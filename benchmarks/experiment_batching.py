"""Profile reusable experiment-level numerical batches on CPU and CUDA."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import torch

from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import (
    GeneratedProblem,
    PublicProblemView,
    SimulationGroundTruth,
)
from graphs.scm import LinearGaussianSCM
from graphs.seeds import AlgorithmSeeds, SimulationSeeds
from numerical import ComputeConfig
from oracle.batching import OracleBatchProblem, OracleInterventionBatchEvaluator

BASE_SEED = 20_260_730
FULL_COUNTS = {
    5: (1, 8, 32, 128),
    20: (1, 8, 32),
    50: (1, 8),
    100: (1, 8),
}
QUICK_COUNTS = {5: (8,), 20: (8,), 50: (1,)}


def _case_grid(*, quick: bool) -> tuple[tuple[int, int], ...]:
    """Return bounded representative dimension/request-count pairs."""
    counts = QUICK_COUNTS if quick else FULL_COUNTS
    return tuple(
        (n_nodes, problem_count)
        for n_nodes, problem_counts in counts.items()
        for problem_count in problem_counts
    )


def _problem(n_nodes: int, seed: int) -> GeneratedProblem:
    """Build one independent SCM, target, and candidate on the CPU."""
    rng = np.random.default_rng(seed)
    probability = min(1.0, 4.0 / max(1, n_nodes - 1))
    support = np.triu(rng.random((n_nodes, n_nodes)) < probability, k=1)
    magnitude = rng.uniform(0.05, 0.20, size=(n_nodes, n_nodes))
    sign = np.where(rng.integers(0, 2, size=(n_nodes, n_nodes)), 1.0, -1.0)
    coefficients = np.where(support, magnitude * sign, 0.0)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(
        (int(parent), int(child)) for parent, child in np.argwhere(coefficients != 0.0)
    )
    scm = LinearGaussianSCM(graph, coefficients)
    target = ShiftScaleIntervention(
        rng.normal(0.0, 0.25, n_nodes),
        rng.uniform(0.75, 1.25, n_nodes),
    )
    current = ShiftScaleIntervention(
        rng.normal(0.0, 0.25, n_nodes),
        rng.uniform(0.75, 1.25, n_nodes),
    )
    target_mean, target_covariance = scm.interventional_moments(target)
    public = PublicProblemView(
        observational_samples=np.zeros((1, n_nodes)),
        target_mean=target_mean,
        target_covariance=target_covariance,
        current_intervention=current,
        algorithm_seeds=AlgorithmSeeds(seed + 1, seed + 2),
    )
    seeds = SimulationSeeds(seed, seed + 3, seed + 4, seed + 5, seed + 6, seed + 7)
    return GeneratedProblem(public, SimulationGroundTruth(scm, target, seeds))


def _case(
    n_nodes: int,
    problem_count: int,
    seed: int,
) -> tuple[tuple[OracleBatchProblem, ...], tuple[ShiftScaleIntervention, ...]]:
    """Construct durable CPU inputs outside every benchmark timing."""
    problems = tuple(
        OracleBatchProblem(
            f"nodes={n_nodes}/problem={index}",
            _problem(n_nodes, seed + index * 101),
        )
        for index in range(problem_count)
    )
    candidates = tuple(item.problem.public.current_intervention for item in problems)
    return problems, candidates


def _repetitions(n_nodes: int, problem_count: int) -> int:
    """Bound profiling time while retaining repeated steady-state rounds."""
    dense_work = n_nodes * n_nodes * problem_count
    if dense_work >= 80_000:
        return 2
    if dense_work >= 10_000:
        return 3
    return 5


def _synchronize(device: str) -> None:
    """Synchronize CUDA only; result materialization already synchronizes too."""
    if device == "cuda":
        torch.cuda.synchronize()


def _time_evaluator(
    problems: tuple[OracleBatchProblem, ...],
    candidates: tuple[ShiftScaleIntervention, ...],
    *,
    backend: str,
    device: str,
    repetitions: int,
) -> tuple[dict[str, object], object]:
    """Measure preparation, first call, and reusable steady evaluation rounds."""
    compute_config = ComputeConfig(
        backend=backend,
        device=device,
        batch_size=len(problems),
    )
    _synchronize(device)
    preparation_start = perf_counter()
    evaluator = OracleInterventionBatchEvaluator(problems, compute_config)
    _synchronize(device)
    preparation_seconds = perf_counter() - preparation_start

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
    else:
        baseline_allocated = 0
    first_start = perf_counter()
    first_result = evaluator.evaluate(candidates, gradients=True)
    _synchronize(device)
    first_seconds = perf_counter() - first_start

    timings = []
    result = first_result
    for _ in range(repetitions):
        start = perf_counter()
        result = evaluator.evaluate(candidates, gradients=True)
        _synchronize(device)
        timings.append(perf_counter() - start)
    median_seconds = statistics.median(timings)
    peak_increment = (
        max(0, torch.cuda.max_memory_allocated() - baseline_allocated)
        if device == "cuda"
        else 0
    )
    return (
        {
            "backend": result.backend_diagnostics.backend,
            "device": result.backend_diagnostics.device,
            "dtype": result.backend_diagnostics.dtype,
            "preparation_seconds": preparation_seconds,
            "first_evaluation_seconds": first_seconds,
            "steady_evaluation_median_seconds": median_seconds,
            "steady_evaluation_minimum_seconds": min(timings),
            "steady_evaluation_maximum_seconds": max(timings),
            "steady_evaluation_repetitions": repetitions,
            "steady_evaluation_timings_seconds": timings,
            "requests_per_second": len(problems) / median_seconds,
            "prepared_bytes": sum(batch.prepared_bytes for batch in result.batches),
            "peak_evaluation_allocated_bytes": peak_increment,
            "host_to_device_bytes_per_round": result.host_to_device_bytes,
            "device_to_host_bytes_per_round": result.device_to_host_bytes,
            "batches": [asdict(batch) for batch in result.batches],
        },
        result,
    )


def _flatten(result: object) -> np.ndarray:
    """Flatten values and heterogeneous gradients for parity reporting."""
    assert result.shift_gradients is not None
    assert result.log_scale_gradients is not None
    return np.concatenate(
        (
            result.values,
            *(result.shift_gradients),
            *(result.log_scale_gradients),
        )
    )


def _errors(actual: object, expected: object) -> tuple[float, float]:
    """Return maximum absolute and guarded relative result errors."""
    actual_flat = _flatten(actual)
    expected_flat = _flatten(expected)
    absolute = np.abs(actual_flat - expected_flat)
    relative = absolute / np.maximum(np.abs(expected_flat), 1e-15)
    return float(np.max(absolute)), float(np.max(relative))


def run_benchmark(*, quick: bool = False) -> dict[str, object]:
    """Run canonical NumPy, Torch CPU, and CUDA experiment batches."""
    if not torch.cuda.is_available():
        raise RuntimeError("The GPU-8 benchmark requires CUDA.")
    torch.cuda.init()
    records = []
    for case_index, (n_nodes, problem_count) in enumerate(_case_grid(quick=quick)):
        problems, candidates = _case(
            n_nodes,
            problem_count,
            BASE_SEED + case_index * 10_000,
        )
        repetitions = _repetitions(n_nodes, problem_count)
        numpy_timing, expected = _time_evaluator(
            problems,
            candidates,
            backend="numpy",
            device="cpu",
            repetitions=repetitions,
        )
        torch_cpu_timing, torch_cpu_result = _time_evaluator(
            problems,
            candidates,
            backend="torch",
            device="cpu",
            repetitions=repetitions,
        )
        cuda_timing, cuda_result = _time_evaluator(
            problems,
            candidates,
            backend="torch",
            device="cuda",
            repetitions=repetitions,
        )
        torch_cpu_error = _errors(torch_cpu_result, expected)
        cuda_error = _errors(cuda_result, expected)
        numpy_seconds = float(numpy_timing["steady_evaluation_median_seconds"])
        torch_cpu_seconds = float(torch_cpu_timing["steady_evaluation_median_seconds"])
        cuda_seconds = float(cuda_timing["steady_evaluation_median_seconds"])
        records.append(
            {
                "n_nodes": n_nodes,
                "problem_count": problem_count,
                "batch_size_cap": problem_count,
                "operation": "value_and_gradient",
                "numpy": numpy_timing,
                "torch_cpu": torch_cpu_timing,
                "cuda": cuda_timing,
                "torch_cpu_maximum_absolute_error": torch_cpu_error[0],
                "torch_cpu_maximum_relative_error": torch_cpu_error[1],
                "cuda_maximum_absolute_error": cuda_error[0],
                "cuda_maximum_relative_error": cuda_error[1],
                "torch_cpu_speedup_vs_numpy": numpy_seconds / torch_cpu_seconds,
                "cuda_speedup_vs_numpy": numpy_seconds / cuda_seconds,
                "cuda_speedup_vs_torch_cpu": torch_cpu_seconds / cuda_seconds,
            }
        )
        del cuda_result
        torch.cuda.empty_cache()

    maximum_absolute_error = max(
        record["cuda_maximum_absolute_error"] for record in records
    )
    maximum_relative_error = max(
        record["cuda_maximum_relative_error"] for record in records
    )
    return {
        "schema_version": 2,
        "kl_direction": "target_to_candidate",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0),
        },
        "methodology": {
            "canonical_reference": "OracleAnalyticGaussianKLLoss through NumPy batch evaluator",
            "operation": "independent_SCM_target_value_and_full_gradient",
            "grouping": "equal_n_nodes_in_first_seen_order",
            "graph_problem_generation_excluded": True,
            "scheduler_and_graph_policy_executed": False,
            "stable_scm_and_target_preparation_reused": True,
            "candidate_numpy_stacking_included": True,
            "cuda_stable_preparation_transfer_count_per_batch": 1,
            "cuda_candidate_transfer_count_per_batch": 1,
            "cuda_combined_result_transfer_count_per_batch": 1,
            "automatic_device_thresholds_added": False,
            "gpu_context_initialization_excluded": True,
            "steady_state_uses_median_wall_time": True,
        },
        "cases": records,
        "summary": {
            "case_count": len(records),
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_error": maximum_relative_error,
            "cuda_faster_than_numpy_count": sum(
                record["cuda_speedup_vs_numpy"] > 1.0 for record in records
            ),
            "cuda_faster_than_torch_cpu_count": sum(
                record["cuda_speedup_vs_torch_cpu"] > 1.0 for record in records
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded profiling CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
    )
    return parser


def main() -> None:
    """Run the profile and write a deterministic JSON schema."""
    arguments = build_parser().parse_args()
    payload = run_benchmark(quick=arguments.quick)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
