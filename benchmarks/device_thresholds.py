"""Derive a conservative CUDA threshold for experiment-batch gradients."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from benchmarks.experiment_batching import _case
from numerical import ComputeConfig
from numerical.torch_backend import TorchBackend
from numerical.torch_gradient import PreparedInterventionKLLoss
from oracle.batching import OracleBatchProblem, OracleInterventionBatchEvaluator

BASE_SEED = 20_260_731
MINIMUM_REQUIRED_SPEEDUP = 1.5
FULL_COUNTS = {
    5: (1, 32, 64, 128),
    10: (1, 16, 32, 64),
    20: (1, 4, 8, 32),
    50: (1, 2, 8),
    100: (1, 2, 8),
}
QUICK_COUNTS = {5: (64, 128), 20: (4, 8), 50: (1, 2), 100: (1,)}


def _case_grid(*, quick: bool) -> tuple[tuple[int, int], ...]:
    """Return crossover-focused dimension and compatible-batch shapes."""
    counts = QUICK_COUNTS if quick else FULL_COUNTS
    return tuple(
        (n_nodes, batch_size)
        for n_nodes, batch_sizes in counts.items()
        for batch_size in batch_sizes
    )


def _synchronize(device: str) -> None:
    """Synchronize only CUDA operations."""
    if device == "cuda":
        torch.cuda.synchronize()


def _repetitions(n_nodes: int, batch_size: int) -> int:
    """Use repeated one-shot trials while bounding dense CPU work."""
    dense_work = n_nodes * n_nodes * batch_size
    if dense_work >= 20_000:
        return 3
    return 5


def _time_one_shot(
    problems: tuple[OracleBatchProblem, ...],
    candidates: tuple[object, ...],
    *,
    backend: str,
    device: str,
    repetitions: int,
) -> tuple[dict[str, object], object]:
    """Time preparation plus one evaluation with warmup and synchronization."""
    config = ComputeConfig(
        backend=backend,
        device=device,
        batch_size=len(problems),
    )
    warmup = OracleInterventionBatchEvaluator(problems, config)
    warmup.evaluate(candidates, gradients=True)
    _synchronize(device)
    del warmup
    if device == "cuda":
        torch.cuda.empty_cache()

    preparation_timings = []
    evaluation_timings = []
    total_timings = []
    result = None
    peak_allocated_bytes = 0
    for _ in range(repetitions):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            baseline_allocated = torch.cuda.memory_allocated()
        else:
            baseline_allocated = 0
        _synchronize(device)
        total_start = perf_counter()
        preparation_start = perf_counter()
        evaluator = OracleInterventionBatchEvaluator(problems, config)
        _synchronize(device)
        preparation_seconds = perf_counter() - preparation_start
        evaluation_start = perf_counter()
        result = evaluator.evaluate(candidates, gradients=True)
        _synchronize(device)
        evaluation_seconds = perf_counter() - evaluation_start
        total_seconds = perf_counter() - total_start
        preparation_timings.append(preparation_seconds)
        evaluation_timings.append(evaluation_seconds)
        total_timings.append(total_seconds)
        if device == "cuda":
            peak_allocated_bytes = max(
                peak_allocated_bytes,
                max(0, torch.cuda.max_memory_allocated() - baseline_allocated),
            )
        del evaluator
    assert result is not None
    return (
        {
            "backend": result.backend_diagnostics.backend,
            "device": result.backend_diagnostics.device,
            "dtype": result.backend_diagnostics.dtype,
            "repetitions": repetitions,
            "preparation_median_seconds": statistics.median(preparation_timings),
            "evaluation_median_seconds": statistics.median(evaluation_timings),
            "total_median_seconds": statistics.median(total_timings),
            "total_minimum_seconds": min(total_timings),
            "total_maximum_seconds": max(total_timings),
            "total_timings_seconds": total_timings,
            "peak_allocated_bytes": peak_allocated_bytes,
            "prepared_bytes": sum(batch.prepared_bytes for batch in result.batches),
            "stable_host_to_device_bytes": sum(
                batch.preparation_host_to_device_bytes for batch in result.batches
            ),
            "candidate_host_to_device_bytes": result.host_to_device_bytes,
            "result_device_to_host_bytes": result.device_to_host_bytes,
        },
        result,
    )


def _cuda_components(
    problems: tuple[OracleBatchProblem, ...],
    candidates: tuple[object, ...],
) -> dict[str, object]:
    """Decompose one warmed CUDA round into host, transfer, and kernel stages."""
    backend = TorchBackend(device="cuda")
    n_nodes = problems[0].problem.public.n_nodes
    batch_size = len(problems)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()

    host_stable_start = perf_counter()
    coefficients = np.stack(
        [item.problem.ground_truth.scm.coefficients for item in problems]
    )
    target_means = np.stack([item.problem.public.target_mean for item in problems])
    target_covariances = np.stack(
        [item.problem.public.target_covariance for item in problems]
    )
    packed_numpy = np.concatenate(
        (
            coefficients.reshape(batch_size, -1),
            target_means,
            target_covariances.reshape(batch_size, -1),
        ),
        axis=1,
    )
    host_stable_pack_seconds = perf_counter() - host_stable_start

    _synchronize("cuda")
    stable_transfer_start = perf_counter()
    packed = torch.tensor(packed_numpy, dtype=torch.float64, device="cuda")
    _synchronize("cuda")
    stable_transfer_seconds = perf_counter() - stable_transfer_start
    coefficient_end = n_nodes * n_nodes
    mean_end = coefficient_end + n_nodes
    preparation_start_event = torch.cuda.Event(enable_timing=True)
    preparation_end_event = torch.cuda.Event(enable_timing=True)
    preparation_start_event.record()
    prepared = PreparedInterventionKLLoss(
        backend.prepare_scm_batch(
            packed[:, :coefficient_end].reshape(batch_size, n_nodes, n_nodes)
        ),
        backend.prepare_gaussian_target(
            packed[:, coefficient_end:mean_end],
            packed[:, mean_end:].reshape(batch_size, n_nodes, n_nodes),
        ),
    )
    preparation_end_event.record()
    _synchronize("cuda")
    device_preparation_seconds = (
        preparation_start_event.elapsed_time(preparation_end_event) / 1_000.0
    )

    host_candidate_start = perf_counter()
    parameters = np.stack(
        (
            np.stack([candidate.shift for candidate in candidates]),
            np.log(np.stack([candidate.scale for candidate in candidates])),
        ),
        axis=1,
    )
    host_candidate_pack_seconds = perf_counter() - host_candidate_start
    input_transfer_start = perf_counter()
    parameter_tensor = torch.tensor(parameters, dtype=torch.float64, device="cuda")
    _synchronize("cuda")
    input_transfer_seconds = perf_counter() - input_transfer_start

    kernel_start = torch.cuda.Event(enable_timing=True)
    kernel_end = torch.cuda.Event(enable_timing=True)
    kernel_start.record()
    values, shift_gradient, log_scale_gradient = (
        prepared.value_and_gradient_from_tensors(
            parameter_tensor[:, 0, :],
            parameter_tensor[:, 1, :],
        )
    )
    kernel_end.record()
    _synchronize("cuda")
    kernel_seconds = kernel_start.elapsed_time(kernel_end) / 1_000.0

    output_transfer_start = perf_counter()
    output = torch.cat(
        (values.unsqueeze(-1), shift_gradient, log_scale_gradient),
        dim=-1,
    ).cpu()
    _synchronize("cuda")
    output_transfer_seconds = perf_counter() - output_transfer_start
    return {
        "host_stable_pack_seconds": host_stable_pack_seconds,
        "stable_host_to_device_seconds": stable_transfer_seconds,
        "device_target_preparation_seconds": device_preparation_seconds,
        "host_candidate_pack_seconds": host_candidate_pack_seconds,
        "candidate_host_to_device_seconds": input_transfer_seconds,
        "value_and_gradient_kernel_seconds": kernel_seconds,
        "combined_result_device_to_host_seconds": output_transfer_seconds,
        "stable_host_to_device_bytes": int(packed_numpy.nbytes),
        "candidate_host_to_device_bytes": int(parameters.nbytes),
        "result_device_to_host_bytes": int(output.numel() * output.element_size()),
        "prepared_bytes": int(prepared.prepared_bytes),
        "peak_allocated_bytes": max(
            0,
            torch.cuda.max_memory_allocated() - baseline_allocated,
        ),
    }


def _flatten(result: object) -> np.ndarray:
    """Flatten durable values and gradients for numerical error checks."""
    assert result.shift_gradients is not None
    assert result.log_scale_gradients is not None
    return np.concatenate(
        (result.values, *result.shift_gradients, *result.log_scale_gradients)
    )


def _errors(actual: object, expected: object) -> tuple[float, float]:
    """Return guarded maximum absolute and relative errors."""
    actual_flat = _flatten(actual)
    expected_flat = _flatten(expected)
    absolute = np.abs(actual_flat - expected_flat)
    relative = absolute / np.maximum(np.abs(expected_flat), 1e-15)
    return float(np.max(absolute)), float(np.max(relative))


def _derive_threshold(records: list[dict[str, object]]) -> dict[str, object]:
    """Choose the lowest dense-work cutoff with a safety-margin crossover."""
    scores = sorted({int(record["dense_work"]) for record in records})
    for score in scores:
        eligible = [record for record in records if int(record["dense_work"]) >= score]
        if all(
            float(record["cuda_conservative_total_speedup_vs_numpy"])
            >= MINIMUM_REQUIRED_SPEEDUP
            for record in eligible
        ):
            frontier = []
            for n_nodes in sorted(
                {int(record["n_nodes"]) for record in records},
                reverse=True,
            ):
                minimum_batch_from_dense_work = math.ceil(score / (n_nodes * n_nodes))
                measured_batches = sorted(
                    int(record["compatible_batch_size"])
                    for record in records
                    if int(record["n_nodes"]) == n_nodes
                    and int(record["compatible_batch_size"])
                    >= minimum_batch_from_dense_work
                )
                if not measured_batches:
                    continue
                minimum_batch_size = measured_batches[0]
                observed_above_frontier = [
                    record
                    for record in records
                    if int(record["n_nodes"]) == n_nodes
                    and int(record["compatible_batch_size"]) >= minimum_batch_size
                ]
                if all(
                    float(record["cuda_conservative_total_speedup_vs_numpy"])
                    >= MINIMUM_REQUIRED_SPEEDUP
                    for record in observed_above_frontier
                ):
                    frontier.append(
                        {
                            "minimum_n_nodes": n_nodes,
                            "minimum_compatible_batch_size": minimum_batch_size,
                        }
                    )
            return {
                "metric": "n_nodes_squared_times_compatible_batch_size",
                "minimum_dense_work": score,
                "minimum_required_observed_speedup": MINIMUM_REQUIRED_SPEEDUP,
                "minimum_observed_eligible_speedup": min(
                    float(record["cuda_conservative_total_speedup_vs_numpy"])
                    for record in eligible
                ),
                "eligible_case_count": len(eligible),
                "ineligible_case_count": len(records) - len(eligible),
                "operation": "intervention_kl_value_and_gradient",
                "precision": "float64",
                "conservative_shape_frontier": frontier,
            }
    raise RuntimeError("No conservative CUDA threshold satisfied the safety margin.")


def run_benchmark(*, quick: bool = False) -> dict[str, object]:
    """Benchmark controlled one-shot batches and derive the CUDA threshold."""
    if not torch.cuda.is_available():
        raise RuntimeError("The GPU-9 benchmark requires CUDA.")
    torch.cuda.init()
    records = []
    for case_index, (n_nodes, batch_size) in enumerate(_case_grid(quick=quick)):
        problems, candidates = _case(
            n_nodes,
            batch_size,
            BASE_SEED + case_index * 10_000,
        )
        repetitions = _repetitions(n_nodes, batch_size)
        numpy_timing, expected = _time_one_shot(
            problems,
            candidates,
            backend="numpy",
            device="cpu",
            repetitions=repetitions,
        )
        torch_cpu_timing, torch_cpu_result = _time_one_shot(
            problems,
            candidates,
            backend="torch",
            device="cpu",
            repetitions=repetitions,
        )
        cuda_timing, cuda_result = _time_one_shot(
            problems,
            candidates,
            backend="torch",
            device="cuda",
            repetitions=repetitions,
        )
        components = _cuda_components(problems, candidates)
        torch_cpu_error = _errors(torch_cpu_result, expected)
        cuda_error = _errors(cuda_result, expected)
        numpy_seconds = float(numpy_timing["total_median_seconds"])
        torch_cpu_seconds = float(torch_cpu_timing["total_median_seconds"])
        cuda_seconds = float(cuda_timing["total_median_seconds"])
        records.append(
            {
                "n_nodes": n_nodes,
                "compatible_batch_size": batch_size,
                "dense_work": n_nodes * n_nodes * batch_size,
                "operation": "intervention_kl_value_and_gradient",
                "numpy": numpy_timing,
                "torch_cpu": torch_cpu_timing,
                "cuda": cuda_timing,
                "cuda_components": components,
                "torch_cpu_maximum_absolute_error": torch_cpu_error[0],
                "torch_cpu_maximum_relative_error": torch_cpu_error[1],
                "cuda_maximum_absolute_error": cuda_error[0],
                "cuda_maximum_relative_error": cuda_error[1],
                "torch_cpu_total_speedup_vs_numpy": (numpy_seconds / torch_cpu_seconds),
                "cuda_total_speedup_vs_numpy": numpy_seconds / cuda_seconds,
                "cuda_conservative_total_speedup_vs_numpy": (
                    float(numpy_timing["total_minimum_seconds"])
                    / float(cuda_timing["total_maximum_seconds"])
                ),
                "cuda_total_speedup_vs_torch_cpu": torch_cpu_seconds / cuda_seconds,
            }
        )
        del cuda_result
        torch.cuda.empty_cache()
    threshold = _derive_threshold(records)
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
            "timed_scope": "evaluator_preparation_plus_one_value_and_gradient_round",
            "warmup_excluded": True,
            "cuda_synchronization": True,
            "transfer_overhead_included": True,
            "cuda_components_measured_separately": True,
            "peak_memory_measured": True,
            "problem_generation_excluded": True,
            "scheduler_and_graph_policy_executed": False,
            "threshold_requires_every_observed_eligible_case_to_meet_margin": True,
            "threshold_speedup_estimator": (
                "minimum_numpy_total_seconds_divided_by_maximum_cuda_total_seconds"
            ),
        },
        "threshold": threshold,
        "cases": records,
        "summary": {
            "case_count": len(records),
            "cuda_faster_than_numpy_count": sum(
                float(record["cuda_total_speedup_vs_numpy"]) > 1.0 for record in records
            ),
            "cuda_faster_than_torch_cpu_count": sum(
                float(record["cuda_total_speedup_vs_torch_cpu"]) > 1.0
                for record in records
            ),
            "maximum_absolute_error": max(
                float(record["cuda_maximum_absolute_error"]) for record in records
            ),
            "maximum_relative_error": max(
                float(record["cuda_maximum_relative_error"]) for record in records
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the controlled threshold-benchmark CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
    )
    return parser


def main() -> None:
    """Run the benchmark and save the threshold evidence."""
    arguments = build_parser().parse_args()
    payload = run_benchmark(quick=arguments.quick)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
