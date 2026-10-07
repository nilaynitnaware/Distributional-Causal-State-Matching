"""Benchmark canonical NumPy sampling against two prepared CUDA samplers."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import torch

from graphs.interventions import ShiftScaleIntervention
from graphs.scm import LinearGaussianSCM
from numerical.torch_backend import PreparedSCMBatch, TorchBackend
from numerical.torch_sampling import SamplingMethod

BASE_SEED = 20_260_728
FULL_NODE_COUNTS = (5, 10, 20, 50, 100, 200, 500)
QUICK_NODE_COUNTS = (5, 100)
METHODS: tuple[SamplingMethod, ...] = ("topological", "dense")


@dataclass(frozen=True)
class _Case:
    """Canonical CPU objects and durable arrays prepared outside timings."""

    n_nodes: int
    candidate_count: int
    sample_count: int
    scm: LinearGaussianSCM
    interventions: tuple[ShiftScaleIntervention, ...]
    shifts: np.ndarray
    scales: np.ndarray


def _case_grid(*, quick: bool) -> tuple[tuple[int, int, int], ...]:
    """Return graph, sample, and candidate shapes with bounded dense outputs."""
    node_counts = QUICK_NODE_COUNTS if quick else FULL_NODE_COUNTS
    cases = []
    for n_nodes in node_counts:
        cases.extend(
            (
                (n_nodes, 100, 1),
                (n_nodes, 10_000, 1),
            )
        )
        if quick or n_nodes >= 500:
            batched_samples = 2_000
        elif n_nodes >= 200:
            batched_samples = 5_000
        else:
            batched_samples = 10_000
        cases.append((n_nodes, batched_samples, 8))
        if not quick and n_nodes <= 100:
            cases.append((n_nodes, 100_000, 1))
    return tuple(cases)


def _coefficient_matrix(n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    """Create a sparse upper-triangular DAG with expected total degree four."""
    edge_probability = min(1.0, 4.0 / max(1, n_nodes - 1))
    support = np.triu(
        rng.random((n_nodes, n_nodes)) < edge_probability,
        k=1,
    )
    magnitudes = rng.uniform(0.05, 0.20, size=(n_nodes, n_nodes))
    signs = np.where(rng.integers(0, 2, size=(n_nodes, n_nodes)), 1.0, -1.0)
    return np.where(support, magnitudes * signs, 0.0)


def _make_case(
    n_nodes: int,
    sample_count: int,
    candidate_count: int,
    seed: int,
) -> _Case:
    """Build one shared SCM and one or many interventions."""
    rng = np.random.default_rng(seed)
    coefficients = _coefficient_matrix(n_nodes, rng)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))
    dag.add_edges_from(
        (int(parent), int(child)) for parent, child in np.argwhere(coefficients != 0.0)
    )
    scm = LinearGaussianSCM(dag, coefficients)
    shifts = rng.normal(0.0, 0.25, size=(candidate_count, n_nodes))
    scales = rng.uniform(0.75, 1.25, size=(candidate_count, n_nodes))
    interventions = tuple(
        ShiftScaleIntervention(shift, scale)
        for shift, scale in zip(shifts, scales, strict=True)
    )
    return _Case(
        n_nodes=n_nodes,
        candidate_count=candidate_count,
        sample_count=sample_count,
        scm=scm,
        interventions=interventions,
        shifts=shifts,
        scales=scales,
    )


def _cpu_samples(case: _Case, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Run the canonical NumPy topological sampler for every candidate."""
    return tuple(
        case.scm.sample_interventional(
            intervention,
            case.sample_count,
            rng,
        )
        for intervention in case.interventions
    )


def _time_cpu(function: Callable[[], object]) -> dict[str, object]:
    """Return stable median per-call CPU wall time with bounded repetition."""
    function()
    start = perf_counter()
    function()
    calibration_seconds = perf_counter() - start
    inner_iterations = min(
        20,
        max(1, math.ceil(0.02 / max(calibration_seconds, 1e-12))),
    )
    repetitions = 3 if calibration_seconds >= 0.20 else 5
    timings = []
    for _ in range(repetitions):
        start = perf_counter()
        for _ in range(inner_iterations):
            function()
        timings.append((perf_counter() - start) / inner_iterations)
    return {
        "median_seconds": statistics.median(timings),
        "minimum_seconds": min(timings),
        "maximum_seconds": max(timings),
        "repetitions": repetitions,
        "inner_iterations": inner_iterations,
        "timings_seconds": timings,
    }


def _cuda_repetitions(case: _Case) -> int:
    """Use more repeats when the output contains at most five million values."""
    output_elements = case.candidate_count * case.sample_count * case.n_nodes
    return 7 if output_elements <= 5_000_000 else 3


def _time_cuda_preparation(
    backend: TorchBackend,
    case: _Case,
) -> tuple[float, PreparedSCMBatch]:
    """Measure coefficient transfer and sparse topology preparation."""
    timings = []
    prepared = backend.prepare_scm_sampler(case.scm.coefficients)
    for _ in range(3):
        torch.cuda.synchronize()
        start = perf_counter()
        prepared = backend.prepare_scm_sampler(case.scm.coefficients)
        torch.cuda.synchronize()
        timings.append(perf_counter() - start)
    return statistics.median(timings), prepared


def _time_cuda_input_transfer(
    prepared: PreparedSCMBatch,
    case: _Case,
    repetitions: int,
) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
    """Measure intervention validation and host-to-device transfer."""
    timings = []
    tensors = prepared.prepare_interventions(case.shifts, case.scales)
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = perf_counter()
        tensors = prepared.prepare_interventions(case.shifts, case.scales)
        torch.cuda.synchronize()
        timings.append(perf_counter() - start)
    return statistics.median(timings), tensors


def _time_cuda_compute(
    prepared: PreparedSCMBatch,
    parameters: tuple[torch.Tensor, torch.Tensor],
    case: _Case,
    generator: torch.Generator,
    method: SamplingMethod,
    repetitions: int,
) -> tuple[float, float, torch.Tensor]:
    """Measure RNG plus transformation using events and synchronized wall time."""
    event_timings = []
    wall_timings = []
    result = None
    for _ in range(repetitions):
        if result is not None:
            del result
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = perf_counter()
        start_event.record()
        result = prepared.sample_interventional_from_tensors(
            *parameters,
            case.sample_count,
            generator,
            method=method,
        )
        end_event.record()
        end_event.synchronize()
        wall_timings.append(perf_counter() - wall_start)
        event_timings.append(start_event.elapsed_time(end_event) / 1_000.0)
    assert result is not None
    return (
        statistics.median(event_timings),
        statistics.median(wall_timings),
        result,
    )


def _time_cuda_output_transfer(
    result: torch.Tensor,
    repetitions: int,
) -> tuple[float, np.ndarray]:
    """Measure full sample D2H transfer and detached NumPy copy."""
    timings = []
    array = result.detach().cpu().numpy().copy()
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = perf_counter()
        array = result.detach().cpu().numpy().copy()
        torch.cuda.synchronize()
        timings.append(perf_counter() - start)
    return statistics.median(timings), array


def _manual_transform(case: _Case, noise: np.ndarray) -> np.ndarray:
    """Transform common explicit noise using canonical node semantics."""
    results = []
    for batch_index, intervention in enumerate(case.interventions):
        samples = noise[batch_index].copy()
        for node in range(case.n_nodes):
            parents = np.flatnonzero(case.scm.coefficients[:, node] != 0.0)
            mechanism = noise[batch_index, :, node].copy()
            if parents.size:
                mechanism += samples[:, parents] @ case.scm.coefficients[parents, node]
            samples[:, node] = (
                intervention.shift[node] + intervention.scale[node] * mechanism
            )
        results.append(samples)
    return np.stack(results)


def _accuracy(
    prepared: PreparedSCMBatch,
    parameters: tuple[torch.Tensor, torch.Tensor],
    case: _Case,
    method: SamplingMethod,
    seed: int,
) -> tuple[float, float]:
    """Compare the CUDA transformation to canonical semantics on common noise."""
    rng = np.random.default_rng(seed)
    accuracy_samples = min(64, case.sample_count)
    noise = rng.normal(size=(case.candidate_count, accuracy_samples, case.n_nodes))
    expected = _manual_transform(case, noise)
    noise_tensor = torch.tensor(noise, dtype=torch.float64, device="cuda")
    actual = (
        prepared.samples_from_noise(
            *parameters,
            noise_tensor,
            method=method,
        )
        .cpu()
        .numpy()
    )
    absolute = np.abs(expected - actual)
    denominator = np.maximum(np.abs(expected), 1e-10)
    return float(absolute.max(initial=0.0)), float(
        (absolute / denominator).max(initial=0.0)
    )


def _benchmark_method(
    backend: TorchBackend,
    case: _Case,
    cpu_timing: dict[str, object],
    method: SamplingMethod,
    seed: int,
) -> dict[str, object]:
    """Benchmark one CUDA algorithm for a controlled shape."""
    repetitions = _cuda_repetitions(case)
    preparation_time, prepared = _time_cuda_preparation(backend, case)
    input_transfer_time, parameters = _time_cuda_input_transfer(
        prepared,
        case,
        repetitions,
    )
    maximum_absolute_error, maximum_relative_error = _accuracy(
        prepared,
        parameters,
        case,
        method,
        seed + 1,
    )

    warm_result = prepared.sample_interventional_from_tensors(
        *parameters,
        case.sample_count,
        backend.make_generator(seed + 2),
        method=method,
    )
    torch.cuda.synchronize()
    del warm_result
    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    compute_time, compute_wall_time, result = _time_cuda_compute(
        prepared,
        parameters,
        case,
        backend.make_generator(seed + 3),
        method,
        repetitions,
    )
    peak_memory = max(
        0,
        torch.cuda.max_memory_allocated() - baseline_memory,
    )
    output_transfer_time, _ = _time_cuda_output_transfer(result, repetitions)
    transfer_time = input_transfer_time + output_transfer_time
    total_time = transfer_time + compute_wall_time
    cpu_time = float(cpu_timing["median_seconds"])
    return {
        "operation": "interventional_sampling",
        "mode": (
            "candidate_batch_shared_scm" if case.candidate_count > 1 else "single_scm"
        ),
        "algorithm": method,
        "n_nodes": case.n_nodes,
        "batch_size": 1,
        "candidate_count": case.candidate_count,
        "effective_batch_size": case.candidate_count,
        "sample_count": case.sample_count,
        "dtype": "float64",
        "cpu_backend": "numpy",
        "backend": "torch",
        "device": "cuda",
        "cpu_time": cpu_time,
        "preparation_time": preparation_time,
        "input_transfer_time": input_transfer_time,
        "output_transfer_time": output_transfer_time,
        "transfer_time": transfer_time,
        "compute_time": compute_time,
        "compute_wall_time": compute_wall_time,
        "total_time": total_time,
        "first_call_total_time": preparation_time + total_time,
        "prepared_memory": prepared.prepared_bytes,
        "peak_memory": peak_memory,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "speedup": cpu_time / total_time,
        "compute_only_speedup": cpu_time / compute_time,
        "cpu_timing": cpu_timing,
        "cuda_repetitions": repetitions,
        "cuda_warmup_runs": 1,
    }


def _environment() -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_compute_capability": ".".join(
            str(item) for item in torch.cuda.get_device_capability(0)
        ),
        "gpu_total_memory": properties.total_memory,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "python_executable": sys.executable,
    }


def _warm_cuda(backend: TorchBackend) -> None:
    """Exclude CUDA, RNG, sparse recurrence, and solve initialization."""
    coefficients = np.array([[0.0, 0.1], [0.0, 0.0]])
    prepared = backend.prepare_scm_sampler(coefficients)
    parameters = prepared.prepare_interventions(np.zeros(2), np.ones(2))
    for method in METHODS:
        prepared.sample_interventional_from_tensors(
            *parameters,
            8,
            backend.make_generator(1),
            method=method,
        )
    torch.cuda.synchronize()


def run_benchmark(*, quick: bool) -> dict[str, object]:
    """Run synchronized NumPy/CUDA sampling comparisons for both algorithms."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Torch sampling benchmark.")
    backend = TorchBackend(device="cuda")
    _warm_cuda(backend)

    records = []
    for index, (n_nodes, sample_count, candidate_count) in enumerate(
        _case_grid(quick=quick)
    ):
        case = _make_case(
            n_nodes,
            sample_count,
            candidate_count,
            BASE_SEED + index,
        )
        cpu_rng = np.random.default_rng(BASE_SEED + 10_000 + index)
        cpu_timing = _time_cpu(
            lambda case=case, cpu_rng=cpu_rng: _cpu_samples(case, cpu_rng)
        )
        for method_index, method in enumerate(METHODS):
            records.append(
                _benchmark_method(
                    backend,
                    case,
                    cpu_timing,
                    method,
                    BASE_SEED + 20_000 + 10 * index + method_index,
                )
            )

    best_record = max(records, key=lambda item: float(item["speedup"]))
    worst_record = min(records, key=lambda item: float(item["speedup"]))
    paired = {}
    for record in records:
        key = (
            record["n_nodes"],
            record["sample_count"],
            record["candidate_count"],
        )
        paired.setdefault(key, {})[record["algorithm"]] = record
    algorithm_wins = {
        method: sum(
            float(items[method]["total_time"])
            == min(float(value["total_time"]) for value in items.values())
            for items in paired.values()
        )
        for method in METHODS
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": _environment(),
        "methodology": {
            "timer": "time.perf_counter plus synchronized CUDA events",
            "cpu_reference": "canonical NumPy sparse topological SCM sampling",
            "gpu_algorithms": list(METHODS),
            "dense_algorithm": (
                "reorder by prepared topological order, then solve the lower-"
                "triangular structural system against all sample columns"
            ),
            "gpu_compute_includes_rng": True,
            "gpu_warmup_runs": 1,
            "gpu_context_initialization_excluded": True,
            "sampling_preparation_reported_separately": True,
            "input_and_output_transfers_reported_separately": True,
            "total_time_definition": (
                "intervention input transfer + synchronized RNG/compute wall + "
                "full sample output transfer"
            ),
            "speedup_definition": "CPU time / steady-state GPU total time",
            "numerical_error_definition": (
                "canonical CPU mechanism versus GPU transform on identical "
                "explicit float64 noise"
            ),
            "statistical_validation": "covered by tests/test_torch_sampling.py",
            "node_counts": list(QUICK_NODE_COUNTS if quick else FULL_NODE_COUNTS),
            "sample_counts": [100, 10_000, 100_000],
            "candidate_counts": [1, 8],
            "large_output_limits": {
                "200_nodes_batched_samples": 5_000,
                "500_nodes_batched_samples": 2_000,
            },
            "peak_memory_definition": (
                "incremental CUDA allocator peak above prepared SCM and inputs"
            ),
            "numerical_fixture": (
                "sparse upper-triangular DAG with expected total degree four, "
                "coefficient magnitudes in [0.05, 0.20], shifts from N(0, 0.25), "
                "and scales in [0.75, 1.25]"
            ),
        },
        "summary": {
            "record_count": len(records),
            "shape_case_count": len(paired),
            "gpu_faster_record_count": sum(
                float(item["speedup"]) > 1.0 for item in records
            ),
            "gpu_faster_by_algorithm": {
                method: sum(
                    item["algorithm"] == method and float(item["speedup"]) > 1.0
                    for item in records
                )
                for method in METHODS
            },
            "algorithm_wins": algorithm_wins,
            "maximum_absolute_error": max(
                float(item["maximum_absolute_error"]) for item in records
            ),
            "maximum_relative_error": max(
                float(item["maximum_relative_error"]) for item in records
            ),
            "best_speedup": best_record["speedup"],
            "best_case": {
                key: best_record[key]
                for key in (
                    "algorithm",
                    "n_nodes",
                    "sample_count",
                    "candidate_count",
                )
            },
            "worst_speedup": worst_record["speedup"],
            "worst_case": {
                key: worst_record[key]
                for key in (
                    "algorithm",
                    "n_nodes",
                    "sample_count",
                    "candidate_count",
                )
            },
        },
        "operations": records,
    }


def _write_result(result: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark NumPy and CUDA linear SCM sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("torch_sampling.json"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced graph and sample-count grid for smoke validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    result = run_benchmark(quick=namespace.quick)
    _write_result(result, namespace.output)
    print(
        f"wrote {len(result['operations'])} Torch sampling benchmark records "
        f"to {namespace.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
