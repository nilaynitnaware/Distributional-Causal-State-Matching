"""Benchmark canonical NumPy moments against prepared batched CUDA moments."""

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
from numerical import NUMPY_BACKEND
from numerical.torch_backend import PreparedSCMBatch, TorchBackend

BASE_SEED = 20_260_726
FULL_NODE_COUNTS = (5, 10, 20, 50, 100, 200, 500)
QUICK_NODE_COUNTS = (5, 20, 100)
FULL_BATCH_SIZES = (1, 8, 32, 128)
QUICK_BATCH_SIZES = (1, 8, 32)
MODES = ("candidate_batch", "independent_scm_batch")


@dataclass(frozen=True)
class _Case:
    """CPU objects and durable NumPy inputs prepared outside timings."""

    mode: str
    n_nodes: int
    effective_batch_size: int
    torch_coefficients: np.ndarray
    scms: tuple[LinearGaussianSCM, ...]
    interventions: tuple[ShiftScaleIntervention, ...]
    shifts: np.ndarray
    scales: np.ndarray


def _batch_sizes(n_nodes: int, *, quick: bool) -> tuple[int, ...]:
    """Bound the largest dense batches at high node counts."""
    requested = QUICK_BATCH_SIZES if quick else FULL_BATCH_SIZES
    if n_nodes >= 500:
        return tuple(size for size in requested if size <= 8)
    if n_nodes >= 200:
        return tuple(size for size in requested if size <= 32)
    return requested


def _case_grid(*, quick: bool) -> tuple[tuple[str, int, int], ...]:
    """Return both batching modes over representative shape-compatible cases."""
    node_counts = QUICK_NODE_COUNTS if quick else FULL_NODE_COUNTS
    return tuple(
        (mode, n_nodes, batch_size)
        for n_nodes in node_counts
        for batch_size in _batch_sizes(n_nodes, quick=quick)
        for mode in MODES
    )


def _coefficient_matrix(n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    """Create a sparse upper-triangular, moderately conditioned DAG matrix."""
    edge_probability = min(1.0, 4.0 / max(1, n_nodes - 1))
    support = np.triu(
        rng.random((n_nodes, n_nodes)) < edge_probability,
        k=1,
    )
    magnitudes = rng.uniform(0.05, 0.20, size=(n_nodes, n_nodes))
    signs = np.where(rng.integers(0, 2, size=(n_nodes, n_nodes)), 1.0, -1.0)
    return np.where(support, magnitudes * signs, 0.0)


def _scm(coefficients: np.ndarray) -> LinearGaussianSCM:
    """Construct the canonical CPU SCM for a prepared coefficient matrix."""
    n_nodes = coefficients.shape[0]
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))
    dag.add_edges_from(
        (int(parent), int(child)) for parent, child in np.argwhere(coefficients != 0.0)
    )
    return LinearGaussianSCM(dag, coefficients)


def _make_case(
    mode: str,
    n_nodes: int,
    effective_batch_size: int,
    seed: int,
) -> _Case:
    """Build one candidate-batch or independent-SCM benchmark case."""
    rng = np.random.default_rng(seed)
    coefficient_count = effective_batch_size if mode == "independent_scm_batch" else 1
    coefficients = tuple(
        _coefficient_matrix(n_nodes, rng) for _ in range(coefficient_count)
    )
    scms = tuple(_scm(coefficient) for coefficient in coefficients)
    shifts = rng.normal(0.0, 0.25, size=(effective_batch_size, n_nodes))
    scales = rng.uniform(0.75, 1.25, size=(effective_batch_size, n_nodes))
    interventions = tuple(
        ShiftScaleIntervention(shift, scale)
        for shift, scale in zip(shifts, scales, strict=True)
    )
    torch_coefficients = (
        coefficients[0] if mode == "candidate_batch" else np.stack(coefficients)
    )
    return _Case(
        mode=mode,
        n_nodes=n_nodes,
        effective_batch_size=effective_batch_size,
        torch_coefficients=torch_coefficients,
        scms=scms,
        interventions=interventions,
        shifts=shifts,
        scales=scales,
    )


def _cpu_moments(case: _Case) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the canonical NumPy backend in its current per-problem loop."""
    if case.mode == "candidate_batch":
        results = tuple(
            NUMPY_BACKEND.interventional_moments(case.scms[0], intervention)
            for intervention in case.interventions
        )
    else:
        results = tuple(
            NUMPY_BACKEND.interventional_moments(scm, intervention)
            for scm, intervention in zip(
                case.scms,
                case.interventions,
                strict=True,
            )
        )
    return (
        np.stack([mean for mean, _ in results]),
        np.stack([covariance for _, covariance in results]),
    )


def _time_cpu(function: Callable[[], object]) -> dict[str, object]:
    """Return stable median per-call CPU wall time with adaptive inner loops."""
    function()
    start = perf_counter()
    function()
    calibration_seconds = perf_counter() - start
    inner_iterations = min(
        100,
        max(1, math.ceil(0.02 / max(calibration_seconds, 1e-12))),
    )
    repetitions = 3 if calibration_seconds >= 0.25 else 5
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


def _timing_repetitions(case: _Case) -> int:
    """Use more repeats for small cases and bounded repeats for large cases."""
    dense_elements = case.effective_batch_size * case.n_nodes * case.n_nodes
    return 7 if dense_elements <= 1_000_000 else 3


def _time_cuda_preparation(
    backend: TorchBackend,
    coefficients: np.ndarray,
) -> tuple[float, PreparedSCMBatch]:
    """Measure one-time coefficient transfer and preparation."""
    timings = []
    prepared = backend.prepare_scm_batch(coefficients)
    for _ in range(3):
        torch.cuda.synchronize()
        start = perf_counter()
        prepared = backend.prepare_scm_batch(coefficients)
        torch.cuda.synchronize()
        timings.append(perf_counter() - start)
    return statistics.median(timings), prepared


def _time_cuda_input_transfer(
    prepared: PreparedSCMBatch,
    shifts: np.ndarray,
    scales: np.ndarray,
    repetitions: int,
) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
    """Measure validation and host-to-device parameter transfer."""
    timings = []
    tensors = prepared.prepare_interventions(shifts, scales)
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = perf_counter()
        tensors = prepared.prepare_interventions(shifts, scales)
        torch.cuda.synchronize()
        timings.append(perf_counter() - start)
    return statistics.median(timings), tensors


def _time_cuda_compute(
    prepared: PreparedSCMBatch,
    parameters: tuple[torch.Tensor, torch.Tensor],
    repetitions: int,
) -> tuple[float, float, tuple[torch.Tensor, torch.Tensor]]:
    """Measure synchronized kernel events and complete compute wall time."""
    for _ in range(2):
        prepared.interventional_moments_from_tensors(*parameters)
    torch.cuda.synchronize()

    event_timings = []
    wall_timings = []
    result = prepared.interventional_moments_from_tensors(*parameters)
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = perf_counter()
        start_event.record()
        result = prepared.interventional_moments_from_tensors(*parameters)
        end_event.record()
        end_event.synchronize()
        wall_timings.append(perf_counter() - wall_start)
        event_timings.append(start_event.elapsed_time(end_event) / 1_000.0)
    return (
        statistics.median(event_timings),
        statistics.median(wall_timings),
        result,
    )


def _time_cuda_output_transfer(
    result: tuple[torch.Tensor, torch.Tensor],
    repetitions: int,
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """Measure device-to-host transfer and detached NumPy copies."""
    timings = []
    arrays = (
        result[0].detach().cpu().numpy().copy(),
        result[1].detach().cpu().numpy().copy(),
    )
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = perf_counter()
        arrays = (
            result[0].detach().cpu().numpy().copy(),
            result[1].detach().cpu().numpy().copy(),
        )
        torch.cuda.synchronize()
        timings.append(perf_counter() - start)
    return statistics.median(timings), arrays


def _errors(
    expected: tuple[np.ndarray, np.ndarray],
    actual: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float]:
    """Return maximum absolute and scale-aware relative moment errors."""
    expected_flat = np.concatenate([array.ravel() for array in expected])
    actual_flat = np.concatenate([array.ravel() for array in actual])
    absolute = np.abs(expected_flat - actual_flat)
    denominator = np.maximum(np.abs(expected_flat), np.finfo(float).tiny)
    return float(absolute.max(initial=0.0)), float(
        (absolute / denominator).max(initial=0.0)
    )


def _benchmark_case(
    backend: TorchBackend,
    case: _Case,
) -> dict[str, object]:
    """Benchmark one controlled CPU/CUDA shape and batching mode."""
    expected = _cpu_moments(case)
    cpu_timing = _time_cpu(lambda: _cpu_moments(case))
    repetitions = _timing_repetitions(case)
    preparation_time, prepared = _time_cuda_preparation(
        backend,
        case.torch_coefficients,
    )
    input_transfer_time, parameters = _time_cuda_input_transfer(
        prepared,
        case.shifts,
        case.scales,
        repetitions,
    )

    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    compute_time, compute_wall_time, result = _time_cuda_compute(
        prepared,
        parameters,
        repetitions,
    )
    peak_memory = max(
        0,
        torch.cuda.max_memory_allocated() - baseline_memory,
    )
    output_transfer_time, actual = _time_cuda_output_transfer(
        result,
        repetitions,
    )
    maximum_absolute_error, maximum_relative_error = _errors(expected, actual)
    transfer_time = input_transfer_time + output_transfer_time
    total_time = transfer_time + compute_wall_time
    cpu_time = float(cpu_timing["median_seconds"])
    return {
        "operation": "interventional_moments",
        "mode": case.mode,
        "n_nodes": case.n_nodes,
        "batch_size": (
            case.effective_batch_size if case.mode == "independent_scm_batch" else 1
        ),
        "candidate_count": (
            case.effective_batch_size if case.mode == "candidate_batch" else 1
        ),
        "effective_batch_size": case.effective_batch_size,
        "sample_count": 0,
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
        "cuda_warmup_runs": 2,
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
        "torch_version": torch.__version__,
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


def run_benchmark(*, quick: bool) -> dict[str, object]:
    """Run synchronized NumPy/CUDA moment comparisons."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Torch moments benchmark.")
    backend = TorchBackend(device="cuda")
    torch.ones(1, dtype=torch.float64, device="cuda").sum()
    torch.cuda.synchronize()

    records = []
    for index, (mode, n_nodes, batch_size) in enumerate(_case_grid(quick=quick)):
        case = _make_case(
            mode,
            n_nodes,
            batch_size,
            BASE_SEED + index,
        )
        records.append(_benchmark_case(backend, case))

    best_record = max(records, key=lambda item: float(item["speedup"]))
    worst_record = min(records, key=lambda item: float(item["speedup"]))
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": _environment(),
        "methodology": {
            "timer": "time.perf_counter plus synchronized CUDA events",
            "cpu_reference": "looped canonical NumPy backend",
            "gpu_backend": "prepared batched Torch float64 moments",
            "gpu_warmup_runs": 2,
            "gpu_context_initialization_excluded": True,
            "coefficient_preparation_reported_separately": True,
            "input_and_output_transfers_reported_separately": True,
            "total_time_definition": (
                "input transfer + synchronized compute wall + output transfer"
            ),
            "speedup_definition": "CPU time / steady-state GPU total time",
            "node_counts": list(QUICK_NODE_COUNTS if quick else FULL_NODE_COUNTS),
            "batching_modes": list(MODES),
            "requested_batch_sizes": list(
                QUICK_BATCH_SIZES if quick else FULL_BATCH_SIZES
            ),
            "large_case_limits": {
                "200_nodes": 32,
                "500_nodes": 8,
            },
            "numerical_fixture": (
                "sparse upper-triangular DAG coefficients with expected total "
                "degree near four, magnitudes in [0.05, 0.20], shifts sampled "
                "from N(0, 0.25), and scales sampled in [0.75, 1.25]"
            ),
        },
        "summary": {
            "record_count": len(records),
            "gpu_faster_record_count": sum(
                float(item["speedup"]) > 1.0 for item in records
            ),
            "maximum_absolute_error": max(
                float(item["maximum_absolute_error"]) for item in records
            ),
            "maximum_relative_error": max(
                float(item["maximum_relative_error"]) for item in records
            ),
            "best_speedup": best_record["speedup"],
            "best_case": {
                key: best_record[key]
                for key in ("mode", "n_nodes", "effective_batch_size")
            },
            "worst_speedup": worst_record["speedup"],
            "worst_case": {
                key: worst_record[key]
                for key in ("mode", "n_nodes", "effective_batch_size")
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
        description="Benchmark NumPy and CUDA analytic SCM moments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("torch_moments.json"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced node and batch grid for smoke validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    result = run_benchmark(quick=namespace.quick)
    _write_result(result, namespace.output)
    print(
        f"wrote {len(result['operations'])} Torch moment benchmark records to "
        f"{namespace.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
