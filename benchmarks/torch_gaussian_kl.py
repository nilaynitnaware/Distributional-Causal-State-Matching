"""Benchmark canonical NumPy KL against cached batched CUDA KL."""

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

import numpy as np
import torch

from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from numerical.torch_backend import TorchBackend
from numerical.torch_gaussian import PreparedGaussianTarget

BASE_SEED = 20_260_727
FULL_NODE_COUNTS = (5, 10, 20, 50, 100, 200, 500)
QUICK_NODE_COUNTS = (5, 20, 100)
FULL_CANDIDATE_COUNTS = (1, 8, 32, 128)
QUICK_CANDIDATE_COUNTS = (1, 8, 32)


@dataclass(frozen=True)
class _Case:
    """Durable NumPy inputs prepared outside all benchmark timings."""

    n_nodes: int
    candidate_count: int
    mean_p: np.ndarray
    covariance_p: np.ndarray
    mean_q: np.ndarray
    covariance_q: np.ndarray


def _candidate_counts(n_nodes: int, *, quick: bool) -> tuple[int, ...]:
    """Bound dense candidate batches at the largest dimensions."""
    requested = QUICK_CANDIDATE_COUNTS if quick else FULL_CANDIDATE_COUNTS
    if n_nodes >= 500:
        return tuple(size for size in requested if size <= 8)
    if n_nodes >= 200:
        return tuple(size for size in requested if size <= 32)
    return requested


def _case_grid(*, quick: bool) -> tuple[tuple[int, int], ...]:
    """Return representative dimensions and candidate batch sizes."""
    node_counts = QUICK_NODE_COUNTS if quick else FULL_NODE_COUNTS
    return tuple(
        (n_nodes, candidate_count)
        for n_nodes in node_counts
        for candidate_count in _candidate_counts(n_nodes, quick=quick)
    )


def _well_conditioned_covariances(
    count: int,
    n_nodes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build dense SPD matrices from a diagonal plus a rank-four update."""
    rank = min(4, n_nodes)
    factors = rng.normal(
        0.0,
        0.20 / math.sqrt(rank),
        size=(count, n_nodes, rank),
    )
    covariance = factors @ factors.transpose(0, 2, 1)
    diagonal = rng.uniform(0.75, 1.25, size=(count, n_nodes))
    indices = np.arange(n_nodes)
    covariance[:, indices, indices] += diagonal
    return covariance


def _make_case(
    n_nodes: int,
    candidate_count: int,
    seed: int,
) -> _Case:
    """Create one shared target and a batch of candidate Gaussians."""
    rng = np.random.default_rng(seed)
    return _Case(
        n_nodes=n_nodes,
        candidate_count=candidate_count,
        mean_p=rng.normal(0.0, 0.5, size=(candidate_count, n_nodes)),
        covariance_p=_well_conditioned_covariances(
            candidate_count,
            n_nodes,
            rng,
        ),
        mean_q=rng.normal(0.0, 0.5, size=n_nodes),
        covariance_q=_well_conditioned_covariances(1, n_nodes, rng)[0],
    )


def _cpu_kl(case: _Case) -> np.ndarray:
    """Evaluate the current canonical NumPy implementation candidate by candidate."""
    return np.array(
        [
            gaussian_kl_target_to_candidate(
                target_mean=case.mean_q,
                target_covariance=case.covariance_q,
                candidate_mean=case.mean_p[index],
                candidate_covariance=case.covariance_p[index],
            )
            for index in range(case.candidate_count)
        ]
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
    dense_elements = case.candidate_count * case.n_nodes * case.n_nodes
    return 7 if dense_elements <= 1_000_000 else 3


def _time_cuda_target_preparation(
    backend: TorchBackend,
    case: _Case,
) -> tuple[float, PreparedGaussianTarget]:
    """Measure target H2D transfer, validation, Cholesky, and log determinant."""
    timings = []
    prepared = backend.prepare_gaussian_target(case.mean_q, case.covariance_q)
    for _ in range(3):
        torch.cuda.synchronize()
        start = perf_counter()
        prepared = backend.prepare_gaussian_target(
            case.mean_q,
            case.covariance_q,
        )
        torch.cuda.synchronize()
        timings.append(perf_counter() - start)
    return statistics.median(timings), prepared


def _time_cuda_input_transfer(
    prepared: PreparedGaussianTarget,
    case: _Case,
    repetitions: int,
) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
    """Measure candidate validation and host-to-device transfer."""
    timings = []
    tensors = prepared.prepare_candidates(case.mean_p, case.covariance_p)
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = perf_counter()
        tensors = prepared.prepare_candidates(case.mean_p, case.covariance_p)
        torch.cuda.synchronize()
        timings.append(perf_counter() - start)
    return statistics.median(timings), tensors


def _time_cuda_compute(
    prepared: PreparedGaussianTarget,
    candidates: tuple[torch.Tensor, torch.Tensor],
    repetitions: int,
) -> tuple[float, float, torch.Tensor]:
    """Measure synchronized CUDA events and full compute wall time."""
    event_timings = []
    wall_timings = []
    result = prepared.gaussian_kl_target_to_candidate_from_tensors(*candidates)
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = perf_counter()
        start_event.record()
        result = prepared.gaussian_kl_target_to_candidate_from_tensors(*candidates)
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
    result: torch.Tensor,
    repetitions: int,
) -> tuple[float, np.ndarray]:
    """Measure result D2H transfer and a detached NumPy copy."""
    timings = []
    array = result.detach().cpu().numpy().copy()
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = perf_counter()
        array = result.detach().cpu().numpy().copy()
        torch.cuda.synchronize()
        timings.append(perf_counter() - start)
    return statistics.median(timings), array


def _errors(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Return maximum absolute and scale-aware relative KL errors."""
    absolute = np.abs(expected - actual)
    denominator = np.maximum(np.abs(expected), 1e-10)
    return float(absolute.max(initial=0.0)), float(
        (absolute / denominator).max(initial=0.0)
    )


def _benchmark_case(
    backend: TorchBackend,
    case: _Case,
) -> dict[str, object]:
    """Benchmark one controlled shared-target candidate batch."""
    expected = _cpu_kl(case)
    cpu_timing = _time_cpu(lambda: _cpu_kl(case))
    repetitions = _timing_repetitions(case)
    preparation_time, prepared = _time_cuda_target_preparation(backend, case)
    input_transfer_time, candidates = _time_cuda_input_transfer(
        prepared,
        case,
        repetitions,
    )

    for _ in range(2):
        prepared.gaussian_kl_target_to_candidate_from_tensors(*candidates)
    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    compute_time, compute_wall_time, result = _time_cuda_compute(
        prepared,
        candidates,
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
        "operation": "gaussian_kl_target_to_candidate",
        "kl_direction": "target_to_candidate",
        "mode": "candidate_batch_shared_target",
        "n_nodes": case.n_nodes,
        "batch_size": 1,
        "candidate_count": case.candidate_count,
        "effective_batch_size": case.candidate_count,
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


def run_benchmark(*, quick: bool) -> dict[str, object]:
    """Run synchronized canonical NumPy versus cached CUDA KL comparisons."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Torch Gaussian KL benchmark.")
    backend = TorchBackend(device="cuda")
    torch.ones(1, dtype=torch.float64, device="cuda").sum()
    torch.cuda.synchronize()

    records = []
    for index, (n_nodes, candidate_count) in enumerate(_case_grid(quick=quick)):
        case = _make_case(
            n_nodes,
            candidate_count,
            BASE_SEED + index,
        )
        records.append(_benchmark_case(backend, case))

    best_record = max(records, key=lambda item: float(item["speedup"]))
    worst_record = min(records, key=lambda item: float(item["speedup"]))
    return {
        "schema_version": 2,
        "kl_direction": "target_to_candidate",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": _environment(),
        "methodology": {
            "timer": "time.perf_counter plus synchronized CUDA events",
            "cpu_reference": "looped canonical NumPy Gaussian KL",
            "gpu_backend": "cached-target batched Torch float64 Gaussian KL",
            "gpu_warmup_runs": 2,
            "gpu_context_initialization_excluded": True,
            "target_preparation_reported_separately": True,
            "input_and_output_transfers_reported_separately": True,
            "total_time_definition": (
                "candidate input transfer + synchronized compute wall + "
                "result output transfer"
            ),
            "speedup_definition": "CPU time / steady-state GPU total time",
            "node_counts": list(QUICK_NODE_COUNTS if quick else FULL_NODE_COUNTS),
            "batching_mode": "candidate_batch_shared_target",
            "requested_candidate_counts": list(
                QUICK_CANDIDATE_COUNTS if quick else FULL_CANDIDATE_COUNTS
            ),
            "large_case_limits": {
                "200_nodes": 32,
                "500_nodes": 8,
            },
            "peak_memory_definition": (
                "incremental CUDA allocator peak above prepared target and inputs"
            ),
            "numerical_fixture": (
                "dense SPD covariance matrices formed from diagonals in "
                "[0.75, 1.25] plus rank-four updates"
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
                key: best_record[key] for key in ("n_nodes", "candidate_count")
            },
            "worst_speedup": worst_record["speedup"],
            "worst_case": {
                key: worst_record[key] for key in ("n_nodes", "candidate_count")
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
        description="Benchmark NumPy and cached CUDA Gaussian KL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("torch_gaussian_kl.json"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced node and candidate grid for smoke validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    result = run_benchmark(quick=namespace.quick)
    _write_result(result, namespace.output)
    print(
        f"wrote {len(result['operations'])} Torch Gaussian KL benchmark "
        f"records to {namespace.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
