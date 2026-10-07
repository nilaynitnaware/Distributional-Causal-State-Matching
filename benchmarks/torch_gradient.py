"""Benchmark canonical analytic KL gradients against prepared Torch autograd."""

from __future__ import annotations

import argparse
import json
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
from graphs.problem_gen import PublicProblemView
from graphs.scm import LinearGaussianSCM
from graphs.seeds import AlgorithmSeeds
from matching.types import FinalSupportBudget, ParameterCoordinate
from numerical.torch_backend import TorchBackend
from numerical.torch_gradient import PreparedInterventionKLLoss
from oracle.analytic_gaussian_kl import OracleAnalyticGaussianKLLoss
from oracle.oracle_optimizer import OracleActiveCoordinateOptimizer
from oracle.torch_gaussian_kl import TorchOracleGaussianKLLoss

BASE_SEED = 20_260_729
FULL_NODE_COUNTS = (5, 10, 20, 50, 100, 200, 500)
QUICK_NODE_COUNTS = (5, 20, 100)
FULL_CANDIDATE_COUNTS = (1, 8, 32, 128)
QUICK_CANDIDATE_COUNTS = (1, 8)
OPTIMIZER_NODE_COUNTS = (5, 20, 50)


@dataclass(frozen=True)
class _Case:
    """Canonical objects and durable arrays prepared outside benchmark timings."""

    n_nodes: int
    candidate_count: int
    scm: LinearGaussianSCM
    view: PublicProblemView
    candidates: tuple[ShiftScaleIntervention, ...]
    shifts: np.ndarray
    log_scales: np.ndarray


def _candidate_counts(n_nodes: int, *, quick: bool) -> tuple[int, ...]:
    """Bound the canonical per-candidate analytic-gradient workload."""
    requested = QUICK_CANDIDATE_COUNTS if quick else FULL_CANDIDATE_COUNTS
    if n_nodes >= 200:
        maximum = 1
    elif n_nodes >= 100:
        maximum = 8
    elif n_nodes >= 50:
        maximum = 32
    else:
        maximum = 128
    return tuple(count for count in requested if count <= maximum)


def _case_grid(*, quick: bool) -> tuple[tuple[int, int], ...]:
    """Return representative node and candidate-batch combinations."""
    node_counts = QUICK_NODE_COUNTS if quick else FULL_NODE_COUNTS
    return tuple(
        (n_nodes, candidate_count)
        for n_nodes in node_counts
        for candidate_count in _candidate_counts(n_nodes, quick=quick)
    )


def _coefficient_matrix(
    n_nodes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a sparse upper-triangular DAG with expected total degree four."""
    edge_probability = min(1.0, 4.0 / max(1, n_nodes - 1))
    support = np.triu(
        rng.random((n_nodes, n_nodes)) < edge_probability,
        k=1,
    )
    magnitudes = rng.uniform(0.05, 0.20, size=(n_nodes, n_nodes))
    signs = np.where(rng.integers(0, 2, size=(n_nodes, n_nodes)), 1.0, -1.0)
    return np.where(support, magnitudes * signs, 0.0)


def _scm_from_coefficients(coefficients: np.ndarray) -> LinearGaussianSCM:
    """Construct the canonical CPU SCM for one coefficient matrix."""
    dag = nx.DiGraph()
    dag.add_nodes_from(range(coefficients.shape[0]))
    dag.add_edges_from(
        (int(parent), int(child)) for parent, child in np.argwhere(coefficients != 0.0)
    )
    return LinearGaussianSCM(dag, coefficients)


def _make_case(
    n_nodes: int,
    candidate_count: int,
    seed: int,
) -> _Case:
    """Create one shared SCM/target and a candidate intervention batch."""
    rng = np.random.default_rng(seed)
    scm = _scm_from_coefficients(_coefficient_matrix(n_nodes, rng))
    target = ShiftScaleIntervention(
        rng.normal(0.0, 0.25, size=n_nodes),
        rng.uniform(0.75, 1.25, size=n_nodes),
    )
    target_mean, target_covariance = scm.interventional_moments(target)
    view = PublicProblemView(
        observational_samples=np.zeros((1, n_nodes)),
        target_mean=target_mean,
        target_covariance=target_covariance,
        current_intervention=ShiftScaleIntervention.identity(n_nodes),
        algorithm_seeds=AlgorithmSeeds(seed + 1, seed + 2),
    )
    shifts = rng.normal(0.0, 0.25, size=(candidate_count, n_nodes))
    scales = rng.uniform(0.75, 1.25, size=(candidate_count, n_nodes))
    candidates = tuple(
        ShiftScaleIntervention(shift, scale)
        for shift, scale in zip(shifts, scales, strict=True)
    )
    return _Case(
        n_nodes=n_nodes,
        candidate_count=candidate_count,
        scm=scm,
        view=view,
        candidates=candidates,
        shifts=shifts,
        log_scales=np.log(scales),
    )


def _cpu_gradients(
    case: _Case,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loop over the current canonical analytic value-and-gradient method."""
    loss = OracleAnalyticGaussianKLLoss(case.scm)
    results = tuple(
        loss.value_and_gradient(candidate, case.view) for candidate in case.candidates
    )
    return (
        np.array([result[0] for result in results]),
        np.stack([result[1] for result in results]),
        np.stack([result[2] for result in results]),
    )


def _cpu_repetitions(case: _Case) -> int:
    """Use bounded repeats because the 500-node reference call is expensive."""
    if case.n_nodes >= 500:
        return 1
    if case.n_nodes >= 100:
        return 3
    if case.n_nodes >= 50:
        return 5
    return 7


def _torch_repetitions(case: _Case) -> int:
    """Use more repeats for small resident autograd workloads."""
    dense_elements = case.candidate_count * case.n_nodes * case.n_nodes
    if dense_elements >= 10_000_000:
        return 1
    if dense_elements >= 1_000_000:
        return 3
    return 7


def _time_cpu_reference(
    case: _Case,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return canonical timing and the final reference result."""
    repetitions = _cpu_repetitions(case)
    if repetitions > 1:
        _cpu_gradients(case)
    timings = []
    result = None
    for _ in range(repetitions):
        start = perf_counter()
        result = _cpu_gradients(case)
        timings.append(perf_counter() - start)
    assert result is not None
    return (
        {
            "median_seconds": statistics.median(timings),
            "minimum_seconds": min(timings),
            "maximum_seconds": max(timings),
            "repetitions": repetitions,
            "timings_seconds": timings,
        },
        result,
    )


def _synchronize(device: str) -> None:
    """Synchronize only CUDA work."""
    if device == "cuda":
        torch.cuda.synchronize()


def _time_preparation(
    case: _Case,
    device: str,
) -> tuple[float, PreparedInterventionKLLoss]:
    """Measure SCM/target transfer, validation, and target Cholesky."""
    timings = []
    prepared = None
    for _ in range(3):
        _synchronize(device)
        start = perf_counter()
        backend = TorchBackend(device=device)
        prepared = PreparedInterventionKLLoss(
            backend.prepare_scm_batch(case.scm.coefficients),
            backend.prepare_gaussian_target(
                case.view.target_mean,
                case.view.target_covariance,
            ),
        )
        _synchronize(device)
        timings.append(perf_counter() - start)
    assert prepared is not None
    return statistics.median(timings), prepared


def _time_input_preparation(
    prepared: PreparedInterventionKLLoss,
    case: _Case,
    repetitions: int,
) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
    """Measure candidate validation, transfer, and batch materialization."""
    timings = []
    parameters = None
    for _ in range(repetitions):
        _synchronize(prepared.device)
        start = perf_counter()
        parameters = prepared.prepare_parameters(case.shifts, case.log_scales)
        _synchronize(prepared.device)
        timings.append(perf_counter() - start)
    assert parameters is not None
    return statistics.median(timings), parameters


def _time_torch_cpu_compute(
    prepared: PreparedInterventionKLLoss,
    parameters: tuple[torch.Tensor, torch.Tensor],
    repetitions: int,
) -> tuple[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Measure prepared Torch CPU autograd wall time."""
    prepared.value_and_gradient_from_tensors(*parameters)
    timings = []
    result = None
    for _ in range(repetitions):
        start = perf_counter()
        result = prepared.value_and_gradient_from_tensors(*parameters)
        timings.append(perf_counter() - start)
    assert result is not None
    return statistics.median(timings), result


def _time_cuda_compute(
    prepared: PreparedInterventionKLLoss,
    parameters: tuple[torch.Tensor, torch.Tensor],
    repetitions: int,
) -> tuple[
    float,
    float,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Measure resident CUDA autograd with events and synchronized wall time."""
    event_timings = []
    wall_timings = []
    result = None
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = perf_counter()
        start_event.record()
        result = prepared.value_and_gradient_from_tensors(*parameters)
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


def _time_output_transfer(
    result: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    repetitions: int,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Measure detachment, D2H transfer where applicable, and NumPy copies."""
    timings = []
    arrays = None
    for _ in range(repetitions):
        _synchronize(result[0].device.type)
        start = perf_counter()
        arrays = tuple(tensor.cpu().numpy().copy() for tensor in result)
        _synchronize(result[0].device.type)
        timings.append(perf_counter() - start)
    assert arrays is not None
    return statistics.median(timings), arrays


def _errors(
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float]:
    """Return maximum absolute and scale-aware relative errors across outputs."""
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for expected_array, actual_array in zip(expected, actual, strict=True):
        absolute = np.abs(expected_array - actual_array)
        relative = absolute / np.maximum(np.abs(expected_array), 1e-10)
        maximum_absolute_error = max(
            maximum_absolute_error,
            float(absolute.max(initial=0.0)),
        )
        maximum_relative_error = max(
            maximum_relative_error,
            float(relative.max(initial=0.0)),
        )
    return maximum_absolute_error, maximum_relative_error


def _benchmark_case(
    case: _Case,
    cpu_timing: dict[str, object],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    """Benchmark one shared-SCM/target candidate gradient batch."""
    repetitions = _torch_repetitions(case)

    torch_cpu_preparation, torch_cpu = _time_preparation(case, "cpu")
    torch_cpu_input, torch_cpu_parameters = _time_input_preparation(
        torch_cpu,
        case,
        repetitions,
    )
    torch_cpu_compute, torch_cpu_result = _time_torch_cpu_compute(
        torch_cpu,
        torch_cpu_parameters,
        repetitions,
    )
    torch_cpu_output, torch_cpu_actual = _time_output_transfer(
        torch_cpu_result,
        repetitions,
    )
    torch_cpu_absolute_error, torch_cpu_relative_error = _errors(
        expected,
        torch_cpu_actual,
    )

    cuda_preparation, cuda = _time_preparation(case, "cuda")
    cuda_input, cuda_parameters = _time_input_preparation(
        cuda,
        case,
        repetitions,
    )
    warm_result = cuda.value_and_gradient_from_tensors(*cuda_parameters)
    torch.cuda.synchronize()
    del warm_result
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    cuda_compute, cuda_compute_wall, cuda_result = _time_cuda_compute(
        cuda,
        cuda_parameters,
        repetitions,
    )
    peak_memory = max(0, torch.cuda.max_memory_allocated() - baseline_memory)
    cuda_output, cuda_actual = _time_output_transfer(cuda_result, repetitions)
    cuda_absolute_error, cuda_relative_error = _errors(expected, cuda_actual)

    cpu_time = float(cpu_timing["median_seconds"])
    torch_cpu_total = torch_cpu_input + torch_cpu_compute + torch_cpu_output
    cuda_transfer = cuda_input + cuda_output
    cuda_total = cuda_transfer + cuda_compute_wall
    return {
        "operation": "gaussian_kl_target_to_candidate_value_and_gradient",
        "kl_direction": "target_to_candidate",
        "mode": "candidate_batch_shared_scm_and_target",
        "n_nodes": case.n_nodes,
        "batch_size": 1,
        "candidate_count": case.candidate_count,
        "effective_batch_size": case.candidate_count,
        "dtype": "float64",
        "cpu_backend": "canonical_numpy_analytic",
        "cpu_time": cpu_time,
        "cpu_timing": cpu_timing,
        "torch_cpu_preparation_time": torch_cpu_preparation,
        "torch_cpu_input_transfer_time": torch_cpu_input,
        "torch_cpu_compute_wall_time": torch_cpu_compute,
        "torch_cpu_output_transfer_time": torch_cpu_output,
        "torch_cpu_total_time": torch_cpu_total,
        "torch_cpu_first_call_total_time": (torch_cpu_preparation + torch_cpu_total),
        "torch_cpu_prepared_memory": torch_cpu.prepared_bytes,
        "torch_cpu_maximum_absolute_error": torch_cpu_absolute_error,
        "torch_cpu_maximum_relative_error": torch_cpu_relative_error,
        "torch_cpu_speedup": cpu_time / torch_cpu_total,
        "backend": "torch",
        "device": "cuda",
        "preparation_time": cuda_preparation,
        "input_transfer_time": cuda_input,
        "output_transfer_time": cuda_output,
        "transfer_time": cuda_transfer,
        "compute_time": cuda_compute,
        "compute_wall_time": cuda_compute_wall,
        "total_time": cuda_total,
        "first_call_total_time": cuda_preparation + cuda_total,
        "prepared_memory": cuda.prepared_bytes,
        "peak_memory": peak_memory,
        "maximum_absolute_error": cuda_absolute_error,
        "maximum_relative_error": cuda_relative_error,
        "speedup": cpu_time / cuda_total,
        "compute_only_speedup": cpu_time / cuda_compute,
        "torch_repetitions": repetitions,
        "cuda_warmup_runs": 1,
    }


def _make_optimizer_case(
    n_nodes: int,
    seed: int,
) -> tuple[
    LinearGaussianSCM,
    PublicProblemView,
    ShiftScaleIntervention,
    tuple[ParameterCoordinate, ...],
]:
    """Create a reachable four-coordinate target for optimizer parity/timing."""
    rng = np.random.default_rng(seed)
    scm = _scm_from_coefficients(_coefficient_matrix(n_nodes, rng))
    current = ShiftScaleIntervention.identity(n_nodes)
    target_shift = np.zeros(n_nodes)
    target_scale = np.ones(n_nodes)
    target_shift[:2] = (0.35, -0.20)
    target_scale[:2] = (1.20, 0.80)
    target = ShiftScaleIntervention(target_shift, target_scale)
    target_mean, target_covariance = scm.interventional_moments(target)
    view = PublicProblemView(
        observational_samples=np.zeros((1, n_nodes)),
        target_mean=target_mean,
        target_covariance=target_covariance,
        current_intervention=current,
        algorithm_seeds=AlgorithmSeeds(seed + 1, seed + 2),
    )
    active = (
        ParameterCoordinate(0, "shift"),
        ParameterCoordinate(0, "log_scale"),
        ParameterCoordinate(1, "shift"),
        ParameterCoordinate(1, "log_scale"),
    )
    return scm, view, current, active


def _time_optimizer(
    function: Callable[[], object],
    repetitions: int,
) -> tuple[float, object]:
    """Return median end-to-end optimizer wall time and final proposal."""
    timings = []
    proposal = None
    for _ in range(repetitions):
        start = perf_counter()
        proposal = function()
        timings.append(perf_counter() - start)
    if proposal is None:
        raise RuntimeError("optimizer timing produced no proposal.")
    return statistics.median(timings), proposal


def _benchmark_optimizer(n_nodes: int, seed: int) -> dict[str, object]:
    """Compare canonical and hybrid SciPy/CUDA optimizer wall times."""
    scm, view, current, active = _make_optimizer_case(n_nodes, seed)
    budget = FinalSupportBudget(len(active))
    cpu_optimizer = OracleActiveCoordinateOptimizer(
        loss=OracleAnalyticGaussianKLLoss(scm),
        scale_min=0.25,
        scale_max=3.0,
        max_iterations=200,
    )
    cpu_time, cpu_proposal = _time_optimizer(
        lambda: cpu_optimizer.optimize(current, active, view, budget),
        3,
    )

    torch.cuda.synchronize()
    start = perf_counter()
    cuda_loss = TorchOracleGaussianKLLoss(scm, device="cuda")
    torch.cuda.synchronize()
    loss_preparation_time = perf_counter() - start
    cuda_optimizer = OracleActiveCoordinateOptimizer(
        loss=cuda_loss,
        scale_min=0.25,
        scale_max=3.0,
        max_iterations=200,
    )
    torch.cuda.synchronize()
    first_start = perf_counter()
    first_proposal = cuda_optimizer.optimize(current, active, view, budget)
    torch.cuda.synchronize()
    first_optimizer_time = perf_counter() - first_start

    def optimize_cuda() -> object:
        proposal = cuda_optimizer.optimize(current, active, view, budget)
        torch.cuda.synchronize()
        return proposal

    cuda_steady_time, cuda_proposal = _time_optimizer(optimize_cuda, 3)
    shift_error = np.max(
        np.abs(cpu_proposal.intervention.shift - cuda_proposal.intervention.shift),
        initial=0.0,
    )
    log_scale_error = np.max(
        np.abs(
            np.log(cpu_proposal.intervention.scale)
            - np.log(cuda_proposal.intervention.scale)
        ),
        initial=0.0,
    )
    return {
        "operation": "oracle_optimizer",
        "algorithm": "SciPy L-BFGS-B",
        "n_nodes": n_nodes,
        "active_coordinate_count": len(active),
        "dtype": "float64",
        "cpu_backend": "canonical_numpy_analytic_gradient",
        "gpu_backend": "prepared_torch_cuda_autograd_gradient",
        "cpu_optimizer_time": cpu_time,
        "gpu_loss_preparation_time": loss_preparation_time,
        "gpu_first_optimizer_time": first_optimizer_time,
        "gpu_first_total_time": loss_preparation_time + first_optimizer_time,
        "gpu_steady_optimizer_time": cuda_steady_time,
        "steady_speedup": cpu_time / cuda_steady_time,
        "maximum_parameter_absolute_error": float(max(shift_error, log_scale_error)),
        "cpu_predicted_loss": cpu_proposal.predicted_loss,
        "gpu_first_predicted_loss": first_proposal.predicted_loss,
        "gpu_predicted_loss": cuda_proposal.predicted_loss,
        "cpu_support_cost": cpu_proposal.intervention.support_cost(),
        "gpu_support_cost": cuda_proposal.intervention.support_cost(),
        "target_preparation_count": cuda_loss.target_preparation_count,
        "stopping_and_acceptance": "canonical OracleActiveCoordinateOptimizer",
    }


def _environment() -> dict[str, object]:
    """Return relevant CPU, Torch, and CUDA runtime metadata."""
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
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "python_executable": sys.executable,
    }


def run_benchmark(*, quick: bool) -> dict[str, object]:
    """Run canonical CPU, prepared Torch CPU, and prepared CUDA comparisons."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Torch gradient benchmark.")
    cases = [
        _make_case(n_nodes, candidate_count, BASE_SEED + index)
        for index, (n_nodes, candidate_count) in enumerate(_case_grid(quick=quick))
    ]
    cpu_references = [_time_cpu_reference(case) for case in cases]

    torch.ones(1, dtype=torch.float64, device="cuda").sum()
    torch.cuda.synchronize()

    records = [
        _benchmark_case(case, cpu_timing, expected)
        for case, (cpu_timing, expected) in zip(
            cases,
            cpu_references,
            strict=True,
        )
    ]
    optimizer_counts = (5,) if quick else OPTIMIZER_NODE_COUNTS
    optimizer_records = [
        _benchmark_optimizer(n_nodes, BASE_SEED + 10_000 + n_nodes)
        for n_nodes in optimizer_counts
    ]
    best_record = max(records, key=lambda item: float(item["speedup"]))
    worst_record = min(records, key=lambda item: float(item["speedup"]))
    return {
        "schema_version": 2,
        "kl_direction": "target_to_candidate",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": _environment(),
        "methodology": {
            "timer": "time.perf_counter plus synchronized CUDA events",
            "cpu_reference": ("looped OracleAnalyticGaussianKLLoss.value_and_gradient"),
            "torch_cpu_backend": "prepared batched float64 autograd",
            "gpu_backend": "prepared batched float64 CUDA autograd",
            "gpu_context_initialization_excluded": True,
            "cpu_reference_phase_precedes_torch_execution": True,
            "scm_and_target_preparation_reported_separately": True,
            "input_and_output_transfers_reported_separately": True,
            "total_time_definition": (
                "candidate input preparation/transfer + synchronized gradient "
                "compute wall + value/full-gradient output transfer"
            ),
            "speedup_definition": "canonical CPU time / steady total time",
            "batching_mode": "candidate_batch_shared_scm_and_target",
            "node_counts": list(QUICK_NODE_COUNTS if quick else FULL_NODE_COUNTS),
            "requested_candidate_counts": list(
                QUICK_CANDIDATE_COUNTS if quick else FULL_CANDIDATE_COUNTS
            ),
            "large_case_limits": {
                "50_nodes": 32,
                "100_nodes": 8,
                "200_nodes": 1,
                "500_nodes": 1,
            },
            "canonical_reference_reason_for_limits": (
                "the analytic gradient performs a Python node loop containing "
                "dense solves; one calibrated 500-node call took 13.45 seconds"
            ),
            "candidate_factorization": (
                "form Sigma = T T^T, then use candidate Cholesky solves"
            ),
            "covariance_semantics": "Sigma = T T^T; no regularization",
            "cuda_bmm_policy": (
                "PyTorch native CUDA bmm is selected in a documented scoped "
                "context during backward; no host compiler is required"
            ),
            "optimizer_boundary": (
                "SciPy L-BFGS-B remains on CPU and calls the prepared CUDA loss; "
                "active masks, log-scale bounds, support budgets, stopping, and "
                "acceptance remain in OracleActiveCoordinateOptimizer"
            ),
        },
        "summary": {
            "record_count": len(records),
            "optimizer_record_count": len(optimizer_records),
            "gpu_faster_record_count": sum(
                float(item["speedup"]) > 1.0 for item in records
            ),
            "torch_cpu_faster_record_count": sum(
                float(item["torch_cpu_speedup"]) > 1.0 for item in records
            ),
            "maximum_absolute_error": max(
                float(item["maximum_absolute_error"]) for item in records
            ),
            "maximum_relative_error": max(
                float(item["maximum_relative_error"]) for item in records
            ),
            "torch_cpu_maximum_absolute_error": max(
                float(item["torch_cpu_maximum_absolute_error"]) for item in records
            ),
            "torch_cpu_maximum_relative_error": max(
                float(item["torch_cpu_maximum_relative_error"]) for item in records
            ),
            "best_speedup": best_record["speedup"],
            "best_case": {
                key: best_record[key] for key in ("n_nodes", "candidate_count")
            },
            "worst_speedup": worst_record["speedup"],
            "worst_case": {
                key: worst_record[key] for key in ("n_nodes", "candidate_count")
            },
            "gpu_faster_optimizer_record_count": sum(
                float(item["steady_speedup"]) > 1.0 for item in optimizer_records
            ),
            "optimizer_maximum_parameter_absolute_error": max(
                float(item["maximum_parameter_absolute_error"])
                for item in optimizer_records
            ),
        },
        "operations": records,
        "optimizer_operations": optimizer_records,
    }


def _write_result(result: dict[str, object], output: Path) -> None:
    """Write one compact, sorted JSON artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark analytic and Torch autograd Gaussian KL gradients.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("torch_gradient.json"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced node/candidate grid and one optimizer case.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested benchmark and write its JSON result."""
    namespace = build_parser().parse_args(argv)
    result = run_benchmark(quick=namespace.quick)
    _write_result(result, namespace.output)
    print(
        f"wrote {len(result['operations'])} Torch gradient records and "
        f"{len(result['optimizer_operations'])} optimizer records to "
        f"{namespace.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
