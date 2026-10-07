"""Run the bounded GPU-10 end-to-end and large-batch readiness smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch

from benchmarks.experiment_batching import _case
from experiments.noiseless_unknown_graph_pilot import (
    NoiselessPilotConfig,
    PilotRunSpec,
    _problem_for_spec,
    _scheduler_settings,
)
from graphs.problem_gen import GeneratedProblem
from matching.scheduler import SchedulerResult
from matching.types import FinalSupportBudget, ParameterCoordinate
from numerical import ComputeConfig
from numerical.torch_backend import TorchBackend
from oracle.analytic_gaussian_kl import OracleAnalyticGaussianKLLoss
from oracle.batching import (
    OracleInterventionBatchEvaluator,
    OracleInterventionBatchResult,
)
from oracle.known_dag_baseline import run_known_dag_oracle_baseline
from oracle.torch_gaussian_kl import TorchOracleGaussianKLLoss

SCHEMA_VERSION = 2
ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-8
MINIMUM_MEANINGFUL_SPEEDUP = 1.5
SMALL_SEED = 20_260_726
LARGE_SEED = 20_260_732
LARGE_N_NODES = 100
LARGE_BATCH_SIZE = 8
LARGE_REPETITIONS = 3


def _synchronize(device: str) -> None:
    """Synchronize only CUDA work included in a wall-clock interval."""
    if device == "cuda":
        torch.cuda.synchronize()


def _maximum_errors(
    actual: np.ndarray,
    expected: np.ndarray,
) -> tuple[float, float]:
    """Return maximum absolute and guarded relative errors."""
    absolute = np.abs(np.asarray(actual) - np.asarray(expected))
    relative = absolute / np.maximum(np.abs(expected), 1e-15)
    return (
        float(np.max(absolute, initial=0.0)),
        float(np.max(relative, initial=0.0)),
    )


def _within_tolerance(actual: np.ndarray, expected: np.ndarray) -> bool:
    """Apply the approved strict float64 analytic tolerance."""
    return bool(
        np.allclose(
            actual,
            expected,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )
    )


def _problem_fingerprint(problem: GeneratedProblem) -> str:
    """Hash every immutable scientific input without storing raw samples."""
    digest = hashlib.sha256()
    metadata = {
        "dag_edges": sorted(
            [int(parent), int(child)]
            for parent, child in problem.ground_truth.scm.dag.edges
        ),
        "algorithm_seeds": asdict(problem.public.algorithm_seeds),
        "simulation_seeds": asdict(problem.ground_truth.seeds),
    }
    digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    arrays = (
        ("coefficients", problem.ground_truth.scm.coefficients),
        ("observational_samples", problem.public.observational_samples),
        ("target_mean", problem.public.target_mean),
        ("target_covariance", problem.public.target_covariance),
        ("current_shift", problem.public.current_intervention.shift),
        ("current_scale", problem.public.current_intervention.scale),
        ("generating_shift", problem.ground_truth.generating_intervention.shift),
        ("generating_scale", problem.ground_truth.generating_intervention.scale),
    )
    for name, array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _active_shift_coordinates(
    problem: GeneratedProblem,
) -> tuple[ParameterCoordinate, ...]:
    """Return the known target support for this oracle-only smoke."""
    return tuple(
        ParameterCoordinate(int(node), "shift")
        for node in np.flatnonzero(
            problem.ground_truth.generating_intervention.shift != 0.0
        )
    )


def _accepted_interventions(result: SchedulerResult) -> list[dict[str, object]]:
    """Serialize the controlled smoke's one accepted scheduler proposal."""
    accepted_count = sum(
        record.phase == "EVALUATE" and record.accepted is True
        for record in result.history
    )
    if accepted_count != 1:
        raise RuntimeError(
            "GPU-10 controlled smoke expected exactly one accepted intervention."
        )
    return [
        {
            "shift": result.final_candidate.shift.tolist(),
            "scale": result.final_candidate.scale.tolist(),
        }
    ]


def _run_small_backend(
    problem: GeneratedProblem,
    config: NoiselessPilotConfig,
    *,
    device: Literal["cpu", "cuda"],
) -> dict[str, object]:
    """Run the same scheduler and experiment interface on one loss backend."""
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
    else:
        baseline_allocated = 0

    _synchronize(device)
    total_start = perf_counter()
    preparation_start = perf_counter()
    if device == "cpu":
        loss: Any = OracleAnalyticGaussianKLLoss(problem.ground_truth.scm)
        backend = "numpy"
    else:
        loss = TorchOracleGaussianKLLoss(
            problem.ground_truth.scm,
            device="cuda",
        )
        backend = "torch"
    _synchronize(device)
    preparation_seconds = perf_counter() - preparation_start

    initial_start = perf_counter()
    initial_loss = loss.value(problem.public.current_intervention, problem.public)
    _synchronize(device)
    initial_loss_seconds = perf_counter() - initial_start

    scheduler_start = perf_counter()
    result = run_known_dag_oracle_baseline(
        problem,
        _active_shift_coordinates(problem),
        _scheduler_settings(config),
        FinalSupportBudget(config.max_parameter_coordinates),
        config.max_experiment_trials,
        required_probe_count=1,
        probe_shift_magnitude=config.target_shift_magnitude,
        n_probe_samples=32,
        candidate_evaluation_requires_experiment=True,
        n_candidate_evaluation_samples=32,
        optimizer_max_iterations=config.optimizer_max_iterations,
        loss=loss,
    )
    _synchronize(device)
    scheduler_seconds = perf_counter() - scheduler_start

    moments_start = perf_counter()
    if device == "cpu":
        final_mean, final_covariance = problem.ground_truth.scm.interventional_moments(
            result.final_candidate
        )
    else:
        final_mean, final_covariance = TorchBackend(
            device="cuda"
        ).interventional_moments(
            problem.ground_truth.scm,
            result.final_candidate,
        )
    _synchronize(device)
    moments_seconds = perf_counter() - moments_start
    total_seconds = perf_counter() - total_start

    peak_allocated_bytes = (
        max(0, torch.cuda.max_memory_allocated() - baseline_allocated)
        if device == "cuda"
        else 0
    )
    ledger = result.budget_ledger
    return {
        "backend": backend,
        "device": device,
        "dtype": "float64",
        "initial_loss": initial_loss,
        "final_loss": result.final_loss,
        "accepted_interventions": _accepted_interventions(result),
        "accepted_intervention_count": ledger.accepted_matching_trials,
        "final_support_cost": result.final_candidate.support_cost(),
        "experiment_trials": {
            "exploration": ledger.exploration_trials,
            "candidate_evaluation": ledger.candidate_evaluation_trials,
            "total": ledger.experiment_trials_used,
        },
        "coordinate_uses": {
            "shift": ledger.shift_coordinate_uses,
            "scale": ledger.scale_coordinate_uses,
            "total": ledger.total_atomic_coordinate_uses,
        },
        "termination_reason": result.termination_reason,
        "scheduler_decisions": [
            {
                "phase": record.phase,
                "action_type": record.action_type,
                "accepted": record.accepted,
            }
            for record in result.history
        ],
        "final_moments": {
            "mean": final_mean.tolist(),
            "covariance": final_covariance.tolist(),
        },
        "runtime": {
            "preparation_seconds": preparation_seconds,
            "initial_loss_seconds": initial_loss_seconds,
            "scheduler_seconds": scheduler_seconds,
            "moments_seconds": moments_seconds,
            "total_seconds": total_seconds,
        },
        "prepared_bytes": int(getattr(loss, "prepared_bytes", 0)),
        "peak_allocated_bytes": int(peak_allocated_bytes),
        "target_preparation_count": int(getattr(loss, "target_preparation_count", 0)),
    }


def _accepted_parameter_array(record: dict[str, object]) -> np.ndarray:
    """Flatten the serialized accepted intervention for parity checks."""
    accepted = record["accepted_interventions"]
    assert isinstance(accepted, list) and len(accepted) == 1
    intervention = accepted[0]
    assert isinstance(intervention, dict)
    return np.concatenate((intervention["shift"], intervention["scale"]))


def _small_comparison(
    cpu: dict[str, object],
    cuda: dict[str, object],
) -> dict[str, object]:
    """Compare all required end-to-end numerical and scientific fields."""
    cpu_losses = np.array([cpu["initial_loss"], cpu["final_loss"]])
    cuda_losses = np.array([cuda["initial_loss"], cuda["final_loss"]])
    loss_errors = _maximum_errors(cuda_losses, cpu_losses)
    cpu_parameters = _accepted_parameter_array(cpu)
    cuda_parameters = _accepted_parameter_array(cuda)
    parameter_errors = _maximum_errors(cuda_parameters, cpu_parameters)
    cpu_moments = cpu["final_moments"]
    cuda_moments = cuda["final_moments"]
    assert isinstance(cpu_moments, dict) and isinstance(cuda_moments, dict)
    cpu_moment_array = np.concatenate(
        (
            np.ravel(cpu_moments["mean"]),
            np.ravel(cpu_moments["covariance"]),
        )
    )
    cuda_moment_array = np.concatenate(
        (
            np.ravel(cuda_moments["mean"]),
            np.ravel(cuda_moments["covariance"]),
        )
    )
    moment_errors = _maximum_errors(cuda_moment_array, cpu_moment_array)

    exact_decision_fields = (
        "accepted_intervention_count",
        "final_support_cost",
        "experiment_trials",
        "coordinate_uses",
        "termination_reason",
        "scheduler_decisions",
    )
    decisions = {field: cpu[field] == cuda[field] for field in exact_decision_fields}
    losses_close = _within_tolerance(cuda_losses, cpu_losses)
    parameters_close = _within_tolerance(cuda_parameters, cpu_parameters)
    moments_close = _within_tolerance(cuda_moment_array, cpu_moment_array)
    decisions_equivalent = all(decisions.values())
    return {
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "maximum_loss_absolute_error": loss_errors[0],
        "maximum_loss_relative_error": loss_errors[1],
        "maximum_accepted_intervention_absolute_error": parameter_errors[0],
        "maximum_accepted_intervention_relative_error": parameter_errors[1],
        "maximum_moment_absolute_error": moment_errors[0],
        "maximum_moment_relative_error": moment_errors[1],
        "losses_within_tolerance": losses_close,
        "accepted_interventions_within_tolerance": parameters_close,
        "moments_within_tolerance": moments_close,
        "exact_scientific_decision_checks": decisions,
        "scientific_decisions_equivalent": decisions_equivalent,
        "ready": (
            losses_close and parameters_close and moments_close and decisions_equivalent
        ),
    }


def _flatten_batch_result(result: OracleInterventionBatchResult) -> np.ndarray:
    """Flatten values and gradients for one batch parity comparison."""
    shift_gradients = result.shift_gradients
    log_scale_gradients = result.log_scale_gradients
    assert shift_gradients is not None and log_scale_gradients is not None
    return np.concatenate((result.values, *shift_gradients, *log_scale_gradients))


def _time_batch(
    problems: tuple[object, ...],
    candidates: tuple[object, ...],
    config: ComputeConfig,
) -> tuple[dict[str, object], OracleInterventionBatchResult]:
    """Time warmup-excluded preparation plus evaluation rounds."""
    warmup = OracleInterventionBatchEvaluator(
        problems,
        config,
        auto_operation="value_and_gradient",
    )
    warmup.evaluate(candidates, gradients=True)
    _synchronize(warmup.diagnostics.device)
    del warmup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    preparation_timings = []
    evaluation_timings = []
    total_timings = []
    peak_allocated_bytes = 0
    result = None
    for _ in range(LARGE_REPETITIONS):
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        total_start = perf_counter()
        preparation_start = perf_counter()
        evaluator = OracleInterventionBatchEvaluator(
            problems,
            config,
            auto_operation="value_and_gradient",
        )
        device = evaluator.diagnostics.device
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
    if result is None:
        raise RuntimeError("GPU-10 batch timing produced no result.")
    return (
        {
            "backend_diagnostics": asdict(result.backend_diagnostics),
            "repetitions": LARGE_REPETITIONS,
            "preparation_median_seconds": statistics.median(preparation_timings),
            "evaluation_median_seconds": statistics.median(evaluation_timings),
            "total_median_seconds": statistics.median(total_timings),
            "total_minimum_seconds": min(total_timings),
            "total_maximum_seconds": max(total_timings),
            "total_timings_seconds": total_timings,
            "prepared_bytes": sum(batch.prepared_bytes for batch in result.batches),
            "peak_allocated_bytes": int(peak_allocated_bytes),
            "host_to_device_bytes": result.host_to_device_bytes,
            "device_to_host_bytes": result.device_to_host_bytes,
        },
        result,
    )


def _large_batch_smoke() -> dict[str, object]:
    """Run one threshold-eligible workload, never a shape grid."""
    problems, candidates = _case(
        LARGE_N_NODES,
        LARGE_BATCH_SIZE,
        LARGE_SEED,
    )
    cpu_timing, cpu_result = _time_batch(
        problems,
        candidates,
        ComputeConfig(
            backend="numpy",
            device="cpu",
            batch_size=LARGE_BATCH_SIZE,
        ),
    )
    auto_timing, auto_result = _time_batch(
        problems,
        candidates,
        ComputeConfig(device="auto", batch_size=LARGE_BATCH_SIZE),
    )
    expected = _flatten_batch_result(cpu_result)
    actual = _flatten_batch_result(auto_result)
    errors = _maximum_errors(actual, expected)
    cpu_seconds = float(cpu_timing["total_median_seconds"])
    auto_seconds = float(auto_timing["total_median_seconds"])
    speedup = cpu_seconds / auto_seconds
    diagnostics = auto_result.backend_diagnostics
    return {
        "operation": "intervention_kl_value_and_gradient",
        "n_nodes": LARGE_N_NODES,
        "batch_size": LARGE_BATCH_SIZE,
        "candidate_count": LARGE_BATCH_SIZE,
        "sample_count": 0,
        "dense_work": LARGE_N_NODES * LARGE_N_NODES * LARGE_BATCH_SIZE,
        "dtype": "float64",
        "problem_generation_excluded": True,
        "warmup_excluded": True,
        "cuda_synchronization": True,
        "transfer_overhead_included": True,
        "full_grid_launched": False,
        "cpu": cpu_timing,
        "auto": auto_timing,
        "auto_selected_cuda": (
            diagnostics.backend == "torch" and diagnostics.device == "cuda"
        ),
        "maximum_absolute_error": errors[0],
        "maximum_relative_error": errors[1],
        "within_tolerance": _within_tolerance(actual, expected),
        "speedup_vs_numpy": speedup,
        "minimum_meaningful_speedup": MINIMUM_MEANINGFUL_SPEEDUP,
        "meaningful_speedup_observed": speedup >= MINIMUM_MEANINGFUL_SPEEDUP,
    }


def _environment() -> dict[str, object]:
    """Return the runtime identity needed to interpret the smoke."""
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": properties.total_memory,
    }


def run_smoke() -> dict[str, object]:
    """Execute exactly one small experiment and one larger batch workload."""
    if not torch.cuda.is_available():
        raise RuntimeError("The GPU-10 readiness smoke requires CUDA.")
    torch.cuda.init()

    config = NoiselessPilotConfig(seeds=(SMALL_SEED,))
    spec = PilotRunSpec(
        n_nodes=5,
        graph_family="path",
        policy="known_dag_oracle",
        seed=SMALL_SEED,
    )
    problem = _problem_for_spec(config, spec)
    fingerprint_before = _problem_fingerprint(problem)
    cpu = _run_small_backend(problem, config, device="cpu")
    fingerprint_after_cpu = _problem_fingerprint(problem)
    cuda = _run_small_backend(problem, config, device="cuda")
    fingerprint_after_cuda = _problem_fingerprint(problem)
    comparison = _small_comparison(cpu, cuda)
    large_batch = _large_batch_smoke()
    input_unchanged = (
        fingerprint_before == fingerprint_after_cpu == fingerprint_after_cuda
    )
    ready = bool(
        input_unchanged
        and comparison["ready"]
        and large_batch["within_tolerance"]
        and large_batch["auto_selected_cuda"]
        and large_batch["meaningful_speedup_observed"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kl_direction": "target_to_candidate",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "step": "GPU-10",
        "full_grid_launched": False,
        "environment": _environment(),
        "small_end_to_end": {
            "input": {
                "spec": asdict(spec),
                "problem_object_reused_for_both_runs": True,
                "problem_object_id_during_process": id(problem),
                "sha256_before_cpu": fingerprint_before,
                "sha256_after_cpu": fingerprint_after_cpu,
                "sha256_after_cuda": fingerprint_after_cuda,
                "input_unchanged": input_unchanged,
                "raw_observational_samples_stored": False,
                "observational_samples_shape": list(
                    problem.public.observational_samples.shape
                ),
            },
            "methodology": {
                "scheduler": "AlternatingMatchingScheduler",
                "optimizer": "SciPy L-BFGS-B",
                "cpu_loss": "OracleAnalyticGaussianKLLoss",
                "cuda_loss": "TorchOracleGaussianKLLoss",
                "same_scheduler_acceptance_and_budget_rules": True,
                "graph_logic_and_experiment_environment_on_cpu": True,
                "one_required_probe": True,
                "candidate_evaluation_requires_experiment": True,
            },
            "cpu": cpu,
            "cuda": cuda,
            "comparison": comparison,
        },
        "large_batch": large_batch,
        "readiness": {
            "small_end_to_end_equivalent": comparison["ready"],
            "input_unchanged": input_unchanged,
            "large_batch_equivalent": large_batch["within_tolerance"],
            "auto_selected_cuda": large_batch["auto_selected_cuda"],
            "meaningful_speedup_observed": large_batch["meaningful_speedup_observed"],
            "ready_for_later_full_experiment_step": ready,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded smoke CLI with no grid option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
    )
    return parser


def main() -> None:
    """Run the bounded smoke and persist its compact summary."""
    arguments = build_parser().parse_args()
    payload = run_smoke()
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not payload["readiness"]["ready_for_later_full_experiment_step"]:
        raise RuntimeError("GPU-10 readiness criteria were not satisfied.")


if __name__ == "__main__":
    main()
