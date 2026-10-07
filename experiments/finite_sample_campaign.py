"""Focused paired finite-sample campaign for graph-aware shift--scale matching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx
import numpy as np

from environments.blackbox import (
    EnvironmentQueryLedger,
    FiniteSampleBlackBoxEnvironment,
)
from experiments.executors import build_production_runtime
from experiments.nonoracle import (
    NonOracleExperimentConfig,
    NonOracleExperimentResult,
    ProbeRecoverySettings,
    run_finite_sample_nonoracle,
)
from experiments.runner import RunSpec, config_from_json, expand_run_specs
from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import PublicProblemView
from matching.coordinate_search import (
    CoordinateSearchBounds,
    ExactCoordinateSearchSettings,
    FiniteCoordinateSearchSettings,
)
from matching.empirical_decisions import EmpiricalAcceptanceSettings
from matching.empirical_moments import (
    CovarianceEstimatorSettings,
    EmpiricalMomentsProvider,
)
from matching.local_losses import LocalLossWeights, PublicPairwiseNeighborhood
from matching.structure_learning import SuppliedExactCPDAGProvider
from matching.types import FinalSupportBudget
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation

PROTOCOL_ID = "finite_sample_focused_v1"
SCHEMA_VERSION = 3
CAMPAIGN_ID = "finite_sample_graph_aware_p08_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXACT_CONFIG = REPOSITORY_ROOT / (
    "launch_configs/ablations/focused_density_sweep/paper/"
    "config-004-erdos-renyi-n10-rho0-4-mixed-moderate-b2-0-moderate-realisable.json"
)
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/finite_sample_focused_v1"
BASE_SEEDS = tuple(range(36001, 36011))
COVARIANCE = CovarianceEstimatorSettings("oas", 1e-8, 1e-8)
BOOTSTRAP_REPLICATES = 100
CONFIDENCE_LEVEL = 0.95
IMPROVEMENT_TOLERANCE = 1e-3
LOCAL_TOLERANCE = 1e-3
GLOBAL_TOLERANCE = 1e-3


@dataclass(frozen=True)
class SampleCondition:
    N_obs: int
    N_target: int
    N_int: int
    sweep_identifiers: tuple[str, ...]

    @property
    def identifier(self) -> str:
        return f"nobs-{self.N_obs}__ntarget-{self.N_target}__nint-{self.N_int}"


CONDITIONS = (
    SampleCondition(2000, 2000, 100, ("intervention_sample_sweep",)),
    SampleCondition(2000, 2000, 500, ("intervention_sample_sweep",)),
    SampleCondition(
        2000,
        2000,
        2000,
        ("intervention_sample_sweep", "joint_sample_size_sweep"),
    ),
    SampleCondition(100, 100, 100, ("joint_sample_size_sweep",)),
    SampleCondition(500, 500, 500, ("joint_sample_size_sweep",)),
)


@dataclass(frozen=True)
class FiniteSampleRunSpec:
    run_id: str
    paired_problem_id: str
    base_seed: int
    exact_run_id: str
    exact_result_path: str
    N_obs: int
    N_target: int
    N_int: int
    sweep_identifiers: tuple[str, ...]
    sample_seeds: dict[str, int]
    fingerprint: str


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seed(pair_id: str, label: str) -> int:
    digest = hashlib.sha256(f"{CAMPAIGN_ID}:{pair_id}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % np.iinfo(np.int64).max


def _sample_seeds(pair_id: str) -> dict[str, int]:
    environment_root = _seed(pair_id, "environment_root")
    first = int(
        np.random.default_rng(environment_root).integers(0, np.iinfo(np.int64).max)
    )
    return {
        "source_observational": first,
        "target": _seed(pair_id, "target"),
        "environment_root": environment_root,
        "bootstrap": _seed(pair_id, "bootstrap"),
    }


def exact_specs() -> tuple[RunSpec, ...]:
    config = config_from_json(EXACT_CONFIG)
    selected = tuple(
        spec for spec in expand_run_specs(config) if spec.base_seed in BASE_SEEDS
    )
    if len(selected) != len(BASE_SEEDS):
        raise RuntimeError(
            "the exact p=0.8 reference does not contain ten paired seeds"
        )
    by_seed = {spec.base_seed: spec for spec in selected}
    if set(by_seed) != set(BASE_SEEDS):
        raise RuntimeError("exact reference base seeds are missing or duplicated")
    return tuple(by_seed[seed] for seed in BASE_SEEDS)


def campaign_specs() -> tuple[FiniteSampleRunSpec, ...]:
    specs: list[FiniteSampleRunSpec] = []
    for exact in exact_specs():
        exact_path = (
            REPOSITORY_ROOT
            / "outputs/ablations/focused_density_sweep/paper/cells/"
            / "004-erdos-renyi-n10-rho0-4-mixed-moderate-b2-0-moderate-realisable"
            / "results"
            / f"{exact.run_id}.json"
        )
        if not exact_path.exists():
            raise FileNotFoundError(f"missing paired exact result: {exact_path}")
        for condition in CONDITIONS:
            scientific = {
                "campaign": CAMPAIGN_ID,
                "protocol": PROTOCOL_ID,
                "exact_run_id": exact.run_id,
                "pair": exact.paired_problem_id,
                "condition": asdict(condition),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "confidence_level": CONFIDENCE_LEVEL,
                "improvement_tolerance": IMPROVEMENT_TOLERANCE,
            }
            fingerprint = _stable_hash(scientific)
            specs.append(
                FiniteSampleRunSpec(
                    run_id=f"finite-{fingerprint[:20]}",
                    paired_problem_id=exact.paired_problem_id,
                    base_seed=exact.base_seed,
                    exact_run_id=exact.run_id,
                    exact_result_path=str(exact_path),
                    N_obs=condition.N_obs,
                    N_target=condition.N_target,
                    N_int=condition.N_int,
                    sweep_identifiers=condition.sweep_identifiers,
                    sample_seeds=_sample_seeds(exact.paired_problem_id),
                    fingerprint=fingerprint,
                )
            )
    if len(specs) != 50 or len({spec.run_id for spec in specs}) != 50:
        raise RuntimeError("finite-sample grid must contain exactly 50 unique jobs")
    return tuple(sorted(specs, key=lambda item: item.run_id))


def manifest_payload() -> dict[str, object]:
    exact_manifest = (
        REPOSITORY_ROOT
        / "outputs/ablations/focused_density_sweep/paper/cells/"
        / "004-erdos-renyi-n10-rho0-4-mixed-moderate-b2-0-moderate-realisable"
        / "manifest.json"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_identifier": CAMPAIGN_ID,
        "protocol_identifier": PROTOCOL_ID,
        "output_root": str(OUTPUT_ROOT),
        "exact_config": str(EXACT_CONFIG),
        "exact_manifest": str(exact_manifest),
        "exact_manifest_checksum": _file_checksum(exact_manifest),
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "base_seeds": list(BASE_SEEDS),
        "environment_cache_policy": (
            "accepted-background samples reused within a step; candidate-search, "
            "probe, and held-out acceptance-validation queries use deterministic "
            "fresh independent samples"
        ),
        "covariance": asdict(COVARIANCE),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": CONFIDENCE_LEVEL,
        "improvement_tolerance": IMPROVEMENT_TOLERANCE,
        "runs": [asdict(spec) for spec in campaign_specs()],
    }


def write_manifest() -> Path:
    path = OUTPUT_ROOT / "manifest.json"
    payload = json.loads(json.dumps(manifest_payload()))
    if path.exists():
        prior = json.loads(path.read_text())
        if prior != payload:
            artifacts = tuple((OUTPUT_ROOT / "results").glob("*.json")) + tuple(
                (OUTPUT_ROOT / "failures").glob("*.json")
            )
            if artifacts:
                raise RuntimeError("existing finite-sample manifest is incompatible")
            _atomic_json(path, payload)
        return path
    _atomic_json(path, payload)
    return path


def _exact_by_id(run_id: str) -> RunSpec:
    return next(spec for spec in exact_specs() if spec.run_id == run_id)


def _condition(spec: FiniteSampleRunSpec) -> SampleCondition:
    return SampleCondition(
        spec.N_obs, spec.N_target, spec.N_int, spec.sweep_identifiers
    )


def _algorithm_config(
    exact: RunSpec, condition: SampleCondition
) -> NonOracleExperimentConfig:
    bounds = CoordinateSearchBounds(-2.0, 2.0, np.log(0.25), np.log(3.0))
    return NonOracleExperimentConfig(
        final_support_budget=FinalSupportBudget(
            exact.final_parameter_coordinate_budget
        ),
        exact_search=ExactCoordinateSearchSettings(bounds, 9, True, 21),
        finite_search=FiniteCoordinateSearchSettings(
            bounds=bounds,
            coarse_grid_size=9,
            initial_sample_size=condition.N_int,
            max_sample_size_per_candidate=condition.N_int,
            max_query_batches=9,
            sample_growth_factor=2.0,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            confidence_level=CONFIDENCE_LEVEL,
            improvement_tolerance=IMPROVEMENT_TOLERANCE,
        ),
        acceptance=EmpiricalAcceptanceSettings(
            initial_candidate_sample_size=condition.N_int,
            max_sample_size_per_arm=condition.N_int,
            sample_growth_factor=2.0,
            max_repeat_queries=1,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            confidence_level=CONFIDENCE_LEVEL,
            improvement_tolerance=IMPROVEMENT_TOLERANCE,
        ),
        covariance=COVARIANCE,
        probe=ProbeRecoverySettings(
            initial_probe_sample_size=condition.N_int,
            max_probe_sample_size=condition.N_int,
            sample_growth_factor=2.0,
            max_probe_repeats=1,
            probe_amplitude_schedule=(0.5,),
            max_probe_targets=exact.n_nodes,
            source_tolerance_absolute=0.1,
            source_tolerance_relative=0.05,
            source_confidence_level=CONFIDENCE_LEVEL,
        ),
        local_weights=LocalLossWeights(1.0, 1.0, 1.0),
        pairwise_neighborhood=PublicPairwiseNeighborhood(),
        local_loss_tolerance=LOCAL_TOLERANCE,
        global_loss_tolerance=GLOBAL_TOLERANCE,
        improvement_tolerance=IMPROVEMENT_TOLERANCE,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        confidence_level=CONFIDENCE_LEVEL,
        baseline_sample_size=condition.N_obs,
        max_steps=20,
        patience=21,
        accepted_baseline_sample_size=condition.N_int,
        record_covariance_estimates=True,
    )


def _kl(
    target_mean: np.ndarray, target_cov: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> float:
    return gaussian_kl_target_to_candidate(target_mean, target_cov, mean, cov)


def _coordinate_labels(intervention: ShiftScaleIntervention) -> tuple[str, ...]:
    labels = [
        f"shift:{node}"
        for node, value in enumerate(intervention.shift)
        if not np.isclose(value, 0.0)
    ]
    labels.extend(
        f"log_scale:{node}"
        for node, value in enumerate(intervention.scale)
        if not np.isclose(value, 1.0)
    )
    return tuple(labels)


def _sample_fingerprint(samples: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(samples)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _graph_diagnostics(
    true_dag: nx.DiGraph, result: NonOracleExperimentResult
) -> dict[str, object]:
    true_skeleton = {frozenset(edge) for edge in true_dag.edges}
    found_skeleton = {
        frozenset(edge)
        for edge in (
            *result.final_graph.directed_edges,
            *result.final_graph.undirected_edges,
        )
    }
    skeleton_accuracy = (
        len(true_skeleton & found_skeleton) / len(true_skeleton | found_skeleton)
        if true_skeleton | found_skeleton
        else 1.0
    )
    directed = set(result.final_graph.directed_edges)
    orientation_accuracy = (
        sum(edge in true_dag.edges for edge in directed) / len(directed)
        if directed
        else 1.0
    )
    return {
        "skeleton_accuracy": skeleton_accuracy,
        "orientation_accuracy": orientation_accuracy,
        "oriented_edge_count": len(directed),
    }


def _accepted_diagnostics(
    problem: Any, result: NonOracleExperimentResult
) -> dict[str, object]:
    target_mean = problem.public.target_mean
    target_cov = problem.public.target_covariance
    current = problem.public.current_intervention
    current_mean, current_cov = problem.ground_truth.scm.interventional_moments(current)
    current_true = _kl(target_mean, target_cov, current_mean, current_cov)
    accepted_rows: list[dict[str, object]] = []
    false_positive = 0
    false_negative = 0
    for record in result.history:
        if record.action != "confidence_aware_candidate_evaluation":
            continue
        candidate = ShiftScaleIntervention(
            np.asarray(record.candidate_shift), np.asarray(record.candidate_scale)
        )
        candidate_mean, candidate_cov = problem.ground_truth.scm.interventional_moments(
            candidate
        )
        candidate_true = _kl(target_mean, target_cov, candidate_mean, candidate_cov)
        true_improvement = current_true - candidate_true
        estimated = float(record.empirical_global_improvement or 0.0)
        if record.decision == "accepted":
            false_positive += int(estimated > 0.0 and true_improvement <= 0.0)
            accepted_rows.append(
                {
                    "step": record.step,
                    "estimated_kl": record.global_loss_after,
                    "true_kl": candidate_true,
                    "estimated_improvement": estimated,
                    "true_improvement": true_improvement,
                    "coordinate": (
                        None
                        if record.coordinate is None
                        else f"{record.coordinate.kind}:{record.coordinate.node}"
                    ),
                    "decision_metadata": list(record.decision_metadata),
                }
            )
            current = candidate
            current_true = candidate_true
        elif record.decision == "rejected" and true_improvement > 0.0:
            false_negative += 1
    return {
        "accepted_update_trajectory": accepted_rows,
        "false_positive_accepted_updates": false_positive,
        "false_negative_rejected_proposals": false_negative,
    }


def _probe_diagnostics(
    true_dag: nx.DiGraph, result: NonOracleExperimentResult
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    correct = 0
    total = 0
    for record in result.history:
        if not record.classifications:
            continue
        targets = {node for node, _ in record.classifications}
        for node, classification in record.classifications:
            true_source = not any(
                ancestor in targets for ancestor in nx.ancestors(true_dag, node)
            )
            expected = "source" if true_source else "non_source"
            is_correct = classification == expected
            correct += int(is_correct)
            total += 1
            detail = next(
                (
                    item
                    for item in record.decision_metadata
                    if int(item.get("node", -1)) == node
                ),
                {},
            )
            rows.append(
                {
                    "step": record.step,
                    "node": node,
                    "classification": classification,
                    "hidden_expected_classification": expected,
                    "correct": is_correct,
                    "uncertainty": detail,
                }
            )
    return {
        "probe_classification_decisions": rows,
        "probe_classification_accuracy": correct / total if total else None,
        "probe_classification_count": total,
    }


def _enrich_query_history(
    query_history: list[dict[str, object]],
) -> list[dict[str, object]]:
    replications: dict[tuple[str, str], int] = {}
    enriched: list[dict[str, object]] = []
    for item in query_history:
        key = (str(item["intervention_id"]), str(item["budget_category"]))
        replication = replications.get(key, 0)
        replications[key] = replication + 1
        row = dict(item)
        row["replication_index"] = replication
        row["query_phase"] = {
            "baseline": "accepted_background",
            "probe": "probe_classification",
            "candidate": "candidate_search_or_held_out_acceptance",
        }.get(str(item["budget_category"]), "repeat")
        enriched.append(row)
    return enriched


def upgrade_existing_query_metadata() -> int:
    upgraded = 0
    for path in sorted((OUTPUT_ROOT / "results").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("campaign_identifier") != CAMPAIGN_ID:
            raise RuntimeError(f"foreign campaign row: {path}")
        history = payload["metrics"]["query_history"]
        if history and "replication_index" in history[0]:
            continue
        payload["metrics"]["query_history"] = _enrich_query_history(history)
        _atomic_json(path, payload)
        upgraded += 1
    return upgraded


def execute(spec: FiniteSampleRunSpec) -> dict[str, object]:
    exact = _exact_by_id(spec.exact_run_id)
    runtime = build_production_runtime(exact)
    problem = runtime.problem
    condition = _condition(spec)
    seeds = spec.sample_seeds

    source_samples = problem.ground_truth.scm.sample_interventional(
        problem.public.current_intervention,
        condition.N_obs,
        np.random.default_rng(seeds["source_observational"]),
    )
    target_samples = problem.ground_truth.scm.sample_interventional(
        problem.ground_truth.generating_intervention,
        condition.N_target,
        np.random.default_rng(seeds["target"]),
    )
    diagnostic_provider = EmpiricalMomentsProvider(COVARIANCE)
    source_estimate = diagnostic_provider.estimate(source_samples)
    target_estimate = diagnostic_provider.estimate(target_samples)
    public_problem = PublicProblemView(
        observational_samples=source_samples,
        target_mean=target_estimate.mean,
        target_covariance=target_estimate.regularized_covariance,
        current_intervention=problem.public.current_intervention,
        algorithm_seeds=problem.public.algorithm_seeds,
    )
    ledger = EnvironmentQueryLedger(max_query_batches=None, max_observations=None)
    environment = FiniteSampleBlackBoxEnvironment(
        problem.ground_truth.scm,
        instance_id=spec.paired_problem_id,
        ledger=ledger,
    )
    structure = SuppliedExactCPDAGProvider(
        cpdag_from_true_dag_for_simulation(problem.ground_truth.scm.dag)
    )
    config = _algorithm_config(exact, condition)

    started = perf_counter()
    result = run_finite_sample_nonoracle(
        public_problem,
        environment,
        structure,
        config,
        environment_seed=seeds["environment_root"],
        bootstrap_seed=seeds["bootstrap"],
        result_mode="finite_sample_non_oracle_supplied_cpdag",
    )
    runtime_seconds = perf_counter() - started

    initial_exact_mean, initial_exact_cov = problem.current_moments()
    final_exact_mean, final_exact_cov = problem.ground_truth.scm.interventional_moments(
        result.final_intervention
    )
    true_initial = _kl(
        problem.public.target_mean,
        problem.public.target_covariance,
        initial_exact_mean,
        initial_exact_cov,
    )
    true_final = _kl(
        problem.public.target_mean,
        problem.public.target_covariance,
        final_exact_mean,
        final_exact_cov,
    )
    estimated_initial = _kl(
        public_problem.target_mean,
        public_problem.target_covariance,
        source_estimate.mean,
        source_estimate.regularized_covariance,
    )
    labels = _coordinate_labels(result.final_intervention)
    query_history = _enrich_query_history(
        [asdict(item) for item in environment.query_history]
    )
    covariance_rows = list(result.covariance_diagnostics)
    warnings = [
        "regularized_covariance_cholesky_failed"
        for row in covariance_rows
        if not bool(row["cholesky_success"])
    ]
    metrics: dict[str, object] = {
        "N_obs": condition.N_obs,
        "N_target": condition.N_target,
        "N_int": condition.N_int,
        "sample_size_sweep_identifier": list(condition.sweep_identifiers),
        "protocol_identifier": PROTOCOL_ID,
        "covariance_estimator": asdict(COVARIANCE),
        "source_covariance_diagnostics": source_estimate.metadata(
            include_matrices=True
        ),
        "target_covariance_diagnostics": target_estimate.metadata(
            include_matrices=True
        ),
        "covariance_estimate_diagnostics": covariance_rows,
        "sample_seeds": seeds,
        "environment_cache_policy": (
            "accepted-background reuse plus fresh deterministic independent "
            "search/probe/held-out-acceptance samples"
        ),
        "estimated_initial_reverse_kl": estimated_initial,
        "true_initial_reverse_kl": true_initial,
        "estimated_final_reverse_kl": result.final_global_loss,
        "true_final_reverse_kl": true_final,
        "estimated_minus_true_final_kl": result.final_global_loss - true_final,
        "true_relative_kl_reduction": (
            (true_initial - true_final) / true_initial if true_initial > 0.0 else 0.0
        ),
        "accepted_shift_coordinates": [
            label for label in labels if label.startswith("shift:")
        ],
        "accepted_scale_coordinates": [
            label for label in labels if label.startswith("log_scale:")
        ],
        "accepted_final_support": list(labels),
        "final_support_size": len(labels),
        "accepted_candidates": result.accepted_proposals,
        "rejected_candidates": result.rejected_proposals,
        "inconclusive_candidates": result.inconclusive_proposals,
        "candidate_evaluations": result.optimizer_search_evaluations,
        "intervention_queries": int(result.query_accounting["candidate_queries"] or 0),
        "probe_queries": int(result.query_accounting["probe_queries"] or 0),
        "total_samples_drawn": condition.N_target
        + int(result.query_accounting["total_observations"] or 0),
        "query_accounting": result.query_accounting,
        "query_history": query_history,
        "covariance_fallback_count": sum(
            bool(row["fallback_required"]) for row in covariance_rows
        ),
        "covariance_pseudoinverse_count": sum(
            bool(row["pseudoinverse_required"]) for row in covariance_rows
        ),
        "numerical_warnings": warnings,
        "termination_reason": result.termination_reason,
        "public_action_signature": list(result.public_action_signature),
        "runtime_seconds": runtime_seconds,
        "problem_fingerprint": runtime.problem_fingerprint,
        "sample_provenance_fingerprint": _stable_hash(
            {
                "source": _sample_fingerprint(source_samples),
                "target": _sample_fingerprint(target_samples),
                "query_seeds": [row["query_seed"] for row in query_history],
                "query_interventions": [
                    row["intervention_id"] for row in query_history
                ],
            }
        ),
        "source_sample_fingerprint": _sample_fingerprint(source_samples),
        "target_sample_fingerprint": _sample_fingerprint(target_samples),
        **_accepted_diagnostics(problem, result),
        **_probe_diagnostics(problem.ground_truth.scm.dag, result),
        **_graph_diagnostics(problem.ground_truth.scm.dag, result),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "campaign_identifier": CAMPAIGN_ID,
        "protocol_identifier": PROTOCOL_ID,
        "run_id": spec.run_id,
        "run_fingerprint": spec.fingerprint,
        "paired_problem_id": spec.paired_problem_id,
        "base_seed": spec.base_seed,
        "exact_reference": {
            "run_id": spec.exact_run_id,
            "path": spec.exact_result_path,
            "checksum": _file_checksum(Path(spec.exact_result_path)),
        },
        "spec": asdict(spec),
        "metrics": metrics,
    }


def result_path(run_id: str) -> Path:
    return OUTPUT_ROOT / "results" / f"{run_id}.json"


def failure_path(run_id: str) -> Path:
    return OUTPUT_ROOT / "failures" / f"{run_id}.json"


def run_one(run_id: str, *, resume: bool = True) -> dict[str, object]:
    write_manifest()
    spec = next((item for item in campaign_specs() if item.run_id == run_id), None)
    if spec is None:
        raise ValueError(f"unknown finite-sample run ID {run_id}")
    completed = result_path(run_id)
    failure = failure_path(run_id)
    if completed.exists():
        payload = json.loads(completed.read_text())
        if payload.get("run_fingerprint") != spec.fingerprint:
            raise RuntimeError(f"incompatible completed artifact: {completed}")
        if failure.exists():
            raise RuntimeError(f"duplicate result and failure for {run_id}")
        return {"status": "skipped_completed", "run_id": run_id}
    if failure.exists() and not resume:
        return {"status": "skipped_failed_use_resume", "run_id": run_id}
    try:
        payload = execute(spec)
        _atomic_json(completed, payload)
        if failure.exists():
            failure.unlink()
        return payload
    except Exception as error:  # noqa: BLE001 - durable failure boundary
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "campaign_identifier": CAMPAIGN_ID,
            "protocol_identifier": PROTOCOL_ID,
            "run_id": run_id,
            "run_fingerprint": spec.fingerprint,
            "spec": asdict(spec),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(failure, payload)
        return payload


def validate_outputs() -> dict[str, object]:
    specs = campaign_specs()
    expected = {spec.run_id: spec for spec in specs}
    results = tuple(sorted((OUTPUT_ROOT / "results").glob("*.json")))
    failures = tuple(sorted((OUTPUT_ROOT / "failures").glob("*.json")))
    duplicates = {path.stem for path in results} & {path.stem for path in failures}
    foreign = {path.stem for path in (*results, *failures) if path.stem not in expected}
    valid = 0
    corrupt: list[str] = []
    for path in results:
        try:
            row = json.loads(path.read_text())
            spec = expected[path.stem]
            metrics = row["metrics"]
            query_history = metrics["query_history"]
            replication_keys = [
                (
                    query["intervention_id"],
                    query["budget_category"],
                    query.get("replication_index"),
                )
                for query in query_history
            ]
            query_seeds = [query.get("query_seed") for query in query_history]
            exact_reference = row["exact_reference"]
            if (
                row["schema_version"] != SCHEMA_VERSION
                or row["status"] != "completed"
                or row["campaign_identifier"] != CAMPAIGN_ID
                or row["protocol_identifier"] != PROTOCOL_ID
                or row["run_fingerprint"] != spec.fingerprint
                or row["paired_problem_id"] != spec.paired_problem_id
                or exact_reference["run_id"] != spec.exact_run_id
                or exact_reference["path"] != spec.exact_result_path
                or exact_reference["checksum"]
                != _file_checksum(Path(spec.exact_result_path))
                or metrics["protocol_identifier"] != PROTOCOL_ID
                or metrics["N_obs"] != spec.N_obs
                or metrics["N_target"] != spec.N_target
                or metrics["N_int"] != spec.N_int
                or len(query_seeds) != len(set(query_seeds))
                or len(replication_keys) != len(set(replication_keys))
                or any(
                    seed is None
                    or "replication_index" not in query
                    or "query_phase" not in query
                    for seed, query in zip(query_seeds, query_history, strict=True)
                )
            ):
                raise ValueError("row provenance mismatch")
            valid += 1
        except Exception:  # noqa: BLE001 - corrupt artifact boundary
            corrupt.append(str(path))
    return {
        "expected_jobs": 50,
        "completed_files": len(results),
        "valid_completed_jobs": valid,
        "failure_files": len(failures),
        "duplicates": sorted(duplicates),
        "foreign_artifacts": sorted(foreign),
        "corrupt_results": corrupt,
        "missing_jobs": sorted(set(expected) - {path.stem for path in results}),
    }


def _file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--upgrade-query-metadata", action="store_true")
    args = parser.parse_args(argv)
    if args.write_manifest:
        print(write_manifest())
    if args.run_id:
        print(json.dumps(run_one(args.run_id), sort_keys=True))
    if args.all:
        write_manifest()
        for index, spec in enumerate(campaign_specs(), 1):
            outcome = run_one(spec.run_id)
            print(
                json.dumps(
                    {
                        "index": index,
                        "total": 50,
                        "run_id": spec.run_id,
                        "status": outcome["status"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if args.validate:
        print(json.dumps(validate_outputs(), indent=2, sort_keys=True))
    if args.upgrade_query_metadata:
        print(json.dumps({"upgraded_rows": upgrade_existing_query_metadata()}))
    if not any(
        (
            args.write_manifest,
            args.run_id,
            args.all,
            args.validate,
            args.upgrade_query_metadata,
        )
    ):
        parser.error("select at least one action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
