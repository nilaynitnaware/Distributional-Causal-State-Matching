"""Restartable per-run experiment runner.

The CLI is safe by default: an execution requires an explicit ``--run-id`` and
an application-provided executor registry.  This module never expands a grid
into in-memory experiment state; every run is reconstructed from its serialized
specification and writes one atomic result or durable failure record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
import tempfile
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from itertools import product
from math import ceil
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Literal

import numpy as np

from graphs.interventions import INTERVENTION_SEMANTICS
from graphs.noise import validate_noise_variance_regime

from .modes import ExperimentMode

RUN_RESULT_SCHEMA_VERSION = 2
_ABLATION_SUMMARY_METRICS = (
    "initial_global_reverse_kl",
    "final_global_reverse_kl",
    "absolute_kl_reduction",
    "relative_kl_reduction",
    "mean_error",
    "covariance_error",
    "accepted_final_coordinates",
    "target_support_recall",
    "recovered_support_precision",
    "target_shift_coordinate_count",
    "target_scale_coordinate_count",
    "accepted_shift_coordinate_count",
    "accepted_scale_coordinate_count",
    "candidate_evaluations",
    "executed_candidate_queries",
    "probe_count",
    "total_environment_calls",
    "executed_environment_queries",
    "skeleton_accuracy",
    "orientation_accuracy",
    "directed_shd",
    "wall_seconds",
)


@dataclass(frozen=True)
class RunnerDimensions:
    """Configuration dimensions supported by the experiment manifest."""

    experiment_modes: tuple[ExperimentMode, ...]
    graph_families: tuple[str, ...]
    node_counts: tuple[int, ...]
    coefficient_regimes: tuple[str, ...]
    target_families: tuple[str, ...]
    target_realisability: tuple[bool, ...]
    support_ratios: tuple[float, ...]
    policies: tuple[str, ...]
    initial_cpdag_providers: tuple[str, ...]
    sample_sizes: tuple[int, ...]
    edge_probabilities: tuple[float | None, ...] = (None,)
    intervention_compositions: tuple[str, ...] = ("shift_only",)
    intervention_magnitude_regimes: tuple[str, ...] = ("moderate",)
    final_coordinate_budget_multipliers: tuple[float, ...] = (2.0,)
    coordinate_selection_rules: tuple[str, ...] = ("best_improvement",)
    noise_variance_regimes: tuple[str, ...] = ("homoskedastic",)


@dataclass(frozen=True)
class RunnerConfig:
    """Full serialized grid schema and shared scientific settings."""

    dimensions: RunnerDimensions
    base_seeds: tuple[int, ...]
    query_budget: int | None
    observation_budget: int
    probe_target_limit: int | Literal["n_nodes"]
    probe_amplitudes: tuple[float, ...]
    local_loss_weights: tuple[float, float, float]
    source_equivalence_absolute: float
    source_equivalence_relative: float
    source_confidence_level: float
    candidate_acceptance_confidence: float
    covariance_estimator: str
    backend: str
    device: str
    output_directory: str
    intervention_semantics: str = INTERVENTION_SEMANTICS
    campaign_design_identifier: str = "unspecified_campaign_design"
    couple_edge_probability: bool = False
    couple_intervention_composition: bool = False
    couple_intervention_magnitude: bool = False
    bootstrap_aggregate_replicates: int = 1_000
    coordinate_coarse_grid_size: int = 9
    coordinate_max_values_per_coordinate: int = 30
    improvement_tolerance: float = 1e-10
    ablation_metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class RunSeeds:
    """Separate deterministic seeds for every stochastic subsystem."""

    graph: int
    coefficients: int
    target: int
    observational_samples: int
    environment_samples: int
    probes: int
    coordinate_search: int
    bootstrap: int
    policy: int
    noise_variances: int | None = None


@dataclass(frozen=True)
class RunSpec:
    """One deterministic run and its paired-instance identity."""

    run_id: str
    paired_problem_id: str
    experiment_mode: ExperimentMode
    graph_family: str
    n_nodes: int
    coefficient_regime: str
    target_family: str
    target_is_realisable: bool
    rho: float
    k: int
    final_parameter_coordinate_budget: int
    query_budget: int | None
    observation_budget: int
    policy: str
    initial_cpdag_provider: str
    probe_target_limit: int
    probe_amplitudes: tuple[float, ...]
    sample_size: int
    local_loss_weights: tuple[float, float, float]
    source_equivalence_absolute: float
    source_equivalence_relative: float
    source_confidence_level: float
    candidate_acceptance_confidence: float
    covariance_estimator: str
    backend: str
    device: str
    base_seed: int
    seeds: RunSeeds
    edge_probability: float | None = None
    intervention_composition: str = "shift_only"
    intervention_action_class: str = "shift_scale"
    target_coordinate_scheme: str = "composition_default"
    couple_intervention_composition: bool = False
    couple_intervention_magnitude: bool = False
    intervention_magnitude_regime: str = "moderate"
    final_coordinate_budget_multiplier: float = 2.0
    coordinate_selection_rule: str = "best_improvement"
    coordinate_coarse_grid_size: int = 9
    coordinate_max_values_per_coordinate: int = 30
    improvement_tolerance: float = 1e-10
    intervention_semantics: str = INTERVENTION_SEMANTICS
    campaign_design_identifier: str = "unspecified_campaign_design"
    noise_variance_regime: str = "homoskedastic"
    noise_variance_seed: int | None = None
    cross_noise_regime_pair_id: str = ""


@dataclass(frozen=True)
class RunOutcome:
    """Stable per-run payload expected from an executor."""

    metrics: Mapping[str, object] | ExperimentRunMetrics
    termination_reason: str
    success: bool
    reliability: Mapping[str, object] | ExperimentReliability
    protocol_version: str
    resolved_protocol: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.protocol_version:
            raise ValueError("run outcomes require a nonempty protocol version.")
        required = {
            "protocol_version",
            "purpose",
            "coordinate_search",
            "unresolved_state",
            "budgets",
        }

        if not required <= set(self.resolved_protocol):
            raise ValueError("run outcomes require complete resolved protocol values.")
        if self.resolved_protocol["protocol_version"] != self.protocol_version:
            raise ValueError(
                "resolved protocol version does not match the run outcome."
            )


@dataclass(frozen=True)
class ExperimentRunMetrics:
    """Stable scientific and cost metrics shared by production executors."""

    initial_global_reverse_kl: float | None = None
    final_global_reverse_kl: float | None = None
    local_composite_losses: tuple[float, ...] = ()
    mean_error: float | None = None
    covariance_error: float | None = None
    accepted_final_support: tuple[int, ...] = ()
    final_parameter_coordinate_budget: int = 0
    target_support: tuple[int, ...] | None = None
    accepted_candidates: int = 0
    rejected_candidates: int = 0
    inconclusive_candidates: int = 0
    directed_shd: int | None = None
    skeleton_accuracy: float | None = None
    orientation_accuracy: float | None = None
    oriented_edge_count: int = 0
    source_accuracy: float | None = None
    nonsource_accuracy: float | None = None
    inconclusive_source_rate: float | None = None
    meek_orientation_count: int = 0
    algorithm_environment_calls: int = 0
    executed_environment_queries: int = 0
    cache_hits: int = 0
    failed_environment_queries: int = 0
    queried_intervention_batches: int = 0
    baseline_queries: int = 0
    probe_queries: int = 0
    candidate_queries: int = 0
    repeat_queries: int = 0
    failed_queries: int = 0
    total_observations: int = 0
    shift_coordinate_uses: int = 0
    scale_coordinate_uses: int = 0
    optimizer_evaluations: int = 0
    wall_seconds: float = 0.0


@dataclass(frozen=True)
class ExperimentReliability:
    """Stable reliability counters and diagnostic-only error labels."""

    numerical_failures: int = 0
    structure_learning_failures: int = 0
    query_budget_exhaustions: int = 0
    observation_budget_exhaustions: int = 0
    support_budget_exhaustions: int = 0
    no_action_terminations: int = 0
    global_mismatch_not_localized: int = 0
    inconclusive_terminations: int = 0
    orientation_conflicts: int = 0
    empirical_false_acceptance: bool | None = None


RunExecutor = Callable[[RunSpec], RunOutcome | Mapping[str, object] | object]


def _resolve_probe_target_limit(
    configured: int | Literal["n_nodes"],
    n_nodes: int,
) -> int:
    """Resolve a per-run probe-action target cap from runner configuration."""
    if configured == "n_nodes":
        return n_nodes
    if isinstance(configured, bool) or not isinstance(configured, int):
        raise TypeError('probe_target_limit must be an integer or "n_nodes".')
    if not 0 < configured <= n_nodes:
        raise ValueError("probe_target_limit must lie in [1, n_nodes].")
    return configured


def expand_run_specs(config: RunnerConfig) -> tuple[RunSpec, ...]:
    """Deterministically expand configuration only into lightweight specs."""
    dimensions = config.dimensions
    intervention_action_class = "shift_scale"
    target_coordinate_scheme = "composition_default"
    metadata = config.ablation_metadata
    if isinstance(metadata, Mapping):
        group_values = metadata.get("group_values")
        if isinstance(group_values, Mapping):
            configured_action_class = group_values.get("intervention_action_class")
            if isinstance(configured_action_class, str):
                intervention_action_class = configured_action_class
            configured_target_scheme = group_values.get("target_coordinate_scheme")
            if isinstance(configured_target_scheme, str):
                target_coordinate_scheme = configured_target_scheme
    specs: list[RunSpec] = []
    axes = product(
        config.base_seeds,
        dimensions.experiment_modes,
        dimensions.graph_families,
        dimensions.edge_probabilities,
        dimensions.node_counts,
        dimensions.coefficient_regimes,
        dimensions.noise_variance_regimes,
        dimensions.target_families,
        dimensions.target_realisability,
        dimensions.support_ratios,
        dimensions.intervention_compositions,
        dimensions.intervention_magnitude_regimes,
        dimensions.final_coordinate_budget_multipliers,
        dimensions.initial_cpdag_providers,
        dimensions.sample_sizes,
        dimensions.policies,
        dimensions.coordinate_selection_rules,
    )
    for (
        base_seed,
        mode,
        graph_family,
        edge_probability,
        n_nodes,
        coefficient_regime,
        noise_variance_regime,
        target_family,
        target_is_realisable,
        rho,
        intervention_composition,
        intervention_magnitude_regime,
        budget_multiplier,
        initial_cpdag_provider,
        sample_size,
        policy,
        coordinate_selection_rule,
    ) in axes:
        validate_noise_variance_regime(noise_variance_regime)
        k = ceil(rho * n_nodes)
        # This is an allowed final-support cap, not a requested support size.
        # A DAG exposes at most 2*n_nodes-1 legal coordinates because sources
        # cannot be scaled, but permitting a configured cap of 2*n_nodes lets
        # ablations demonstrate that the optimizer stops before a loose cap.
        budget = min(ceil(budget_multiplier * k), 2 * n_nodes)
        problem_payload = {
            "graph_family": graph_family,
            "edge_probability": edge_probability,
            "n_nodes": n_nodes,
            "coefficient_regime": coefficient_regime,
            "target_family": target_family,
            "target_is_realisable": target_is_realisable,
            "rho": rho,
            "intervention_composition": intervention_composition,
            "intervention_magnitude_regime": intervention_magnitude_regime,
            "base_seed": base_seed,
            "intervention_semantics": config.intervention_semantics,
            "campaign_design_identifier": config.campaign_design_identifier,
        }
        if target_coordinate_scheme != "composition_default":
            problem_payload["target_coordinate_scheme"] = target_coordinate_scheme
        seed_problem_payload = dict(problem_payload)
        if config.couple_edge_probability:
            seed_problem_payload.pop("edge_probability")
        if config.couple_intervention_composition:
            seed_problem_payload["intervention_composition"] = "mixed"
        if config.couple_intervention_magnitude:
            seed_problem_payload["intervention_magnitude_regime"] = "moderate"
        structural_pair_id = _stable_id("pair", seed_problem_payload)
        cross_noise_pair_id = _stable_id(
            "cross-noise-pair",
            seed_problem_payload,
        )
        if noise_variance_regime == "homoskedastic":
            paired_id = structural_pair_id
        else:
            paired_id = _stable_id(
                "pair",
                {
                    **seed_problem_payload,
                    "noise_variance_regime": noise_variance_regime,
                },
            )
            problem_payload["noise_variance_regime"] = noise_variance_regime
        shared_seeds = _shared_run_seeds(structural_pair_id, base_seed)
        noise_variance_seed = _seed_from_parts(
            structural_pair_id,
            base_seed,
            "noise_variances",
        )
        shared_seeds["noise_variances"] = noise_variance_seed
        run_payload = {
            **problem_payload,
            "experiment_mode": mode,
            **(
                {"intervention_action_class": intervention_action_class}
                if intervention_action_class != "shift_scale"
                else {}
            ),
            "final_coordinate_budget_multiplier": budget_multiplier,
            "initial_cpdag_provider": initial_cpdag_provider,
            "sample_size": sample_size,
            "policy": policy,
            "coordinate_selection_rule": coordinate_selection_rule,
            "coordinate_coarse_grid_size": config.coordinate_coarse_grid_size,
            "coordinate_max_values_per_coordinate": (
                config.coordinate_max_values_per_coordinate
            ),
            "improvement_tolerance": config.improvement_tolerance,
        }
        run_id = _stable_id("run", run_payload)
        seeds = RunSeeds(
            **shared_seeds,
            policy=_seed_from_parts(
                structural_pair_id,
                policy,
                coordinate_selection_rule,
                "policy",
            ),
        )
        specs.append(
            RunSpec(
                run_id=run_id,
                paired_problem_id=paired_id,
                experiment_mode=mode,
                graph_family=graph_family,
                n_nodes=n_nodes,
                coefficient_regime=coefficient_regime,
                target_family=target_family,
                target_is_realisable=target_is_realisable,
                rho=rho,
                k=k,
                final_parameter_coordinate_budget=budget,
                query_budget=config.query_budget,
                observation_budget=config.observation_budget,
                policy=policy,
                initial_cpdag_provider=initial_cpdag_provider,
                probe_target_limit=_resolve_probe_target_limit(
                    config.probe_target_limit,
                    n_nodes,
                ),
                probe_amplitudes=config.probe_amplitudes,
                sample_size=sample_size,
                local_loss_weights=config.local_loss_weights,
                source_equivalence_absolute=config.source_equivalence_absolute,
                source_equivalence_relative=config.source_equivalence_relative,
                source_confidence_level=config.source_confidence_level,
                candidate_acceptance_confidence=(
                    config.candidate_acceptance_confidence
                ),
                covariance_estimator=config.covariance_estimator,
                backend=config.backend,
                device=config.device,
                base_seed=base_seed,
                seeds=seeds,
                edge_probability=edge_probability,
                intervention_composition=intervention_composition,
                intervention_action_class=intervention_action_class,
                target_coordinate_scheme=target_coordinate_scheme,
                couple_intervention_composition=(
                    config.couple_intervention_composition
                ),
                couple_intervention_magnitude=config.couple_intervention_magnitude,
                intervention_magnitude_regime=intervention_magnitude_regime,
                final_coordinate_budget_multiplier=budget_multiplier,
                coordinate_selection_rule=coordinate_selection_rule,
                coordinate_coarse_grid_size=config.coordinate_coarse_grid_size,
                coordinate_max_values_per_coordinate=(
                    config.coordinate_max_values_per_coordinate
                ),
                improvement_tolerance=config.improvement_tolerance,
                intervention_semantics=config.intervention_semantics,
                campaign_design_identifier=config.campaign_design_identifier,
                noise_variance_regime=noise_variance_regime,
                noise_variance_seed=noise_variance_seed,
                cross_noise_regime_pair_id=cross_noise_pair_id,
            )
        )
    return tuple(sorted(specs, key=lambda item: item.run_id))


def runner_config_fingerprint(config: RunnerConfig) -> str:
    """Return a stable fingerprint of all scientifically relevant config fields."""
    payload = _jsonable(config)
    dimensions = payload.get("dimensions")
    if isinstance(dimensions, dict) and dimensions.get(
        "noise_variance_regimes"
    ) == ["homoskedastic"]:
        dimensions.pop("noise_variance_regimes")
    metadata = payload.get("ablation_metadata")
    if isinstance(metadata, dict):
        metadata.pop("config_fingerprint", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _validate_artifact_fingerprint(path: Path, expected: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("config_fingerprint") != expected
        or payload.get("intervention_semantics") != INTERVENTION_SEMANTICS
    ):
        raise RuntimeError(
            f"{path} belongs to different RunnerConfig fingerprint or intervention semantics; "
            "use a fresh output directory."
        )


class RestartableExperimentRunner:
    """Execute, skip, resume, and durably record independent run specs."""

    def __init__(
        self,
        config: RunnerConfig,
        executors: Mapping[ExperimentMode, RunExecutor],
    ) -> None:
        if config.intervention_semantics != INTERVENTION_SEMANTICS:
            raise ValueError("RunnerConfig has stale intervention semantics.")
        self.config = config
        self.executors = dict(executors)
        self.output_directory = Path(config.output_directory)
        self.config_fingerprint = runner_config_fingerprint(config)

    def dry_run(self) -> dict[str, object]:
        """Return a manifest preview without creating problem instances."""
        specs = expand_run_specs(self.config)
        return {
            "schema_version": RUN_RESULT_SCHEMA_VERSION,
            "dry_run": True,
            "intervention_semantics": self.config.intervention_semantics,
            "campaign_design_identifier": self.config.campaign_design_identifier,
            "config_fingerprint": self.config_fingerprint,
            "configured_run_count": len(specs),
            "run_ids": [spec.run_id for spec in specs],
            "paired_problem_count": len({spec.paired_problem_id for spec in specs}),
            "config": _jsonable(self.config),
        }

    def write_manifest(self, path: Path | None = None) -> Path:
        """Atomically serialize all run specifications and full configuration."""
        specs = expand_run_specs(self.config)
        manifest_path = path or self.output_directory / "manifest.json"
        payload = {
            "schema_version": RUN_RESULT_SCHEMA_VERSION,
            "intervention_semantics": self.config.intervention_semantics,
            "campaign_design_identifier": self.config.campaign_design_identifier,
            "config": _jsonable(self.config),
            "config_fingerprint": self.config_fingerprint,
            "runs": [_jsonable(spec) for spec in specs],
        }
        _atomic_json_write(manifest_path, payload)
        return manifest_path

    def run_id(self, run_id: str, *, resume: bool = False) -> dict[str, object]:
        """Run exactly one ID, recording either result or failure atomically."""
        spec_by_id = {spec.run_id: spec for spec in expand_run_specs(self.config)}
        try:
            spec = spec_by_id[run_id]
        except KeyError as error:
            raise ValueError(f"unknown run ID {run_id!r}.") from error
        result_path = self._result_path(spec)
        failure_path = self._failure_path(spec)
        if failure_path.exists():
            _validate_artifact_fingerprint(failure_path, self.config_fingerprint)
        if result_path.exists():
            _validate_artifact_fingerprint(result_path, self.config_fingerprint)
            return {
                "status": "skipped_completed",
                "run_id": run_id,
                "path": str(result_path),
            }
        if failure_path.exists() and not resume:
            return {
                "status": "skipped_failed_use_resume",
                "run_id": run_id,
                "path": str(failure_path),
            }
        executor = self.executors.get(spec.experiment_mode)
        if executor is None:
            raise RuntimeError(
                f"no executor registered for mode {spec.experiment_mode!r}."
            )
        try:
            raw_outcome = executor(spec)
            outcome = _normalize_outcome(raw_outcome)
            payload = {
                "schema_version": RUN_RESULT_SCHEMA_VERSION,
                "status": "completed",
                "intervention_semantics": self.config.intervention_semantics,
                "campaign_design_identifier": self.config.campaign_design_identifier,
                "config_fingerprint": self.config_fingerprint,
                "run_id": spec.run_id,
                "paired_problem_id": spec.paired_problem_id,
                "cross_noise_regime_pair_id": spec.cross_noise_regime_pair_id,
                "noise_variance_regime": spec.noise_variance_regime,
                "noise_variance_seed": spec.noise_variance_seed,
                "spec": _jsonable(spec),
                "full_config": _jsonable(self.config),
                "metrics": _jsonable(outcome.metrics),
                "termination_reason": outcome.termination_reason,
                "success": outcome.success,
                "scientific_matching_success": _scientific_success_from_values(
                    outcome.metrics,
                    outcome.termination_reason,
                ),
                "reliability": _jsonable(outcome.reliability),
                "protocol_version": outcome.protocol_version,
                "resolved_protocol": _jsonable(outcome.resolved_protocol),
            }
            _atomic_json_write(result_path, payload)
            if failure_path.exists():
                failure_path.unlink()
            return payload
        except Exception as error:  # noqa: BLE001 - durable failure boundary
            failure = {
                "schema_version": RUN_RESULT_SCHEMA_VERSION,
                "status": "failed",
                "intervention_semantics": self.config.intervention_semantics,
                "campaign_design_identifier": self.config.campaign_design_identifier,
                "config_fingerprint": self.config_fingerprint,
                "run_id": spec.run_id,
                "paired_problem_id": spec.paired_problem_id,
                "cross_noise_regime_pair_id": spec.cross_noise_regime_pair_id,
                "noise_variance_regime": spec.noise_variance_regime,
                "noise_variance_seed": spec.noise_variance_seed,
                "spec": _jsonable(spec),
                "full_config": _jsonable(self.config),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
            }
            _atomic_json_write(failure_path, failure)
            return failure

    def _result_path(self, spec: RunSpec) -> Path:
        return self.output_directory / "results" / f"{spec.run_id}.json"

    def _failure_path(self, spec: RunSpec) -> Path:
        return self.output_directory / "failures" / f"{spec.run_id}.json"


def aggregate_results(
    output_directory: Path,
    *,
    bootstrap_replicates: int = 1_000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Aggregate durable result files without changing the legacy top-level keys."""
    result_files = sorted((output_directory / "results").glob("*.json"))
    failure_files = sorted((output_directory / "failures").glob("*.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]
    failure_payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in failure_files
    ]
    return aggregate_result_payloads(
        payloads,
        failure_files=tuple(path.name for path in failure_files),
        failure_payloads=failure_payloads,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )


def aggregate_result_payloads(
    payloads: Sequence[Mapping[str, object]],
    *,
    failure_files: Sequence[str] = (),
    failure_payloads: Sequence[Mapping[str, object]] = (),
    bootstrap_replicates: int = 1_000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Aggregate result payloads overall and by declarative ablation dimensions."""
    semantics = {
        payload.get("intervention_semantics")
        for payload in (*payloads, *failure_payloads)
    }
    if semantics and semantics != {INTERVENTION_SEMANTICS}:
        raise ValueError(
            "analysis refuses missing, old, or mixed intervention semantics: "
            f"{sorted(map(str, semantics))}."
        )
    overall = _summarize_result_payloads(
        payloads,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    unexpected_failure_count = max(len(failure_files), len(failure_payloads))
    expected_run_count = len(payloads) + unexpected_failure_count
    execution_success_rate = (
        len(payloads) / expected_run_count if expected_run_count else None
    )
    groups: dict[
        str,
        tuple[
            dict[str, object],
            list[Mapping[str, object]],
            list[Mapping[str, object]],
        ],
    ] = {}
    for payload in payloads:
        dimensions = _ablation_group_dimensions(payload)
        key = json.dumps(dimensions, sort_keys=True, separators=(",", ":"))
        groups.setdefault(key, (dimensions, [], []))[1].append(payload)
    for failure in failure_payloads:
        dimensions = _ablation_group_dimensions(failure)
        key = json.dumps(dimensions, sort_keys=True, separators=(",", ":"))
        groups.setdefault(key, (dimensions, [], []))[2].append(failure)
    grouped = []
    for index, key in enumerate(sorted(groups), start=1):
        dimensions, group_payloads, group_failures = groups[key]
        summary = _summarize_result_payloads(
            group_payloads,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_seed_from_parts(bootstrap_seed, key),
        )
        group_expected = len(group_payloads) + len(group_failures)
        summary.update(
            {
                "expected_run_count": group_expected,
                "completed_result_count": len(group_payloads),
                "unexpected_execution_failures": len(group_failures),
                "execution_success_rate": (
                    len(group_payloads) / group_expected if group_expected else None
                ),
            }
        )
        grouped.append(
            {
                "group_id": f"group-{index:04d}",
                "dimensions": dimensions,
                **summary,
            }
        )
    return {
        "schema_version": RUN_RESULT_SCHEMA_VERSION,
        "intervention_semantics": INTERVENTION_SEMANTICS,
        "expected_run_count": expected_run_count,
        "completed_result_count": len(payloads),
        "unexpected_execution_failures": unexpected_failure_count,
        "completed_runs": len(payloads),
        "failure_count": unexpected_failure_count,
        "scientific_matching_successes": overall["scientific_matching_successes"],
        "scientific_matching_failures": overall["scientific_matching_failures"],
        "orientation_conflict_frequency": overall["orientation_conflict_frequency"],
        "success_rate": execution_success_rate,
        "execution_success_rate": execution_success_rate,
        "scientific_success_rate": overall["scientific_success_rate"],
        "termination_reason_frequencies": overall["termination_reason_frequencies"],
        "metric_summaries": overall["metric_summaries"],
        "probe_target_sequences": overall["probe_target_sequences"],
        "grouped_ablation_summaries": grouped,
        "failure_files": list(failure_files),
    }


def _summarize_result_payloads(
    payloads: Sequence[Mapping[str, object]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    numeric_metrics: dict[str, list[float]] = {}
    execution_successes: list[bool] = []
    scientific_successes: list[bool] = []
    probe_targets = []
    for payload in payloads:
        metrics = _analysis_ready_metrics(payload)
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_metrics.setdefault(name, []).append(float(value))
        execution_successes.append(bool(payload.get("success", False)))
        scientific_successes.append(_scientific_success(payload))
        source_outcomes = metrics.get("source_outcomes", [])
        if isinstance(source_outcomes, list):
            sequences = []
            for outcome in source_outcomes:
                if not isinstance(outcome, Mapping):
                    continue
                classifications = outcome.get("classifications", [])
                sequences.append(
                    [
                        int(item[0])
                        for item in classifications
                        if isinstance(item, list)
                        and len(item) == 2
                        and isinstance(item[0], int)
                    ]
                )
            probe_targets.append(
                {"run_id": payload.get("run_id"), "targets": sequences}
            )
    rng = np.random.default_rng(bootstrap_seed)
    summaries = {
        name: _numeric_summary(values, bootstrap_replicates, rng)
        for name, values in sorted(numeric_metrics.items())
    }
    for name in _ABLATION_SUMMARY_METRICS:
        summaries.setdefault(name, _empty_numeric_summary())
    terminations = Counter(
        str(payload.get("termination_reason", "missing")) for payload in payloads
    )
    return {
        "expected_run_count": len(payloads),
        "completed_result_count": len(payloads),
        "unexpected_execution_failures": 0,
        "run_count": len(payloads),
        "execution_success_rate": (
            float(np.mean(execution_successes)) if execution_successes else None
        ),
        "scientific_success_rate": (
            float(np.mean(scientific_successes)) if scientific_successes else None
        ),
        "scientific_matching_successes": sum(scientific_successes),
        "scientific_matching_failures": (
            len(scientific_successes) - sum(scientific_successes)
        ),
        "orientation_conflict_frequency": terminations.get(
            "inconsistent_estimated_cpdag", 0
        ),
        "termination_reason_frequencies": dict(sorted(terminations.items())),
        "metric_summaries": summaries,
        "probe_target_sequences": probe_targets,
    }


def _analysis_ready_metrics(payload: Mapping[str, object]) -> dict[str, object]:
    raw_metrics = payload.get("metrics", {})
    metrics = dict(raw_metrics) if isinstance(raw_metrics, Mapping) else {}
    initial = metrics.get("initial_global_reverse_kl")
    final = metrics.get("final_global_reverse_kl")
    if _is_number(initial) and _is_number(final):
        reduction = float(initial) - float(final)
        metrics["absolute_kl_reduction"] = reduction
        if float(initial) > 0.0:
            metrics["relative_kl_reduction"] = reduction / float(initial)
    target = metrics.get("target_parameter_coordinate_support")
    accepted = metrics.get("accepted_final_support")
    if isinstance(target, list) and isinstance(accepted, list):
        target_set = {str(value) for value in target}
        accepted_set = {str(value) for value in accepted}
        overlap = len(target_set & accepted_set)
        metrics["target_support_recall"] = (
            overlap / len(target_set) if target_set else 1.0
        )
        metrics["recovered_support_precision"] = (
            overlap / len(accepted_set) if accepted_set else float(not target_set)
        )
    aliases = {
        "accepted_final_coordinates": metrics.get(
            "accepted_final_coordinate_count",
            len(accepted) if isinstance(accepted, list) else None,
        ),
        "candidate_evaluations": metrics.get(
            "candidate_search_evaluations",
            metrics.get("optimizer_search_evaluations"),
        ),
        "executed_candidate_queries": metrics.get("candidate_queries"),
        "total_environment_calls": metrics.get("algorithm_environment_calls"),
    }
    metrics.update(
        {name: value for name, value in aliases.items() if _is_number(value)}
    )
    metrics["runner_execution_success"] = float(bool(payload.get("success", False)))
    metrics["scientific_matching_success"] = float(_scientific_success(payload))
    return metrics


def _scientific_success(payload: Mapping[str, object]) -> bool:
    declared = payload.get("scientific_matching_success")
    if isinstance(declared, bool):
        return declared
    metrics = payload.get("metrics", {})
    return _scientific_success_from_values(
        metrics,
        str(payload.get("termination_reason", "missing")),
    )


def _scientific_success_from_values(
    metrics: object,
    termination_reason: str,
) -> bool:
    if termination_reason == "inconsistent_estimated_cpdag":
        return False
    values = _jsonable(metrics)
    if not isinstance(values, Mapping):
        return False
    final = values.get("final_global_reverse_kl")
    if not _is_number(final):
        return False
    tolerance = 1e-8
    settings = values.get("effective_runtime_settings")
    if isinstance(settings, Mapping) and _is_number(
        settings.get("global_loss_tolerance")
    ):
        tolerance = float(settings["global_loss_tolerance"])
    return float(final) <= tolerance


def _ablation_group_dimensions(payload: Mapping[str, object]) -> dict[str, object]:
    full_config = payload.get("full_config", {})
    metadata = (
        full_config.get("ablation_metadata", {})
        if isinstance(full_config, Mapping)
        else {}
    )
    group_values = (
        metadata.get("group_values", {}) if isinstance(metadata, Mapping) else {}
    )
    dimensions = dict(group_values) if isinstance(group_values, Mapping) else {}
    spec = payload.get("spec", {})
    metrics = payload.get("metrics", {})
    for output_name, source_name in (
        ("graph_family", "graph_family"),
        ("edge_probability", "edge_probability"),
        ("node_count", "n_nodes"),
        ("support_ratio", "rho"),
        ("intervention_composition", "intervention_composition"),
        ("intervention_action_class", "intervention_action_class"),
        ("magnitude_regime", "intervention_magnitude_regime"),
        ("policy", "policy"),
        ("coefficient_regime", "coefficient_regime"),
        ("final_coordinate_budget", "final_parameter_coordinate_budget"),
    ):
        if output_name in dimensions:
            continue
        if isinstance(spec, Mapping) and source_name in spec:
            dimensions[output_name] = spec[source_name]
        elif isinstance(metrics, Mapping) and source_name in metrics:
            dimensions[output_name] = metrics[source_name]
    return dimensions


def _empty_numeric_summary() -> dict[str, float | int | None]:
    return {
        "count": 0,
        "mean": None,
        "standard_deviation": None,
        "median": None,
        "quantile_05": None,
        "quantile_25": None,
        "quantile_75": None,
        "quantile_95": None,
        "bootstrap_mean_ci_lower": None,
        "bootstrap_mean_ci_upper": None,
    }


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def write_aggregate_tables(
    output_directory: Path,
    aggregate: Mapping[str, object],
) -> tuple[Path, Path]:
    """Write machine-readable JSON and CSV aggregate tables atomically."""
    json_path = output_directory / "aggregate.json"
    csv_path = output_directory / "aggregate.csv"
    _atomic_json_write(json_path, dict(aggregate))
    rows = []
    metric_summaries = aggregate.get("metric_summaries", {})
    if isinstance(metric_summaries, Mapping):
        for metric, summary in metric_summaries.items():
            if isinstance(summary, Mapping):
                rows.append({"metric": metric, **summary})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(csv_path)
    fieldnames = ["metric"] + sorted(
        {key for row in rows for key in row if key != "metric"}
    )
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, csv_path)
    return json_path, csv_path


def config_from_json(path: Path) -> RunnerConfig:
    """Reconstruct and fingerprint-check the complete runner configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("intervention_semantics") != INTERVENTION_SEMANTICS:
        raise ValueError(
            f"{path} is missing or has stale intervention_semantics; "
            f"expected {INTERVENTION_SEMANTICS!r}."
        )
    raw_dimensions = raw.pop("dimensions")
    if "graph_densities" in raw_dimensions:
        if "edge_probabilities" in raw_dimensions:
            raise ValueError(f"{path} mixes old and new edge-probability fields.")
        raw_dimensions["edge_probabilities"] = raw_dimensions.pop("graph_densities")
    dimensions = RunnerDimensions(**raw_dimensions)
    config = RunnerConfig(dimensions=dimensions, **raw)
    metadata = config.ablation_metadata
    if isinstance(metadata, Mapping):
        declared = metadata.get("config_fingerprint")
        if declared is not None and declared != runner_config_fingerprint(config):
            raise ValueError(f"{path} has an invalid config_fingerprint.")
    return config


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for dry-run, manifest, one-run resume, and aggregation."""
    parser = argparse.ArgumentParser(
        description="Restartable one-RunSpec experiment runner.",
        epilog=(
            "Built-in exact/noiseless execution:\n"
            "  python -m experiments.runner --config CONFIG --run-id RUN_ID "
            "--executor-module experiments.executors\n"
            "Finite-sample production executors are intentionally unavailable until "
            "scoped calibration readiness passes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="preview run IDs only")
    parser.add_argument(
        "--write-manifest", action="store_true", help="atomically write manifest.json"
    )
    parser.add_argument("--run-id", help="execute exactly one manifest run ID")
    parser.add_argument(
        "--executor-module",
        help=(
            "import path exposing EXPERIMENT_EXECUTORS; built-in exact/noiseless "
            "module: experiments.executors"
        ),
    )
    parser.add_argument(
        "--resume", action="store_true", help="retry a recorded failed run ID"
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="write aggregate.json and aggregate.csv",
    )
    arguments = parser.parse_args(argv)
    config = config_from_json(arguments.config)
    executors: Mapping[ExperimentMode, RunExecutor] = {}
    if arguments.executor_module:
        module = importlib.import_module(arguments.executor_module)
        registry = getattr(module, "EXPERIMENT_EXECUTORS", None)
        if not isinstance(registry, Mapping):
            parser.error(
                "--executor-module must expose a mapping named EXPERIMENT_EXECUTORS"
            )
        executors = registry
    runner = RestartableExperimentRunner(config, executors=executors)
    if arguments.dry_run:
        print(json.dumps(runner.dry_run(), indent=2, sort_keys=True))
    if arguments.write_manifest:
        print(runner.write_manifest())
    if arguments.aggregate:
        aggregate = aggregate_results(
            Path(config.output_directory),
            bootstrap_replicates=config.bootstrap_aggregate_replicates,
        )
        paths = write_aggregate_tables(Path(config.output_directory), aggregate)
        print(*(str(path) for path in paths))
    if arguments.run_id:
        if not arguments.executor_module:
            parser.error(
                "--run-id requires --executor-module; use "
                "--executor-module experiments.executors for built-in exact/noiseless modes"
            )
        print(
            json.dumps(
                runner.run_id(arguments.run_id, resume=arguments.resume),
                indent=2,
                sort_keys=True,
            )
        )
    if not any(
        (
            arguments.dry_run,
            arguments.write_manifest,
            arguments.aggregate,
            arguments.run_id,
        )
    ):
        parser.error("select --dry-run, --write-manifest, --run-id, or --aggregate")
    return 0


def _shared_run_seeds(paired_id: str, base_seed: int) -> dict[str, int]:
    return {
        name: _seed_from_parts(paired_id, base_seed, name)
        for name in (
            "graph",
            "coefficients",
            "target",
            "observational_samples",
            "environment_samples",
            "probes",
            "coordinate_search",
            "bootstrap",
        )
    }


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % np.iinfo(np.int32).max


def _stable_id(prefix: str, payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{prefix}-{digest[:20]}"


def _normalize_outcome(raw: object) -> RunOutcome:
    if isinstance(raw, RunOutcome):
        return raw
    if isinstance(raw, Mapping):
        if {
            "metrics",
            "termination_reason",
            "success",
            "reliability",
        } <= set(raw):
            return RunOutcome(
                metrics=raw["metrics"],  # type: ignore[arg-type]
                termination_reason=str(raw["termination_reason"]),
                protocol_version=str(raw.get("protocol_version", "")),
                resolved_protocol=raw.get("resolved_protocol", {}),  # type: ignore[arg-type]
                success=bool(raw["success"]),
                reliability=raw["reliability"],  # type: ignore[arg-type]
            )
        raise ValueError(
            "mapping executor outcomes must include metrics, termination, reliability, "
            "success, protocol_version, and resolved_protocol."
        )
    protocol_version = getattr(raw, "protocol_version", None)
    resolved_protocol = getattr(raw, "resolved_protocol", None)
    if not isinstance(protocol_version, str) or not isinstance(
        resolved_protocol, Mapping
    ):
        raise TypeError("executor outcomes must expose complete protocol metadata.")
    return RunOutcome(
        metrics={"result": _jsonable(raw)},
        termination_reason=str(getattr(raw, "termination_reason", "unspecified")),
        success=bool(getattr(raw, "success", False)),
        reliability={},
        protocol_version=protocol_version,
        resolved_protocol=resolved_protocol,
    )


def _numeric_summary(
    values: list[float],
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    bootstrap_means = np.empty(bootstrap_replicates)
    for replicate in range(bootstrap_replicates):
        bootstrap_means[replicate] = np.mean(
            rng.choice(array, size=array.size, replace=True)
        )
    return {
        "count": len(values),
        "mean": mean(values),
        "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "quantile_05": float(np.quantile(array, 0.05)),
        "quantile_25": float(np.quantile(array, 0.25)),
        "quantile_75": float(np.quantile(array, 0.75)),
        "quantile_95": float(np.quantile(array, 0.95)),
        "bootstrap_mean_ci_lower": float(np.quantile(bootstrap_means, 0.025)),
        "bootstrap_mean_ci_upper": float(np.quantile(bootstrap_means, 0.975)),
    }


def _atomic_json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    with temporary.open("w") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _temporary_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(name)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    sys.modules.setdefault("experiments.runner", sys.modules[__name__])
    raise SystemExit(main())
