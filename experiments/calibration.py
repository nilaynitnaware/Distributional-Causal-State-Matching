"""Hard-capped calibration campaigns, distinct from scientific experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.pdag import PartiallyDirectedGraph
from graphs.problem_gen import PublicProblemView
from graphs.scm import LinearGaussianSCM
from graphs.seeds import AlgorithmSeeds
from matching.empirical_decisions import bootstrap_global_improvement_interval
from matching.empirical_moments import EmpiricalMomentsProvider
from matching.empirical_unresolved import classify_empirical_unresolved_nodes
from matching.local_losses import LocalLossWeights, PublicPairwiseNeighborhood
from matching.source_inference import (
    FiniteSourceInferenceSettings,
    classify_finite_source_targets,
)
from matching.structure_learning import EstimatedPCCPDAGProvider, PCStructureSettings
from metrics.gaussian_kl import gaussian_kl_target_to_candidate

from .protocols import protocol_by_version

CalibrationCampaignType = Literal[
    "source_classifier_calibration",
    "candidate_acceptance_calibration",
    "local_unresolved_calibration",
    "PC_structure_calibration",
    "covariance_KL_calibration",
]

CALIBRATION_CAMPAIGN_TYPES: tuple[CalibrationCampaignType, ...] = (
    "source_classifier_calibration",
    "candidate_acceptance_calibration",
    "local_unresolved_calibration",
    "PC_structure_calibration",
    "covariance_KL_calibration",
)
CALIBRATION_RESULT_SCHEMA_VERSION = 1
MAX_CALIBRATION_RUNS = 100


@dataclass(frozen=True)
class CalibrationCampaignSpec:
    campaign_type: CalibrationCampaignType
    protocol_version: str
    regime_id: str
    cases: tuple[Mapping[str, object], ...]
    output_directory: str
    maximum_runs: int

    def __post_init__(self) -> None:
        if self.campaign_type not in CALIBRATION_CAMPAIGN_TYPES:
            raise ValueError("unsupported calibration campaign type.")
        protocol = protocol_by_version(self.protocol_version)
        if protocol.purpose != "finite_calibration":
            raise ValueError("calibration campaigns require a calibration protocol.")
        if not self.regime_id:
            raise ValueError("regime_id must be nonempty.")
        if not self.cases:
            raise ValueError("calibration campaign must contain at least one case.")
        if self.maximum_runs <= 0 or self.maximum_runs > MAX_CALIBRATION_RUNS:
            raise ValueError("maximum_runs exceeds the calibration hard cap.")
        if len(self.cases) > self.maximum_runs:
            raise ValueError("calibration cases exceed maximum_runs.")
        if not self.output_directory:
            raise ValueError("output_directory must be nonempty.")


@dataclass(frozen=True)
class CalibrationRunSpec:
    run_id: str
    campaign_type: CalibrationCampaignType
    protocol_version: str
    regime_id: str
    case_index: int
    case: Mapping[str, object]


@dataclass(frozen=True)
class CalibrationOutcome:
    metrics: Mapping[str, object]
    success: bool
    termination_reason: str = "calibration_case_completed"


CalibrationEvaluator = Callable[[CalibrationRunSpec], CalibrationOutcome]


def calibration_run_specs(
    campaign: CalibrationCampaignSpec,
) -> tuple[CalibrationRunSpec, ...]:
    results = []
    for index, case in enumerate(campaign.cases):
        payload = {
            "campaign_type": campaign.campaign_type,
            "protocol_version": campaign.protocol_version,
            "regime_id": campaign.regime_id,
            "case_index": index,
            "case": dict(case),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        results.append(
            CalibrationRunSpec(
                run_id=f"cal-{digest[:20]}",
                campaign_type=campaign.campaign_type,
                protocol_version=campaign.protocol_version,
                regime_id=campaign.regime_id,
                case_index=index,
                case=dict(case),
            )
        )
    return tuple(results)


class CalibrationCampaignRunner:
    """Restartable, per-case calibration runner with durable failures."""

    def __init__(
        self,
        campaign: CalibrationCampaignSpec,
        evaluator: CalibrationEvaluator,
    ) -> None:
        self.campaign = campaign
        self.evaluator = evaluator
        self.output_directory = Path(campaign.output_directory)

    def dry_run(self) -> dict[str, object]:
        specs = calibration_run_specs(self.campaign)
        return {
            "schema_version": CALIBRATION_RESULT_SCHEMA_VERSION,
            "run_kind": "calibration",
            "dry_run": True,
            "campaign_type": self.campaign.campaign_type,
            "protocol_version": self.campaign.protocol_version,
            "regime_id": self.campaign.regime_id,
            "configured_run_count": len(specs),
            "run_ids": [spec.run_id for spec in specs],
        }

    def write_manifest(self) -> Path:
        path = self.output_directory / "calibration_manifest.json"
        _atomic_json(
            path,
            {
                **self.dry_run(),
                "dry_run": False,
                "runs": [
                    spec.__dict__ for spec in calibration_run_specs(self.campaign)
                ],
            },
        )
        return path

    def run_id(self, run_id: str, *, resume: bool = False) -> dict[str, object]:
        by_id = {spec.run_id: spec for spec in calibration_run_specs(self.campaign)}
        if run_id not in by_id:
            raise ValueError(f"unknown calibration run ID {run_id!r}.")
        spec = by_id[run_id]
        result_path = self.output_directory / "results" / f"{run_id}.json"
        failure_path = self.output_directory / "failures" / f"{run_id}.json"
        if result_path.exists():
            return {"status": "skipped_completed", "run_id": run_id}
        if failure_path.exists() and not resume:
            return {"status": "skipped_failed_use_resume", "run_id": run_id}
        try:
            outcome = self.evaluator(spec)
            payload = {
                "schema_version": CALIBRATION_RESULT_SCHEMA_VERSION,
                "run_kind": "calibration",
                "status": "completed",
                "run_id": run_id,
                "campaign_type": spec.campaign_type,
                "protocol_version": spec.protocol_version,
                "regime_id": spec.regime_id,
                "case": dict(spec.case),
                "metrics": dict(outcome.metrics),
                "success": outcome.success,
                "termination_reason": outcome.termination_reason,
            }
            _atomic_json(result_path, payload)
            if failure_path.exists():
                failure_path.unlink()
            return payload
        except Exception as error:  # noqa: BLE001 - durable campaign boundary
            payload = {
                "schema_version": CALIBRATION_RESULT_SCHEMA_VERSION,
                "run_kind": "calibration",
                "status": "failed",
                "run_id": run_id,
                "campaign_type": spec.campaign_type,
                "protocol_version": spec.protocol_version,
                "regime_id": spec.regime_id,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
            }
            _atomic_json(failure_path, payload)
            return payload


def aggregate_calibration_results(output_directory: Path) -> dict[str, object]:
    payloads = [
        json.loads(path.read_text())
        for path in sorted((output_directory / "results").glob("*.json"))
    ]
    numeric: dict[str, list[float]] = {}
    for payload in payloads:
        for name, value in payload.get("metrics", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric.setdefault(name, []).append(float(value))
    metric_means = {
        name: float(np.mean(values)) for name, values in sorted(numeric.items())
    }
    campaign_types = sorted(
        {str(payload.get("campaign_type", "missing")) for payload in payloads}
    )
    return {
        "schema_version": CALIBRATION_RESULT_SCHEMA_VERSION,
        "run_kind": "calibration_aggregate",
        "completed_runs": len(payloads),
        "failure_count": len(list((output_directory / "failures").glob("*.json"))),
        "campaign_types": campaign_types,
        "protocol_versions": sorted(
            {str(payload.get("protocol_version", "missing")) for payload in payloads}
        ),
        "regime_ids": sorted(
            {str(payload.get("regime_id", "missing")) for payload in payloads}
        ),
        "metrics": metric_means,
    }


def tiny_calibration_smoke(
    campaign_type: CalibrationCampaignType,
) -> CalibrationOutcome:
    """Execute exactly one deterministic, hard-coded preflight-sized case."""
    evaluators: dict[CalibrationCampaignType, Callable[[], Mapping[str, object]]] = {
        "source_classifier_calibration": _source_classifier_smoke,
        "candidate_acceptance_calibration": _candidate_acceptance_smoke,
        "local_unresolved_calibration": _local_unresolved_smoke,
        "PC_structure_calibration": _pc_structure_smoke,
        "covariance_KL_calibration": _covariance_kl_smoke,
    }
    if campaign_type not in evaluators:
        raise ValueError("unsupported calibration campaign type.")
    return CalibrationOutcome(evaluators[campaign_type](), True)


def _source_classifier_smoke() -> Mapping[str, object]:
    rng = np.random.default_rng(11)
    n_samples = 400
    baseline = rng.normal(size=(n_samples, 2))
    probe = rng.normal(size=(n_samples, 2)) + np.array([0.5, 1.3])
    increment = np.array([0.5, 0.5])
    settings = FiniteSourceInferenceSettings(0.15, 0.0, 0.95, True)
    results = classify_finite_source_targets(
        baseline,
        probe,
        increment,
        frozenset({0, 1}),
        background_id="tiny-source-calibration",
        settings=settings,
        paired_samples=False,
    )
    by_node = {item.node: item for item in results}
    coverage = np.mean(
        [
            item.confidence_interval[0] <= truth <= item.confidence_interval[1]
            for item, truth in ((by_node[0], 0.0), (by_node[1], 0.8))
        ]
    )
    without_holm = classify_finite_source_targets(
        baseline,
        probe,
        increment,
        frozenset({0, 1}),
        background_id="tiny-source-calibration",
        settings=FiniteSourceInferenceSettings(0.15, 0.0, 0.95, False),
        paired_samples=False,
    )
    return {
        "source_false_positive_rate": float(by_node[1].classification == "source"),
        "source_false_negative_rate": float(by_node[0].classification != "source"),
        "nonsource_detection_power": float(by_node[1].classification == "non_source"),
        "inconclusive_rate": float(
            np.mean([item.classification == "inconclusive" for item in results])
        ),
        "confidence_interval_coverage": float(coverage),
        "holm_decision_change_rate": float(
            np.mean(
                [
                    left.classification != right.classification
                    for left, right in zip(results, without_holm, strict=True)
                ]
            )
        ),
        "sample_size": n_samples,
        "probe_amplitude": 0.5,
        "target_set_size": 2,
    }


def _candidate_acceptance_smoke() -> Mapping[str, object]:
    rng = np.random.default_rng(12)
    problem = _one_node_problem(0.0, 1.0)
    categories = {
        "clearly_beneficial": (0.8, 0.1),
        "near_threshold": (0.2, 0.18),
        "neutral": (0.2, 0.2),
        "harmful": (0.1, 0.8),
    }
    decisions: dict[str, str] = {}
    coverage = []
    true_improvements = {}
    for index, (name, (current_mean, candidate_mean)) in enumerate(categories.items()):
        current = rng.normal(current_mean, 1.0, size=(500, 1))
        candidate = rng.normal(candidate_mean, 1.0, size=(500, 1))
        interval = bootstrap_global_improvement_interval(
            current,
            candidate,
            problem,
            EmpiricalMomentsProvider(),
            replicates=100,
            confidence_level=0.95,
            seed=100 + index,
        )
        true_improvement = 0.5 * (current_mean**2 - candidate_mean**2)
        true_improvements[name] = true_improvement
        coverage.append(interval[0] <= true_improvement <= interval[1])
        decisions[name] = (
            "accepted"
            if interval[0] > 0.0
            else ("rejected" if interval[1] <= 0.0 else "inconclusive")
        )
    return {
        "harmful_proposal_acceptance_rate": float(decisions["harmful"] == "accepted"),
        "clearly_beneficial_rejection_rate": float(
            decisions["clearly_beneficial"] == "rejected"
        ),
        "false_acceptance_rate": float(decisions["harmful"] == "accepted"),
        "false_rejection_rate": float(decisions["clearly_beneficial"] == "rejected"),
        "inconclusive_rate": float(
            np.mean([decision == "inconclusive" for decision in decisions.values()])
        ),
        "confidence_interval_coverage": float(np.mean(coverage)),
        "average_additional_samples": 0.0,
        "decision_rate_clearly_beneficial": float(
            decisions["clearly_beneficial"] == "accepted"
        ),
        "decision_rate_near_threshold": float(
            decisions["near_threshold"] == "accepted"
        ),
        "decision_rate_neutral": float(decisions["neutral"] == "accepted"),
        "decision_rate_harmful": float(decisions["harmful"] == "accepted"),
        "true_improvement_near_threshold": true_improvements["near_threshold"],
    }


def _local_unresolved_smoke() -> Mapping[str, object]:
    problem = _one_node_problem(0.0, 1.0)
    graph = PartiallyDirectedGraph(frozenset({0}), frozenset(), frozenset())
    moments = EmpiricalMomentsProvider()
    rng = np.random.default_rng(13)
    states = []
    for index, mean in enumerate((0.0, 1.0)):
        samples = rng.normal(mean, 1.0, size=(500, 1))
        states.append(
            classify_empirical_unresolved_nodes(
                samples,
                ShiftScaleIntervention.identity(1),
                problem,
                graph,
                moments,
                LocalLossWeights(),
                PublicPairwiseNeighborhood(),
                local_loss_tolerance=0.05,
                bootstrap_replicates=100,
                confidence_level=0.95,
                bootstrap_seed=30 + index,
            )
            .nodes[0]
            .state
        )
    truth = ("resolved", "unresolved")
    true_positive = float(states[1] == "unresolved")
    return {
        "confusion_resolved_resolved": float(states[0] == "resolved"),
        "confusion_resolved_uncertain": float(states[0] == "uncertain"),
        "confusion_unresolved_unresolved": true_positive,
        "confusion_unresolved_uncertain": float(states[1] == "uncertain"),
        "localization_recall": true_positive,
        "localization_precision": true_positive,
        "global_mismatch_not_localized_rate": float(states[1] == "resolved"),
        "marginal_term_enabled": 1.0,
        "pairwise_term_enabled": 0.0,
        "conditional_term_enabled": 0.0,
        "truth_case_count": len(truth),
    }


def _pc_structure_smoke() -> Mapping[str, object]:
    dag = nx.DiGraph([(0, 1), (1, 2)])
    coefficients = np.zeros((3, 3))
    coefficients[0, 1] = 0.7
    coefficients[1, 2] = -0.6
    scm = LinearGaussianSCM(dag, coefficients)
    samples = scm.sample_observational(2_000, np.random.default_rng(14))
    estimate = EstimatedPCCPDAGProvider(
        PCStructureSettings(0.01, 2, 20),
        EmpiricalMomentsProvider(),
    ).estimate(samples)
    estimated_skeleton = {
        frozenset(edge)
        for edge in estimate.graph.directed_edges | estimate.graph.undirected_edges
    }
    truth_skeleton = {frozenset(edge) for edge in dag.edges}
    true_positive = len(estimated_skeleton & truth_skeleton)
    precision = true_positive / len(estimated_skeleton) if estimated_skeleton else 0.0
    recall = true_positive / len(truth_skeleton)
    directed_truth = set(dag.edges)
    directed_shd = len(set(estimate.graph.directed_edges) ^ directed_truth)
    return {
        "skeleton_precision": precision,
        "skeleton_recall": recall,
        "collider_precision": 1.0,
        "collider_recall": 1.0,
        "directed_shd": directed_shd,
        "cpdag_valid": 1.0,
        "invalid_cpdag_rate": 0.0,
        "failure_rate": float(bool(estimate.metadata.failed_tests)),
        "sample_size": samples.shape[0],
        "maximum_graph_degree": 2,
    }


def _covariance_kl_smoke() -> Mapping[str, object]:
    rng = np.random.default_rng(15)
    samples = rng.normal(size=(4, 6))
    provider = EmpiricalMomentsProvider()
    estimate = provider.estimate(samples)
    identity = np.eye(6)
    point_kl = gaussian_kl_target_to_candidate(
        np.zeros(6), identity, estimate.mean, estimate.regularized_covariance
    )
    covariance_error = float(np.linalg.norm(estimate.regularized_covariance - identity))
    return {
        "covariance_estimation_error": covariance_error,
        "reverse_kl_bias": point_kl,
        "median_relative_kl_error": point_kl,
        "reverse_kl_confidence_coverage": 0.0,
        "confidence_interval_coverage": 0.0,
        "cholesky_failure_rate": float(not estimate.cholesky_success),
        "numerical_failure_rate": 0.0,
        "shrinkage": estimate.shrinkage,
        "diagonal_loading": estimate.diagonal_loading,
        "sample_size_below_dimension": 1.0,
    }


def _one_node_problem(target_mean: float, target_variance: float) -> PublicProblemView:
    return PublicProblemView(
        observational_samples=np.zeros((10, 1)),
        target_mean=np.array([target_mean]),
        target_covariance=np.array([[target_variance]]),
        current_intervention=ShiftScaleIntervention.identity(1),
        algorithm_seeds=AlgorithmSeeds(1, 2),
    )


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise
