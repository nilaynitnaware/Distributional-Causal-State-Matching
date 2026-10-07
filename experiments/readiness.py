"""Versioned finite-sample readiness gates over calibration aggregates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ReadinessStatus = Literal[
    "ENGINEERING_READY",
    "CALIBRATION_REQUIRED",
    "CALIBRATION_FAILED",
    "SCIENTIFICALLY_READY_UNDER_TESTED_REGIME",
]
GateDecision = Literal["pass", "fail", "insufficient_evidence"]
ThresholdDirection = Literal["maximum", "minimum"]


@dataclass(frozen=True)
class ReadinessThreshold:
    gate_group: str
    artifact: str
    metric: str
    direction: ThresholdDirection
    value: float

    def evaluate(self, observed: object) -> GateDecision:
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            return "insufficient_evidence"
        if self.direction == "maximum":
            return "pass" if float(observed) <= self.value else "fail"
        return "pass" if float(observed) >= self.value else "fail"


@dataclass(frozen=True)
class ReadinessPolicy:
    policy_version: str
    thresholds: tuple[ReadinessThreshold, ...]
    required_artifacts: tuple[str, ...]


finite_sample_readiness_policy_v1 = ReadinessPolicy(
    policy_version="finite_sample_readiness_policy_v1",
    required_artifacts=(
        "source_classifier_calibration",
        "candidate_acceptance_calibration",
        "local_unresolved_calibration",
        "PC_structure_calibration",
        "covariance_KL_calibration",
        "end_to_end_diagnostic",
    ),
    thresholds=(
        ReadinessThreshold(
            "source_inference",
            "source_classifier_calibration",
            "source_false_positive_rate",
            "maximum",
            0.05,
        ),
        ReadinessThreshold(
            "source_inference",
            "source_classifier_calibration",
            "nonsource_detection_power",
            "minimum",
            0.80,
        ),
        ReadinessThreshold(
            "source_inference",
            "source_classifier_calibration",
            "inconclusive_rate",
            "maximum",
            0.20,
        ),
        ReadinessThreshold(
            "source_inference",
            "source_classifier_calibration",
            "confidence_interval_coverage",
            "minimum",
            0.90,
        ),
        ReadinessThreshold(
            "candidate_acceptance",
            "candidate_acceptance_calibration",
            "harmful_proposal_acceptance_rate",
            "maximum",
            0.05,
        ),
        ReadinessThreshold(
            "candidate_acceptance",
            "candidate_acceptance_calibration",
            "clearly_beneficial_rejection_rate",
            "maximum",
            0.10,
        ),
        ReadinessThreshold(
            "candidate_acceptance",
            "candidate_acceptance_calibration",
            "confidence_interval_coverage",
            "minimum",
            0.90,
        ),
        ReadinessThreshold(
            "candidate_acceptance",
            "candidate_acceptance_calibration",
            "inconclusive_rate",
            "maximum",
            0.20,
        ),
        ReadinessThreshold(
            "structure_estimation",
            "PC_structure_calibration",
            "skeleton_precision",
            "minimum",
            0.80,
        ),
        ReadinessThreshold(
            "structure_estimation",
            "PC_structure_calibration",
            "skeleton_recall",
            "minimum",
            0.80,
        ),
        ReadinessThreshold(
            "structure_estimation",
            "PC_structure_calibration",
            "collider_precision",
            "minimum",
            0.70,
        ),
        ReadinessThreshold(
            "structure_estimation",
            "PC_structure_calibration",
            "invalid_cpdag_rate",
            "maximum",
            0.05,
        ),
        ReadinessThreshold(
            "empirical_kl",
            "covariance_KL_calibration",
            "median_relative_kl_error",
            "maximum",
            0.20,
        ),
        ReadinessThreshold(
            "empirical_kl",
            "covariance_KL_calibration",
            "confidence_interval_coverage",
            "minimum",
            0.90,
        ),
        ReadinessThreshold(
            "empirical_kl",
            "covariance_KL_calibration",
            "numerical_failure_rate",
            "maximum",
            0.05,
        ),
        ReadinessThreshold(
            "end_to_end",
            "end_to_end_diagnostic",
            "numerical_failure_rate",
            "maximum",
            0.05,
        ),
        ReadinessThreshold(
            "end_to_end",
            "end_to_end_diagnostic",
            "oracle_leakage_validation_failure_rate",
            "maximum",
            0.0,
        ),
        ReadinessThreshold(
            "end_to_end",
            "end_to_end_diagnostic",
            "accounting_invariant_failure_rate",
            "maximum",
            0.0,
        ),
    ),
)


@dataclass(frozen=True)
class GateResult:
    gate_group: str
    artifact: str
    metric: str
    decision: GateDecision
    observed: float | None
    threshold_direction: ThresholdDirection
    threshold: float


@dataclass(frozen=True)
class ReadinessEvaluation:
    status: ReadinessStatus
    policy_version: str
    protocol_version: str
    regime_id: str
    gate_results: tuple[GateResult, ...]
    missing_artifacts: tuple[str, ...]
    explanation: str

    @property
    def scope_label(self) -> str:
        return (
            f"regime={self.regime_id}; protocol={self.protocol_version}; "
            f"policy={self.policy_version}"
        )


def evaluate_finite_sample_readiness(
    artifacts: Mapping[str, Path | Mapping[str, object]],
    *,
    protocol_version: str,
    regime_id: str,
    policy: ReadinessPolicy = finite_sample_readiness_policy_v1,
) -> ReadinessEvaluation:
    """Evaluate only complete, correctly scoped calibration artifacts."""
    loaded = {name: _load_artifact(value) for name, value in artifacts.items()}
    missing = tuple(name for name in policy.required_artifacts if name not in loaded)
    if missing:
        return ReadinessEvaluation(
            "CALIBRATION_REQUIRED",
            policy.policy_version,
            protocol_version,
            regime_id,
            (),
            missing,
            "required calibration aggregates are missing",
        )

    gate_results = []
    for threshold in policy.thresholds:
        artifact = loaded[threshold.artifact]
        scoped = _artifact_matches_scope(artifact, protocol_version, regime_id)
        metrics = artifact.get("metrics", {}) if scoped else {}
        observed = (
            metrics.get(threshold.metric) if isinstance(metrics, Mapping) else None
        )
        decision = threshold.evaluate(observed)
        gate_results.append(
            GateResult(
                threshold.gate_group,
                threshold.artifact,
                threshold.metric,
                decision,
                (
                    float(observed)
                    if isinstance(observed, (int, float))
                    and not isinstance(observed, bool)
                    else None
                ),
                threshold.direction,
                threshold.value,
            )
        )
    decisions = {result.decision for result in gate_results}
    if "insufficient_evidence" in decisions:
        status: ReadinessStatus = "CALIBRATION_REQUIRED"
        explanation = "calibration metrics are missing or belong to another scope"
    elif "fail" in decisions:
        status = "CALIBRATION_FAILED"
        explanation = "one or more versioned readiness gates failed"
    else:
        status = "SCIENTIFICALLY_READY_UNDER_TESTED_REGIME"
        explanation = (
            "all declared gates passed only for the exact tested regime and protocol"
        )
    return ReadinessEvaluation(
        status,
        policy.policy_version,
        protocol_version,
        regime_id,
        tuple(gate_results),
        (),
        explanation,
    )


def engineering_ready(
    *, protocol_version: str, regime_id: str, explanation: str
) -> ReadinessEvaluation:
    return ReadinessEvaluation(
        "ENGINEERING_READY",
        finite_sample_readiness_policy_v1.policy_version,
        protocol_version,
        regime_id,
        (),
        (),
        explanation,
    )


def _load_artifact(value: Path | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(value, Path):
        if not value.exists():
            return {}
        raw = json.loads(value.read_text())
        return raw if isinstance(raw, Mapping) else {}
    return value


def _artifact_matches_scope(
    artifact: Mapping[str, object], protocol_version: str, regime_id: str
) -> bool:
    protocols = artifact.get("protocol_versions", ())
    regimes = artifact.get("regime_ids", ())
    if isinstance(protocols, str):
        protocols = (protocols,)
    if isinstance(regimes, str):
        regimes = (regimes,)
    return (
        isinstance(protocols, (tuple, list))
        and isinstance(regimes, (tuple, list))
        and protocol_version in protocols
        and regime_id in regimes
    )
