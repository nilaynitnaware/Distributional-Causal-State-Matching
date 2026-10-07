"""Explicit policies for converting local-loss evidence into actionability."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.pdag import PartiallyDirectedGraph
from graphs.problem_gen import PublicProblemView

from .capabilities import ProviderCapability
from .empirical_unresolved import EmpiricalUnresolvedState
from .local_losses import LocalLossWeights, PublicPairwiseNeighborhood


@dataclass(frozen=True)
class LocalEvidenceApplication:
    """Policy-adjusted state plus complete per-node decision records."""

    state: EmpiricalUnresolvedState
    decisions: tuple[dict[str, object], ...]


@runtime_checkable
class LocalEvidenceProvider(Protocol):
    """Convert already-computed empirical local evidence into gate labels."""

    policy_identifier: str
    capabilities: frozenset[ProviderCapability]

    def apply(
        self,
        empirical_state: EmpiricalUnresolvedState,
        *,
        intervention: ShiftScaleIntervention,
        problem: PublicProblemView,
        graph: PartiallyDirectedGraph,
        weights: LocalLossWeights,
        neighborhood: PublicPairwiseNeighborhood,
    ) -> LocalEvidenceApplication:
        """Return the state consumed downstream and auditable decisions."""


@dataclass(frozen=True)
class AdaptiveEvidenceSampleBatch:
    """One public sample batch returned by an adaptive target sampler."""

    samples: np.ndarray
    distribution_role: str
    adaptive_stage: int
    replication_index: int
    batch_seed: int
    sample_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.samples, np.ndarray) or self.samples.ndim != 2:
            raise TypeError("adaptive samples must be a two-dimensional ndarray.")
        if self.samples.shape[0] <= 0 or not np.all(np.isfinite(self.samples)):
            raise ValueError("adaptive samples must be nonempty and finite.")
        if self.distribution_role != "target":
            raise ValueError("adaptive target sampler returned the wrong role.")
        for name in ("adaptive_stage", "replication_index", "batch_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if not isinstance(self.sample_fingerprint, str) or not self.sample_fingerprint:
            raise ValueError("sample_fingerprint must be a nonempty string.")


@runtime_checkable
class AdaptiveTargetSampleProvider(Protocol):
    """Sample-only boundary for deterministic adaptive target batches."""

    capabilities: frozenset[ProviderCapability]

    def sample_batch(
        self,
        *,
        sample_size: int,
        adaptive_stage: int,
        replication_index: int,
    ) -> AdaptiveEvidenceSampleBatch:
        """Return one deterministic independent target sample batch."""


def _record(item, *, policy: str, applied_state: str, reason: str) -> dict[str, object]:
    return {
        **asdict(item),
        "policy_identifier": policy,
        "empirical_confidence_state": item.state,
        "applied_state": applied_state,
        "actionable": applied_state == "unresolved",
        "configured_threshold": item.local_loss_tolerance,
        "decision_reason": reason,
        "hidden_information_used": False,
        "true_local_loss": None,
        "structural_certificate_state": "pending_structural_analysis",
        "structural_decision_reason": None,
        "structurally_actionable_source": None,
        "top_unresolved_chain_component": None,
    }


@dataclass(frozen=True)
class ConfidenceCertificateLocalEvidenceProvider:
    """Preserve the existing conservative bootstrap-certificate decision."""

    policy_identifier: str = "confidence_certificate"
    capabilities = frozenset({ProviderCapability.EMPIRICAL_SAMPLES})

    def apply(
        self,
        empirical_state: EmpiricalUnresolvedState,
        **_: object,
    ) -> LocalEvidenceApplication:
        decisions = tuple(
            _record(
                item,
                policy=self.policy_identifier,
                applied_state=item.state,
                reason={
                    "unresolved": "ci_above_threshold",
                    "resolved": "ci_below_threshold",
                    "uncertain": "ci_crosses_threshold",
                }[item.state],
            )
            for item in empirical_state.nodes
        )
        return LocalEvidenceApplication(empirical_state, decisions)


@dataclass(frozen=True)
class AdaptiveConfidenceCertificateLocalEvidenceProvider:
    """Select the additive adaptive version of the confidence certificate."""

    policy_identifier: str = "adaptive_confidence_certificate"
    capabilities = frozenset({ProviderCapability.EMPIRICAL_SAMPLES})

    def apply(
        self,
        empirical_state: EmpiricalUnresolvedState,
        **_: object,
    ) -> LocalEvidenceApplication:
        decisions = tuple(
            _record(
                item,
                policy=self.policy_identifier,
                applied_state=item.state,
                reason={
                    "unresolved": "ci_above_threshold",
                    "resolved": "ci_below_threshold",
                    "uncertain": "ci_crosses_threshold",
                }[item.state],
            )
            for item in empirical_state.nodes
        )
        return LocalEvidenceApplication(empirical_state, decisions)


@dataclass(frozen=True)
class PointEstimateLocalEvidenceProvider:
    """Use point local loss for the binary gate while retaining its CI."""

    policy_identifier: str = "point_estimate_diagnostic"
    capabilities = frozenset({ProviderCapability.EMPIRICAL_SAMPLES})

    def apply(
        self,
        empirical_state: EmpiricalUnresolvedState,
        **_: object,
    ) -> LocalEvidenceApplication:
        nodes = []
        decisions = []
        for item in empirical_state.nodes:
            state = (
                "unresolved"
                if item.point_local_loss > item.local_loss_tolerance
                else "resolved"
            )
            nodes.append(replace(item, state=state))
            decisions.append(
                _record(
                    item,
                    policy=self.policy_identifier,
                    applied_state=state,
                    reason="point_estimate_above_threshold"
                    if state == "unresolved"
                    else "point_estimate_below_threshold",
                )
            )
        return LocalEvidenceApplication(
            EmpiricalUnresolvedState(tuple(nodes)), tuple(decisions)
        )
