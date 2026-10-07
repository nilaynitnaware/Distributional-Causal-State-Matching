"""Oracle local-actionability label override for diagnostic ablations only."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.pdag import PartiallyDirectedGraph
from graphs.problem_gen import PublicProblemView
from matching.capabilities import ProviderCapability
from matching.coordinate_search import _FixedMomentsProvider
from matching.empirical_unresolved import EmpiricalUnresolvedState
from matching.local_evidence import LocalEvidenceApplication, _record
from matching.local_losses import (
    CompositeGaussianLocalLoss,
    LocalLossWeights,
    PublicPairwiseNeighborhood,
)


@dataclass
class OracleLocalActionabilityProvider:
    """Override only local binary labels using hidden population local loss."""

    hidden_moments: Callable[[ShiftScaleIntervention], tuple[np.ndarray, np.ndarray]]
    hidden_target_mean: np.ndarray
    hidden_target_covariance: np.ndarray
    policy_identifier: str = "oracle_local_actionability"
    capabilities = frozenset({ProviderCapability.ORACLE})

    def __post_init__(self) -> None:
        if not callable(self.hidden_moments):
            raise TypeError("hidden_moments must be callable.")
        self.__moments = self.hidden_moments
        self.__target_mean = np.asarray(self.hidden_target_mean, dtype=float).copy()
        self.__target_covariance = np.asarray(
            self.hidden_target_covariance, dtype=float
        ).copy()
        del self.hidden_moments, self.hidden_target_mean, self.hidden_target_covariance

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
        mean, covariance = self.__moments(intervention)
        true_problem = replace(
            problem,
            target_mean=self.__target_mean,
            target_covariance=self.__target_covariance,
        )
        losses = CompositeGaussianLocalLoss(
            _FixedMomentsProvider(mean, covariance), weights, neighborhood
        ).node_losses(true_problem, intervention, graph)
        nodes = []
        decisions = []
        for item, loss in zip(empirical_state.nodes, losses, strict=True):
            state = (
                "unresolved"
                if loss.composite > item.local_loss_tolerance
                else "resolved"
            )
            nodes.append(replace(item, state=state))
            record = _record(
                item,
                policy=self.policy_identifier,
                applied_state=state,
                reason="oracle_true_loss_above_threshold"
                if state == "unresolved"
                else "oracle_true_loss_below_threshold",
            )
            record.update(
                {
                    "hidden_information_used": True,
                    "hidden_information_scope": "binary_local_actionability_only",
                    "true_local_loss": loss.composite,
                    "true_local_loss_used_terms": loss.used_terms,
                    "true_local_loss_omitted_terms": loss.omitted_terms,
                }
            )
            decisions.append(record)
        return LocalEvidenceApplication(
            EmpiricalUnresolvedState(tuple(nodes)), tuple(decisions)
        )
