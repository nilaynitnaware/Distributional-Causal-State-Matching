"""Three-way unresolved-node state from empirical local-loss uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.pdag import PartiallyDirectedGraph
from graphs.problem_gen import PublicProblemView

from .coordinate_search import _FixedMomentsProvider
from .empirical_moments import EmpiricalMomentsProvider
from .local_losses import (
    CompositeGaussianLocalLoss,
    LocalLossWeights,
    PublicPairwiseNeighborhood,
)

LocalResolution = Literal["resolved", "unresolved", "uncertain"]


@dataclass(frozen=True)
class EmpiricalNodeResolution:
    """One node's local-loss interval and three-way state."""

    node: int
    state: LocalResolution
    point_local_loss: float
    confidence_interval: tuple[float, float]
    local_loss_tolerance: float
    bootstrap_replicates: int
    bootstrap_seed: int
    used_terms: tuple[str, ...]
    omitted_terms: tuple[str, ...]


@dataclass(frozen=True)
class EmpiricalUnresolvedState:
    """All node states and convenience index sets."""

    nodes: tuple[EmpiricalNodeResolution, ...]

    @property
    def resolved_nodes(self) -> frozenset[int]:
        return frozenset(item.node for item in self.nodes if item.state == "resolved")

    @property
    def unresolved_nodes(self) -> frozenset[int]:
        return frozenset(item.node for item in self.nodes if item.state == "unresolved")

    @property
    def uncertain_nodes(self) -> frozenset[int]:
        return frozenset(item.node for item in self.nodes if item.state == "uncertain")


def classify_empirical_unresolved_nodes(
    samples: np.ndarray,
    intervention: ShiftScaleIntervention,
    problem: PublicProblemView,
    graph: PartiallyDirectedGraph,
    moments_provider: EmpiricalMomentsProvider,
    weights: LocalLossWeights,
    neighborhood: PublicPairwiseNeighborhood,
    *,
    local_loss_tolerance: float,
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> EmpiricalUnresolvedState:
    """Classify nodes using bootstrap intervals around public local loss."""
    if local_loss_tolerance < 0.0:
        raise ValueError("local_loss_tolerance must be nonnegative.")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1).")
    point_estimate = moments_provider.estimate(samples)
    point_provider = _FixedMomentsProvider(
        point_estimate.mean,
        point_estimate.regularized_covariance,
    )
    point_losses = CompositeGaussianLocalLoss(
        point_provider,
        weights,
        neighborhood,
    ).node_losses(problem, intervention, graph)
    bootstrap_losses = np.empty((bootstrap_replicates, problem.n_nodes))
    rng = np.random.default_rng(bootstrap_seed)
    for replicate in range(bootstrap_replicates):
        indices = rng.integers(0, samples.shape[0], samples.shape[0])
        estimate = moments_provider.estimate(samples[indices])
        provider = _FixedMomentsProvider(
            estimate.mean,
            estimate.regularized_covariance,
        )
        losses = CompositeGaussianLocalLoss(
            provider,
            weights,
            neighborhood,
        ).node_losses(problem, intervention, graph)
        bootstrap_losses[replicate] = [loss.composite for loss in losses]
    tail = (1.0 - confidence_level) / 2.0
    records: list[EmpiricalNodeResolution] = []
    for node, point_loss in enumerate(point_losses):
        interval = (
            float(np.quantile(bootstrap_losses[:, node], tail)),
            float(np.quantile(bootstrap_losses[:, node], 1.0 - tail)),
        )
        if interval[0] > local_loss_tolerance:
            state: LocalResolution = "unresolved"
        elif interval[1] <= local_loss_tolerance:
            state = "resolved"
        else:
            state = "uncertain"
        records.append(
            EmpiricalNodeResolution(
                node=node,
                state=state,
                point_local_loss=point_loss.composite,
                confidence_interval=interval,
                local_loss_tolerance=local_loss_tolerance,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
                used_terms=point_loss.used_terms,
                omitted_terms=point_loss.omitted_terms,
            )
        )
    return EmpiricalUnresolvedState(tuple(records))
