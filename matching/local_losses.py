"""Configurable public local Gaussian losses and mismatch localization."""

from dataclasses import dataclass, field
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.pdag import PartiallyDirectedGraph
from graphs.problem_gen import PublicProblemView
from metrics.conditional import conditional_gaussian_kl_target_to_candidate
from metrics.marginal import marginal_gaussian_kl_target_to_candidate
from metrics.pairwise import pairwise_gaussian_kl_target_to_candidate

from .types import InterventionBackground, ParameterCoordinate


@runtime_checkable
class CandidateMomentsProvider(Protocol):
    """Provide candidate moments without defining how they are obtained."""

    def moments(
        self,
        candidate: ShiftScaleIntervention,
        public_problem: PublicProblemView,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return candidate mean and covariance."""
        ...


@dataclass(frozen=True)
class LocalLossWeights:
    """Configurable marginal, pairwise, and conditional score weights."""

    marginal: float = 1.0
    pairwise: float = 1.0
    conditional: float = 1.0

    def __post_init__(self) -> None:
        """Require finite nonnegative weights and one enabled component."""
        normalized: list[float] = []
        for name in ("marginal", "pairwise", "conditional"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} weight must be a real number.")
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} weight must be finite and nonnegative.")
            object.__setattr__(self, name, numeric)
            normalized.append(numeric)
        if not any(value > 0.0 for value in normalized):
            raise ValueError("at least one local-loss weight must be positive.")


@dataclass(frozen=True)
class PublicPairwiseNeighborhood:
    """Select neighbors from only the current public PDAG.

    The default neighborhood contains parents, children, and undirected
    neighbors. With ``assign_directed_to_child=True``, a directed pair is
    scored at its child endpoint, so a node's directed children remain part of
    its public neighborhood metadata but their discrepancy is assigned to the
    child rather than duplicated at the parent.
    """

    include_parents: bool = True
    include_children: bool = True
    include_undirected: bool = True
    assign_directed_to_child: bool = True

    def __post_init__(self) -> None:
        for name in (
            "include_parents",
            "include_children",
            "include_undirected",
            "assign_directed_to_child",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")


@dataclass(frozen=True)
class NodeLocalLoss:
    """One node's normalized composite and available component values."""

    node: int
    composite: float
    marginal: float | None
    pairwise: float | None
    conditional: float | None
    used_terms: tuple[str, ...]
    omitted_terms: tuple[str, ...]
    public_pairwise_neighborhood: tuple[int, ...]
    assigned_pairwise_neighbors: tuple[int, ...]


_CacheKey = tuple[
    int,
    str,
    frozenset[tuple[int, int]],
    frozenset[tuple[int, int]],
]


@dataclass
class CompositeGaussianLocalLoss:
    """Compute normalized local scores using only supplied public graph context."""

    moments_provider: CandidateMomentsProvider
    weights: LocalLossWeights = field(default_factory=LocalLossWeights)
    pairwise_neighborhood: PublicPairwiseNeighborhood = field(
        default_factory=PublicPairwiseNeighborhood
    )
    _cache: dict[_CacheKey, tuple[NodeLocalLoss, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _evaluation_count: int = field(default=0, init=False, repr=False)
    _invalidation_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the injected moment boundary and weight configuration."""
        if not isinstance(self.moments_provider, CandidateMomentsProvider):
            raise TypeError("moments_provider must implement CandidateMomentsProvider.")
        if not isinstance(self.weights, LocalLossWeights):
            raise TypeError("weights must be LocalLossWeights.")
        if not isinstance(self.pairwise_neighborhood, PublicPairwiseNeighborhood):
            raise TypeError("pairwise_neighborhood must be PublicPairwiseNeighborhood.")

    @property
    def evaluation_count(self) -> int:
        """Return the number of uncached candidate-moment evaluations."""
        return self._evaluation_count

    @property
    def invalidation_count(self) -> int:
        """Return the number of accepted-background invalidations."""
        return self._invalidation_count

    def invalidate_background(
        self,
        previous: ShiftScaleIntervention,
        current: ShiftScaleIntervention,
    ) -> None:
        """Discard all candidate and ranking caches after accepted state changes."""
        if not isinstance(previous, ShiftScaleIntervention) or not isinstance(
            current,
            ShiftScaleIntervention,
        ):
            raise TypeError("backgrounds must be ShiftScaleIntervention objects.")
        if InterventionBackground.from_intervention(previous).identifier == (
            InterventionBackground.from_intervention(current).identifier
        ):
            return
        self._cache.clear()
        self._invalidation_count += 1

    def node_losses(
        self,
        public_problem: PublicProblemView,
        candidate: ShiftScaleIntervention,
        graph: PartiallyDirectedGraph | None = None,
    ) -> tuple[NodeLocalLoss, ...]:
        """Return one normalized composite score per node.

        Conditional terms use only directed parents in ``graph``. Pairwise terms
        use certified parents plus unresolved undirected neighbors, assigning an
        oriented pair to its child endpoint instead of contaminating the parent's
        local score with a child's mismatch. Without graph context, pairwise terms
        use all other public variables and conditional terms are unavailable.
        """
        if not isinstance(public_problem, PublicProblemView):
            raise TypeError("public_problem must be a PublicProblemView.")
        if not isinstance(candidate, ShiftScaleIntervention):
            raise TypeError("candidate must be a ShiftScaleIntervention.")
        if graph is not None:
            if not isinstance(graph, PartiallyDirectedGraph):
                raise TypeError("graph must be a PartiallyDirectedGraph or None.")
            if graph.nodes != frozenset(range(public_problem.n_nodes)):
                raise ValueError("graph nodes must match public problem indices.")
        background = InterventionBackground.from_intervention(candidate)
        key: _CacheKey = (
            id(public_problem),
            background.identifier,
            graph.directed_edges if graph is not None else frozenset(),
            graph.undirected_edges if graph is not None else frozenset(),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        candidate_mean, candidate_covariance = self.moments_provider.moments(
            candidate,
            public_problem,
        )
        self._evaluation_count += 1
        results = tuple(
            self._node_loss(
                node,
                public_problem,
                candidate_mean,
                candidate_covariance,
                graph,
            )
            for node in range(public_problem.n_nodes)
        )
        self._cache[key] = results
        return results

    def coordinate_loss(
        self,
        coordinate: ParameterCoordinate,
        public_problem: PublicProblemView,
        candidate: ShiftScaleIntervention,
        graph: PartiallyDirectedGraph | None = None,
    ) -> float:
        """Return the local score associated with one parameter coordinate."""
        if not isinstance(coordinate, ParameterCoordinate):
            raise TypeError("coordinate must be a ParameterCoordinate.")
        return self.node_losses(public_problem, candidate, graph)[
            coordinate.node
        ].composite

    def _node_loss(
        self,
        node: int,
        public_problem: PublicProblemView,
        candidate_mean: np.ndarray,
        candidate_covariance: np.ndarray,
        graph: PartiallyDirectedGraph | None,
    ) -> NodeLocalLoss:
        """Compute and renormalize the components available for one node."""
        target_mean = public_problem.target_mean
        target_covariance = public_problem.target_covariance
        marginal = marginal_gaussian_kl_target_to_candidate(
            target_mean,
            target_covariance,
            candidate_mean,
            candidate_covariance,
            node,
        )
        relevant = (
            self._public_pairwise_neighborhood(node, graph)
            if graph is not None
            else frozenset(range(public_problem.n_nodes)) - {node}
        )
        assigned = (
            self._assigned_pairwise_neighbors(node, graph)
            if graph is not None
            else relevant
        )
        pairwise_values = tuple(
            pairwise_gaussian_kl_target_to_candidate(
                target_mean,
                target_covariance,
                candidate_mean,
                candidate_covariance,
                node,
                other,
            )
            for other in sorted(assigned)
        )
        pairwise = float(np.mean(pairwise_values)) if pairwise_values else None
        parents = graph.parents_of(node) if graph is not None else frozenset()
        conditional = (
            conditional_gaussian_kl_target_to_candidate(
                target_mean,
                target_covariance,
                candidate_mean,
                candidate_covariance,
                node,
                sorted(parents),
            )
            if parents
            else None
        )
        weighted_components = [
            (self.weights.marginal, marginal),
            (self.weights.pairwise, pairwise),
            (self.weights.conditional, conditional),
        ]
        available = [
            (name, weight, value)
            for name, (weight, value) in zip(
                ("marginal", "pairwise", "conditional"),
                weighted_components,
                strict=True,
            )
            if weight > 0.0 and value is not None
        ]
        normalization = sum(weight for _, weight, _ in available)
        composite = (
            sum(weight * value for _, weight, value in available) / normalization
            if normalization > 0.0
            else 0.0
        )
        used_terms = tuple(name for name, _, _ in available)
        omitted_terms = tuple(
            name
            for name, (weight, value) in zip(
                ("marginal", "pairwise", "conditional"),
                weighted_components,
                strict=True,
            )
            if weight > 0.0 and value is None
        )
        return NodeLocalLoss(
            node,
            composite,
            marginal,
            pairwise,
            conditional,
            used_terms,
            omitted_terms,
            tuple(sorted(relevant)),
            tuple(sorted(assigned)),
        )

    def _public_pairwise_neighborhood(
        self,
        node: int,
        graph: PartiallyDirectedGraph,
    ) -> frozenset[int]:
        """Return configured neighbors from only the public graph."""
        neighbors: set[int] = set()
        if self.pairwise_neighborhood.include_parents:
            neighbors.update(graph.parents_of(node))
        if self.pairwise_neighborhood.include_children:
            neighbors.update(graph.children_of(node))
        if self.pairwise_neighborhood.include_undirected:
            neighbors.update(graph.undirected_neighbors_of(node))
        return frozenset(neighbors)

    def _assigned_pairwise_neighbors(
        self,
        node: int,
        graph: PartiallyDirectedGraph,
    ) -> frozenset[int]:
        """Assign each directed-pair discrepancy to its child by default."""
        neighborhood = self._public_pairwise_neighborhood(node, graph)
        if not self.pairwise_neighborhood.assign_directed_to_child:
            return neighborhood
        return frozenset(
            neighbor
            for neighbor in neighborhood
            if not graph.has_directed_edge(node, neighbor)
        )


@dataclass
class CompositeGaussianMismatchProvider:
    """Mark nodes unresolved when their composite local KL exceeds a threshold."""

    local_loss: CompositeGaussianLocalLoss
    unresolved_threshold: float

    def __post_init__(self) -> None:
        """Validate the local-loss dependency and unresolved threshold."""
        if not isinstance(self.local_loss, CompositeGaussianLocalLoss):
            raise TypeError("local_loss must be CompositeGaussianLocalLoss.")
        if isinstance(self.unresolved_threshold, bool) or not isinstance(
            self.unresolved_threshold,
            Real,
        ):
            raise TypeError("unresolved_threshold must be a real number.")
        threshold = float(self.unresolved_threshold)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("unresolved_threshold must be finite and nonnegative.")
        self.unresolved_threshold = threshold

    def unresolved_nodes(
        self,
        public_problem: PublicProblemView,
        candidate: ShiftScaleIntervention,
    ) -> frozenset[int]:
        """Localize mismatch without graph-specific conditional information."""
        return self.unresolved_nodes_for_graph(public_problem, candidate, None)

    def unresolved_nodes_for_graph(
        self,
        public_problem: PublicProblemView,
        candidate: ShiftScaleIntervention,
        graph: PartiallyDirectedGraph | None,
    ) -> frozenset[int]:
        """Localize mismatch using only the supplied certified public graph."""
        return frozenset(
            loss.node
            for loss in self.local_loss.node_losses(public_problem, candidate, graph)
            if loss.composite > self.unresolved_threshold
        )
