"""Public mismatch and unresolved-graph state contracts."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Protocol, runtime_checkable

from graphs.interventions import ShiftScaleIntervention
from graphs.pdag import PartiallyDirectedGraph
from graphs.problem_gen import PublicProblemView


def _normalize_node_set(value: object, *, name: str) -> frozenset[int]:
    """Validate and defensively freeze one set of node indices."""
    if not isinstance(value, frozenset):
        raise TypeError(f"{name} must be a frozenset of node indices.")
    if any(isinstance(node, bool) or not isinstance(node, Integral) for node in value):
        raise TypeError(f"{name} must contain integer node indices.")
    normalized = frozenset(int(node) for node in value)
    if any(node < 0 for node in normalized):
        raise ValueError(f"{name} must contain nonnegative node indices.")
    return normalized


@runtime_checkable
class MismatchProvider(Protocol):
    """Classify unresolved nodes without imposing one loss-specific rule."""

    def unresolved_nodes(
        self,
        public_problem: PublicProblemView,
        candidate: ShiftScaleIntervention,
    ) -> frozenset[int]:
        """Return the nodes unresolved under ``candidate``."""
        ...


@dataclass
class ScriptedMismatchProvider:
    """Return a deterministic sequence of unresolved-node sets for tests.

    Each call advances to the next entry. Once the sequence is exhausted, the
    final entry is retained so repeated state queries remain well-defined.
    """

    script: tuple[frozenset[int], ...]
    _call_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and freeze the scripted public mismatch states."""
        if not isinstance(self.script, tuple):
            raise TypeError("script must be a tuple of unresolved-node sets.")
        if not self.script:
            raise ValueError("script must contain at least one unresolved-node set.")
        normalized = tuple(
            _normalize_node_set(nodes, name=f"script entry {position}")
            for position, nodes in enumerate(self.script)
        )
        self.script = normalized

    @property
    def call_count(self) -> int:
        """Return the number of successful state queries."""
        return self._call_count

    def unresolved_nodes(
        self,
        public_problem: PublicProblemView,
        candidate: ShiftScaleIntervention,
    ) -> frozenset[int]:
        """Return the next scripted mismatch state after public validation."""
        if not isinstance(public_problem, PublicProblemView):
            raise TypeError("public_problem must be a PublicProblemView.")
        if not isinstance(candidate, ShiftScaleIntervention):
            raise TypeError("candidate must be a ShiftScaleIntervention.")
        if candidate.shift.shape != (public_problem.n_nodes,):
            raise ValueError("candidate dimension must agree with the public problem.")

        position = min(self._call_count, len(self.script) - 1)
        unresolved = self.script[position]
        if any(node >= public_problem.n_nodes for node in unresolved):
            raise ValueError(
                "scripted unresolved nodes must fit the public problem dimension."
            )
        self._call_count += 1
        return frozenset(unresolved)


@dataclass(frozen=True)
class UnresolvedStateKnowledge:
    """One immutable public snapshot of unresolved graph actionability."""

    unresolved_graph: PartiallyDirectedGraph
    unresolved_nodes: frozenset[int]
    certified_actionable_sources: frozenset[int]
    top_unresolved_chain_components: tuple[frozenset[int], ...]
    needs_exploration: bool = field(init=False)

    def __post_init__(self) -> None:
        """Validate the snapshot and derive its exploration gate."""
        if not isinstance(self.unresolved_graph, PartiallyDirectedGraph):
            raise TypeError("unresolved_graph must be a PartiallyDirectedGraph.")
        unresolved = _normalize_node_set(
            self.unresolved_nodes,
            name="unresolved_nodes",
        )
        sources = _normalize_node_set(
            self.certified_actionable_sources,
            name="certified_actionable_sources",
        )
        if not sources <= unresolved:
            raise ValueError(
                "certified_actionable_sources must be a subset of unresolved_nodes."
            )
        if not isinstance(self.top_unresolved_chain_components, tuple):
            raise TypeError("top_unresolved_chain_components must be a tuple.")

        components = tuple(
            _normalize_node_set(component, name=f"top component {position}")
            for position, component in enumerate(self.top_unresolved_chain_components)
        )
        if self.unresolved_graph.nodes != unresolved:
            raise ValueError("unresolved_graph nodes must equal unresolved_nodes.")
        expected_sources = frozenset(
            node
            for node in self.unresolved_graph.nodes
            if not self.unresolved_graph.parents_of(node)
            and not self.unresolved_graph.undirected_neighbors_of(node)
        )
        if sources != expected_sources:
            raise ValueError(
                "certified_actionable_sources must match unresolved_graph."
            )
        if components != self.unresolved_graph.top_chain_components():
            raise ValueError(
                "top_unresolved_chain_components must match unresolved_graph."
            )
        if any(not component for component in components):
            raise ValueError("top unresolved chain components must not be empty.")
        covered: set[int] = set()
        for component in components:
            if not component <= unresolved:
                raise ValueError(
                    "top unresolved chain components must contain only "
                    "unresolved nodes."
                )
            if covered & component:
                raise ValueError(
                    "top unresolved chain components must be pairwise disjoint."
                )
            covered.update(component)
        if not sources <= covered:
            raise ValueError(
                "every certified actionable source must belong to a top "
                "unresolved chain component."
            )

        object.__setattr__(self, "unresolved_nodes", unresolved)
        object.__setattr__(self, "certified_actionable_sources", sources)
        object.__setattr__(
            self,
            "top_unresolved_chain_components",
            components,
        )
        object.__setattr__(
            self,
            "needs_exploration",
            bool(unresolved and not sources),
        )


def analyze_unresolved_state(
    graph: PartiallyDirectedGraph,
    unresolved_nodes: frozenset[int],
) -> UnresolvedStateKnowledge:
    """Derive source actionability from the PDAG induced by unresolved nodes.

    A node is certified only when it has neither a directed parent nor an
    unresolved undirected neighbor in the induced graph. This certificate
    identifies a source node only; it does not reveal a complete parent set.
    """
    if not isinstance(graph, PartiallyDirectedGraph):
        raise TypeError("graph must be a PartiallyDirectedGraph.")
    unresolved = _normalize_node_set(unresolved_nodes, name="unresolved_nodes")
    if not unresolved <= graph.nodes:
        raise ValueError("unresolved_nodes must be a subset of graph nodes.")

    induced = graph.induced_subgraph(unresolved)
    sources = frozenset(
        node
        for node in induced.nodes
        if not induced.parents_of(node) and not induced.undirected_neighbors_of(node)
    )
    return UnresolvedStateKnowledge(
        unresolved_graph=induced,
        unresolved_nodes=unresolved,
        certified_actionable_sources=sources,
        top_unresolved_chain_components=induced.top_chain_components(),
    )
