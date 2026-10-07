"""Maximal-clique-tree exploration for unresolved chordal components."""

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral

import networkx as nx
import numpy as np

from matching.probes import ShiftProbe
from matching.scheduler import AlgorithmStateView, BudgetView
from matching.unresolved_state import UnresolvedStateKnowledge

from .exact_minmaxc import (
    _finite_nonzero_real,
    _largest_residual_component_size,
    _positive_integer,
    _selected_component_graph,
)

Clique = tuple[int, ...]


def _validated_chordal_component(
    component_graph: nx.Graph,
) -> tuple[nx.Graph, tuple[int, ...]]:
    """Return a canonical copy of one simple connected chordal graph."""
    if not isinstance(component_graph, nx.Graph) or component_graph.is_directed():
        raise TypeError("component_graph must be an undirected networkx.Graph.")
    if component_graph.is_multigraph():
        raise TypeError("component_graph must be a simple networkx.Graph.")
    if component_graph.number_of_nodes() == 0:
        raise ValueError("component_graph must not be empty.")
    if nx.number_of_selfloops(component_graph):
        raise ValueError("component_graph must not contain self-loops.")
    if any(
        isinstance(node, bool) or not isinstance(node, Integral)
        for node in component_graph.nodes
    ):
        raise TypeError("component_graph nodes must be integer indices.")

    nodes = tuple(sorted(int(node) for node in component_graph.nodes))
    if any(node < 0 for node in nodes):
        raise ValueError("component_graph nodes must be nonnegative.")

    canonical = nx.Graph()
    canonical.add_nodes_from(nodes)
    canonical.add_edges_from(
        sorted(
            tuple(sorted((int(node_a), int(node_b))))
            for node_a, node_b in component_graph.edges
        )
    )
    if not nx.is_connected(canonical):
        raise ValueError("component_graph must be connected.")
    if not nx.is_chordal(canonical):
        raise ValueError("component_graph must be chordal.")
    return canonical, nodes


def _maximal_cliques(component_graph: nx.Graph) -> tuple[Clique, ...]:
    """Return canonical maximal cliques of a validated chordal graph."""
    cliques = tuple(
        sorted(
            tuple(sorted(int(node) for node in clique))
            for clique in nx.chordal_graph_cliques(component_graph)
        )
    )
    if not cliques:
        raise RuntimeError("a nonempty chordal graph must have a maximal clique.")
    return cliques


def _clique_tree_from_validated(component_graph: nx.Graph) -> nx.Graph:
    """Build a deterministic maximum-weight maximal-clique tree."""
    cliques = _maximal_cliques(component_graph)
    clique_tree = nx.Graph()
    clique_tree.add_nodes_from(cliques)
    if len(cliques) == 1:
        return clique_tree

    weighted_edges: list[tuple[int, Clique, Clique, Clique]] = []
    for clique_a, clique_b in combinations(cliques, 2):
        separator = tuple(sorted(set(clique_a) & set(clique_b)))
        if separator:
            weighted_edges.append((-len(separator), clique_a, clique_b, separator))

    parent = {clique: clique for clique in cliques}

    def find(clique: Clique) -> Clique:
        root = clique
        while parent[root] != root:
            root = parent[root]
        while parent[clique] != clique:
            next_clique = parent[clique]
            parent[clique] = root
            clique = next_clique
        return root

    for negative_weight, clique_a, clique_b, separator in sorted(weighted_edges):
        root_a = find(clique_a)
        root_b = find(clique_b)
        if root_a == root_b:
            continue
        parent[root_b] = root_a
        clique_tree.add_edge(
            clique_a,
            clique_b,
            weight=-negative_weight,
            separator=separator,
        )
        if clique_tree.number_of_edges() == len(cliques) - 1:
            break

    if not nx.is_tree(clique_tree):
        raise RuntimeError("maximal-clique intersection graph did not form a tree.")
    for node in component_graph.nodes:
        containing = tuple(clique for clique in cliques if node in clique)
        if len(containing) > 1 and not nx.is_connected(
            clique_tree.subgraph(containing)
        ):
            raise RuntimeError("constructed clique tree violates running intersection.")
    return clique_tree


def deterministic_clique_tree(component_graph: nx.Graph) -> nx.Graph:
    """Return a deterministic maximum-weight clique tree.

    Tree nodes are sorted tuples representing maximal cliques. Each edge stores
    its intersection as ``separator`` and its cardinality as ``weight``.
    """
    canonical, _ = _validated_chordal_component(component_graph)
    return _clique_tree_from_validated(canonical)


@dataclass(frozen=True)
class _CliqueTreeChoice:
    targets: frozenset[int]
    central_clique: Clique
    central_clique_score: int
    residual_score: int
    maximal_clique_count: int
    selection_rule: str


def _choose_clique_tree_targets(
    component_graph: nx.Graph,
    *,
    max_targets: int,
    rng: np.random.Generator,
) -> _CliqueTreeChoice:
    """Select one paper-style central maximal clique, then enforce width."""
    max_targets = _positive_integer("max_targets", max_targets)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    canonical, nodes = _validated_chordal_component(component_graph)
    clique_tree = _clique_tree_from_validated(canonical)
    cliques = tuple(sorted(clique_tree.nodes))

    central_clique = min(
        cliques,
        key=lambda clique: (
            _largest_residual_component_size(canonical, clique),
            len(clique),
            clique,
        ),
    )
    central_score = _largest_residual_component_size(
        canonical,
        central_clique,
    )

    if len(nodes) <= max_targets:
        targets = frozenset(nodes)
        selection_rule = "whole-component"
    elif len(central_clique) <= max_targets:
        targets = frozenset(central_clique)
        selection_rule = "full-maximal-clique"
    else:
        selected = rng.choice(
            np.asarray(central_clique, dtype=int),
            size=max_targets,
            replace=False,
        )
        targets = frozenset(int(node) for node in np.atleast_1d(selected))
        selection_rule = "rng-subset-of-maximal-clique"

    return _CliqueTreeChoice(
        targets=targets,
        central_clique=central_clique,
        central_clique_score=central_score,
        residual_score=_largest_residual_component_size(canonical, targets),
        maximal_clique_count=len(cliques),
        selection_rule=selection_rule,
    )


def clique_tree_targets(
    component_graph: nx.Graph,
    *,
    max_targets: int,
    rng: np.random.Generator,
) -> frozenset[int]:
    """Return bounded CliqueTree targets for one chordal component.

    Candidate objects are maximal cliques, not separator sets. Full maximal
    cliques are scored by the largest connected component remaining after
    removal. If the winning clique is wider than ``max_targets``, the returned
    set is a reproducible random subset drawn only from that clique.
    """
    return _choose_clique_tree_targets(
        component_graph,
        max_targets=max_targets,
        rng=rng,
    ).targets


@dataclass(frozen=True)
class CliqueTreeProbePolicy:
    """Apply maximal-clique-tree exploration to one selected top component."""

    max_probe_targets: int
    shift_magnitude: float

    def __post_init__(self) -> None:
        """Validate probe width and additive shift magnitude."""
        object.__setattr__(
            self,
            "max_probe_targets",
            _positive_integer("max_probe_targets", self.max_probe_targets),
        )
        object.__setattr__(
            self,
            "shift_magnitude",
            _finite_nonzero_real("shift_magnitude", self.shift_magnitude),
        )

    def propose_probe(
        self,
        public_state: AlgorithmStateView,
        knowledge: UnresolvedStateKnowledge,
        budget: BudgetView,
        rng: np.random.Generator,
        *,
        allow_when_actionable: bool = False,
    ) -> ShiftProbe | None:
        """Return a bounded CliqueTree probe when exploration is legal."""
        if not isinstance(public_state, AlgorithmStateView):
            raise TypeError("public_state must be an AlgorithmStateView.")
        if not isinstance(knowledge, UnresolvedStateKnowledge):
            raise TypeError("knowledge must be an UnresolvedStateKnowledge.")
        if not isinstance(budget, BudgetView):
            raise TypeError("budget must be a BudgetView.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        if not isinstance(allow_when_actionable, bool):
            raise TypeError("allow_when_actionable must be a boolean.")

        n_nodes = public_state.problem_view.n_nodes
        if any(node >= n_nodes for node in knowledge.unresolved_nodes):
            raise ValueError("knowledge nodes must fit the public problem dimension.")
        if budget.remaining_experiment_trials == 0:
            return None
        if not knowledge.unresolved_nodes:
            return None
        if knowledge.certified_actionable_sources and not allow_when_actionable:
            return None
        if not knowledge.top_unresolved_chain_components:
            return None

        component = min(
            knowledge.top_unresolved_chain_components,
            key=lambda nodes: (-len(nodes), tuple(sorted(nodes))),
        )
        component_graph = _selected_component_graph(knowledge, component)
        choice = _choose_clique_tree_targets(
            component_graph,
            max_targets=self.max_probe_targets,
            rng=rng,
        )

        increment = np.zeros(n_nodes, dtype=float)
        increment[list(choice.targets)] = self.shift_magnitude
        component_label = ",".join(str(node) for node in sorted(component))
        clique_label = ",".join(str(node) for node in choice.central_clique)
        target_label = ",".join(str(node) for node in sorted(choice.targets))
        return ShiftProbe(
            shift_increment=increment,
            targets=choice.targets,
            reason=(
                "clique-tree:"
                f"component={component_label};"
                f"central-clique={clique_label};"
                f"targets={target_label};"
                f"clique-score={choice.central_clique_score};"
                f"score={choice.residual_score};"
                f"maximal-cliques={choice.maximal_clique_count};"
                f"selection={choice.selection_rule}"
            ),
        )
