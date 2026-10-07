"""Exact small-component MinMaxC exploration baseline."""

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral, Real

import networkx as nx
import numpy as np

from matching.probes import ShiftProbe
from matching.scheduler import AlgorithmStateView, BudgetView
from matching.unresolved_state import UnresolvedStateKnowledge


class ExactMinMaxCSizeLimitError(ValueError):
    """Raised when exact subset enumeration exceeds its configured limit."""


def _positive_integer(name: str, value: object) -> int:
    """Validate and return one positive Python integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _finite_nonzero_real(name: str, value: object) -> float:
    """Validate and return one finite, nonzero real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    if normalized == 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return normalized


def _largest_residual_component_size(
    graph: nx.Graph,
    removed: tuple[int, ...] | frozenset[int],
) -> int:
    """Return the MinMaxC objective after removing ``removed`` nodes."""
    residual = graph.copy()
    residual.remove_nodes_from(removed)
    return max(
        (len(component) for component in nx.connected_components(residual)),
        default=0,
    )


def exact_minmaxc_targets(
    component_graph: nx.Graph,
    *,
    max_targets: int,
    max_component_size: int,
) -> frozenset[int]:
    """Solve the bounded MinMaxC objective by exhaustive subset enumeration.

    Ties are resolved by smaller target cardinality and then the
    lexicographically smallest sorted node tuple.
    """
    max_targets = _positive_integer("max_targets", max_targets)
    max_component_size = _positive_integer(
        "max_component_size",
        max_component_size,
    )
    if not isinstance(component_graph, nx.Graph) or component_graph.is_directed():
        raise TypeError("component_graph must be an undirected networkx.Graph.")
    if component_graph.is_multigraph():
        raise TypeError("component_graph must be a simple networkx.Graph.")
    if component_graph.number_of_nodes() == 0:
        raise ValueError("component_graph must not be empty.")
    if any(
        isinstance(node, bool) or not isinstance(node, Integral)
        for node in component_graph.nodes
    ):
        raise TypeError("component_graph nodes must be integer indices.")
    nodes = tuple(sorted(int(node) for node in component_graph.nodes))
    if any(node < 0 for node in nodes):
        raise ValueError("component_graph nodes must be nonnegative.")
    if len(nodes) > max_component_size:
        raise ExactMinMaxCSizeLimitError(
            f"component size {len(nodes)} exceeds exact limit {max_component_size}."
        )
    if not nx.is_connected(component_graph):
        raise ValueError("component_graph must be connected.")
    if not nx.is_chordal(component_graph):
        raise ValueError("component_graph must be chordal.")

    best_key: tuple[int, int, tuple[int, ...]] | None = None
    for target_count in range(min(max_targets, len(nodes)) + 1):
        for targets in combinations(nodes, target_count):
            key = (
                _largest_residual_component_size(component_graph, targets),
                target_count,
                targets,
            )
            if best_key is None or key < best_key:
                best_key = key

    if best_key is None or not best_key[2]:
        raise RuntimeError("exact MinMaxC failed to find a nonempty target set.")
    return frozenset(best_key[2])


def _selected_component_graph(
    knowledge: UnresolvedStateKnowledge,
    component: frozenset[int],
) -> nx.Graph:
    """Return the selected fully undirected public chain component."""
    induced = knowledge.unresolved_graph.induced_subgraph(component)
    if induced.directed_edges:
        raise ValueError("selected top chain component must be fully undirected.")
    graph = nx.Graph()
    graph.add_nodes_from(induced.nodes)
    graph.add_edges_from(induced.undirected_edges)
    return graph


@dataclass(frozen=True)
class ExactMinMaxCExplorationPolicy:
    """Apply exact MinMaxC to one deterministically selected top component."""

    max_probe_targets: int
    shift_magnitude: float
    max_component_size: int

    def __post_init__(self) -> None:
        """Validate the exact-enumeration and probe-design controls."""
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
        object.__setattr__(
            self,
            "max_component_size",
            _positive_integer("max_component_size", self.max_component_size),
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
        """Return the deterministic exact MinMaxC probe when exploration is legal."""
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
        targets = exact_minmaxc_targets(
            component_graph,
            max_targets=self.max_probe_targets,
            max_component_size=self.max_component_size,
        )
        score = _largest_residual_component_size(
            component_graph,
            targets,
        )

        increment = np.zeros(n_nodes, dtype=float)
        increment[list(targets)] = self.shift_magnitude
        component_label = ",".join(str(node) for node in sorted(component))
        target_label = ",".join(str(node) for node in sorted(targets))
        return ShiftProbe(
            shift_increment=increment,
            targets=targets,
            reason=(
                "exact-minmaxc:"
                f"component={component_label};targets={target_label};score={score}"
            ),
        )
