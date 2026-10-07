"""Bounded faithful Supermodular exploration over chordal components."""

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral, Real

import networkx as nx
import numpy as np

from matching.probes import ShiftProbe
from matching.scheduler import AlgorithmStateView, BudgetView
from matching.unresolved_state import UnresolvedStateKnowledge

from .clique_tree import _validated_chordal_component
from .exact_minmaxc import (
    _finite_nonzero_real,
    _positive_integer,
    _selected_component_graph,
)


class SupermodularPathLimitError(ValueError):
    """Raised when faithful simple-path precomputation exceeds its hard cap."""


def _saturation_tolerance(value: object) -> float:
    """Validate one nontrivial absolute tolerance for SATURATE."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("saturation_tolerance must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError("saturation_tolerance must be finite.")
    if not 0.0 < normalized < 1.0:
        raise ValueError("saturation_tolerance must satisfy 0 < value < 1.")
    return normalized


def _normalize_targets(
    targets: object,
    *,
    nodes: frozenset[int],
) -> frozenset[int]:
    """Validate and freeze one target set against the evaluator nodes."""
    if not isinstance(targets, frozenset):
        raise TypeError("targets must be a frozenset of node indices.")
    if any(
        isinstance(node, bool) or not isinstance(node, Integral) for node in targets
    ):
        raise TypeError("targets must contain integer node indices.")
    normalized = frozenset(int(node) for node in targets)
    if not normalized <= nodes:
        raise ValueError("targets must be a subset of the component nodes.")
    return normalized


@dataclass(frozen=True, init=False)
class BoundedPathSurrogate:
    """Precompute the faithful normalized simple-path surrogate within a cap.

    Self-pairs are synthesized with one zero-edge path. Trees use connected
    components directly, so they never enumerate their pairwise paths and do
    not consume the general-graph path cap.
    """

    nodes: tuple[int, ...]
    max_total_simple_paths: int
    total_simple_paths: int
    evaluation_mode: str
    _tree_neighbors: tuple[tuple[int, ...], ...] | None
    _pair_path_masks: tuple[tuple[int, int, tuple[int, ...]], ...]

    def __init__(
        self,
        component_graph: nx.Graph,
        *,
        max_total_simple_paths: int,
    ) -> None:
        """Validate the component and prepare one immutable evaluator."""
        path_limit = _positive_integer(
            "max_total_simple_paths",
            max_total_simple_paths,
        )
        canonical, nodes = _validated_chordal_component(component_graph)
        node_position = {node: position for position, node in enumerate(nodes)}

        if nx.is_tree(canonical):
            neighbors = tuple(
                tuple(
                    sorted(
                        node_position[neighbor]
                        for neighbor in canonical.neighbors(node)
                    )
                )
                for node in nodes
            )
            object.__setattr__(self, "nodes", nodes)
            object.__setattr__(self, "max_total_simple_paths", path_limit)
            object.__setattr__(
                self,
                "total_simple_paths",
                len(nodes) * (len(nodes) - 1) // 2,
            )
            object.__setattr__(self, "evaluation_mode", "tree-fast-path")
            object.__setattr__(self, "_tree_neighbors", neighbors)
            object.__setattr__(self, "_pair_path_masks", ())
            return

        pair_path_masks: list[tuple[int, int, tuple[int, ...]]] = []
        total_paths = 0
        for node_a, node_b in combinations(nodes, 2):
            masks: list[int] = []
            for path in nx.all_simple_paths(canonical, node_a, node_b):
                total_paths += 1
                if total_paths > path_limit:
                    raise SupermodularPathLimitError(
                        "simple-path precomputation exceeds configured limit "
                        f"{path_limit}."
                    )
                mask = 0
                for node in path:
                    mask |= 1 << node_position[int(node)]
                masks.append(mask)
            if not masks:
                raise RuntimeError(
                    "a connected component must have a path between every pair."
                )
            pair_path_masks.append(
                (
                    node_position[node_a],
                    node_position[node_b],
                    tuple(sorted(masks)),
                )
            )

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "max_total_simple_paths", path_limit)
        object.__setattr__(self, "total_simple_paths", total_paths)
        object.__setattr__(self, "evaluation_mode", "bounded-simple-paths")
        object.__setattr__(self, "_tree_neighbors", None)
        object.__setattr__(
            self,
            "_pair_path_masks",
            tuple(pair_path_masks),
        )

    def values(self, targets: frozenset[int]) -> tuple[float, ...]:
        """Return ``hat f_i(targets)`` aligned with ``nodes``."""
        selected = _normalize_targets(targets, nodes=frozenset(self.nodes))
        selected_positions = {
            position for position, node in enumerate(self.nodes) if node in selected
        }
        if self._tree_neighbors is not None:
            return self._tree_values(selected_positions)

        selected_mask = 0
        for position in selected_positions:
            selected_mask |= 1 << position

        values = [
            0.0 if position in selected_positions else 1.0
            for position in range(len(self.nodes))
        ]
        for position_a, position_b, path_masks in self._pair_path_masks:
            if position_a in selected_positions or position_b in selected_positions:
                continue
            surviving_paths = sum(
                path_mask & selected_mask == 0 for path_mask in path_masks
            )
            ratio = surviving_paths / len(path_masks)
            values[position_a] += ratio
            values[position_b] += ratio
        return tuple(values)

    def robust_value(self, targets: frozenset[int]) -> float:
        """Return the maximum faithful surrogate value."""
        return max(self.values(targets), default=0.0)

    def benefits(self, targets: frozenset[int]) -> tuple[float, ...]:
        """Return increasing submodular benefits ``n - hat f_i``."""
        n_nodes = float(len(self.nodes))
        return tuple(n_nodes - value for value in self.values(targets))

    def _tree_values(
        self,
        selected_positions: set[int],
    ) -> tuple[float, ...]:
        """Evaluate tree connectivity without enumerating unique paths."""
        if self._tree_neighbors is None:
            raise RuntimeError("tree evaluation requires tree neighbors.")
        values = [0.0] * len(self.nodes)
        remaining = set(range(len(self.nodes))) - selected_positions
        while remaining:
            root = min(remaining)
            component = {root}
            frontier = [root]
            while frontier:
                position = frontier.pop()
                for neighbor in self._tree_neighbors[position]:
                    if neighbor in remaining and neighbor not in component:
                        component.add(neighbor)
                        frontier.append(neighbor)
            component_size = float(len(component))
            for position in component:
                values[position] = component_size
            remaining.difference_update(component)
        return tuple(values)


@dataclass(frozen=True)
class _SaturationChoice:
    targets: frozenset[int]
    surrogate_score: float
    total_simple_paths: int
    evaluation_mode: str
    iterations: int


def _cached_benefits(
    evaluator: BoundedPathSurrogate,
    targets: frozenset[int],
    cache: dict[frozenset[int], tuple[float, ...]],
) -> tuple[float, ...]:
    """Return memoized benefit values for one target set."""
    if targets not in cache:
        cache[targets] = evaluator.benefits(targets)
    return cache[targets]


def _truncated_average(
    benefits: tuple[float, ...],
    threshold: float,
) -> float:
    """Return the SATURATE truncated-average objective."""
    return sum(min(value, threshold) for value in benefits) / len(benefits)


def _greedy_partial_cover(
    evaluator: BoundedPathSurrogate,
    *,
    threshold: float,
    max_targets: int,
    tolerance: float,
    cache: dict[frozenset[int], tuple[float, ...]],
) -> tuple[frozenset[int], bool]:
    """Run deterministic GPC without exceeding the hard target cap."""
    selected = frozenset()
    current = _truncated_average(
        _cached_benefits(evaluator, selected, cache),
        threshold,
    )
    while current < threshold - tolerance and len(selected) < max_targets:
        best_node: int | None = None
        best_value = current
        best_gain = float("-inf")
        for node in evaluator.nodes:
            if node in selected:
                continue
            candidate = selected | {node}
            candidate_value = _truncated_average(
                _cached_benefits(evaluator, candidate, cache),
                threshold,
            )
            gain = candidate_value - current
            if gain > best_gain + 1e-15 or (
                abs(gain - best_gain) <= 1e-15
                and (best_node is None or node < best_node)
            ):
                best_node = node
                best_value = candidate_value
                best_gain = gain

        if best_node is None or best_gain <= 1e-15:
            break
        selected = selected | {best_node}
        current = best_value

    return selected, current >= threshold - tolerance


def _choose_saturation_targets(
    component_graph: nx.Graph,
    *,
    max_targets: int,
    max_total_simple_paths: int,
    saturation_tolerance: float,
    max_saturation_iterations: int,
) -> _SaturationChoice:
    """Run bounded hard-cap SATURATE and return the best visited legal set."""
    max_targets = _positive_integer("max_targets", max_targets)
    path_limit = _positive_integer(
        "max_total_simple_paths",
        max_total_simple_paths,
    )
    tolerance = _saturation_tolerance(saturation_tolerance)
    max_iterations = _positive_integer(
        "max_saturation_iterations",
        max_saturation_iterations,
    )
    canonical, nodes = _validated_chordal_component(component_graph)
    if len(nodes) <= max_targets:
        targets = frozenset(nodes)
        return _SaturationChoice(
            targets=targets,
            surrogate_score=0.0,
            total_simple_paths=0,
            evaluation_mode="whole-component",
            iterations=0,
        )

    evaluator = BoundedPathSurrogate(
        canonical,
        max_total_simple_paths=path_limit,
    )
    benefit_cache: dict[frozenset[int], tuple[float, ...]] = {}
    robust_cache: dict[frozenset[int], float] = {}

    def candidate_key(
        targets: frozenset[int],
    ) -> tuple[float, int, tuple[int, ...]]:
        if targets not in robust_cache:
            robust_cache[targets] = evaluator.robust_value(targets)
        return robust_cache[targets], len(targets), tuple(sorted(targets))

    lower = 0.0
    upper = float(len(nodes))
    visited: set[frozenset[int]] = set()
    iterations = 0
    while iterations < max_iterations and upper - lower > tolerance:
        iterations += 1
        threshold = (lower + upper) / 2.0
        targets, covered = _greedy_partial_cover(
            evaluator,
            threshold=threshold,
            max_targets=max_targets,
            tolerance=tolerance,
            cache=benefit_cache,
        )
        if targets:
            visited.add(targets)
        if covered:
            lower = threshold
        else:
            upper = threshold

    if not visited:
        visited.update(frozenset({node}) for node in nodes)
    targets = min(visited, key=candidate_key)
    return _SaturationChoice(
        targets=targets,
        surrogate_score=robust_cache[targets],
        total_simple_paths=evaluator.total_simple_paths,
        evaluation_mode=evaluator.evaluation_mode,
        iterations=iterations,
    )


def saturate_supermodular_targets(
    component_graph: nx.Graph,
    *,
    max_targets: int,
    max_total_simple_paths: int,
    saturation_tolerance: float,
    max_saturation_iterations: int,
) -> frozenset[int]:
    """Return deterministic hard-cap SATURATE targets.

    This uses ``alpha=1`` to preserve the intervention-width contract. It is
    therefore a heuristic without the bicriteria size-relaxation guarantee of
    the original SATURATE theorem.
    """
    return _choose_saturation_targets(
        component_graph,
        max_targets=max_targets,
        max_total_simple_paths=max_total_simple_paths,
        saturation_tolerance=saturation_tolerance,
        max_saturation_iterations=max_saturation_iterations,
    ).targets


@dataclass(frozen=True)
class SupermodularProbePolicy:
    """Apply bounded faithful Supermodular exploration to one top component."""

    max_probe_targets: int
    shift_magnitude: float
    max_total_simple_paths: int
    saturation_tolerance: float
    max_saturation_iterations: int

    def __post_init__(self) -> None:
        """Validate all probe, path, and SATURATE controls."""
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
            "max_total_simple_paths",
            _positive_integer(
                "max_total_simple_paths",
                self.max_total_simple_paths,
            ),
        )
        object.__setattr__(
            self,
            "saturation_tolerance",
            _saturation_tolerance(self.saturation_tolerance),
        )
        object.__setattr__(
            self,
            "max_saturation_iterations",
            _positive_integer(
                "max_saturation_iterations",
                self.max_saturation_iterations,
            ),
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
        """Return one bounded faithful Supermodular probe when legal."""
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
        choice = _choose_saturation_targets(
            component_graph,
            max_targets=self.max_probe_targets,
            max_total_simple_paths=self.max_total_simple_paths,
            saturation_tolerance=self.saturation_tolerance,
            max_saturation_iterations=self.max_saturation_iterations,
        )

        increment = np.zeros(n_nodes, dtype=float)
        increment[list(choice.targets)] = self.shift_magnitude
        component_label = ",".join(str(node) for node in sorted(component))
        target_label = ",".join(str(node) for node in sorted(choice.targets))
        return ShiftProbe(
            shift_increment=increment,
            targets=choice.targets,
            reason=(
                "supermodular-saturate:"
                f"component={component_label};targets={target_label};"
                f"score={choice.surrogate_score:.12g};"
                f"simple-paths={choice.total_simple_paths};"
                f"mode={choice.evaluation_mode};"
                f"iterations={choice.iterations};alpha=1"
            ),
        )
