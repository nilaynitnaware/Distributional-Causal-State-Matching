"""Small-n exact matching-specific singleton-query complexity prototype.

This module is deliberately isolated from the production matching pipeline.  It
enumerates a PDAG's observational MEC, reconstructs each candidate Gaussian SCM
from public moments, reuses the exact known-DAG sparse shift--scale solver, and
solves the matching- and DAG-identification Bellman recurrences exactly.

The implementation is exponential by design and intended for roughly n <= 10.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from math import inf
from types import MappingProxyType

import networkx as nx
import numpy as np

from graphs.pdag import PartiallyDirectedGraph
from graphs.support import DEFAULT_SUPPORT_ATOL, parameter_support_cost
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from oracle.sparse_shift_scale_kl import (
    LocalActionState,
    SparseShiftScaleOracleResult,
    interventional_moments_from_parameters,
    solve_sparse_shift_scale_kl_from_parameters,
)

DEFAULT_MOMENT_ATOL = 1e-10
DEFAULT_MOMENT_RTOL = 1e-9
DEFAULT_ACTION_ATOL = 1e-10
DEFAULT_ACTION_RTOL = 1e-9
DEFAULT_LOSS_ATOL = 1e-12
DEFAULT_LOSS_RTOL = 1e-11

DagId = tuple[tuple[int, int], ...]
ParentSignature = tuple[int, ...]


def _loss_tolerance(left: float, right: float, atol: float, rtol: float) -> float:
    return atol + rtol * max(1.0, abs(left), abs(right))


def _loss_tied(left: float, right: float, atol: float, rtol: float) -> bool:
    return abs(left - right) <= _loss_tolerance(left, right, atol, rtol)


def _skeleton(edges: Iterable[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    return frozenset(tuple(sorted((int(a), int(b)))) for a, b in edges)


def _pdag_skeleton(pdag: PartiallyDirectedGraph) -> frozenset[tuple[int, int]]:
    return _skeleton(pdag.directed_edges | pdag.undirected_edges)


def unshielded_colliders(dag: nx.DiGraph) -> frozenset[tuple[int, int, int]]:
    """Return canonical ``(outer_low, center, outer_high)`` collider triples."""
    skeleton = _skeleton(dag.edges)
    colliders: set[tuple[int, int, int]] = set()
    for center in dag.nodes:
        parents = sorted(int(parent) for parent in dag.predecessors(center))
        for left, right in combinations(parents, 2):
            if tuple(sorted((left, right))) not in skeleton:
                colliders.add((min(left, right), int(center), max(left, right)))
    return frozenset(colliders)


def required_pdag_colliders(
    pdag: PartiallyDirectedGraph,
) -> frozenset[tuple[int, int, int]]:
    """Return unshielded colliders already encoded by directed PDAG edges."""
    skeleton = _pdag_skeleton(pdag)
    by_child: dict[int, list[int]] = {node: [] for node in pdag.nodes}
    for parent, child in pdag.directed_edges:
        by_child[child].append(parent)
    output: set[tuple[int, int, int]] = set()
    for center, parents in by_child.items():
        for left, right in combinations(sorted(parents), 2):
            if tuple(sorted((left, right))) not in skeleton:
                output.add((min(left, right), center, max(left, right)))
    return frozenset(output)


@dataclass(frozen=True)
class CandidateDAG:
    """One validated consistent extension with a stable canonical edge ID."""

    id: DagId
    dag: nx.DiGraph
    parent_sets: tuple[ParentSignature, ...]
    colliders: frozenset[tuple[int, int, int]]


def validate_consistent_dag(
    pdag: PartiallyDirectedGraph,
    dag: nx.DiGraph,
) -> None:
    """Validate skeleton, compelled directions, acyclicity, and MEC membership."""
    if set(dag.nodes) != set(pdag.nodes):
        raise ValueError("candidate DAG nodes differ from the PDAG nodes")
    if _skeleton(dag.edges) != _pdag_skeleton(pdag):
        raise ValueError("candidate DAG skeleton differs from the PDAG skeleton")
    if not pdag.directed_edges <= frozenset(
        (int(parent), int(child)) for parent, child in dag.edges
    ):
        raise ValueError("candidate DAG violates a compelled/public orientation")
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("candidate orientation is cyclic")
    required = required_pdag_colliders(pdag)
    actual = unshielded_colliders(dag)
    if actual != required:
        missing = tuple(sorted(required - actual))
        added = tuple(sorted(actual - required))
        raise ValueError(
            f"candidate DAG has the wrong unshielded colliders; missing={missing}, "
            f"added={added}"
        )


def enumerate_consistent_dags(
    pdag: PartiallyDirectedGraph,
    *,
    max_undirected_edges: int = 24,
) -> tuple[CandidateDAG, ...]:
    """Brute-force every consistent DAG extension of a small observational PDAG.

    Equality of skeleton and unshielded-collider sets is the standard
    observational Markov-equivalence criterion for DAGs.  The explicit cap is a
    guard against accidentally invoking this prototype on a large graph.
    """
    nodes = tuple(sorted(pdag.nodes))
    if nodes != tuple(range(len(nodes))):
        raise ValueError("prototype requires nodes labeled 0, ..., n - 1")
    undirected = tuple(sorted(pdag.undirected_edges))
    if len(undirected) > max_undirected_edges:
        raise ValueError("too many undirected edges for brute-force enumeration")

    candidates: list[CandidateDAG] = []
    for bits in product((0, 1), repeat=len(undirected)):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(pdag.directed_edges)
        graph.add_edges_from(
            (edge if bit == 0 else (edge[1], edge[0]))
            for edge, bit in zip(undirected, bits, strict=True)
        )
        try:
            validate_consistent_dag(pdag, graph)
        except ValueError:
            continue
        edge_id: DagId = tuple(sorted((int(a), int(b)) for a, b in graph.edges))
        parents = tuple(
            tuple(sorted(int(parent) for parent in graph.predecessors(node)))
            for node in nodes
        )
        candidates.append(
            CandidateDAG(
                id=edge_id,
                dag=nx.freeze(nx.DiGraph(graph)),
                parent_sets=parents,
                colliders=unshielded_colliders(graph),
            )
        )
    candidates.sort(key=lambda item: item.id)
    if len({candidate.id for candidate in candidates}) != len(candidates):
        raise RuntimeError("canonical DAG IDs are not unique")
    return tuple(candidates)


@dataclass(frozen=True)
class CandidateSCM:
    """Gaussian SCM reconstructed from public moments for one candidate DAG."""

    candidate: CandidateDAG
    coefficients: np.ndarray
    intercepts: np.ndarray
    noise_variances: np.ndarray
    system_matrix: np.ndarray
    total_effect_matrix: np.ndarray
    reproduction_mean_error: float
    reproduction_covariance_error: float

    def observational_moments(self) -> tuple[np.ndarray, np.ndarray]:
        mean = self.total_effect_matrix @ self.intercepts
        covariance = (
            self.total_effect_matrix
            @ np.diag(self.noise_variances)
            @ self.total_effect_matrix.T
        )
        return mean, covariance

    def singleton_shift_response(
        self, node: int, *, shift_magnitude: float = 1.0
    ) -> np.ndarray:
        return float(shift_magnitude) * self.total_effect_matrix[:, int(node)]

    def interventional_moments(
        self, shift: Sequence[float], scale: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        return interventional_moments_from_parameters(
            source_intercepts=self.intercepts,
            coefficients=self.coefficients,
            noise_variances=self.noise_variances,
            shift=np.asarray(shift, dtype=float),
            scale=np.asarray(scale, dtype=float),
        )


def reconstruct_candidate_scm(
    candidate: CandidateDAG,
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    *,
    moment_atol: float = DEFAULT_MOMENT_ATOL,
    moment_rtol: float = DEFAULT_MOMENT_RTOL,
) -> CandidateSCM:
    """Recover the unique linear-Gaussian parameterization for a candidate DAG."""
    mean = np.asarray(source_mean, dtype=float)
    covariance = np.asarray(source_covariance, dtype=float)
    n = len(candidate.parent_sets)
    if mean.shape != (n,) or covariance.shape != (n, n):
        raise ValueError("source moments do not match the candidate dimension")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError("source moments must be finite")
    if not np.allclose(covariance, covariance.T, atol=moment_atol, rtol=moment_rtol):
        raise ValueError("source covariance must be symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("source covariance must be positive definite") from error

    coefficients = np.zeros((n, n), dtype=float)
    intercepts = np.zeros(n, dtype=float)
    noise = np.zeros(n, dtype=float)
    for node, parents in enumerate(candidate.parent_sets):
        if parents:
            index = np.asarray(parents, dtype=int)
            beta = np.linalg.solve(
                covariance[np.ix_(index, index)], covariance[index, node]
            )
            coefficients[index, node] = beta
            intercepts[node] = mean[node] - beta @ mean[index]
            noise[node] = (
                covariance[node, node] - covariance[node, index] @ beta
            )
        else:
            intercepts[node] = mean[node]
            noise[node] = covariance[node, node]
    if np.any(noise <= 0.0):
        raise RuntimeError("candidate conditional noise variance is not positive")

    system = np.eye(n) - coefficients.T
    total_effect = np.linalg.solve(system, np.eye(n))
    reconstructed_mean = total_effect @ intercepts
    reconstructed_covariance = total_effect @ np.diag(noise) @ total_effect.T
    mean_error = float(np.max(np.abs(reconstructed_mean - mean), initial=0.0))
    covariance_error = float(
        np.max(np.abs(reconstructed_covariance - covariance), initial=0.0)
    )
    if not np.allclose(
        reconstructed_mean, mean, atol=moment_atol, rtol=moment_rtol
    ) or not np.allclose(
        reconstructed_covariance,
        covariance,
        atol=moment_atol,
        rtol=moment_rtol,
    ):
        raise RuntimeError("candidate SCM does not reproduce the public moments")

    frozen_arrays: list[np.ndarray] = []
    for value in (coefficients, intercepts, noise, system, total_effect):
        copied = np.array(value, dtype=float, copy=True)
        copied.setflags(write=False)
        frozen_arrays.append(copied)
    return CandidateSCM(
        candidate=candidate,
        coefficients=frozen_arrays[0],
        intercepts=frozen_arrays[1],
        noise_variances=frozen_arrays[2],
        system_matrix=frozen_arrays[3],
        total_effect_matrix=frozen_arrays[4],
        reproduction_mean_error=mean_error,
        reproduction_covariance_error=covariance_error,
    )


def reconstruct_all_candidate_scms(
    candidates: Sequence[CandidateDAG],
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    **kwargs: float,
) -> dict[DagId, CandidateSCM]:
    return {
        candidate.id: reconstruct_candidate_scm(
            candidate, source_mean, source_covariance, **kwargs
        )
        for candidate in candidates
    }


@dataclass(frozen=True, order=True)
class CanonicalAction:
    """One canonical *effective* known-DAG optimum.

    Root scale entries are one.  This object is not, by itself, the complete
    physical-command optimum set when root status varies across candidate DAGs.
    OptimalCommandFamily performs that symbolic lift.
    """

    shift: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def from_arrays(
        cls,
        shift: Sequence[float],
        scale: Sequence[float],
        *,
        root_nodes: Iterable[int] = (),
    ) -> CanonicalAction:
        shift_values = [0.0 if float(value) == 0.0 else float(value) for value in shift]
        scale_values = [1.0 if float(value) == 1.0 else float(value) for value in scale]
        if len(shift_values) != len(scale_values):
            raise ValueError("shift and scale vectors must have equal length")
        for root in root_nodes:
            scale_values[int(root)] = 1.0
        return cls(tuple(shift_values), tuple(scale_values))

    @property
    def dimension(self) -> int:
        return len(self.shift)


@dataclass(frozen=True, order=True)
class PhysicalCommand:
    """One DAG-independent executable shift--scale command vector."""

    shift: tuple[float, ...]
    scale: tuple[float, ...]

    @property
    def dimension(self) -> int:
        return len(self.shift)


ActionVector = CanonicalAction | PhysicalCommand


def actions_close(
    left: ActionVector,
    right: ActionVector,
    *,
    atol: float = DEFAULT_ACTION_ATOL,
    rtol: float = DEFAULT_ACTION_RTOL,
) -> bool:
    return left.dimension == right.dimension and np.allclose(
        left.shift, right.shift, atol=atol, rtol=rtol
    ) and np.allclose(left.scale, right.scale, atol=atol, rtol=rtol)


@dataclass(frozen=True)
class OptimalCommandFamily:
    """Symbolic lift of one canonical optimum to physical command space.

    Every shift coordinate and every non-root scale is fixed by
    canonical_action. Scales at free_root_scale_nodes may take any value in
    scale_bounds. The family is finally restricted by one global commanded-
    coordinate budget; a nonidentity root scale therefore consumes budget even
    though it is a no-op for this DAG.

    Completeness assumes canonical_action belongs to the complete retained set
    of tied effective optima. Continuous non-root degeneracies remain a
    fail-loud condition in the existing optimizer/campaign layer.
    """

    canonical_action: CanonicalAction
    free_root_scale_nodes: frozenset[int]
    shift_bounds: tuple[float, float]
    scale_bounds: tuple[float, float]
    budget: int

    @classmethod
    def from_canonical(
        cls,
        action: CanonicalAction,
        *,
        root_nodes: Iterable[int],
        shift_bounds: tuple[float, float],
        scale_bounds: tuple[float, float],
        budget: int,
        action_atol: float = DEFAULT_ACTION_ATOL,
        action_rtol: float = DEFAULT_ACTION_RTOL,
    ) -> OptimalCommandFamily:
        roots = frozenset(int(node) for node in root_nodes)
        if any(node < 0 or node >= action.dimension for node in roots):
            raise ValueError("root node is outside the action dimension")
        if not scale_bounds[0] > 0.0 or scale_bounds[0] > scale_bounds[1]:
            raise ValueError("scale bounds must be positive and ordered")
        if shift_bounds[0] > shift_bounds[1]:
            raise ValueError("shift bounds must be ordered")
        if budget < 0:
            raise ValueError("command budget must be nonnegative")
        if any(
            not shift_bounds[0] <= value <= shift_bounds[1]
            for value in action.shift
        ):
            raise ValueError("canonical shift lies outside command bounds")
        if any(value <= 0.0 for value in action.scale):
            raise ValueError("canonical scales must be positive")
        for node, value in enumerate(action.scale):
            if node in roots:
                if not np.isclose(value, 1.0, atol=action_atol, rtol=action_rtol):
                    raise ValueError("canonical root scale is not one")
            elif not scale_bounds[0] <= value <= scale_bounds[1]:
                raise ValueError("canonical non-root scale lies outside bounds")
        return cls(
            canonical_action=action,
            free_root_scale_nodes=roots,
            shift_bounds=shift_bounds,
            scale_bounds=scale_bounds,
            budget=int(budget),
        )


@dataclass(frozen=True)
class CandidateCommandLoss:
    dag_id: DagId
    direct_kl: float
    optimal_kl: float
    absolute_residual: float
    tolerance: float


@dataclass(frozen=True)
class CommonCommandCertificate:
    """Independently checked witness for a positive matching terminal."""

    command: PhysicalCommand
    commanded_support: int
    selected_family_indices: tuple[tuple[DagId, int], ...]
    candidate_losses: tuple[CandidateCommandLoss, ...]
    maximum_loss_residual: float


@dataclass(frozen=True)
class OptimalActionSet:
    """Canonical effective optima plus their complete root-scale command lift."""

    dag_id: DagId
    actions: tuple[CanonicalAction, ...]
    command_families: tuple[OptimalCommandFamily, ...]
    optimal_kl: float
    oracle_result: SparseShiftScaleOracleResult
    numerical_diagnostics: tuple[str, ...]
    scm: CandidateSCM
    target_mean: np.ndarray
    target_covariance: np.ndarray
    budget: int
    shift_bounds: tuple[float, float]
    log_scale_bounds: tuple[float, float]
    scale_bounds: tuple[float, float]
    support_atol: float
    loss_atol: float
    loss_rtol: float
    action_atol: float
    action_rtol: float


def _enumerate_feasible_state_products(
    states_by_node: tuple[tuple[LocalActionState, ...], ...],
    budget: int,
) -> Iterable[tuple[LocalActionState, ...]]:
    chosen: list[LocalActionState] = []

    def visit(node: int, cost: int) -> Iterable[tuple[LocalActionState, ...]]:
        if node == len(states_by_node):
            yield tuple(chosen)
            return
        for state in states_by_node[node]:
            next_cost = cost + state.effective_cost
            if next_cost > budget:
                continue
            chosen.append(state)
            yield from visit(node + 1, next_cost)
            chosen.pop()

    return visit(0, 0)


def compute_optimal_action_set(
    scm: CandidateSCM,
    *,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    budget: int,
    shift_bounds: tuple[float, float],
    log_scale_bounds: tuple[float, float],
    loss_atol: float = DEFAULT_LOSS_ATOL,
    loss_rtol: float = DEFAULT_LOSS_RTOL,
    action_atol: float = DEFAULT_ACTION_ATOL,
    action_rtol: float = DEFAULT_ACTION_RTOL,
) -> OptimalActionSet:
    """Reuse the exact solver, then preserve every globally tied full action.

    Production's deterministic backpointer remains unchanged.  This prototype
    enumerates the already validated local states under the sparse budget and
    retains all full vectors tied at the declared objective tolerance.
    """
    result = solve_sparse_shift_scale_kl_from_parameters(
        dag=scm.candidate.dag,
        source_intercepts=scm.intercepts,
        coefficients=scm.coefficients,
        noise_variances=scm.noise_variances,
        target_mean=np.asarray(target_mean, dtype=float),
        target_covariance=np.asarray(target_covariance, dtype=float),
        budget=budget,
        shift_bounds=shift_bounds,
        log_scale_bounds=log_scale_bounds,
        loss_atol=loss_atol,
        loss_rtol=loss_rtol,
        return_all_optimal_supports=True,
        provenance={"research_module": "matching_query_complexity"},
    )
    feasible: list[tuple[float, CanonicalAction]] = []
    roots = tuple(
        node for node, parents in enumerate(scm.candidate.parent_sets) if not parents
    )
    for states in _enumerate_feasible_state_products(
        result.node_local_candidate_states, result.effective_budget
    ):
        action = CanonicalAction.from_arrays(
            [state.shift for state in states],
            [state.scale for state in states],
            root_nodes=roots,
        )
        local_sum = float(sum(state.raw_local_loss for state in states))
        feasible.append((result.kl_constant + local_sum, action))
    if not feasible:
        raise RuntimeError("no feasible action survived local-state enumeration")
    exact_minimum = min(value for value, _ in feasible)
    tied = [
        action
        for value, action in feasible
        if _loss_tied(value, exact_minimum, loss_atol, loss_rtol)
    ]
    unique: list[CanonicalAction] = []
    for action in sorted(tied):
        if not any(
            actions_close(action, existing, atol=action_atol, rtol=action_rtol)
            for existing in unique
        ):
            unique.append(action)
    diagnostics = list(result.tie_diagnostics)
    diagnostics.extend(
        f"node={node},state={state.label}:local_numerical_tie"
        for node, states in enumerate(result.node_local_candidate_states)
        for state in states
        if state.numerically_tied
    )
    if not _loss_tied(result.raw_K_B_star, exact_minimum, loss_atol, loss_rtol):
        raise RuntimeError("tie-preserving enumeration disagrees with production DP")
    if len(unique) > 1:
        diagnostics.append(f"global_optimal_action_tie_count={len(unique)}")
    scale_bounds = (
        float(np.exp(log_scale_bounds[0])),
        float(np.exp(log_scale_bounds[1])),
    )
    command_families = tuple(
        OptimalCommandFamily.from_canonical(
            action,
            root_nodes=roots,
            shift_bounds=shift_bounds,
            scale_bounds=scale_bounds,
            budget=budget,
            action_atol=action_atol,
            action_rtol=action_rtol,
        )
        for action in unique
    )
    frozen_target_mean = np.array(target_mean, dtype=float, copy=True)
    frozen_target_covariance = np.array(target_covariance, dtype=float, copy=True)
    frozen_target_mean.setflags(write=False)
    frozen_target_covariance.setflags(write=False)
    return OptimalActionSet(
        dag_id=scm.candidate.id,
        actions=tuple(unique),
        command_families=command_families,
        optimal_kl=float(exact_minimum),
        oracle_result=result,
        numerical_diagnostics=tuple(diagnostics),
        scm=scm,
        target_mean=frozen_target_mean,
        target_covariance=frozen_target_covariance,
        budget=int(budget),
        shift_bounds=tuple(float(value) for value in shift_bounds),
        log_scale_bounds=tuple(float(value) for value in log_scale_bounds),
        scale_bounds=scale_bounds,
        support_atol=float(result.tolerances["support_atol"]),
        loss_atol=float(loss_atol),
        loss_rtol=float(loss_rtol),
        action_atol=float(action_atol),
        action_rtol=float(action_rtol),
    )


def compute_all_optimal_action_sets(
    scms: Mapping[DagId, CandidateSCM],
    **kwargs: object,
) -> dict[DagId, OptimalActionSet]:
    return {
        dag_id: compute_optimal_action_set(scm, **kwargs)
        for dag_id, scm in sorted(scms.items())
    }


class CommandCertificationError(RuntimeError):
    """A structural common-command candidate could not be KL-certified."""


def _merge_fixed_value(
    current: float | None,
    proposed: float,
    *,
    action_atol: float,
    action_rtol: float,
) -> float | None:
    if current is None:
        return float(proposed)
    if np.isclose(current, proposed, atol=action_atol, rtol=action_rtol):
        return current
    return None


def _iter_common_command_candidates(
    model_ids: Iterable[DagId],
    families_by_model: Mapping[DagId, Sequence[OptimalCommandFamily]],
    *,
    budget: int,
    support_atol: float,
    action_atol: float,
    action_rtol: float,
) -> Iterable[tuple[PhysicalCommand, tuple[tuple[DagId, int], ...]]]:
    """Exhaust all simultaneous tied-family choices and yield feasible commands."""
    ids = tuple(sorted(model_ids))
    if not ids:
        raise ValueError("model set must be nonempty")
    if budget < 0:
        raise ValueError("command budget must be nonnegative")
    if any(not families_by_model[dag_id] for dag_id in ids):
        raise ValueError("every candidate must provide at least one command family")
    first = families_by_model[ids[0]][0]
    dimension = first.canonical_action.dimension
    yielded: list[PhysicalCommand] = []

    def visit(
        model_index: int,
        shifts: tuple[float | None, ...],
        scales: tuple[float | None, ...],
        shift_lower: float,
        shift_upper: float,
        scale_lower: float,
        scale_upper: float,
        selected: tuple[tuple[DagId, int], ...],
    ) -> Iterable[tuple[PhysicalCommand, tuple[tuple[DagId, int], ...]]]:
        if model_index == len(ids):
            if shift_lower > shift_upper or scale_lower > scale_upper:
                return
            if not shift_lower <= 0.0 <= shift_upper:
                return
            if not scale_lower <= 1.0 <= scale_upper:
                return
            if any(value is None for value in shifts):
                raise RuntimeError("a command family left a shift coordinate free")
            command = PhysicalCommand(
                shift=tuple(float(value) for value in shifts if value is not None),
                scale=tuple(
                    1.0 if value is None else float(value) for value in scales
                ),
            )
            if any(not shift_lower <= value <= shift_upper for value in command.shift):
                return
            if any(
                value <= 0.0 or not scale_lower <= value <= scale_upper
                for value in command.scale
            ):
                return
            support = parameter_support_cost(
                np.asarray(command.shift, dtype=float),
                np.asarray(command.scale, dtype=float),
                support_atol=support_atol,
            )
            if support > budget:
                return
            if any(
                actions_close(
                    command, existing, atol=action_atol, rtol=action_rtol
                )
                for existing in yielded
            ):
                return
            yielded.append(command)
            yield command, selected
            return

        dag_id = ids[model_index]
        for family_index, family in enumerate(families_by_model[dag_id]):
            action = family.canonical_action
            if action.dimension != dimension:
                raise ValueError("command-family dimensions differ")
            if family.budget != budget:
                raise ValueError("command-family budgets differ")
            next_shift_lower = max(shift_lower, family.shift_bounds[0])
            next_shift_upper = min(shift_upper, family.shift_bounds[1])
            next_scale_lower = max(scale_lower, family.scale_bounds[0])
            next_scale_upper = min(scale_upper, family.scale_bounds[1])
            if (
                next_shift_lower > next_shift_upper
                or next_scale_lower > next_scale_upper
            ):
                continue
            next_shifts = list(shifts)
            next_scales = list(scales)
            compatible = True
            for node, value in enumerate(action.shift):
                merged = _merge_fixed_value(
                    next_shifts[node],
                    value,
                    action_atol=action_atol,
                    action_rtol=action_rtol,
                )
                if merged is None:
                    compatible = False
                    break
                next_shifts[node] = merged
            if not compatible:
                continue
            for node, value in enumerate(action.scale):
                if node in family.free_root_scale_nodes:
                    continue
                merged = _merge_fixed_value(
                    next_scales[node],
                    value,
                    action_atol=action_atol,
                    action_rtol=action_rtol,
                )
                if merged is None:
                    compatible = False
                    break
                next_scales[node] = merged
            if not compatible:
                continue
            lower_bound_command = PhysicalCommand(
                shift=tuple(
                    0.0 if value is None else float(value) for value in next_shifts
                ),
                scale=tuple(
                    1.0 if value is None else float(value) for value in next_scales
                ),
            )
            lower_bound_support = parameter_support_cost(
                np.asarray(lower_bound_command.shift, dtype=float),
                np.asarray(lower_bound_command.scale, dtype=float),
                support_atol=support_atol,
            )
            if lower_bound_support > budget:
                continue
            yield from visit(
                model_index + 1,
                tuple(next_shifts),
                tuple(next_scales),
                next_shift_lower,
                next_shift_upper,
                next_scale_lower,
                next_scale_upper,
                selected + ((dag_id, family_index),),
            )

    yield from visit(
        0,
        tuple(None for _ in range(dimension)),
        tuple(None for _ in range(dimension)),
        -inf,
        inf,
        0.0,
        inf,
        (),
    )


def intersect_optimal_command_families(
    model_ids: Iterable[DagId],
    families_by_model: Mapping[DagId, Sequence[OptimalCommandFamily]],
    *,
    budget: int,
    support_atol: float = DEFAULT_SUPPORT_ATOL,
    action_atol: float = DEFAULT_ACTION_ATOL,
    action_rtol: float = DEFAULT_ACTION_RTOL,
) -> tuple[PhysicalCommand, ...]:
    """Return all distinct structural witnesses from simultaneous family choices.

    This low-level routine checks bounds, positivity, and one global commanded
    support budget. It intentionally does not claim optimality certification;
    production Bellman stopping uses find_common_optimal_command below.
    """
    return tuple(
        command
        for command, _ in _iter_common_command_candidates(
            model_ids,
            families_by_model,
            budget=budget,
            support_atol=support_atol,
            action_atol=action_atol,
            action_rtol=action_rtol,
        )
    )


def _common_conventions(
    ids: tuple[DagId, ...],
    action_sets: Mapping[DagId, OptimalActionSet],
) -> tuple[int, float, float, float]:
    first = action_sets[ids[0]]
    conventions = (
        first.budget,
        first.support_atol,
        first.action_atol,
        first.action_rtol,
    )
    for dag_id in ids[1:]:
        current = action_sets[dag_id]
        if (
            current.budget,
            current.support_atol,
            current.action_atol,
            current.action_rtol,
        ) != conventions:
            raise ValueError("candidate command budgets or tolerances differ")
    return conventions


def _certify_common_command(
    command: PhysicalCommand,
    selected: tuple[tuple[DagId, int], ...],
    ids: tuple[DagId, ...],
    action_sets: Mapping[DagId, OptimalActionSet],
    *,
    budget: int,
    support_atol: float,
    instrumentation: MutableMapping[str, int] | None = None,
) -> CommonCommandCertificate | None:
    shift = np.asarray(command.shift, dtype=float)
    scale = np.asarray(command.scale, dtype=float)
    support = parameter_support_cost(shift, scale, support_atol=support_atol)
    if support > budget:
        raise CommandCertificationError("structural witness exceeds command budget")
    losses: list[CandidateCommandLoss] = []
    for dag_id in ids:
        if instrumentation is not None:
            instrumentation["direct_kl_evaluations"] = (
                instrumentation.get("direct_kl_evaluations", 0) + 1
            )
        action_set = action_sets[dag_id]
        if np.any(shift < action_set.shift_bounds[0]) or np.any(
            shift > action_set.shift_bounds[1]
        ):
            raise CommandCertificationError("structural witness violates shift bounds")
        if (
            np.any(scale <= 0.0)
            or np.any(scale < action_set.scale_bounds[0])
            or np.any(scale > action_set.scale_bounds[1])
        ):
            raise CommandCertificationError("structural witness violates scale bounds")
        candidate_mean, candidate_covariance = action_set.scm.interventional_moments(
            shift, scale
        )
        direct_kl = gaussian_kl_target_to_candidate(
            action_set.target_mean,
            action_set.target_covariance,
            candidate_mean,
            candidate_covariance,
        )
        tolerance = _loss_tolerance(
            direct_kl,
            action_set.optimal_kl,
            action_set.loss_atol,
            action_set.loss_rtol,
        )
        residual = abs(direct_kl - action_set.optimal_kl)
        losses.append(
            CandidateCommandLoss(
                dag_id=dag_id,
                direct_kl=float(direct_kl),
                optimal_kl=float(action_set.optimal_kl),
                absolute_residual=float(residual),
                tolerance=float(tolerance),
            )
        )
    if any(item.absolute_residual > item.tolerance for item in losses):
        return None
    return CommonCommandCertificate(
        command=command,
        commanded_support=support,
        selected_family_indices=selected,
        candidate_losses=tuple(losses),
        maximum_loss_residual=max(
            (item.absolute_residual for item in losses), default=0.0
        ),
    )


def find_common_optimal_command(
    model_ids: Iterable[DagId],
    action_sets: Mapping[DagId, OptimalActionSet],
    *,
    instrumentation: MutableMapping[str, int] | None = None,
) -> CommonCommandCertificate | None:
    """Find and independently certify one simultaneous optimal physical command."""
    ids = tuple(sorted(model_ids))
    if not ids:
        raise ValueError("model set must be nonempty")
    if not all(isinstance(action_sets[dag_id], OptimalActionSet) for dag_id in ids):
        raise TypeError("certified command search requires OptimalActionSet values")
    budget, support_atol, action_atol, action_rtol = _common_conventions(
        ids, action_sets
    )
    families = {dag_id: action_sets[dag_id].command_families for dag_id in ids}
    if instrumentation is not None:
        instrumentation.setdefault("structural_candidates_visited", 0)
        instrumentation.setdefault("direct_kl_evaluations", 0)
    structural_candidates = 0
    largest_failed_residual = 0.0
    for command, selected in _iter_common_command_candidates(
        ids,
        families,
        budget=budget,
        support_atol=support_atol,
        action_atol=action_atol,
        action_rtol=action_rtol,
    ):
        structural_candidates += 1
        if instrumentation is not None:
            instrumentation["structural_candidates_visited"] += 1
        certificate = _certify_common_command(
            command,
            selected,
            ids,
            action_sets,
            budget=budget,
            support_atol=support_atol,
            instrumentation=instrumentation,
        )
        if certificate is not None:
            return certificate
        for dag_id in ids:
            if instrumentation is not None:
                instrumentation["direct_kl_evaluations"] += 1
            action_set = action_sets[dag_id]
            mean, covariance = action_set.scm.interventional_moments(
                np.asarray(command.shift), np.asarray(command.scale)
            )
            value = gaussian_kl_target_to_candidate(
                action_set.target_mean,
                action_set.target_covariance,
                mean,
                covariance,
            )
            largest_failed_residual = max(
                largest_failed_residual, abs(value - action_set.optimal_kl)
            )
    if structural_candidates:
        raise CommandCertificationError(
            "common-command geometry produced only uncertified witnesses; "
            f"largest_loss_residual={largest_failed_residual:.17g}"
        )
    return None


def common_optimal_actions(
    model_ids: Iterable[DagId],
    action_sets: Mapping[DagId, OptimalActionSet | Sequence[CanonicalAction]],
    *,
    action_atol: float = DEFAULT_ACTION_ATOL,
    action_rtol: float = DEFAULT_ACTION_RTOL,
) -> tuple[ActionVector, ...]:
    """Compatibility wrapper returning a certified witness when lifts exist.

    For OptimalActionSet inputs this returns zero or one physical command. Bare
    canonical-action sequences retain the legacy finite-set intersection
    behavior for diagnostic callers, but Bellman stopping rejects that
    uncertified representation.
    """
    ids = tuple(sorted(model_ids))
    if not ids:
        raise ValueError("model set must be nonempty")
    if all(isinstance(action_sets[dag_id], OptimalActionSet) for dag_id in ids):
        certified_sets = {
            dag_id: action_sets[dag_id]
            for dag_id in ids
            if isinstance(action_sets[dag_id], OptimalActionSet)
        }
        certificate = find_common_optimal_command(ids, certified_sets)
        return () if certificate is None else (certificate.command,)

    def actions(dag_id: DagId) -> tuple[CanonicalAction, ...]:
        value = action_sets[dag_id]
        return value.actions if isinstance(value, OptimalActionSet) else tuple(value)

    output: list[CanonicalAction] = []
    for candidate in actions(ids[0]):
        if all(
            any(
                actions_close(
                    candidate, other, atol=action_atol, rtol=action_rtol
                )
                for other in actions(dag_id)
            )
            for dag_id in ids[1:]
        ) and not any(
            actions_close(candidate, current, atol=action_atol, rtol=action_rtol)
            for current in output
        ):
            output.append(candidate)
    return tuple(sorted(output))


def parent_signature(candidate: CandidateDAG, node: int) -> ParentSignature:
    return candidate.parent_sets[int(node)]


def singleton_query_partition(
    model_ids: Iterable[DagId],
    candidates: Mapping[DagId, CandidateDAG],
    node: int,
) -> tuple[frozenset[DagId], ...]:
    groups: dict[ParentSignature, set[DagId]] = {}
    for dag_id in model_ids:
        signature = parent_signature(candidates[dag_id], node)
        groups.setdefault(signature, set()).add(dag_id)
    return tuple(
        sorted(
            (frozenset(group) for group in groups.values()),
            key=lambda group: tuple(sorted(group)),
        )
    )


@dataclass(frozen=True)
class QueryPartitionAudit:
    passed: bool
    precision_identity_max_error: float
    same_signature_max_error: float
    different_signature_min_separation: float
    failures: tuple[str, ...]


def audit_singleton_query_signatures(
    scms: Mapping[DagId, CandidateSCM],
    *,
    response_atol: float = DEFAULT_MOMENT_ATOL,
    response_rtol: float = DEFAULT_MOMENT_RTOL,
) -> QueryPartitionAudit:
    """Numerically audit parent-set signatures against total-effect responses."""
    ids = tuple(sorted(scms))
    if not ids:
        raise ValueError("at least one candidate SCM is required")
    n = len(scms[ids[0]].candidate.parent_sets)
    _, source_covariance = scms[ids[0]].observational_moments()
    source_precision = np.linalg.solve(source_covariance, np.eye(n))
    precision_identity_max = 0.0
    same_max = 0.0
    different_min = inf
    failures: list[str] = []
    for node in range(n):
        for left_index, left_id in enumerate(ids):
            left = scms[left_id]
            left_response = left.singleton_shift_response(node)
            identity_error = float(
                np.max(
                    np.abs(
                        source_precision @ left_response
                        - left.system_matrix[node] / left.noise_variances[node]
                    )
                )
            )
            precision_identity_max = max(precision_identity_max, identity_error)
            identity_scale = max(
                1.0,
                float(
                    np.max(
                        np.abs(
                            left.system_matrix[node] / left.noise_variances[node]
                        )
                    )
                ),
            )
            if identity_error > response_atol + response_rtol * identity_scale:
                failures.append(
                    f"node={node}:precision-times-response identity fails for "
                    f"{left_id}"
                )
            for right_id in ids[left_index + 1 :]:
                right = scms[right_id]
                right_response = right.singleton_shift_response(node)
                distance = float(np.max(np.abs(left_response - right_response)))
                same = (
                    left.candidate.parent_sets[node]
                    == right.candidate.parent_sets[node]
                )
                if same:
                    same_max = max(same_max, distance)
                    if not np.allclose(
                        left_response,
                        right_response,
                        atol=response_atol,
                        rtol=response_rtol,
                    ):
                        failures.append(
                            f"node={node}:same-parent responses differ for "
                            f"{left_id} and {right_id}"
                        )
                else:
                    different_min = min(different_min, distance)
                    tolerance = response_atol + response_rtol * max(
                        1.0,
                        float(np.max(np.abs(left_response))),
                        float(np.max(np.abs(right_response))),
                    )
                    if distance <= tolerance:
                        failures.append(
                            f"node={node}:different-parent responses coincide for "
                            f"{left_id} and {right_id}"
                        )
    return QueryPartitionAudit(
        passed=not failures,
        precision_identity_max_error=precision_identity_max,
        same_signature_max_error=same_max,
        different_signature_min_separation=(
            different_min if np.isfinite(different_min) else float("nan")
        ),
        failures=tuple(failures),
    )


@dataclass(frozen=True)
class BellmanState:
    model_ids: frozenset[DagId]
    value: int
    terminal_reason: str | None
    common_actions: tuple[ActionVector, ...]
    common_command_certificate: CommonCommandCertificate | None
    optimal_queries: tuple[int, ...]
    partitions: Mapping[int, tuple[frozenset[DagId], ...]]
    worst_children: Mapping[int, tuple[frozenset[DagId], ...]]


@dataclass(frozen=True)
class BellmanResult:
    objective: str
    initial_model_ids: frozenset[DagId]
    complexity: int
    states: Mapping[frozenset[DagId], BellmanState]

    @property
    def initial_state(self) -> BellmanState:
        return self.states[self.initial_model_ids]


def _bellman_solve(
    *,
    candidates: Mapping[DagId, CandidateDAG],
    initial_model_ids: Iterable[DagId] | None,
    action_sets: Mapping[DagId, OptimalActionSet | Sequence[CanonicalAction]] | None,
) -> BellmanResult:
    objective = "matching" if action_sets is not None else "dag"
    initial = frozenset(candidates if initial_model_ids is None else initial_model_ids)
    if not initial:
        raise ValueError("initial model set must be nonempty")
    if not initial <= frozenset(candidates):
        raise ValueError("initial model set contains a DAG outside the supplied MEC")
    if action_sets is not None and not initial <= frozenset(action_sets):
        raise ValueError("an initial candidate is missing its optimal action set")
    if action_sets is not None and any(
        not isinstance(action_sets[dag_id], OptimalActionSet) for dag_id in initial
    ):
        raise TypeError(
            "matching Bellman stopping requires lifted OptimalActionSet values"
        )
    n = len(next(iter(candidates.values())).parent_sets)
    cache: dict[frozenset[DagId], BellmanState] = {}

    def solve(state_ids: frozenset[DagId]) -> BellmanState:
        cached = cache.get(state_ids)
        if cached is not None:
            return cached
        certificate = (
            find_common_optimal_command(
                state_ids,
                {
                    dag_id: action_sets[dag_id]
                    for dag_id in state_ids
                    if isinstance(action_sets[dag_id], OptimalActionSet)
                },
            )
            if action_sets is not None
            else None
        )
        if certificate is not None:
            state = BellmanState(
                model_ids=state_ids,
                value=0,
                terminal_reason="common_optimal_action",
                common_actions=(certificate.command,),
                common_command_certificate=certificate,
                optimal_queries=(),
                partitions=MappingProxyType({}),
                worst_children=MappingProxyType({}),
            )
            cache[state_ids] = state
            return state
        if action_sets is None and len(state_ids) == 1:
            state = BellmanState(
                model_ids=state_ids,
                value=0,
                terminal_reason="singleton_dag",
                common_actions=(),
                common_command_certificate=None,
                optimal_queries=(),
                partitions=MappingProxyType({}),
                worst_children=MappingProxyType({}),
            )
            cache[state_ids] = state
            return state

        evaluated: dict[
            int, tuple[int, tuple[frozenset[DagId], ...], tuple[frozenset[DagId], ...]]
        ] = {}
        for node in range(n):
            partition = singleton_query_partition(state_ids, candidates, node)
            if len(partition) < 2:
                continue
            child_states = tuple(solve(child) for child in partition)
            worst_value = max(child.value for child in child_states)
            worst = tuple(
                child.model_ids for child in child_states if child.value == worst_value
            )
            evaluated[node] = (1 + worst_value, partition, worst)
        if not evaluated:
            raise RuntimeError(
                f"nonterminal {objective} state has no informative singleton query"
            )
        optimum = min(item[0] for item in evaluated.values())
        queries = tuple(node for node, item in evaluated.items() if item[0] == optimum)
        state = BellmanState(
            model_ids=state_ids,
            value=optimum,
            terminal_reason=None,
            common_actions=(),
            common_command_certificate=None,
            optimal_queries=queries,
            partitions=MappingProxyType(
                {node: evaluated[node][1] for node in sorted(evaluated)}
            ),
            worst_children=MappingProxyType(
                {node: evaluated[node][2] for node in sorted(evaluated)}
            ),
        )
        cache[state_ids] = state
        return state

    root = solve(initial)
    return BellmanResult(
        objective=objective,
        initial_model_ids=initial,
        complexity=root.value,
        states=MappingProxyType(dict(cache)),
    )


def solve_matching_complexity(
    *,
    candidates: Mapping[DagId, CandidateDAG],
    action_sets: Mapping[DagId, OptimalActionSet | Sequence[CanonicalAction]],
    initial_model_ids: Iterable[DagId] | None = None,
) -> BellmanResult:
    return _bellman_solve(
        candidates=candidates,
        initial_model_ids=initial_model_ids,
        action_sets=action_sets,
    )


def solve_dag_complexity(
    *,
    candidates: Mapping[DagId, CandidateDAG],
    initial_model_ids: Iterable[DagId] | None = None,
) -> BellmanResult:
    return _bellman_solve(
        candidates=candidates,
        initial_model_ids=initial_model_ids,
        action_sets=None,
    )


def verify_matching_leq_dag_for_all_subsets(
    *,
    candidates: Mapping[DagId, CandidateDAG],
    action_sets: Mapping[DagId, OptimalActionSet | Sequence[CanonicalAction]],
    max_candidates: int = 16,
) -> tuple[int, int]:
    """Exhaust every nonempty candidate subset and assert F_match <= F_DAG."""
    ids = tuple(sorted(candidates))
    if len(ids) > max_candidates:
        raise ValueError("too many candidate subsets for exhaustive verification")
    checked = 0
    maximum_gap = 0
    for size in range(1, len(ids) + 1):
        for subset_tuple in combinations(ids, size):
            subset = frozenset(subset_tuple)
            matching = solve_matching_complexity(
                candidates=candidates,
                action_sets=action_sets,
                initial_model_ids=subset,
            ).complexity
            dag = solve_dag_complexity(
                candidates=candidates, initial_model_ids=subset
            ).complexity
            if matching > dag:
                raise AssertionError(
                    f"F_match={matching} exceeds F_DAG={dag} for {subset_tuple}"
                )
            checked += 1
            maximum_gap = max(maximum_gap, dag - matching)
    return checked, maximum_gap


def tree_root_partition(
    tree: nx.Graph,
    roots_by_id: Mapping[DagId, int],
    model_ids: Iterable[DagId],
    query_node: int,
) -> tuple[frozenset[DagId], ...]:
    """Specialized collider-free-tree root/component query partition."""
    if not nx.is_tree(tree):
        raise ValueError("specialized root partition requires a tree")
    selected = frozenset(model_ids)
    groups: list[frozenset[DagId]] = []
    at_query = frozenset(
        dag_id for dag_id in selected if roots_by_id[dag_id] == query_node
    )
    if at_query:
        groups.append(at_query)
    reduced = nx.Graph(tree)
    reduced.remove_node(query_node)
    for component_nodes in nx.connected_components(reduced):
        group = frozenset(
            dag_id
            for dag_id in selected
            if roots_by_id[dag_id] in component_nodes
        )
        if group:
            groups.append(group)
    return tuple(sorted(groups, key=lambda group: tuple(sorted(group))))


def infer_outward_tree_root(candidate: CandidateDAG) -> int:
    roots = [
        node for node, parents in enumerate(candidate.parent_sets) if not parents
    ]
    if len(roots) != 1 or any(
        len(parents) != (0 if node == roots[0] else 1)
        for node, parents in enumerate(candidate.parent_sets)
    ):
        raise ValueError("candidate is not a collider-free outward-oriented tree")
    return roots[0]


def solve_specialized_tree_root_complexity(
    *,
    tree: nx.Graph,
    candidates: Mapping[DagId, CandidateDAG],
    initial_model_ids: Iterable[DagId] | None = None,
    action_sets: Mapping[DagId, OptimalActionSet | Sequence[CanonicalAction]] | None = None,
) -> int:
    """Independent root-location Bellman reference for collider-free tree MECs."""
    roots = {dag_id: infer_outward_tree_root(dag) for dag_id, dag in candidates.items()}
    initial = frozenset(candidates if initial_model_ids is None else initial_model_ids)
    if action_sets is not None and any(
        not isinstance(action_sets[dag_id], OptimalActionSet) for dag_id in initial
    ):
        raise TypeError(
            "matching tree reference requires lifted OptimalActionSet values"
        )
    cache: dict[frozenset[DagId], int] = {}

    def solve(state: frozenset[DagId]) -> int:
        if state in cache:
            return cache[state]
        if action_sets is not None and find_common_optimal_command(
            state,
            {
                dag_id: action_sets[dag_id]
                for dag_id in state
                if isinstance(action_sets[dag_id], OptimalActionSet)
            },
        ) is not None:
            cache[state] = 0
            return 0
        if action_sets is None and len(state) == 1:
            cache[state] = 0
            return 0
        values: list[int] = []
        for node in sorted(tree.nodes):
            partition = tree_root_partition(tree, roots, state, int(node))
            if len(partition) >= 2:
                values.append(1 + max(solve(child) for child in partition))
        if not values:
            raise RuntimeError("nonterminal tree-root state has no informative query")
        cache[state] = min(values)
        return cache[state]

    return solve(initial)


def candidate_map(candidates: Sequence[CandidateDAG]) -> dict[DagId, CandidateDAG]:
    return {candidate.id: candidate for candidate in candidates}


def pdag_from_undirected_graph(graph: nx.Graph) -> PartiallyDirectedGraph:
    nodes = frozenset(int(node) for node in graph.nodes)
    return PartiallyDirectedGraph(
        nodes=nodes,
        directed_edges=frozenset(),
        undirected_edges=frozenset(
            tuple(sorted((int(left), int(right)))) for left, right in graph.edges
        ),
    )


def outward_tree_dag(tree: nx.Graph, root: int) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(tree.nodes)
    for parent, child in nx.bfs_edges(tree, root):
        dag.add_edge(parent, child)
    return dag


def moments_from_parameters(
    *,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
    noise_variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(intercepts)
    system = np.eye(n) - np.asarray(coefficients, dtype=float).T
    total = np.linalg.solve(system, np.eye(n))
    mean = total @ np.asarray(intercepts, dtype=float)
    covariance = total @ np.diag(noise_variances) @ total.T
    return mean, covariance


def direct_action_kl(
    *,
    scm: CandidateSCM,
    action: ActionVector,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
) -> float:
    candidate_mean, candidate_covariance = scm.interventional_moments(
        action.shift, action.scale
    )
    return gaussian_kl_target_to_candidate(
        np.asarray(target_mean, dtype=float),
        np.asarray(target_covariance, dtype=float),
        candidate_mean,
        candidate_covariance,
    )


__all__ = [
    "BellmanResult",
    "BellmanState",
    "CandidateCommandLoss",
    "CandidateDAG",
    "CandidateSCM",
    "CanonicalAction",
    "CommandCertificationError",
    "CommonCommandCertificate",
    "DagId",
    "OptimalCommandFamily",
    "OptimalActionSet",
    "PhysicalCommand",
    "QueryPartitionAudit",
    "actions_close",
    "audit_singleton_query_signatures",
    "candidate_map",
    "common_optimal_actions",
    "compute_all_optimal_action_sets",
    "compute_optimal_action_set",
    "direct_action_kl",
    "enumerate_consistent_dags",
    "find_common_optimal_command",
    "infer_outward_tree_root",
    "intersect_optimal_command_families",
    "moments_from_parameters",
    "outward_tree_dag",
    "parent_signature",
    "pdag_from_undirected_graph",
    "reconstruct_all_candidate_scms",
    "reconstruct_candidate_scm",
    "required_pdag_colliders",
    "singleton_query_partition",
    "solve_dag_complexity",
    "solve_matching_complexity",
    "solve_specialized_tree_root_complexity",
    "tree_root_partition",
    "unshielded_colliders",
    "validate_consistent_dag",
    "verify_matching_leq_dag_for_all_subsets",
]
