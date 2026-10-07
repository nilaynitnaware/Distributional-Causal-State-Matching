"""Public exact-query methods for KL-optimal sparse matching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Literal, Protocol

import networkx as nx
import numpy as np

from environments.blackbox import ExactMomentBlackBoxEnvironment, QueryMetadata
from graphs.interventions import ShiftScaleIntervention
from graphs.pdag import PartiallyDirectedGraph
from graphs.scm import DEFAULT_BETA_MIN
from graphs.support import DEFAULT_SUPPORT_ATOL
from matching.graph_updates import update_graph_from_probe
from matching.probe_outcomes import ProbeEnvironment, ProbeOutcome
from matching.probes import ShiftProbe
from matching.types import InterventionBackground
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from oracle.sparse_shift_scale_kl import (
    LocalObjective,
    LocalObjectiveMoments,
    NodeMoments,
    SparseShiftScaleOracleResult,
    local_kl_loss,
    solve_sparse_shift_scale_kl_from_parameters,
)
from query_complexity.local_mechanism import (
    RecoveredLocalMechanism,
    orient_incident_edges_from_local_mechanism,
    recover_local_mechanism_from_additive_probe,
)
from query_complexity.public_probe import construct_public_singleton_shift_outcome

PROTOCOL_VERSION = "kl_query_complexity_v1"
Method = Literal[
    "graph_free_blackbox_dp",
    "graph_free_total_effect_dp",
    "passive_cpdag_hybrid_dp",
    "active_graph_hybrid_dp",
    "adaptive_singleton_shift_dp",
    "public_adaptive_singleton_shift_dp",
    "public_adaptive_local_mechanism_dp",
]
ProbePolicy = Literal[
    "no_probe",
    "deterministic_singleton_shift_probe",
    "existing_public_probe_policy",
    "deterministic_unresolved_singleton_shift",
]


class ExactMomentEnvironment(Protocol):
    def query(
        self, intervention: ShiftScaleIntervention, *, kind: str, background_id: str
    ): ...


class ExecutedSingletonShiftOutcomeProvider(Protocol):
    """Derive structural facts from an already charged singleton shift."""

    def outcome_from_executed_shift(
        self,
        public_candidate: ShiftScaleIntervention,
        probe: ShiftProbe,
        *,
        exact_interventional_mean: np.ndarray,
        exact_interventional_covariance: np.ndarray,
        background_id: str,
    ) -> ProbeOutcome: ...


class SingletonShiftPolicy(Protocol):
    """Pluggable public-state policy for distinct singleton targets."""

    def select_target(
        self,
        *,
        graph: PartiallyDirectedGraph,
        queried_nodes: frozenset[int],
        resolved_nodes: frozenset[int],
    ) -> int: ...


@dataclass(frozen=True)
class DeterministicUnresolvedSingletonShiftPolicy:
    """Choose the lowest-index unqueried node in an unresolved region."""

    name: str = "deterministic_unresolved_singleton_shift"

    def select_target(
        self,
        *,
        graph: PartiallyDirectedGraph,
        queried_nodes: frozenset[int],
        resolved_nodes: frozenset[int],
    ) -> int:
        incident = sorted(
            node
            for node in graph.nodes
            if node not in queried_nodes
            and node not in resolved_nodes
            and graph.undirected_neighbors_of(node)
        )
        if incident:
            return incident[0]
        unresolved = sorted(
            node
            for node in graph.nodes
            if node not in queried_nodes and node not in resolved_nodes
        )
        if unresolved:
            return unresolved[0]
        remaining = sorted(graph.nodes - queried_nodes)
        if not remaining:
            raise RuntimeError("singleton-shift policy has no unqueried target")
        return remaining[0]


@dataclass(frozen=True)
class FirstUnresolvedPolicy:
    """Choose the smallest-index unresolved, previously unqueried node."""

    name: str = "first_unresolved"

    def select_target(
        self,
        *,
        graph: PartiallyDirectedGraph,
        queried_nodes: frozenset[int],
        resolved_nodes: frozenset[int],
    ) -> int:
        candidates = sorted(graph.nodes - queried_nodes - resolved_nodes)
        if not candidates:
            raise RuntimeError("first_unresolved policy has no legal target")
        return candidates[0]


@dataclass(frozen=True)
class MaxUndirectedDegreePolicy:
    """Choose the unresolved node of greatest current undirected degree."""

    name: str = "max_undirected_degree"

    def select_target(
        self,
        *,
        graph: PartiallyDirectedGraph,
        queried_nodes: frozenset[int],
        resolved_nodes: frozenset[int],
    ) -> int:
        candidates = graph.nodes - queried_nodes - resolved_nodes
        if not candidates:
            raise RuntimeError("max_undirected_degree policy has no legal target")
        return min(
            candidates,
            key=lambda node: (-len(graph.undirected_neighbors_of(node)), node),
        )


@dataclass(frozen=True)
class SingletonShiftObservation:
    """Public dual-use record for one exact singleton-shift environment."""

    target_node: int
    shift_value: float
    background_id: str
    exact_interventional_mean: np.ndarray
    exact_interventional_covariance: np.ndarray
    query_metadata: QueryMetadata
    total_effect_column: np.ndarray
    structural_probe_outcome: ProbeOutcome
    provenance: tuple[str, ...]
    recovered_local_mechanism: RecoveredLocalMechanism | None = None

    def __post_init__(self) -> None:
        if self.shift_value == 0.0 or not np.isfinite(self.shift_value):
            raise ValueError("shift_value must be finite and nonzero")
        if self.structural_probe_outcome.targets != frozenset({self.target_node}):
            raise ValueError(
                "structural outcome target disagrees with singleton target"
            )
        if self.query_metadata.background_id != self.background_id:
            raise ValueError("query metadata background disagrees with observation")
        mean = np.array(self.exact_interventional_mean, dtype=float, copy=True)
        covariance = np.array(
            self.exact_interventional_covariance, dtype=float, copy=True
        )
        column = np.array(self.total_effect_column, dtype=float, copy=True)
        if mean.ndim != 1 or covariance.shape != (mean.size, mean.size):
            raise ValueError("exact moment dimensions must agree")
        if column.shape != mean.shape:
            raise ValueError("total-effect column has the wrong shape")
        if not all(np.all(np.isfinite(x)) for x in (mean, covariance, column)):
            raise ValueError("dual-use observation arrays must be finite")
        mean.setflags(write=False)
        covariance.setflags(write=False)
        column.setflags(write=False)
        object.__setattr__(self, "exact_interventional_mean", mean)
        object.__setattr__(self, "exact_interventional_covariance", covariance)
        object.__setattr__(self, "total_effect_column", column)

    @property
    def executed_environment_query(self) -> bool:
        return self.query_metadata.executed_environment_query

    @property
    def cache_hit(self) -> bool:
        return self.query_metadata.cache_hit


@dataclass(frozen=True)
class GraphUpdateRecord:
    """Public graph/progress provenance for one dual-use query."""

    query_index: int
    target_node: int
    directed_edges_before: int
    directed_edges_after: int
    undirected_edges_before: int
    undirected_edges_after: int
    orientations_gained: int
    newly_resolved_nodes: tuple[int, ...]
    graph_changed: bool
    orientations_from_local_mechanism: int = 0
    orientations_from_meek: int = 0


@dataclass(frozen=True)
class Quadratic:
    """Node-local KL change in natural parameters ``(a, d=s-1)``."""

    A: float
    B: float
    C: float
    D: float
    E: float

    def value(self, shift: float, dscale: float) -> float:
        return float(
            self.A * shift**2
            + self.B * shift
            + self.C * dscale**2
            + self.D * dscale
            + self.E * shift * dscale
        )


@dataclass(frozen=True)
class TotalEffectReconstruction:
    """Public mechanism identified from singleton shift mean responses."""

    shift_magnitude: float
    total_effect_matrix: np.ndarray
    system_matrix: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    noise_covariance: np.ndarray
    noise_variances: np.ndarray
    directed_edges: tuple[tuple[int, int], ...]
    condition_number: float
    coefficient_zero_tolerance: float
    max_mean_reconstruction_error: float
    max_covariance_reconstruction_error: float
    max_noise_offdiagonal: float
    identification: str = "singleton_shift_total_effects"


@dataclass(frozen=True)
class QueryCounts:
    algorithm_environment_calls: int
    executed_environment_queries: int
    cache_hits: int
    baseline_queries: int
    graph_probe_queries: int
    blackbox_reconstruction_queries: int
    total_effect_reconstruction_queries: int
    singleton_shift_queries: int = 0
    singleton_shift_used_for_graph_probe: int = 0
    singleton_shift_used_for_total_effect: int = 0
    singleton_shift_used_for_local_mechanism: int = 0
    singleton_shift_used_for_incident_orientation: int = 0

    @property
    def total_query_count(self) -> int:
        return self.executed_environment_queries


@dataclass(frozen=True)
class QueryMatchingResult:
    method: Method
    protocol_version: str
    budget_B: int
    experimental_sparsity_S: int
    shift: np.ndarray
    scale: np.ndarray
    support: tuple[str, ...]
    final_kl: float
    initial_kl: float
    analytic_nodes_resolved_initially: int
    analytic_nodes_resolved_after_probing: int
    experimentally_reconstructed_nodes: int
    counts: QueryCounts
    dp_result: SparseShiftScaleOracleResult
    final_graph: PartiallyDirectedGraph | None
    total_effect_reconstruction: TotalEffectReconstruction | None
    runtime_seconds: float
    hidden_information_access: str = "none_public_algorithm"
    singleton_shift_observations: tuple[SingletonShiftObservation, ...] = ()
    graph_history: tuple[GraphUpdateRecord, ...] = ()
    queried_nodes: tuple[int, ...] = ()
    structural_orientations_gained: int = 0
    all_parent_sets_resolved_at_query: int | None = None
    termination_route: str | None = None
    singleton_shift_policy: str | None = None
    query_fraction: float = 0.0
    queried_node_count: int = 0
    initially_resolved_nodes: tuple[int, ...] = ()
    directly_recovered_nodes: tuple[int, ...] = ()
    graph_resolved_without_query_nodes: tuple[int, ...] = ()
    orientations_from_local_mechanism: int = 0
    orientations_from_meek: int = 0
    termination_reason: str | None = None
    certified_K_B_star: float | None = None
    kl_gap: float | None = None

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.pop("dp_result")
        return result


def parent_set_is_resolved(graph: PartiallyDirectedGraph, node: int) -> bool:
    """Certify a parent set only when no incident edge remains undirected."""
    if not isinstance(graph, PartiallyDirectedGraph):
        raise TypeError("graph must be a PartiallyDirectedGraph.")
    return not graph.undirected_neighbors_of(node)


def _isolated(n: int, node: int, shift: float, dscale: float) -> ShiftScaleIntervention:
    shifts = np.zeros(n)
    scales = np.ones(n)
    shifts[node] = shift
    scales[node] = 1.0 + dscale
    return ShiftScaleIntervention(shifts, scales)


def _query_objective(
    environment: ExactMomentEnvironment,
    intervention: ShiftScaleIntervention,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    background_id: str,
) -> tuple[float, bool]:
    response = environment.query(
        intervention, kind="candidate", background_id=background_id
    )
    value = gaussian_kl_target_to_candidate(
        target_mean, target_covariance, response.mean, response.covariance
    )
    return float(value), bool(response.metadata.executed_environment_query)


def reconstruct_node_quadratic(
    *,
    node: int,
    n_nodes: int,
    environment: ExactMomentEnvironment,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    baseline_kl: float,
    shift_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
    background_id: str,
) -> tuple[Quadratic, int, int]:
    """Recover one quadratic with the deterministic five-point design."""
    h_shift = 0.5 * min(-shift_bounds[0], shift_bounds[1])
    d_bounds = (scale_bounds[0] - 1.0, scale_bounds[1] - 1.0)
    h_scale = 0.5 * min(-d_bounds[0], d_bounds[1])
    if h_shift <= 0.0 or h_scale <= 0.0:
        raise ValueError("shift/scale domains must contain identity in their interior.")
    points = (
        (h_shift, 0.0),
        (-h_shift, 0.0),
        (0.0, h_scale),
        (0.0, -h_scale),
        (h_shift, h_scale),
    )
    values: dict[tuple[float, float], float] = {}
    executed = 0
    for point in points:
        value, did_execute = _query_objective(
            environment,
            _isolated(n_nodes, node, *point),
            target_mean,
            target_covariance,
            background_id,
        )
        values[point] = value - baseline_kl
        executed += int(did_execute)
    fp, fm = values[(h_shift, 0.0)], values[(-h_shift, 0.0)]
    gp, gm = values[(0.0, h_scale)], values[(0.0, -h_scale)]
    A = (fp + fm) / (2.0 * h_shift**2)
    B = (fp - fm) / (2.0 * h_shift)
    C = (gp + gm) / (2.0 * h_scale**2)
    D = (gp - gm) / (2.0 * h_scale)
    E = (
        values[(h_shift, h_scale)]
        - A * h_shift**2
        - B * h_shift
        - C * h_scale**2
        - D * h_scale
    ) / (h_shift * h_scale)
    return Quadratic(A, B, C, D, E), len(points), executed


def _quadratic_to_local_objective(
    node: int,
    quadratic: Quadratic,
    shift_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
    degeneracy_atol: float,
) -> LocalObjectiveMoments:
    """Translate an identified objective without inventing topology metadata."""
    if quadratic.A <= 0.0:
        raise RuntimeError(f"node {node} reconstructed nonpositive shift curvature")
    noise_variance = 1.0 / (2.0 * quadratic.A)
    mean_z = quadratic.E * noise_variance
    mean_y = mean_z - quadratic.B * noise_variance
    second_z = max(0.0, 2.0 * quadratic.C * noise_variance)
    second_yz = second_z - quadratic.D * noise_variance
    variance_z = max(0.0, second_z - mean_z**2)
    second_y = 1.0 + max(
        abs(mean_y) ** 2,
        max(abs(x) for x in shift_bounds) ** 2,
        max(abs(x) for x in scale_bounds) ** 2 * second_z,
    )
    signal = max(abs(quadratic.C), abs(quadratic.D), abs(quadratic.E))
    scale_eligible = signal > 256.0 * np.finfo(float).eps * max(
        1.0, abs(quadratic.A), abs(quadratic.B), signal
    )
    eigenvalues = np.linalg.eigvalsh(np.array([[1.0, mean_z], [mean_z, second_z]]))
    positive = eigenvalues[eigenvalues > degeneracy_atol * max(1.0, eigenvalues[-1])]
    return LocalObjectiveMoments(
        node=node,
        noise_variance=noise_variance,
        mean_y=mean_y,
        mean_z=mean_z,
        second_y=second_y,
        second_z=second_z,
        second_yz=second_yz,
        variance_y=max(0.0, second_y - mean_y**2),
        variance_z=variance_z,
        covariance_yz=second_yz - mean_y * mean_z,
        scale_eligible=scale_eligible,
        hessian_eigenvalues=(float(eigenvalues[0]), float(eigenvalues[1])),
        hessian_condition_number=(
            float(eigenvalues[-1] / positive[0]) if positive.size == 2 else None
        ),
        numerically_degenerate=not scale_eligible,
        degeneracy_reason=(
            None if scale_eligible else "objective_identified_scale_noop"
        ),
    )


def analytic_node_moment(
    *,
    node: int,
    parents: frozenset[int],
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    degeneracy_atol: float = 1e-14,
) -> NodeMoments:
    """Recover one mechanism from public observational moments and known parents."""
    n = source_mean.size
    beta = np.zeros(n)
    ordered = tuple(sorted(parents))
    if ordered:
        pp = source_covariance[np.ix_(ordered, ordered)]
        pi = source_covariance[np.array(ordered), node]
        beta[np.array(ordered)] = np.linalg.solve(pp, pi)
    alpha = float(source_mean[node] - beta @ source_mean)
    noise = float(
        source_covariance[node, node]
        - source_covariance[node, np.array(ordered)] @ beta[np.array(ordered)]
        if ordered
        else source_covariance[node, node]
    )
    if noise <= 0.0:
        raise RuntimeError("public conditional residual variance is not positive")
    second = target_covariance + np.outer(target_mean, target_mean)
    mean_y = float(target_mean[node] - alpha)
    mean_z = float(beta @ target_mean)
    second_y = float(target_covariance[node, node] + mean_y**2)
    second_z = float(beta @ second @ beta)
    second_yz = float(beta @ (target_covariance[:, node] + target_mean * mean_y))
    variance_z = max(0.0, second_z - mean_z**2)
    eig = np.linalg.eigvalsh(np.array([[1.0, mean_z], [mean_z, second_z]]))
    positive = eig[eig > degeneracy_atol * max(1.0, eig[-1])]
    root = not ordered
    return NodeMoments(
        node=node,
        parents=ordered,
        source_intercept=alpha,
        noise_variance=noise,
        mean_y=mean_y,
        mean_z=mean_z,
        second_y=second_y,
        second_z=second_z,
        second_yz=second_yz,
        variance_y=float(target_covariance[node, node]),
        variance_z=variance_z,
        covariance_yz=second_yz - mean_y * mean_z,
        coefficient_norm=float(np.linalg.norm(beta)),
        root=root,
        scale_eligible=not root and second_z != 0.0,
        hessian_eigenvalues=(float(eig[0]), float(eig[1])),
        hessian_condition_number=(
            float(eig[-1] / positive[0]) if positive.size == 2 else None
        ),
        numerically_degenerate=root
        or variance_z <= degeneracy_atol * max(1.0, second_z),
        degeneracy_reason="zero_parental_predictor" if root else None,
    )


def recover_mechanism_from_total_effects(
    *,
    total_effect_matrix: np.ndarray,
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    shift_magnitude: float,
) -> TotalEffectReconstruction:
    """Recover repository-oriented structural parameters from public responses."""
    n = source_mean.size
    if total_effect_matrix.shape != (n, n):
        raise ValueError("total_effect_matrix has the wrong shape")
    if not np.all(np.isfinite(total_effect_matrix)):
        raise ValueError("total_effect_matrix must be finite")
    condition = float(np.linalg.cond(total_effect_matrix))
    if not np.isfinite(condition):
        raise ValueError("total-effect matrix is singular")
    try:
        system = np.linalg.solve(total_effect_matrix, np.eye(n))
    except np.linalg.LinAlgError as error:
        raise ValueError("total-effect matrix is not invertible") from error
    validation_tolerance = (
        256.0 * np.finfo(float).eps * max(1.0, condition, float(np.max(np.abs(system))))
    )
    coefficient_zero_tolerance = DEFAULT_BETA_MIN
    diagonal_error = float(np.max(np.abs(np.diag(system) - 1.0)))
    if diagonal_error > validation_tolerance:
        raise RuntimeError("reconstructed structural system has non-unit diagonal")
    coefficients = (np.eye(n) - system).T
    if float(np.max(np.abs(np.diag(coefficients)))) > validation_tolerance:
        raise RuntimeError("reconstructed coefficient matrix has nonzero diagonal")
    coefficients = np.array(coefficients, copy=True)
    coefficients[np.abs(coefficients) < coefficient_zero_tolerance] = 0.0
    np.fill_diagonal(coefficients, 0.0)
    intercepts = system @ source_mean
    omega_raw = system @ source_covariance @ system.T
    symmetry_error = float(np.max(np.abs(omega_raw - omega_raw.T)))
    covariance_scale = max(1.0, float(np.max(np.abs(omega_raw))))
    covariance_tolerance = 1e-12 + 1e-10 * covariance_scale
    if symmetry_error > covariance_tolerance:
        raise RuntimeError("reconstructed noise covariance is not symmetric")
    omega = 0.5 * (omega_raw + omega_raw.T)
    offdiagonal = omega - np.diag(np.diag(omega))
    max_offdiagonal = float(np.max(np.abs(offdiagonal)))
    if max_offdiagonal > covariance_tolerance:
        raise RuntimeError("reconstructed invariant noise is not node-independent")
    noise_variances = np.diag(omega).copy()
    if np.any(noise_variances <= 0.0) or not np.all(np.isfinite(noise_variances)):
        raise RuntimeError("reconstructed noise variances must be finite and positive")
    reconstructed_mean = np.linalg.solve(system, intercepts)
    reconstructed_covariance = np.linalg.solve(
        system, np.diag(np.sqrt(noise_variances))
    )
    reconstructed_covariance = reconstructed_covariance @ reconstructed_covariance.T
    mean_error = float(np.max(np.abs(reconstructed_mean - source_mean)))
    covariance_error = float(
        np.max(np.abs(reconstructed_covariance - source_covariance))
    )
    if mean_error > covariance_tolerance or covariance_error > covariance_tolerance:
        raise RuntimeError("reconstructed mechanism does not reproduce source moments")
    edges = tuple(
        (parent, child)
        for parent in range(n)
        for child in range(n)
        if coefficients[parent, child] != 0.0
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("reconstructed structural support is not acyclic")
    for array in (
        total_effect_matrix,
        system,
        coefficients,
        intercepts,
        omega,
        noise_variances,
    ):
        if not np.all(np.isfinite(array)):
            raise RuntimeError("structural reconstruction produced nonfinite values")
    return TotalEffectReconstruction(
        shift_magnitude=float(shift_magnitude),
        total_effect_matrix=np.array(total_effect_matrix, copy=True),
        system_matrix=system,
        coefficients=coefficients,
        intercepts=intercepts,
        noise_covariance=omega,
        noise_variances=noise_variances,
        directed_edges=edges,
        condition_number=condition,
        coefficient_zero_tolerance=coefficient_zero_tolerance,
        max_mean_reconstruction_error=mean_error,
        max_covariance_reconstruction_error=covariance_error,
        max_noise_offdiagonal=max_offdiagonal,
    )


def identify_total_effects(
    *,
    environment: ExactMomentEnvironment,
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    shift_bounds: tuple[float, float],
    background_id: str,
) -> tuple[TotalEffectReconstruction, int, int]:
    """Identify all total-effect columns with one singleton shift per node."""
    n = source_mean.size
    shift_magnitude = 0.5 * min(-shift_bounds[0], shift_bounds[1])
    if shift_magnitude <= 0.0:
        raise ValueError("shift bounds must contain zero in their interior")
    total_effect = np.empty((n, n), dtype=float)
    executed = 0
    for node in range(n):
        intervention = _isolated(n, node, shift_magnitude, 0.0)
        response = environment.query(
            intervention, kind="candidate", background_id=background_id
        )
        mean_response = response.mean - source_mean
        response_at_target = float(mean_response[node])
        if not np.isfinite(response_at_target) or response_at_target == 0.0:
            raise RuntimeError("singleton probe has zero or nonfinite target response")
        total_effect[:, node] = mean_response / response_at_target
        executed += int(response.metadata.executed_environment_query)
    reconstruction = recover_mechanism_from_total_effects(
        total_effect_matrix=total_effect,
        source_mean=source_mean,
        source_covariance=source_covariance,
        shift_magnitude=shift_magnitude,
    )
    return reconstruction, n, executed


def _run_validated_local_dp(
    objectives: tuple[LocalObjective, ...],
    baseline: float,
    budget: int,
    shift_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
    support_atol: float,
    loss_atol: float,
    loss_rtol: float,
    degeneracy_atol: float,
    method: Method,
) -> SparseShiftScaleOracleResult:
    constant = baseline - sum(
        local_kl_loss(objective, 0.0, 1.0) for objective in objectives
    )
    n = len(objectives)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    return solve_sparse_shift_scale_kl_from_parameters(
        dag=graph,
        source_intercepts=np.zeros(n),
        coefficients=np.zeros((n, n)),
        noise_variances=np.ones(n),
        target_mean=np.zeros(n),
        target_covariance=np.eye(n),
        budget=budget,
        shift_bounds=shift_bounds,
        log_scale_bounds=(
            float(np.log(scale_bounds[0])),
            float(np.log(scale_bounds[1])),
        ),
        support_atol=support_atol,
        loss_atol=loss_atol,
        loss_rtol=loss_rtol,
        degeneracy_atol=degeneracy_atol,
        provenance={
            "query_protocol": PROTOCOL_VERSION,
            "query_method": method,
            "information_regime": (
                "idealized_oracle_assisted_structural_probe"
                if method == "adaptive_singleton_shift_dp"
                else "public_moments_and_queries_only"
            ),
            "hidden_dag_evaluation_only": method == "adaptive_singleton_shift_dp",
            "topology_metadata": "structural_or_explicitly_topology_neutral",
        },
        precomputed_node_moments=objectives,
        precomputed_kl_constant=constant,
    )


def _graph_snapshot(
    graph: PartiallyDirectedGraph,
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    return graph.directed_edges, graph.undirected_edges


def _run_adaptive_singleton_shift_dp(
    *,
    environment: ExactMomentBlackBoxEnvironment,
    method: Method,
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    budget_B: int,
    experimental_sparsity_S: int,
    shift_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
    initial_graph: PartiallyDirectedGraph,
    outcome_provider: object,
    singleton_shift_policy: SingletonShiftPolicy | None,
    public_probe_effect_atol: float,
    support_atol: float,
    loss_atol: float,
    loss_rtol: float,
    degeneracy_atol: float,
    started: float,
) -> QueryMatchingResult:
    """Run the exact dual-use adaptive singleton-shift method."""
    n = source_mean.size
    graph_state = initial_graph.copy()
    if graph_state.nodes != frozenset(range(n)):
        raise ValueError("initial graph nodes must be indexed 0, ..., n - 1")
    initial_resolved = {
        node for node in range(n) if parent_set_is_resolved(graph_state, node)
    }
    resolved = set(initial_resolved)
    policy = singleton_shift_policy or DeterministicUnresolvedSingletonShiftPolicy()
    if not hasattr(policy, "select_target"):
        raise TypeError("singleton_shift_policy must implement select_target")
    if (
        method == "adaptive_singleton_shift_dp"
        and len(resolved) < n
        and not hasattr(outcome_provider, "outcome_from_executed_shift")
    ):
        raise TypeError(
            "adaptive singleton shifts require an outcome provider for an "
            "already executed shift; a separate ProbeEnvironment is not accepted"
        )

    shift_magnitude = 0.5 * min(-shift_bounds[0], shift_bounds[1])
    if shift_magnitude <= 0.0:
        raise ValueError("shift bounds must contain zero in their interior")
    baseline = gaussian_kl_target_to_candidate(
        target_mean, target_covariance, source_mean, source_covariance
    )
    ledger_before = environment.ledger.snapshot()
    identity = ShiftScaleIntervention.identity(n)
    background_id = InterventionBackground.from_intervention(identity).identifier
    observations: list[SingletonShiftObservation] = []
    history: list[GraphUpdateRecord] = []
    queried_nodes: set[int] = set()
    total_effect_columns: dict[int, np.ndarray] = {}
    reconstruction: TotalEffectReconstruction | None = None
    resolved_at_query: int | None = 0 if len(resolved) == n else None

    def graph_objectives() -> tuple[LocalObjective, ...]:
        return tuple(
            analytic_node_moment(
                node=node,
                parents=graph_state.parents_of(node),
                source_mean=source_mean,
                source_covariance=source_covariance,
                target_mean=target_mean,
                target_covariance=target_covariance,
                degeneracy_atol=degeneracy_atol,
            )
            for node in range(n)
        )

    if len(resolved) == n:
        termination_route = "graph_resolution"
        dp = _run_validated_local_dp(
            graph_objectives(),
            baseline,
            budget_B,
            shift_bounds,
            scale_bounds,
            support_atol,
            loss_atol,
            loss_rtol,
            degeneracy_atol,
            method,
        )
    else:
        dp = None
        while len(queried_nodes) < n:
            node = policy.select_target(
                graph=graph_state,
                queried_nodes=frozenset(queried_nodes),
                resolved_nodes=frozenset(resolved),
            )
            if node not in graph_state.nodes:
                raise RuntimeError("singleton-shift policy selected an unknown node")
            if node in queried_nodes:
                raise RuntimeError("singleton-shift policy repeated a target")
            queried_nodes.add(node)
            intervention = _isolated(n, node, shift_magnitude, 0.0)
            response = environment.query(
                intervention, kind="probe", background_id=background_id
            )
            mean_response = response.mean - source_mean
            response_at_target = float(mean_response[node])
            if not np.isfinite(response_at_target) or response_at_target == 0.0:
                raise RuntimeError(
                    "singleton probe has zero or nonfinite target response"
                )
            column = mean_response / response_at_target
            total_effect_columns[node] = column
            increment = np.zeros(n)
            increment[node] = shift_magnitude
            probe = ShiftProbe(
                increment,
                reason=f"dual-use-singleton-shift-node-{node}",
            )
            if len(probe.targets) > experimental_sparsity_S:
                raise RuntimeError("singleton shift violates experimental sparsity S")
            if method == "public_adaptive_singleton_shift_dp":
                outcome = construct_public_singleton_shift_outcome(
                    source_mean=source_mean,
                    source_covariance=source_covariance,
                    exact_interventional_mean=response.mean,
                    exact_interventional_covariance=response.covariance,
                    probe=probe,
                    current_graph=graph_state,
                    assumption="conservative",
                    effect_atol=public_probe_effect_atol,
                )
                structural_provenance = "public_nonzero_response_orientations_only"
            else:
                outcome = outcome_provider.outcome_from_executed_shift(
                    identity,
                    probe,
                    exact_interventional_mean=response.mean,
                    exact_interventional_covariance=response.covariance,
                    background_id=background_id,
                )
                structural_provenance = "idealized_oracle_assisted_orientations"
            observation = SingletonShiftObservation(
                target_node=node,
                shift_value=shift_magnitude,
                background_id=background_id,
                exact_interventional_mean=response.mean,
                exact_interventional_covariance=response.covariance,
                query_metadata=response.metadata,
                total_effect_column=column,
                structural_probe_outcome=outcome,
                provenance=(
                    "one_exact_singleton_shift_environment",
                    "graph_orientation_role",
                    "total_effect_column_role",
                    "no_hidden_structure_in_public_record",
                    structural_provenance,
                ),
            )
            observations.append(observation)
            before = _graph_snapshot(graph_state)
            updated = update_graph_from_probe(graph_state, outcome)
            after = _graph_snapshot(updated)
            newly_resolved = {
                candidate
                for candidate in range(n)
                if parent_set_is_resolved(updated, candidate)
            } - resolved
            orientations_gained = len(after[0] - before[0])
            history.append(
                GraphUpdateRecord(
                    query_index=len(observations),
                    target_node=node,
                    directed_edges_before=len(before[0]),
                    directed_edges_after=len(after[0]),
                    undirected_edges_before=len(before[1]),
                    undirected_edges_after=len(after[1]),
                    orientations_gained=orientations_gained,
                    newly_resolved_nodes=tuple(sorted(newly_resolved)),
                    graph_changed=after != before,
                )
            )
            graph_state = updated
            resolved = {
                candidate
                for candidate in range(n)
                if parent_set_is_resolved(graph_state, candidate)
            }
            if len(resolved) == n:
                resolved_at_query = len(observations)
                termination_route = "graph_resolution"
                dp = _run_validated_local_dp(
                    graph_objectives(),
                    baseline,
                    budget_B,
                    shift_bounds,
                    scale_bounds,
                    support_atol,
                    loss_atol,
                    loss_rtol,
                    degeneracy_atol,
                    method,
                )
                break

        if dp is None:
            if queried_nodes != set(range(n)) or len(total_effect_columns) != n:
                raise RuntimeError(
                    "full total-effect fallback lacks all singleton columns"
                )
            total_effect = np.column_stack(
                [total_effect_columns[node] for node in range(n)]
            )
            reconstruction = recover_mechanism_from_total_effects(
                total_effect_matrix=total_effect,
                source_mean=source_mean,
                source_covariance=source_covariance,
                shift_magnitude=shift_magnitude,
            )
            reconstructed_graph = nx.DiGraph()
            reconstructed_graph.add_nodes_from(range(n))
            reconstructed_graph.add_edges_from(reconstruction.directed_edges)
            dp = solve_sparse_shift_scale_kl_from_parameters(
                dag=reconstructed_graph,
                source_intercepts=reconstruction.intercepts,
                coefficients=reconstruction.coefficients,
                noise_variances=reconstruction.noise_variances,
                target_mean=target_mean,
                target_covariance=target_covariance,
                budget=budget_B,
                shift_bounds=shift_bounds,
                log_scale_bounds=(
                    float(np.log(scale_bounds[0])),
                    float(np.log(scale_bounds[1])),
                ),
                support_atol=support_atol,
                loss_atol=loss_atol,
                loss_rtol=loss_rtol,
                degeneracy_atol=degeneracy_atol,
                provenance={
                    "query_protocol": PROTOCOL_VERSION,
                    "query_method": method,
                    "information_regime": (
                        "public_moments_conservative_orientation"
                        if method == "public_adaptive_singleton_shift_dp"
                        else "idealized_oracle_assisted_structural_probe"
                    ),
                    "termination_route": "full_total_effect",
                    "hidden_dag_evaluation_only": method
                    == "adaptive_singleton_shift_dp",
                    "structure_experimentally_identified": True,
                },
            )
            termination_route = "full_total_effect"

    ledger_after = environment.ledger.snapshot()
    api_delta = int(ledger_after["algorithm_environment_calls"]) - int(
        ledger_before["algorithm_environment_calls"]
    )
    cache_delta = int(ledger_after["cache_hits"]) - int(ledger_before["cache_hits"])
    executed_delta = int(ledger_after["executed_environment_queries"]) - int(
        ledger_before["executed_environment_queries"]
    )
    role_count = len(observations)
    counts = QueryCounts(
        algorithm_environment_calls=api_delta,
        executed_environment_queries=executed_delta,
        cache_hits=cache_delta,
        baseline_queries=0,
        graph_probe_queries=role_count,
        blackbox_reconstruction_queries=0,
        total_effect_reconstruction_queries=role_count,
        singleton_shift_queries=executed_delta,
        singleton_shift_used_for_graph_probe=role_count,
        singleton_shift_used_for_total_effect=role_count,
    )
    return QueryMatchingResult(
        method=method,
        protocol_version=PROTOCOL_VERSION,
        budget_B=budget_B,
        experimental_sparsity_S=experimental_sparsity_S,
        shift=dp.budgeted_shift,
        scale=dp.budgeted_scale,
        support=dp.budgeted_support,
        final_kl=dp.raw_K_B_star,
        initial_kl=float(baseline),
        analytic_nodes_resolved_initially=len(initial_resolved),
        analytic_nodes_resolved_after_probing=len(resolved - initial_resolved),
        experimentally_reconstructed_nodes=(
            n if termination_route == "full_total_effect" else 0
        ),
        counts=counts,
        dp_result=dp,
        final_graph=graph_state,
        total_effect_reconstruction=reconstruction,
        runtime_seconds=perf_counter() - started,
        hidden_information_access=(
            "idealized_hidden_dag_orientation_payload"
            if method == "adaptive_singleton_shift_dp"
            else "none_public_algorithm"
        ),
        singleton_shift_observations=tuple(observations),
        graph_history=tuple(history),
        queried_nodes=tuple(observation.target_node for observation in observations),
        structural_orientations_gained=sum(
            record.orientations_gained for record in history
        ),
        all_parent_sets_resolved_at_query=resolved_at_query,
        termination_route=termination_route,
        singleton_shift_policy=getattr(policy, "name", type(policy).__name__),
    )


def _run_public_local_mechanism_dp(
    *,
    environment: ExactMomentBlackBoxEnvironment,
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    budget_B: int,
    experimental_sparsity_S: int,
    shift_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
    initial_graph: PartiallyDirectedGraph,
    singleton_shift_policy: SingletonShiftPolicy | None,
    support_atol: float,
    loss_atol: float,
    loss_rtol: float,
    degeneracy_atol: float,
    started: float,
) -> QueryMatchingResult:
    """Run the exact public adaptive local-mechanism method."""
    method: Method = "public_adaptive_local_mechanism_dp"
    n = source_mean.size
    graph_state = initial_graph.copy()
    if graph_state.nodes != frozenset(range(n)):
        raise ValueError("initial graph nodes must be indexed 0, ..., n - 1")
    if experimental_sparsity_S < 1:
        raise ValueError("experimental_sparsity_S must allow a singleton shift")
    policy = singleton_shift_policy or FirstUnresolvedPolicy()
    if not hasattr(policy, "select_target"):
        raise TypeError("singleton_shift_policy must implement select_target")

    shift_magnitude = 0.5 * min(-shift_bounds[0], shift_bounds[1])
    if shift_magnitude <= 0.0:
        raise ValueError("shift bounds must contain zero in their interior")
    baseline = gaussian_kl_target_to_candidate(
        target_mean, target_covariance, source_mean, source_covariance
    )
    ledger_before = environment.ledger.snapshot()
    background_id = InterventionBackground.from_intervention(
        ShiftScaleIntervention.identity(n)
    ).identifier

    initially_resolved = {
        node for node in range(n) if parent_set_is_resolved(graph_state, node)
    }
    objectives: dict[int, LocalObjective] = {}
    for node in sorted(initially_resolved):
        objectives[node] = analytic_node_moment(
            node=node,
            parents=graph_state.parents_of(node),
            source_mean=source_mean,
            source_covariance=source_covariance,
            target_mean=target_mean,
            target_covariance=target_covariance,
            degeneracy_atol=degeneracy_atol,
        )

    queried_nodes: set[int] = set()
    directly_recovered: set[int] = set()
    graph_resolved_without_query: set[int] = set()
    observations: list[SingletonShiftObservation] = []
    history: list[GraphUpdateRecord] = []
    direct_orientation_count = 0
    meek_orientation_count = 0

    while len(objectives) < n:
        resolved_nodes = {
            node for node in range(n) if parent_set_is_resolved(graph_state, node)
        }
        node = policy.select_target(
            graph=graph_state,
            queried_nodes=frozenset(queried_nodes),
            resolved_nodes=frozenset(resolved_nodes),
        )
        if node not in graph_state.nodes:
            raise RuntimeError("singleton-shift policy selected an unknown node")
        if node in queried_nodes or node in resolved_nodes:
            raise RuntimeError("singleton-shift policy selected an ineligible node")
        queried_nodes.add(node)
        response = environment.query(
            _isolated(n, node, shift_magnitude, 0.0),
            kind="probe",
            background_id=background_id,
        )
        recovered = recover_local_mechanism_from_additive_probe(
            node=node,
            source_mean=source_mean,
            source_covariance=source_covariance,
            exact_shifted_mean=response.mean,
            coefficient_support_tolerance=DEFAULT_BETA_MIN,
        )
        directly_recovered.add(node)
        objectives[node] = analytic_node_moment(
            node=node,
            parents=frozenset(recovered.parents),
            source_mean=source_mean,
            source_covariance=source_covariance,
            target_mean=target_mean,
            target_covariance=target_covariance,
            degeneracy_atol=degeneracy_atol,
        )

        before = _graph_snapshot(graph_state)
        orientation = orient_incident_edges_from_local_mechanism(graph_state, recovered)
        graph_state = orientation.graph
        after = _graph_snapshot(graph_state)
        direct_orientation_count += len(orientation.direct_orientations)
        meek_orientation_count += len(orientation.meek_orientations)
        now_resolved = {
            candidate
            for candidate in range(n)
            if parent_set_is_resolved(graph_state, candidate)
        }
        newly_resolved = now_resolved - set(objectives)
        for candidate in sorted(newly_resolved):
            objectives[candidate] = analytic_node_moment(
                node=candidate,
                parents=graph_state.parents_of(candidate),
                source_mean=source_mean,
                source_covariance=source_covariance,
                target_mean=target_mean,
                target_covariance=target_covariance,
                degeneracy_atol=degeneracy_atol,
            )
            if candidate not in queried_nodes and candidate not in initially_resolved:
                graph_resolved_without_query.add(candidate)

        outcome = ProbeOutcome(
            targets=frozenset({node}),
            source_targets=frozenset({node}),
            identified_orientations=frozenset(orientation.direct_orientations),
            samples=None,
            trial_cost=1,
        )
        observations.append(
            SingletonShiftObservation(
                target_node=node,
                shift_value=shift_magnitude,
                background_id=background_id,
                exact_interventional_mean=response.mean,
                exact_interventional_covariance=response.covariance,
                query_metadata=response.metadata,
                total_effect_column=recovered.total_effect_column,
                structural_probe_outcome=outcome,
                recovered_local_mechanism=recovered,
                provenance=(
                    "one_exact_singleton_shift_environment",
                    "total_effect_column_observed",
                    "local_mechanism_recovered",
                    "incident_edges_oriented",
                    "roles_are_non_additive",
                    "public_precision_identity_no_path_faithfulness",
                ),
            )
        )
        history.append(
            GraphUpdateRecord(
                query_index=len(observations),
                target_node=node,
                directed_edges_before=len(before[0]),
                directed_edges_after=len(after[0]),
                undirected_edges_before=len(before[1]),
                undirected_edges_after=len(after[1]),
                orientations_gained=len(after[0] - before[0]),
                newly_resolved_nodes=tuple(sorted(newly_resolved)),
                graph_changed=after != before,
                orientations_from_local_mechanism=len(orientation.direct_orientations),
                orientations_from_meek=len(orientation.meek_orientations),
            )
        )

    ordered_objectives = tuple(objectives[node] for node in range(n))
    dp = _run_validated_local_dp(
        ordered_objectives,
        baseline,
        budget_B,
        shift_bounds,
        scale_bounds,
        support_atol,
        loss_atol,
        loss_rtol,
        degeneracy_atol,
        method,
    )
    ledger_after = environment.ledger.snapshot()
    api_delta = int(ledger_after["algorithm_environment_calls"]) - int(
        ledger_before["algorithm_environment_calls"]
    )
    cache_delta = int(ledger_after["cache_hits"]) - int(ledger_before["cache_hits"])
    executed_delta = int(ledger_after["executed_environment_queries"]) - int(
        ledger_before["executed_environment_queries"]
    )
    role_count = len(observations)
    counts = QueryCounts(
        algorithm_environment_calls=api_delta,
        executed_environment_queries=executed_delta,
        cache_hits=cache_delta,
        baseline_queries=0,
        graph_probe_queries=role_count,
        blackbox_reconstruction_queries=0,
        total_effect_reconstruction_queries=role_count,
        singleton_shift_queries=executed_delta,
        singleton_shift_used_for_graph_probe=role_count,
        singleton_shift_used_for_total_effect=role_count,
        singleton_shift_used_for_local_mechanism=role_count,
        singleton_shift_used_for_incident_orientation=role_count,
    )
    return QueryMatchingResult(
        method=method,
        protocol_version=PROTOCOL_VERSION,
        budget_B=budget_B,
        experimental_sparsity_S=experimental_sparsity_S,
        shift=dp.budgeted_shift,
        scale=dp.budgeted_scale,
        support=dp.budgeted_support,
        final_kl=dp.raw_K_B_star,
        initial_kl=float(baseline),
        analytic_nodes_resolved_initially=len(initially_resolved),
        analytic_nodes_resolved_after_probing=len(graph_resolved_without_query),
        experimentally_reconstructed_nodes=len(directly_recovered),
        counts=counts,
        dp_result=dp,
        final_graph=graph_state,
        total_effect_reconstruction=None,
        runtime_seconds=perf_counter() - started,
        hidden_information_access="none_public_algorithm",
        singleton_shift_observations=tuple(observations),
        graph_history=tuple(history),
        queried_nodes=tuple(item.target_node for item in observations),
        structural_orientations_gained=direct_orientation_count
        + meek_orientation_count,
        all_parent_sets_resolved_at_query=role_count,
        termination_route="local_mechanisms_complete",
        singleton_shift_policy=getattr(policy, "name", type(policy).__name__),
        query_fraction=executed_delta / n,
        queried_node_count=len(queried_nodes),
        initially_resolved_nodes=tuple(sorted(initially_resolved)),
        directly_recovered_nodes=tuple(sorted(directly_recovered)),
        graph_resolved_without_query_nodes=tuple(sorted(graph_resolved_without_query)),
        orientations_from_local_mechanism=direct_orientation_count,
        orientations_from_meek=meek_orientation_count,
        termination_reason="all_local_mechanisms_known",
        certified_K_B_star=dp.raw_K_B_star,
        kl_gap=0.0,
    )


def run_query_method(
    *,
    method: Method,
    environment: ExactMomentBlackBoxEnvironment,
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    budget_B: int,
    experimental_sparsity_S: int,
    shift_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
    initial_graph: PartiallyDirectedGraph | None = None,
    probe_policy: ProbePolicy = "no_probe",
    probe_environment: ProbeEnvironment | None = None,
    probe_shift: float = 0.5,
    max_probe_queries: int | None = None,
    singleton_shift_policy: SingletonShiftPolicy | None = None,
    public_probe_effect_atol: float = 1e-12,
    support_atol: float = DEFAULT_SUPPORT_ATOL,
    loss_atol: float = 1e-12,
    loss_rtol: float = 1e-11,
    degeneracy_atol: float = 1e-14,
) -> QueryMatchingResult:
    """Run graph-free or hybrid exact methods with explicit B/S/T accounting."""
    started = perf_counter()
    if experimental_sparsity_S < 1:
        raise ValueError("experimental_sparsity_S must be at least one")
    if max_probe_queries is not None and max_probe_queries < 0:
        raise ValueError("max_probe_queries must be nonnegative or None")
    n = target_mean.size
    if source_mean.shape != (n,) or source_covariance.shape != (n, n):
        raise ValueError("public source moments have incompatible dimensions")
    if method in {
        "passive_cpdag_hybrid_dp",
        "active_graph_hybrid_dp",
        "adaptive_singleton_shift_dp",
        "public_adaptive_singleton_shift_dp",
        "public_adaptive_local_mechanism_dp",
    } and (initial_graph is None):
        raise ValueError("hybrid methods require a public graph")
    if method == "public_adaptive_local_mechanism_dp":
        return _run_public_local_mechanism_dp(
            environment=environment,
            source_mean=source_mean,
            source_covariance=source_covariance,
            target_mean=target_mean,
            target_covariance=target_covariance,
            budget_B=budget_B,
            experimental_sparsity_S=experimental_sparsity_S,
            shift_bounds=shift_bounds,
            scale_bounds=scale_bounds,
            initial_graph=initial_graph,
            singleton_shift_policy=singleton_shift_policy,
            support_atol=support_atol,
            loss_atol=loss_atol,
            loss_rtol=loss_rtol,
            degeneracy_atol=degeneracy_atol,
            started=started,
        )
    if method in {
        "adaptive_singleton_shift_dp",
        "public_adaptive_singleton_shift_dp",
    }:
        return _run_adaptive_singleton_shift_dp(
            method=method,
            environment=environment,
            source_mean=source_mean,
            source_covariance=source_covariance,
            target_mean=target_mean,
            target_covariance=target_covariance,
            budget_B=budget_B,
            experimental_sparsity_S=experimental_sparsity_S,
            shift_bounds=shift_bounds,
            scale_bounds=scale_bounds,
            initial_graph=initial_graph,
            outcome_provider=probe_environment,
            singleton_shift_policy=singleton_shift_policy,
            public_probe_effect_atol=public_probe_effect_atol,
            support_atol=support_atol,
            loss_atol=loss_atol,
            loss_rtol=loss_rtol,
            degeneracy_atol=degeneracy_atol,
            started=started,
        )
    baseline = gaussian_kl_target_to_candidate(
        target_mean, target_covariance, source_mean, source_covariance
    )
    ledger_before = environment.ledger.snapshot()
    identity = ShiftScaleIntervention.identity(n)
    background_id = InterventionBackground.from_intervention(identity).identifier

    if method == "graph_free_total_effect_dp":
        reconstruction, _, total_effect_executed = identify_total_effects(
            environment=environment,
            source_mean=source_mean,
            source_covariance=source_covariance,
            shift_bounds=shift_bounds,
            background_id=background_id,
        )
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(reconstruction.directed_edges)
        dp = solve_sparse_shift_scale_kl_from_parameters(
            dag=graph,
            source_intercepts=reconstruction.intercepts,
            coefficients=reconstruction.coefficients,
            noise_variances=reconstruction.noise_variances,
            target_mean=target_mean,
            target_covariance=target_covariance,
            budget=budget_B,
            shift_bounds=shift_bounds,
            log_scale_bounds=(
                float(np.log(scale_bounds[0])),
                float(np.log(scale_bounds[1])),
            ),
            support_atol=support_atol,
            loss_atol=loss_atol,
            loss_rtol=loss_rtol,
            degeneracy_atol=degeneracy_atol,
            provenance={
                "query_protocol": PROTOCOL_VERSION,
                "query_method": method,
                "information_regime": "public_observational_and_singleton_shifts",
                "hidden_dag_evaluation_only": False,
                "structure_experimentally_identified": True,
                "coefficient_zero_tolerance": reconstruction.coefficient_zero_tolerance,
            },
        )
        graph_result = None
        initial_resolved_count = 0
        after_probe_count = 0
        reconstructed_nodes = 0
        graph_probe_calls = 0
        blackbox_executed = 0
    else:
        reconstruction = None
        total_effect_executed = 0
        graph_state = initial_graph.copy() if initial_graph is not None else None
        initial_resolved = {
            node
            for node in range(n)
            if graph_state is not None and parent_set_is_resolved(graph_state, node)
        }
        resolved = set(initial_resolved)
        analytic_cache: dict[int, NodeMoments] = {}
        if graph_state is not None:
            for node in resolved:
                analytic_cache[node] = analytic_node_moment(
                    node=node,
                    parents=graph_state.parents_of(node),
                    source_mean=source_mean,
                    source_covariance=source_covariance,
                    target_mean=target_mean,
                    target_covariance=target_covariance,
                    degeneracy_atol=degeneracy_atol,
                )
        graph_probe_calls = 0
        normalized_policy = (
            "deterministic_singleton_shift_probe"
            if probe_policy == "existing_public_probe_policy"
            else probe_policy
        )
        if (
            method == "active_graph_hybrid_dp"
            and normalized_policy == "deterministic_singleton_shift_probe"
        ):
            if probe_environment is None:
                raise ValueError(
                    "deterministic probe policy requires a probe environment"
                )
            probed_nodes: set[int] = set()
            while graph_state is not None:
                if (
                    max_probe_queries is not None
                    and graph_probe_calls >= max_probe_queries
                ):
                    break
                candidates = sorted(
                    node
                    for node in range(n)
                    if node not in resolved
                    and node not in probed_nodes
                    and graph_state.undirected_neighbors_of(node)
                )
                if not candidates:
                    break
                node = candidates[0]
                probed_nodes.add(node)
                increment = np.zeros(n)
                increment[node] = probe_shift
                probe = ShiftProbe(
                    increment, reason=f"deterministic-singleton-shift-node-{node}"
                )
                if len(probe.targets) > experimental_sparsity_S:
                    raise RuntimeError("probe violates experimental sparsity S")
                before_graph = _graph_snapshot(graph_state)
                outcome = probe_environment.execute_probe(identity, probe)
                graph_probe_calls += 1
                updated = update_graph_from_probe(graph_state, outcome)
                if _graph_snapshot(updated) == before_graph:
                    break
                graph_state = updated
                newly_resolved = {
                    candidate
                    for candidate in range(n)
                    if parent_set_is_resolved(graph_state, candidate)
                } - resolved
                resolved |= newly_resolved
                for candidate in newly_resolved:
                    analytic_cache[candidate] = analytic_node_moment(
                        node=candidate,
                        parents=graph_state.parents_of(candidate),
                        source_mean=source_mean,
                        source_covariance=source_covariance,
                        target_mean=target_mean,
                        target_covariance=target_covariance,
                        degeneracy_atol=degeneracy_atol,
                    )
                if len(resolved) == n:
                    break
        objectives: list[LocalObjective] = []
        blackbox_executed = 0
        for node in range(n):
            if node in resolved:
                objectives.append(analytic_cache[node])
            else:
                quadratic, _, executed = reconstruct_node_quadratic(
                    node=node,
                    n_nodes=n,
                    environment=environment,
                    target_mean=target_mean,
                    target_covariance=target_covariance,
                    baseline_kl=baseline,
                    shift_bounds=shift_bounds,
                    scale_bounds=scale_bounds,
                    background_id=background_id,
                )
                blackbox_executed += executed
                objectives.append(
                    _quadratic_to_local_objective(
                        node,
                        quadratic,
                        shift_bounds,
                        scale_bounds,
                        degeneracy_atol,
                    )
                )
        dp = _run_validated_local_dp(
            tuple(objectives),
            baseline,
            budget_B,
            shift_bounds,
            scale_bounds,
            support_atol,
            loss_atol,
            loss_rtol,
            degeneracy_atol,
            method,
        )
        graph_result = graph_state
        initial_resolved_count = len(initial_resolved)
        after_probe_count = len(resolved - initial_resolved)
        reconstructed_nodes = n - len(resolved)

    ledger_after = environment.ledger.snapshot()
    api_delta = int(ledger_after["algorithm_environment_calls"]) - int(
        ledger_before["algorithm_environment_calls"]
    )
    cache_delta = int(ledger_after["cache_hits"]) - int(ledger_before["cache_hits"])
    executed_delta = int(ledger_after["executed_environment_queries"]) - int(
        ledger_before["executed_environment_queries"]
    )
    counts = QueryCounts(
        algorithm_environment_calls=api_delta + graph_probe_calls,
        executed_environment_queries=executed_delta + graph_probe_calls,
        cache_hits=cache_delta,
        baseline_queries=0,
        graph_probe_queries=graph_probe_calls,
        blackbox_reconstruction_queries=blackbox_executed,
        total_effect_reconstruction_queries=total_effect_executed,
    )
    return QueryMatchingResult(
        method=method,
        protocol_version=PROTOCOL_VERSION,
        budget_B=budget_B,
        experimental_sparsity_S=experimental_sparsity_S,
        shift=dp.budgeted_shift,
        scale=dp.budgeted_scale,
        support=dp.budgeted_support,
        final_kl=dp.raw_K_B_star,
        initial_kl=float(baseline),
        analytic_nodes_resolved_initially=initial_resolved_count,
        analytic_nodes_resolved_after_probing=after_probe_count,
        experimentally_reconstructed_nodes=reconstructed_nodes,
        counts=counts,
        dp_result=dp,
        final_graph=graph_result,
        total_effect_reconstruction=reconstruction,
        runtime_seconds=perf_counter() - started,
    )
