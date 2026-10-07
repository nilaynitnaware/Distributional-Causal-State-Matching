"""Exact known-DAG sparse shift--scale target-to-candidate KL oracle.

This module is evaluation-only: it consumes the hidden source DAG and
structural parameters.  It does not call the public greedy matcher, issue
environment queries, or infer a graph.  The scale parameter multiplies only
the endogenous parental contribution; exogenous noise remains invariant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from time import perf_counter
from typing import Any, Literal

import networkx as nx
import numpy as np

from graphs.interventions import INTERVENTION_SEMANTICS
from graphs.scm import LinearGaussianSCM
from graphs.support import DEFAULT_SUPPORT_ATOL, validate_support_atol
from metrics.gaussian_kl import gaussian_kl_target_to_candidate

ORACLE_SCHEMA_VERSION = "sparse_shift_scale_kl_oracle_v1"
SUPPORT_SEMANTICS = "canonical_snap_then_exact_count_v1"
KL_DIRECTION = "target_to_candidate"

ActionLabel = Literal["none", "shift", "scale", "shift_scale"]
_STATE_RANK: dict[ActionLabel, int] = {
    "none": 0,
    "shift": 1,
    "scale": 2,
    "shift_scale": 3,
}


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _finite_real(name: str, value: object, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _immutable(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _loss_tolerance(left: float, right: float, atol: float, rtol: float) -> float:
    return atol + rtol * max(1.0, abs(left), abs(right))


def _loss_order(left: float, right: float, atol: float, rtol: float) -> int:
    """Return -1/0/1 when left is better/equivalent/worse than right."""
    tolerance = _loss_tolerance(left, right, atol, rtol)
    if left < right - tolerance:
        return -1
    if left > right + tolerance:
        return 1
    return 0


@dataclass(frozen=True)
class NodeMoments:
    node: int
    parents: tuple[int, ...]
    source_intercept: float
    noise_variance: float
    mean_y: float
    mean_z: float
    second_y: float
    second_z: float
    second_yz: float
    variance_y: float
    variance_z: float
    covariance_yz: float
    coefficient_norm: float
    root: bool
    scale_eligible: bool
    hessian_eigenvalues: tuple[float, float]
    hessian_condition_number: float | None
    numerically_degenerate: bool
    degeneracy_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class LocalObjectiveMoments:
    """Topology-neutral sufficient moments for one reconstructed objective.

    This record makes no claim about parents, coefficients, intercepts, or root
    status. It is for public objective-surface identification only.
    """

    node: int
    noise_variance: float
    mean_y: float
    mean_z: float
    second_y: float
    second_z: float
    second_yz: float
    variance_y: float
    variance_z: float
    covariance_yz: float
    scale_eligible: bool
    hessian_eigenvalues: tuple[float, float]
    hessian_condition_number: float | None
    numerically_degenerate: bool
    degeneracy_reason: str | None
    identification: str = "public_objective_quadratic"

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


LocalObjective = NodeMoments | LocalObjectiveMoments


@dataclass(frozen=True)
class JointCandidate:
    derivation: str
    shift: float
    scale: float
    raw_local_loss: float

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class LocalActionState:
    node: int
    label: ActionLabel
    shift: float
    scale: float
    log_scale: float
    raw_local_loss: float
    unweighted_squared_residual: float
    nominal_cost: int
    effective_cost: int
    active_coordinates: tuple[str, ...]
    shift_boundary: str | None
    scale_boundary: str | None
    unique: bool
    derivation: str
    snapped_shift: bool
    snapped_scale: bool
    numerically_tied: bool
    degeneracy_reason: str | None
    joint_candidates: tuple[JointCandidate, ...] = ()

    @property
    def parameter_key(self) -> tuple[float, float]:
        return (self.shift, self.scale)

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class DPTracebackStep:
    node: int
    previous_budget: int
    current_budget: int
    local_state_index: int
    local_state_label: ActionLabel
    local_cost: int
    local_loss: float

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class SparseShiftScaleOracleResult:
    schema_version: str
    kl_direction: str
    intervention_semantics: str
    support_semantics: str
    n_nodes: int
    requested_budget: int
    effective_budget: int
    legal_coordinate_count: int
    budget_used: int
    dense_coordinate_cost: int
    kl_constant: float
    raw_initial_kl: float
    raw_K_B_star: float
    raw_K_dense_star: float
    presentation_initial_kl: float
    presentation_K_B_star: float
    presentation_K_dense_star: float
    direct_budgeted_kl: float
    direct_dense_kl: float
    budgeted_direct_absolute_error: float
    dense_direct_absolute_error: float
    budgeted_shift: np.ndarray
    budgeted_scale: np.ndarray
    dense_shift: np.ndarray
    dense_scale: np.ndarray
    budgeted_support: tuple[str, ...]
    dense_support: tuple[str, ...]
    node_moments: tuple[LocalObjective, ...]
    node_local_candidate_states: tuple[tuple[LocalActionState, ...], ...]
    selected_budgeted_state_indices: tuple[int, ...]
    selected_budgeted_state_labels: tuple[ActionLabel, ...]
    selected_dense_state_indices: tuple[int, ...]
    selected_dense_state_labels: tuple[ActionLabel, ...]
    selected_budgeted_local_losses: tuple[float, ...]
    selected_dense_local_losses: tuple[float, ...]
    dp_value_table: tuple[tuple[float | None, ...], ...]
    dp_traceback: tuple[DPTracebackStep, ...]
    optimal_support_count: int
    all_optimal_supports: tuple[tuple[str, ...], ...] | None
    all_optimal_supports_truncated: bool
    tie_diagnostics: tuple[str, ...]
    degeneracy_diagnostics: tuple[str, ...]
    snap_events: tuple[str, ...]
    runtime_seconds: dict[str, float]
    tolerances: dict[str, float]
    provenance: dict[str, object]

    @property
    def K_initial(self) -> float:
        return self.raw_initial_kl

    @property
    def K_B_star(self) -> float:
        return self.raw_K_B_star

    @property
    def K_dense_star(self) -> float:
        return self.raw_K_dense_star

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class _DPChoice:
    previous_budget: int
    state_index: int


@dataclass(frozen=True)
class _DPCell:
    loss: float
    canonical_choice: _DPChoice
    tied_choices: tuple[_DPChoice, ...]
    path_count: int
    support_key: tuple[str, ...]
    state_rank_key: tuple[int, ...]
    parameter_key: tuple[float, ...]
    numerical_tie: bool


def _validate_indexed_dag(dag: object) -> tuple[nx.DiGraph, int]:
    if not isinstance(dag, nx.DiGraph) or dag.is_multigraph():
        raise TypeError("dag must be a simple networkx.DiGraph.")
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("dag must be acyclic.")
    n_nodes = dag.number_of_nodes()
    if set(dag.nodes) != set(range(n_nodes)):
        raise ValueError("dag nodes must be labeled 0, ..., n_nodes - 1.")
    return dag, n_nodes


def _validated_array(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not np.issubdtype(value.dtype, np.number) or not np.isrealobj(value):
        raise TypeError(f"{name} must contain real numeric values.")
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _validated_bounds(
    name: str,
    bounds: object,
    *,
    require_zero: bool,
) -> tuple[float, float]:
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise TypeError(f"{name} must be a two-element tuple.")
    lower = _finite_real(f"{name}[0]", bounds[0])
    upper = _finite_real(f"{name}[1]", bounds[1])
    if lower >= upper:
        raise ValueError(f"{name} must be strictly increasing.")
    if require_zero and not lower <= 0.0 <= upper:
        raise ValueError(f"{name} must contain zero.")
    return lower, upper


def _boundary(value: float, lower: float, upper: float) -> str | None:
    if value == lower:
        return "lower"
    if value == upper:
        return "upper"
    return None


def extract_node_moments(
    *,
    dag: nx.DiGraph,
    source_intercepts: np.ndarray,
    coefficients: np.ndarray,
    noise_variances: np.ndarray,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    degeneracy_atol: float = 1e-14,
) -> tuple[NodeMoments, ...]:
    """Extract the five exact target moments required by each local quadratic."""
    _, n_nodes = _validate_indexed_dag(dag)
    alpha = _validated_array("source_intercepts", source_intercepts, (n_nodes,))
    beta = _validated_array("coefficients", coefficients, (n_nodes, n_nodes))
    variances = _validated_array("noise_variances", noise_variances, (n_nodes,))
    mean = _validated_array("target_mean", target_mean, (n_nodes,))
    covariance = _validated_array(
        "target_covariance", target_covariance, (n_nodes, n_nodes)
    )
    tolerance = _finite_real("degeneracy_atol", degeneracy_atol, nonnegative=True)
    second = covariance + np.outer(mean, mean)
    records: list[NodeMoments] = []
    for node in range(n_nodes):
        column = beta[:, node]
        parents = tuple(sorted(int(parent) for parent in dag.predecessors(node)))
        mean_y = float(mean[node] - alpha[node])
        mean_z = float(column @ mean)
        second_y = float(covariance[node, node] + mean_y**2)
        second_z = float(column @ second @ column)
        second_yz = float(column @ (covariance[:, node] + mean * mean_y))
        variance_y = float(covariance[node, node])
        variance_z_raw = float(second_z - mean_z**2)
        covariance_yz = float(second_yz - mean_y * mean_z)
        scale = max(1.0, abs(second_z), abs(mean_z**2))
        cancellation_tolerance = 64.0 * np.finfo(float).eps * scale
        if variance_z_raw < -cancellation_tolerance:
            raise RuntimeError("computed parental-predictor variance is negative.")
        variance_z = max(variance_z_raw, 0.0)
        eigenvalues = np.linalg.eigvalsh(
            np.array([[1.0, mean_z], [mean_z, second_z]], dtype=float)
        )
        positive = eigenvalues[eigenvalues > tolerance * max(1.0, eigenvalues[-1])]
        condition = float(eigenvalues[-1] / positive[0]) if positive.size == 2 else None
        coefficient_norm = float(np.linalg.norm(column))
        root = not parents
        exact_noop = bool(second_z == 0.0 or coefficient_norm == 0.0)
        numerical = bool(
            not exact_noop
            and (
                second_z <= tolerance * max(1.0, second_y)
                or variance_z <= tolerance * max(1.0, second_z)
            )
        )
        reason = None
        if exact_noop:
            reason = "zero_parental_predictor"
        elif numerical:
            reason = "near_singular_parental_predictor"
        records.append(
            NodeMoments(
                node=node,
                parents=parents,
                source_intercept=float(alpha[node]),
                noise_variance=float(variances[node]),
                mean_y=mean_y,
                mean_z=mean_z,
                second_y=second_y,
                second_z=second_z,
                second_yz=second_yz,
                variance_y=variance_y,
                variance_z=variance_z,
                covariance_yz=covariance_yz,
                coefficient_norm=coefficient_norm,
                root=root,
                scale_eligible=bool(not root and not exact_noop),
                hessian_eigenvalues=(float(eigenvalues[0]), float(eigenvalues[1])),
                hessian_condition_number=condition,
                numerically_degenerate=numerical or exact_noop,
                degeneracy_reason=reason,
            )
        )
    return tuple(records)


def local_squared_residual(moment: LocalObjective, shift: float, scale: float) -> float:
    """Return ``E_Q[(Y-a-sZ)^2]`` from the five node-local moments."""
    value = (
        moment.second_y
        + shift**2
        + scale**2 * moment.second_z
        - 2.0 * shift * moment.mean_y
        - 2.0 * scale * moment.second_yz
        + 2.0 * shift * scale * moment.mean_z
    )
    if not np.isfinite(value):
        raise RuntimeError("local squared residual is non-finite.")
    numerical_scale = max(
        1.0,
        abs(moment.second_y),
        abs(shift**2),
        abs(scale**2 * moment.second_z),
    )
    if value < -128.0 * np.finfo(float).eps * numerical_scale:
        raise RuntimeError("local squared residual is negative beyond roundoff.")
    return 0.0 if value < 0.0 else float(value)


def local_kl_loss(moment: LocalObjective, shift: float, scale: float) -> float:
    return local_squared_residual(moment, shift, scale) / (2.0 * moment.noise_variance)


def kl_additive_constant(
    target_covariance: np.ndarray,
    noise_variances: np.ndarray,
) -> float:
    """Return the complete intervention-independent Gaussian KL constant."""
    n_nodes = target_covariance.shape[0]
    cholesky = np.linalg.cholesky(target_covariance)
    logdet_target = 2.0 * float(np.log(np.diag(cholesky)).sum())
    return 0.5 * (float(np.log(noise_variances).sum()) - logdet_target - n_nodes)


def interventional_moments_from_parameters(
    *,
    source_intercepts: np.ndarray,
    coefficients: np.ndarray,
    noise_variances: np.ndarray,
    shift: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute moments for the implemented invariant-noise shift--scale SCM."""
    n_nodes = source_intercepts.size
    matrix = np.eye(n_nodes) - scale[:, None] * coefficients.T
    mean = np.linalg.solve(matrix, source_intercepts + shift)
    noise_factor = np.diag(np.sqrt(noise_variances))
    transformed_noise = np.linalg.solve(matrix, noise_factor)
    covariance = transformed_noise @ transformed_noise.T
    return mean, covariance


def decomposed_kl(
    *,
    moments: tuple[LocalObjective, ...],
    constant: float,
    shift: np.ndarray,
    scale: np.ndarray,
) -> float:
    return float(
        constant
        + sum(
            local_kl_loss(moment, float(shift[moment.node]), float(scale[moment.node]))
            for moment in moments
        )
    )


def _snap(
    shift: float,
    scale: float,
    support_atol: float,
) -> tuple[float, float, bool, bool]:
    snapped_shift = abs(shift) <= support_atol
    snapped_scale = abs(float(np.log(scale))) <= support_atol
    return (
        0.0 if snapped_shift else float(shift),
        1.0 if snapped_scale else float(scale),
        snapped_shift and shift != 0.0,
        snapped_scale and scale != 1.0,
    )


def _state(
    moment: LocalObjective,
    label: ActionLabel,
    shift: float,
    scale: float,
    *,
    nominal_cost: int,
    derivation: str,
    shift_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
    support_atol: float,
    unique: bool,
    degeneracy_reason: str | None = None,
    joint_candidates: tuple[JointCandidate, ...] = (),
) -> LocalActionState:
    snapped_shift_value, snapped_scale_value, snapped_shift, snapped_scale = _snap(
        shift, scale, support_atol
    )
    loss = local_kl_loss(moment, snapped_shift_value, snapped_scale_value)
    residual = local_squared_residual(moment, snapped_shift_value, snapped_scale_value)
    active: list[str] = []
    if snapped_shift_value != 0.0:
        active.append(f"shift:{moment.node}")
    if snapped_scale_value != 1.0:
        active.append(f"log_scale:{moment.node}")
    return LocalActionState(
        node=moment.node,
        label=label,
        shift=snapped_shift_value,
        scale=snapped_scale_value,
        log_scale=float(np.log(snapped_scale_value)),
        raw_local_loss=loss,
        unweighted_squared_residual=residual,
        nominal_cost=nominal_cost,
        effective_cost=len(active),
        active_coordinates=tuple(active),
        shift_boundary=_boundary(snapped_shift_value, shift_bounds[0], shift_bounds[1]),
        scale_boundary=_boundary(snapped_scale_value, scale_bounds[0], scale_bounds[1]),
        unique=unique,
        derivation=derivation,
        snapped_shift=snapped_shift,
        snapped_scale=snapped_scale,
        numerically_tied=False,
        degeneracy_reason=degeneracy_reason,
        joint_candidates=joint_candidates,
    )


def _candidate_better(
    left: LocalActionState,
    right: LocalActionState,
    loss_atol: float,
    loss_rtol: float,
) -> bool:
    order = _loss_order(left.raw_local_loss, right.raw_local_loss, loss_atol, loss_rtol)
    if order:
        return order < 0
    return (
        left.effective_cost,
        left.active_coordinates,
        _STATE_RANK[left.label],
        left.shift,
        left.scale,
    ) < (
        right.effective_cost,
        right.active_coordinates,
        _STATE_RANK[right.label],
        right.shift,
        right.scale,
    )


def solve_local_action_states(
    moment: LocalObjective,
    *,
    shift_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
    support_atol: float = DEFAULT_SUPPORT_ATOL,
    loss_atol: float = 1e-12,
    loss_rtol: float = 1e-11,
    degeneracy_atol: float = 1e-14,
) -> tuple[LocalActionState, ...]:
    """Solve, snap, deduplicate, and Pareto-prune one node's legal states."""
    support_tolerance = validate_support_atol(support_atol)
    absolute_tolerance = _finite_real("loss_atol", loss_atol, nonnegative=True)
    relative_tolerance = _finite_real("loss_rtol", loss_rtol, nonnegative=True)
    division_tolerance = _finite_real(
        "degeneracy_atol", degeneracy_atol, nonnegative=True
    ) * max(1.0, abs(moment.second_z), abs(moment.variance_z))
    shift_lower, shift_upper = shift_bounds
    scale_lower, scale_upper = scale_bounds
    states: list[LocalActionState] = [
        _state(
            moment,
            "none",
            0.0,
            1.0,
            nominal_cost=0,
            derivation="identity",
            shift_bounds=shift_bounds,
            scale_bounds=scale_bounds,
            support_atol=support_tolerance,
            unique=True,
        )
    ]
    shift_optimum = float(
        np.clip(moment.mean_y - moment.mean_z, shift_lower, shift_upper)
    )
    states.append(
        _state(
            moment,
            "shift",
            shift_optimum,
            1.0,
            nominal_cost=1,
            derivation="closed_form_shift",
            shift_bounds=shift_bounds,
            scale_bounds=scale_bounds,
            support_atol=support_tolerance,
            unique=True,
        )
    )
    if moment.scale_eligible:
        if moment.second_z > division_tolerance:
            scale_optimum = float(
                np.clip(moment.second_yz / moment.second_z, scale_lower, scale_upper)
            )
            scale_unique = True
            scale_reason = None
        else:
            scale_optimum = 1.0
            scale_unique = False
            scale_reason = "numerically_flat_scale"
        states.append(
            _state(
                moment,
                "scale",
                0.0,
                scale_optimum,
                nominal_cost=1,
                derivation="closed_form_scale",
                shift_bounds=shift_bounds,
                scale_bounds=scale_bounds,
                support_atol=support_tolerance,
                unique=scale_unique,
                degeneracy_reason=scale_reason,
            )
        )

        raw_joint: list[tuple[str, float, float]] = []
        if moment.variance_z > division_tolerance:
            regression_scale = moment.covariance_yz / moment.variance_z
            regression_shift = moment.mean_y - regression_scale * moment.mean_z
            if (
                shift_lower <= regression_shift <= shift_upper
                and scale_lower <= regression_scale <= scale_upper
            ):
                raw_joint.append(
                    ("interior_regression", regression_shift, regression_scale)
                )
        for fixed_shift, name in (
            (shift_lower, "shift_lower_edge"),
            (shift_upper, "shift_upper_edge"),
        ):
            if moment.second_z > division_tolerance:
                edge_scale = float(
                    np.clip(
                        (moment.second_yz - fixed_shift * moment.mean_z)
                        / moment.second_z,
                        scale_lower,
                        scale_upper,
                    )
                )
                raw_joint.append((name, fixed_shift, edge_scale))
            else:
                raw_joint.extend(
                    (f"{name}_{suffix}", fixed_shift, value)
                    for suffix, value in (
                        ("identity", 1.0),
                        ("lower", scale_lower),
                        ("upper", scale_upper),
                    )
                )
        for fixed_scale, name in (
            (scale_lower, "scale_lower_edge"),
            (scale_upper, "scale_upper_edge"),
        ):
            edge_shift = float(
                np.clip(
                    moment.mean_y - fixed_scale * moment.mean_z,
                    shift_lower,
                    shift_upper,
                )
            )
            raw_joint.append((name, edge_shift, fixed_scale))
        for fixed_shift, shift_name in (
            (shift_lower, "shift_lower"),
            (shift_upper, "shift_upper"),
        ):
            for fixed_scale, scale_name in (
                (scale_lower, "scale_lower"),
                (scale_upper, "scale_upper"),
            ):
                raw_joint.append(
                    (f"corner_{shift_name}_{scale_name}", fixed_shift, fixed_scale)
                )
        evaluated = tuple(
            JointCandidate(
                derivation=name,
                shift=float(shift),
                scale=float(scale),
                raw_local_loss=local_kl_loss(moment, float(shift), float(scale)),
            )
            for name, shift, scale in raw_joint
        )
        if not evaluated:
            raise RuntimeError("joint solver produced no boundary candidates.")
        best_joint = evaluated[0]
        joint_tied = False
        for candidate in evaluated[1:]:
            order = _loss_order(
                candidate.raw_local_loss,
                best_joint.raw_local_loss,
                absolute_tolerance,
                relative_tolerance,
            )
            if order < 0 or (
                order == 0
                and (candidate.derivation, candidate.shift, candidate.scale)
                < (best_joint.derivation, best_joint.shift, best_joint.scale)
            ):
                joint_tied = joint_tied or order == 0
                best_joint = candidate
            elif order == 0:
                joint_tied = True
        joint_state = _state(
            moment,
            "shift_scale",
            best_joint.shift,
            best_joint.scale,
            nominal_cost=2,
            derivation=best_joint.derivation,
            shift_bounds=shift_bounds,
            scale_bounds=scale_bounds,
            support_atol=support_tolerance,
            unique=bool(moment.variance_z > division_tolerance and not joint_tied),
            degeneracy_reason=(
                None
                if moment.variance_z > division_tolerance
                else "singular_joint_quadratic"
            ),
            joint_candidates=evaluated,
        )
        if joint_tied:
            joint_state = replace(joint_state, numerically_tied=True)
        states.append(joint_state)

    deduplicated: dict[tuple[float, float], LocalActionState] = {}
    for state in states:
        current = deduplicated.get(state.parameter_key)
        if current is None or _candidate_better(
            state, current, absolute_tolerance, relative_tolerance
        ):
            deduplicated[state.parameter_key] = state
    ordered = sorted(
        deduplicated.values(),
        key=lambda state: (
            state.effective_cost,
            _STATE_RANK[state.label],
            state.shift,
            state.scale,
        ),
    )
    retained: list[LocalActionState] = []
    for state in ordered:
        dominated = any(
            other.effective_cost <= state.effective_cost
            and _loss_order(
                other.raw_local_loss,
                state.raw_local_loss,
                absolute_tolerance,
                relative_tolerance,
            )
            <= 0
            and (
                other.effective_cost < state.effective_cost
                or _loss_order(
                    other.raw_local_loss,
                    state.raw_local_loss,
                    absolute_tolerance,
                    relative_tolerance,
                )
                < 0
            )
            for other in ordered
            if other is not state
        )
        if not dominated:
            retained.append(state)
    return tuple(retained)


def _path_transition_key(
    previous: _DPCell | None,
    state: LocalActionState,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[float, ...]]:
    support = (
        () if previous is None else previous.support_key
    ) + state.active_coordinates
    ranks = (() if previous is None else previous.state_rank_key) + (
        _STATE_RANK[state.label],
    )
    parameters = (() if previous is None else previous.parameter_key) + (
        state.shift,
        state.scale,
    )
    return support, ranks, parameters


def _enumerate_optimal_supports(
    table: list[list[_DPCell | None]],
    states_by_node: tuple[tuple[LocalActionState, ...], ...],
    node_count: int,
    budget: int,
    cap: int,
) -> tuple[tuple[tuple[str, ...], ...], bool]:
    output: list[tuple[str, ...]] = []

    def visit(node: int, current_budget: int, support: tuple[str, ...]) -> None:
        if len(output) >= cap:
            return
        if node == 0:
            output.append(tuple(sorted(support)))
            return
        cell = table[node][current_budget]
        if cell is None:
            return
        for choice in cell.tied_choices:
            state = states_by_node[node - 1][choice.state_index]
            visit(node - 1, choice.previous_budget, support + state.active_coordinates)

    visit(node_count, budget, ())
    unique = tuple(sorted(set(output)))
    cell = table[node_count][budget]
    truncated = bool(cell is not None and cell.path_count > len(output))
    return unique, truncated


def solve_sparse_shift_scale_kl_from_parameters(
    *,
    dag: nx.DiGraph,
    source_intercepts: np.ndarray,
    coefficients: np.ndarray,
    noise_variances: np.ndarray,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    budget: int,
    shift_bounds: tuple[float, float],
    log_scale_bounds: tuple[float, float],
    support_atol: float = DEFAULT_SUPPORT_ATOL,
    loss_atol: float = 1e-12,
    loss_rtol: float = 1e-11,
    degeneracy_atol: float = 1e-14,
    return_all_optimal_supports: bool = False,
    all_optimal_supports_cap: int = 10_000,
    provenance: dict[str, object] | None = None,
    precomputed_node_moments: tuple[LocalObjective, ...] | None = None,
    precomputed_kl_constant: float | None = None,
) -> SparseShiftScaleOracleResult:
    """Return the exact bounded sparse and dense target-to-candidate KL optima.

    ``precomputed_node_moments`` is an additive adapter for public experimental
    reconstruction. When supplied together with ``precomputed_kl_constant``,
    it bypasses only hidden-parameter moment extraction; the validated local
    solver, snapping, tie logic, and dynamic program below are unchanged.
    Existing callers omit both arguments and retain the historical behavior.
    """
    total_started = perf_counter()
    graph, n_nodes = _validate_indexed_dag(dag)
    alpha = _validated_array("source_intercepts", source_intercepts, (n_nodes,))
    beta = _validated_array("coefficients", coefficients, (n_nodes, n_nodes))
    variances = _validated_array("noise_variances", noise_variances, (n_nodes,))
    if np.any(variances <= 0.0):
        raise ValueError("noise_variances must be strictly positive.")
    edge_mask = np.zeros((n_nodes, n_nodes), dtype=bool)
    for parent, child in graph.edges:
        edge_mask[int(parent), int(child)] = True
    if np.any(beta[~edge_mask] != 0.0):
        raise ValueError("coefficients must be zero outside DAG edges.")
    mean = _validated_array("target_mean", target_mean, (n_nodes,))
    covariance = _validated_array(
        "target_covariance", target_covariance, (n_nodes, n_nodes)
    )
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("target_covariance must be symmetric.")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("target_covariance must be positive definite.") from error
    shift_box = _validated_bounds("shift_bounds", shift_bounds, require_zero=True)
    log_scale_box = _validated_bounds(
        "log_scale_bounds", log_scale_bounds, require_zero=True
    )
    scale_box = (float(np.exp(log_scale_box[0])), float(np.exp(log_scale_box[1])))
    if not 0.0 < scale_box[0] <= 1.0 <= scale_box[1]:
        raise ValueError("converted natural-scale bounds must contain one.")
    if isinstance(budget, bool) or not isinstance(budget, Integral):
        raise TypeError("budget must be an integer.")
    requested_budget = int(budget)
    if requested_budget < 0:
        raise ValueError("budget must be nonnegative.")
    if not isinstance(return_all_optimal_supports, bool):
        raise TypeError("return_all_optimal_supports must be boolean.")
    if (
        isinstance(all_optimal_supports_cap, bool)
        or not isinstance(all_optimal_supports_cap, Integral)
        or int(all_optimal_supports_cap) <= 0
    ):
        raise ValueError("all_optimal_supports_cap must be a positive integer.")
    support_tolerance = validate_support_atol(support_atol)
    absolute_tolerance = _finite_real("loss_atol", loss_atol, nonnegative=True)
    relative_tolerance = _finite_real("loss_rtol", loss_rtol, nonnegative=True)
    degeneracy_tolerance = _finite_real(
        "degeneracy_atol", degeneracy_atol, nonnegative=True
    )

    if (precomputed_node_moments is None) != (precomputed_kl_constant is None):
        raise ValueError(
            "precomputed_node_moments and precomputed_kl_constant must be "
            "supplied together."
        )
    moments_started = perf_counter()
    if precomputed_node_moments is None:
        moments = extract_node_moments(
            dag=graph,
            source_intercepts=alpha,
            coefficients=beta,
            noise_variances=variances,
            target_mean=mean,
            target_covariance=covariance,
            degeneracy_atol=degeneracy_tolerance,
        )
        constant = kl_additive_constant(covariance, variances)
    else:
        moments = tuple(precomputed_node_moments)
        if len(moments) != n_nodes or any(
            not isinstance(moment, (NodeMoments, LocalObjectiveMoments))
            or moment.node != node
            for node, moment in enumerate(moments)
        ):
            raise ValueError(
                "precomputed_node_moments must contain one ordered local-objective "
                "record per node."
            )
        constant = _finite_real("precomputed_kl_constant", precomputed_kl_constant)
    moments_seconds = perf_counter() - moments_started

    local_started = perf_counter()
    states_by_node = tuple(
        solve_local_action_states(
            moment,
            shift_bounds=shift_box,
            scale_bounds=scale_box,
            support_atol=support_tolerance,
            loss_atol=absolute_tolerance,
            loss_rtol=relative_tolerance,
            degeneracy_atol=degeneracy_tolerance,
        )
        for moment in moments
    )
    local_seconds = perf_counter() - local_started
    legal_coordinate_count = n_nodes + sum(moment.scale_eligible for moment in moments)
    effective_budget = min(requested_budget, legal_coordinate_count)

    dp_started = perf_counter()
    table: list[list[_DPCell | None]] = [
        [None for _ in range(effective_budget + 1)] for _ in range(n_nodes + 1)
    ]
    table[0][0] = _DPCell(
        loss=0.0,
        canonical_choice=_DPChoice(0, 0),
        tied_choices=(_DPChoice(0, 0),),
        path_count=1,
        support_key=(),
        state_rank_key=(),
        parameter_key=(),
        numerical_tie=False,
    )
    tie_messages: list[str] = []
    for node_index, states in enumerate(states_by_node, start=1):
        for used_budget in range(effective_budget + 1):
            candidates: list[tuple[float, _DPChoice, _DPCell, tuple[Any, ...]]] = []
            for state_index, state in enumerate(states):
                previous_budget = used_budget - state.effective_cost
                if previous_budget < 0:
                    continue
                previous = table[node_index - 1][previous_budget]
                if previous is None:
                    continue
                support_key, rank_key, parameter_key = _path_transition_key(
                    previous, state
                )
                candidates.append(
                    (
                        previous.loss + state.raw_local_loss,
                        _DPChoice(previous_budget, state_index),
                        previous,
                        (support_key, rank_key, parameter_key),
                    )
                )
            if not candidates:
                continue
            best = candidates[0]
            tied = [best]
            numerical_tie = False
            for candidate in candidates[1:]:
                order = _loss_order(
                    candidate[0], best[0], absolute_tolerance, relative_tolerance
                )
                if order < 0:
                    best = candidate
                    tied = [candidate]
                    numerical_tie = False
                elif order == 0:
                    tied.append(candidate)
                    numerical_tie = True
                    if candidate[3] < best[3]:
                        best = candidate
            if numerical_tie:
                tie_messages.append(
                    f"dp numerical tie at node={node_index - 1}, budget={used_budget}"
                )
            table[node_index][used_budget] = _DPCell(
                loss=float(best[0]),
                canonical_choice=best[1],
                tied_choices=tuple(item[1] for item in tied),
                path_count=sum(item[2].path_count for item in tied),
                support_key=best[3][0],
                state_rank_key=best[3][1],
                parameter_key=best[3][2],
                numerical_tie=numerical_tie,
            )
    terminal_budget: int | None = None
    terminal: _DPCell | None = None
    for used_budget, cell in enumerate(table[n_nodes]):
        if cell is None:
            continue
        if terminal is None:
            terminal_budget, terminal = used_budget, cell
            continue
        order = _loss_order(
            cell.loss, terminal.loss, absolute_tolerance, relative_tolerance
        )
        if order < 0 or (
            order == 0
            and (used_budget, cell.support_key, cell.state_rank_key, cell.parameter_key)
            < (
                int(terminal_budget),
                terminal.support_key,
                terminal.state_rank_key,
                terminal.parameter_key,
            )
        ):
            if order == 0:
                tie_messages.append(
                    f"terminal numerical tie between budgets {terminal_budget} and {used_budget}"
                )
            terminal_budget, terminal = used_budget, cell
    if terminal is None or terminal_budget is None:
        raise RuntimeError("dynamic program found no feasible intervention.")
    selected_indices = [0] * n_nodes
    traceback: list[DPTracebackStep] = []
    current_budget = terminal_budget
    for node_index in range(n_nodes, 0, -1):
        cell = table[node_index][current_budget]
        if cell is None:
            raise RuntimeError("dynamic-program traceback reached an empty cell.")
        choice = cell.canonical_choice
        state = states_by_node[node_index - 1][choice.state_index]
        selected_indices[node_index - 1] = choice.state_index
        traceback.append(
            DPTracebackStep(
                node=node_index - 1,
                previous_budget=choice.previous_budget,
                current_budget=current_budget,
                local_state_index=choice.state_index,
                local_state_label=state.label,
                local_cost=state.effective_cost,
                local_loss=state.raw_local_loss,
            )
        )
        current_budget = choice.previous_budget
    if current_budget != 0:
        raise RuntimeError(
            "dynamic-program traceback did not terminate at budget zero."
        )
    traceback.reverse()
    selected_states = tuple(
        states_by_node[node][state_index]
        for node, state_index in enumerate(selected_indices)
    )
    budgeted_shift = _immutable(np.array([state.shift for state in selected_states]))
    budgeted_scale = _immutable(np.array([state.scale for state in selected_states]))
    budgeted_local_sum = float(sum(state.raw_local_loss for state in selected_states))
    traceback_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(1.0, abs(budgeted_local_sum), abs(terminal.loss))
    )
    if not np.isclose(
        budgeted_local_sum,
        terminal.loss,
        atol=traceback_tolerance,
        rtol=0.0,
    ):
        raise RuntimeError("traceback local losses disagree with terminal DP cell.")
    if sum(state.effective_cost for state in selected_states) != terminal_budget:
        raise RuntimeError("traceback effective cost disagrees with terminal budget.")

    dense_indices: list[int] = []
    dense_states: list[LocalActionState] = []
    for states in states_by_node:
        best_index = 0
        for state_index in range(1, len(states)):
            if _candidate_better(
                states[state_index],
                states[best_index],
                absolute_tolerance,
                relative_tolerance,
            ):
                best_index = state_index
        dense_indices.append(best_index)
        dense_states.append(states[best_index])
    dense_tuple = tuple(dense_states)
    dense_shift = _immutable(np.array([state.shift for state in dense_tuple]))
    dense_scale = _immutable(np.array([state.scale for state in dense_tuple]))
    dp_seconds = perf_counter() - dp_started

    initial_shift = np.zeros(n_nodes, dtype=float)
    initial_scale = np.ones(n_nodes, dtype=float)
    raw_initial = decomposed_kl(
        moments=moments,
        constant=constant,
        shift=initial_shift,
        scale=initial_scale,
    )
    raw_budgeted = float(constant + budgeted_local_sum)
    raw_dense = float(constant + sum(state.raw_local_loss for state in dense_tuple))
    if raw_dense > raw_budgeted + _loss_tolerance(
        raw_dense, raw_budgeted, absolute_tolerance, relative_tolerance
    ):
        raise RuntimeError("dense optimum is worse than budgeted optimum.")
    if precomputed_node_moments is None:
        budgeted_mean, budgeted_covariance = interventional_moments_from_parameters(
            source_intercepts=alpha,
            coefficients=beta,
            noise_variances=variances,
            shift=budgeted_shift,
            scale=budgeted_scale,
        )
        dense_mean, dense_covariance = interventional_moments_from_parameters(
            source_intercepts=alpha,
            coefficients=beta,
            noise_variances=variances,
            shift=dense_shift,
            scale=dense_scale,
        )
        direct_budgeted = gaussian_kl_target_to_candidate(
            mean, covariance, budgeted_mean, budgeted_covariance
        )
        direct_dense = gaussian_kl_target_to_candidate(
            mean, covariance, dense_mean, dense_covariance
        )
    else:
        direct_budgeted = raw_budgeted
        direct_dense = raw_dense
    if return_all_optimal_supports:
        all_supports, supports_truncated = _enumerate_optimal_supports(
            table,
            states_by_node,
            n_nodes,
            terminal_budget,
            int(all_optimal_supports_cap),
        )
    else:
        all_supports, supports_truncated = None, False
    degeneracy_messages = tuple(
        f"node={moment.node}:{moment.degeneracy_reason}"
        for moment in moments
        if moment.degeneracy_reason is not None
    )
    snap_messages = tuple(
        f"node={state.node},state={state.label},shift={state.snapped_shift},scale={state.snapped_scale}"
        for states in states_by_node
        for state in states
        if state.snapped_shift or state.snapped_scale
    )
    total_seconds = perf_counter() - total_started
    dp_values = tuple(
        tuple(None if cell is None else float(cell.loss) for cell in row)
        for row in table
    )
    base_provenance: dict[str, object] = {
        "oracle_schema_version": ORACLE_SCHEMA_VERSION,
        "coefficient_convention": "coefficients[parent,child]",
        "source_intercepts": "explicit_array",
        "noise_variances": "node_specific_invariant",
        "shift_bounds": shift_box,
        "log_scale_bounds": log_scale_box,
        "natural_scale_bounds": scale_box,
        "support_atol": support_tolerance,
        "support_semantics": SUPPORT_SEMANTICS,
        "kl_direction": KL_DIRECTION,
        "hidden_dag_evaluation_only": precomputed_node_moments is None,
        "precomputed_public_node_moments": precomputed_node_moments is not None,
    }
    if provenance:
        base_provenance.update(provenance)
    return SparseShiftScaleOracleResult(
        schema_version=ORACLE_SCHEMA_VERSION,
        kl_direction=KL_DIRECTION,
        intervention_semantics=INTERVENTION_SEMANTICS,
        support_semantics=SUPPORT_SEMANTICS,
        n_nodes=n_nodes,
        requested_budget=requested_budget,
        effective_budget=effective_budget,
        legal_coordinate_count=legal_coordinate_count,
        budget_used=terminal_budget,
        dense_coordinate_cost=sum(state.effective_cost for state in dense_tuple),
        kl_constant=constant,
        raw_initial_kl=raw_initial,
        raw_K_B_star=raw_budgeted,
        raw_K_dense_star=raw_dense,
        presentation_initial_kl=max(0.0, raw_initial),
        presentation_K_B_star=max(0.0, raw_budgeted),
        presentation_K_dense_star=max(0.0, raw_dense),
        direct_budgeted_kl=direct_budgeted,
        direct_dense_kl=direct_dense,
        budgeted_direct_absolute_error=abs(raw_budgeted - direct_budgeted),
        dense_direct_absolute_error=abs(raw_dense - direct_dense),
        budgeted_shift=budgeted_shift,
        budgeted_scale=budgeted_scale,
        dense_shift=dense_shift,
        dense_scale=dense_scale,
        budgeted_support=tuple(
            coordinate
            for state in selected_states
            for coordinate in state.active_coordinates
        ),
        dense_support=tuple(
            coordinate
            for state in dense_tuple
            for coordinate in state.active_coordinates
        ),
        node_moments=moments,
        node_local_candidate_states=states_by_node,
        selected_budgeted_state_indices=tuple(selected_indices),
        selected_budgeted_state_labels=tuple(state.label for state in selected_states),
        selected_dense_state_indices=tuple(dense_indices),
        selected_dense_state_labels=tuple(state.label for state in dense_tuple),
        selected_budgeted_local_losses=tuple(
            state.raw_local_loss for state in selected_states
        ),
        selected_dense_local_losses=tuple(
            state.raw_local_loss for state in dense_tuple
        ),
        dp_value_table=dp_values,
        dp_traceback=tuple(traceback),
        optimal_support_count=terminal.path_count,
        all_optimal_supports=all_supports,
        all_optimal_supports_truncated=supports_truncated,
        tie_diagnostics=tuple(tie_messages),
        degeneracy_diagnostics=degeneracy_messages,
        snap_events=snap_messages,
        runtime_seconds={
            "moment_extraction": moments_seconds,
            "local_solve": local_seconds,
            "dynamic_program": dp_seconds,
            "total": total_seconds,
        },
        tolerances={
            "support_atol": support_tolerance,
            "loss_atol": absolute_tolerance,
            "loss_rtol": relative_tolerance,
            "degeneracy_atol": degeneracy_tolerance,
        },
        provenance=base_provenance,
    )


def solve_sparse_shift_scale_kl(
    *,
    scm: LinearGaussianSCM,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    budget: int,
    shift_bounds: tuple[float, float],
    log_scale_bounds: tuple[float, float],
    source_intercepts: np.ndarray | None = None,
    **kwargs: Any,
) -> SparseShiftScaleOracleResult:
    """Repository-facing wrapper around the exact low-level array solver."""
    if not isinstance(scm, LinearGaussianSCM):
        raise TypeError("scm must be a LinearGaussianSCM.")
    n_nodes = scm.dag.number_of_nodes()
    if source_intercepts is None:
        intercepts = np.zeros(n_nodes, dtype=float)
        convention = "repository_zero_vector_default"
    else:
        intercepts = source_intercepts
        convention = "explicit_array"
    supplied_provenance = dict(kwargs.pop("provenance", {}) or {})
    supplied_provenance.update(
        {
            "adapter": "LinearGaussianSCM",
            "source_intercept_convention": convention,
        }
    )
    return solve_sparse_shift_scale_kl_from_parameters(
        dag=scm.dag,
        source_intercepts=intercepts,
        coefficients=scm.coefficients,
        noise_variances=scm.noise_variances,
        target_mean=target_mean,
        target_covariance=target_covariance,
        budget=budget,
        shift_bounds=shift_bounds,
        log_scale_bounds=log_scale_bounds,
        provenance=supplied_provenance,
        **kwargs,
    )


def residual_decomposition(
    *,
    K_greedy: float,
    result: SparseShiftScaleOracleResult,
    tolerance: float = 1e-10,
) -> dict[str, object]:
    """Build raw diagnostic components without hiding material violations."""
    greedy = _finite_real("K_greedy", K_greedy)
    tol = _finite_real("tolerance", tolerance, nonnegative=True)
    family_floor = result.raw_K_dense_star
    budget_penalty = result.raw_K_B_star - result.raw_K_dense_star
    greedy_gap = greedy - result.raw_K_B_star
    available = result.raw_initial_kl - result.raw_K_B_star
    material_violation = greedy_gap < -tol * max(
        1.0, abs(greedy), abs(result.raw_K_B_star)
    )
    negligible = available <= tol * max(
        1.0, abs(result.raw_initial_kl), abs(result.raw_K_B_star)
    )
    return {
        "family_floor_raw": family_floor,
        "sparsity_budget_penalty_raw": budget_penalty,
        "greedy_search_gap_raw": greedy_gap,
        "family_floor_presentation": max(0.0, family_floor),
        "sparsity_budget_penalty_presentation": max(0.0, budget_penalty),
        "greedy_search_gap_presentation": (
            greedy_gap if material_violation else max(0.0, greedy_gap)
        ),
        "fraction_oracle_improvement_attained": (
            None if negligible else (result.raw_initial_kl - greedy) / available
        ),
        "negligible_oracle_improvement": negligible,
        "archive_below_oracle_anomaly": material_violation,
        "roundoff_negative_budget_component": -tol <= budget_penalty < 0.0,
        "roundoff_negative_greedy_gap": -tol <= greedy_gap < 0.0,
    }


__all__ = [
    "KL_DIRECTION",
    "ORACLE_SCHEMA_VERSION",
    "SUPPORT_SEMANTICS",
    "DPTracebackStep",
    "JointCandidate",
    "LocalActionState",
    "NodeMoments",
    "SparseShiftScaleOracleResult",
    "decomposed_kl",
    "extract_node_moments",
    "interventional_moments_from_parameters",
    "kl_additive_constant",
    "local_kl_loss",
    "local_squared_residual",
    "residual_decomposition",
    "solve_local_action_states",
    "solve_sparse_shift_scale_kl",
    "solve_sparse_shift_scale_kl_from_parameters",
]
