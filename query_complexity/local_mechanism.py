"""Public local-mechanism recovery from one exact singleton shift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graphs.meek import apply_meek_closure
from graphs.pdag import PartiallyDirectedGraph
from graphs.scm import DEFAULT_BETA_MIN


def _immutable(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LocalMechanismDiagnostics:
    """Numerical checks attached to a publicly recovered local mechanism."""

    precision_route: str
    source_covariance_condition_number: float | None
    raw_self_precision_effect: float
    normalized_self_error: float
    coefficient_support_tolerance: float
    max_selected_coefficient_regression_error: float
    noise_variance_reconstruction_error: float
    intercept_reconstruction_error: float
    max_rejected_coefficient_magnitude: float


@dataclass(frozen=True)
class RecoveredLocalMechanism:
    """Topology-clean public reconstruction for one structural equation."""

    node: int
    parents: tuple[int, ...]
    incoming_coefficients: np.ndarray
    mechanism_row: np.ndarray
    noise_variance: float
    source_intercept: float
    root: bool
    total_effect_column: np.ndarray
    diagnostics: LocalMechanismDiagnostics
    provenance: tuple[str, ...] = (
        "public_source_moments",
        "one_exact_singleton_shift_mean",
        "precision_times_total_effect_identity",
        "no_hidden_structural_inputs",
    )

    def __post_init__(self) -> None:
        incoming = _immutable(self.incoming_coefficients)
        row = _immutable(self.mechanism_row)
        column = _immutable(self.total_effect_column)
        if (
            incoming.ndim != 1
            or row.shape != incoming.shape
            or column.shape != row.shape
        ):
            raise ValueError("local mechanism arrays must be equal-length vectors")
        if not all(np.all(np.isfinite(item)) for item in (incoming, row, column)):
            raise ValueError("local mechanism arrays must be finite")
        object.__setattr__(self, "incoming_coefficients", incoming)
        object.__setattr__(self, "mechanism_row", row)
        object.__setattr__(self, "total_effect_column", column)


@dataclass(frozen=True)
class IncidentOrientationResult:
    """Direct local-mechanism orientations followed by public Meek closure."""

    graph: PartiallyDirectedGraph
    direct_orientations: tuple[tuple[int, int], ...]
    meek_orientations: tuple[tuple[int, int], ...]


def recover_local_mechanism_from_singleton_shift(
    *,
    node: int,
    source_mean: np.ndarray,
    shift_magnitude: float | None = None,
    exact_shifted_mean: np.ndarray,
    source_covariance: np.ndarray | None = None,
    source_precision: np.ndarray | None = None,
    coefficient_support_tolerance: float = DEFAULT_BETA_MIN,
    validation_atol: float = 1e-11,
    validation_rtol: float = 1e-10,
) -> RecoveredLocalMechanism:
    """Recover node ``i`` from one exact local additive mean response.

    When ``shift_magnitude`` is supplied, this uses the familiar identity
    ``K_0 (mu_i^h-mu_0)/h = m_i/omega_i``.  When it is ``None``, the physical
    magnitude need not be calibrated: with ``d_i=mu_i^h-mu_0``, the routine
    normalizes ``K_0 d_i`` by its node-``i`` coordinate and recovers the source
    noise variance from the source covariance.  Acyclicity gives
    ``d_i[i]=h``, so the total-effect column remains available as
    ``d_i/d_i[i]``.

    Exactly one of ``source_covariance`` and ``source_precision`` must be
    supplied.  The covariance route uses a linear solve and never forms an
    explicit inverse.  No SCM, DAG, coefficient, or noise input is accepted.
    """
    if not isinstance(source_mean, np.ndarray) or source_mean.ndim != 1:
        raise TypeError("source_mean must be a one-dimensional numpy array")
    n = source_mean.size
    if isinstance(node, bool) or not isinstance(node, (int, np.integer)):
        raise TypeError("node must be an integer")
    node = int(node)
    if node < 0 or node >= n:
        raise ValueError("node is outside the moment dimension")
    if not isinstance(exact_shifted_mean, np.ndarray) or exact_shifted_mean.shape != (
        n,
    ):
        raise TypeError("exact_shifted_mean must match source_mean")
    if not np.all(np.isfinite(source_mean)) or not np.all(
        np.isfinite(exact_shifted_mean)
    ):
        raise ValueError("mean inputs must be finite")
    shift: float | None
    if shift_magnitude is None:
        shift = None
    else:
        shift = float(shift_magnitude)
        if not np.isfinite(shift) or shift == 0.0:
            raise ValueError("shift_magnitude must be finite and nonzero")
    support_tolerance = float(coefficient_support_tolerance)
    if not np.isfinite(support_tolerance) or support_tolerance < 0.0:
        raise ValueError("coefficient_support_tolerance must be finite and nonnegative")
    if (source_covariance is None) == (source_precision is None):
        raise ValueError("supply exactly one of source_covariance or source_precision")

    mean_response = exact_shifted_mean - source_mean
    response_at_node = float(mean_response[node])
    if not np.isfinite(response_at_node) or response_at_node == 0.0:
        raise ValueError(
            "additive probe must have a finite nonzero mean response at its target"
        )
    total_effect_column = mean_response / (
        shift if shift is not None else response_at_node
    )
    covariance_condition: float | None = None
    covariance: np.ndarray | None = None
    if source_covariance is not None:
        if not isinstance(source_covariance, np.ndarray) or source_covariance.shape != (
            n,
            n,
        ):
            raise TypeError("source_covariance must have shape (n, n)")
        covariance = np.asarray(source_covariance, dtype=float)
        if not np.all(np.isfinite(covariance)):
            raise ValueError("source_covariance must be finite")
        if not np.allclose(
            covariance, covariance.T, atol=validation_atol, rtol=validation_rtol
        ):
            raise ValueError("source_covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("source_covariance must be positive definite") from error
        covariance_condition = float(np.linalg.cond(covariance))
        if not np.isfinite(covariance_condition):
            raise ValueError("source_covariance must be nonsingular")
        try:
            precision_effect = np.linalg.solve(
                covariance,
                total_effect_column if shift is not None else mean_response,
            )
        except np.linalg.LinAlgError as error:
            raise ValueError("source_covariance must be positive definite") from error
        precision_route = "solve_source_covariance"
    else:
        if not isinstance(source_precision, np.ndarray) or source_precision.shape != (
            n,
            n,
        ):
            raise TypeError("source_precision must have shape (n, n)")
        precision = np.asarray(source_precision, dtype=float)
        if not np.all(np.isfinite(precision)):
            raise ValueError("source_precision must be finite")
        if not np.allclose(
            precision, precision.T, atol=validation_atol, rtol=validation_rtol
        ):
            raise ValueError("source_precision must be symmetric")
        try:
            np.linalg.cholesky(precision)
        except np.linalg.LinAlgError as error:
            raise ValueError("source_precision must be positive definite") from error
        precision_effect = precision @ (
            total_effect_column if shift is not None else mean_response
        )
        try:
            covariance = np.linalg.solve(precision, np.eye(n))
        except np.linalg.LinAlgError as error:
            raise ValueError("source_precision must be nonsingular") from error
        precision_route = "precomputed_source_precision"

    if not np.all(np.isfinite(precision_effect)):
        raise RuntimeError("precision-times-effect calculation is nonfinite")
    self_entry = float(precision_effect[node])
    if not np.isfinite(self_entry) or (
        self_entry <= 0.0 if shift is not None else self_entry == 0.0
    ):
        detail = (
            "inverse noise variance must be finite and positive"
            if shift is not None
            else "probe normalization denominator must be finite and nonzero"
        )
        raise RuntimeError(f"recovered {detail}")
    mechanism_row = precision_effect / self_entry
    self_error = abs(float(mechanism_row[node]) - 1.0)
    scale = max(1.0, float(np.max(np.abs(mechanism_row))))
    tolerance = validation_atol + validation_rtol * scale
    if self_error > tolerance:
        raise RuntimeError(
            "recovered mechanism row does not have unit self coefficient"
        )
    mechanism_row = np.array(mechanism_row, copy=True)
    mechanism_row[node] = 1.0
    incoming = -mechanism_row
    incoming[node] = 0.0
    rejected = np.abs(incoming) < support_tolerance
    rejected[node] = True
    max_rejected = (
        float(np.max(np.abs(incoming[rejected]))) if np.any(rejected) else 0.0
    )
    incoming[rejected] = 0.0
    parents = tuple(int(index) for index in np.flatnonzero(incoming != 0.0))
    mechanism_row = np.eye(n)[node] - incoming
    noise_variance = float(mechanism_row @ covariance @ mechanism_row)
    if not np.isfinite(noise_variance) or noise_variance <= 0.0:
        raise RuntimeError(
            "source covariance produced a nonpositive recovered noise variance"
        )
    source_intercept = float(mechanism_row @ source_mean)

    # Validate against the same public Gaussian conditional moments used for
    # graph-resolved nodes.  Hidden structural metadata is neither needed nor
    # fabricated here.
    if parents:
        parent_index = np.array(parents)
        parent_covariance = covariance[np.ix_(parent_index, parent_index)]
        regression = np.linalg.solve(parent_covariance, covariance[parent_index, node])
        coefficient_error = float(np.max(np.abs(regression - incoming[parent_index])))
        conditional_noise = float(
            covariance[node, node]
            - covariance[node, parent_index] @ incoming[parent_index]
        )
    else:
        coefficient_error = 0.0
        conditional_noise = float(covariance[node, node])
    conditional_intercept = float(source_mean[node] - incoming @ source_mean)
    noise_error = abs(conditional_noise - noise_variance)
    intercept_error = abs(conditional_intercept - source_intercept)
    check_scale = max(
        1.0,
        abs(noise_variance),
        float(np.max(np.abs(incoming))),
        abs(source_intercept),
    )
    check_tolerance = validation_atol + validation_rtol * check_scale
    if max(coefficient_error, noise_error, intercept_error) > check_tolerance:
        raise RuntimeError(
            "recovered mechanism is inconsistent with public conditional moments"
        )

    return RecoveredLocalMechanism(
        node=node,
        parents=parents,
        incoming_coefficients=incoming,
        mechanism_row=mechanism_row,
        noise_variance=float(noise_variance),
        source_intercept=source_intercept,
        root=not parents,
        total_effect_column=total_effect_column,
        diagnostics=LocalMechanismDiagnostics(
            precision_route=precision_route,
            source_covariance_condition_number=covariance_condition,
            raw_self_precision_effect=self_entry,
            normalized_self_error=self_error,
            coefficient_support_tolerance=support_tolerance,
            max_selected_coefficient_regression_error=coefficient_error,
            noise_variance_reconstruction_error=noise_error,
            intercept_reconstruction_error=intercept_error,
            max_rejected_coefficient_magnitude=max_rejected,
        ),
        provenance=(
            (
                "public_source_moments",
                "one_exact_uncalibrated_additive_probe_mean",
                "precision_times_mean_response_normalization",
                "source_variance_from_observational_covariance",
                "no_hidden_structural_inputs",
            )
            if shift is None
            else (
                "public_source_moments",
                "one_exact_singleton_shift_mean",
                "precision_times_total_effect_identity",
                "no_hidden_structural_inputs",
            )
        ),
    )


def recover_local_mechanism_from_additive_probe(
    *,
    node: int,
    source_mean: np.ndarray,
    exact_shifted_mean: np.ndarray,
    source_covariance: np.ndarray | None = None,
    source_precision: np.ndarray | None = None,
    coefficient_support_tolerance: float = DEFAULT_BETA_MIN,
    validation_atol: float = 1e-11,
    validation_rtol: float = 1e-10,
) -> RecoveredLocalMechanism:
    """Recover a source local mechanism without a calibrated probe magnitude."""
    return recover_local_mechanism_from_singleton_shift(
        node=node,
        source_mean=source_mean,
        exact_shifted_mean=exact_shifted_mean,
        shift_magnitude=None,
        source_covariance=source_covariance,
        source_precision=source_precision,
        coefficient_support_tolerance=coefficient_support_tolerance,
        validation_atol=validation_atol,
        validation_rtol=validation_rtol,
    )


def orient_incident_edges_from_local_mechanism(
    graph: PartiallyDirectedGraph,
    mechanism: RecoveredLocalMechanism,
) -> IncidentOrientationResult:
    """Orient every ambiguous edge incident to the recovered node, then close."""
    if not isinstance(graph, PartiallyDirectedGraph):
        raise TypeError("graph must be a PartiallyDirectedGraph")
    if not isinstance(mechanism, RecoveredLocalMechanism):
        raise TypeError("mechanism must be a RecoveredLocalMechanism")
    node = mechanism.node
    if node not in graph.nodes:
        raise ValueError("recovered node is not present in graph")
    parents = frozenset(mechanism.parents)
    adjacency = graph.adjacent_nodes(node)
    if not parents <= adjacency:
        raise ValueError("recovered parent support is absent from the public skeleton")
    if not graph.parents_of(node) <= parents:
        raise ValueError("recovered parents conflict with an existing incoming edge")
    if graph.children_of(node) & parents:
        raise ValueError("recovered parents conflict with an existing outgoing edge")

    direct = tuple(
        sorted(
            (neighbor, node) if neighbor in parents else (node, neighbor)
            for neighbor in graph.undirected_neighbors_of(node)
        )
    )
    updated = graph.copy()
    for source, target in direct:
        try:
            updated = updated.orient(source, target)
        except ValueError as error:
            raise ValueError(
                f"local-mechanism orientation {source}->{target} is inconsistent: {error}"
            ) from error
    before_meek = updated.directed_edges
    try:
        closed = apply_meek_closure(updated)
    except ValueError as error:
        raise ValueError(
            f"local-mechanism orientations are inconsistent under Meek closure: {error}"
        ) from error
    meek = tuple(sorted(closed.directed_edges - before_meek))
    return IncidentOrientationResult(closed, direct, meek)
