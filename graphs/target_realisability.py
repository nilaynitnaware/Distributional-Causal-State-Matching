"""Analytic validation of Gaussian targets against linear intervention families."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np

from .interventions import ShiftScaleIntervention
from .scm import LinearGaussianSCM

DEFAULT_TARGET_VALIDATION_RTOL = 1e-8
DEFAULT_TARGET_VALIDATION_ATOL = 1e-8


@dataclass(frozen=True)
class GaussianDAGParameters:
    """Conditional linear-Gaussian parameters recovered for a supplied DAG."""

    intercepts: np.ndarray
    coefficients: np.ndarray
    residual_covariance: np.ndarray

    @property
    def residual_variances(self) -> np.ndarray:
        return np.diag(self.residual_covariance)


@dataclass(frozen=True)
class TargetRealisabilityValidation:
    """Fail-loud audit record for a corrected coefficient target."""

    selected_node: int
    selected_parent: int
    selected_node_indegree: int
    source_incoming_coefficients: tuple[float, ...]
    allowed_mixed_incoming_coefficients: tuple[float, ...]
    final_target_incoming_coefficients: tuple[float, ...]
    incoming_parents: tuple[int, ...]
    multiplier: float
    source_residual_variances: tuple[float, ...]
    target_residual_variances: tuple[float, ...]
    markov_with_respect_to_source_dag: bool
    residual_variances_preserved: bool
    selected_node_positive_proportionality: bool
    node_wise_shift_scale_realisable: bool
    intercept_coefficient_realisable: bool
    intervention_family_realisability_classification: str
    minimum_required_coordinate_support: int | None
    minimum_required_intercept_coefficient_support: int | None
    exactly_one_extra_coefficient_modified: bool
    other_incoming_coefficient_unchanged: bool
    no_scale_coordinate_on_root: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def gaussian_dag_parameters(
    dag: nx.DiGraph,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> GaussianDAGParameters:
    """Recover Gaussian conditionals and mutually comparable residuals on ``dag``."""
    if not isinstance(dag, nx.DiGraph) or not nx.is_directed_acyclic_graph(dag):
        raise ValueError("dag must be a directed acyclic graph.")
    n_nodes = dag.number_of_nodes()
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if mean.shape != (n_nodes,) or covariance.shape != (n_nodes, n_nodes):
        raise ValueError("mean/covariance dimensions must agree with the DAG.")
    coefficients = np.zeros((n_nodes, n_nodes), dtype=float)
    intercepts = np.zeros(n_nodes, dtype=float)
    for child in nx.lexicographical_topological_sort(dag):
        parents = np.asarray(sorted(int(parent) for parent in dag.predecessors(child)))
        if parents.size:
            incoming = np.linalg.solve(
                covariance[np.ix_(parents, parents)], covariance[parents, child]
            )
            coefficients[parents, child] = incoming
            intercepts[child] = mean[child] - incoming @ mean[parents]
        else:
            intercepts[child] = mean[child]
    residual_map = np.eye(n_nodes) - coefficients.T
    residual_covariance = residual_map @ covariance @ residual_map.T
    residual_covariance = 0.5 * (residual_covariance + residual_covariance.T)
    return GaussianDAGParameters(intercepts, coefficients, residual_covariance)


def positive_proportionality(
    source: np.ndarray,
    target: np.ndarray,
    *,
    rtol: float = DEFAULT_TARGET_VALIDATION_RTOL,
    atol: float = DEFAULT_TARGET_VALIDATION_ATOL,
) -> bool:
    """Return whether ``target = s * source`` for one finite scalar ``s > 0``."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 1 or source.size == 0:
        raise ValueError("proportionality requires equally shaped nonempty vectors.")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        return False
    nonzero = np.abs(source) > atol
    if not np.any(nonzero) or np.any(np.abs(target[~nonzero]) > atol):
        return False
    ratios = target[nonzero] / source[nonzero]
    return bool(
        np.all(ratios > 0.0)
        and np.allclose(ratios, ratios[0], rtol=rtol, atol=atol)
    )


def validate_nonproportional_coefficient_target(
    source_scm: LinearGaussianSCM,
    allowed_intervention: ShiftScaleIntervention,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    *,
    selected_node: int,
    selected_parent: int,
    multiplier: float,
    configured_support_budget: int | None = None,
    rtol: float = DEFAULT_TARGET_VALIDATION_RTOL,
    atol: float = DEFAULT_TARGET_VALIDATION_ATOL,
) -> TargetRealisabilityValidation:
    """Validate every defining property of the corrected coefficient target."""
    source_scm.validate_target_intervention(allowed_intervention)
    if not np.isfinite(multiplier) or multiplier <= 0.0 or np.isclose(multiplier, 1.0):
        raise ValueError("multiplier must be finite, positive, and different from one.")
    parents = tuple(sorted(int(parent) for parent in source_scm.dag.predecessors(selected_node)))
    if selected_parent not in parents:
        raise ValueError("selected_parent must be an incoming parent of selected_node.")
    if len(parents) < 2:
        raise ValueError("selected_node must have at least two nonzero parents.")

    source = np.asarray(source_scm.coefficients, dtype=float)
    allowed = source * np.asarray(allowed_intervention.scale)[None, :]
    expected_final = np.array(allowed, copy=True)
    expected_final[selected_parent, selected_node] *= multiplier
    parameters = gaussian_dag_parameters(
        source_scm.dag, np.asarray(target_mean), np.asarray(target_covariance)
    )
    final = parameters.coefficients
    changed = ~np.isclose(final, allowed, rtol=rtol, atol=atol)
    exactly_one = bool(
        int(changed.sum()) == 1
        and changed[selected_parent, selected_node]
        and np.allclose(final, expected_final, rtol=rtol, atol=atol)
    )
    other_parents = tuple(parent for parent in parents if parent != selected_parent)
    other_unchanged = bool(
        other_parents
        and np.allclose(
            final[list(other_parents), selected_node],
            allowed[list(other_parents), selected_node],
            rtol=rtol,
            atol=atol,
        )
    )
    off_diagonal = parameters.residual_covariance - np.diag(
        parameters.residual_variances
    )
    markov = bool(np.allclose(off_diagonal, 0.0, rtol=rtol, atol=atol))
    source_residual_variances = np.asarray(source_scm.noise_variances)
    residuals_preserved = bool(
        np.allclose(
            parameters.residual_variances,
            source_residual_variances,
            rtol=rtol,
            atol=atol,
        )
    )
    selected_source = source[list(parents), selected_node]
    selected_final = final[list(parents), selected_node]
    selected_proportional = positive_proportionality(
        selected_source, selected_final, rtol=rtol, atol=atol
    )
    roots = tuple(node for node, degree in source_scm.dag.in_degree if degree == 0)
    no_root_scale = bool(
        np.allclose(allowed_intervention.scale[list(roots)], 1.0, rtol=rtol, atol=atol)
    )

    every_node_proportional = True
    scale_support = 0
    for node, degree in source_scm.dag.in_degree:
        if degree == 0:
            continue
        node_parents = sorted(int(parent) for parent in source_scm.dag.predecessors(node))
        source_incoming = source[node_parents, node]
        final_incoming = final[node_parents, node]
        if not positive_proportionality(
            source_incoming, final_incoming, rtol=rtol, atol=atol
        ):
            every_node_proportional = False
            continue
        ratio = float(final_incoming[0] / source_incoming[0])
        scale_support += int(not np.isclose(ratio, 1.0, rtol=rtol, atol=atol))
    intercepts_preserved = bool(
        np.allclose(
            parameters.intercepts,
            allowed_intervention.shift,
            rtol=rtol,
            atol=atol,
        )
    )
    node_wise_realisable = bool(
        markov
        and residuals_preserved
        and every_node_proportional
        and no_root_scale
        and intercepts_preserved
    )
    stronger_realisable = bool(markov and residuals_preserved and intercepts_preserved)
    shift_support = int(
        np.count_nonzero(
            ~np.isclose(parameters.intercepts, 0.0, rtol=rtol, atol=atol)
        )
    )
    stronger_support = shift_support + int(
        np.count_nonzero(~np.isclose(final, source, rtol=rtol, atol=atol))
    )
    minimum_node_wise_support = (
        shift_support + scale_support if node_wise_realisable else None
    )

    checks = {
        "selected node has at least two nonzero parents": len(parents) >= 2,
        "exactly one coefficient received the extra multiplier": exactly_one,
        "another incoming coefficient remained unchanged": other_unchanged,
        "target is Markov with respect to source DAG": markov,
        "target residual variances equal source residual variances": residuals_preserved,
        "positive proportionality fails at selected node": not selected_proportional,
        "target is non-realisable under node-wise shift-scale": not node_wise_realisable,
        "target is reachable under intercept-coefficient interventions": stronger_realisable,
        "no scale coordinate is assigned to a root": no_root_scale,
        "target intercepts preserve the allowed mixed intervention": intercepts_preserved,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("invalid non-proportional coefficient target: " + "; ".join(failed))

    return TargetRealisabilityValidation(
        selected_node=int(selected_node),
        selected_parent=int(selected_parent),
        selected_node_indegree=len(parents),
        source_incoming_coefficients=tuple(float(source[parent, selected_node]) for parent in parents),
        allowed_mixed_incoming_coefficients=tuple(float(allowed[parent, selected_node]) for parent in parents),
        final_target_incoming_coefficients=tuple(float(final[parent, selected_node]) for parent in parents),
        incoming_parents=parents,
        multiplier=float(multiplier),
        source_residual_variances=tuple(float(value) for value in source_residual_variances),
        target_residual_variances=tuple(float(value) for value in parameters.residual_variances),
        markov_with_respect_to_source_dag=markov,
        residual_variances_preserved=residuals_preserved,
        selected_node_positive_proportionality=selected_proportional,
        node_wise_shift_scale_realisable=node_wise_realisable,
        intercept_coefficient_realisable=stronger_realisable,
        intervention_family_realisability_classification=(
            "non_realisable_node_wise_shift_scale_due_to_nonproportional_coefficients"
        ),
        minimum_required_coordinate_support=minimum_node_wise_support,
        minimum_required_intercept_coefficient_support=stronger_support,
        exactly_one_extra_coefficient_modified=exactly_one,
        other_incoming_coefficient_unchanged=other_unchanged,
        no_scale_coordinate_on_root=no_root_scale,
    )
