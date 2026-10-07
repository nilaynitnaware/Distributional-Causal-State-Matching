"""Non-realisable Gaussian target families with public-only outputs."""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np

from .interventions import ShiftScaleIntervention
from .scm import LinearGaussianSCM
from .target_gen import GaussianTarget
from .target_realisability import validate_nonproportional_coefficient_target


def nonproportional_coefficient_target(
    original_scm: LinearGaussianSCM,
    hidden_target_intervention: ShiftScaleIntervention,
    *,
    selected_node: int,
    selected_parent: int,
    multiplier: float = 1.35,
    configured_support_budget: int | None = None,
) -> GaussianTarget:
    """Modify one incoming coefficient after the allowed mixed target is defined."""
    if not isinstance(original_scm, LinearGaussianSCM):
        raise TypeError("original_scm must be LinearGaussianSCM.")
    modified_coefficients = np.array(original_scm.coefficients, copy=True)
    modified_coefficients[selected_parent, selected_node] *= multiplier
    alternative = LinearGaussianSCM(
        nx.DiGraph(original_scm.dag),
        modified_coefficients,
        beta_min=original_scm.beta_min,
        noise_variances=original_scm.noise_variances,
    )
    mean, covariance = alternative.interventional_moments(hidden_target_intervention)
    validation = validate_nonproportional_coefficient_target(
        original_scm,
        hidden_target_intervention,
        mean,
        covariance,
        selected_node=selected_node,
        selected_parent=selected_parent,
        multiplier=multiplier,
        configured_support_budget=configured_support_budget,
    )
    metadata = validation.as_dict()
    metadata.update(
        {
            "target_family": "nonproportional_structural_coefficients",
            "configured_support_budget": configured_support_budget,
            "coefficient_modification_stage": (
                "after_allowed_sparse_mixed_shift_scale_target"
            ),
        }
    )
    return GaussianTarget(
        mean=mean,
        covariance=covariance,
        generating_intervention=None,
        is_realisable=False,
        public_metadata=metadata,
    )


def graph_perturbed_target(
    hidden_alternative_scm: LinearGaussianSCM,
    hidden_target_intervention: ShiftScaleIntervention,
) -> GaussianTarget:
    """Generate public moments from a hidden graph-perturbed SCM."""
    return _hidden_alternative_target(
        hidden_alternative_scm,
        hidden_target_intervention,
        family="graph_perturbed_linear_gaussian",
    )


def correlated_noise_target(
    original_scm: LinearGaussianSCM,
    noise_covariance: np.ndarray,
    hidden_target_intervention: ShiftScaleIntervention,
) -> GaussianTarget:
    """Generate moments with hidden correlated exogenous Gaussian noise."""
    if not isinstance(original_scm, LinearGaussianSCM):
        raise TypeError("original_scm must be LinearGaussianSCM.")
    if not isinstance(noise_covariance, np.ndarray):
        raise TypeError("noise_covariance must be a numpy.ndarray.")
    n_nodes = original_scm.dag.number_of_nodes()
    if noise_covariance.shape != (n_nodes, n_nodes):
        raise ValueError("noise_covariance dimension must match the SCM.")
    if not np.allclose(noise_covariance, noise_covariance.T):
        raise ValueError("noise_covariance must be symmetric.")
    try:
        np.linalg.cholesky(noise_covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("noise_covariance must be positive definite.") from error
    if not isinstance(hidden_target_intervention, ShiftScaleIntervention):
        raise TypeError("hidden_target_intervention must be ShiftScaleIntervention.")
    original_scm.validate_target_intervention(hidden_target_intervention)
    scale = hidden_target_intervention.scale
    system = np.eye(n_nodes) - scale[:, None] * original_scm.coefficients.T
    mean = np.linalg.solve(system, hidden_target_intervention.shift)
    transformed_noise = np.linalg.solve(system, np.eye(n_nodes))
    covariance = transformed_noise @ noise_covariance @ transformed_noise.T
    return GaussianTarget(
        mean=mean,
        covariance=covariance,
        generating_intervention=None,
        is_realisable=False,
        public_metadata={"target_family": "correlated_noise_gaussian"},
    )


def external_gaussian_target(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    public_metadata: Mapping[str, object] | None = None,
) -> GaussianTarget:
    """Wrap an externally supplied positive-definite Gaussian target."""
    metadata = {"target_family": "external_spd_gaussian"}
    if public_metadata is not None:
        metadata.update(dict(public_metadata))
    return GaussianTarget(
        mean=mean,
        covariance=covariance,
        generating_intervention=None,
        is_realisable=False,
        public_metadata=metadata,
    )


def _hidden_alternative_target(
    hidden_scm: LinearGaussianSCM,
    hidden_intervention: ShiftScaleIntervention,
    *,
    family: str,
) -> GaussianTarget:
    if not isinstance(hidden_scm, LinearGaussianSCM):
        raise TypeError("hidden_scm must be LinearGaussianSCM.")
    if not isinstance(hidden_intervention, ShiftScaleIntervention):
        raise TypeError("hidden_intervention must be ShiftScaleIntervention.")
    hidden_scm.validate_target_intervention(hidden_intervention)
    mean, covariance = hidden_scm.interventional_moments(hidden_intervention)
    return GaussianTarget(
        mean=mean,
        covariance=covariance,
        generating_intervention=None,
        is_realisable=False,
        public_metadata={"target_family": family},
    )
