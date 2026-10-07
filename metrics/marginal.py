"""Marginal target-to-candidate Gaussian discrepancies."""

from numbers import Integral

import numpy as np

from .gaussian_kl import gaussian_kl_target_to_candidate


def marginal_gaussian_kl_target_to_candidate(
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_covariance: np.ndarray,
    node: int,
) -> float:
    """Return ``KL(Q(X_node) || P(X_node))``."""
    if isinstance(node, bool) or not isinstance(node, Integral):
        raise TypeError("node must be an integer.")
    normalized_node = int(node)
    if normalized_node < 0 or normalized_node >= target_mean.size:
        raise ValueError("node must index the target Gaussian.")
    indices = np.array([normalized_node])
    return gaussian_kl_target_to_candidate(
        target_mean[indices],
        target_covariance[np.ix_(indices, indices)],
        candidate_mean[indices],
        candidate_covariance[np.ix_(indices, indices)],
    )
