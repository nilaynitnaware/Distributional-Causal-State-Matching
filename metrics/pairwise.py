"""Pairwise target-to-candidate Gaussian discrepancies."""

from numbers import Integral

import numpy as np

from .gaussian_kl import gaussian_kl_target_to_candidate


def pairwise_gaussian_kl_target_to_candidate(
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_covariance: np.ndarray,
    first_node: int,
    second_node: int,
) -> float:
    """Return ``KL(Q(X_i, X_j) || P(X_i, X_j))``."""
    nodes = (first_node, second_node)
    if any(isinstance(node, bool) or not isinstance(node, Integral) for node in nodes):
        raise TypeError("pairwise nodes must be integers.")
    normalized = tuple(int(node) for node in nodes)
    if normalized[0] == normalized[1]:
        raise ValueError("pairwise nodes must be distinct.")
    if any(node < 0 or node >= target_mean.size for node in normalized):
        raise ValueError("pairwise nodes must index the target Gaussian.")
    indices = np.array(normalized)
    return gaussian_kl_target_to_candidate(
        target_mean[indices],
        target_covariance[np.ix_(indices, indices)],
        candidate_mean[indices],
        candidate_covariance[np.ix_(indices, indices)],
    )
