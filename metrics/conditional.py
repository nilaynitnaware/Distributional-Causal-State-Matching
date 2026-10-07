"""Conditional target-to-candidate Gaussian discrepancies."""

from collections.abc import Iterable
from numbers import Integral

import numpy as np

from .gaussian_kl import _validated_gaussian


def conditional_gaussian_kl_target_to_candidate(
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_covariance: np.ndarray,
    node: int,
    parents: Iterable[int],
) -> float:
    """Return target-averaged ``KL(Q(X_i|X_pa) || P(X_i|X_pa))``.

    The conditional KL is averaged over the target marginal ``Q(X_pa)``.
    Empty parent sets reduce exactly to the scalar marginal Gaussian KL.
    """
    validated_target_mean, validated_target_covariance, _ = _validated_gaussian(
        "target",
        target_mean,
        target_covariance,
    )
    validated_candidate_mean, validated_candidate_covariance, _ = _validated_gaussian(
        "candidate",
        candidate_mean,
        candidate_covariance,
    )
    if validated_target_mean.shape != validated_candidate_mean.shape:
        raise ValueError("target and candidate dimensions must agree.")
    if isinstance(node, bool) or not isinstance(node, Integral):
        raise TypeError("node must be an integer.")
    normalized_node = int(node)
    if not 0 <= normalized_node < validated_target_mean.size:
        raise ValueError("node must index the Gaussian.")
    try:
        parent_values = tuple(parents)
    except TypeError as error:
        raise TypeError(
            "parents must be an iterable of integer node indices."
        ) from error
    if any(
        isinstance(parent, bool) or not isinstance(parent, Integral)
        for parent in parent_values
    ):
        raise TypeError("parents must contain integer node indices.")
    normalized_parents = tuple(int(parent) for parent in parent_values)
    if len(set(normalized_parents)) != len(normalized_parents):
        raise ValueError("parents must not contain duplicates.")
    if normalized_node in normalized_parents:
        raise ValueError("node must not be its own parent.")
    if any(
        parent < 0 or parent >= validated_target_mean.size
        for parent in normalized_parents
    ):
        raise ValueError("parents must index the Gaussian.")

    parent_indices = np.array(normalized_parents, dtype=int)
    if not normalized_parents:
        target_variance = validated_target_covariance[
            normalized_node,
            normalized_node,
        ]
        candidate_variance = validated_candidate_covariance[
            normalized_node,
            normalized_node,
        ]
        mean_difference = (
            validated_candidate_mean[normalized_node]
            - validated_target_mean[normalized_node]
        )
        return float(
            0.5
            * (
                target_variance / candidate_variance
                + mean_difference**2 / candidate_variance
                - 1.0
                + np.log(candidate_variance)
                - np.log(target_variance)
            )
        )

    target_parent_covariance = validated_target_covariance[
        np.ix_(parent_indices, parent_indices)
    ]
    candidate_parent_covariance = validated_candidate_covariance[
        np.ix_(parent_indices, parent_indices)
    ]
    target_regression = np.linalg.solve(
        target_parent_covariance,
        validated_target_covariance[parent_indices, normalized_node],
    )
    candidate_regression = np.linalg.solve(
        candidate_parent_covariance,
        validated_candidate_covariance[parent_indices, normalized_node],
    )
    target_conditional_variance = float(
        validated_target_covariance[normalized_node, normalized_node]
        - validated_target_covariance[normalized_node, parent_indices]
        @ target_regression
    )
    candidate_conditional_variance = float(
        validated_candidate_covariance[normalized_node, normalized_node]
        - validated_candidate_covariance[normalized_node, parent_indices]
        @ candidate_regression
    )
    if target_conditional_variance <= 0.0 or candidate_conditional_variance <= 0.0:
        raise RuntimeError("conditional variances must be positive.")

    target_intercept = float(
        validated_target_mean[normalized_node]
        - target_regression @ validated_target_mean[parent_indices]
    )
    candidate_intercept = float(
        validated_candidate_mean[normalized_node]
        - candidate_regression @ validated_candidate_mean[parent_indices]
    )
    regression_difference = candidate_regression - target_regression
    intercept_difference = candidate_intercept - target_intercept
    target_parent_mean = validated_target_mean[parent_indices]
    conditional_mean_difference = (
        intercept_difference + regression_difference @ target_parent_mean
    )
    expected_squared_mean_difference = float(
        regression_difference @ target_parent_covariance @ regression_difference
        + conditional_mean_difference**2
    )
    raw_value = 0.5 * (
        (target_conditional_variance + expected_squared_mean_difference)
        / candidate_conditional_variance
        - 1.0
        + np.log(candidate_conditional_variance)
        - np.log(target_conditional_variance)
    )
    return float(max(raw_value, 0.0))
