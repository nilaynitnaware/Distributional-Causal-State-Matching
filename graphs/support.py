"""Central support semantics for shift and log-scale coordinates."""

from numbers import Real

import numpy as np

DEFAULT_SUPPORT_ATOL = 1e-9


def validate_support_atol(value: object) -> float:
    """Validate and normalize an absolute support tolerance."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("support_atol must be a real number.")
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("support_atol must be finite and nonnegative.")
    return tolerance


def coordinate_support_masks(
    shift: np.ndarray,
    scale: np.ndarray,
    *,
    support_atol: object = DEFAULT_SUPPORT_ATOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Return active shift and log-scale masks under one tolerance rule."""
    if not isinstance(shift, np.ndarray) or not isinstance(scale, np.ndarray):
        raise TypeError("shift and scale must be numpy.ndarray objects.")
    if shift.ndim != 1 or scale.ndim != 1 or shift.shape != scale.shape:
        raise ValueError("shift and scale must be one-dimensional with equal shape.")
    if np.any(scale <= 0.0):
        raise ValueError("all scales must be strictly positive.")

    tolerance = validate_support_atol(support_atol)
    shifted = ~np.isclose(shift, 0.0, atol=tolerance, rtol=0.0)
    log_scale = np.log(scale)
    scaled = ~np.isclose(log_scale, 0.0, atol=tolerance, rtol=0.0)
    return shifted, scaled


def targeted_nodes(
    shift: np.ndarray,
    scale: np.ndarray,
    *,
    support_atol: object = DEFAULT_SUPPORT_ATOL,
) -> frozenset[int]:
    """Return nodes with at least one active intervention coordinate."""
    shifted, scaled = coordinate_support_masks(
        shift,
        scale,
        support_atol=support_atol,
    )
    active = np.flatnonzero(shifted | scaled)
    return frozenset(int(node) for node in active)


def parameter_support_cost(
    shift: np.ndarray,
    scale: np.ndarray,
    *,
    support_atol: object = DEFAULT_SUPPORT_ATOL,
) -> int:
    """Count active shift and log-scale parameters separately."""
    shifted, scaled = coordinate_support_masks(
        shift,
        scale,
        support_atol=support_atol,
    )
    return int(np.count_nonzero(shifted) + np.count_nonzero(scaled))
