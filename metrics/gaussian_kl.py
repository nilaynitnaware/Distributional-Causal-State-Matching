"""Stable KL divergence between multivariate Gaussian distributions."""

import numpy as np


def _validated_gaussian(
    name: str,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate Gaussian parameters and return float arrays plus Cholesky."""
    if not isinstance(mean, np.ndarray):
        raise TypeError(f"{name}_mean must be a numpy.ndarray.")
    if not isinstance(covariance, np.ndarray):
        raise TypeError(f"{name}_covariance must be a numpy.ndarray.")
    if mean.ndim != 1:
        raise ValueError(f"{name}_mean must be one-dimensional.")
    if covariance.ndim != 2 or (covariance.shape[0] != covariance.shape[1]):
        raise ValueError(f"{name}_covariance must be a square matrix.")
    if covariance.shape != (mean.size, mean.size):
        raise ValueError(f"{name} mean and covariance dimensions must agree.")
    arrays = {f"{name}_mean": mean, f"{name}_covariance": covariance}
    for array_name, array in arrays.items():
        if not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array):
            raise TypeError(f"{array_name} must contain real numeric values.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{array_name} must contain only finite values.")
    mean_float = np.asarray(mean, dtype=float)
    covariance_float = np.asarray(covariance, dtype=float)
    if not np.allclose(
        covariance_float,
        covariance_float.T,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(f"{name}_covariance must be symmetric.")
    if mean.size:
        try:
            cholesky = np.linalg.cholesky(covariance_float)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{name}_covariance must be positive definite.") from error
    else:
        cholesky = np.empty((0, 0), dtype=float)
    return mean_float, covariance_float, cholesky


def gaussian_kl_target_to_candidate(
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_covariance: np.ndarray,
) -> float:
    """Compute ``D_KL(Q_target || P_candidate)``.

    The expectation is taken under the target distribution. Cholesky factors
    provide log determinants, and triangular solves with the candidate factor
    provide the trace and Mahalanobis terms. No covariance inverse is formed.
    """
    validated_target_mean, _, target_cholesky = _validated_gaussian(
        "target",
        target_mean,
        target_covariance,
    )
    validated_candidate_mean, _, candidate_cholesky = _validated_gaussian(
        "candidate",
        candidate_mean,
        candidate_covariance,
    )
    if validated_target_mean.shape != validated_candidate_mean.shape:
        raise ValueError("target and candidate dimensions must agree.")
    if not validated_target_mean.size:
        return 0.0
    whitened_covariance_factor = np.linalg.solve(
        candidate_cholesky,
        target_cholesky,
    )
    trace_term = float(np.sum(whitened_covariance_factor**2))
    residual = validated_candidate_mean - validated_target_mean
    whitened_residual = np.linalg.solve(candidate_cholesky, residual)
    quadratic_term = float(whitened_residual @ whitened_residual)
    log_determinant_target = 2.0 * float(np.log(np.diag(target_cholesky)).sum())
    log_determinant_candidate = 2.0 * float(np.log(np.diag(candidate_cholesky)).sum())
    raw_value = 0.5 * (
        trace_term
        + quadratic_term
        - validated_target_mean.size
        + log_determinant_candidate
        - log_determinant_target
    )
    numerical_scale = max(
        1.0,
        abs(trace_term),
        abs(quadratic_term),
        abs(log_determinant_target),
        abs(log_determinant_candidate),
        float(validated_target_mean.size),
    )
    roundoff_tolerance = 64.0 * np.finfo(float).eps * numerical_scale
    if raw_value < -roundoff_tolerance:
        raise RuntimeError(
            "computed Gaussian KL is negative beyond numerical roundoff."
        )
    if abs(raw_value) <= roundoff_tolerance:
        return 0.0
    return float(raw_value)
