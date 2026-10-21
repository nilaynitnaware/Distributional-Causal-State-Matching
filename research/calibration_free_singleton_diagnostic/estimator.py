"""Paired calibrated and calibration-free finite-sample mechanism fits.

This module is diagnostic-only.  It deliberately does not alter the production
finite-sample Algorithm 1 estimator.  Both entry points normalize a complete
precision-times-response row before applying the existing graph-free bootstrap
support rule.  The calibration-free entry point has no query-magnitude input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiagnosticMechanismFit:
    """One point fit plus graph-free bootstrap support diagnostics."""

    estimator: str
    node: int
    valid: bool
    invalid_reason: str | None
    signed_denominator: float
    raw_mechanism_row: np.ndarray
    selected_mechanism_row: np.ndarray
    selected_incoming_coefficients: np.ndarray
    selected_parents: tuple[int, ...]
    source_intercept: float
    noise_variance: float
    raw_quadratic_variance: float
    selected_quadratic_variance: float
    bootstrap_valid_replicates: int
    bootstrap_total_replicates: int
    bootstrap_parent_probabilities: np.ndarray
    bootstrap_confidence_intervals: np.ndarray


def _validate_common(
    *,
    node: int,
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    source_precision: np.ndarray,
    mean_response: np.ndarray,
    bootstrap_source_means: np.ndarray,
    bootstrap_precisions: np.ndarray,
    bootstrap_intervention_means: np.ndarray,
) -> tuple[int, int]:
    if source_mean.ndim != 1:
        raise ValueError("source_mean must be one-dimensional")
    n = source_mean.size
    if not 0 <= int(node) < n:
        raise ValueError("node is outside the moment dimension")
    if source_covariance.shape != (n, n) or source_precision.shape != (n, n):
        raise ValueError("source covariance and precision must be n by n")
    if mean_response.shape != (n,):
        raise ValueError("mean_response must have length n")
    if bootstrap_source_means.ndim != 2 or bootstrap_source_means.shape[1] != n:
        raise ValueError("bootstrap_source_means must have shape (R, n)")
    r = bootstrap_source_means.shape[0]
    if bootstrap_precisions.shape != (r, n, n):
        raise ValueError("bootstrap_precisions must have shape (R, n, n)")
    if bootstrap_intervention_means.shape != (r, n):
        raise ValueError("bootstrap_intervention_means must have shape (R, n)")
    arrays = (source_mean, source_covariance, source_precision, mean_response)
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("finite point and bootstrap source inputs are required")
    return n, r


def _support_select(
    *,
    node: int,
    point_row: np.ndarray,
    bootstrap_rows: np.ndarray,
    point_valid: bool,
    confidence_level: float,
    orientation_confidence: float,
    support_probability_threshold: float,
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray]:
    """Apply the production graph-free support rule to complete normalized rows."""
    n = point_row.size
    valid_rows = np.all(np.isfinite(bootstrap_rows), axis=1)
    distribution = -bootstrap_rows[valid_rows]
    distribution[:, node] = 0.0
    point_coefficients = -np.array(point_row, copy=True)
    point_coefficients[node] = 0.0
    probabilities = np.zeros(n)
    intervals = np.full((n, 2), np.nan)
    selected: list[int] = []
    alpha = 1.0 - float(confidence_level)
    for candidate in range(n):
        if candidate == node:
            continue
        values = distribution[:, candidate]
        if values.size:
            intervals[candidate] = np.quantile(
                values, (alpha / 2.0, 1.0 - alpha / 2.0)
            )
            sign_stability = max(
                float(np.mean(values > 0.0)), float(np.mean(values < 0.0))
            )
            probabilities[candidate] = max(
                0.0, min(1.0, 2.0 * sign_stability - 1.0)
            )
        low, high = intervals[candidate]
        excludes_zero = bool(low > 0.0 or high < 0.0)
        contains_zero = bool(low <= 0.0 <= high)
        if (
            point_valid
            and excludes_zero
            and probabilities[candidate] >= orientation_confidence
        ):
            is_selected = True
        elif (
            point_valid
            and contains_zero
            and probabilities[candidate] <= 1.0 - orientation_confidence
        ):
            is_selected = False
        else:
            is_selected = bool(
                point_valid
                and excludes_zero
                and probabilities[candidate] >= support_probability_threshold
            )
        if is_selected:
            selected.append(candidate)
    incoming = np.zeros(n)
    if selected:
        index = np.asarray(selected, dtype=int)
        incoming[index] = point_coefficients[index]
    return incoming, tuple(selected), probabilities, intervals


def _finish(
    *,
    estimator: str,
    node: int,
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    raw_row: np.ndarray,
    signed_denominator: float,
    point_variance: float,
    point_valid: bool,
    invalid_reason: str | None,
    bootstrap_rows: np.ndarray,
    bootstrap_total: int,
    confidence_level: float,
    orientation_confidence: float,
    support_probability_threshold: float,
) -> DiagnosticMechanismFit:
    valid_bootstrap = np.all(np.isfinite(bootstrap_rows), axis=1)
    valid_count = int(valid_bootstrap.sum())
    if point_valid and valid_count < 2:
        point_valid = False
        invalid_reason = "fewer_than_two_valid_bootstrap_replicates"
    incoming, parents, probabilities, intervals = _support_select(
        node=node,
        point_row=raw_row,
        bootstrap_rows=bootstrap_rows,
        point_valid=point_valid,
        confidence_level=confidence_level,
        orientation_confidence=orientation_confidence,
        support_probability_threshold=support_probability_threshold,
    )
    selected_row = np.eye(source_mean.size)[node] - incoming
    intercept = float(selected_row @ source_mean)
    selected_quadratic = float(selected_row @ source_covariance @ selected_row)
    raw_quadratic = (
        float(raw_row @ source_covariance @ raw_row)
        if np.all(np.isfinite(raw_row))
        else float("nan")
    )
    variance = float(point_variance) if point_valid else float("nan")
    return DiagnosticMechanismFit(
        estimator=estimator,
        node=int(node),
        valid=bool(point_valid),
        invalid_reason=invalid_reason,
        signed_denominator=float(signed_denominator),
        raw_mechanism_row=np.array(raw_row, copy=True),
        selected_mechanism_row=selected_row,
        selected_incoming_coefficients=incoming,
        selected_parents=parents,
        source_intercept=intercept,
        noise_variance=variance,
        raw_quadratic_variance=raw_quadratic,
        selected_quadratic_variance=selected_quadratic,
        bootstrap_valid_replicates=valid_count,
        bootstrap_total_replicates=int(bootstrap_total),
        bootstrap_parent_probabilities=probabilities,
        bootstrap_confidence_intervals=intervals,
    )


def fit_calibration_free(
    *,
    node: int,
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    source_precision: np.ndarray,
    mean_response: np.ndarray,
    bootstrap_source_means: np.ndarray,
    bootstrap_precisions: np.ndarray,
    bootstrap_intervention_means: np.ndarray,
    minimum_abs_denominator: float,
    confidence_level: float,
    orientation_confidence: float,
    support_probability_threshold: float,
) -> DiagnosticMechanismFit:
    """Fit without receiving or reconstructing the physical query magnitude."""
    n, r = _validate_common(
        node=node,
        source_mean=source_mean,
        source_covariance=source_covariance,
        source_precision=source_precision,
        mean_response=mean_response,
        bootstrap_source_means=bootstrap_source_means,
        bootstrap_precisions=bootstrap_precisions,
        bootstrap_intervention_means=bootstrap_intervention_means,
    )
    threshold = float(minimum_abs_denominator)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_abs_denominator must be finite and positive")
    what = source_precision @ mean_response
    denominator = float(what[node])
    valid = bool(np.isfinite(denominator) and abs(denominator) > threshold)
    reason = None if valid else "near_zero_or_nonfinite_signed_denominator"
    raw_row = what / denominator if valid else np.full(n, np.nan)
    if valid:
        raw_row = np.array(raw_row, copy=True)
        raw_row[node] = 1.0

    boot_rows = np.full((r, n), np.nan)
    for replicate in range(r):
        if (
            not np.all(np.isfinite(bootstrap_source_means[replicate]))
            or not np.all(np.isfinite(bootstrap_precisions[replicate]))
        ):
            continue
        response = (
            bootstrap_intervention_means[replicate]
            - bootstrap_source_means[replicate]
        )
        boot_what = bootstrap_precisions[replicate] @ response
        boot_denominator = float(boot_what[node])
        if np.isfinite(boot_denominator) and abs(boot_denominator) > threshold:
            boot_rows[replicate] = boot_what / boot_denominator
            boot_rows[replicate, node] = 1.0
    point_variance = (
        float(raw_row @ source_covariance @ raw_row) if valid else float("nan")
    )
    return _finish(
        estimator="calibration_free",
        node=node,
        source_mean=source_mean,
        source_covariance=source_covariance,
        raw_row=raw_row,
        signed_denominator=denominator,
        point_variance=point_variance,
        point_valid=valid,
        invalid_reason=reason,
        bootstrap_rows=boot_rows,
        bootstrap_total=r,
        confidence_level=confidence_level,
        orientation_confidence=orientation_confidence,
        support_probability_threshold=support_probability_threshold,
    )


def fit_calibrated(
    *,
    node: int,
    signed_h: float,
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    source_precision: np.ndarray,
    mean_response: np.ndarray,
    bootstrap_source_means: np.ndarray,
    bootstrap_precisions: np.ndarray,
    bootstrap_intervention_means: np.ndarray,
    minimum_self_precision_effect: float,
    confidence_level: float,
    orientation_confidence: float,
    support_probability_threshold: float,
) -> DiagnosticMechanismFit:
    """Existing calibrated precision-times-total-effect comparator."""
    n, r = _validate_common(
        node=node,
        source_mean=source_mean,
        source_covariance=source_covariance,
        source_precision=source_precision,
        mean_response=mean_response,
        bootstrap_source_means=bootstrap_source_means,
        bootstrap_precisions=bootstrap_precisions,
        bootstrap_intervention_means=bootstrap_intervention_means,
    )
    h = float(signed_h)
    threshold = float(minimum_self_precision_effect)
    if not np.isfinite(h) or h == 0.0:
        raise ValueError("signed_h must be finite and nonzero")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_self_precision_effect must be finite and positive")
    precision_effect = source_precision @ (mean_response / h)
    denominator = float(precision_effect[node])
    valid = bool(np.isfinite(denominator) and denominator > threshold)
    reason = None if valid else "nonpositive_or_nonfinite_self_precision_effect"
    raw_row = precision_effect / denominator if valid else np.full(n, np.nan)
    if valid:
        raw_row = np.array(raw_row, copy=True)
        raw_row[node] = 1.0

    boot_rows = np.full((r, n), np.nan)
    for replicate in range(r):
        if (
            not np.all(np.isfinite(bootstrap_source_means[replicate]))
            or not np.all(np.isfinite(bootstrap_precisions[replicate]))
        ):
            continue
        response = (
            bootstrap_intervention_means[replicate]
            - bootstrap_source_means[replicate]
        )
        boot_effect = bootstrap_precisions[replicate] @ (response / h)
        boot_denominator = float(boot_effect[node])
        if np.isfinite(boot_denominator) and boot_denominator > threshold:
            boot_rows[replicate] = boot_effect / boot_denominator
            boot_rows[replicate, node] = 1.0
    point_variance = 1.0 / denominator if valid else float("nan")
    return _finish(
        estimator="calibrated",
        node=node,
        source_mean=source_mean,
        source_covariance=source_covariance,
        raw_row=raw_row,
        signed_denominator=denominator,
        point_variance=point_variance,
        point_valid=valid,
        invalid_reason=reason,
        bootstrap_rows=boot_rows,
        bootstrap_total=r,
        confidence_level=confidence_level,
        orientation_confidence=orientation_confidence,
        support_probability_threshold=support_probability_threshold,
    )
