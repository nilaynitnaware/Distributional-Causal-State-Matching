from __future__ import annotations

import inspect

import numpy as np

from research.calibration_free_singleton_diagnostic.estimator import (
    fit_calibrated,
    fit_calibration_free,
)


def _inputs(response: np.ndarray, replicates: int = 8) -> dict[str, object]:
    n = response.size
    return {
        "node": 1,
        "source_mean": np.zeros(n),
        "source_covariance": np.eye(n),
        "source_precision": np.eye(n),
        "mean_response": response,
        "bootstrap_source_means": np.zeros((replicates, n)),
        "bootstrap_precisions": np.repeat(np.eye(n)[None, :, :], replicates, axis=0),
        "bootstrap_intervention_means": np.repeat(response[None, :], replicates, axis=0),
        "confidence_level": 0.95,
        "orientation_confidence": 0.95,
        "support_probability_threshold": 0.5,
    }


def test_calibration_free_signature_has_no_query_magnitude() -> None:
    parameters = set(inspect.signature(fit_calibration_free).parameters)
    assert not parameters & {"h", "signed_h", "shift_magnitude", "query_magnitude"}


def test_raw_rows_match_calibrated_for_positive_and_negative_queries() -> None:
    true_row = np.array([-0.4, 1.0, 0.25])
    for h in (-2.0, -0.05, 0.1, 1.5):
        response = h * true_row
        common = _inputs(response)
        calibrated = fit_calibrated(
            signed_h=h,
            minimum_self_precision_effect=1e-10,
            **common,
        )
        uncalibrated = fit_calibration_free(
            minimum_abs_denominator=1e-10,
            **common,
        )
        assert calibrated.valid
        assert uncalibrated.valid
        np.testing.assert_allclose(calibrated.raw_mechanism_row, true_row)
        np.testing.assert_allclose(uncalibrated.raw_mechanism_row, true_row)
        np.testing.assert_allclose(
            calibrated.raw_mechanism_row, uncalibrated.raw_mechanism_row
        )


def test_negative_uncalibrated_denominator_is_valid() -> None:
    response = np.array([0.2, -0.5, 0.0])
    fit = fit_calibration_free(minimum_abs_denominator=1e-10, **_inputs(response))
    assert fit.valid
    assert fit.signed_denominator < 0.0


def test_near_zero_denominator_is_explicitly_invalid() -> None:
    response = np.zeros(3)
    fit = fit_calibration_free(minimum_abs_denominator=1e-10, **_inputs(response))
    assert not fit.valid
    assert fit.invalid_reason == "near_zero_or_nonfinite_signed_denominator"
    assert fit.bootstrap_valid_replicates == 0


def test_variance_definitions_are_explicitly_different() -> None:
    response = np.array([-0.2, 0.5, 0.1])
    common = _inputs(response)
    calibrated = fit_calibrated(
        signed_h=0.5,
        minimum_self_precision_effect=1e-10,
        **common,
    )
    uncalibrated = fit_calibration_free(
        minimum_abs_denominator=1e-10,
        **common,
    )
    assert calibrated.noise_variance == 1.0
    assert uncalibrated.noise_variance == np.dot(
        uncalibrated.selected_mechanism_row,
        uncalibrated.selected_mechanism_row,
    )
    assert calibrated.noise_variance != uncalibrated.noise_variance
