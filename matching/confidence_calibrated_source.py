"""Confidence-interval probe-source classification using empirical samples only."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from statistics import NormalDist

import numpy as np
from scipy.stats import t as student_t

from .source_inference import (
    FiniteSourceInferenceSettings,
    SourceClassification,
    _response_statistics,
)


def _nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


@dataclass(frozen=True)
class ConfidenceCalibratedSourceSettings:
    """Prespecified residual-response confidence-interval settings."""

    equivalence_absolute: float
    equivalence_relative: float
    confidence_level: float = 0.95
    minimum_sample_size: int = 30

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "equivalence_absolute",
            _nonnegative("equivalence_absolute", self.equivalence_absolute),
        )
        object.__setattr__(
            self,
            "equivalence_relative",
            _nonnegative("equivalence_relative", self.equivalence_relative),
        )
        confidence = _nonnegative("confidence_level", self.confidence_level)
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie in (0, 1).")
        object.__setattr__(self, "confidence_level", confidence)
        if (
            isinstance(self.minimum_sample_size, bool)
            or not isinstance(self.minimum_sample_size, int)
            or self.minimum_sample_size < 2
        ):
            raise ValueError("minimum_sample_size must be at least two.")


@dataclass(frozen=True)
class ConfidenceCalibratedSourceClassification:
    """One three-way CI-equivalence decision and its empirical evidence."""

    node: int
    classification: SourceClassification
    probe_increment: float
    estimated_response: float
    response_error: float
    standard_error: float
    degrees_of_freedom: float
    confidence_interval: tuple[float, float]
    equivalence_interval: tuple[float, float]
    confidence_level: float
    baseline_sample_size: int
    probe_sample_size: int
    minimum_sample_size: int
    minimum_evidence_met: bool
    background_id: str
    paired_samples: bool
    reason: str


def classify_confidence_calibrated_source_targets(
    baseline_samples: np.ndarray,
    probe_samples: np.ndarray,
    shift_increment: np.ndarray,
    targets: frozenset[int],
    *,
    background_id: str,
    settings: ConfidenceCalibratedSourceSettings,
    paired_samples: bool = False,
) -> tuple[ConfidenceCalibratedSourceClassification, ...]:
    """Apply a full-CI-inside/full-CI-outside equivalence rule."""
    if not isinstance(settings, ConfidenceCalibratedSourceSettings):
        raise TypeError("settings must be ConfidenceCalibratedSourceSettings.")
    if not isinstance(background_id, str) or not background_id:
        raise ValueError("background_id must be nonempty.")
    for name, samples in (("baseline", baseline_samples), ("probe", probe_samples)):
        if not isinstance(samples, np.ndarray) or samples.ndim != 2:
            raise TypeError(f"{name} samples must be a two-dimensional array.")
        if samples.shape[0] < 2 or not np.all(np.isfinite(samples)):
            raise ValueError(f"{name} samples must contain finite observations.")
    if baseline_samples.shape[1] != probe_samples.shape[1]:
        raise ValueError("sample dimensions must agree.")
    if not isinstance(shift_increment, np.ndarray) or shift_increment.shape != (
        baseline_samples.shape[1],
    ):
        raise ValueError("shift_increment has the wrong dimension.")
    if not isinstance(targets, frozenset) or any(
        isinstance(node, bool) or not isinstance(node, int) for node in targets
    ):
        raise TypeError("targets must be a frozenset of integers.")
    if any(node < 0 or node >= baseline_samples.shape[1] for node in targets):
        raise ValueError("targets must index sample variables.")
    if paired_samples and baseline_samples.shape[0] != probe_samples.shape[0]:
        raise ValueError("paired samples must have equal sizes.")

    legacy_settings = FiniteSourceInferenceSettings(
        settings.equivalence_absolute,
        settings.equivalence_relative,
        settings.confidence_level,
        False,
    )
    rows = []
    for node in sorted(targets):
        statistics = _response_statistics(
            node,
            baseline_samples,
            probe_samples,
            float(shift_increment[node]),
            legacy_settings,
            paired_samples,
        )
        (
            _,
            increment,
            response,
            residual,
            standard_error,
            degrees_of_freedom,
            epsilon,
            _,
            _,
        ) = statistics
        alpha = 1.0 - settings.confidence_level
        if standard_error == 0.0:
            critical = 0.0
        elif np.isfinite(degrees_of_freedom):
            critical = float(student_t.ppf(1.0 - alpha / 2.0, degrees_of_freedom))
        else:
            critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        interval = (
            residual - critical * standard_error,
            residual + critical * standard_error,
        )
        minimum_met = (
            baseline_samples.shape[0] >= settings.minimum_sample_size
            and probe_samples.shape[0] >= settings.minimum_sample_size
        )
        if increment == 0.0 or not np.isfinite(residual):
            classification: SourceClassification = "inconclusive"
            reason = "invalid_or_zero_probe_increment"
        elif not minimum_met:
            classification = "inconclusive"
            reason = "minimum_evidence_not_met"
        elif interval[0] >= -epsilon and interval[1] <= epsilon:
            classification = "source"
            reason = "confidence_interval_inside_equivalence_band"
        elif interval[1] < -epsilon or interval[0] > epsilon:
            classification = "non_source"
            reason = "confidence_interval_outside_equivalence_band"
        else:
            classification = "inconclusive"
            reason = "confidence_interval_overlaps_equivalence_boundary"
        rows.append(
            ConfidenceCalibratedSourceClassification(
                node,
                classification,
                increment,
                response,
                residual,
                standard_error,
                degrees_of_freedom,
                interval,
                (-epsilon, epsilon),
                settings.confidence_level,
                baseline_samples.shape[0],
                probe_samples.shape[0],
                settings.minimum_sample_size,
                minimum_met,
                background_id,
                paired_samples,
                reason,
            )
        )
    return tuple(rows)
