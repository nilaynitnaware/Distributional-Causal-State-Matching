"""Public source classification from exact or finite-sample probe responses."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from statistics import NormalDist
from typing import Literal

import numpy as np
from scipy.stats import t as student_t

SourceClassification = Literal["source", "non_source", "inconclusive"]


def _nonnegative_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


@dataclass(frozen=True)
class SourceTolerance:
    """Absolute-plus-relative response-error tolerance."""

    absolute: float
    relative: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "absolute",
            _nonnegative_real("absolute", self.absolute),
        )
        object.__setattr__(
            self,
            "relative",
            _nonnegative_real("relative", self.relative),
        )

    def for_increment(self, increment: float) -> float:
        """Return the tolerance for one pure-shift increment."""
        return self.absolute + self.relative * abs(float(increment))


@dataclass(frozen=True)
class ExactSourceClassification:
    """One exact response classification with auditable numerical evidence."""

    node: int
    classification: SourceClassification
    probe_increment: float
    estimated_response: float
    response_error: float
    tolerance: float
    reason: str


def classify_exact_source_targets(
    baseline_mean: np.ndarray,
    probe_mean: np.ndarray,
    shift_increment: np.ndarray,
    targets: frozenset[int],
    tolerance: SourceTolerance,
) -> tuple[ExactSourceClassification, ...]:
    """Classify targets using only ``mean_probe - mean_baseline``.

    This criterion assumes intervention faithfulness for the chosen probe:
    non-source targets do not have a total-effect cancellation that exactly
    reproduces their own additive increment.
    """
    arrays = {
        "baseline_mean": baseline_mean,
        "probe_mean": probe_mean,
        "shift_increment": shift_increment,
    }
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise TypeError(f"{name} must be a one-dimensional numpy.ndarray.")
    if not (baseline_mean.shape == probe_mean.shape == shift_increment.shape):
        raise ValueError("mean and increment dimensions must agree.")
    if not isinstance(targets, frozenset) or any(
        isinstance(node, bool) or not isinstance(node, int) for node in targets
    ):
        raise TypeError("targets must be a frozenset of integer nodes.")
    if any(node < 0 or node >= baseline_mean.size for node in targets):
        raise ValueError("target nodes must fit the mean dimension.")
    if not isinstance(tolerance, SourceTolerance):
        raise TypeError("tolerance must be SourceTolerance.")

    results: list[ExactSourceClassification] = []
    response = probe_mean - baseline_mean
    for node in sorted(targets):
        increment = float(shift_increment[node])
        estimated = float(response[node])
        threshold = tolerance.for_increment(increment)
        if not np.isfinite(increment) or not np.isfinite(estimated) or increment == 0.0:
            classification: SourceClassification = "inconclusive"
            error = float("nan")
            reason = "invalid_or_zero_probe_increment"
        else:
            error = estimated - increment
            classification = "source" if abs(error) <= threshold else "non_source"
            reason = (
                "response_matches_increment"
                if classification == "source"
                else ("response_differs_from_increment")
            )
        results.append(
            ExactSourceClassification(
                node=node,
                classification=classification,
                probe_increment=increment,
                estimated_response=estimated,
                response_error=error,
                tolerance=threshold,
                reason=reason,
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class FiniteSourceInferenceSettings:
    """Configurable Welch/equivalence source-test settings."""

    equivalence_absolute: float
    equivalence_relative: float
    confidence_level: float = 0.95
    holm_correction: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "equivalence_absolute",
            _nonnegative_real(
                "equivalence_absolute",
                self.equivalence_absolute,
            ),
        )
        object.__setattr__(
            self,
            "equivalence_relative",
            _nonnegative_real(
                "equivalence_relative",
                self.equivalence_relative,
            ),
        )
        confidence = _nonnegative_real("confidence_level", self.confidence_level)
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie in (0, 1).")
        object.__setattr__(self, "confidence_level", confidence)
        if not isinstance(self.holm_correction, bool):
            raise TypeError("holm_correction must be a boolean.")


@dataclass(frozen=True)
class FiniteSourceClassification:
    """One three-way source classification and corrected confidence interval."""

    node: int
    classification: SourceClassification
    probe_increment: float
    estimated_response: float
    response_error: float
    confidence_interval: tuple[float, float]
    equivalence_interval: tuple[float, float]
    corrected_alpha: float
    standard_error: float
    degrees_of_freedom: float
    baseline_sample_size: int
    probe_sample_size: int
    background_id: str
    paired_samples: bool
    source_p_value: float
    non_source_p_value: float
    source_adjusted_p_value: float
    non_source_adjusted_p_value: float
    source_rejected: bool
    non_source_rejected: bool
    source_holm_rank: int
    non_source_holm_rank: int
    source_holm_threshold: float
    non_source_holm_threshold: float
    multiplicity_method: str


def classify_finite_source_targets(
    baseline_samples: np.ndarray,
    probe_samples: np.ndarray,
    shift_increment: np.ndarray,
    targets: frozenset[int],
    *,
    background_id: str,
    settings: FiniteSourceInferenceSettings,
    paired_samples: bool = False,
) -> tuple[FiniteSourceClassification, ...]:
    """Classify responses with equivalence/outside tests in one Holm family.

    Each target contributes an intersection-union equivalence p-value for the
    source decision and a conservative two-sided outside-interval p-value for
    the non-source decision. Holm step-down is applied to all 2m hypotheses;
    neither ordinary failure to reject nor overlap of a confidence interval is
    treated as evidence of equivalence.
    """
    if not isinstance(settings, FiniteSourceInferenceSettings):
        raise TypeError("settings must be FiniteSourceInferenceSettings.")
    if not isinstance(background_id, str) or not background_id:
        raise ValueError("background_id must be a nonempty string.")
    if not isinstance(paired_samples, bool):
        raise TypeError("paired_samples must be a boolean.")
    for name, samples in (
        ("baseline_samples", baseline_samples),
        ("probe_samples", probe_samples),
    ):
        if not isinstance(samples, np.ndarray) or samples.ndim != 2:
            raise TypeError(f"{name} must be a two-dimensional numpy.ndarray.")
        if samples.shape[0] < 2:
            raise ValueError(f"{name} must contain at least two observations.")
        if not np.all(np.isfinite(samples)):
            raise ValueError(f"{name} must be finite.")
    if baseline_samples.shape[1] != probe_samples.shape[1]:
        raise ValueError("baseline and probe dimensions must agree.")
    if not isinstance(shift_increment, np.ndarray) or shift_increment.shape != (
        baseline_samples.shape[1],
    ):
        raise ValueError("shift_increment dimension must match the samples.")
    if not isinstance(targets, frozenset) or any(
        isinstance(node, bool) or not isinstance(node, int) for node in targets
    ):
        raise TypeError("targets must be a frozenset of integer nodes.")
    if any(node < 0 or node >= baseline_samples.shape[1] for node in targets):
        raise ValueError("targets must index sample variables.")
    if paired_samples and baseline_samples.shape[0] != probe_samples.shape[0]:
        raise ValueError("paired samples must have equal row counts.")

    preliminary = [
        _response_statistics(
            node,
            baseline_samples,
            probe_samples,
            float(shift_increment[node]),
            settings,
            paired_samples,
        )
        for node in sorted(targets)
    ]
    family_alpha = 1.0 - settings.confidence_level
    p_values = [p_value for statistics in preliminary for p_value in statistics[-2:]]
    decisions = (
        _holm_step_down(p_values, family_alpha)
        if settings.holm_correction and preliminary
        else _uncorrected_decisions(p_values, family_alpha)
    )

    results: list[FiniteSourceClassification] = []
    for index, statistics in enumerate(preliminary):
        (
            node,
            increment,
            estimated_response,
            response_error,
            standard_error,
            degrees_of_freedom,
            epsilon,
            source_p_value,
            non_source_p_value,
        ) = statistics
        source_decision = decisions[2 * index]
        non_source_decision = decisions[2 * index + 1]
        corrected_alpha = min(source_decision.threshold, non_source_decision.threshold)
        if standard_error == 0.0:
            critical = 0.0
        elif np.isfinite(degrees_of_freedom):
            critical = float(
                student_t.ppf(1.0 - corrected_alpha / 2.0, degrees_of_freedom)
            )
        else:
            critical = NormalDist().inv_cdf(1.0 - corrected_alpha / 2.0)
        interval = (
            response_error - critical * standard_error,
            response_error + critical * standard_error,
        )
        if increment == 0.0 or not np.isfinite(response_error):
            classification: SourceClassification = "inconclusive"
        elif source_decision.rejected and not non_source_decision.rejected:
            classification = "source"
        elif non_source_decision.rejected and not source_decision.rejected:
            classification = "non_source"
        else:
            classification = "inconclusive"
        results.append(
            FiniteSourceClassification(
                node=node,
                classification=classification,
                probe_increment=increment,
                estimated_response=estimated_response,
                response_error=response_error,
                confidence_interval=interval,
                equivalence_interval=(-epsilon, epsilon),
                corrected_alpha=corrected_alpha,
                standard_error=standard_error,
                degrees_of_freedom=degrees_of_freedom,
                baseline_sample_size=baseline_samples.shape[0],
                probe_sample_size=probe_samples.shape[0],
                background_id=background_id,
                paired_samples=paired_samples,
                source_p_value=source_p_value,
                non_source_p_value=non_source_p_value,
                source_adjusted_p_value=source_decision.adjusted_p_value,
                non_source_adjusted_p_value=non_source_decision.adjusted_p_value,
                source_rejected=source_decision.rejected,
                non_source_rejected=non_source_decision.rejected,
                source_holm_rank=source_decision.rank,
                non_source_holm_rank=non_source_decision.rank,
                source_holm_threshold=source_decision.threshold,
                non_source_holm_threshold=non_source_decision.threshold,
                multiplicity_method=(
                    "holm_step_down" if settings.holm_correction else "none"
                ),
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class _MultiplicityDecision:
    rejected: bool
    adjusted_p_value: float
    rank: int
    threshold: float


def _response_statistics(
    node: int,
    baseline: np.ndarray,
    probe: np.ndarray,
    increment: float,
    settings: FiniteSourceInferenceSettings,
    paired: bool,
) -> tuple[int, float, float, float, float, float, float, float, float]:
    baseline_values = baseline[:, node]
    probe_values = probe[:, node]
    estimated_response = float(np.mean(probe_values) - np.mean(baseline_values))
    response_error = estimated_response - increment
    if paired:
        differences = probe_values - baseline_values
        standard_error = float(np.std(differences, ddof=1) / np.sqrt(differences.size))
        degrees_of_freedom = float(differences.size - 1)
    else:
        baseline_variance = float(np.var(baseline_values, ddof=1))
        probe_variance = float(np.var(probe_values, ddof=1))
        baseline_term = baseline_variance / baseline_values.size
        probe_term = probe_variance / probe_values.size
        standard_error = float(np.sqrt(baseline_term + probe_term))
        denominator = baseline_term**2 / (baseline_values.size - 1) + probe_term**2 / (
            probe_values.size - 1
        )
        degrees_of_freedom = (
            (baseline_term + probe_term) ** 2 / denominator
            if denominator > 0.0
            else float("inf")
        )
    epsilon = settings.equivalence_absolute + settings.equivalence_relative * abs(
        increment
    )
    if increment == 0.0 or not np.isfinite(response_error):
        source_p_value = 1.0
        non_source_p_value = 1.0
    elif standard_error == 0.0:
        source_p_value = 0.0 if -epsilon < response_error < epsilon else 1.0
        non_source_p_value = 0.0 if abs(response_error) > epsilon else 1.0
    else:
        source_p_value = max(
            _survival((response_error + epsilon) / standard_error, degrees_of_freedom),
            _cdf((response_error - epsilon) / standard_error, degrees_of_freedom),
        )
        non_source_p_value = min(
            1.0,
            2.0
            * min(
                _survival(
                    (response_error - epsilon) / standard_error, degrees_of_freedom
                ),
                _cdf((response_error + epsilon) / standard_error, degrees_of_freedom),
            ),
        )
    return (
        node,
        increment,
        estimated_response,
        response_error,
        standard_error,
        degrees_of_freedom,
        epsilon,
        float(source_p_value),
        float(non_source_p_value),
    )


def _cdf(value: float, degrees_of_freedom: float) -> float:
    if np.isfinite(degrees_of_freedom):
        return float(student_t.cdf(value, degrees_of_freedom))
    return float(NormalDist().cdf(value))


def _survival(value: float, degrees_of_freedom: float) -> float:
    if np.isfinite(degrees_of_freedom):
        return float(student_t.sf(value, degrees_of_freedom))
    return float(1.0 - NormalDist().cdf(value))


def _holm_step_down(
    p_values: list[float],
    family_alpha: float,
) -> list[_MultiplicityDecision]:
    """Apply deterministic Holm step-down testing, including the stop rule."""
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    decisions: list[_MultiplicityDecision | None] = [None] * len(p_values)
    continue_rejecting = True
    running_adjusted = 0.0
    family_size = len(p_values)
    for zero_rank, index in enumerate(order):
        threshold = family_alpha / (family_size - zero_rank)
        p_value = float(p_values[index])
        running_adjusted = max(
            running_adjusted,
            min(1.0, (family_size - zero_rank) * p_value),
        )
        rejected = continue_rejecting and p_value <= threshold
        if not rejected:
            continue_rejecting = False
        decisions[index] = _MultiplicityDecision(
            rejected=rejected,
            adjusted_p_value=running_adjusted,
            rank=zero_rank + 1,
            threshold=threshold,
        )
    return [decision for decision in decisions if decision is not None]


def _uncorrected_decisions(
    p_values: list[float],
    alpha: float,
) -> list[_MultiplicityDecision]:
    return [
        _MultiplicityDecision(
            rejected=float(p_value) <= alpha,
            adjusted_p_value=float(p_value),
            rank=index + 1,
            threshold=alpha,
        )
        for index, p_value in enumerate(p_values)
    ]
