"""Accepted-background baseline management for exact and sampled modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graphs._immutability import immutable_array
from graphs.interventions import ShiftScaleIntervention

from .blackbox import (
    ExactMomentBlackBoxEnvironment,
    ExactMomentQueryResult,
    FiniteSampleBlackBoxEnvironment,
    SampleQueryResult,
)
from .empirical_moments import EmpiricalMomentEstimate, EmpiricalMomentsProvider
from .types import InterventionBackground


@dataclass(frozen=True)
class AcceptedBackgroundBaseline:
    """Numerical baseline attached to exactly one accepted intervention."""

    background_id: str
    accepted_shift: np.ndarray
    accepted_scale: np.ndarray
    exact_mean: np.ndarray | None
    exact_covariance: np.ndarray | None
    samples: np.ndarray | None
    sample_size: int
    query_ids: tuple[str, ...]
    covariance_estimator_metadata: dict[str, object] | None

    def __post_init__(self) -> None:
        intervention = ShiftScaleIntervention(
            self.accepted_shift,
            self.accepted_scale,
        )
        expected = InterventionBackground.from_intervention(intervention).identifier
        if self.background_id != expected:
            raise ValueError("background_id does not match accepted intervention.")
        object.__setattr__(
            self,
            "accepted_shift",
            immutable_array(intervention.shift, dtype=float),
        )
        object.__setattr__(
            self,
            "accepted_scale",
            immutable_array(intervention.scale, dtype=float),
        )
        for name in ("exact_mean", "exact_covariance", "samples"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, immutable_array(value, dtype=float))
        exact = self.exact_mean is not None or self.exact_covariance is not None
        sampled = self.samples is not None
        if exact == sampled:
            raise ValueError("baseline must contain exactly one evidence type.")
        if exact and (
            self.exact_mean is None
            or self.exact_covariance is None
            or self.sample_size != 0
        ):
            raise ValueError("exact baseline fields are inconsistent.")
        if sampled and (
            self.samples is None
            or self.samples.shape[0] != self.sample_size
            or self.covariance_estimator_metadata is None
        ):
            raise ValueError("sample baseline fields are inconsistent.")
        if not self.query_ids:
            raise ValueError("baseline must retain at least one query ID.")

    @property
    def intervention(self) -> ShiftScaleIntervention:
        """Return the accepted intervention represented by this baseline."""
        return ShiftScaleIntervention(self.accepted_shift, self.accepted_scale)


class AcceptedBackgroundBaselineManager:
    """Reuse baselines only while their accepted background remains current."""

    def __init__(self, accepted: ShiftScaleIntervention) -> None:
        if not isinstance(accepted, ShiftScaleIntervention):
            raise TypeError("accepted must be a ShiftScaleIntervention.")
        self._accepted = accepted
        self._current: AcceptedBackgroundBaseline | None = None
        self._history: list[AcceptedBackgroundBaseline] = []
        self._invalidation_count = 0

    @property
    def accepted(self) -> ShiftScaleIntervention:
        """Return the current accepted intervention."""
        return self._accepted

    @property
    def current(self) -> AcceptedBackgroundBaseline | None:
        """Return the current baseline, if it has been obtained."""
        return self._current

    @property
    def history(self) -> tuple[AcceptedBackgroundBaseline, ...]:
        """Return every baseline obtained across accepted backgrounds."""
        return tuple(self._history)

    @property
    def invalidation_count(self) -> int:
        """Return the number of accepted-background changes."""
        return self._invalidation_count

    @property
    def background_id(self) -> str:
        """Return the current accepted background identifier."""
        return InterventionBackground.from_intervention(self._accepted).identifier

    def accept(self, intervention: ShiftScaleIntervention) -> None:
        """Change accepted state and invalidate all numerical baseline caches."""
        if not isinstance(intervention, ShiftScaleIntervention):
            raise TypeError("intervention must be a ShiftScaleIntervention.")
        new_id = InterventionBackground.from_intervention(intervention).identifier
        if new_id == self.background_id:
            return
        self._accepted = intervention
        self._current = None
        self._invalidation_count += 1

    def ensure_exact(
        self,
        environment: ExactMomentBlackBoxEnvironment,
    ) -> AcceptedBackgroundBaseline:
        """Obtain or reuse exact moments for the accepted background."""
        if self._current is not None:
            if self._current.exact_mean is None:
                raise RuntimeError("current baseline is sampled, not exact.")
            return self._current
        result = environment.query(
            self._accepted,
            kind="baseline",
            background_id=self.background_id,
        )
        self._current = _exact_baseline(self._accepted, result)
        self._history.append(self._current)
        return self._current

    def ensure_sampled(
        self,
        environment: FiniteSampleBlackBoxEnvironment,
        *,
        sample_size: int,
        query_seed: int,
        moments_provider: EmpiricalMomentsProvider,
    ) -> AcceptedBackgroundBaseline:
        """Obtain or reuse finite samples for the accepted background."""
        if self._current is not None:
            if self._current.samples is None:
                raise RuntimeError("current baseline is exact, not sampled.")
            return self._current
        result = environment.query(
            self._accepted,
            sample_size=sample_size,
            query_seed=query_seed,
            kind="baseline",
            background_id=self.background_id,
        )
        estimate = moments_provider.estimate(result.samples)
        self._current = _sampled_baseline(self._accepted, result, estimate)
        self._history.append(self._current)
        return self._current

    def extend_sampled(
        self,
        environment: FiniteSampleBlackBoxEnvironment,
        *,
        sample_size: int,
        query_seed: int,
        moments_provider: EmpiricalMomentsProvider,
    ) -> AcceptedBackgroundBaseline:
        """Add independent samples to the current accepted-background baseline."""
        if self._current is None or self._current.samples is None:
            raise RuntimeError(
                "a sampled baseline must exist before it can be extended."
            )
        if isinstance(sample_size, bool) or not isinstance(sample_size, int):
            raise TypeError("sample_size must be an integer.")
        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")
        result = environment.query(
            self._accepted,
            sample_size=sample_size,
            query_seed=query_seed,
            kind="repeat",
            budget_category="baseline",
            background_id=self.background_id,
        )
        combined = np.concatenate((self._current.samples, result.samples), axis=0)
        estimate = moments_provider.estimate(combined)
        extended = AcceptedBackgroundBaseline(
            background_id=self.background_id,
            accepted_shift=self._accepted.shift,
            accepted_scale=self._accepted.scale,
            exact_mean=None,
            exact_covariance=None,
            samples=combined,
            sample_size=combined.shape[0],
            query_ids=self._current.query_ids + (result.metadata.query_id,),
            covariance_estimator_metadata=estimate.metadata(),
        )
        self._current = extended
        if not self._history or self._history[-1].background_id != self.background_id:
            raise RuntimeError(
                "baseline history is inconsistent with current background."
            )
        self._history[-1] = extended
        return extended


def _exact_baseline(
    accepted: ShiftScaleIntervention,
    result: ExactMomentQueryResult,
) -> AcceptedBackgroundBaseline:
    return AcceptedBackgroundBaseline(
        background_id=InterventionBackground.from_intervention(accepted).identifier,
        accepted_shift=accepted.shift,
        accepted_scale=accepted.scale,
        exact_mean=result.mean,
        exact_covariance=result.covariance,
        samples=None,
        sample_size=0,
        query_ids=(result.metadata.query_id,),
        covariance_estimator_metadata=None,
    )


def _sampled_baseline(
    accepted: ShiftScaleIntervention,
    result: SampleQueryResult,
    estimate: EmpiricalMomentEstimate,
) -> AcceptedBackgroundBaseline:
    return AcceptedBackgroundBaseline(
        background_id=InterventionBackground.from_intervention(accepted).identifier,
        accepted_shift=accepted.shift,
        accepted_scale=accepted.scale,
        exact_mean=None,
        exact_covariance=None,
        samples=result.samples,
        sample_size=result.sample_size,
        query_ids=(result.metadata.query_id,),
        covariance_estimator_metadata=estimate.metadata(),
    )
