"""Confidence-aware empirical candidate acceptance and global success."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import PublicProblemView
from metrics.gaussian_kl import gaussian_kl_target_to_candidate

from .baselines import AcceptedBackgroundBaseline
from .blackbox import FiniteSampleBlackBoxEnvironment
from .capabilities import ProviderCapability
from .empirical_moments import EmpiricalMomentsProvider

EmpiricalDecision = Literal["accepted", "rejected", "inconclusive"]


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


@dataclass(frozen=True)
class EmpiricalAcceptanceSettings:
    """Bootstrap decision thresholds and repeat caps."""

    initial_candidate_sample_size: int
    max_sample_size_per_arm: int
    sample_growth_factor: float
    max_repeat_queries: int
    bootstrap_replicates: int
    confidence_level: float
    improvement_tolerance: float

    def __post_init__(self) -> None:
        for name in (
            "initial_candidate_sample_size",
            "max_sample_size_per_arm",
            "max_repeat_queries",
            "bootstrap_replicates",
        ):
            _positive_integer(name, getattr(self, name))
        if self.initial_candidate_sample_size > self.max_sample_size_per_arm:
            raise ValueError("initial candidate sample size exceeds arm cap.")
        growth = _finite_real("sample_growth_factor", self.sample_growth_factor)
        if growth <= 1.0:
            raise ValueError("sample_growth_factor must exceed one.")
        confidence = _finite_real("confidence_level", self.confidence_level)
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie in (0, 1).")
        tolerance = _finite_real(
            "improvement_tolerance",
            self.improvement_tolerance,
        )
        if tolerance < 0.0:
            raise ValueError("improvement_tolerance must be nonnegative.")


@dataclass(frozen=True)
class EmpiricalCandidateDecision:
    """Final three-way decision with all uncertainty and query metadata."""

    decision: EmpiricalDecision
    current_global_kl: float
    candidate_global_kl: float
    point_improvement: float
    improvement_interval: tuple[float, float]
    bootstrap_seed: int
    bootstrap_replicates: int
    confidence_level: float
    current_sample_size: int
    candidate_sample_size: int
    current_query_ids: tuple[str, ...]
    candidate_query_ids: tuple[str, ...]
    extra_queries: int
    current_covariance_metadata: dict[str, object]
    candidate_covariance_metadata: dict[str, object]
    reused_current_sample_size: int
    newly_drawn_current_sample_size: int
    newly_drawn_candidate_sample_size: int
    comparison_sampling: str
    final_status: str


@dataclass
class ConfidenceAwareCandidateEvaluator:
    """Query samples and never force an empirical acceptance decision."""

    environment: FiniteSampleBlackBoxEnvironment
    moments_provider: EmpiricalMomentsProvider
    settings: EmpiricalAcceptanceSettings
    capabilities = frozenset({ProviderCapability.EMPIRICAL_SAMPLES})

    def evaluate(
        self,
        current: ShiftScaleIntervention,
        candidate: ShiftScaleIntervention,
        problem: PublicProblemView,
        baseline: AcceptedBackgroundBaseline,
        *,
        environment_seed: int,
        bootstrap_seed: int,
    ) -> EmpiricalCandidateDecision:
        """Evaluate and repeat an inconclusive candidate within configured caps."""
        if baseline.samples is None:
            raise ValueError("empirical acceptance requires baseline samples.")
        rng = np.random.default_rng(environment_seed)
        current_chunks = [baseline.samples]
        candidate_result = self.environment.query(
            candidate,
            sample_size=self.settings.initial_candidate_sample_size,
            query_seed=int(rng.integers(0, np.iinfo(np.int64).max)),
            kind="candidate",
            background_id=baseline.background_id,
        )
        candidate_chunks = [candidate_result.samples]
        current_query_ids = list(baseline.query_ids)
        candidate_query_ids = [candidate_result.metadata.query_id]
        repeat_queries = 0

        decision = self._decision(
            np.concatenate(current_chunks),
            np.concatenate(candidate_chunks),
            problem,
            bootstrap_seed,
        )
        while (
            decision.decision == "inconclusive"
            and repeat_queries < self.settings.max_repeat_queries
        ):
            current_size = sum(chunk.shape[0] for chunk in current_chunks)
            candidate_size = sum(chunk.shape[0] for chunk in candidate_chunks)
            if (
                current_size >= self.settings.max_sample_size_per_arm
                and candidate_size >= self.settings.max_sample_size_per_arm
            ):
                break
            add_to_current = current_size <= candidate_size
            intervention = current if add_to_current else candidate
            arm_size = current_size if add_to_current else candidate_size
            remaining = self.settings.max_sample_size_per_arm - arm_size
            requested = min(
                remaining,
                max(
                    2,
                    int(np.ceil(arm_size * (self.settings.sample_growth_factor - 1.0))),
                ),
            )
            if requested <= 0:
                add_to_current = not add_to_current
                intervention = current if add_to_current else candidate
                arm_size = current_size if add_to_current else candidate_size
                remaining = self.settings.max_sample_size_per_arm - arm_size
                requested = min(
                    remaining,
                    max(
                        2,
                        int(
                            np.ceil(
                                arm_size * (self.settings.sample_growth_factor - 1.0)
                            )
                        ),
                    ),
                )
            if requested <= 0:
                break
            result = self.environment.query(
                intervention,
                sample_size=requested,
                query_seed=int(rng.integers(0, np.iinfo(np.int64).max)),
                kind="repeat",
                budget_category="candidate",
                background_id=baseline.background_id,
            )
            if add_to_current:
                current_chunks.append(result.samples)
                current_query_ids.append(result.metadata.query_id)
            else:
                candidate_chunks.append(result.samples)
                candidate_query_ids.append(result.metadata.query_id)
            repeat_queries += 1
            decision = self._decision(
                np.concatenate(current_chunks),
                np.concatenate(candidate_chunks),
                problem,
                bootstrap_seed + repeat_queries,
            )

        return EmpiricalCandidateDecision(
            decision=decision.decision,
            current_global_kl=decision.current_global_kl,
            candidate_global_kl=decision.candidate_global_kl,
            point_improvement=decision.point_improvement,
            improvement_interval=decision.improvement_interval,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=self.settings.bootstrap_replicates,
            confidence_level=self.settings.confidence_level,
            current_sample_size=sum(chunk.shape[0] for chunk in current_chunks),
            candidate_sample_size=sum(chunk.shape[0] for chunk in candidate_chunks),
            current_query_ids=tuple(current_query_ids),
            candidate_query_ids=tuple(candidate_query_ids),
            extra_queries=repeat_queries,
            current_covariance_metadata=decision.current_covariance_metadata,
            candidate_covariance_metadata=decision.candidate_covariance_metadata,
            reused_current_sample_size=baseline.sample_size,
            newly_drawn_current_sample_size=(
                sum(chunk.shape[0] for chunk in current_chunks) - baseline.sample_size
            ),
            newly_drawn_candidate_sample_size=sum(
                chunk.shape[0] for chunk in candidate_chunks
            ),
            comparison_sampling=(
                "explicitly_paired"
                if self.environment.paired_samples
                else "independent_with_reused_baseline"
            ),
            final_status=(
                "inconclusive_evaluation_budget_exhausted"
                if decision.decision == "inconclusive"
                else f"candidate_{decision.decision}"
            ),
        )

    def _decision(
        self,
        current_samples: np.ndarray,
        candidate_samples: np.ndarray,
        problem: PublicProblemView,
        bootstrap_seed: int,
    ) -> EmpiricalCandidateDecision:
        current_estimate = self.moments_provider.estimate(current_samples)
        candidate_estimate = self.moments_provider.estimate(candidate_samples)
        current_loss = gaussian_kl_target_to_candidate(
            problem.target_mean,
            problem.target_covariance,
            current_estimate.mean,
            current_estimate.regularized_covariance,
        )
        candidate_loss = gaussian_kl_target_to_candidate(
            problem.target_mean,
            problem.target_covariance,
            candidate_estimate.mean,
            candidate_estimate.regularized_covariance,
        )
        interval = bootstrap_global_improvement_interval(
            current_samples,
            candidate_samples,
            problem,
            self.moments_provider,
            replicates=self.settings.bootstrap_replicates,
            confidence_level=self.settings.confidence_level,
            seed=bootstrap_seed,
        )
        if interval[0] > self.settings.improvement_tolerance:
            decision: EmpiricalDecision = "accepted"
        elif interval[1] <= self.settings.improvement_tolerance:
            decision = "rejected"
        else:
            decision = "inconclusive"
        return EmpiricalCandidateDecision(
            decision=decision,
            current_global_kl=current_loss,
            candidate_global_kl=candidate_loss,
            point_improvement=current_loss - candidate_loss,
            improvement_interval=interval,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=self.settings.bootstrap_replicates,
            confidence_level=self.settings.confidence_level,
            current_sample_size=current_samples.shape[0],
            candidate_sample_size=candidate_samples.shape[0],
            current_query_ids=(),
            candidate_query_ids=(),
            extra_queries=0,
            current_covariance_metadata=current_estimate.metadata(),
            candidate_covariance_metadata=candidate_estimate.metadata(),
            reused_current_sample_size=0,
            newly_drawn_current_sample_size=current_samples.shape[0],
            newly_drawn_candidate_sample_size=candidate_samples.shape[0],
            comparison_sampling="independent_analysis",
            final_status=f"candidate_{decision}",
        )


@dataclass(frozen=True)
class EmpiricalGlobalKLEstimate:
    """Point and confidence interval for finite-sample global reverse KL."""

    point_estimate: float
    confidence_interval: tuple[float, float]
    sample_size: int
    bootstrap_replicates: int
    bootstrap_seed: int
    success: bool
    covariance_metadata: dict[str, object]


def estimate_empirical_global_success(
    samples: np.ndarray,
    problem: PublicProblemView,
    moments_provider: EmpiricalMomentsProvider,
    *,
    global_tolerance: float,
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> EmpiricalGlobalKLEstimate:
    """Declare success only when the upper global-KL bound meets tolerance."""
    tolerance = _finite_real("global_tolerance", global_tolerance)
    if tolerance < 0.0:
        raise ValueError("global_tolerance must be nonnegative.")
    _positive_integer("bootstrap_replicates", bootstrap_replicates)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1).")
    estimate = moments_provider.estimate(samples)
    point = gaussian_kl_target_to_candidate(
        problem.target_mean,
        problem.target_covariance,
        estimate.mean,
        estimate.regularized_covariance,
    )
    rng = np.random.default_rng(bootstrap_seed)
    losses = np.empty(bootstrap_replicates)
    for replicate in range(bootstrap_replicates):
        indices = rng.integers(0, samples.shape[0], samples.shape[0])
        bootstrap_estimate = moments_provider.estimate(samples[indices])
        losses[replicate] = gaussian_kl_target_to_candidate(
            problem.target_mean,
            problem.target_covariance,
            bootstrap_estimate.mean,
            bootstrap_estimate.regularized_covariance,
        )
    tail = (1.0 - confidence_level) / 2.0
    interval = (
        float(np.quantile(losses, tail)),
        float(np.quantile(losses, 1.0 - tail)),
    )
    return EmpiricalGlobalKLEstimate(
        point_estimate=point,
        confidence_interval=interval,
        sample_size=samples.shape[0],
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        success=interval[1] <= tolerance,
        covariance_metadata=estimate.metadata(),
    )


def bootstrap_global_improvement_interval(
    current_samples: np.ndarray,
    candidate_samples: np.ndarray,
    problem: PublicProblemView,
    moments_provider: EmpiricalMomentsProvider,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap current-minus-candidate global reverse KL."""
    rng = np.random.default_rng(seed)
    improvements = np.empty(replicates)
    for replicate in range(replicates):
        current_indices = rng.integers(
            0,
            current_samples.shape[0],
            current_samples.shape[0],
        )
        candidate_indices = rng.integers(
            0,
            candidate_samples.shape[0],
            candidate_samples.shape[0],
        )
        current_estimate = moments_provider.estimate(current_samples[current_indices])
        candidate_estimate = moments_provider.estimate(
            candidate_samples[candidate_indices]
        )
        current_loss = gaussian_kl_target_to_candidate(
            problem.target_mean,
            problem.target_covariance,
            current_estimate.mean,
            current_estimate.regularized_covariance,
        )
        candidate_loss = gaussian_kl_target_to_candidate(
            problem.target_mean,
            problem.target_covariance,
            candidate_estimate.mean,
            candidate_estimate.regularized_covariance,
        )
        improvements[replicate] = current_loss - candidate_loss
    tail = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(improvements, tail)),
        float(np.quantile(improvements, 1.0 - tail)),
    )
