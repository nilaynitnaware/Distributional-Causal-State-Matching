"""Held-out global KL-improvement inference with prespecified looks."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from metrics.gaussian_kl import gaussian_kl_target_to_candidate

from .empirical_moments import EmpiricalMomentsProvider
from .sample_roles import (
    KeyedTargetValidationSampler,
    NestedRoleSamplePool,
    RoleSeparatedValidationEnvironment,
)
from .types import InterventionBackground

HeldoutDecision = Literal["accepted", "rejected", "unresolved"]
VALIDATION_LOOKS = (2000, 4000, 8000)
SHORTLIST_SIZE = 3
OVERALL_ALPHA = 0.05
MULTIPLICITY_METHOD = "bonferroni_candidates_times_prespecified_looks"


@dataclass(frozen=True)
class HeldoutImprovementSettings:
    """Frozen familywise and repeated-look settings for the new protocol."""

    bootstrap_replicates: int = 999
    acceptance_margin: float = 1e-3
    validation_looks: tuple[int, int, int] = VALIDATION_LOOKS
    shortlist_size: int = SHORTLIST_SIZE
    overall_alpha: float = OVERALL_ALPHA

    def __post_init__(self) -> None:
        if self.validation_looks != VALIDATION_LOOKS:
            raise ValueError("validation looks are fixed at 2000, 4000, and 8000.")
        if self.shortlist_size != SHORTLIST_SIZE:
            raise ValueError("shortlist size is fixed at three.")
        if not np.isclose(self.overall_alpha, OVERALL_ALPHA, rtol=0.0, atol=1e-15):
            raise ValueError("overall familywise alpha is fixed at 0.05.")
        if (
            isinstance(self.bootstrap_replicates, bool)
            or not isinstance(self.bootstrap_replicates, int)
            or self.bootstrap_replicates <= 0
        ):
            raise ValueError("bootstrap_replicates must be a positive integer.")
        if not np.isfinite(self.acceptance_margin) or self.acceptance_margin < 0.0:
            raise ValueError("acceptance_margin must be finite and nonnegative.")

    @property
    def alpha_test(self) -> float:
        return self.overall_alpha / (self.shortlist_size * len(self.validation_looks))

    @property
    def one_sided_confidence_level(self) -> float:
        return 1.0 - self.alpha_test


@dataclass(frozen=True)
class HeldoutDeltaEstimate:
    """One prespecified-look shared-target bootstrap result."""

    decision: HeldoutDecision
    point_delta_kl: float
    current_kl: float
    candidate_kl: float
    lower_bound: float
    upper_bound: float
    validation_look: int
    validation_cap: int
    bootstrap_seed: int
    bootstrap_replicates: int
    overall_alpha: float
    alpha_test: float
    one_sided_confidence_level: float
    multiplicity_method: str
    acceptance_margin: float
    target_sample_size: int
    current_sample_size: int
    candidate_sample_size: int
    target_covariance_metadata: dict[str, object]
    current_covariance_metadata: dict[str, object]
    candidate_covariance_metadata: dict[str, object]
    first_resample_audit: dict[str, object]


def _index_fingerprint(indices: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()


def shared_target_bootstrap_delta(
    target_samples: np.ndarray,
    current_samples: np.ndarray,
    candidate_samples: np.ndarray,
    moments_provider: EmpiricalMomentsProvider,
    settings: HeldoutImprovementSettings,
    *,
    validation_look: int,
    validation_cap: int,
    bootstrap_seed: int,
) -> HeldoutDeltaEstimate:
    """Refit target/current/candidate OAS moments in every bootstrap replicate."""
    if not isinstance(moments_provider, EmpiricalMomentsProvider):
        raise TypeError("moments_provider must be EmpiricalMomentsProvider.")
    if validation_look not in range(len(settings.validation_looks)):
        raise ValueError("validation_look must be one of the three scheduled looks.")
    if settings.validation_looks[validation_look] != validation_cap:
        raise ValueError("validation cap does not match its scheduled look.")
    arrays = (target_samples, current_samples, candidate_samples)
    if any(
        not isinstance(samples, np.ndarray)
        or samples.ndim != 2
        or samples.shape[0] != validation_cap
        or not np.all(np.isfinite(samples))
        for samples in arrays
    ):
        raise ValueError("all validation arrays must match the scheduled cap.")
    if len({samples.shape[1] for samples in arrays}) != 1:
        raise ValueError("validation sample dimensions must agree.")

    target_estimate = moments_provider.estimate(target_samples)
    current_estimate = moments_provider.estimate(current_samples)
    candidate_estimate = moments_provider.estimate(candidate_samples)
    current_kl = gaussian_kl_target_to_candidate(
        target_estimate.mean,
        target_estimate.regularized_covariance,
        current_estimate.mean,
        current_estimate.regularized_covariance,
    )
    candidate_kl = gaussian_kl_target_to_candidate(
        target_estimate.mean,
        target_estimate.regularized_covariance,
        candidate_estimate.mean,
        candidate_estimate.regularized_covariance,
    )
    point_delta = current_kl - candidate_kl

    rng = np.random.default_rng(bootstrap_seed)
    improvements = np.empty(settings.bootstrap_replicates)
    first_audit: dict[str, object] = {}
    for replicate in range(settings.bootstrap_replicates):
        target_indices = rng.integers(0, target_samples.shape[0], target_samples.shape[0])
        current_indices = rng.integers(
            0, current_samples.shape[0], current_samples.shape[0]
        )
        candidate_indices = rng.integers(
            0, candidate_samples.shape[0], candidate_samples.shape[0]
        )
        bootstrap_target = moments_provider.estimate(target_samples[target_indices])
        bootstrap_current = moments_provider.estimate(current_samples[current_indices])
        bootstrap_candidate = moments_provider.estimate(
            candidate_samples[candidate_indices]
        )
        current_loss = gaussian_kl_target_to_candidate(
            bootstrap_target.mean,
            bootstrap_target.regularized_covariance,
            bootstrap_current.mean,
            bootstrap_current.regularized_covariance,
        )
        candidate_loss = gaussian_kl_target_to_candidate(
            bootstrap_target.mean,
            bootstrap_target.regularized_covariance,
            bootstrap_candidate.mean,
            bootstrap_candidate.regularized_covariance,
        )
        improvements[replicate] = current_loss - candidate_loss
        if replicate == 0:
            target_hash = _index_fingerprint(target_indices)
            first_audit = {
                "target_indices_current_kl_fingerprint": target_hash,
                "target_indices_candidate_kl_fingerprint": target_hash,
                "current_indices_fingerprint": _index_fingerprint(current_indices),
                "candidate_indices_fingerprint": _index_fingerprint(candidate_indices),
                "current_candidate_indices_are_distinct": not np.array_equal(
                    current_indices, candidate_indices
                ),
            }
    lower = float(np.quantile(improvements, settings.alpha_test))
    upper = float(np.quantile(improvements, 1.0 - settings.alpha_test))
    if lower > settings.acceptance_margin:
        decision: HeldoutDecision = "accepted"
    elif upper <= settings.acceptance_margin:
        decision = "rejected"
    else:
        decision = "unresolved"
    return HeldoutDeltaEstimate(
        decision=decision,
        point_delta_kl=point_delta,
        current_kl=current_kl,
        candidate_kl=candidate_kl,
        lower_bound=lower,
        upper_bound=upper,
        validation_look=validation_look,
        validation_cap=validation_cap,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=settings.bootstrap_replicates,
        overall_alpha=settings.overall_alpha,
        alpha_test=settings.alpha_test,
        one_sided_confidence_level=settings.one_sided_confidence_level,
        multiplicity_method=MULTIPLICITY_METHOD,
        acceptance_margin=settings.acceptance_margin,
        target_sample_size=target_samples.shape[0],
        current_sample_size=current_samples.shape[0],
        candidate_sample_size=candidate_samples.shape[0],
        target_covariance_metadata=target_estimate.metadata(),
        current_covariance_metadata=current_estimate.metadata(),
        candidate_covariance_metadata=candidate_estimate.metadata(),
        first_resample_audit=first_audit,
    )


@dataclass
class CandidateComparisonState:
    """Validation evidence retained for one exact background/candidate pair."""

    candidate: ShiftScaleIntervention
    shortlist_position: int
    background_id: str
    paired_problem_id: str
    pool: NestedRoleSamplePool = field(init=False)
    interval_history: list[HeldoutDeltaEstimate] = field(default_factory=list)
    terminal_decision: HeldoutDecision = "unresolved"

    def __post_init__(self) -> None:
        candidate_id = InterventionBackground.from_intervention(self.candidate).identifier
        self.pool = NestedRoleSamplePool(
            "validation_candidate",
            self.paired_problem_id,
            self.background_id,
            candidate_id,
        )

    @property
    def candidate_id(self) -> str:
        return self.pool.candidate_id


@dataclass(frozen=True)
class HeldoutShortlistDecision:
    """Terminal decision across at most three fixed training candidates."""

    decision: HeldoutDecision
    accepted_candidate_id: str | None
    accepted_shortlist_position: int | None
    accepted_intervention: ShiftScaleIntervention | None
    final_look: int
    final_cap: int
    comparison_states: tuple[CandidateComparisonState, ...]
    shared_target_history: dict[str, object]
    shared_current_history: dict[str, object]


class HeldoutShortlistEvaluator:
    """Allocate only scheduled target/current/relevant-candidate validation pools."""

    def __init__(
        self,
        validation_environment: RoleSeparatedValidationEnvironment,
        target_sampler: KeyedTargetValidationSampler,
        moments_provider: EmpiricalMomentsProvider,
        settings: HeldoutImprovementSettings,
    ) -> None:
        self.validation_environment = validation_environment
        self.target_sampler = target_sampler
        self.moments_provider = moments_provider
        self.settings = settings

    def evaluate(
        self,
        current: ShiftScaleIntervention,
        candidates: Sequence[ShiftScaleIntervention],
        *,
        paired_problem_id: str,
        max_validation_cap: int,
        bootstrap_seed: int,
    ) -> HeldoutShortlistDecision:
        if len(candidates) > self.settings.shortlist_size:
            raise ValueError("candidate count exceeds the fixed top-three shortlist.")
        if not candidates:
            raise ValueError("held-out evaluation requires at least one candidate.")
        if max_validation_cap not in self.settings.validation_looks:
            raise ValueError("max_validation_cap must be a prespecified look.")
        background_id = InterventionBackground.from_intervention(current).identifier
        target_pool = NestedRoleSamplePool(
            "validation_target", paired_problem_id, background_id, "shared_target"
        )
        current_pool = NestedRoleSamplePool(
            "validation_current", paired_problem_id, background_id, background_id
        )
        states = [
            CandidateComparisonState(candidate, position, background_id, paired_problem_id)
            for position, candidate in enumerate(candidates)
        ]
        active = list(states)
        final_look = 0
        final_cap = self.settings.validation_looks[0]
        for look, cap in enumerate(self.settings.validation_looks):
            if cap > max_validation_cap:
                break
            final_look, final_cap = look, cap
            increment = cap - target_pool.sample_size
            target_pool.append(
                self.target_sampler.sample_batch(
                    background_id=background_id,
                    validation_look=look,
                    validation_cap=cap,
                    replication_index=len(target_pool.batches),
                    sample_size=increment,
                )
            )
            current_pool.append(
                self.validation_environment.query_batch(
                    current,
                    role="validation_current",
                    background_id=background_id,
                    candidate_id=background_id,
                    validation_look=look,
                    validation_cap=cap,
                    replication_index=len(current_pool.batches),
                    sample_size=increment,
                )
            )
            for state in active:
                candidate_increment = cap - state.pool.sample_size
                state.pool.append(
                    self.validation_environment.query_batch(
                        state.candidate,
                        role="validation_candidate",
                        background_id=background_id,
                        candidate_id=state.candidate_id,
                        validation_look=look,
                        validation_cap=cap,
                        replication_index=len(state.pool.batches),
                        sample_size=candidate_increment,
                    )
                )
                estimate = shared_target_bootstrap_delta(
                    target_pool.samples,
                    current_pool.samples,
                    state.pool.samples,
                    self.moments_provider,
                    self.settings,
                    validation_look=look,
                    validation_cap=cap,
                    bootstrap_seed=bootstrap_seed + look * 10_003 + state.shortlist_position,
                )
                state.interval_history.append(estimate)
                state.terminal_decision = estimate.decision
            accepted = [state for state in active if state.terminal_decision == "accepted"]
            if accepted:
                selected = max(
                    accepted,
                    key=lambda state: (
                        state.interval_history[-1].lower_bound,
                        state.interval_history[-1].point_delta_kl,
                        -state.shortlist_position,
                    ),
                )
                return HeldoutShortlistDecision(
                    "accepted",
                    selected.candidate_id,
                    selected.shortlist_position,
                    selected.candidate,
                    look,
                    cap,
                    tuple(states),
                    target_pool.history(),
                    current_pool.history(),
                )
            active = [state for state in active if state.terminal_decision == "unresolved"]
            if not active:
                return HeldoutShortlistDecision(
                    "rejected",
                    None,
                    None,
                    None,
                    look,
                    cap,
                    tuple(states),
                    target_pool.history(),
                    current_pool.history(),
                )
        return HeldoutShortlistDecision(
            "unresolved" if active else "rejected",
            None,
            None,
            None,
            final_look,
            final_cap,
            tuple(states),
            target_pool.history(),
            current_pool.history(),
        )
