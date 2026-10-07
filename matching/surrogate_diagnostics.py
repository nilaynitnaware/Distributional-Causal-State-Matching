"""Diagnostics for disagreement between local surrogates and global KL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import rankdata

from .types import ParameterCoordinate

ProposalStatus = Literal["accepted", "rejected", "inconclusive", "not_selected"]


@dataclass(frozen=True)
class SurrogateProposalDiagnostic:
    predicted_local_improvement: float
    actual_exact_global_improvement: float | None
    empirical_global_improvement: float | None
    status: ProposalStatus
    coordinate: ParameterCoordinate
    background_id: str
    local_terms_used: tuple[str, ...]

    @property
    def comparable_global_improvement(self) -> float | None:
        return (
            self.actual_exact_global_improvement
            if self.actual_exact_global_improvement is not None
            else self.empirical_global_improvement
        )


@dataclass(frozen=True)
class SurrogateDiagnosticSummary:
    proposal_count: int
    comparable_proposal_count: int
    local_global_improvement_rank_correlation: float | None
    locally_positive_globally_negative_rate: float | None
    globally_rejected_rate: float
    shift_mean_global_improvement: float | None
    scale_mean_global_improvement: float | None
    shift_minus_scale_mean_global_improvement: float | None


def global_acceptance_decision(
    current_global_kl: float,
    candidate_global_kl: float,
    improvement_tolerance: float,
) -> bool:
    """Accept solely from global KL, irrespective of local predictions."""
    if improvement_tolerance < 0.0:
        raise ValueError("improvement_tolerance must be nonnegative.")
    return candidate_global_kl <= current_global_kl - improvement_tolerance


def summarize_surrogate_diagnostics(
    diagnostics: tuple[SurrogateProposalDiagnostic, ...],
) -> SurrogateDiagnosticSummary:
    comparable = [
        (item, item.comparable_global_improvement)
        for item in diagnostics
        if item.comparable_global_improvement is not None
    ]
    if len(comparable) >= 2:
        local_ranks = rankdata(
            [item.predicted_local_improvement for item, _ in comparable]
        )
        global_ranks = rankdata([value for _, value in comparable])
        if np.std(local_ranks) == 0.0 or np.std(global_ranks) == 0.0:
            correlation = None
        else:
            correlation = float(np.corrcoef(local_ranks, global_ranks)[0, 1])
    else:
        correlation = None
    locally_positive = [
        (item, value)
        for item, value in comparable
        if item.predicted_local_improvement > 0.0
    ]
    disagreement = (
        float(np.mean([value < 0.0 for _, value in locally_positive]))
        if locally_positive
        else None
    )
    shift = [value for item, value in comparable if item.coordinate.kind == "shift"]
    scale = [value for item, value in comparable if item.coordinate.kind == "log_scale"]
    shift_mean = float(np.mean(shift)) if shift else None
    scale_mean = float(np.mean(scale)) if scale else None
    return SurrogateDiagnosticSummary(
        proposal_count=len(diagnostics),
        comparable_proposal_count=len(comparable),
        local_global_improvement_rank_correlation=correlation,
        locally_positive_globally_negative_rate=disagreement,
        globally_rejected_rate=(
            float(np.mean([item.status == "rejected" for item in diagnostics]))
            if diagnostics
            else 0.0
        ),
        shift_mean_global_improvement=shift_mean,
        scale_mean_global_improvement=scale_mean,
        shift_minus_scale_mean_global_improvement=(
            shift_mean - scale_mean
            if shift_mean is not None and scale_mean is not None
            else None
        ),
    )
