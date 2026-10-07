"""Explicit non-realisable execution and evaluation-only gap reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from graphs.problem_gen import GeneratedProblem, PublicProblemView
from matching.blackbox import (
    ExactMomentBlackBoxEnvironment,
    FiniteSampleBlackBoxEnvironment,
)
from matching.structure_learning import (
    EstimatedPCCPDAGProvider,
    SuppliedExactCPDAGProvider,
)
from oracle.comparator import (
    NonRealisableGapMetrics,
    OracleComparatorResult,
    nonrealisable_gap_metrics,
)

from .nonoracle import (
    NonOracleExperimentConfig,
    NonOracleExperimentResult,
    run_finite_sample_nonoracle,
    run_nonoracle_noiseless,
)

NonRealisableTerminationReason = Literal[
    "target_tolerance_reached",
    "no_significant_improvement",
    "iteration_limit_reached",
    "final_support_exhausted",
    "query_budget_exhausted",
    "observation_budget_exhausted",
    "patience_exhausted",
    "no_legal_action",
    "global_mismatch_not_localized",
    "insufficient_evidence",
    "numerical_failure",
]


@dataclass(frozen=True)
class NonRealisableExperimentResult:
    """Algorithm result plus explicitly evaluation-only comparator metrics."""

    mode: Literal["exact_nonrealisable", "finite_sample_nonrealisable"]
    underlying_result: NonOracleExperimentResult = field(repr=False, compare=False)
    termination_reason: NonRealisableTerminationReason
    initial_reverse_kl: float
    final_reverse_kl: float
    positive_residual_kl_permitted: bool
    no_significant_improvement_reported: bool
    best_found_comparator_loss: float | None
    algorithm_to_best_found_gap: float | None
    fraction_observed_gap_closed: float | None
    comparator_label: str | None
    comparator_proven_lower_bound: bool | None
    target_support_metrics: None = None


def normalize_nonrealisable_termination(
    reason: str,
) -> NonRealisableTerminationReason:
    """Map legacy algorithm exits onto the finite non-realisable vocabulary."""
    aliases: dict[str, NonRealisableTerminationReason] = {
        "loss_tolerance_reached": "target_tolerance_reached",
        "confidence_aware_loss_tolerance_reached": "target_tolerance_reached",
        "target_tolerance_reached": "target_tolerance_reached",
        "iteration_limit_reached": "iteration_limit_reached",
        "no_significant_improvement": "no_significant_improvement",
        "final_support_budget_exhausted": "final_support_exhausted",
        "final_support_exhausted": "final_support_exhausted",
        "experiment_budget_exhausted": "query_budget_exhausted",
        "query_budget_exhausted": "query_budget_exhausted",
        "observation_budget_exhausted": "observation_budget_exhausted",
        "patience_exhausted": "patience_exhausted",
        "no_legal_action": "no_legal_action",
        "global_mismatch_not_localized": "global_mismatch_not_localized",
        "insufficient_evidence": "insufficient_evidence",
        "inconclusive_evaluation_budget_exhausted": "insufficient_evidence",
        "numerical_failure": "numerical_failure",
    }
    return aliases.get(reason, "numerical_failure")


def audit_nonrealisable_result(
    result: NonOracleExperimentResult,
    *,
    evaluation_problem: GeneratedProblem | None = None,
    comparator: OracleComparatorResult | None = None,
) -> NonRealisableExperimentResult:
    """Normalize stopping and attach optional, evaluation-only comparator gaps."""
    if comparator is not None and evaluation_problem is None:
        raise ValueError("a comparator requires an evaluation-only generated problem.")
    gap: NonRealisableGapMetrics | None = None
    if comparator is not None and evaluation_problem is not None:
        gap = nonrealisable_gap_metrics(
            evaluation_problem,
            result.final_intervention,
            comparator,
        )
    initial = next(
        (
            record.global_loss_before
            for record in result.history
            if record.global_loss_before is not None
        ),
        result.final_global_loss,
    )
    normalized = normalize_nonrealisable_termination(result.termination_reason)
    return NonRealisableExperimentResult(
        mode=(
            "finite_sample_nonrealisable"
            if result.mode.startswith("finite_sample")
            else "exact_nonrealisable"
        ),
        underlying_result=result,
        termination_reason=normalized,
        initial_reverse_kl=float(initial),
        final_reverse_kl=result.final_global_loss,
        positive_residual_kl_permitted=True,
        no_significant_improvement_reported=(
            normalized == "no_significant_improvement"
        ),
        best_found_comparator_loss=(
            gap.oracle_comparator_loss if gap is not None else None
        ),
        algorithm_to_best_found_gap=(
            gap.final_algorithm_loss - gap.oracle_comparator_loss
            if gap is not None
            else None
        ),
        fraction_observed_gap_closed=(
            gap.fraction_observed_gap_closed if gap is not None else None
        ),
        comparator_label=gap.comparator_label if gap is not None else None,
        comparator_proven_lower_bound=(
            comparator.proven_lower_bound if comparator is not None else None
        ),
    )


def run_exact_nonrealisable(
    public_problem: PublicProblemView,
    environment: ExactMomentBlackBoxEnvironment,
    structure_provider: SuppliedExactCPDAGProvider | EstimatedPCCPDAGProvider,
    config: NonOracleExperimentConfig,
    *,
    evaluation_problem: GeneratedProblem | None = None,
    comparator: OracleComparatorResult | None = None,
) -> NonRealisableExperimentResult:
    result = run_nonoracle_noiseless(
        public_problem,
        environment,
        structure_provider,
        config,
        result_mode="exact_nonrealisable",
    )
    return audit_nonrealisable_result(
        result,
        evaluation_problem=evaluation_problem,
        comparator=comparator,
    )


def run_finite_sample_nonrealisable(
    public_problem: PublicProblemView,
    environment: FiniteSampleBlackBoxEnvironment,
    structure_provider: SuppliedExactCPDAGProvider | EstimatedPCCPDAGProvider,
    config: NonOracleExperimentConfig,
    *,
    environment_seed: int,
    bootstrap_seed: int,
    evaluation_problem: GeneratedProblem | None = None,
    comparator: OracleComparatorResult | None = None,
) -> NonRealisableExperimentResult:
    result = run_finite_sample_nonoracle(
        public_problem,
        environment,
        structure_provider,
        config,
        environment_seed=environment_seed,
        bootstrap_seed=bootstrap_seed,
        result_mode="finite_sample_nonrealisable",
    )
    return audit_nonrealisable_result(
        result,
        evaluation_problem=evaluation_problem,
        comparator=comparator,
    )
