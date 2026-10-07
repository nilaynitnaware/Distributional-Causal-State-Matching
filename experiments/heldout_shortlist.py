"""Operational sample-only runner for the opt-in held-out shortlist protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from environments.blackbox import FiniteSampleBlackBoxEnvironment
from graphs.interventions import ShiftScaleIntervention
from graphs.pdag import PartiallyDirectedGraph
from graphs.problem_gen import PublicProblemView
from matching.baselines import AcceptedBackgroundBaselineManager
from matching.coordinate_search import (
    FiniteCoordinateSearchResult,
    FiniteCoordinateSearchSettings,
    FiniteSampleCoordinateSearch,
    _FixedMomentsProvider,
)
from matching.empirical_moments import (
    CovarianceEstimatorSettings,
    EmpiricalMomentsProvider,
)
from matching.finite_shortlist import deterministic_top3_shortlist
from matching.heldout_improvement import (
    HeldoutImprovementSettings,
    HeldoutShortlistDecision,
    HeldoutShortlistEvaluator,
)
from matching.local_losses import (
    CompositeGaussianLocalLoss,
    LocalLossWeights,
    PublicPairwiseNeighborhood,
)
from matching.sample_roles import (
    KeyedTargetValidationSampler,
    RoleSeparatedValidationEnvironment,
)
from matching.types import FinalSupportBudget, InterventionBackground, ParameterKind


@dataclass(frozen=True)
class HeldoutShortlistConfig:
    """Configuration isolated from every existing exact/finite protocol."""

    final_support_budget: FinalSupportBudget
    finite_search: FiniteCoordinateSearchSettings
    heldout: HeldoutImprovementSettings
    covariance: CovarianceEstimatorSettings
    local_weights: LocalLossWeights = field(default_factory=LocalLossWeights)
    pairwise_neighborhood: PublicPairwiseNeighborhood = field(
        default_factory=PublicPairwiseNeighborhood
    )
    training_current_sample_size: int = 2000
    max_steps: int = 20
    allowed_coordinate_kinds: tuple[ParameterKind, ...] = ("shift", "log_scale")

    def __post_init__(self) -> None:
        if self.finite_search.proposal_mode != "training_point":
            raise ValueError("held-out protocol requires training_point proposal mode.")
        if self.training_current_sample_size != 2000:
            raise ValueError("training current sample size is fixed at 2000.")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise TypeError("max_steps must be an integer.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")


@dataclass(frozen=True)
class HeldoutShortlistRunResult:
    """Public operational result; hidden evaluation is added by the driver."""

    final_intervention: ShiftScaleIntervention
    termination_reason: str
    accepted_updates: int
    rejected_rounds: int
    unresolved_rounds: int
    shortlist_events: tuple[dict[str, object], ...]
    training_search_events: tuple[dict[str, object], ...]
    validation_events: tuple[dict[str, object], ...]
    accepted_interventions: tuple[ShiftScaleIntervention, ...]
    training_background_ids: tuple[str, ...]


def _serialize_search(result: FiniteCoordinateSearchResult) -> dict[str, object]:
    return {
        "coordinate": {
            "node": result.coordinate.node,
            "kind": result.coordinate.kind,
        },
        "status": result.status,
        "reason": result.reason,
        "selected_value": result.selected_value,
        "query_batches": result.query_batches,
        "observations": result.observations,
        "bootstrap_seed": result.bootstrap_seed,
        "proposal_mode": "training_point",
        "selected_intervention": None
        if result.proposal is None
        else {
            "shift": result.proposal.intervention.shift.tolist(),
            "scale": result.proposal.intervention.scale.tolist(),
        },
        "evidence": [
            {
                "value": item.value,
                "candidate_shift": item.intervention.shift.tolist(),
                "candidate_scale": item.intervention.scale.tolist(),
                "point_local_loss": item.point_local_loss,
                "point_improvement": item.point_improvement,
                "local_improvement_interval_diagnostic_only": list(
                    item.improvement_interval
                ),
                "sample_size": item.sample_size,
                "query_ids": list(item.query_ids),
                "covariance_metadata": item.covariance_metadata,
            }
            for item in result.evidence
        ],
    }


def _serialize_validation(decision: HeldoutShortlistDecision) -> dict[str, object]:
    comparisons = []
    for state in decision.comparison_states:
        comparisons.append(
            {
                "candidate_id": state.candidate_id,
                "candidate_shift": state.candidate.shift.tolist(),
                "candidate_scale": state.candidate.scale.tolist(),
                "shortlist_position": state.shortlist_position,
                "terminal_decision": state.terminal_decision,
                "candidate_pool": state.pool.history(),
                "interval_history": [asdict(item) for item in state.interval_history],
            }
        )
    return {
        "decision": decision.decision,
        "accepted_candidate_id": decision.accepted_candidate_id,
        "accepted_shortlist_position": decision.accepted_shortlist_position,
        "accepted_intervention": None
        if decision.accepted_intervention is None
        else {
            "shift": decision.accepted_intervention.shift.tolist(),
            "scale": decision.accepted_intervention.scale.tolist(),
        },
        "final_validation_look": decision.final_look,
        "final_validation_cap": decision.final_cap,
        "shared_target_pool": decision.shared_target_history,
        "shared_current_pool": decision.shared_current_history,
        "sharing_policy": (
            "target_and_current_shared_within_background; candidate_arms_independent"
        ),
        "comparisons": comparisons,
    }


def run_heldout_shortlist(
    public_problem: PublicProblemView,
    graph: PartiallyDirectedGraph,
    training_environment: FiniteSampleBlackBoxEnvironment,
    validation_environment: RoleSeparatedValidationEnvironment,
    target_validation_sampler: KeyedTargetValidationSampler,
    config: HeldoutShortlistConfig,
    *,
    paired_problem_id: str,
    max_validation_cap: int,
    training_seed: int,
    bootstrap_seed: int,
) -> HeldoutShortlistRunResult:
    """Run proposal-only local search and held-out global decisions."""
    if not isinstance(public_problem, PublicProblemView):
        raise TypeError("public_problem must be PublicProblemView.")
    if not isinstance(graph, PartiallyDirectedGraph):
        raise TypeError("graph must be PartiallyDirectedGraph.")
    moments = EmpiricalMomentsProvider(config.covariance)
    current = public_problem.current_intervention
    baselines = AcceptedBackgroundBaselineManager(current)
    seed_rng = np.random.default_rng(training_seed)

    def next_seed() -> int:
        return int(seed_rng.integers(0, np.iinfo(np.int64).max))

    baseline = baselines.ensure_sampled(
        training_environment,
        sample_size=config.training_current_sample_size,
        query_seed=next_seed(),
        moments_provider=moments,
    )
    shortlist_events: list[dict[str, object]] = []
    search_events: list[dict[str, object]] = []
    validation_events: list[dict[str, object]] = []
    accepted_interventions: list[ShiftScaleIntervention] = []
    accepted = rejected = unresolved = 0
    termination = "iteration_limit_reached"

    for step in range(config.max_steps):
        if baseline.samples is None:
            raise RuntimeError("training baseline omitted explicit samples.")
        estimate = moments.estimate(baseline.samples)
        local_provider = _FixedMomentsProvider(
            estimate.mean,
            estimate.regularized_covariance,
        )
        node_losses = CompositeGaussianLocalLoss(
            local_provider,
            config.local_weights,
            config.pairwise_neighborhood,
        ).node_losses(public_problem, current, graph)
        shortlist = deterministic_top3_shortlist(
            [item.composite for item in node_losses],
            graph,
            current,
            config.final_support_budget,
            allowed_coordinate_kinds=config.allowed_coordinate_kinds,
        )
        shortlist_events.append(
            {
                "step": step,
                "background_id": baselines.background_id,
                "training_current_sample_size": baseline.sample_size,
                "training_current_query_ids": list(baseline.query_ids),
                "training_current_covariance_metadata": estimate.metadata(),
                "node_local_losses": [item.composite for item in node_losses],
                "rule_identifier": shortlist.rule_identifier,
                "shortlist_size_configuration": shortlist.shortlist_size,
                "kind_order": list(shortlist.kind_order),
                "node_order": list(shortlist.node_order),
                "shortlist": [entry.as_dict() for entry in shortlist.entries],
                "legality_and_tie_trace": list(shortlist.trace),
                "information_scope": "public_training_only",
            }
        )
        if not shortlist.entries:
            termination = "no_legal_shortlist_coordinate"
            break

        searcher = FiniteSampleCoordinateSearch(
            training_environment,
            moments,
            config.local_weights,
            config.pairwise_neighborhood,
            graph,
            config.finite_search,
        )
        candidates: list[ShiftScaleIntervention] = []
        candidate_ids: set[str] = set()
        step_searches = []
        for entry in shortlist.entries:
            result = searcher.search(
                current,
                entry.coordinate,
                public_problem,
                baseline,
                config.final_support_budget,
                search_seed=next_seed(),
                bootstrap_seed=bootstrap_seed + step * 1_009 + entry.shortlist_position,
            )
            serialized = _serialize_search(result)
            serialized["step"] = step
            serialized["shortlist_position"] = entry.shortlist_position
            serialized["information_scope"] = "public_training_only"
            step_searches.append(serialized)
            if result.proposal is None:
                continue
            candidate = result.proposal.intervention
            candidate_id = InterventionBackground.from_intervention(candidate).identifier
            current_id = InterventionBackground.from_intervention(current).identifier
            if candidate_id == current_id or candidate_id in candidate_ids:
                serialized["validation_eligibility"] = "duplicate_or_current_intervention"
                continue
            serialized["validation_eligibility"] = "eligible_frozen_scalar_candidate"
            candidate_ids.add(candidate_id)
            candidates.append(candidate)
        search_events.extend(step_searches)
        if not candidates:
            rejected += 1
            termination = "no_distinct_training_candidate"
            break

        heldout = HeldoutShortlistEvaluator(
            validation_environment,
            target_validation_sampler,
            moments,
            config.heldout,
        ).evaluate(
            current,
            tuple(candidates),
            paired_problem_id=paired_problem_id,
            max_validation_cap=max_validation_cap,
            bootstrap_seed=bootstrap_seed + step * 100_003,
        )
        validation_row = _serialize_validation(heldout)
        validation_row.update(
            {
                "step": step,
                "background_id": baselines.background_id,
                "candidate_scalar_values_frozen_before_validation": True,
                "information_scope": "public_validation_only",
            }
        )
        validation_events.append(validation_row)
        if heldout.decision == "accepted":
            if heldout.accepted_intervention is None:
                raise RuntimeError("accepted held-out decision omitted intervention.")
            current = heldout.accepted_intervention
            accepted_interventions.append(current)
            accepted += 1
            baselines.accept(current)
            baseline = baselines.ensure_sampled(
                training_environment,
                sample_size=config.training_current_sample_size,
                query_seed=next_seed(),
                moments_provider=moments,
            )
            continue
        if heldout.decision == "rejected":
            rejected += 1
            termination = "all_shortlisted_candidates_rejected"
        else:
            unresolved += 1
            termination = "heldout_comparison_unresolved_at_cap"
        break

    return HeldoutShortlistRunResult(
        final_intervention=current,
        termination_reason=termination,
        accepted_updates=accepted,
        rejected_rounds=rejected,
        unresolved_rounds=unresolved,
        shortlist_events=tuple(shortlist_events),
        training_search_events=tuple(search_events),
        validation_events=tuple(validation_events),
        accepted_interventions=tuple(accepted_interventions),
        training_background_ids=tuple(
            item.background_id for item in baselines.history
        ),
    )
