"""Runnable exact-moment and finite-sample non-oracle experiment assemblies."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, replace
from itertools import count
from time import perf_counter
from typing import Literal

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.meek import MeekOrientationConflict, MeekOrientationConflictDetail
from graphs.pdag import PartiallyDirectedGraph
from graphs.problem_gen import PublicProblemView
from graphs.support import coordinate_support_masks
from matching.baselines import AcceptedBackgroundBaselineManager
from matching.blackbox import (
    EnvironmentQueryLedger,
    ExactMomentBlackBoxEnvironment,
    FiniteSampleBlackBoxEnvironment,
)
from matching.capabilities import (
    AssemblyComponent,
    ProviderCapability,
    normalized_capabilities,
    validate_experiment_components,
)
from matching.coordinate_search import (
    ExactBlackBoxMomentsProvider,
    ExactCoordinateSearchSettings,
    ExactPublicMomentCoordinateOptimizer,
    FiniteCoordinateSearchSettings,
    FiniteSampleCoordinateSearch,
    _FixedMomentsProvider,
)
from matching.empirical_decisions import (
    ConfidenceAwareCandidateEvaluator,
    EmpiricalAcceptanceSettings,
    EmpiricalCandidateDecision,
    EmpiricalGlobalKLEstimate,
    estimate_empirical_global_success,
)
from matching.empirical_moments import (
    CovarianceEstimatorSettings,
    EmpiricalMomentsProvider,
)
from matching.empirical_unresolved import (
    EmpiricalUnresolvedState,
    classify_empirical_unresolved_nodes,
)
from matching.local_evidence import (
    AdaptiveConfidenceCertificateLocalEvidenceProvider,
    AdaptiveTargetSampleProvider,
    LocalEvidenceProvider,
)
from matching.local_losses import (
    CompositeGaussianLocalLoss,
    LocalLossWeights,
    NodeLocalLoss,
    PublicPairwiseNeighborhood,
)
from matching.optimizers import FinalSupportBudgetExceededError
from matching.public_orientation import (
    OrientationEvidence,
    orient_from_public_source_classifications,
)
from matching.source_inference import (
    FiniteSourceInferenceSettings,
    SourceTolerance,
    classify_exact_source_targets,
)
from matching.source_providers import (
    FiniteSourceInferenceProvider,
    StatisticalFiniteSourceInferenceProvider,
)
from matching.structure_learning import (
    EstimatedPCCPDAGProvider,
    StructureEstimate,
    SuppliedExactCPDAGProvider,
)
from matching.surrogate_diagnostics import (
    SurrogateDiagnosticSummary,
    SurrogateProposalDiagnostic,
    global_acceptance_decision,
    summarize_surrogate_diagnostics,
)
from matching.types import (
    CandidateProposal,
    FinalSupportBudget,
    ParameterCoordinate,
    ParameterKind,
)
from matching.unresolved_state import analyze_unresolved_state
from metrics.gaussian_kl import gaussian_kl_target_to_candidate

NonOracleMode = Literal[
    "non_oracle_noiseless",
    "finite_sample_non_oracle",
    "non_oracle_noiseless_supplied_cpdag",
    "non_oracle_noiseless_estimated_cpdag",
    "finite_sample_non_oracle_supplied_cpdag",
    "finite_sample_non_oracle_estimated_cpdag",
    "exact_nonrealisable",
    "finite_sample_nonrealisable",
]


@dataclass(frozen=True)
class ProbeRecoverySettings:
    """Probe amplitudes, sample growth, repeats, and target-set recovery."""

    initial_probe_sample_size: int
    max_probe_sample_size: int
    sample_growth_factor: float
    max_probe_repeats: int
    probe_amplitude_schedule: tuple[float, ...]
    max_probe_targets: int
    source_tolerance_absolute: float
    source_tolerance_relative: float
    source_confidence_level: float

    def __post_init__(self) -> None:
        for name in (
            "initial_probe_sample_size",
            "max_probe_sample_size",
            "max_probe_repeats",
            "max_probe_targets",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.initial_probe_sample_size > self.max_probe_sample_size:
            raise ValueError("initial probe sample size exceeds maximum.")
        if self.sample_growth_factor <= 1.0:
            raise ValueError("sample_growth_factor must exceed one.")
        if not self.probe_amplitude_schedule or any(
            not np.isfinite(value) or value == 0.0
            for value in self.probe_amplitude_schedule
        ):
            raise ValueError("probe amplitudes must be finite and nonzero.")
        if self.source_tolerance_absolute < 0.0:
            raise ValueError("source_tolerance_absolute must be nonnegative.")
        if self.source_tolerance_relative < 0.0:
            raise ValueError("source_tolerance_relative must be nonnegative.")
        if not 0.0 < self.source_confidence_level < 1.0:
            raise ValueError("source_confidence_level must lie in (0, 1).")


@dataclass(frozen=True)
class NonOracleExperimentConfig:
    """Configuration shared by exact and finite non-oracle runners."""

    final_support_budget: FinalSupportBudget
    exact_search: ExactCoordinateSearchSettings
    finite_search: FiniteCoordinateSearchSettings
    acceptance: EmpiricalAcceptanceSettings
    covariance: CovarianceEstimatorSettings
    probe: ProbeRecoverySettings
    local_weights: LocalLossWeights = field(default_factory=LocalLossWeights)
    pairwise_neighborhood: PublicPairwiseNeighborhood = field(
        default_factory=PublicPairwiseNeighborhood
    )
    local_loss_tolerance: float = 1e-4
    global_loss_tolerance: float = 1e-4
    improvement_tolerance: float = 1e-6
    bootstrap_replicates: int = 100
    confidence_level: float = 0.95
    baseline_sample_size: int = 200
    max_steps: int | None = 50
    patience: int = 5
    allowed_coordinate_kinds: tuple[ParameterKind, ...] = ("shift", "log_scale")
    accepted_baseline_sample_size: int | None = None
    record_covariance_estimates: bool = False
    adaptive_local_evidence_initial_sample_size: int | None = None
    adaptive_local_evidence_batch_size: int | None = None
    adaptive_local_evidence_max_sample_size: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.allowed_coordinate_kinds, tuple)
            or not self.allowed_coordinate_kinds
            or len(set(self.allowed_coordinate_kinds))
            != len(self.allowed_coordinate_kinds)
            or any(
                kind not in {"shift", "log_scale"}
                for kind in self.allowed_coordinate_kinds
            )
        ):
            raise ValueError(
                "allowed_coordinate_kinds must contain shift and/or log_scale."
            )
        if not isinstance(self.final_support_budget, FinalSupportBudget):
            raise TypeError("final_support_budget must be FinalSupportBudget.")
        if self.local_loss_tolerance < 0.0:
            raise ValueError("local_loss_tolerance must be nonnegative.")
        if self.global_loss_tolerance < 0.0:
            raise ValueError("global_loss_tolerance must be nonnegative.")
        if self.improvement_tolerance < 0.0:
            raise ValueError("improvement_tolerance must be nonnegative.")
        for name in (
            "bootstrap_replicates",
            "baseline_sample_size",
            "patience",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_steps is not None and (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer or None.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1).")
        if self.accepted_baseline_sample_size is not None and (
            isinstance(self.accepted_baseline_sample_size, bool)
            or not isinstance(self.accepted_baseline_sample_size, int)
            or self.accepted_baseline_sample_size <= 0
        ):
            raise ValueError("accepted_baseline_sample_size must be positive or None.")
        if not isinstance(self.record_covariance_estimates, bool):
            raise TypeError("record_covariance_estimates must be a boolean.")
        adaptive_values = (
            self.adaptive_local_evidence_initial_sample_size,
            self.adaptive_local_evidence_batch_size,
            self.adaptive_local_evidence_max_sample_size,
        )
        if any(value is not None for value in adaptive_values):
            if any(value is None for value in adaptive_values):
                raise ValueError(
                    "adaptive local-evidence settings must be configured together."
                )
            initial, batch, maximum = adaptive_values
            for name, value in zip(
                (
                    "adaptive_local_evidence_initial_sample_size",
                    "adaptive_local_evidence_batch_size",
                    "adaptive_local_evidence_max_sample_size",
                ),
                adaptive_values,
                strict=True,
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer.")
            if maximum < initial:
                raise ValueError(
                    "adaptive sample cap cannot be below the initial size."
                )
            if (maximum - initial) % batch:
                raise ValueError(
                    "adaptive sample cap must be reachable by whole sample batches."
                )


@dataclass(frozen=True)
class NonOracleStepRecord:
    """One public algorithm action and its decision metadata."""

    step: int
    action: str
    background_id: str
    query_ids: tuple[str, ...]
    coordinate: ParameterCoordinate | None
    decision: str
    global_loss_before: float | None
    global_loss_after: float | None
    predicted_local_improvement: float | None = None
    actual_exact_global_improvement: float | None = None
    empirical_global_improvement: float | None = None
    local_terms_used: tuple[str, ...] = ()
    classifications: tuple[tuple[int, str], ...] = ()
    candidate_shift: tuple[float, ...] = ()
    candidate_scale: tuple[float, ...] = ()
    decision_metadata: tuple[dict[str, object], ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class NonOracleExperimentResult:
    """Final public state, graph evidence, uncertainty, and cost accounting."""

    mode: NonOracleMode
    final_intervention: ShiftScaleIntervention
    final_graph: PartiallyDirectedGraph
    structure_estimate: StructureEstimate
    final_global_loss: float
    final_global_interval: tuple[float, float] | None
    termination_reason: str
    history: tuple[NonOracleStepRecord, ...]
    direct_orientation_evidence: tuple[OrientationEvidence, ...]
    meek_orientation_evidence: tuple[OrientationEvidence, ...]
    query_accounting: dict[str, int | None]
    baseline_background_ids: tuple[str, ...]
    accepted_proposals: int
    rejected_proposals: int
    inconclusive_proposals: int
    optimizer_search_evaluations: int
    final_local_losses: tuple[float, ...]
    surrogate_proposal_diagnostics: tuple[SurrogateProposalDiagnostic, ...]
    surrogate_diagnostic_summary: SurrogateDiagnosticSummary
    public_action_signature: tuple[str, ...]
    orientation_conflicts: tuple[MeekOrientationConflictDetail, ...] = ()
    covariance_diagnostics: tuple[dict[str, object], ...] = ()


@dataclass
class FiniteStageAudit:
    """Optional default-off finite-pipeline reachability audit."""

    actionable_source_events: int = 0
    coordinate_searches_started: int = 0
    candidate_proposals_constructed: int = 0
    acceptance_evaluations_started: int = 0
    coordinates: list[str] = field(default_factory=list)
    local_state_events: list[dict[str, object]] = field(default_factory=list)
    local_evidence_events: list[dict[str, object]] = field(default_factory=list)
    structural_state_events: list[dict[str, object]] = field(default_factory=list)


def _finalize_unreached_structural_audit(
    stage_audit: FiniteStageAudit | None,
    *,
    state: str,
    reason: str,
) -> None:
    """Record why structural certification was not evaluated at this step."""
    if stage_audit is None or not stage_audit.local_evidence_events:
        return
    for item in stage_audit.local_evidence_events[-1]["node_decisions"]:
        item["structural_certificate_state"] = state
        item["structural_decision_reason"] = reason
        item["structurally_actionable_source"] = None
        item["top_unresolved_chain_component"] = None


InitialStructureProvider = SuppliedExactCPDAGProvider | EstimatedPCCPDAGProvider


def run_nonoracle_noiseless(
    public_problem: PublicProblemView,
    environment: ExactMomentBlackBoxEnvironment,
    structure_provider: InitialStructureProvider,
    config: NonOracleExperimentConfig,
    *,
    result_mode: NonOracleMode | None = None,
) -> NonOracleExperimentResult:
    """Run exact black-box matching without DAG, B, support, or gradients."""
    _validate_common_inputs(public_problem, config)
    structure = _structure_estimate(structure_provider, public_problem)
    if result_mode is None:
        result_mode = (
            "non_oracle_noiseless_estimated_cpdag"
            if isinstance(structure_provider, EstimatedPCCPDAGProvider)
            else "non_oracle_noiseless_supplied_cpdag"
        )
    validate_experiment_components(
        "non_oracle_noiseless",
        (
            AssemblyComponent("environment", environment),
            AssemblyComponent("initial_structure", structure_provider),
        ),
    )
    current = public_problem.current_intervention
    baselines = AcceptedBackgroundBaselineManager(current)
    baseline = baselines.ensure_exact(environment)
    if baseline.exact_mean is None or baseline.exact_covariance is None:
        raise RuntimeError("exact baseline was not initialized.")
    current_loss = gaussian_kl_target_to_candidate(
        public_problem.target_mean,
        public_problem.target_covariance,
        baseline.exact_mean,
        baseline.exact_covariance,
    )
    graph = structure.graph
    history: list[NonOracleStepRecord] = []
    direct_evidence: list[OrientationEvidence] = []
    meek_evidence: list[OrientationEvidence] = []
    rejected_coordinates: set[ParameterCoordinate] = set()
    accepted = 0
    rejected = 0
    inconclusive = 0
    search_evaluations = 0
    patience = 0
    final_local: tuple[NodeLocalLoss, ...] = ()
    termination = "iteration_limit_reached"
    orientation_conflicts = list(structure.metadata.orientation_conflicts)
    if orientation_conflicts and not isinstance(
        structure_provider, EstimatedPCCPDAGProvider
    ):
        raise MeekOrientationConflict(orientation_conflicts[0])

    steps = (
        ()
        if orientation_conflicts
        else (count() if config.max_steps is None else range(config.max_steps))
    )
    if orientation_conflicts:
        termination = "inconsistent_estimated_cpdag"
    for step in steps:
        final_local = _exact_node_losses(
            public_problem,
            current,
            graph,
            baseline.exact_mean,
            baseline.exact_covariance,
            config,
        )
        unresolved = frozenset(
            item.node
            for item in final_local
            if item.composite > config.local_loss_tolerance
        )
        if current_loss <= config.global_loss_tolerance:
            termination = "loss_tolerance_reached"
            break
        if not unresolved:
            termination = "global_mismatch_not_localized"
            break
        snapshot = analyze_unresolved_state(graph, unresolved)
        sources = snapshot.certified_actionable_sources

        if not sources:
            try:
                update = _exact_exploration(
                    environment,
                    baselines,
                    baseline.exact_mean,
                    graph,
                    unresolved,
                    config,
                    step,
                    history,
                )
            except RuntimeError as error:
                termination = _budget_or_evidence_reason(error)
                break
            except MeekOrientationConflict as error:
                if not isinstance(structure_provider, EstimatedPCCPDAGProvider):
                    raise
                orientation_conflicts.append(error.detail)
                termination = "inconsistent_estimated_cpdag"
                break
            if update is None:
                termination = "insufficient_evidence"
                break
            graph, new_direct, new_meek = update
            direct_evidence.extend(new_direct)
            meek_evidence.extend(new_meek)
            continue

        moments_provider = ExactBlackBoxMomentsProvider(environment, current)
        local_loss = CompositeGaussianLocalLoss(
            moments_provider,
            config.local_weights,
            config.pairwise_neighborhood,
        )
        optimizer = ExactPublicMomentCoordinateOptimizer(
            moments_provider,
            local_loss,
            graph,
            config.exact_search,
        )
        validate_experiment_components(
            "non_oracle_noiseless",
            (
                AssemblyComponent("environment", environment),
                AssemblyComponent("optimizer", optimizer),
                AssemblyComponent("moments", moments_provider),
            ),
        )
        shifted, scaled = coordinate_support_masks(current.shift, current.scale)
        remaining_support = config.final_support_budget.remaining_after(current)
        coordinates = tuple(
            ParameterCoordinate(node, kind)
            for node in sorted(sources)
            for kind in (
                ("shift", "log_scale") if graph.parents_of(node) else ("shift",)
            )
            if kind in config.allowed_coordinate_kinds
            if ParameterCoordinate(node, kind) not in rejected_coordinates
            and (
                bool(shifted[node] if kind == "shift" else scaled[node])
                or remaining_support > 0
            )
        )
        proposals: list[
            tuple[float, CandidateProposal, ParameterCoordinate, tuple[str, ...]]
        ] = []
        try:
            for coordinate in coordinates:
                query_start = len(moments_provider.query_ids)
                proposal = optimizer.optimize(
                    current,
                    (coordinate,),
                    public_problem,
                    config.final_support_budget,
                )
                trace = optimizer.last_trace
                if trace is None:
                    raise RuntimeError("exact optimizer omitted its trace.")
                search_evaluations += len(trace.tested_values)
                local_before = next(
                    item.composite
                    for item in final_local
                    if item.node == coordinate.node
                )
                proposals.append(
                    (
                        local_before - trace.selected_local_loss,
                        proposal,
                        coordinate,
                        moments_provider.query_ids[query_start:],
                    )
                )
        except FinalSupportBudgetExceededError:
            termination = "final_support_exhausted"
            break
        except RuntimeError as error:
            termination = _budget_or_evidence_reason(error)
            break
        if not proposals:
            termination = (
                "final_support_exhausted"
                if remaining_support == 0
                else "no_legal_action"
            )
            break
        predicted_local_improvement, proposal, coordinate, query_ids = max(
            proposals,
            key=lambda item: (item[0], -item[2].node, item[2].kind),
        )
        proposal_mean, proposal_covariance = moments_provider.moments(
            proposal.intervention,
            public_problem,
        )
        candidate_loss = gaussian_kl_target_to_candidate(
            public_problem.target_mean,
            public_problem.target_covariance,
            proposal_mean,
            proposal_covariance,
        )
        is_accepted = global_acceptance_decision(
            current_loss,
            candidate_loss,
            config.improvement_tolerance,
        )
        history.append(
            NonOracleStepRecord(
                step=step,
                action="exact_coordinate_evaluation",
                background_id=baselines.background_id,
                query_ids=query_ids,
                coordinate=coordinate,
                decision="accepted" if is_accepted else "rejected",
                global_loss_before=current_loss,
                global_loss_after=candidate_loss,
                predicted_local_improvement=predicted_local_improvement,
                actual_exact_global_improvement=current_loss - candidate_loss,
                local_terms_used=next(
                    item.used_terms
                    for item in final_local
                    if item.node == coordinate.node
                ),
            )
        )
        if is_accepted:
            current = proposal.intervention
            current_loss = candidate_loss
            accepted += 1
            patience = 0
            rejected_coordinates.clear()
            baselines.accept(current)
            baseline = baselines.ensure_exact(environment)
            if baseline.exact_mean is None or baseline.exact_covariance is None:
                raise RuntimeError("new exact baseline was not initialized.")
        else:
            rejected += 1
            patience += 1
            rejected_coordinates.add(coordinate)
            if patience >= config.patience:
                termination = "patience_exhausted"
                break

    if final_local:
        final_local_losses = tuple(item.composite for item in final_local)
    else:
        final_local_losses = ()
    return _result(
        mode=result_mode,
        current=current,
        graph=graph,
        structure=structure,
        current_loss=current_loss,
        interval=None,
        termination=termination,
        history=history,
        direct_evidence=direct_evidence,
        meek_evidence=meek_evidence,
        ledger=environment.ledger,
        baselines=baselines,
        accepted=accepted,
        rejected=rejected,
        inconclusive=inconclusive,
        search_evaluations=search_evaluations,
        final_local_losses=final_local_losses,
        orientation_conflicts=orientation_conflicts,
    )


def _sample_pool_fingerprint(samples: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(samples.shape).encode())
    digest.update(str(samples.dtype).encode())
    digest.update(np.ascontiguousarray(samples).tobytes())
    return digest.hexdigest()


def _adaptive_confidence_application(
    *,
    raw_state: EmpiricalUnresolvedState,
    provider: AdaptiveConfidenceCertificateLocalEvidenceProvider,
    target_provider: AdaptiveTargetSampleProvider,
    target_samples: np.ndarray,
    public_problem: PublicProblemView,
    environment: FiniteSampleBlackBoxEnvironment,
    baselines: AcceptedBackgroundBaselineManager,
    baseline,
    moments_provider: EmpiricalMomentsProvider,
    config: NonOracleExperimentConfig,
    current: ShiftScaleIntervention,
    graph: PartiallyDirectedGraph,
    seed_rng: np.random.Generator,
    bootstrap_seed: int,
    step: int,
) -> tuple[
    EmpiricalUnresolvedState,
    tuple[dict[str, object], ...],
    object,
    np.ndarray,
    EmpiricalGlobalKLEstimate,
]:
    """Add deterministic batches until the local certificate resolves or caps."""
    initial = config.adaptive_local_evidence_initial_sample_size
    batch_size = config.adaptive_local_evidence_batch_size
    sample_cap = config.adaptive_local_evidence_max_sample_size
    if initial is None or batch_size is None or sample_cap is None:
        raise ValueError("adaptive confidence requires explicit sample settings.")
    started = perf_counter()
    local_problem = public_problem
    application = provider.apply(
        raw_state,
        intervention=current,
        problem=local_problem,
        graph=graph,
        weights=config.local_weights,
        neighborhood=config.pairwise_neighborhood,
    )
    initial_state = application.state
    round_states: list[EmpiricalUnresolvedState] = [initial_state]
    evidence_states: list[EmpiricalUnresolvedState] = [initial_state]
    locked_nodes = {
        item.node: item for item in initial_state.nodes if item.state != "uncertain"
    }
    round_records: list[dict[str, object]] = [
        {
            "adaptive_stage": 0,
            "current_sample_count": baseline.sample_size,
            "target_sample_count": int(target_samples.shape[0]),
            "current_pool_fingerprint": _sample_pool_fingerprint(baseline.samples),
            "target_pool_fingerprint": _sample_pool_fingerprint(target_samples),
            "current_batch_size": 0,
            "target_batch_size": 0,
            "current_batch_seed": None,
            "target_batch_seed": None,
            "current_batch_fingerprint": None,
            "target_batch_fingerprint": None,
        }
    ]
    current_extra = 0
    target_extra = 0
    adaptive_rounds = 0
    while application.state.uncertain_nodes and baseline.sample_size < sample_cap:
        adaptive_rounds += 1
        current_batch_size = min(batch_size, sample_cap - baseline.sample_size)
        target_batch_size = min(batch_size, sample_cap - target_samples.shape[0])
        current_seed = _next_seed(seed_rng)
        baseline = baselines.extend_sampled(
            environment,
            sample_size=current_batch_size,
            query_seed=current_seed,
            moments_provider=moments_provider,
        )
        current_metadata = environment.query_history[-1]
        current_batch = baseline.samples[-current_batch_size:]
        current_extra += current_batch_size
        target_batch_seed = None
        target_batch_fingerprint = None
        if target_batch_size:
            target_batch = target_provider.sample_batch(
                sample_size=target_batch_size,
                adaptive_stage=adaptive_rounds,
                replication_index=(target_samples.shape[0] - initial) // batch_size,
            )
            if target_batch.samples.shape[1] != public_problem.n_nodes:
                raise ValueError("adaptive target sample dimension mismatch.")
            target_samples = np.concatenate(
                (target_samples, target_batch.samples), axis=0
            )
            target_batch_seed = target_batch.batch_seed
            target_batch_fingerprint = target_batch.sample_fingerprint
            target_extra += target_batch_size
        target_estimate = moments_provider.estimate(target_samples)
        local_problem = replace(
            public_problem,
            target_mean=target_estimate.mean,
            target_covariance=target_estimate.regularized_covariance,
        )
        raw_state = classify_empirical_unresolved_nodes(
            baseline.samples,
            current,
            local_problem,
            graph,
            moments_provider,
            config.local_weights,
            config.pairwise_neighborhood,
            local_loss_tolerance=config.local_loss_tolerance,
            bootstrap_replicates=config.bootstrap_replicates,
            confidence_level=config.confidence_level,
            bootstrap_seed=bootstrap_seed + step * 101 + adaptive_rounds,
        )
        evidence_states.append(raw_state)
        effective_nodes = []
        for item in raw_state.nodes:
            effective = locked_nodes.get(item.node, item)
            if effective.state != "uncertain":
                locked_nodes[item.node] = effective
            effective_nodes.append(effective)
        effective_state = EmpiricalUnresolvedState(tuple(effective_nodes))
        application = provider.apply(
            effective_state,
            intervention=current,
            problem=local_problem,
            graph=graph,
            weights=config.local_weights,
            neighborhood=config.pairwise_neighborhood,
        )
        round_states.append(application.state)
        round_records.append(
            {
                "adaptive_stage": adaptive_rounds,
                "current_sample_count": baseline.sample_size,
                "target_sample_count": int(target_samples.shape[0]),
                "current_pool_fingerprint": _sample_pool_fingerprint(baseline.samples),
                "target_pool_fingerprint": _sample_pool_fingerprint(target_samples),
                "current_batch_size": current_batch_size,
                "target_batch_size": target_batch_size,
                "current_batch_seed": current_metadata.query_seed,
                "target_batch_seed": target_batch_seed,
                "current_batch_fingerprint": _sample_pool_fingerprint(current_batch),
                "target_batch_fingerprint": target_batch_fingerprint,
            }
        )
    current_is_source = np.array_equal(
        current.shift, public_problem.current_intervention.shift
    ) and np.array_equal(current.scale, public_problem.current_intervention.scale)
    enriched = []
    for final_decision in application.decisions:
        node = int(final_decision["node"])
        evidence_by_round = []
        for state, metadata in zip(evidence_states, round_records, strict=True):
            item = state.nodes[node]
            evidence_by_round.append(
                {
                    **metadata,
                    "point_local_loss": item.point_local_loss,
                    "confidence_interval": list(item.confidence_interval),
                    "confidence_state": item.state,
                    "bootstrap_seed": item.bootstrap_seed,
                    "bootstrap_replicates": item.bootstrap_replicates,
                }
            )
        initial_node = round_states[0].nodes[node]
        final_node = round_states[-1].nodes[node]
        unresolved_at_cap = (
            final_node.state == "uncertain" and baseline.sample_size >= sample_cap
        )
        resolved_by_sampling = (
            initial_node.state == "uncertain"
            and final_node.state != "uncertain"
            and adaptive_rounds > 0
        )
        if unresolved_at_cap:
            reason_code = "unresolved_at_sample_cap"
        elif resolved_by_sampling:
            reason_code = (
                "resolved_above_threshold_after_sampling"
                if final_node.state == "unresolved"
                else "resolved_below_threshold_after_sampling"
            )
        elif initial_node.state != "uncertain":
            reason_code = "initial_interval_decisive"
        else:
            reason_code = str(final_decision["decision_reason"])
        record = dict(final_decision)
        record.update(
            {
                "initial_sample_count": initial,
                "initial_current_sample_count": round_records[0][
                    "current_sample_count"
                ],
                "initial_target_sample_count": round_records[0]["target_sample_count"],
                "final_pooled_sample_count": min(
                    baseline.sample_size, target_samples.shape[0]
                ),
                "configured_sample_cap": sample_cap,
                "number_of_adaptive_rounds": adaptive_rounds,
                "sample_count_per_distribution": {
                    "source_or_candidate": baseline.sample_size,
                    "target": int(target_samples.shape[0]),
                },
                "batch_sizes": {
                    "source_or_candidate": [
                        item["current_batch_size"] for item in round_records[1:]
                    ],
                    "target": [item["target_batch_size"] for item in round_records[1:]],
                },
                "batch_seeds": {
                    "source_or_candidate": [
                        item["current_batch_seed"] for item in round_records[1:]
                    ],
                    "target": [item["target_batch_seed"] for item in round_records[1:]],
                },
                "nested_sample_fingerprints": [
                    {
                        "adaptive_stage": item["adaptive_stage"],
                        "source_or_candidate": item["current_pool_fingerprint"],
                        "target": item["target_pool_fingerprint"],
                    }
                    for item in round_records
                ],
                "adaptive_round_history": evidence_by_round,
                "empirical_point_loss_after_each_round": [
                    item["point_local_loss"] for item in evidence_by_round
                ],
                "confidence_interval_after_each_round": [
                    item["confidence_interval"] for item in evidence_by_round
                ],
                "final_decision": final_node.state,
                "reason_code": reason_code,
                "resolved_because_of_additional_sampling": resolved_by_sampling,
                "unresolved_at_cap": unresolved_at_cap,
                "additional_source_samples": current_extra if current_is_source else 0,
                "additional_target_samples": target_extra,
                "additional_candidate_environment_samples": (
                    0 if current_is_source else current_extra
                ),
                "total_extra_samples": current_extra + target_extra,
                "adaptive_sampling_runtime_seconds": perf_counter() - started,
            }
        )
        enriched.append(record)
    global_estimate = _global_estimate(
        baseline.samples,
        public_problem,
        moments_provider,
        config,
        bootstrap_seed + step * 101 + adaptive_rounds,
    )
    return (
        application.state,
        tuple(enriched),
        baseline,
        target_samples,
        global_estimate,
    )


def run_finite_sample_nonoracle(
    public_problem: PublicProblemView,
    environment: FiniteSampleBlackBoxEnvironment,
    structure_provider: InitialStructureProvider,
    config: NonOracleExperimentConfig,
    *,
    environment_seed: int,
    bootstrap_seed: int,
    source_inference_provider: FiniteSourceInferenceProvider | None = None,
    local_evidence_provider: LocalEvidenceProvider | None = None,
    adaptive_target_samples: np.ndarray | None = None,
    adaptive_target_sample_provider: AdaptiveTargetSampleProvider | None = None,
    diagnostic_mode: str | None = None,
    result_mode: NonOracleMode | None = None,
    stage_audit: FiniteStageAudit | None = None,
) -> NonOracleExperimentResult:
    """Run sample-only matching with three-way inference and acceptance."""
    if source_inference_provider is None:
        source_inference_provider = StatisticalFiniteSourceInferenceProvider(
            FiniteSourceInferenceSettings(
                config.probe.source_tolerance_absolute,
                config.probe.source_tolerance_relative,
                config.probe.source_confidence_level,
                True,
            )
        )
    if not isinstance(source_inference_provider, FiniteSourceInferenceProvider):
        raise TypeError("source_inference_provider has the wrong interface.")
    source_capabilities = normalized_capabilities(source_inference_provider)
    oracle_source = ProviderCapability.ORACLE in source_capabilities
    permitted_oracle_diagnostics = {
        "exact_cpdag_exact_source_empirical_matching",
        "estimated_cpdag_exact_source_empirical_matching",
        "exact_cpdag_exact_source_oracle_local_actionability_empirical_matching",
    }
    if oracle_source and diagnostic_mode not in permitted_oracle_diagnostics:
        raise ValueError(
            "oracle source inference is allowed only in explicitly labelled diagnostics."
        )
    if local_evidence_provider is not None and not isinstance(
        local_evidence_provider, LocalEvidenceProvider
    ):
        raise TypeError("local_evidence_provider has the wrong interface.")
    adaptive_local_evidence = isinstance(
        local_evidence_provider,
        AdaptiveConfidenceCertificateLocalEvidenceProvider,
    )
    adaptive_configured = config.adaptive_local_evidence_initial_sample_size is not None
    if adaptive_local_evidence != adaptive_configured:
        raise ValueError(
            "adaptive policy selection and adaptive configuration must agree."
        )
    if adaptive_local_evidence:
        if (
            adaptive_target_samples is None
            or adaptive_target_sample_provider is None
            or not isinstance(
                adaptive_target_sample_provider, AdaptiveTargetSampleProvider
            )
        ):
            raise ValueError(
                "adaptive confidence requires initial target samples and a "
                "sample-only target provider."
            )
        if (
            not isinstance(adaptive_target_samples, np.ndarray)
            or adaptive_target_samples.ndim != 2
            or adaptive_target_samples.shape[1] != public_problem.n_nodes
            or not np.all(np.isfinite(adaptive_target_samples))
        ):
            raise ValueError("adaptive initial target samples are invalid.")
        if ProviderCapability.ORACLE in normalized_capabilities(
            adaptive_target_sample_provider
        ):
            raise ValueError("adaptive target sampling must expose samples only.")
        adaptive_target_pool = np.asarray(adaptive_target_samples, dtype=float).copy()
    else:
        if (
            adaptive_target_samples is not None
            or adaptive_target_sample_provider is not None
        ):
            raise ValueError(
                "adaptive target inputs require the explicit adaptive policy."
            )
        adaptive_target_pool = None
    oracle_local_evidence = local_evidence_provider is not None and (
        ProviderCapability.ORACLE in normalized_capabilities(local_evidence_provider)
    )
    if oracle_local_evidence and diagnostic_mode != (
        "exact_cpdag_exact_source_oracle_local_actionability_empirical_matching"
    ):
        raise ValueError(
            "oracle local evidence is allowed only in the explicitly labelled "
            "oracle-local-actionability diagnostic."
        )
    if not oracle_source:
        validate_experiment_components(
            "finite_sample_non_oracle",
            (
                AssemblyComponent("environment", environment),
                AssemblyComponent("source_inference", source_inference_provider),
            ),
        )
    _validate_common_inputs(public_problem, config)
    structure = _structure_estimate(structure_provider, public_problem)
    moments_provider = EmpiricalMomentsProvider(
        config.covariance,
        record_estimates=config.record_covariance_estimates,
    )
    if result_mode is None:
        result_mode = (
            "finite_sample_non_oracle_estimated_cpdag"
            if isinstance(structure_provider, EstimatedPCCPDAGProvider)
            else "finite_sample_non_oracle_supplied_cpdag"
        )
    validate_experiment_components(
        "finite_sample_non_oracle",
        (
            AssemblyComponent("environment", environment),
            AssemblyComponent("initial_structure", structure_provider),
            AssemblyComponent("moments", moments_provider),
        ),
    )
    seed_rng = np.random.default_rng(environment_seed)
    current = public_problem.current_intervention
    baselines = AcceptedBackgroundBaselineManager(current)
    baseline = baselines.ensure_sampled(
        environment,
        sample_size=config.baseline_sample_size,
        query_seed=_next_seed(seed_rng),
        moments_provider=moments_provider,
    )
    if baseline.samples is None:
        raise RuntimeError("finite baseline was not initialized.")
    if adaptive_local_evidence:
        expected_initial = config.adaptive_local_evidence_initial_sample_size
        if (
            baseline.sample_size != expected_initial
            or adaptive_target_pool is None
            or adaptive_target_pool.shape[0] != expected_initial
        ):
            raise ValueError(
                "adaptive current and target pools must start at the configured size."
            )
    global_estimate = _global_estimate(
        baseline.samples,
        public_problem,
        moments_provider,
        config,
        bootstrap_seed,
    )
    graph = structure.graph
    history: list[NonOracleStepRecord] = []
    direct_evidence: list[OrientationEvidence] = []
    meek_evidence: list[OrientationEvidence] = []
    accepted = 0
    rejected = 0
    inconclusive = 0
    search_evaluations = 0
    patience = 0
    termination = "iteration_limit_reached"
    final_local_state: EmpiricalUnresolvedState | None = None
    orientation_conflicts = list(structure.metadata.orientation_conflicts)
    if orientation_conflicts and not isinstance(
        structure_provider, EstimatedPCCPDAGProvider
    ):
        raise MeekOrientationConflict(orientation_conflicts[0])

    if config.max_steps is None:
        raise ValueError("finite-sample execution requires max_steps.")
    if orientation_conflicts:
        termination = "inconsistent_estimated_cpdag"
    steps = () if orientation_conflicts else range(config.max_steps)
    for step in steps:
        if global_estimate.success:
            termination = "confidence_aware_loss_tolerance_reached"
            break
        local_problem = public_problem
        if adaptive_local_evidence:
            if adaptive_target_pool is None:
                raise RuntimeError("adaptive target pool was not initialized.")
            adaptive_target_estimate = moments_provider.estimate(adaptive_target_pool)
            local_problem = replace(
                public_problem,
                target_mean=adaptive_target_estimate.mean,
                target_covariance=(adaptive_target_estimate.regularized_covariance),
            )
        raw_local_state = classify_empirical_unresolved_nodes(
            baseline.samples,
            current,
            local_problem,
            graph,
            moments_provider,
            config.local_weights,
            config.pairwise_neighborhood,
            local_loss_tolerance=config.local_loss_tolerance,
            bootstrap_replicates=config.bootstrap_replicates,
            confidence_level=config.confidence_level,
            bootstrap_seed=bootstrap_seed + step * 101,
        )
        local_decisions: tuple[dict[str, object], ...] = ()
        if local_evidence_provider is None:
            final_local_state = raw_local_state
        else:
            application = local_evidence_provider.apply(
                raw_local_state,
                intervention=current,
                problem=local_problem,
                graph=graph,
                weights=config.local_weights,
                neighborhood=config.pairwise_neighborhood,
            )
            final_local_state = application.state
            local_decisions = application.decisions
        if adaptive_local_evidence:
            if (
                adaptive_target_pool is None
                or adaptive_target_sample_provider is None
                or not isinstance(
                    local_evidence_provider,
                    AdaptiveConfidenceCertificateLocalEvidenceProvider,
                )
            ):
                raise RuntimeError("adaptive evidence inputs disappeared.")
            try:
                (
                    final_local_state,
                    local_decisions,
                    baseline,
                    adaptive_target_pool,
                    global_estimate,
                ) = _adaptive_confidence_application(
                    raw_state=raw_local_state,
                    provider=local_evidence_provider,
                    target_provider=adaptive_target_sample_provider,
                    target_samples=adaptive_target_pool,
                    public_problem=public_problem,
                    environment=environment,
                    baselines=baselines,
                    baseline=baseline,
                    moments_provider=moments_provider,
                    config=config,
                    current=current,
                    graph=graph,
                    seed_rng=seed_rng,
                    bootstrap_seed=bootstrap_seed,
                    step=step,
                )
            except RuntimeError as error:
                termination = _budget_or_evidence_reason(error)
                break
        repeat_failure: str | None = None
        repeat_queries = 0
        baseline_sample_cap = max(
            config.baseline_sample_size,
            config.acceptance.max_sample_size_per_arm,
        )
        while (
            not adaptive_local_evidence
            and final_local_state.uncertain_nodes
            and repeat_queries < config.acceptance.max_repeat_queries
            and baseline.sample_size < baseline_sample_cap
        ):
            previous_loss = global_estimate.point_estimate
            additional_size = min(
                baseline_sample_cap - baseline.sample_size,
                max(
                    2,
                    int(
                        np.ceil(
                            baseline.sample_size
                            * (config.acceptance.sample_growth_factor - 1.0)
                        )
                    ),
                ),
            )
            try:
                baseline = baselines.extend_sampled(
                    environment,
                    sample_size=additional_size,
                    query_seed=_next_seed(seed_rng),
                    moments_provider=moments_provider,
                )
            except RuntimeError as error:
                repeat_failure = _budget_or_evidence_reason(error)
                break
            repeat_queries += 1
            if baseline.samples is None:
                raise RuntimeError("extended finite baseline omitted samples.")
            global_estimate = _global_estimate(
                baseline.samples,
                public_problem,
                moments_provider,
                config,
                bootstrap_seed + step * 101 + repeat_queries,
            )
            raw_local_state = classify_empirical_unresolved_nodes(
                baseline.samples,
                current,
                public_problem,
                graph,
                moments_provider,
                config.local_weights,
                config.pairwise_neighborhood,
                local_loss_tolerance=config.local_loss_tolerance,
                bootstrap_replicates=config.bootstrap_replicates,
                confidence_level=config.confidence_level,
                bootstrap_seed=bootstrap_seed + step * 101 + repeat_queries,
            )
            if local_evidence_provider is None:
                final_local_state = raw_local_state
                local_decisions = ()
            else:
                application = local_evidence_provider.apply(
                    raw_local_state,
                    intervention=current,
                    problem=public_problem,
                    graph=graph,
                    weights=config.local_weights,
                    neighborhood=config.pairwise_neighborhood,
                )
                final_local_state = application.state
                local_decisions = application.decisions
            history.append(
                NonOracleStepRecord(
                    step=step,
                    action="finite_baseline_repeat",
                    background_id=baselines.background_id,
                    query_ids=(baseline.query_ids[-1],),
                    coordinate=None,
                    decision=(
                        "reclassified"
                        if not final_local_state.uncertain_nodes
                        else "inconclusive"
                    ),
                    global_loss_before=previous_loss,
                    global_loss_after=global_estimate.point_estimate,
                    detail=f"baseline_sample_size={baseline.sample_size}",
                )
            )
        if repeat_failure is not None:
            termination = repeat_failure
            break
        if stage_audit is not None:
            stage_audit.local_state_events.append(
                {
                    "step": step,
                    "resolved_nodes": sorted(final_local_state.resolved_nodes),
                    "unresolved_nodes": sorted(final_local_state.unresolved_nodes),
                    "uncertain_nodes": sorted(final_local_state.uncertain_nodes),
                    "node_evidence": [asdict(item) for item in final_local_state.nodes],
                }
            )
            if local_evidence_provider is not None:
                sample_digest = hashlib.sha256()
                sample_digest.update(str(baseline.samples.shape).encode())
                sample_digest.update(str(baseline.samples.dtype).encode())
                sample_digest.update(np.ascontiguousarray(baseline.samples).tobytes())
                stage_audit.local_evidence_events.append(
                    {
                        "step": step,
                        "policy_identifier": local_evidence_provider.policy_identifier,
                        "baseline_sample_size": baseline.sample_size,
                        "baseline_sample_fingerprint": sample_digest.hexdigest(),
                        "intervention_shift": [float(value) for value in current.shift],
                        "intervention_scale": [float(value) for value in current.scale],
                        "graph_directed_edges": sorted(graph.directed_edges),
                        "graph_undirected_edges": sorted(graph.undirected_edges),
                        "node_decisions": [
                            {
                                **dict(item),
                                "directed_parent_set": sorted(
                                    graph.parents_of(int(item["node"]))
                                ),
                            }
                            for item in local_decisions
                        ],
                    }
                )
        if global_estimate.success:
            _finalize_unreached_structural_audit(
                stage_audit,
                state="not_evaluated_global_tolerance_reached",
                reason="no_localizable_mismatch",
            )
            termination = "confidence_aware_loss_tolerance_reached"
            break
        if final_local_state.uncertain_nodes:
            _finalize_unreached_structural_audit(
                stage_audit,
                state="not_evaluated_local_evidence_block",
                reason="ci_crosses_threshold",
            )
            termination = "insufficient_evidence"
            break
        active_nodes = (
            final_local_state.unresolved_nodes | final_local_state.uncertain_nodes
        )
        if not active_nodes:
            _finalize_unreached_structural_audit(
                stage_audit,
                state="not_evaluated_no_localizable_mismatch",
                reason="no_localizable_mismatch",
            )
            termination = "global_mismatch_not_localized"
            break
        snapshot = analyze_unresolved_state(graph, active_nodes)
        sources = snapshot.certified_actionable_sources
        if stage_audit is not None:
            stage_audit.structural_state_events.append(
                {
                    "step": step,
                    "active_nodes": sorted(active_nodes),
                    "certified_actionable_sources": sorted(sources),
                    "top_unresolved_chain_components": [
                        sorted(component)
                        for component in snapshot.top_unresolved_chain_components
                    ],
                    "induced_directed_edges": sorted(
                        snapshot.unresolved_graph.directed_edges
                    ),
                    "induced_undirected_edges": sorted(
                        snapshot.unresolved_graph.undirected_edges
                    ),
                }
            )
            if (
                local_evidence_provider is not None
                and stage_audit.local_evidence_events
            ):
                for item in stage_audit.local_evidence_events[-1]["node_decisions"]:
                    node = int(item["node"])
                    is_source = node in sources
                    item["structurally_actionable_source"] = is_source
                    if is_source:
                        item["structural_certificate_state"] = (
                            "certified_actionable_source"
                        )
                        item["structural_decision_reason"] = (
                            "structural_certificate_obtained"
                        )
                    elif node in active_nodes:
                        item["structural_certificate_state"] = (
                            "structural_source_unavailable"
                        )
                        item["structural_decision_reason"] = (
                            "structural_source_unavailable"
                        )
                    else:
                        item["structural_certificate_state"] = (
                            "not_applicable_local_mismatch_absent"
                        )
                        item["structural_decision_reason"] = "no_localizable_mismatch"
                    item["top_unresolved_chain_component"] = next(
                        (
                            sorted(component)
                            for component in snapshot.top_unresolved_chain_components
                            if node in component
                        ),
                        None,
                    )
        if not sources:
            try:
                update = _finite_exploration(
                    environment,
                    baselines,
                    graph,
                    active_nodes,
                    config,
                    step,
                    seed_rng,
                    history,
                    source_inference_provider,
                )
            except RuntimeError as error:
                termination = _budget_or_evidence_reason(error)
                break
            except MeekOrientationConflict as error:
                if not isinstance(structure_provider, EstimatedPCCPDAGProvider):
                    raise
                orientation_conflicts.append(error.detail)
                termination = "inconsistent_estimated_cpdag"
                break
            if update is None:
                termination = "insufficient_evidence"
                break
            graph, new_direct, new_meek = update
            direct_evidence.extend(new_direct)
            meek_evidence.extend(new_meek)
            continue

        if stage_audit is not None:
            stage_audit.actionable_source_events += len(sources)
        coordinate_search = FiniteSampleCoordinateSearch(
            environment,
            moments_provider,
            config.local_weights,
            config.pairwise_neighborhood,
            graph,
            config.finite_search,
        )
        validate_experiment_components(
            "finite_sample_non_oracle",
            (
                AssemblyComponent("environment", environment),
                AssemblyComponent("optimizer", coordinate_search),
            ),
        )
        search_results = []
        try:
            for node in sorted(sources):
                for kind in (
                    ("shift", "log_scale") if graph.parents_of(node) else ("shift",)
                ):
                    if kind not in config.allowed_coordinate_kinds:
                        continue
                    if stage_audit is not None:
                        stage_audit.coordinate_searches_started += 1
                        stage_audit.coordinates.append(f"{kind}:{node}")
                    result = coordinate_search.search(
                        current,
                        ParameterCoordinate(node, kind),
                        public_problem,
                        baseline,
                        config.final_support_budget,
                        search_seed=_next_seed(seed_rng),
                        bootstrap_seed=bootstrap_seed + step * 1_009 + node,
                    )
                    search_results.append(result)
                    search_evaluations += sum(
                        len(item.query_ids) for item in result.evidence
                    )
        except FinalSupportBudgetExceededError:
            termination = "final_support_exhausted"
            break
        except RuntimeError as error:
            termination = _budget_or_evidence_reason(error)
            break
        supported = [
            result
            for result in search_results
            if result.status == "proposal" and result.proposal is not None
        ]
        if stage_audit is not None:
            stage_audit.candidate_proposals_constructed += len(supported)
        if not supported:
            if any(result.status == "inconclusive" for result in search_results):
                inconclusive += 1
                termination = "insufficient_evidence"
            else:
                rejected += 1
                termination = "no_significant_improvement"
            break
        selected_search = max(
            supported,
            key=lambda result: next(
                item.point_improvement
                for item in result.evidence
                if item.value == result.selected_value
            ),
        )
        proposal = selected_search.proposal
        if proposal is None:
            raise RuntimeError("supported search omitted its proposal.")
        selected_evidence = next(
            item
            for item in selected_search.evidence
            if item.value == selected_search.selected_value
        )
        selected_terms = next(
            item.used_terms
            for item in final_local_state.nodes
            if item.node == selected_search.coordinate.node
        )
        evaluator = ConfidenceAwareCandidateEvaluator(
            environment,
            moments_provider,
            config.acceptance,
        )
        if stage_audit is not None:
            stage_audit.acceptance_evaluations_started += 1
        try:
            decision = evaluator.evaluate(
                current,
                proposal.intervention,
                public_problem,
                baseline,
                environment_seed=_next_seed(seed_rng),
                bootstrap_seed=bootstrap_seed + step * 10_007,
            )
        except RuntimeError as error:
            termination = _budget_or_evidence_reason(error)
            break
        history.append(
            _finite_decision_record(
                step,
                baselines.background_id,
                selected_search.coordinate,
                proposal.intervention,
                decision,
                selected_evidence.point_improvement,
                selected_terms,
                selected_evidence.covariance_metadata,
            )
        )
        if decision.decision == "accepted":
            current = proposal.intervention
            accepted += 1
            patience = 0
            baselines.accept(current)
            baseline = baselines.ensure_sampled(
                environment,
                sample_size=(
                    config.accepted_baseline_sample_size or config.baseline_sample_size
                ),
                query_seed=_next_seed(seed_rng),
                moments_provider=moments_provider,
            )
            if baseline.samples is None:
                raise RuntimeError("new finite baseline was not initialized.")
            global_estimate = _global_estimate(
                baseline.samples,
                public_problem,
                moments_provider,
                config,
                bootstrap_seed + step + 1,
            )
        elif decision.decision == "rejected":
            rejected += 1
            patience += 1
        else:
            inconclusive += 1
            termination = "inconclusive_evaluation_budget_exhausted"
            break
        if patience >= config.patience:
            termination = "patience_exhausted"
            break

    final_local_losses = (
        tuple(item.point_local_loss for item in final_local_state.nodes)
        if final_local_state is not None
        else ()
    )
    return _result(
        mode=result_mode,
        current=current,
        graph=graph,
        structure=structure,
        current_loss=global_estimate.point_estimate,
        interval=global_estimate.confidence_interval,
        termination=termination,
        history=history,
        direct_evidence=direct_evidence,
        meek_evidence=meek_evidence,
        ledger=environment.ledger,
        baselines=baselines,
        accepted=accepted,
        rejected=rejected,
        inconclusive=inconclusive,
        search_evaluations=search_evaluations,
        final_local_losses=final_local_losses,
        orientation_conflicts=orientation_conflicts,
        covariance_diagnostics=moments_provider.audit_log,
    )


def _exact_exploration(
    environment: ExactMomentBlackBoxEnvironment,
    baselines: AcceptedBackgroundBaselineManager,
    baseline_mean: np.ndarray,
    graph: PartiallyDirectedGraph,
    active_nodes: frozenset[int],
    config: NonOracleExperimentConfig,
    step: int,
    history: list[NonOracleStepRecord],
) -> (
    tuple[
        PartiallyDirectedGraph,
        tuple[OrientationEvidence, ...],
        tuple[OrientationEvidence, ...],
    ]
    | None
):
    for target_set in _probe_target_sets(
        graph,
        active_nodes,
        config.probe.max_probe_targets,
    ):
        for amplitude in config.probe.probe_amplitude_schedule:
            increment = np.zeros(baselines.accepted.shift.size)
            increment[list(target_set)] = amplitude
            probe = ShiftScaleIntervention(
                baselines.accepted.shift + increment,
                baselines.accepted.scale,
            )
            result = environment.query(
                probe,
                kind="probe",
                background_id=baselines.background_id,
            )
            classifications = classify_exact_source_targets(
                baseline_mean,
                result.mean,
                increment,
                target_set,
                SourceTolerance(
                    config.probe.source_tolerance_absolute,
                    config.probe.source_tolerance_relative,
                ),
            )
            mapping = {item.node: item.classification for item in classifications}
            update = orient_from_public_source_classifications(
                graph,
                targets=target_set,
                classifications=mapping,
                probe_id=result.metadata.query_id,
                background_id=baselines.background_id,
            )
            history.append(
                NonOracleStepRecord(
                    step=step,
                    action="exact_public_probe",
                    background_id=baselines.background_id,
                    query_ids=(result.metadata.query_id,),
                    coordinate=None,
                    decision=(
                        "oriented" if update.direct_evidence else "no_safe_orientation"
                    ),
                    global_loss_before=None,
                    global_loss_after=None,
                    classifications=tuple(sorted(mapping.items())),
                    detail="intervention_faithfulness_assumed",
                )
            )
            if update.graph != graph:
                return (
                    update.graph,
                    update.direct_evidence,
                    update.meek_evidence,
                )
    return None


def _finite_exploration(
    environment: FiniteSampleBlackBoxEnvironment,
    baselines: AcceptedBackgroundBaselineManager,
    graph: PartiallyDirectedGraph,
    active_nodes: frozenset[int],
    config: NonOracleExperimentConfig,
    step: int,
    seed_rng: np.random.Generator,
    history: list[NonOracleStepRecord],
    source_inference_provider: FiniteSourceInferenceProvider,
) -> (
    tuple[
        PartiallyDirectedGraph,
        tuple[OrientationEvidence, ...],
        tuple[OrientationEvidence, ...],
    ]
    | None
):
    baseline = baselines.current
    if baseline is None or baseline.samples is None:
        raise RuntimeError("finite exploration requires current baseline samples.")
    for target_set in _probe_target_sets(
        graph,
        active_nodes,
        config.probe.max_probe_targets,
    ):
        for amplitude in config.probe.probe_amplitude_schedule:
            increment = np.zeros(baselines.accepted.shift.size)
            increment[list(target_set)] = amplitude
            probe = ShiftScaleIntervention(
                baselines.accepted.shift + increment,
                baselines.accepted.scale,
            )
            probe_chunks: list[np.ndarray] = []
            query_ids: list[str] = []
            sample_size = config.probe.initial_probe_sample_size
            for repeat in range(config.probe.max_probe_repeats):
                result = environment.query(
                    probe,
                    sample_size=sample_size,
                    query_seed=_next_seed(seed_rng),
                    kind="probe" if repeat == 0 else "repeat",
                    budget_category="probe",
                    background_id=baselines.background_id,
                )
                probe_chunks.append(result.samples)
                query_ids.append(result.metadata.query_id)
                classifications = source_inference_provider.classify(
                    baseline.samples,
                    np.concatenate(probe_chunks),
                    baselines.accepted,
                    increment,
                    target_set,
                    background_id=baselines.background_id,
                    paired_samples=environment.paired_samples,
                )
                mapping = {item.node: item.classification for item in classifications}
                update = orient_from_public_source_classifications(
                    graph,
                    targets=target_set,
                    classifications=mapping,
                    probe_id=result.metadata.query_id,
                    background_id=baselines.background_id,
                )
                history.append(
                    NonOracleStepRecord(
                        step=step,
                        action=(
                            "finite_public_probe"
                            if repeat == 0
                            else "finite_public_probe_repeat"
                        ),
                        background_id=baselines.background_id,
                        query_ids=tuple(query_ids),
                        coordinate=None,
                        decision=(
                            "oriented"
                            if update.direct_evidence
                            else (
                                "inconclusive"
                                if "inconclusive" in mapping.values()
                                else "no_safe_orientation"
                            )
                        ),
                        global_loss_before=None,
                        global_loss_after=None,
                        classifications=tuple(sorted(mapping.items())),
                        decision_metadata=tuple(
                            asdict(classification) for classification in classifications
                        ),
                    )
                )
                if update.graph != graph:
                    return (
                        update.graph,
                        update.direct_evidence,
                        update.meek_evidence,
                    )
                if "inconclusive" not in mapping.values():
                    break
                accumulated = sum(chunk.shape[0] for chunk in probe_chunks)
                if accumulated >= config.probe.max_probe_sample_size:
                    break
                sample_size = min(
                    config.probe.max_probe_sample_size - accumulated,
                    max(
                        2,
                        int(np.ceil(sample_size * config.probe.sample_growth_factor)),
                    ),
                )
    return None


def _probe_target_sets(
    graph: PartiallyDirectedGraph,
    active_nodes: frozenset[int],
    max_targets: int,
) -> tuple[frozenset[int], ...]:
    if not active_nodes <= graph.nodes:
        raise ValueError("probe targets must be endogenous graph nodes.")
    if not 0 < max_targets <= len(graph.nodes):
        raise ValueError("max probe targets must lie within endogenous nodes.")
    pairs = [
        frozenset((node_a, node_b))
        for node_a, node_b in sorted(graph.undirected_edges)
        if node_a in active_nodes and node_b in active_nodes
    ]
    larger = [
        frozenset(sorted(component)[:max_targets])
        for component in graph.top_chain_components()
        if len(component & active_nodes) >= 2
    ]
    singletons = [frozenset({node}) for node in sorted(active_nodes)]
    result: list[frozenset[int]] = []
    for targets in (*pairs, *larger, *singletons):
        limited = frozenset(sorted(targets & active_nodes)[:max_targets])
        if limited and limited not in result:
            result.append(limited)
    return tuple(result)


def _exact_node_losses(
    problem: PublicProblemView,
    intervention: ShiftScaleIntervention,
    graph: PartiallyDirectedGraph,
    mean: np.ndarray,
    covariance: np.ndarray,
    config: NonOracleExperimentConfig,
) -> tuple[NodeLocalLoss, ...]:
    return CompositeGaussianLocalLoss(
        _FixedMomentsProvider(mean, covariance),
        config.local_weights,
        config.pairwise_neighborhood,
    ).node_losses(problem, intervention, graph)


def _structure_estimate(
    provider: InitialStructureProvider,
    problem: PublicProblemView,
) -> StructureEstimate:
    if isinstance(provider, SuppliedExactCPDAGProvider):
        estimate = provider.estimate()
    elif isinstance(provider, EstimatedPCCPDAGProvider):
        estimate = provider.estimate(problem.observational_samples)
    else:
        raise TypeError("unsupported non-oracle initial structure provider.")
    if estimate.graph.nodes != frozenset(range(problem.n_nodes)):
        raise ValueError("initial structure dimension must match public problem.")
    return estimate


def _global_estimate(
    samples: np.ndarray,
    problem: PublicProblemView,
    moments: EmpiricalMomentsProvider,
    config: NonOracleExperimentConfig,
    seed: int,
) -> EmpiricalGlobalKLEstimate:
    return estimate_empirical_global_success(
        samples,
        problem,
        moments,
        global_tolerance=config.global_loss_tolerance,
        bootstrap_replicates=config.bootstrap_replicates,
        confidence_level=config.confidence_level,
        bootstrap_seed=seed,
    )


def _finite_decision_record(
    step: int,
    background_id: str,
    coordinate: ParameterCoordinate,
    candidate: ShiftScaleIntervention,
    decision: EmpiricalCandidateDecision,
    predicted_local_improvement: float,
    local_terms_used: tuple[str, ...],
    search_covariance_metadata: dict[str, object],
) -> NonOracleStepRecord:
    return NonOracleStepRecord(
        step=step,
        action="confidence_aware_candidate_evaluation",
        background_id=background_id,
        query_ids=decision.current_query_ids + decision.candidate_query_ids,
        coordinate=coordinate,
        decision=decision.decision,
        global_loss_before=decision.current_global_kl,
        global_loss_after=decision.candidate_global_kl,
        predicted_local_improvement=predicted_local_improvement,
        empirical_global_improvement=decision.point_improvement,
        local_terms_used=local_terms_used,
        candidate_shift=tuple(float(value) for value in candidate.shift),
        candidate_scale=tuple(float(value) for value in candidate.scale),
        decision_metadata=(
            {
                "improvement_interval": list(decision.improvement_interval),
                "bootstrap_seed": decision.bootstrap_seed,
                "bootstrap_replicates": decision.bootstrap_replicates,
                "confidence_level": decision.confidence_level,
                "current_sample_size": decision.current_sample_size,
                "candidate_sample_size": decision.candidate_sample_size,
                "current_covariance": decision.current_covariance_metadata,
                "candidate_covariance": decision.candidate_covariance_metadata,
                "search_covariance": search_covariance_metadata,
                "comparison_sampling": decision.comparison_sampling,
            },
        ),
        detail=decision.final_status,
    )


def _next_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int64).max))


def _budget_or_evidence_reason(error: RuntimeError) -> str:
    message = str(error).lower()
    if "observation budget" in message:
        return "observation_budget_exhausted"
    if "query budget" in message or "experiment budget" in message:
        return "query_budget_exhausted"
    return "numerical_failure"


def _validate_common_inputs(
    problem: PublicProblemView,
    config: NonOracleExperimentConfig,
) -> None:
    if not isinstance(problem, PublicProblemView):
        raise TypeError("problem must be PublicProblemView.")
    if not isinstance(config, NonOracleExperimentConfig):
        raise TypeError("config must be NonOracleExperimentConfig.")
    if not config.final_support_budget.allows(problem.current_intervention):
        raise ValueError("initial intervention exceeds final support budget.")


def _result(
    *,
    mode: NonOracleMode,
    current: ShiftScaleIntervention,
    graph: PartiallyDirectedGraph,
    structure: StructureEstimate,
    current_loss: float,
    interval: tuple[float, float] | None,
    termination: str,
    history: list[NonOracleStepRecord],
    direct_evidence: list[OrientationEvidence],
    meek_evidence: list[OrientationEvidence],
    ledger: EnvironmentQueryLedger,
    baselines: AcceptedBackgroundBaselineManager,
    accepted: int,
    rejected: int,
    inconclusive: int,
    search_evaluations: int,
    final_local_losses: tuple[float, ...],
    orientation_conflicts: list[MeekOrientationConflictDetail],
    covariance_diagnostics: tuple[dict[str, object], ...] = (),
) -> NonOracleExperimentResult:
    diagnostics = tuple(
        SurrogateProposalDiagnostic(
            predicted_local_improvement=record.predicted_local_improvement,
            actual_exact_global_improvement=record.actual_exact_global_improvement,
            empirical_global_improvement=record.empirical_global_improvement,
            status=record.decision,
            coordinate=record.coordinate,
            background_id=record.background_id,
            local_terms_used=record.local_terms_used,
        )
        for record in history
        if record.coordinate is not None
        and record.predicted_local_improvement is not None
        and record.decision in {"accepted", "rejected", "inconclusive", "not_selected"}
    )
    return NonOracleExperimentResult(
        mode=mode,
        final_intervention=current,
        final_graph=graph,
        structure_estimate=structure,
        final_global_loss=current_loss,
        final_global_interval=interval,
        termination_reason=termination,
        history=tuple(history),
        direct_orientation_evidence=tuple(direct_evidence),
        meek_orientation_evidence=tuple(meek_evidence),
        query_accounting=ledger.snapshot(),
        baseline_background_ids=tuple(
            baseline.background_id for baseline in baselines.history
        ),
        accepted_proposals=accepted,
        rejected_proposals=rejected,
        inconclusive_proposals=inconclusive,
        optimizer_search_evaluations=search_evaluations,
        final_local_losses=final_local_losses,
        surrogate_proposal_diagnostics=diagnostics,
        surrogate_diagnostic_summary=summarize_surrogate_diagnostics(diagnostics),
        public_action_signature=tuple(
            f"{record.action}:{record.coordinate}:{record.decision}"
            for record in history
        ),
        orientation_conflicts=tuple(orientation_conflicts),
        covariance_diagnostics=covariance_diagnostics,
    )
