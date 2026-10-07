"""Derivative-free public-moment and finite-sample coordinate search."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar

from graphs.interventions import ShiftScaleIntervention
from graphs.pdag import PartiallyDirectedGraph
from graphs.problem_gen import PublicProblemView

from .baselines import AcceptedBackgroundBaseline
from .blackbox import (
    ExactMomentBlackBoxEnvironment,
    ExactMomentQueryResult,
    FiniteSampleBlackBoxEnvironment,
)
from .capabilities import ProviderCapability
from .empirical_moments import EmpiricalMomentsProvider
from .local_losses import (
    CandidateMomentsProvider,
    CompositeGaussianLocalLoss,
    LocalLossWeights,
    PublicPairwiseNeighborhood,
)
from .optimizers import (
    CoordinateSelectionRequiredError,
    FinalSupportBudgetExceededError,
)
from .types import (
    CandidateProposal,
    FinalSupportBudget,
    InterventionBackground,
    ParameterCoordinate,
)

FiniteSearchStatus = Literal["proposal", "rejected", "inconclusive"]
FiniteProposalMode = Literal["confidence_supported", "training_point"]


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
class CoordinateSearchBounds:
    """Mandatory bounded shift and log-scale search domains."""

    shift_min: float
    shift_max: float
    log_scale_min: float
    log_scale_max: float

    def __post_init__(self) -> None:
        for name in (
            "shift_min",
            "shift_max",
            "log_scale_min",
            "log_scale_max",
        ):
            object.__setattr__(self, name, _finite_real(name, getattr(self, name)))
        if self.shift_min >= self.shift_max:
            raise ValueError("shift_min must be smaller than shift_max.")
        if self.log_scale_min >= self.log_scale_max:
            raise ValueError("log_scale_min must be smaller than log_scale_max.")

    def domain(self, coordinate: ParameterCoordinate) -> tuple[float, float]:
        """Return the configured scalar domain for one coordinate."""
        return (
            (self.shift_min, self.shift_max)
            if coordinate.kind == "shift"
            else (self.log_scale_min, self.log_scale_max)
        )


@dataclass(frozen=True)
class ExactCoordinateSearchSettings:
    """Coarse-grid and optional bounded refinement settings."""

    bounds: CoordinateSearchBounds
    coarse_grid_size: int = 9
    refine: bool = True
    refinement_max_evaluations: int = 32
    refinement_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, CoordinateSearchBounds):
            raise TypeError("bounds must be CoordinateSearchBounds.")
        if _positive_integer("coarse_grid_size", self.coarse_grid_size) < 3:
            raise ValueError("coarse_grid_size must be at least three.")
        _positive_integer(
            "refinement_max_evaluations",
            self.refinement_max_evaluations,
        )
        tolerance = _finite_real(
            "refinement_tolerance",
            self.refinement_tolerance,
        )
        if tolerance <= 0.0:
            raise ValueError("refinement_tolerance must be positive.")
        object.__setattr__(self, "refinement_tolerance", tolerance)
        if not isinstance(self.refine, bool):
            raise TypeError("refine must be a boolean.")


@dataclass(frozen=True)
class ExactCoordinateSearchTrace:
    """Auditable exact black-box evaluations from one optimizer call."""

    coordinate: ParameterCoordinate
    tested_values: tuple[float, ...]
    local_losses: tuple[float, ...]
    query_ids: tuple[str, ...]
    selected_value: float
    selected_local_loss: float


@dataclass
class ExactBlackBoxMomentsProvider:
    """Candidate-moment provider backed only by the exact public API."""

    environment: ExactMomentBlackBoxEnvironment
    accepted_background: ShiftScaleIntervention
    _cache: dict[str, ExactMomentQueryResult] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _query_ids: list[str] = field(default_factory=list, init=False, repr=False)
    capabilities = frozenset({ProviderCapability.EXACT_PUBLIC_MOMENTS})

    def __post_init__(self) -> None:
        if not isinstance(self.environment, ExactMomentBlackBoxEnvironment):
            raise TypeError("environment must be ExactMomentBlackBoxEnvironment.")
        if not isinstance(self.accepted_background, ShiftScaleIntervention):
            raise TypeError("accepted_background must be ShiftScaleIntervention.")

    @property
    def background_id(self) -> str:
        """Return the currently accepted background identifier."""
        return InterventionBackground.from_intervention(
            self.accepted_background
        ).identifier

    @property
    def query_ids(self) -> tuple[str, ...]:
        """Return all black-box queries made through this provider."""
        return tuple(self._query_ids)

    def accept_background(self, intervention: ShiftScaleIntervention) -> None:
        """Invalidate proposal caches after an accepted state change."""
        if not isinstance(intervention, ShiftScaleIntervention):
            raise TypeError("intervention must be ShiftScaleIntervention.")
        if (
            InterventionBackground.from_intervention(intervention).identifier
            == self.background_id
        ):
            return
        self.accepted_background = intervention
        self._cache.clear()

    def preload(self, result: ExactMomentQueryResult) -> None:
        """Reuse a public exact baseline or already queried candidate result."""
        if not isinstance(result, ExactMomentQueryResult):
            raise TypeError("result must be ExactMomentQueryResult.")
        self._cache[result.metadata.intervention_id] = result

    def moments(
        self,
        candidate: ShiftScaleIntervention,
        public_problem: PublicProblemView,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cached or newly queried public moments."""
        if not isinstance(public_problem, PublicProblemView):
            raise TypeError("public_problem must be PublicProblemView.")
        identifier = InterventionBackground.from_intervention(candidate).identifier
        result = self._cache.get(identifier)
        if result is None:
            result = self.environment.query(
                candidate,
                kind="candidate",
                background_id=self.background_id,
            )
            self._cache[identifier] = result
            self._query_ids.append(result.metadata.query_id)
        return result.mean, result.covariance


@dataclass
class ExactPublicMomentCoordinateOptimizer:
    """One-coordinate optimizer using only charged public moment queries."""

    moments_provider: ExactBlackBoxMomentsProvider
    local_loss: CompositeGaussianLocalLoss
    graph: PartiallyDirectedGraph
    settings: ExactCoordinateSearchSettings
    last_trace: ExactCoordinateSearchTrace | None = field(
        default=None,
        init=False,
    )
    capabilities = frozenset({ProviderCapability.EXACT_PUBLIC_MOMENTS})

    def __post_init__(self) -> None:
        if not isinstance(self.moments_provider, ExactBlackBoxMomentsProvider):
            raise TypeError("moments_provider has the wrong type.")
        if not isinstance(self.local_loss, CompositeGaussianLocalLoss):
            raise TypeError("local_loss must be CompositeGaussianLocalLoss.")
        if self.local_loss.moments_provider is not self.moments_provider:
            raise ValueError("local_loss must share the optimizer moments provider.")
        if not isinstance(self.graph, PartiallyDirectedGraph):
            raise TypeError("graph must be PartiallyDirectedGraph.")
        if not isinstance(self.settings, ExactCoordinateSearchSettings):
            raise TypeError("settings must be ExactCoordinateSearchSettings.")

    def optimize(
        self,
        current_candidate: ShiftScaleIntervention,
        active_coordinates: tuple[ParameterCoordinate, ...],
        problem_view: PublicProblemView,
        final_support_budget: FinalSupportBudget,
    ) -> CandidateProposal:
        """Evaluate a bounded grid and optional local bounded refinement."""
        if len(active_coordinates) != 1:
            raise CoordinateSelectionRequiredError(
                "exact public search requires exactly one active coordinate."
            )
        coordinate = active_coordinates[0]
        if coordinate.node >= problem_view.n_nodes:
            raise ValueError("coordinate node must fit the public problem.")
        lower, upper = self.settings.bounds.domain(coordinate)
        coarse_values = np.linspace(
            lower,
            upper,
            self.settings.coarse_grid_size,
        )
        current_value = _coordinate_value(current_candidate, coordinate)
        values = sorted(
            {
                *map(float, coarse_values),
                float(np.clip(current_value, lower, upper)),
            }
        )
        tested_values: list[float] = []
        losses: list[float] = []
        query_start = len(self.moments_provider.query_ids)

        def objective(value: float) -> float:
            candidate = _with_coordinate(current_candidate, coordinate, value)
            if not final_support_budget.allows(candidate):
                return float("inf")
            try:
                loss = self.local_loss.coordinate_loss(
                    coordinate,
                    problem_view,
                    candidate,
                    self.graph,
                )
            except (
                FloatingPointError,
                np.linalg.LinAlgError,
                RuntimeError,
                ValueError,
            ):
                loss = float("inf")
            tested_values.append(float(value))
            losses.append(float(loss))
            return float(loss)

        for value in values:
            objective(value)
        finite_positions = [
            index for index, loss in enumerate(losses) if np.isfinite(loss)
        ]
        if not finite_positions:
            raise FinalSupportBudgetExceededError(
                "no exact-search grid point fits the support budget."
            )
        best_position = min(finite_positions, key=lambda index: losses[index])
        best_value = tested_values[best_position]
        best_loss = losses[best_position]

        if self.settings.refine:
            ordered = sorted(set(tested_values))
            best_order = ordered.index(best_value)
            refinement_lower = ordered[max(0, best_order - 1)]
            refinement_upper = ordered[min(len(ordered) - 1, best_order + 1)]
            if refinement_lower < refinement_upper:
                result = minimize_scalar(
                    objective,
                    bounds=(refinement_lower, refinement_upper),
                    method="bounded",
                    options={
                        "maxiter": self.settings.refinement_max_evaluations,
                        "xatol": self.settings.refinement_tolerance,
                    },
                )
                if result.success and float(result.fun) < best_loss:
                    best_value = float(result.x)
                    best_loss = float(result.fun)

        selected = _with_coordinate(current_candidate, coordinate, best_value)
        query_ids = self.moments_provider.query_ids[query_start:]
        self.last_trace = ExactCoordinateSearchTrace(
            coordinate=coordinate,
            tested_values=tuple(tested_values),
            local_losses=tuple(losses),
            query_ids=query_ids,
            selected_value=best_value,
            selected_local_loss=best_loss,
        )
        return CandidateProposal(selected, active_coordinates, best_loss)


@dataclass(frozen=True)
class FiniteCoordinateSearchSettings:
    """Discrete noisy search and sequential-allocation limits."""

    bounds: CoordinateSearchBounds
    coarse_grid_size: int
    initial_sample_size: int
    max_sample_size_per_candidate: int
    max_query_batches: int
    sample_growth_factor: float
    bootstrap_replicates: int
    confidence_level: float
    improvement_tolerance: float
    proposal_mode: FiniteProposalMode = "confidence_supported"

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, CoordinateSearchBounds):
            raise TypeError("bounds must be CoordinateSearchBounds.")
        if _positive_integer("coarse_grid_size", self.coarse_grid_size) < 3:
            raise ValueError("coarse_grid_size must be at least three.")
        for name in (
            "initial_sample_size",
            "max_sample_size_per_candidate",
            "max_query_batches",
            "bootstrap_replicates",
        ):
            _positive_integer(name, getattr(self, name))
        if self.initial_sample_size > self.max_sample_size_per_candidate:
            raise ValueError("initial_sample_size exceeds per-candidate cap.")
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
        if self.proposal_mode not in {"confidence_supported", "training_point"}:
            raise ValueError(
                "proposal_mode must be 'confidence_supported' or 'training_point'."
            )


@dataclass(frozen=True)
class FiniteCandidateEvidence:
    """Samples and local-improvement uncertainty for one scalar value."""

    value: float
    intervention: ShiftScaleIntervention
    point_local_loss: float
    point_improvement: float
    improvement_interval: tuple[float, float]
    sample_size: int
    query_ids: tuple[str, ...]
    covariance_metadata: dict[str, object]


@dataclass(frozen=True)
class FiniteCoordinateSearchResult:
    """Three-way outcome from noisy one-coordinate search."""

    status: FiniteSearchStatus
    coordinate: ParameterCoordinate
    proposal: CandidateProposal | None
    evidence: tuple[FiniteCandidateEvidence, ...]
    selected_value: float | None
    query_batches: int
    observations: int
    bootstrap_seed: int
    reason: str


@dataclass
class FiniteSampleCoordinateSearch:
    """Discrete public-sample search with sequential allocation."""

    environment: FiniteSampleBlackBoxEnvironment
    moments_provider: EmpiricalMomentsProvider
    weights: LocalLossWeights
    neighborhood: PublicPairwiseNeighborhood
    graph: PartiallyDirectedGraph
    settings: FiniteCoordinateSearchSettings
    capabilities = frozenset({ProviderCapability.EMPIRICAL_SAMPLES})

    def search(
        self,
        current_candidate: ShiftScaleIntervention,
        coordinate: ParameterCoordinate,
        problem_view: PublicProblemView,
        baseline: AcceptedBackgroundBaseline,
        final_support_budget: FinalSupportBudget,
        *,
        search_seed: int,
        bootstrap_seed: int,
    ) -> FiniteCoordinateSearchResult:
        """Search a coarse grid and repeat only statistically promising values."""
        if baseline.samples is None:
            raise ValueError("finite search requires a sampled baseline.")
        if (
            baseline.background_id
            != InterventionBackground.from_intervention(current_candidate).identifier
        ):
            raise ValueError("baseline does not match current accepted background.")
        lower, upper = self.settings.bounds.domain(coordinate)
        values = tuple(
            float(value)
            for value in np.linspace(
                lower,
                upper,
                self.settings.coarse_grid_size,
            )
        )
        rng = np.random.default_rng(search_seed)
        allocations: dict[float, list[np.ndarray]] = {}
        query_ids: dict[float, list[str]] = {}
        query_batches = 0

        for value in values:
            candidate = _with_coordinate(current_candidate, coordinate, value)
            if not final_support_budget.allows(candidate):
                continue
            if query_batches >= self.settings.max_query_batches:
                break
            result = self.environment.query(
                candidate,
                sample_size=self.settings.initial_sample_size,
                query_seed=int(rng.integers(0, np.iinfo(np.int64).max)),
                kind="candidate",
                background_id=baseline.background_id,
            )
            allocations[value] = [result.samples]
            query_ids[value] = [result.metadata.query_id]
            query_batches += 1

        if not allocations:
            return FiniteCoordinateSearchResult(
                status="rejected",
                coordinate=coordinate,
                proposal=None,
                evidence=(),
                selected_value=None,
                query_batches=0,
                observations=0,
                bootstrap_seed=bootstrap_seed,
                reason="no_legal_grid_value",
            )

        evidence = self._evaluate_allocations(
            allocations,
            query_ids,
            current_candidate,
            coordinate,
            problem_view,
            baseline.samples,
            bootstrap_seed,
        )
        while query_batches < self.settings.max_query_batches:
            supported = [
                item
                for item in evidence
                if item.improvement_interval[0] > self.settings.improvement_tolerance
            ]
            if supported:
                break
            uncertain = [
                item
                for item in evidence
                if item.improvement_interval[1] > self.settings.improvement_tolerance
                and item.sample_size < self.settings.max_sample_size_per_candidate
            ]
            if not uncertain:
                break
            promising = max(
                uncertain,
                key=lambda item: (item.point_improvement, -item.value),
            )
            remaining = (
                self.settings.max_sample_size_per_candidate - promising.sample_size
            )
            requested = min(
                remaining,
                max(
                    2,
                    int(
                        np.ceil(
                            promising.sample_size
                            * (self.settings.sample_growth_factor - 1.0)
                        )
                    ),
                ),
            )
            candidate = promising.intervention
            result = self.environment.query(
                candidate,
                sample_size=requested,
                query_seed=int(rng.integers(0, np.iinfo(np.int64).max)),
                kind="repeat",
                budget_category="candidate",
                background_id=baseline.background_id,
            )
            allocations[promising.value].append(result.samples)
            query_ids[promising.value].append(result.metadata.query_id)
            query_batches += 1
            evidence = self._evaluate_allocations(
                allocations,
                query_ids,
                current_candidate,
                coordinate,
                problem_view,
                baseline.samples,
                bootstrap_seed,
            )

        if self.settings.proposal_mode == "training_point":
            selected = max(
                evidence,
                key=lambda item: (item.point_improvement, -item.value),
            )
            proposal = CandidateProposal(
                selected.intervention,
                (coordinate,),
                selected.point_local_loss,
            )
            return FiniteCoordinateSearchResult(
                status="proposal",
                coordinate=coordinate,
                proposal=proposal,
                evidence=evidence,
                selected_value=selected.value,
                query_batches=query_batches,
                observations=sum(
                    samples.shape[0]
                    for chunks in allocations.values()
                    for samples in chunks
                ),
                bootstrap_seed=bootstrap_seed,
                reason="training_point_best_bounded_value",
            )

        supported = [
            item
            for item in evidence
            if item.improvement_interval[0] > self.settings.improvement_tolerance
        ]
        uncertain = [
            item
            for item in evidence
            if item.improvement_interval[0]
            <= self.settings.improvement_tolerance
            < item.improvement_interval[1]
        ]
        if supported:
            selected = max(
                supported,
                key=lambda item: (item.point_improvement, -item.value),
            )
            proposal = CandidateProposal(
                selected.intervention,
                (coordinate,),
                selected.point_local_loss,
            )
            status: FiniteSearchStatus = "proposal"
            reason = "confidence_supported_local_improvement"
        elif uncertain:
            selected = max(
                uncertain,
                key=lambda item: (item.point_improvement, -item.value),
            )
            proposal = None
            status = "inconclusive"
            reason = "sequential_allocation_cap_reached"
        else:
            selected = None
            proposal = None
            status = "rejected"
            reason = "no_confidence_supported_improvement"
        return FiniteCoordinateSearchResult(
            status=status,
            coordinate=coordinate,
            proposal=proposal,
            evidence=evidence,
            selected_value=None if selected is None else selected.value,
            query_batches=query_batches,
            observations=sum(
                samples.shape[0]
                for chunks in allocations.values()
                for samples in chunks
            ),
            bootstrap_seed=bootstrap_seed,
            reason=reason,
        )

    def _evaluate_allocations(
        self,
        allocations: dict[float, list[np.ndarray]],
        query_ids: dict[float, list[str]],
        current: ShiftScaleIntervention,
        coordinate: ParameterCoordinate,
        problem: PublicProblemView,
        baseline_samples: np.ndarray,
        bootstrap_seed: int,
    ) -> tuple[FiniteCandidateEvidence, ...]:
        baseline_loss = _local_loss_from_samples(
            baseline_samples,
            current,
            coordinate,
            problem,
            self.graph,
            self.moments_provider,
            self.weights,
            self.neighborhood,
        )[0]
        results: list[FiniteCandidateEvidence] = []
        for position, value in enumerate(sorted(allocations)):
            samples = np.concatenate(allocations[value], axis=0)
            candidate = _with_coordinate(current, coordinate, value)
            candidate_loss, metadata = _local_loss_from_samples(
                samples,
                candidate,
                coordinate,
                problem,
                self.graph,
                self.moments_provider,
                self.weights,
                self.neighborhood,
            )
            interval = _bootstrap_local_improvement(
                baseline_samples,
                samples,
                current,
                candidate,
                coordinate,
                problem,
                self.graph,
                self.moments_provider,
                self.weights,
                self.neighborhood,
                replicates=self.settings.bootstrap_replicates,
                confidence_level=self.settings.confidence_level,
                seed=bootstrap_seed + position,
            )
            results.append(
                FiniteCandidateEvidence(
                    value=value,
                    intervention=candidate,
                    point_local_loss=candidate_loss,
                    point_improvement=baseline_loss - candidate_loss,
                    improvement_interval=interval,
                    sample_size=samples.shape[0],
                    query_ids=tuple(query_ids[value]),
                    covariance_metadata=metadata,
                )
            )
        return tuple(results)


@dataclass(frozen=True)
class _FixedMomentsProvider:
    mean: np.ndarray
    covariance: np.ndarray

    def moments(
        self,
        candidate: ShiftScaleIntervention,
        public_problem: PublicProblemView,
    ) -> tuple[np.ndarray, np.ndarray]:
        del candidate, public_problem
        return self.mean, self.covariance


def _local_loss_from_samples(
    samples: np.ndarray,
    intervention: ShiftScaleIntervention,
    coordinate: ParameterCoordinate,
    problem: PublicProblemView,
    graph: PartiallyDirectedGraph,
    moments_provider: EmpiricalMomentsProvider,
    weights: LocalLossWeights,
    neighborhood: PublicPairwiseNeighborhood,
) -> tuple[float, dict[str, object]]:
    estimate = moments_provider.estimate(samples)
    provider: CandidateMomentsProvider = _FixedMomentsProvider(
        estimate.mean,
        estimate.regularized_covariance,
    )
    loss = CompositeGaussianLocalLoss(
        provider,
        weights,
        neighborhood,
    ).coordinate_loss(coordinate, problem, intervention, graph)
    return loss, estimate.metadata()


def _bootstrap_local_improvement(
    baseline_samples: np.ndarray,
    candidate_samples: np.ndarray,
    current: ShiftScaleIntervention,
    candidate: ShiftScaleIntervention,
    coordinate: ParameterCoordinate,
    problem: PublicProblemView,
    graph: PartiallyDirectedGraph,
    moments_provider: EmpiricalMomentsProvider,
    weights: LocalLossWeights,
    neighborhood: PublicPairwiseNeighborhood,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    improvements = np.empty(replicates)
    for replicate in range(replicates):
        baseline_indices = rng.integers(
            0,
            baseline_samples.shape[0],
            baseline_samples.shape[0],
        )
        candidate_indices = rng.integers(
            0,
            candidate_samples.shape[0],
            candidate_samples.shape[0],
        )
        baseline_loss = _local_loss_from_samples(
            baseline_samples[baseline_indices],
            current,
            coordinate,
            problem,
            graph,
            moments_provider,
            weights,
            neighborhood,
        )[0]
        candidate_loss = _local_loss_from_samples(
            candidate_samples[candidate_indices],
            candidate,
            coordinate,
            problem,
            graph,
            moments_provider,
            weights,
            neighborhood,
        )[0]
        improvements[replicate] = baseline_loss - candidate_loss
    tail = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(improvements, tail)),
        float(np.quantile(improvements, 1.0 - tail)),
    )


def _coordinate_value(
    intervention: ShiftScaleIntervention,
    coordinate: ParameterCoordinate,
) -> float:
    return (
        float(intervention.shift[coordinate.node])
        if coordinate.kind == "shift"
        else float(np.log(intervention.scale[coordinate.node]))
    )


def _with_coordinate(
    intervention: ShiftScaleIntervention,
    coordinate: ParameterCoordinate,
    value: float,
) -> ShiftScaleIntervention:
    shift = intervention.shift.copy()
    scale = intervention.scale.copy()
    if coordinate.kind == "shift":
        shift[coordinate.node] = value
    else:
        scale[coordinate.node] = np.exp(value)
    return ShiftScaleIntervention(shift, scale)
