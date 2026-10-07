"""Dependency-injected alternating exploration and matching scheduler."""

from dataclasses import dataclass
from numbers import Real
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from graphs._immutability import immutable_array
from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import PublicProblemView
from graphs.support import (
    DEFAULT_SUPPORT_ATOL,
    coordinate_support_masks,
    validate_support_atol,
)

from .losses import InterventionLoss
from .optimizers import (
    CoordinateSelectionRequiredError,
    FinalSupportBudgetExceededError,
    InterventionOptimizer,
)
from .probe_outcomes import ProbeOutcome
from .types import (
    AlgorithmStepRecord,
    AtomicCoordinateCost,
    CandidateProposal,
    ExperimentBudgetLedger,
    FinalSupportBudget,
    InterventionBackground,
    ParameterCoordinate,
    ParameterKind,
    ShiftProbe,
)

TerminationReason = Literal[
    "loss_tolerance_reached",
    "no_unresolved_variables",
    "global_mismatch_not_localized",
    "experiment_budget_exhausted",
    "final_support_budget_exhausted",
    "coordinate_selection_required",
    "no_legal_action",
    "patience_exhausted",
    "max_consecutive_exploration_trials_reached",
    "iteration_limit_reached",
    "numerical_failure",
]


def _finite_real(name: str, value: object) -> float:
    """Validate and normalize one finite real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _positive_integer(name: str, value: object) -> int:
    """Validate one positive integer scheduler setting."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


@dataclass(frozen=True)
class SchedulerSettings:
    """Numerical tolerances and caps for one scheduler run."""

    loss_tolerance: float
    improvement_tolerance: float
    patience: int
    max_consecutive_exploration_trials: int
    max_probe_targets: int
    scale_min: float
    scale_max: float
    support_atol: float = DEFAULT_SUPPORT_ATOL
    max_steps: int = 1_000
    global_loss_tolerance: float | None = None
    allowed_coordinate_kinds: tuple[ParameterKind, ...] = ("shift", "log_scale")

    def __post_init__(self) -> None:
        """Validate and normalize every scheduler setting."""
        loss_tolerance = _finite_real("loss_tolerance", self.loss_tolerance)
        improvement_tolerance = _finite_real(
            "improvement_tolerance",
            self.improvement_tolerance,
        )
        if loss_tolerance < 0.0:
            raise ValueError("loss_tolerance must be nonnegative.")
        if improvement_tolerance < 0.0:
            raise ValueError("improvement_tolerance must be nonnegative.")
        global_loss_tolerance = (
            loss_tolerance
            if self.global_loss_tolerance is None
            else _finite_real(
                "global_loss_tolerance",
                self.global_loss_tolerance,
            )
        )
        if global_loss_tolerance < 0.0:
            raise ValueError("global_loss_tolerance must be nonnegative.")
        scale_min = _finite_real("scale_min", self.scale_min)
        scale_max = _finite_real("scale_max", self.scale_max)
        if not 0.0 < scale_min <= scale_max:
            raise ValueError("scale bounds must satisfy 0 < scale_min <= scale_max.")

        allowed = self.allowed_coordinate_kinds
        if (
            not isinstance(allowed, tuple)
            or not allowed
            or len(set(allowed)) != len(allowed)
            or any(kind not in {"shift", "log_scale"} for kind in allowed)
        ):
            raise ValueError(
                "allowed_coordinate_kinds must be a unique nonempty tuple of "
                "shift and/or log_scale."
            )
        object.__setattr__(self, "loss_tolerance", loss_tolerance)
        object.__setattr__(
            self,
            "improvement_tolerance",
            improvement_tolerance,
        )
        object.__setattr__(
            self,
            "global_loss_tolerance",
            global_loss_tolerance,
        )
        object.__setattr__(
            self,
            "patience",
            _positive_integer("patience", self.patience),
        )
        object.__setattr__(
            self,
            "max_consecutive_exploration_trials",
            _positive_integer(
                "max_consecutive_exploration_trials",
                self.max_consecutive_exploration_trials,
            ),
        )
        object.__setattr__(
            self,
            "max_probe_targets",
            _positive_integer("max_probe_targets", self.max_probe_targets),
        )
        object.__setattr__(self, "scale_min", scale_min)
        object.__setattr__(self, "scale_max", scale_max)
        object.__setattr__(
            self,
            "support_atol",
            validate_support_atol(self.support_atol),
        )
        object.__setattr__(
            self,
            "max_steps",
            _positive_integer("max_steps", self.max_steps),
        )


@dataclass(frozen=True)
class AlgorithmStateView:
    """Public scheduler state passed to knowledge and exploration policies."""

    problem_view: PublicProblemView
    current_candidate: ShiftScaleIntervention
    current_loss: float
    step: int
    consecutive_rejections: int

    def __post_init__(self) -> None:
        """Validate one public-only state snapshot."""
        if not isinstance(self.problem_view, PublicProblemView):
            raise TypeError("problem_view must be a PublicProblemView.")
        if not isinstance(self.current_candidate, ShiftScaleIntervention):
            raise TypeError("current_candidate must be a ShiftScaleIntervention.")
        if self.current_candidate.shift.shape != (self.problem_view.n_nodes,):
            raise ValueError(
                "current candidate dimension must agree with the problem view."
            )
        object.__setattr__(
            self,
            "current_loss",
            _finite_real("current_loss", self.current_loss),
        )
        for name in ("step", "consecutive_rejections"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")


@dataclass(frozen=True)
class BudgetView:
    """Read-only budget summary passed to exploration policies."""

    remaining_experiment_trials: int
    remaining_final_support: int
    max_experiment_trials: int
    max_parameter_coordinates: int

    def __post_init__(self) -> None:
        """Validate nonnegative public budget counts."""
        for name in (
            "remaining_experiment_trials",
            "remaining_final_support",
            "max_experiment_trials",
            "max_parameter_coordinates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")


@dataclass(frozen=True)
class ExplorationObservation:
    """Public evidence produced by one temporary pure-shift probe.

    Statistical environments may return ``samples``. The exact graph-level
    simulation environment instead returns a public ``ProbeOutcome`` whose
    hidden DAG has already been discarded at the environment boundary.
    """

    proposal: ShiftProbe
    intervention: ShiftScaleIntervention
    samples: np.ndarray | None
    outcome: ProbeOutcome | None = None
    background: InterventionBackground | None = None

    def __post_init__(self) -> None:
        """Validate and freeze the public probe observation."""
        if not isinstance(self.proposal, ShiftProbe):
            raise TypeError("proposal must be a ShiftProbe.")
        if not isinstance(self.intervention, ShiftScaleIntervention):
            raise TypeError("intervention must be a ShiftScaleIntervention.")
        accepted_shift = self.intervention.shift - self.proposal.shift_increment
        accepted = ShiftScaleIntervention(accepted_shift, self.intervention.scale)
        expected_background = InterventionBackground.from_intervention(accepted)
        if self.background is None:
            object.__setattr__(self, "background", expected_background)
        elif not isinstance(self.background, InterventionBackground):
            raise TypeError("background must be an InterventionBackground or None.")
        elif self.background.identifier != expected_background.identifier:
            raise ValueError(
                "background must identify the accepted state beneath the probe."
            )
        if self.outcome is not None:
            if not isinstance(self.outcome, ProbeOutcome):
                raise TypeError("outcome must be a ProbeOutcome or None.")
            if self.outcome.targets != self.proposal.targets:
                raise ValueError("outcome targets must match the probe proposal.")
            if self.outcome.samples is None and self.samples is not None:
                raise ValueError(
                    "samples must be None when the probe outcome has no samples."
                )
            if self.outcome.samples is not None and (
                self.samples is None
                or not np.array_equal(self.samples, self.outcome.samples)
            ):
                raise ValueError("samples must match the samples in the probe outcome.")
        if self.samples is None:
            if self.outcome is None:
                raise ValueError("an exploration observation requires public evidence.")
            return
        if not isinstance(self.samples, np.ndarray):
            raise TypeError("samples must be a numpy.ndarray or None.")
        if self.samples.ndim != 2:
            raise ValueError("samples must be two-dimensional.")
        if self.samples.shape[1] != self.intervention.shift.size:
            raise ValueError("sample dimension must agree with the probe intervention.")
        if not np.issubdtype(self.samples.dtype, np.number) or not np.isrealobj(
            self.samples
        ):
            raise TypeError("samples must contain real numeric values.")
        if not np.all(np.isfinite(self.samples)):
            raise ValueError("samples must contain only finite values.")
        object.__setattr__(
            self,
            "samples",
            immutable_array(self.samples, dtype=float),
        )


@dataclass(frozen=True)
class CandidateEvaluation:
    """A candidate loss with empirical moments and uncertainty metadata."""

    loss: float
    mean: np.ndarray | None = None
    covariance: np.ndarray | None = None
    samples: np.ndarray | None = None
    sample_count: int = 0
    covariance_regularization: float = 0.0
    loss_standard_error: float | None = None

    def __post_init__(self) -> None:
        """Validate scalar metadata and defensively freeze empirical arrays."""
        object.__setattr__(self, "loss", _finite_real("loss", self.loss))
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count, int
        ):
            raise TypeError("sample_count must be an integer.")
        if self.sample_count < 0:
            raise ValueError("sample_count must be nonnegative.")
        regularization = _finite_real(
            "covariance_regularization",
            self.covariance_regularization,
        )
        if regularization < 0.0:
            raise ValueError("covariance_regularization must be nonnegative.")
        object.__setattr__(self, "covariance_regularization", regularization)
        if self.loss_standard_error is not None:
            standard_error = _finite_real(
                "loss_standard_error",
                self.loss_standard_error,
            )
            if standard_error < 0.0:
                raise ValueError("loss_standard_error must be nonnegative.")
            object.__setattr__(self, "loss_standard_error", standard_error)
        for name in ("mean", "covariance", "samples"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray or None.")
            if not np.issubdtype(value.dtype, np.number) or not np.isrealobj(value):
                raise TypeError(f"{name} must contain real numeric values.")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values.")
            object.__setattr__(self, name, immutable_array(value, dtype=float))
        if self.mean is not None and self.mean.ndim != 1:
            raise ValueError("mean must be one-dimensional.")
        if self.covariance is not None and (
            self.covariance.ndim != 2
            or self.covariance.shape != (self.covariance.shape[0],) * 2
        ):
            raise ValueError("covariance must be square.")
        if self.samples is not None and self.samples.ndim != 2:
            raise ValueError("samples must be two-dimensional.")
        if self.samples is not None and self.samples.shape[0] != self.sample_count:
            raise ValueError("sample_count must equal the number of sample rows.")


@runtime_checkable
class MatchingKnowledge(Protocol):
    """Public graph-knowledge interface consumed by the scheduler."""

    def unresolved_variables_remain(self, state: AlgorithmStateView) -> bool:
        """Return whether any matching variables remain unresolved."""
        ...

    def actionable_coordinates(
        self,
        state: AlgorithmStateView,
    ) -> tuple[ParameterCoordinate, ...]:
        """Return the complete currently legal coordinate set."""
        ...

    def update_from_probe(
        self,
        observation: ExplorationObservation,
        state: AlgorithmStateView,
    ) -> None:
        """Update knowledge from one public probe observation."""
        ...


@runtime_checkable
class LocalProposalLossProvider(Protocol):
    """Optional knowledge capability for local coordinate ranking."""

    def proposal_local_loss(
        self,
        coordinate: ParameterCoordinate,
        candidate: ShiftScaleIntervention,
        state: AlgorithmStateView,
    ) -> float:
        """Return the coordinate's composite local loss under ``candidate``."""
        ...


@runtime_checkable
class AcceptedBackgroundListener(Protocol):
    """Optional hook for invalidating numerical state after acceptance."""

    def accepted_background_changed(
        self,
        previous: ShiftScaleIntervention,
        current: ShiftScaleIntervention,
    ) -> None:
        """Invalidate background-dependent state while retaining graph facts."""
        ...


@runtime_checkable
class ExplorationKnowledgeProvider(Protocol):
    """Optionally expose a policy-specific immutable knowledge snapshot."""

    def exploration_knowledge(self, state: AlgorithmStateView) -> object:
        """Return the public snapshot to pass to an exploration policy."""
        ...


@runtime_checkable
class ExplorationPolicy(Protocol):
    """Policy for temporary pure-shift experiments."""

    def propose_probe(
        self,
        public_state: AlgorithmStateView,
        knowledge: MatchingKnowledge,
        budget: BudgetView,
        rng: np.random.Generator,
    ) -> ShiftProbe | None:
        """Return a legal probe proposal, or ``None`` when none is available."""
        ...


@runtime_checkable
class MatchingEnvironment(Protocol):
    """Execution and permitted candidate-evaluation interface."""

    candidate_evaluation_requires_experiment: bool

    def execute_probe(
        self,
        current_candidate: ShiftScaleIntervention,
        proposal: ShiftProbe,
        problem_view: PublicProblemView,
        rng: np.random.Generator,
    ) -> np.ndarray | ProbeOutcome:
        """Execute a temporary probe and return only public evidence."""
        ...

    def evaluate_candidate(
        self,
        candidate: ShiftScaleIntervention,
        problem_view: PublicProblemView,
        loss: InterventionLoss,
        rng: np.random.Generator,
    ) -> CandidateEvaluation | float:
        """Evaluate a proposed candidate through the permitted interface."""
        ...


@dataclass(frozen=True)
class SchedulerResult:
    """Final state, immutable history, and explicit termination reason."""

    final_candidate: ShiftScaleIntervention
    final_loss: float
    termination_reason: TerminationReason
    history: tuple[AlgorithmStepRecord, ...]
    budget_ledger: ExperimentBudgetLedger

    def __post_init__(self) -> None:
        """Validate the returned scheduler summary."""
        if not isinstance(self.final_candidate, ShiftScaleIntervention):
            raise TypeError("final_candidate must be a ShiftScaleIntervention.")
        object.__setattr__(
            self,
            "final_loss",
            _finite_real("final_loss", self.final_loss),
        )
        if not isinstance(self.history, tuple) or any(
            not isinstance(record, AlgorithmStepRecord) for record in self.history
        ):
            raise TypeError("history must be a tuple of AlgorithmStepRecord objects.")
        if not isinstance(self.budget_ledger, ExperimentBudgetLedger):
            raise TypeError("budget_ledger must be an ExperimentBudgetLedger.")


@dataclass(frozen=True)
class AlternatingMatchingScheduler:
    """Alternate source-triggered exploration with proposal evaluation."""

    knowledge: MatchingKnowledge
    exploration_policy: ExplorationPolicy
    loss: InterventionLoss
    optimizer: InterventionOptimizer
    environment: MatchingEnvironment
    final_support_budget: FinalSupportBudget
    settings: SchedulerSettings

    def __post_init__(self) -> None:
        """Validate injected interfaces and immutable scheduler settings."""
        interface_values = {
            "knowledge": (self.knowledge, MatchingKnowledge),
            "exploration_policy": (
                self.exploration_policy,
                ExplorationPolicy,
            ),
            "loss": (self.loss, InterventionLoss),
            "optimizer": (self.optimizer, InterventionOptimizer),
            "environment": (self.environment, MatchingEnvironment),
        }
        for name, (value, interface) in interface_values.items():
            if not isinstance(value, interface):
                raise TypeError(f"{name} does not implement its required protocol.")
        if not isinstance(self.final_support_budget, FinalSupportBudget):
            raise TypeError("final_support_budget must be a FinalSupportBudget.")
        if not isinstance(self.settings, SchedulerSettings):
            raise TypeError("settings must be SchedulerSettings.")
        if not isinstance(
            self.environment.candidate_evaluation_requires_experiment,
            bool,
        ):
            raise TypeError(
                "candidate_evaluation_requires_experiment must be a boolean."
            )

    def run(
        self,
        problem_view: PublicProblemView,
        budget_ledger: ExperimentBudgetLedger,
    ) -> SchedulerResult:
        """Run until one explicit scheduler termination condition is met."""
        if not isinstance(problem_view, PublicProblemView):
            raise TypeError("problem_view must be a PublicProblemView.")
        if not isinstance(budget_ledger, ExperimentBudgetLedger):
            raise TypeError("budget_ledger must be an ExperimentBudgetLedger.")

        current_candidate = problem_view.current_intervention
        self._validate_scale_bounds(current_candidate)
        current_loss = self._validated_loss(
            self.loss.value(current_candidate, problem_view)
        )
        history: list[AlgorithmStepRecord] = []
        rejected_coordinates: set[ParameterCoordinate] = set()
        if not self.final_support_budget.allows(
            current_candidate,
            self.settings.support_atol,
        ):
            return self._finish(
                current_candidate,
                current_loss,
                "final_support_budget_exhausted",
                history,
                budget_ledger,
            )
        consecutive_rejections = 0
        exploration_rng = np.random.default_rng(
            problem_view.algorithm_seeds.exploration_seed
        )
        evaluation_rng = np.random.default_rng(
            problem_view.algorithm_seeds.candidate_evaluation_seed
        )

        while True:
            if len(history) >= self.settings.max_steps:
                return self._finish(
                    current_candidate,
                    current_loss,
                    "iteration_limit_reached",
                    history,
                    budget_ledger,
                )
            if current_loss <= self.settings.global_loss_tolerance:
                return self._finish(
                    current_candidate,
                    current_loss,
                    "loss_tolerance_reached",
                    history,
                    budget_ledger,
                )

            state = self._state(
                problem_view,
                current_candidate,
                current_loss,
                history,
                consecutive_rejections,
            )
            unresolved = self.knowledge.unresolved_variables_remain(state)
            if not isinstance(unresolved, bool):
                raise TypeError("unresolved_variables_remain must return a boolean.")
            if not unresolved:
                return self._finish(
                    current_candidate,
                    current_loss,
                    "global_mismatch_not_localized",
                    history,
                    budget_ledger,
                )
            active_coordinates = tuple(
                coordinate
                for coordinate in self._actionable_coordinates(state)
                if coordinate not in rejected_coordinates
            )

            if not active_coordinates:
                termination_reason = self._explore_until_actionable(
                    problem_view,
                    current_candidate,
                    current_loss,
                    consecutive_rejections,
                    history,
                    budget_ledger,
                    exploration_rng,
                    frozenset(rejected_coordinates),
                )
                if termination_reason is not None:
                    return self._finish(
                        current_candidate,
                        current_loss,
                        termination_reason,
                        history,
                        budget_ledger,
                    )
                continue

            legal_coordinates = self._legal_actionable_coordinates(
                current_candidate,
                active_coordinates,
            )
            if not legal_coordinates:
                self._record_failure(
                    "OPTIMIZE",
                    "no_coordinate_fits_final_support",
                    "final_support_budget_exhausted",
                    current_candidate,
                    current_loss,
                    history,
                    budget_ledger,
                )
                return self._finish(
                    current_candidate,
                    current_loss,
                    "final_support_budget_exhausted",
                    history,
                    budget_ledger,
                )

            loss_before = current_loss
            try:
                proposal, selected_coordinate, local_loss_before, local_loss_after = (
                    self._best_coordinate_proposal(
                        current_candidate,
                        legal_coordinates,
                        problem_view,
                        state,
                    )
                )
            except CoordinateSelectionRequiredError:
                self._record_failure(
                    "OPTIMIZE",
                    "coordinate_selection_failed",
                    "coordinate_selection_required",
                    current_candidate,
                    current_loss,
                    history,
                    budget_ledger,
                )
                return self._finish(
                    current_candidate,
                    current_loss,
                    "coordinate_selection_required",
                    history,
                    budget_ledger,
                )
            except FinalSupportBudgetExceededError:
                self._record_failure(
                    "OPTIMIZE",
                    "support_budget_exceeded",
                    "final_support_budget_exhausted",
                    current_candidate,
                    current_loss,
                    history,
                    budget_ledger,
                )
                return self._finish(
                    current_candidate,
                    current_loss,
                    "final_support_budget_exhausted",
                    history,
                    budget_ledger,
                )
            except (
                FloatingPointError,
                np.linalg.LinAlgError,
                RuntimeError,
                ValueError,
            ):
                self._record_failure(
                    "OPTIMIZE",
                    "optimization_failed",
                    "numerical_failure",
                    current_candidate,
                    current_loss,
                    history,
                    budget_ledger,
                )
                return self._finish(
                    current_candidate,
                    current_loss,
                    "numerical_failure",
                    history,
                    budget_ledger,
                )

            if not self.final_support_budget.allows(
                proposal.intervention,
                self.settings.support_atol,
            ):
                self._record_failure(
                    "OPTIMIZE",
                    "support_budget_exceeded",
                    "final_support_budget_exhausted",
                    current_candidate,
                    current_loss,
                    history,
                    budget_ledger,
                )
                return self._finish(
                    current_candidate,
                    current_loss,
                    "final_support_budget_exhausted",
                    history,
                    budget_ledger,
                )

            history.append(
                AlgorithmStepRecord(
                    iteration=len(history),
                    phase="OPTIMIZE",
                    action_type=(
                        f"coordinate_proposed:{selected_coordinate.kind}:"
                        f"{selected_coordinate.node}"
                    ),
                    loss_before=current_loss,
                    loss_after=proposal.predicted_loss,
                    accepted=None,
                    termination_reason=None,
                    final_support_cost=self.final_support_budget.support_cost(
                        proposal.intervention,
                        self.settings.support_atol,
                    ),
                    experiment_trials_used=budget_ledger.experiment_trials_used,
                    local_loss_before=local_loss_before,
                    local_loss_after=local_loss_after,
                )
            )

            if (
                self.environment.candidate_evaluation_requires_experiment
                and budget_ledger.remaining_experiment_trials == 0
            ):
                self._record_failure(
                    "EVALUATE",
                    "evaluation_budget_exhausted",
                    "experiment_budget_exhausted",
                    current_candidate,
                    current_loss,
                    history,
                    budget_ledger,
                )
                return self._finish(
                    current_candidate,
                    current_loss,
                    "experiment_budget_exhausted",
                    history,
                    budget_ledger,
                )

            evaluation_cost = AtomicCoordinateCost(0, 0)
            if self.environment.candidate_evaluation_requires_experiment:
                evaluation_cost = budget_ledger.reserve_candidate_evaluation(
                    proposal.intervention,
                    support_atol=self.settings.support_atol,
                )
            try:
                evaluation = self.environment.evaluate_candidate(
                    proposal.intervention,
                    problem_view,
                    self.loss,
                    evaluation_rng,
                )
                candidate_loss = self._validated_loss(
                    evaluation.loss
                    if isinstance(evaluation, CandidateEvaluation)
                    else evaluation
                )
            except (
                FloatingPointError,
                np.linalg.LinAlgError,
                RuntimeError,
                ValueError,
            ):
                self._record_failure(
                    "EVALUATE",
                    "evaluation_failed",
                    "numerical_failure",
                    current_candidate,
                    current_loss,
                    history,
                    budget_ledger,
                )
                return self._finish(
                    current_candidate,
                    current_loss,
                    "numerical_failure",
                    history,
                    budget_ledger,
                )

            accepted = (
                candidate_loss <= current_loss - self.settings.improvement_tolerance
            )
            if self.environment.candidate_evaluation_requires_experiment:
                budget_ledger.record_candidate_outcome(accepted=accepted)
            if accepted:
                previous_candidate = current_candidate
                current_candidate = proposal.intervention
                current_loss = candidate_loss
                consecutive_rejections = 0
                rejected_coordinates.clear()
                if isinstance(self.knowledge, AcceptedBackgroundListener):
                    self.knowledge.accepted_background_changed(
                        previous_candidate,
                        current_candidate,
                    )
            else:
                consecutive_rejections += 1
                rejected_coordinates.add(selected_coordinate)

            patience_exhausted = (
                not accepted and consecutive_rejections >= self.settings.patience
            )
            history.append(
                AlgorithmStepRecord(
                    iteration=len(history),
                    phase="EVALUATE",
                    action_type=(
                        "candidate_accepted" if accepted else "candidate_rejected"
                    ),
                    loss_before=loss_before,
                    loss_after=candidate_loss,
                    accepted=accepted,
                    termination_reason=(
                        "patience_exhausted" if patience_exhausted else None
                    ),
                    final_support_cost=self.final_support_budget.support_cost(
                        current_candidate,
                        self.settings.support_atol,
                    ),
                    experiment_trials_used=budget_ledger.experiment_trials_used,
                    shift_coordinate_uses=evaluation_cost.shift_coordinate_uses,
                    scale_coordinate_uses=evaluation_cost.scale_coordinate_uses,
                    local_loss_before=local_loss_before,
                    local_loss_after=local_loss_after,
                )
            )
            if patience_exhausted:
                return self._finish(
                    current_candidate,
                    current_loss,
                    "patience_exhausted",
                    history,
                    budget_ledger,
                )

    def _legal_actionable_coordinates(
        self,
        current_candidate: ShiftScaleIntervention,
        coordinates: tuple[ParameterCoordinate, ...],
    ) -> tuple[ParameterCoordinate, ...]:
        """Keep adjustable active coordinates and inactive coordinates that fit."""
        shifted, scaled = coordinate_support_masks(
            current_candidate.shift,
            current_candidate.scale,
            support_atol=self.settings.support_atol,
        )
        remaining = self.final_support_budget.remaining_after(
            current_candidate,
            self.settings.support_atol,
        )
        return tuple(
            coordinate
            for coordinate in coordinates
            if (
                shifted[coordinate.node]
                if coordinate.kind == "shift"
                else scaled[coordinate.node]
            )
            or remaining > 0
        )

    def _best_coordinate_proposal(
        self,
        current_candidate: ShiftScaleIntervention,
        coordinates: tuple[ParameterCoordinate, ...],
        problem_view: PublicProblemView,
        state: AlgorithmStateView,
    ) -> tuple[CandidateProposal, ParameterCoordinate, float, float]:
        """Optimize every legal coordinate and return the best local reduction."""
        best: (
            tuple[
                float,
                CandidateProposal,
                ParameterCoordinate,
                float,
                float,
            ]
            | None
        ) = None
        for coordinate in coordinates:
            active = (coordinate,)
            proposal = self.optimizer.optimize(
                current_candidate,
                active,
                problem_view,
                self.final_support_budget,
            )
            if not self._proposal_respects_contract(
                proposal,
                current_candidate,
                active,
            ):
                raise RuntimeError("optimizer violated one-coordinate semantics.")
            if not self.final_support_budget.allows(
                proposal.intervention,
                self.settings.support_atol,
            ):
                continue
            if isinstance(self.knowledge, LocalProposalLossProvider):
                local_before = self._validated_loss(
                    self.knowledge.proposal_local_loss(
                        coordinate,
                        current_candidate,
                        state,
                    )
                )
                local_after = self._validated_loss(
                    self.knowledge.proposal_local_loss(
                        coordinate,
                        proposal.intervention,
                        state,
                    )
                )
            else:
                local_before = state.current_loss
                local_after = self._validated_loss(
                    proposal.predicted_loss
                    if proposal.predicted_loss is not None
                    else self.loss.value(proposal.intervention, problem_view)
                )
            reduction = local_before - local_after
            if best is None or reduction > best[0]:
                best = (
                    reduction,
                    proposal,
                    coordinate,
                    local_before,
                    local_after,
                )
        if best is None:
            raise FinalSupportBudgetExceededError(
                "no one-coordinate proposal fits the final-support budget."
            )
        _, proposal, coordinate, local_before, local_after = best
        return proposal, coordinate, local_before, local_after

    def _explore_until_actionable(
        self,
        problem_view: PublicProblemView,
        current_candidate: ShiftScaleIntervention,
        current_loss: float,
        consecutive_rejections: int,
        history: list[AlgorithmStepRecord],
        budget_ledger: ExperimentBudgetLedger,
        rng: np.random.Generator,
        excluded_coordinates: frozenset[ParameterCoordinate],
    ) -> TerminationReason | None:
        """Explore only while graph uncertainty leaves no actionable coordinate."""
        for _ in range(self.settings.max_consecutive_exploration_trials):
            if len(history) >= self.settings.max_steps:
                return "iteration_limit_reached"
            if budget_ledger.remaining_experiment_trials == 0:
                self._record_failure(
                    "EXPLORE",
                    "exploration_budget_exhausted",
                    "experiment_budget_exhausted",
                    current_candidate,
                    current_loss,
                    history,
                    budget_ledger,
                )
                return "experiment_budget_exhausted"

            state = self._state(
                problem_view,
                current_candidate,
                current_loss,
                history,
                consecutive_rejections,
            )
            policy_knowledge: object = self.knowledge
            if isinstance(self.knowledge, ExplorationKnowledgeProvider):
                policy_knowledge = self.knowledge.exploration_knowledge(state)
            proposal = self.exploration_policy.propose_probe(
                state,
                policy_knowledge,
                self._budget_view(current_candidate, budget_ledger),
                rng,
            )
            if proposal is None:
                self._record_failure(
                    "EXPLORE",
                    "no_legal_probe",
                    "no_legal_action",
                    current_candidate,
                    current_loss,
                    history,
                    budget_ledger,
                )
                return "no_legal_action"
            if not self._legal_probe(proposal, problem_view.n_nodes):
                self._record_failure(
                    "EXPLORE",
                    "illegal_probe",
                    "no_legal_action",
                    current_candidate,
                    current_loss,
                    history,
                    budget_ledger,
                )
                return "no_legal_action"

            probe_intervention = proposal.apply_to(
                current_candidate,
                max_probe_targets=self.settings.max_probe_targets,
            )
            cost = budget_ledger.record_exploration(proposal)
            try:
                evidence = self.environment.execute_probe(
                    current_candidate,
                    proposal,
                    problem_view,
                    rng,
                )
                if isinstance(evidence, ProbeOutcome):
                    samples = evidence.samples
                    outcome = evidence
                else:
                    samples = evidence
                    outcome = None
                observation = ExplorationObservation(
                    proposal=proposal,
                    intervention=probe_intervention,
                    samples=samples,
                    outcome=outcome,
                )
            except (
                FloatingPointError,
                np.linalg.LinAlgError,
                RuntimeError,
                ValueError,
            ):
                self._record_failure(
                    "EXPLORE",
                    "probe_execution_failed",
                    "numerical_failure",
                    current_candidate,
                    current_loss,
                    history,
                    budget_ledger,
                )
                return "numerical_failure"

            history.append(
                AlgorithmStepRecord(
                    iteration=len(history),
                    phase="EXPLORE",
                    action_type=f"probe:{proposal.reason}",
                    loss_before=current_loss,
                    loss_after=None,
                    accepted=None,
                    termination_reason=None,
                    final_support_cost=self.final_support_budget.support_cost(
                        current_candidate,
                        self.settings.support_atol,
                    ),
                    experiment_trials_used=budget_ledger.experiment_trials_used,
                    shift_coordinate_uses=cost.shift_coordinate_uses,
                    scale_coordinate_uses=cost.scale_coordinate_uses,
                )
            )
            try:
                self.knowledge.update_from_probe(observation, state)
            except (
                FloatingPointError,
                np.linalg.LinAlgError,
                RuntimeError,
                ValueError,
            ):
                return "numerical_failure"

            next_state = self._state(
                problem_view,
                current_candidate,
                current_loss,
                history,
                consecutive_rejections,
            )
            unresolved = self.knowledge.unresolved_variables_remain(next_state)
            if not isinstance(unresolved, bool):
                raise TypeError("unresolved_variables_remain must return a boolean.")
            if not unresolved:
                return None
            if any(
                coordinate not in excluded_coordinates
                for coordinate in self._actionable_coordinates(next_state)
            ):
                return None

        if budget_ledger.remaining_experiment_trials == 0:
            return "experiment_budget_exhausted"
        return "max_consecutive_exploration_trials_reached"

    def _state(
        self,
        problem_view: PublicProblemView,
        current_candidate: ShiftScaleIntervention,
        current_loss: float,
        history: list[AlgorithmStepRecord],
        consecutive_rejections: int,
    ) -> AlgorithmStateView:
        """Build one immutable public scheduler state."""
        return AlgorithmStateView(
            problem_view=problem_view,
            current_candidate=current_candidate,
            current_loss=current_loss,
            step=len(history),
            consecutive_rejections=consecutive_rejections,
        )

    def _budget_view(
        self,
        current_candidate: ShiftScaleIntervention,
        budget_ledger: ExperimentBudgetLedger,
    ) -> BudgetView:
        """Build one read-only budget view for an exploration policy."""
        return BudgetView(
            remaining_experiment_trials=(budget_ledger.remaining_experiment_trials),
            remaining_final_support=self.final_support_budget.remaining_after(
                current_candidate,
                self.settings.support_atol,
            ),
            max_experiment_trials=budget_ledger.max_experiment_trials,
            max_parameter_coordinates=(
                self.final_support_budget.max_parameter_coordinates
            ),
        )

    def _actionable_coordinates(
        self,
        state: AlgorithmStateView,
    ) -> tuple[ParameterCoordinate, ...]:
        """Query and validate the complete active coordinate tuple."""
        active_coordinates = tuple(
            coordinate
            for coordinate in self.knowledge.actionable_coordinates(state)
            if coordinate.kind in self.settings.allowed_coordinate_kinds
        )
        CandidateProposal(
            intervention=state.current_candidate,
            active_coordinates=active_coordinates,
            predicted_loss=None,
        )
        return active_coordinates

    def _legal_probe(self, proposal: object, n_nodes: int) -> bool:
        """Return whether a proposal is a nonempty bounded-width pure shift."""
        if not isinstance(proposal, ShiftProbe):
            return False
        if proposal.shift_increment.shape != (n_nodes,):
            return False
        return 0 < len(proposal.targets) <= self.settings.max_probe_targets

    def _proposal_respects_contract(
        self,
        proposal: object,
        current_candidate: ShiftScaleIntervention,
        active_coordinates: tuple[ParameterCoordinate, ...],
    ) -> bool:
        """Verify bounds, active set, and frozen-coordinate semantics."""
        if not isinstance(proposal, CandidateProposal):
            return False
        if proposal.active_coordinates != active_coordinates:
            return False
        if proposal.intervention.shift.shape != current_candidate.shift.shape:
            return False
        active_shifts = {
            coordinate.node
            for coordinate in active_coordinates
            if coordinate.kind == "shift"
        }
        active_scales = {
            coordinate.node
            for coordinate in active_coordinates
            if coordinate.kind == "log_scale"
        }
        for node in range(current_candidate.shift.size):
            if (
                node not in active_shifts
                and proposal.intervention.shift[node] != current_candidate.shift[node]
            ):
                return False
            if (
                node not in active_scales
                and proposal.intervention.scale[node] != current_candidate.scale[node]
            ):
                return False
        try:
            self._validate_scale_bounds(proposal.intervention)
        except ValueError:
            return False
        return True

    def _validate_scale_bounds(
        self,
        intervention: ShiftScaleIntervention,
    ) -> None:
        """Require every scale to satisfy the scheduler's hard bounds."""
        if np.any(intervention.scale < self.settings.scale_min) or np.any(
            intervention.scale > self.settings.scale_max
        ):
            raise ValueError(
                "intervention scales must lie within the configured bounds."
            )

    @staticmethod
    def _validated_loss(value: object) -> float:
        """Validate one finite loss returned by an injected dependency."""
        return _finite_real("loss", value)

    def _record_failure(
        self,
        phase: Literal["EXPLORE", "OPTIMIZE", "EVALUATE"],
        action_type: str,
        termination_reason: TerminationReason,
        current_candidate: ShiftScaleIntervention,
        current_loss: float,
        history: list[AlgorithmStepRecord],
        budget_ledger: ExperimentBudgetLedger,
    ) -> None:
        """Append a zero-cost phase record for a blocked or failed action."""
        history.append(
            AlgorithmStepRecord(
                iteration=len(history),
                phase=phase,
                action_type=action_type,
                loss_before=current_loss,
                loss_after=None,
                accepted=None,
                termination_reason=termination_reason,
                final_support_cost=self.final_support_budget.support_cost(
                    current_candidate,
                    self.settings.support_atol,
                ),
                experiment_trials_used=budget_ledger.experiment_trials_used,
            )
        )

    def _finish(
        self,
        current_candidate: ShiftScaleIntervention,
        current_loss: float,
        reason: TerminationReason,
        history: list[AlgorithmStepRecord],
        budget_ledger: ExperimentBudgetLedger,
    ) -> SchedulerResult:
        """Append the explicit STOP transition and return the final result."""
        history.append(
            AlgorithmStepRecord(
                iteration=len(history),
                phase="STOP",
                action_type="terminate",
                loss_before=current_loss,
                loss_after=current_loss,
                accepted=None,
                termination_reason=reason,
                final_support_cost=self.final_support_budget.support_cost(
                    current_candidate,
                    self.settings.support_atol,
                ),
                experiment_trials_used=budget_ledger.experiment_trials_used,
            )
        )
        return SchedulerResult(
            final_candidate=current_candidate,
            final_loss=current_loss,
            termination_reason=reason,
            history=tuple(history),
            budget_ledger=budget_ledger,
        )
