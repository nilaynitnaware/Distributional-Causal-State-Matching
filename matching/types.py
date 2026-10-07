"""Validated data types for matching proposals and budget accounting."""

from dataclasses import dataclass, field
from hashlib import sha256
from numbers import Real
from typing import Literal, Self

import numpy as np

from graphs._immutability import immutable_array
from graphs.interventions import ShiftScaleIntervention
from graphs.support import DEFAULT_SUPPORT_ATOL, coordinate_support_masks

from .probes import ShiftProbe

ProbeProposal = ShiftProbe

ParameterKind = Literal["shift", "log_scale"]
AlgorithmPhase = Literal["EXPLORE", "OPTIMIZE", "EVALUATE", "STOP"]

_PARAMETER_KINDS = frozenset({"shift", "log_scale"})
_ALGORITHM_PHASES = frozenset({"EXPLORE", "OPTIMIZE", "EVALUATE", "STOP"})


def _validate_nonnegative_integer(name: str, value: object) -> None:
    """Validate a counter or index."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _finite_real(name: str, value: object) -> float:
    """Validate and normalize one finite real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _validate_nonempty_string(name: str, value: object) -> None:
    """Validate a descriptive string without imposing scientific semantics."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


@dataclass(frozen=True)
class InterventionBackground:
    """Accepted shift/scale state under which evidence was collected."""

    shift: np.ndarray = field(compare=False)
    scale: np.ndarray = field(compare=False)
    identifier: str = ""

    def __post_init__(self) -> None:
        """Freeze arrays and derive a deterministic content identifier."""
        intervention = ShiftScaleIntervention(self.shift, self.scale)
        frozen_shift = immutable_array(intervention.shift, dtype=float)
        frozen_scale = immutable_array(intervention.scale, dtype=float)
        digest = sha256()
        digest.update(frozen_shift.dtype.str.encode())
        digest.update(str(frozen_shift.shape).encode())
        digest.update(frozen_shift.tobytes())
        digest.update(frozen_scale.tobytes())
        computed_identifier = digest.hexdigest()
        if self.identifier and self.identifier != computed_identifier:
            raise ValueError("background identifier does not match its intervention.")
        object.__setattr__(self, "shift", frozen_shift)
        object.__setattr__(self, "scale", frozen_scale)
        object.__setattr__(self, "identifier", computed_identifier)

    @classmethod
    def from_intervention(
        cls,
        intervention: ShiftScaleIntervention,
    ) -> Self:
        """Construct a background record from one accepted intervention."""
        if not isinstance(intervention, ShiftScaleIntervention):
            raise TypeError("intervention must be a ShiftScaleIntervention.")
        return cls(intervention.shift, intervention.scale)

    def matches(self, intervention: ShiftScaleIntervention) -> bool:
        """Return whether ``intervention`` is exactly this accepted state."""
        background = type(self).from_intervention(intervention)
        return self.identifier == background.identifier


@dataclass(frozen=True, order=True)
class ParameterCoordinate:
    """One optimizable shift or log-scale coordinate at a node."""

    node: int
    kind: ParameterKind

    def __post_init__(self) -> None:
        """Validate the node index and coordinate kind."""
        _validate_nonnegative_integer("node", self.node)
        if not isinstance(self.kind, str):
            raise TypeError("kind must be a string.")
        if self.kind not in _PARAMETER_KINDS:
            raise ValueError("kind must be 'shift' or 'log_scale'.")


@dataclass(frozen=True)
class AtomicCoordinateCost:
    """Shift and scale coordinate uses for one executed intervention."""

    shift_coordinate_uses: int
    scale_coordinate_uses: int

    def __post_init__(self) -> None:
        """Validate both atomic-use counts."""
        _validate_nonnegative_integer(
            "shift_coordinate_uses",
            self.shift_coordinate_uses,
        )
        _validate_nonnegative_integer(
            "scale_coordinate_uses",
            self.scale_coordinate_uses,
        )

    @classmethod
    def from_intervention(
        cls,
        intervention: ShiftScaleIntervention,
        support_atol: object = DEFAULT_SUPPORT_ATOL,
    ) -> Self:
        """Count active shift and log-scale coordinates for one experiment."""
        if not isinstance(intervention, ShiftScaleIntervention):
            raise TypeError("intervention must be a ShiftScaleIntervention.")
        shifted, scaled = coordinate_support_masks(
            intervention.shift,
            intervention.scale,
            support_atol=support_atol,
        )
        return cls(
            shift_coordinate_uses=int(np.count_nonzero(shifted)),
            scale_coordinate_uses=int(np.count_nonzero(scaled)),
        )

    @classmethod
    def from_probe(cls, probe: ShiftProbe) -> Self:
        """Count only the temporary nonzero shift increments of a probe."""
        if not isinstance(probe, ShiftProbe):
            raise TypeError("probe must be a ShiftProbe.")
        return cls(
            shift_coordinate_uses=len(probe.targets),
            scale_coordinate_uses=0,
        )

    @property
    def total_atomic_coordinate_uses(self) -> int:
        """Return the total atomic action cost."""
        return self.shift_coordinate_uses + self.scale_coordinate_uses


@dataclass(frozen=True)
class FinalSupportBudget:
    """Cardinality limit for the final candidate's parameter coordinates."""

    max_parameter_coordinates: int

    def __post_init__(self) -> None:
        """Validate the final unique-support limit."""
        _validate_nonnegative_integer(
            "max_parameter_coordinates",
            self.max_parameter_coordinates,
        )

    @classmethod
    def twice_k(cls, k: int) -> Self:
        """Construct a ``2k`` parameter-coordinate budget."""
        _validate_nonnegative_integer("k", k)
        return cls(max_parameter_coordinates=2 * k)

    @classmethod
    def twice_target_support(cls, target_support_size: int) -> Self:
        """Compatibility alias; callers should pass experiment hyperparameter k."""
        return cls.twice_k(target_support_size)

    def support_cost(
        self,
        intervention: ShiftScaleIntervention,
        support_atol: object = DEFAULT_SUPPORT_ATOL,
    ) -> int:
        """Return the candidate's unique final support cost."""
        if not isinstance(intervention, ShiftScaleIntervention):
            raise TypeError("intervention must be a ShiftScaleIntervention.")
        return intervention.support_cost(support_atol)

    def allows(
        self,
        intervention: ShiftScaleIntervention,
        support_atol: object = DEFAULT_SUPPORT_ATOL,
    ) -> bool:
        """Return whether a candidate fits the final support budget."""
        return (
            self.support_cost(intervention, support_atol)
            <= self.max_parameter_coordinates
        )

    def remaining_after(
        self,
        intervention: ShiftScaleIntervention,
        support_atol: object = DEFAULT_SUPPORT_ATOL,
    ) -> int:
        """Return remaining capacity, negative when the candidate exceeds it."""
        return self.max_parameter_coordinates - self.support_cost(
            intervention,
            support_atol,
        )


@dataclass
class ExperimentBudgetLedger:
    """Cumulative experimental-trial and atomic-action accounting."""

    max_experiment_trials: int
    exploration_trials: int = 0
    candidate_evaluation_trials: int = 0
    accepted_matching_trials: int = 0
    rejected_matching_trials: int = 0
    failed_matching_trials: int = 0
    shift_coordinate_uses: int = 0
    scale_coordinate_uses: int = 0

    def __post_init__(self) -> None:
        """Validate initial counters and cross-counter invariants."""
        self._validate_state()

    @property
    def experiment_trials_used(self) -> int:
        """Count physical environment queries without double-counting outcomes."""
        return self.exploration_trials + self.candidate_evaluation_trials

    @property
    def remaining_experiment_trials(self) -> int:
        """Return unused physical-query capacity."""
        return self.max_experiment_trials - self.experiment_trials_used

    @property
    def total_atomic_coordinate_uses(self) -> int:
        """Return cumulative shift-plus-scale coordinate uses."""
        return self.shift_coordinate_uses + self.scale_coordinate_uses

    def record_exploration(
        self,
        probe: ShiftProbe,
    ) -> AtomicCoordinateCost:
        """Record one trial and the probe increment's shift-coordinate uses."""
        cost = AtomicCoordinateCost.from_probe(probe)
        self._ensure_trial_available()
        self.exploration_trials += 1
        self._add_cost(cost)
        self._validate_state()
        return cost

    def reserve_candidate_evaluation(
        self,
        intervention: ShiftScaleIntervention,
        support_atol: object = DEFAULT_SUPPORT_ATOL,
    ) -> AtomicCoordinateCost:
        """Charge a physical query immediately before candidate execution."""
        cost = AtomicCoordinateCost.from_intervention(intervention, support_atol)
        self._ensure_trial_available()
        self.candidate_evaluation_trials += 1
        self.failed_matching_trials += 1
        self._add_cost(cost)
        self._validate_state()
        return cost

    def record_candidate_outcome(self, *, accepted: bool) -> None:
        """Replace one pessimistic failed reservation by its decision."""
        if not isinstance(accepted, bool):
            raise TypeError("accepted must be a boolean.")
        if self.failed_matching_trials == 0:
            raise RuntimeError("no reserved candidate evaluation awaits an outcome.")
        self.failed_matching_trials -= 1
        if accepted:
            self.accepted_matching_trials += 1
        else:
            self.rejected_matching_trials += 1
        self._validate_state()

    def record_candidate_evaluation(
        self,
        intervention: ShiftScaleIntervention,
        *,
        accepted: bool,
        support_atol: object = DEFAULT_SUPPORT_ATOL,
    ) -> AtomicCoordinateCost:
        """Compatibility helper that reserves and immediately records a decision."""
        if not isinstance(accepted, bool):
            raise TypeError("accepted must be a boolean.")
        cost = self.reserve_candidate_evaluation(intervention, support_atol)
        self.record_candidate_outcome(accepted=accepted)
        return cost

    def _ensure_trial_available(self) -> None:
        """Reject physical queries once the trial budget is exhausted."""
        self._validate_state()
        if self.remaining_experiment_trials == 0:
            raise RuntimeError("experimental trial budget is exhausted.")

    def _add_cost(self, cost: AtomicCoordinateCost) -> None:
        """Add one already validated atomic action cost."""
        self.shift_coordinate_uses += cost.shift_coordinate_uses
        self.scale_coordinate_uses += cost.scale_coordinate_uses

    def _validate_state(self) -> None:
        """Validate mutable counters and their accounting relationships."""
        for name in (
            "max_experiment_trials",
            "exploration_trials",
            "candidate_evaluation_trials",
            "accepted_matching_trials",
            "rejected_matching_trials",
            "failed_matching_trials",
            "shift_coordinate_uses",
            "scale_coordinate_uses",
        ):
            _validate_nonnegative_integer(name, getattr(self, name))
        if (
            self.accepted_matching_trials
            + self.rejected_matching_trials
            + self.failed_matching_trials
            != self.candidate_evaluation_trials
        ):
            raise ValueError(
                "accepted, rejected, and failed matching trials must partition "
                "candidate_evaluation_trials."
            )
        if self.experiment_trials_used > self.max_experiment_trials:
            raise ValueError("experiment trial counts exceed max_experiment_trials.")


@dataclass(frozen=True)
class CandidateProposal:
    """A matching proposal over an explicit active-coordinate set."""

    intervention: ShiftScaleIntervention
    active_coordinates: tuple[ParameterCoordinate, ...]
    predicted_loss: float | None

    def __post_init__(self) -> None:
        """Validate the intervention, active set, and optional prediction."""
        if not isinstance(self.intervention, ShiftScaleIntervention):
            raise TypeError("intervention must be a ShiftScaleIntervention.")
        if not isinstance(self.active_coordinates, tuple):
            raise TypeError("active_coordinates must be a tuple.")
        if any(
            not isinstance(coordinate, ParameterCoordinate)
            for coordinate in self.active_coordinates
        ):
            raise TypeError(
                "active_coordinates must contain ParameterCoordinate objects."
            )
        if len(set(self.active_coordinates)) != len(self.active_coordinates):
            raise ValueError("active_coordinates must not contain duplicates.")
        n_nodes = self.intervention.shift.size
        if any(coordinate.node >= n_nodes for coordinate in self.active_coordinates):
            raise ValueError(
                "active coordinate nodes must fit the intervention dimension."
            )
        if self.predicted_loss is not None:
            object.__setattr__(
                self,
                "predicted_loss",
                _finite_real("predicted_loss", self.predicted_loss),
            )


@dataclass(frozen=True)
class AlgorithmStepRecord:
    """One immutable history record for a future alternating scheduler."""

    iteration: int
    phase: AlgorithmPhase
    action_type: str
    loss_before: float
    loss_after: float | None
    accepted: bool | None
    termination_reason: str | None
    final_support_cost: int
    experiment_trials_used: int
    shift_coordinate_uses: int = 0
    scale_coordinate_uses: int = 0
    local_loss_before: float | None = None
    local_loss_after: float | None = None

    def __post_init__(self) -> None:
        """Validate one scheduler-independent history record."""
        _validate_nonnegative_integer("iteration", self.iteration)
        if not isinstance(self.phase, str):
            raise TypeError("phase must be a string.")
        if self.phase not in _ALGORITHM_PHASES:
            raise ValueError("phase must be EXPLORE, OPTIMIZE, EVALUATE, or STOP.")
        _validate_nonempty_string("action_type", self.action_type)
        object.__setattr__(
            self,
            "loss_before",
            _finite_real("loss_before", self.loss_before),
        )
        if self.loss_after is not None:
            object.__setattr__(
                self,
                "loss_after",
                _finite_real("loss_after", self.loss_after),
            )
        for name in ("local_loss_before", "local_loss_after"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_real(name, value),
                )
        if self.accepted is not None and not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a boolean or None.")
        if self.termination_reason is not None:
            _validate_nonempty_string(
                "termination_reason",
                self.termination_reason,
            )
        for name in (
            "final_support_cost",
            "experiment_trials_used",
            "shift_coordinate_uses",
            "scale_coordinate_uses",
        ):
            _validate_nonnegative_integer(name, getattr(self, name))

    @property
    def total_atomic_coordinate_uses(self) -> int:
        """Return this action's shift-plus-scale coordinate use."""
        return self.shift_coordinate_uses + self.scale_coordinate_uses
