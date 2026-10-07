"""Generic optimizer interfaces for intervention matching."""

from typing import Protocol, runtime_checkable

from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import PublicProblemView

from .types import CandidateProposal, FinalSupportBudget, ParameterCoordinate


class FinalSupportBudgetExceededError(ValueError):
    """Raised when a current or proposed intervention exceeds final support."""


class CoordinateSelectionRequiredError(FinalSupportBudgetExceededError):
    """Raised when an active set requires a scientific selection rule."""


@runtime_checkable
class InterventionOptimizer(Protocol):
    """Optimizer contract exposed to the future non-oracle scheduler."""

    def optimize(
        self,
        current_candidate: ShiftScaleIntervention,
        active_coordinates: tuple[ParameterCoordinate, ...],
        problem_view: PublicProblemView,
        final_support_budget: FinalSupportBudget,
    ) -> CandidateProposal:
        """Return a proposal optimized only over ``active_coordinates``."""
        ...
