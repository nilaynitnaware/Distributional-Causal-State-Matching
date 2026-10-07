"""Pluggable loss interfaces for intervention matching."""

from typing import Protocol, runtime_checkable

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import PublicProblemView


@runtime_checkable
class InterventionLoss(Protocol):
    """A loss evaluable from a candidate and public problem state."""

    def value(
        self,
        candidate: ShiftScaleIntervention,
        problem_view: PublicProblemView,
    ) -> float:
        """Return the scalar loss for ``candidate``."""
        ...


@runtime_checkable
class DifferentiableInterventionLoss(InterventionLoss, Protocol):
    """An intervention loss with full shift and log-scale gradients."""

    def value_and_gradient(
        self,
        candidate: ShiftScaleIntervention,
        problem_view: PublicProblemView,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Return loss, shift gradient, and log-scale gradient."""
        ...
