"""Oracle bounded optimization over explicitly supplied coordinates."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.optimize import minimize

from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import PublicProblemView
from graphs.support import (
    DEFAULT_SUPPORT_ATOL,
    coordinate_support_masks,
    validate_support_atol,
)
from matching.losses import DifferentiableInterventionLoss
from matching.optimizers import (
    CoordinateSelectionRequiredError,
    FinalSupportBudgetExceededError,
)
from matching.types import CandidateProposal, FinalSupportBudget, ParameterCoordinate


def _finite_real(name: str, value: object) -> float:
    """Validate and normalize one finite real optimizer setting."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


@dataclass(frozen=True)
class OracleActiveCoordinateOptimizer:
    """Minimize oracle Gaussian KL with L-BFGS-B on a compact active vector."""

    loss: DifferentiableInterventionLoss
    scale_min: float
    scale_max: float
    support_atol: float = DEFAULT_SUPPORT_ATOL
    max_iterations: int = 1_000

    def __post_init__(self) -> None:
        """Validate optimizer dependencies and numerical settings."""
        if not isinstance(self.loss, DifferentiableInterventionLoss):
            raise TypeError(
                "loss must be a DifferentiableInterventionLoss with value() "
                "and value_and_gradient()."
            )
        scale_min = _finite_real("scale_min", self.scale_min)
        scale_max = _finite_real("scale_max", self.scale_max)
        if not 0.0 < scale_min <= scale_max:
            raise ValueError("scale bounds must satisfy 0 < scale_min <= scale_max.")
        support_atol = validate_support_atol(self.support_atol)
        if isinstance(self.max_iterations, bool) or not isinstance(
            self.max_iterations,
            int,
        ):
            raise TypeError("max_iterations must be an integer.")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        object.__setattr__(self, "scale_min", scale_min)
        object.__setattr__(self, "scale_max", scale_max)
        object.__setattr__(self, "support_atol", support_atol)

    def optimize(
        self,
        current_candidate: ShiftScaleIntervention,
        active_coordinates: tuple[ParameterCoordinate, ...],
        problem_view: PublicProblemView,
        final_support_budget: FinalSupportBudget,
    ) -> CandidateProposal:
        """Optimize exactly the supplied coordinates and freeze all others."""
        self._validate_request(
            current_candidate,
            active_coordinates,
            problem_view,
            final_support_budget,
        )
        self._ensure_active_set_fits_budget(
            current_candidate,
            active_coordinates,
            final_support_budget,
        )
        if not active_coordinates:
            return CandidateProposal(
                intervention=current_candidate,
                active_coordinates=active_coordinates,
                predicted_loss=self.loss.value(current_candidate, problem_view),
            )

        initial_point = self._pack(current_candidate, active_coordinates)
        bounds = [
            (
                (None, None)
                if coordinate.kind == "shift"
                else (np.log(self.scale_min), np.log(self.scale_max))
            )
            for coordinate in active_coordinates
        ]

        def objective(compact_parameters: np.ndarray) -> tuple[float, np.ndarray]:
            candidate = self._unpack(
                compact_parameters,
                current_candidate,
                active_coordinates,
            )
            value, shift_gradient, log_scale_gradient = self.loss.value_and_gradient(
                candidate, problem_view
            )
            gradient = np.array(
                [
                    (
                        shift_gradient[coordinate.node]
                        if coordinate.kind == "shift"
                        else log_scale_gradient[coordinate.node]
                    )
                    for coordinate in active_coordinates
                ],
                dtype=float,
            )
            return value, gradient

        result = minimize(
            objective,
            initial_point,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxiter": self.max_iterations,
                "maxls": 50,
            },
        )
        reached_numerical_zero = (
            hasattr(result, "fun")
            and np.isfinite(result.fun)
            and float(result.fun) == 0.0
        )
        if not result.success and not reached_numerical_zero:
            raise RuntimeError(f"oracle optimization failed: {result.message}")
        if not np.all(np.isfinite(result.x)):
            raise RuntimeError("oracle optimization returned nonfinite parameters.")

        proposed_intervention = self._unpack(
            result.x,
            current_candidate,
            active_coordinates,
        )
        if not final_support_budget.allows(
            proposed_intervention,
            self.support_atol,
        ):
            raise FinalSupportBudgetExceededError(
                "optimized proposal exceeds the final-support budget."
            )
        return CandidateProposal(
            intervention=proposed_intervention,
            active_coordinates=active_coordinates,
            predicted_loss=self.loss.value(proposed_intervention, problem_view),
        )

    def _validate_request(
        self,
        current_candidate: ShiftScaleIntervention,
        active_coordinates: tuple[ParameterCoordinate, ...],
        problem_view: PublicProblemView,
        final_support_budget: FinalSupportBudget,
    ) -> None:
        """Validate public request data without selecting active coordinates."""
        CandidateProposal(
            intervention=current_candidate,
            active_coordinates=active_coordinates,
            predicted_loss=None,
        )
        if not isinstance(problem_view, PublicProblemView):
            raise TypeError("problem_view must be a PublicProblemView.")
        if not isinstance(final_support_budget, FinalSupportBudget):
            raise TypeError("final_support_budget must be a FinalSupportBudget.")
        if current_candidate.shift.shape != (problem_view.n_nodes,):
            raise ValueError(
                "current candidate dimension must agree with the problem view."
            )
        if np.any(current_candidate.scale < self.scale_min) or np.any(
            current_candidate.scale > self.scale_max
        ):
            raise ValueError(
                "current candidate scales must lie within the configured bounds."
            )

    def _ensure_active_set_fits_budget(
        self,
        current_candidate: ShiftScaleIntervention,
        active_coordinates: tuple[ParameterCoordinate, ...],
        final_support_budget: FinalSupportBudget,
    ) -> None:
        """Reject infeasible support without inventing an L0 selection rule."""
        current_support_cost = final_support_budget.support_cost(
            current_candidate,
            self.support_atol,
        )
        if current_support_cost > final_support_budget.max_parameter_coordinates:
            raise FinalSupportBudgetExceededError(
                "current candidate exceeds the final-support budget."
            )

        shifted, scaled = coordinate_support_masks(
            current_candidate.shift,
            current_candidate.scale,
            support_atol=self.support_atol,
        )
        potentially_new_coordinates = sum(
            (
                not shifted[coordinate.node]
                if coordinate.kind == "shift"
                else not scaled[coordinate.node]
            )
            for coordinate in active_coordinates
        )
        if (
            current_support_cost + potentially_new_coordinates
            > final_support_budget.max_parameter_coordinates
        ):
            raise CoordinateSelectionRequiredError(
                "active coordinates exceed the remaining final-support budget; "
                "a coordinate-selection rule is required."
            )

    @staticmethod
    def _pack(
        intervention: ShiftScaleIntervention,
        active_coordinates: tuple[ParameterCoordinate, ...],
    ) -> np.ndarray:
        """Pack active full-state coordinates into one compact vector."""
        return np.array(
            [
                (
                    intervention.shift[coordinate.node]
                    if coordinate.kind == "shift"
                    else np.log(intervention.scale[coordinate.node])
                )
                for coordinate in active_coordinates
            ],
            dtype=float,
        )

    @staticmethod
    def _unpack(
        compact_parameters: np.ndarray,
        current_candidate: ShiftScaleIntervention,
        active_coordinates: tuple[ParameterCoordinate, ...],
    ) -> ShiftScaleIntervention:
        """Map a compact vector into full arrays while freezing other entries."""
        shift = current_candidate.shift.copy()
        scale = current_candidate.scale.copy()
        for value, coordinate in zip(
            compact_parameters,
            active_coordinates,
            strict=True,
        ):
            if coordinate.kind == "shift":
                shift[coordinate.node] = value
            else:
                scale[coordinate.node] = np.exp(value)
        return ShiftScaleIntervention(shift=shift, scale=scale)
