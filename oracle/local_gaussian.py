"""Explicitly oracle candidate-moment provider for local Gaussian losses."""

from dataclasses import dataclass

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import PublicProblemView
from graphs.scm import LinearGaussianSCM


@dataclass(frozen=True)
class OracleCandidateMomentsProvider:
    """Compute exact candidate moments using the hidden true coefficient matrix."""

    scm: LinearGaussianSCM

    def __post_init__(self) -> None:
        """Require an explicit hidden SCM."""
        if not isinstance(self.scm, LinearGaussianSCM):
            raise TypeError("scm must be a LinearGaussianSCM.")

    def moments(
        self,
        candidate: ShiftScaleIntervention,
        public_problem: PublicProblemView,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return analytic moments after validating public dimensions."""
        if not isinstance(candidate, ShiftScaleIntervention):
            raise TypeError("candidate must be a ShiftScaleIntervention.")
        if not isinstance(public_problem, PublicProblemView):
            raise TypeError("public_problem must be a PublicProblemView.")
        n_nodes = self.scm.dag.number_of_nodes()
        if public_problem.n_nodes != n_nodes or candidate.shift.shape != (n_nodes,):
            raise ValueError("candidate and public problem must match the oracle SCM.")
        return self.scm.interventional_moments(candidate)
