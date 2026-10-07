"""Oracle-only exact mean-mismatch classification for idealized pilots."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import PublicProblemView
from graphs.scm import LinearGaussianSCM


@dataclass(frozen=True)
class OracleMeanMismatchProvider:
    """Classify exact mean mismatch using the hidden simulation SCM.

    This provider is intentionally oracle-only. It returns only a node set
    through the public ``MismatchProvider`` interface; neither the SCM nor its
    coefficient matrix crosses into exploration policies.
    """

    scm: LinearGaussianSCM
    mean_atol: float

    def __post_init__(self) -> None:
        """Validate the hidden environment and explicit mismatch tolerance."""
        if not isinstance(self.scm, LinearGaussianSCM):
            raise TypeError("scm must be a LinearGaussianSCM.")
        if isinstance(self.mean_atol, bool) or not isinstance(self.mean_atol, Real):
            raise TypeError("mean_atol must be a real number.")
        mean_atol = float(self.mean_atol)
        if not np.isfinite(mean_atol):
            raise ValueError("mean_atol must be finite.")
        if mean_atol < 0.0:
            raise ValueError("mean_atol must be nonnegative.")
        object.__setattr__(self, "mean_atol", mean_atol)

    def unresolved_nodes(
        self,
        public_problem: PublicProblemView,
        candidate: ShiftScaleIntervention,
    ) -> frozenset[int]:
        """Return nodes whose exact candidate mean misses the public target."""
        if not isinstance(public_problem, PublicProblemView):
            raise TypeError("public_problem must be a PublicProblemView.")
        if not isinstance(candidate, ShiftScaleIntervention):
            raise TypeError("candidate must be a ShiftScaleIntervention.")
        n_nodes = self.scm.dag.number_of_nodes()
        if public_problem.n_nodes != n_nodes:
            raise ValueError("public problem dimension must agree with the hidden SCM.")
        if candidate.shift.shape != (n_nodes,):
            raise ValueError("candidate dimension must agree with the hidden SCM.")

        candidate_mean, _ = self.scm.interventional_moments(candidate)
        mismatch = np.abs(candidate_mean - public_problem.target_mean) > self.mean_atol
        return frozenset(int(node) for node in np.flatnonzero(mismatch))
