"""Public and oracle-only containers for causal state-matching problems."""

from dataclasses import dataclass

import numpy as np

from ._immutability import immutable_array
from .interventions import ShiftScaleIntervention
from .scm import LinearGaussianSCM
from .seeds import AlgorithmSeeds, SimulationSeeds


@dataclass(frozen=True)
class PublicProblemView:
    """Algorithm-facing problem data with no hidden causal ground truth.

    Non-oracle matching code should accept this type rather than
    :class:`GeneratedProblem`. In particular, this view contains neither the
    true graph and coefficient matrix nor the target-generating intervention.
    """

    observational_samples: np.ndarray
    target_mean: np.ndarray
    target_covariance: np.ndarray
    current_intervention: ShiftScaleIntervention
    algorithm_seeds: AlgorithmSeeds

    def __post_init__(self) -> None:
        """Validate dimensions and defensively freeze all public arrays."""
        if not isinstance(self.current_intervention, ShiftScaleIntervention):
            raise TypeError("current_intervention must be a ShiftScaleIntervention.")
        if not isinstance(self.algorithm_seeds, AlgorithmSeeds):
            raise TypeError("algorithm_seeds must be an AlgorithmSeeds.")

        arrays = {
            "observational_samples": (self.observational_samples, 2),
            "target_mean": (self.target_mean, 1),
            "target_covariance": (self.target_covariance, 2),
        }
        frozen_arrays = {}
        for name, (array, ndim) in arrays.items():
            if not isinstance(array, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray.")
            if array.ndim != ndim:
                dimensionality = "one-dimensional" if ndim == 1 else "two-dimensional"
                raise ValueError(f"{name} must be {dimensionality}.")
            if not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array):
                raise TypeError(f"{name} must contain real numeric values.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values.")
            frozen_arrays[name] = immutable_array(array, dtype=float)

        frozen_samples = frozen_arrays["observational_samples"]
        frozen_mean = frozen_arrays["target_mean"]
        frozen_covariance = frozen_arrays["target_covariance"]
        n_nodes = frozen_mean.size
        if frozen_covariance.shape != (n_nodes, n_nodes):
            raise ValueError("target mean and covariance dimensions must agree.")
        if frozen_samples.shape[1] != n_nodes:
            raise ValueError(
                "observational sample dimension must agree with the target dimension."
            )
        if self.current_intervention.shift.shape != (n_nodes,):
            raise ValueError(
                "current intervention dimension must agree with the target dimension."
            )
        if not np.allclose(
            frozen_covariance,
            frozen_covariance.T,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("target_covariance must be symmetric.")
        if n_nodes:
            try:
                np.linalg.cholesky(frozen_covariance)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "target_covariance must be positive definite."
                ) from error

        object.__setattr__(self, "observational_samples", frozen_samples)
        object.__setattr__(self, "target_mean", frozen_mean)
        object.__setattr__(self, "target_covariance", frozen_covariance)

    @property
    def n_nodes(self) -> int:
        """Return the target dimension."""
        return int(self.target_mean.size)


@dataclass(frozen=True)
class SimulationGroundTruth:
    """Oracle-only simulation state that must not reach non-oracle code."""

    scm: LinearGaussianSCM
    generating_intervention: ShiftScaleIntervention
    seeds: SimulationSeeds

    def __post_init__(self) -> None:
        """Validate the hidden intervention against the hidden SCM."""
        if not isinstance(self.scm, LinearGaussianSCM):
            raise TypeError("scm must be a LinearGaussianSCM.")
        if not isinstance(
            self.generating_intervention,
            ShiftScaleIntervention,
        ):
            raise TypeError("generating_intervention must be a ShiftScaleIntervention.")
        if not isinstance(self.seeds, SimulationSeeds):
            raise TypeError("seeds must be a SimulationSeeds.")

        self.scm.validate_target_intervention(self.generating_intervention)
        n_nodes = self.scm.dag.number_of_nodes()
        if self.generating_intervention.shift.shape != (n_nodes,):
            raise ValueError(
                "generating intervention dimension must agree with the SCM."
            )


@dataclass(frozen=True)
class GeneratedProblem:
    """A public problem paired explicitly with oracle-only simulation truth."""

    public: PublicProblemView
    ground_truth: SimulationGroundTruth

    def __post_init__(self) -> None:
        """Validate the public and hidden dimensions without merging their APIs."""
        if not isinstance(self.public, PublicProblemView):
            raise TypeError("public must be a PublicProblemView.")
        if not isinstance(self.ground_truth, SimulationGroundTruth):
            raise TypeError("ground_truth must be a SimulationGroundTruth.")
        if self.ground_truth.scm.dag.number_of_nodes() != self.public.n_nodes:
            raise ValueError("public and ground-truth dimensions must agree.")

    def current_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Return oracle exact moments under the current public intervention."""
        return self.ground_truth.scm.interventional_moments(
            self.public.current_intervention
        )

    def sample_current(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Query the hidden simulation environment at the current intervention."""
        return self.ground_truth.scm.sample_interventional(
            self.public.current_intervention,
            n_samples,
            rng,
        )
