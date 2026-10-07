"""Backend contract for array-heavy numerical operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from graphs.interventions import ShiftScaleIntervention
    from graphs.scm import LinearGaussianSCM


@runtime_checkable
class MomentBackend(Protocol):
    """Analytic moment operations implemented without graph algorithms."""

    name: str
    device: Literal["cpu", "cuda"]
    dtype: Literal["float64"]

    def observational_moments(
        self,
        scm: LinearGaussianSCM,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact observational mean and covariance."""
        ...

    def interventional_moments(
        self,
        scm: LinearGaussianSCM,
        intervention: ShiftScaleIntervention,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact mean and covariance for an intervention."""
        ...


@runtime_checkable
class SamplingBackend(MomentBackend, Protocol):
    """SCM sampling operations using an explicit backend-specific generator."""

    def sample_observational(
        self,
        scm: LinearGaussianSCM,
        n_samples: int,
        rng: object,
    ) -> np.ndarray:
        """Return observational samples without using global RNG state."""
        ...

    def sample_interventional(
        self,
        scm: LinearGaussianSCM,
        intervention: ShiftScaleIntervention,
        n_samples: int,
        rng: object,
    ) -> np.ndarray:
        """Return interventional samples without using global RNG state."""
        ...


@runtime_checkable
class NumericalBackend(SamplingBackend, Protocol):
    """Complete numerical interface as backend capabilities are added."""

    def gaussian_kl_target_to_candidate(
        self,
        target_mean: np.ndarray,
        target_covariance: np.ndarray,
        candidate_mean: np.ndarray,
        candidate_covariance: np.ndarray,
    ) -> float:
        """Compute ``D_KL(Q_target || P_candidate)``."""
        ...
