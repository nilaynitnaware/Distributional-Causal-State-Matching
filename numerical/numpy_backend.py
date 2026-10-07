"""Canonical NumPy implementation of the numerical backend contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from metrics.gaussian_kl import gaussian_kl_target_to_candidate

if TYPE_CHECKING:
    from graphs.interventions import ShiftScaleIntervention
    from graphs.scm import LinearGaussianSCM


class NumPyBackend:
    """Delegate numerical work to the existing tested NumPy implementation."""

    name: Literal["numpy"] = "numpy"
    device: Literal["cpu"] = "cpu"
    dtype: Literal["float64"] = "float64"

    def observational_moments(
        self,
        scm: LinearGaussianSCM,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reuse the SCM's canonical float64 observational moments."""
        return scm.observational_moments()

    def interventional_moments(
        self,
        scm: LinearGaussianSCM,
        intervention: ShiftScaleIntervention,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reuse the SCM's canonical float64 moment calculation."""
        return scm.interventional_moments(intervention)

    def sample_observational(
        self,
        scm: LinearGaussianSCM,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Reuse the canonical topological NumPy observational sampler."""
        return scm.sample_observational(n_samples, rng)

    def sample_interventional(
        self,
        scm: LinearGaussianSCM,
        intervention: ShiftScaleIntervention,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Reuse the canonical topological NumPy interventional sampler."""
        return scm.sample_interventional(intervention, n_samples, rng)

    def gaussian_kl_target_to_candidate(
        self,
        target_mean: np.ndarray,
        target_covariance: np.ndarray,
        candidate_mean: np.ndarray,
        candidate_covariance: np.ndarray,
    ) -> float:
        """Compute target-to-candidate KL with the canonical NumPy helper."""
        return gaussian_kl_target_to_candidate(
            target_mean=target_mean,
            target_covariance=target_covariance,
            candidate_mean=candidate_mean,
            candidate_covariance=candidate_covariance,
        )


NUMPY_BACKEND = NumPyBackend()
