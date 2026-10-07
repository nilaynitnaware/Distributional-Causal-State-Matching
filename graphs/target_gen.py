"""Gaussian target containers and realizable-target construction."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from ._immutability import immutable_array
from .interventions import ShiftScaleIntervention
from .scm import LinearGaussianSCM


@dataclass(frozen=True)
class GaussianTarget:
    """A Gaussian target distribution with optional evaluation-only metadata.

    ``mean`` and ``covariance`` are the algorithm-facing target. The hidden
    ``generating_intervention`` is retained only for simulation evaluation and
    must not be supplied to non-oracle matching methods.
    """

    mean: np.ndarray
    covariance: np.ndarray
    generating_intervention: ShiftScaleIntervention | None
    is_realisable: bool
    public_metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        """Validate and defensively freeze target data."""
        if not isinstance(self.mean, np.ndarray):
            raise TypeError("mean must be a numpy.ndarray.")
        if not isinstance(self.covariance, np.ndarray):
            raise TypeError("covariance must be a numpy.ndarray.")
        if self.mean.ndim != 1:
            raise ValueError("mean must be one-dimensional.")
        if self.covariance.ndim != 2 or (
            self.covariance.shape[0] != self.covariance.shape[1]
        ):
            raise ValueError("covariance must be a square matrix.")
        if self.covariance.shape != (self.mean.size, self.mean.size):
            raise ValueError("mean and covariance dimensions must agree.")

        arrays = {"mean": self.mean, "covariance": self.covariance}
        for name, array in arrays.items():
            if not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array):
                raise TypeError(f"{name} must contain real numeric values.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values.")

        if not np.allclose(
            self.covariance,
            self.covariance.T,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("covariance must be symmetric.")
        if self.mean.size:
            try:
                np.linalg.cholesky(self.covariance)
            except np.linalg.LinAlgError as error:
                raise ValueError("covariance must be positive definite.") from error

        if not isinstance(self.is_realisable, bool):
            raise TypeError("is_realisable must be a boolean.")
        if self.generating_intervention is not None and not isinstance(
            self.generating_intervention, ShiftScaleIntervention
        ):
            raise TypeError(
                "generating_intervention must be a ShiftScaleIntervention or None."
            )
        if self.is_realisable and self.generating_intervention is None:
            raise ValueError(
                "a realizable target must retain its generating intervention."
            )
        if (
            self.generating_intervention is not None
            and self.generating_intervention.shift.shape != self.mean.shape
        ):
            raise ValueError(
                "generating intervention dimension must agree with target dimension."
            )

        object.__setattr__(self, "mean", immutable_array(self.mean, dtype=float))
        object.__setattr__(
            self,
            "covariance",
            immutable_array(self.covariance, dtype=float),
        )
        if not isinstance(self.public_metadata, Mapping):
            raise TypeError("public_metadata must be a mapping.")
        if any(not isinstance(key, str) or not key for key in self.public_metadata):
            raise ValueError("public_metadata keys must be nonempty strings.")
        object.__setattr__(
            self, "public_metadata", MappingProxyType(dict(self.public_metadata))
        )


def generate_realisable_target(
    scm: LinearGaussianSCM,
    generating_intervention: ShiftScaleIntervention,
) -> GaussianTarget:
    """Construct a realizable target from a caller-supplied hidden intervention.

    Sampling a prior over hidden interventions is intentionally outside this
    function because no target-node, shift, or scale distribution has yet been
    specified. This function performs the defined part of target generation:
    calculating exact moments and retaining the intervention for evaluation.

    Args:
        scm: Hidden simulation environment.
        generating_intervention: Already sampled hidden intervention.

    Returns:
        An immutable realizable Gaussian target.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If the intervention dimension disagrees with the SCM.
    """
    if not isinstance(scm, LinearGaussianSCM):
        raise TypeError("scm must be a LinearGaussianSCM.")
    if not isinstance(generating_intervention, ShiftScaleIntervention):
        raise TypeError("generating_intervention must be a ShiftScaleIntervention.")

    scm.validate_target_intervention(generating_intervention)
    mean, covariance = scm.interventional_moments(generating_intervention)
    return GaussianTarget(
        mean=mean,
        covariance=covariance,
        generating_intervention=generating_intervention,
        is_realisable=True,
    )
