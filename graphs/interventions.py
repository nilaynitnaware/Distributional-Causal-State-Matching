"""Immutable shift-scale intervention representations."""

from dataclasses import dataclass
from typing import Self

import numpy as np

from ._immutability import immutable_array
from .support import DEFAULT_SUPPORT_ATOL, parameter_support_cost, targeted_nodes

INTERVENTION_SEMANTICS = "mechanism_scale_noise_fixed_v1"


@dataclass(frozen=True)
class ShiftScaleIntervention:
    """Node-wise affine changes to structural parent mechanisms.

    For node ``i``, the intervention represents
    ``X_i^I = shift[i] + scale[i] * sum_j B[j, i] X_j^I + epsilon_i``.
    Exogenous noise remains independent standard normal and is never scaled.
    Arrays are defensively copied and backed by immutable buffers.
    """

    shift: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        """Validate and defensively freeze the intervention arrays."""
        arrays = {"shift": self.shift, "scale": self.scale}
        for name, array in arrays.items():
            if not isinstance(array, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray.")
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            if not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array):
                raise TypeError(f"{name} must contain real numeric values.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values.")

        if self.shift.shape != self.scale.shape:
            raise ValueError("shift and scale must have the same shape.")
        if np.any(self.scale <= 0.0):
            raise ValueError("all scales must be strictly positive.")

        object.__setattr__(
            self,
            "shift",
            immutable_array(self.shift, dtype=float),
        )
        object.__setattr__(
            self,
            "scale",
            immutable_array(self.scale, dtype=float),
        )

    @classmethod
    def identity(cls, n_nodes: int) -> Self:
        """Create an intervention that leaves every mechanism unchanged."""
        if isinstance(n_nodes, bool) or not isinstance(n_nodes, int):
            raise TypeError("n_nodes must be an integer.")
        if n_nodes < 0:
            raise ValueError("n_nodes must be nonnegative.")

        return cls(
            shift=np.zeros(n_nodes, dtype=float),
            scale=np.ones(n_nodes, dtype=float),
        )

    def targeted_nodes(
        self,
        support_atol: object = DEFAULT_SUPPORT_ATOL,
    ) -> frozenset[int]:
        """Return nodes active under the centralized support tolerance."""
        return targeted_nodes(
            self.shift,
            self.scale,
            support_atol=support_atol,
        )

    def support_cost(self, support_atol: object = DEFAULT_SUPPORT_ATOL) -> int:
        """Count active shift and log-scale coordinates separately."""
        return parameter_support_cost(
            self.shift,
            self.scale,
            support_atol=support_atol,
        )

    @property
    def targets(self) -> frozenset[int]:
        """Return nodes active under the default support tolerance."""
        return self.targeted_nodes(DEFAULT_SUPPORT_ATOL)
