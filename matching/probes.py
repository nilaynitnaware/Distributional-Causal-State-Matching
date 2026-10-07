"""Validated temporary pure-shift probe semantics."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from graphs._immutability import immutable_array
from graphs.interventions import ShiftScaleIntervention


def _validate_nonempty_reason(reason: object) -> None:
    """Validate a descriptive probe reason."""
    if not isinstance(reason, str):
        raise TypeError("reason must be a string.")
    if not reason.strip():
        raise ValueError("reason must not be empty.")


@dataclass(frozen=True, init=False)
class ShiftProbe:
    """A temporary additive shift applied on top of an accepted candidate.

    ``targets`` is exactly the mathematical nonzero support of
    ``shift_increment``. The initializer also accepts the established
    ``ProbeProposal(increment, reason)`` call shape and derives its targets.
    """

    shift_increment: np.ndarray
    targets: frozenset[int]
    reason: str

    def __init__(
        self,
        shift_increment: np.ndarray,
        targets: frozenset[int] | str | None = None,
        reason: str | None = None,
    ) -> None:
        """Support explicit targets and the legacy derived-target call form."""
        if isinstance(targets, str) and reason is None:
            reason = targets
            targets = None
        if targets is None:
            targets = (
                frozenset(int(node) for node in np.flatnonzero(shift_increment != 0.0))
                if isinstance(shift_increment, np.ndarray)
                else frozenset()
            )

        object.__setattr__(self, "shift_increment", shift_increment)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "reason", reason)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate, normalize, and irreversibly freeze the probe."""
        if not isinstance(self.shift_increment, np.ndarray):
            raise TypeError("shift_increment must be a numpy.ndarray.")
        if self.shift_increment.ndim != 1:
            raise ValueError("shift_increment must be one-dimensional.")
        if not np.issubdtype(
            self.shift_increment.dtype,
            np.number,
        ) or not np.isrealobj(self.shift_increment):
            raise TypeError("shift_increment must contain real numeric values.")
        if not np.all(np.isfinite(self.shift_increment)):
            raise ValueError("shift_increment must contain only finite values.")
        if not isinstance(self.targets, frozenset):
            raise TypeError("targets must be a frozenset of node indices.")
        if any(
            isinstance(node, bool) or not isinstance(node, Integral)
            for node in self.targets
        ):
            raise TypeError("targets must contain integer node indices.")

        normalized_targets = frozenset(int(node) for node in self.targets)
        if any(
            node < 0 or node >= self.shift_increment.size for node in normalized_targets
        ):
            raise ValueError("targets must fit the shift increment dimension.")
        expected_targets = frozenset(
            int(node) for node in np.flatnonzero(self.shift_increment != 0.0)
        )
        if normalized_targets != expected_targets:
            raise ValueError("targets must exactly match the nonzero shift increments.")

        _validate_nonempty_reason(self.reason)
        object.__setattr__(
            self,
            "shift_increment",
            immutable_array(self.shift_increment, dtype=float),
        )
        object.__setattr__(self, "targets", normalized_targets)

    def apply_to(
        self,
        current_candidate: ShiftScaleIntervention,
        *,
        max_probe_targets: int,
    ) -> ShiftScaleIntervention:
        """Return ``(delta_current + alpha, s_current)`` for this probe.

        The returned intervention is a new immutable snapshot. The accepted
        candidate is neither mutated nor retained by this probe.
        """
        if not isinstance(current_candidate, ShiftScaleIntervention):
            raise TypeError("current_candidate must be a ShiftScaleIntervention.")
        if isinstance(max_probe_targets, bool) or not isinstance(
            max_probe_targets,
            int,
        ):
            raise TypeError("max_probe_targets must be an integer.")
        if max_probe_targets <= 0:
            raise ValueError("max_probe_targets must be positive.")
        if self.shift_increment.shape != current_candidate.shift.shape:
            raise ValueError("probe dimension must agree with the current candidate.")
        if not 0 < len(self.targets) <= max_probe_targets:
            raise ValueError(
                "probe target count must be positive and at most max_probe_targets."
            )

        temporary = ShiftScaleIntervention(
            shift=current_candidate.shift + self.shift_increment,
            scale=current_candidate.scale,
        )
        if not np.array_equal(temporary.scale, current_candidate.scale):
            raise RuntimeError("a pure-shift probe must not modify scale.")
        return temporary


# Backwards-compatible public name established before Step 9D.
ProbeProposal = ShiftProbe
