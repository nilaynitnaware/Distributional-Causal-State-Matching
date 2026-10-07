"""Public outcomes and execution interface for graph-refinement probes."""

from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, runtime_checkable

import numpy as np

from graphs._immutability import immutable_array
from graphs.interventions import ShiftScaleIntervention

from .probes import ShiftProbe

DirectedEdge = tuple[int, int]


def _normalize_node_set(value: object, *, name: str) -> frozenset[int]:
    """Validate and normalize one immutable node set."""
    if not isinstance(value, frozenset):
        raise TypeError(f"{name} must be a frozenset of node indices.")
    if any(isinstance(node, bool) or not isinstance(node, Integral) for node in value):
        raise TypeError(f"{name} must contain integer node indices.")
    normalized = frozenset(int(node) for node in value)
    if any(node < 0 for node in normalized):
        raise ValueError(f"{name} must contain nonnegative node indices.")
    return normalized


def _normalize_orientations(value: object) -> frozenset[DirectedEdge]:
    """Validate and normalize immutable directed-edge evidence."""
    if not isinstance(value, frozenset):
        raise TypeError(
            "identified_orientations must be a frozenset of directed edges."
        )

    normalized: set[DirectedEdge] = set()
    for edge in value:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise TypeError("identified_orientations must contain two-node tuples.")
        source, target = edge
        if (
            isinstance(source, bool)
            or not isinstance(source, Integral)
            or isinstance(target, bool)
            or not isinstance(target, Integral)
        ):
            raise TypeError(
                "identified orientation endpoints must be integer node indices."
            )
        normalized_edge = (int(source), int(target))
        if normalized_edge[0] < 0 or normalized_edge[1] < 0:
            raise ValueError("identified orientation endpoints must be nonnegative.")
        if normalized_edge[0] == normalized_edge[1]:
            raise ValueError("identified orientations must not contain self-loops.")
        if (normalized_edge[1], normalized_edge[0]) in normalized:
            raise ValueError(
                "identified orientations must not contain opposing directions."
            )
        normalized.add(normalized_edge)
    return frozenset(normalized)


@dataclass(frozen=True)
class ProbeOutcome:
    """Only the idealized information publicly identified by one probe."""

    targets: frozenset[int]
    source_targets: frozenset[int]
    identified_orientations: frozenset[DirectedEdge]
    samples: np.ndarray | None
    trial_cost: int

    def __post_init__(self) -> None:
        """Validate and defensively freeze every public outcome field."""
        targets = _normalize_node_set(self.targets, name="targets")
        source_targets = _normalize_node_set(
            self.source_targets,
            name="source_targets",
        )
        if not source_targets <= targets:
            raise ValueError("source_targets must be a subset of targets.")
        orientations = _normalize_orientations(self.identified_orientations)

        if self.samples is not None:
            if not isinstance(self.samples, np.ndarray):
                raise TypeError("samples must be a numpy.ndarray or None.")
            if self.samples.ndim != 2:
                raise ValueError("samples must be two-dimensional.")
            if not np.issubdtype(
                self.samples.dtype,
                np.number,
            ) or not np.isrealobj(self.samples):
                raise TypeError("samples must contain real numeric values.")
            if not np.all(np.isfinite(self.samples)):
                raise ValueError("samples must contain only finite values.")
            object.__setattr__(
                self,
                "samples",
                immutable_array(self.samples, dtype=float),
            )

        if isinstance(self.trial_cost, bool) or not isinstance(
            self.trial_cost,
            int,
        ):
            raise TypeError("trial_cost must be an integer.")
        if self.trial_cost != 1:
            raise ValueError("trial_cost must equal one for one executed probe.")

        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "source_targets", source_targets)
        object.__setattr__(self, "identified_orientations", orientations)


@runtime_checkable
class ProbeEnvironment(Protocol):
    """Hidden execution boundary that returns only a public probe outcome."""

    def execute_probe(
        self,
        public_candidate: ShiftScaleIntervention,
        probe: ShiftProbe,
    ) -> ProbeOutcome:
        """Execute one temporary probe without exposing hidden simulation state."""
        ...
