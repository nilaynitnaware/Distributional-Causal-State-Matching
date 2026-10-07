"""Hidden-free structural evidence from exact singleton-shift responses."""

from __future__ import annotations

from typing import Literal

import numpy as np

from graphs.pdag import PartiallyDirectedGraph
from matching.probe_outcomes import ProbeOutcome
from matching.probes import ShiftProbe

PublicProbeAssumption = Literal["conservative", "total_effect_faithful"]


def construct_public_singleton_shift_outcome(
    *,
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    exact_interventional_mean: np.ndarray,
    exact_interventional_covariance: np.ndarray,
    probe: ShiftProbe,
    current_graph: PartiallyDirectedGraph,
    assumption: PublicProbeAssumption = "conservative",
    effect_atol: float = 1e-12,
    covariance_atol: float = 1e-12,
    covariance_rtol: float = 1e-10,
) -> ProbeOutcome:
    """Construct public orientation evidence from one exact singleton shift.

    The conservative rule orients only ``i -> j`` when an undirected neighbor
    ``j`` has a publicly nonzero response to the shift at ``i``.  The faithful
    audit rule additionally interprets a zero response as ``j -> i``; that rule
    requires total-effect faithfulness/separation and is not the production
    default.
    """
    if assumption not in {"conservative", "total_effect_faithful"}:
        raise ValueError("unsupported public probe assumption")
    if not isinstance(probe, ShiftProbe) or len(probe.targets) != 1:
        raise ValueError("public constructor requires exactly one shift target")
    if not isinstance(current_graph, PartiallyDirectedGraph):
        raise TypeError("current_graph must be a PartiallyDirectedGraph")
    arrays = {
        "source_mean": source_mean,
        "source_covariance": source_covariance,
        "exact_interventional_mean": exact_interventional_mean,
        "exact_interventional_covariance": exact_interventional_covariance,
    }
    if any(not isinstance(value, np.ndarray) for value in arrays.values()):
        raise TypeError("all moment inputs must be numpy arrays")
    n = source_mean.size
    if (
        source_mean.shape != (n,)
        or exact_interventional_mean.shape != (n,)
        or source_covariance.shape != (n, n)
        or exact_interventional_covariance.shape != (n, n)
        or probe.shift_increment.shape != (n,)
        or current_graph.nodes != frozenset(range(n))
    ):
        raise ValueError("public singleton-shift inputs have incompatible dimensions")
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("public singleton-shift moments must be finite")
    for name, value in {
        "effect_atol": effect_atol,
        "covariance_atol": covariance_atol,
        "covariance_rtol": covariance_rtol,
    }.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not np.allclose(
        exact_interventional_covariance,
        source_covariance,
        atol=covariance_atol,
        rtol=covariance_rtol,
    ):
        raise ValueError("pure deterministic shift unexpectedly changed covariance")

    target = next(iter(probe.targets))
    shift = float(probe.shift_increment[target])
    if shift == 0.0 or not np.isfinite(shift):
        raise ValueError("singleton shift must be finite and nonzero")
    total_effect_column = (exact_interventional_mean - source_mean) / shift
    orientations: set[tuple[int, int]] = set()
    for neighbor in sorted(current_graph.undirected_neighbors_of(target)):
        if abs(float(total_effect_column[neighbor])) > effect_atol:
            orientations.add((target, neighbor))
        elif assumption == "total_effect_faithful":
            orientations.add((neighbor, target))

    return ProbeOutcome(
        targets=probe.targets,
        source_targets=probe.targets,
        identified_orientations=frozenset(orientations),
        samples=None,
        trial_cost=1,
    )
