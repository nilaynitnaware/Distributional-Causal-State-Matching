"""Hidden exact probe outcomes for graph-level simulation experiments."""

from dataclasses import dataclass
from numbers import Integral

import networkx as nx
import numpy as np

from graphs.interventions import ShiftScaleIntervention
from matching.probe_outcomes import ProbeOutcome
from matching.probes import ShiftProbe


def _frozen_indexed_dag(true_dag: object) -> nx.DiGraph:
    """Validate, normalize, defensively copy, and freeze a simulation DAG."""
    if not isinstance(true_dag, nx.DiGraph) or true_dag.is_multigraph():
        raise TypeError("true_dag must be a simple networkx.DiGraph.")
    if not nx.is_directed_acyclic_graph(true_dag):
        raise ValueError("true_dag must be acyclic.")

    n_nodes = true_dag.number_of_nodes()
    if any(
        isinstance(node, bool) or not isinstance(node, Integral)
        for node in true_dag.nodes
    ) or {int(node) for node in true_dag.nodes} != set(range(n_nodes)):
        raise ValueError("true_dag nodes must be labeled 0, ..., n_nodes - 1.")

    copied = nx.DiGraph()
    copied.add_nodes_from(range(n_nodes))
    copied.add_edges_from(
        (int(source), int(target)) for source, target in true_dag.edges
    )
    return nx.freeze(copied)


def _crosses_boundary(
    source: int,
    target: int,
    node_set: frozenset[int],
) -> bool:
    """Return whether exactly one edge endpoint belongs to ``node_set``."""
    return (source in node_set) != (target in node_set)


@dataclass(frozen=True)
class NoiselessOracleProbeEnvironment:
    """Use hidden truth to return only idealized probe-identifiable facts.

    The true DAG remains confined to this explicitly oracle-side object.
    ``execute_probe`` returns neither the DAG nor coefficients, only the
    source subset and true directions crossing the full-target or source-target
    intervention boundaries.
    """

    true_dag: nx.DiGraph

    def __post_init__(self) -> None:
        """Defensively freeze the hidden simulation DAG."""
        object.__setattr__(self, "true_dag", _frozen_indexed_dag(self.true_dag))

    def outcome_from_executed_shift(
        self,
        public_candidate: ShiftScaleIntervention,
        probe: ShiftProbe,
        *,
        exact_interventional_mean: np.ndarray,
        exact_interventional_covariance: np.ndarray,
        background_id: str,
    ) -> ProbeOutcome:
        """Attach ideal oracle structural facts to one already executed shift.

        This adapter performs no second environment execution. Its moment
        arguments are the public result of the physical shift and are checked
        only for dimensional/provenance consistency; they do not determine the
        returned directions. The historical idealized
        orientation mathematics remains in ``execute_probe``.
        """
        n_nodes = self.true_dag.number_of_nodes()
        if (
            not isinstance(exact_interventional_mean, np.ndarray)
            or exact_interventional_mean.shape != (n_nodes,)
            or not np.all(np.isfinite(exact_interventional_mean))
        ):
            raise ValueError("exact interventional mean has invalid dimensions")
        if (
            not isinstance(exact_interventional_covariance, np.ndarray)
            or exact_interventional_covariance.shape != (n_nodes, n_nodes)
            or not np.all(np.isfinite(exact_interventional_covariance))
        ):
            raise ValueError("exact interventional covariance has invalid dimensions")
        if not isinstance(background_id, str) or not background_id:
            raise ValueError("background_id must be a nonempty string")
        return self._derive_outcome(public_candidate, probe)

    def execute_probe(
        self,
        public_candidate: ShiftScaleIntervention,
        probe: ShiftProbe,
    ) -> ProbeOutcome:
        """Return the historical noiseless public structural outcome."""
        return self._derive_outcome(public_candidate, probe)

    def _derive_outcome(
        self,
        public_candidate: ShiftScaleIntervention,
        probe: ShiftProbe,
    ) -> ProbeOutcome:
        """Return the noiseless source subset and identifiable orientations."""
        if not isinstance(public_candidate, ShiftScaleIntervention):
            raise TypeError("public_candidate must be a ShiftScaleIntervention.")
        if not isinstance(probe, ShiftProbe):
            raise TypeError("probe must be a ShiftProbe.")

        n_nodes = self.true_dag.number_of_nodes()
        if public_candidate.shift.shape != (n_nodes,):
            raise ValueError(
                "public candidate dimension must agree with the hidden DAG."
            )
        if probe.shift_increment.shape != (n_nodes,):
            raise ValueError("probe dimension must agree with the hidden DAG.")
        if not probe.targets:
            raise ValueError("probe must target at least one node.")

        # Construct and discard the temporary intervention to enforce
        # (delta_probe, s_probe) = (delta_current + alpha, s_current).
        probe.apply_to(
            public_candidate,
            max_probe_targets=len(probe.targets),
        )

        source_targets = frozenset(
            target
            for target in probe.targets
            if not (
                (probe.targets - {target})
                & {int(node) for node in nx.ancestors(self.true_dag, target)}
            )
        )
        identified_orientations = frozenset(
            (int(source), int(target))
            for source, target in self.true_dag.edges
            if _crosses_boundary(source, target, probe.targets)
            or _crosses_boundary(source, target, source_targets)
        )

        return ProbeOutcome(
            targets=probe.targets,
            source_targets=source_targets,
            identified_orientations=identified_orientations,
            samples=None,
            trial_cost=1,
        )
