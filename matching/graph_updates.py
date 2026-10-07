"""Public graph refinement from validated probe outcomes."""

from graphs.meek import apply_meek_closure
from graphs.pdag import PartiallyDirectedGraph

from .probe_outcomes import ProbeOutcome


def _crosses_boundary(
    source: int,
    target: int,
    node_set: frozenset[int],
) -> bool:
    """Return whether exactly one endpoint belongs to ``node_set``."""
    return (source in node_set) != (target in node_set)


def _validate_direct_evidence(
    graph: PartiallyDirectedGraph,
    outcome: ProbeOutcome,
) -> None:
    """Validate all outcome evidence before constructing a refined snapshot."""
    if not outcome.targets:
        raise ValueError("probe outcome targets must not be empty.")
    if not outcome.targets <= graph.nodes:
        raise ValueError("probe outcome targets must belong to graph nodes.")

    for source, target in sorted(outcome.identified_orientations):
        if source not in graph.nodes or target not in graph.nodes:
            raise ValueError(
                f"identified orientation {source}->{target} has an unknown endpoint."
            )
        if target not in graph.adjacent_nodes(source):
            raise ValueError(
                f"identified orientation {source}->{target} is not an existing "
                "graph adjacency."
            )
        if not (
            _crosses_boundary(source, target, outcome.targets)
            or _crosses_boundary(source, target, outcome.source_targets)
        ):
            raise ValueError(
                f"identified orientation {source}->{target} crosses neither "
                "probe-identifiable boundary."
            )
        if graph.has_directed_edge(target, source):
            raise ValueError(
                f"identified orientation {source}->{target} conflicts with "
                f"existing direction {target}->{source}."
            )


def update_graph_from_probe(
    graph: PartiallyDirectedGraph,
    outcome: ProbeOutcome,
) -> PartiallyDirectedGraph:
    """Return a graph refined only by one public probe outcome.

    The update is transactional: all direct evidence is validated against the
    original immutable graph before a private copy is changed. Already-present
    directions are accepted idempotently, conflicting or missing adjacencies
    are rejected, and Meek R1--R4 closure is applied before the new snapshot is
    returned. This function has no hidden-DAG or coefficient input.
    """
    if not isinstance(graph, PartiallyDirectedGraph):
        raise TypeError("graph must be a PartiallyDirectedGraph.")
    if not isinstance(outcome, ProbeOutcome):
        raise TypeError("outcome must be a ProbeOutcome.")

    _validate_direct_evidence(graph, outcome)
    updated = graph.copy()
    for source, target in sorted(outcome.identified_orientations):
        if updated.has_directed_edge(source, target):
            continue
        try:
            updated = updated.orient(source, target)
        except ValueError as error:
            raise ValueError(
                f"identified orientation {source}->{target} is inconsistent: {error}"
            ) from error

    try:
        return apply_meek_closure(updated)
    except ValueError as error:
        raise ValueError(
            f"probe orientations are inconsistent under Meek closure: {error}"
        ) from error
