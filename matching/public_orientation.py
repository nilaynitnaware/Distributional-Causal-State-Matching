"""Graph orientation using only public source classifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from graphs.meek import apply_meek_closure
from graphs.pdag import DirectedEdge, PartiallyDirectedGraph

from .source_inference import SourceClassification


@dataclass(frozen=True)
class OrientationEvidence:
    """Provenance for one direction learned from a public probe."""

    edge: DirectedEdge
    provenance: str
    probe_id: str
    background_id: str


@dataclass(frozen=True)
class PublicOrientationUpdate:
    """Transactional graph update and separated direct/Meek provenance."""

    graph: PartiallyDirectedGraph
    direct_evidence: tuple[OrientationEvidence, ...]
    meek_evidence: tuple[OrientationEvidence, ...]


def orient_from_public_source_classifications(
    graph: PartiallyDirectedGraph,
    *,
    targets: frozenset[int],
    classifications: Mapping[int, SourceClassification],
    probe_id: str,
    background_id: str,
) -> PublicOrientationUpdate:
    """Apply only the co-targeted source-to-non-source adjacency rule."""
    if not isinstance(graph, PartiallyDirectedGraph):
        raise TypeError("graph must be a PartiallyDirectedGraph.")
    if not isinstance(targets, frozenset) or not targets:
        raise ValueError("targets must be a nonempty frozenset.")
    if not targets <= graph.nodes:
        raise ValueError("targets must belong to graph.")
    if set(classifications) != set(targets):
        raise ValueError("classifications must contain exactly all probe targets.")
    if any(
        value not in {"source", "non_source", "inconclusive"}
        for value in classifications.values()
    ):
        raise ValueError("unsupported source classification.")
    for name, value in (("probe_id", probe_id), ("background_id", background_id)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty string.")

    inferred: list[DirectedEdge] = []
    for node_a in sorted(targets):
        for node_b in sorted(target for target in targets if target > node_a):
            class_a = classifications[node_a]
            class_b = classifications[node_b]
            if {class_a, class_b} != {"source", "non_source"}:
                continue
            source, target = (
                (node_a, node_b) if class_a == "source" else (node_b, node_a)
            )
            if graph.has_directed_edge(target, source):
                raise ValueError(
                    f"public probe direction {source}->{target} conflicts with "
                    f"existing direction {target}->{source}."
                )
            if graph.has_directed_edge(source, target):
                continue
            if graph.has_undirected_edge(source, target):
                inferred.append((source, target))

    updated = graph.copy()
    direct_evidence: list[OrientationEvidence] = []
    for source, target in inferred:
        updated = updated.orient(source, target)
        direct_evidence.append(
            OrientationEvidence(
                edge=(source, target),
                provenance="direct_public_source_rule",
                probe_id=probe_id,
                background_id=background_id,
            )
        )
    closed = apply_meek_closure(updated)
    direct_edges = set(updated.directed_edges)
    meek_evidence = tuple(
        OrientationEvidence(
            edge=edge,
            provenance="meek_r1_r4",
            probe_id=probe_id,
            background_id=background_id,
        )
        for edge in sorted(closed.directed_edges - direct_edges)
    )
    return PublicOrientationUpdate(
        graph=closed,
        direct_evidence=tuple(direct_evidence),
        meek_evidence=meek_evidence,
    )
