"""Public-graph estimation, controlled corruption, and evaluation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np

from graphs.pdag import PartiallyDirectedGraph
from matching.empirical_moments import (
    CovarianceEstimatorSettings,
    EmpiricalMomentsProvider,
)
from matching.structure_learning import (
    EstimatedPCCPDAGProvider,
    PCStructureSettings,
    StructureEstimationMetadata,
)


def canonical_edge(node_a: int, node_b: int) -> tuple[int, int]:
    return (node_a, node_b) if node_a < node_b else (node_b, node_a)


def skeleton(graph: PartiallyDirectedGraph) -> frozenset[tuple[int, int]]:
    return frozenset(
        canonical_edge(source, target)
        for source, target in graph.directed_edges | graph.undirected_edges
    )


def graph_metrics(
    supplied: PartiallyDirectedGraph,
    exact_cpdag: PartiallyDirectedGraph,
    true_dag: nx.DiGraph,
) -> dict[str, object]:
    """Evaluate a public graph without modifying or repairing it."""
    estimated_skeleton = skeleton(supplied)
    true_skeleton = skeleton(exact_cpdag)
    tp = len(estimated_skeleton & true_skeleton)
    fp = len(estimated_skeleton - true_skeleton)
    fn = len(true_skeleton - estimated_skeleton)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    exact_oriented = set(exact_cpdag.directed_edges)
    supplied_oriented = set(supplied.directed_edges)
    orientation_tp = len(exact_oriented & supplied_oriented)
    orientation_fp = len(supplied_oriented - exact_oriented)
    orientation_fn = len(exact_oriented - supplied_oriented)
    orientation_precision = (
        orientation_tp / len(supplied_oriented) if supplied_oriented else 1.0
    )
    orientation_recall = (
        orientation_tp / len(exact_oriented) if exact_oriented else 1.0
    )
    wrong_truth_directions = sum(
        not true_dag.has_edge(source, target)
        for source, target in supplied.directed_edges
    )
    reversed_compelled = sum(
        (target, source) in exact_oriented
        for source, target in supplied.directed_edges
    )
    return {
        "skeleton_shd": fp + fn,
        "adjacency_true_positives": tp,
        "false_positive_adjacencies": fp,
        "false_negative_adjacencies": fn,
        "adjacency_precision": precision,
        "adjacency_recall": recall,
        "adjacency_f1": f1,
        "retains_all_true_adjacencies": fn == 0,
        "orientation_true_positives": orientation_tp,
        "orientation_false_positives": orientation_fp,
        "orientation_false_negatives": orientation_fn,
        "orientation_precision": orientation_precision,
        "orientation_recall": orientation_recall,
        "incorrectly_oriented_compelled_edges": reversed_compelled,
        "initial_wrong_truth_directions": wrong_truth_directions,
        "number_directed_edges": len(supplied.directed_edges),
        "number_undirected_edges": len(supplied.undirected_edges),
    }


def estimate_pc_cpdag(
    samples: np.ndarray, settings: dict[str, object]
) -> tuple[PartiallyDirectedGraph, StructureEstimationMetadata]:
    """Call the repository PC learner on public samples only."""
    provider = EstimatedPCCPDAGProvider(
        PCStructureSettings(
            significance_level=float(settings["significance_level"]),
            max_conditioning_set_size=int(settings["maximum_conditioning_set_size"]),
            minimum_effective_sample_size=int(settings["minimum_effective_sample_size"]),
        ),
        EmpiricalMomentsProvider(
            CovarianceEstimatorSettings(
                estimator=str(settings["covariance_estimator"]),
                diagonal_loading=float(settings["diagonal_loading"]),
                minimum_eigenvalue=float(settings["minimum_eigenvalue"]),
            )
        ),
    )
    estimate = provider.estimate(samples)
    return estimate.graph, estimate.metadata


def metadata_payload(metadata: StructureEstimationMetadata | None) -> dict[str, object] | None:
    if metadata is None:
        return None
    payload = asdict(metadata)
    payload["orientation_conflicts"] = [
        {
            "edge": list(item.edge),
            "proposals": [
                {
                    "direction": list(proposal.direction),
                    "rules": list(proposal.rules),
                    "witnesses": [asdict(witness) for witness in proposal.witnesses],
                }
                for proposal in item.proposals
            ],
        }
        for item in metadata.orientation_conflicts
    ]
    return payload


@dataclass(frozen=True)
class CorruptionResult:
    graph: PartiallyDirectedGraph | None
    status: str
    requested_count: int
    applied_count: int
    affected_edges: tuple[tuple[int, int], ...]
    construction_used_hidden_truth: bool
    input_is_valid_pdag_representation: bool
    note: str


def severity_count(severity: str, eligible_count: int) -> int:
    if severity == "one":
        return 1
    fractions = {"five_percent": 0.05, "ten_percent": 0.10}
    if severity not in fractions:
        raise ValueError(f"unknown corruption severity: {severity}")
    return max(1, round(fractions[severity] * eligible_count))


def _new_graph(
    graph: PartiallyDirectedGraph,
    *,
    directed: set[tuple[int, int]],
    undirected: set[tuple[int, int]],
) -> PartiallyDirectedGraph:
    return PartiallyDirectedGraph(
        nodes=graph.nodes,
        directed_edges=frozenset(directed),
        undirected_edges=frozenset(undirected),
    )


def corrupt_graph(
    exact_cpdag: PartiallyDirectedGraph,
    true_dag: nx.DiGraph,
    *,
    corruption_type: str,
    severity: str,
    seed: int,
) -> CorruptionResult:
    """Construct a deterministic corruption; truth is used only in this labeled arm."""
    rng = np.random.default_rng(seed)
    directed = set(exact_cpdag.directed_edges)
    undirected = set(exact_cpdag.undirected_edges)
    current_skeleton = skeleton(exact_cpdag)

    if corruption_type == "edge_deletion":
        eligible = sorted(current_skeleton)
        requested = severity_count(severity, len(eligible))
        order = rng.permutation(len(eligible)) if eligible else np.array([], dtype=int)
        selected = [eligible[index] for index in order[:requested]]
        for edge in selected:
            undirected.discard(edge)
            directed = {item for item in directed if canonical_edge(*item) != edge}
        if len(selected) < requested:
            return CorruptionResult(None, "NO_VALID_CORRUPTION_AVAILABLE", requested, 0, (), False, False, "too few adjacencies")
        return CorruptionResult(
            _new_graph(exact_cpdag, directed=directed, undirected=undirected),
            "SUCCESS", requested, len(selected), tuple(selected), False, True,
            "adjacencies selected without consulting target support",
        )

    if corruption_type == "edge_addition":
        eligible = [
            (a, b)
            for a in sorted(exact_cpdag.nodes)
            for b in sorted(exact_cpdag.nodes)
            if a < b and (a, b) not in current_skeleton
        ]
        requested = severity_count(severity, len(current_skeleton))
        order = rng.permutation(len(eligible)) if eligible else np.array([], dtype=int)
        selected = [eligible[index] for index in order[:requested]]
        if len(selected) < requested:
            return CorruptionResult(None, "NO_VALID_CORRUPTION_AVAILABLE", requested, 0, (), False, False, "too few nonadjacent pairs")
        undirected.update(selected)
        return CorruptionResult(
            _new_graph(exact_cpdag, directed=directed, undirected=undirected),
            "SUCCESS", requested, len(selected), tuple(selected), False, True,
            "false adjacencies added as undirected PDAG edges",
        )

    if corruption_type != "orientation_error":
        raise ValueError(f"unknown corruption type: {corruption_type}")

    # Orient CPDAG-undirected adjacencies opposite to the hidden simulation DAG.
    # Hidden truth is permitted here solely as controlled-corruption construction.
    eligible = sorted(exact_cpdag.undirected_edges)
    requested = severity_count(severity, max(1, len(current_skeleton)))
    order = rng.permutation(len(eligible)) if eligible else np.array([], dtype=int)
    selected: list[tuple[int, int]] = []
    candidate_directed = set(directed)
    candidate_undirected = set(undirected)
    for index in order:
        a, b = eligible[index]
        wrong = (b, a) if true_dag.has_edge(a, b) else (a, b)
        trial_directed = candidate_directed | {wrong}
        trial_undirected = candidate_undirected - {(a, b)}
        try:
            _new_graph(
                exact_cpdag,
                directed=trial_directed,
                undirected=trial_undirected,
            )
        except ValueError:
            continue
        candidate_directed = trial_directed
        candidate_undirected = trial_undirected
        selected.append(wrong)
        if len(selected) == requested:
            break
    if len(selected) < requested:
        return CorruptionResult(
            None, "NO_VALID_CORRUPTION_AVAILABLE", requested, len(selected),
            tuple(selected), True, False,
            "insufficient cycle-free wrong orientations among CPDAG-undirected edges",
        )
    return CorruptionResult(
        _new_graph(
            exact_cpdag,
            directed=candidate_directed,
            undirected=candidate_undirected,
        ),
        "SUCCESS", requested, len(selected), tuple(selected), True, True,
        "wrong directions use hidden DAG only in explicitly labeled corruption construction",
    )


def graph_payload(graph: PartiallyDirectedGraph) -> dict[str, object]:
    return {
        "nodes": sorted(graph.nodes),
        "directed_edges": [list(edge) for edge in sorted(graph.directed_edges)],
        "undirected_edges": [list(edge) for edge in sorted(graph.undirected_edges)],
    }
