"""Restricted population matching-aware acquisition on collider-free trees.

The complete observational MEC of a collider-free tree is represented by its
possible roots.  This module deliberately keeps stopping separate from query
selection: a policy can be replayed to singleton-DAG recovery and its MATCH
trajectory is the certified prefix of exactly the same path.

Certificate searches are scoped to one source/target/budget instance.  A cap
or incomplete tied-family geometry is reported as ``inconclusive`` and is
never treated as a terminal state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from time import perf_counter
from typing import Literal, Mapping

import networkx as nx
import numpy as np

from research.matching_query_complexity import (
    CandidateDAG,
    CommonCommandCertificate,
    DagId,
    OptimalActionSet,
    candidate_map,
    find_common_optimal_command,
    infer_outward_tree_root,
    outward_tree_dag,
    singleton_query_partition,
    tree_root_partition,
    unshielded_colliders,
)

CertificateStatus = Literal["certified", "absent", "inconclusive"]
PolicyName = Literal[
    "one_step_matching_aware",
    "balance",
    "first_unresolved",
    "max_undirected_degree",
]


@dataclass(frozen=True)
class CertificateSearchResult:
    status: CertificateStatus
    certificate: CommonCommandCertificate | None
    tied_family_combinations: int
    seconds: float
    structural_candidates_visited: int = 0
    direct_kl_evaluations: int = 0
    reason: str | None = None


@dataclass(frozen=True)
class SelectionResult:
    node: int
    score: tuple[int, ...]
    partition: tuple[frozenset[DagId], ...]
    child_statuses: tuple[CertificateStatus, ...] = ()


@dataclass(frozen=True)
class TreeCandidateClass:
    tree: nx.Graph
    candidates: Mapping[DagId, CandidateDAG]
    roots_by_id: Mapping[DagId, int]
    ids_by_root: Mapping[int, DagId]

    @property
    def initial_state(self) -> frozenset[DagId]:
        return frozenset(self.candidates)


def _candidate(dag: nx.DiGraph) -> CandidateDAG:
    nodes = tuple(sorted(int(node) for node in dag.nodes))
    if nodes != tuple(range(len(nodes))):
        raise ValueError("tree nodes must be labeled 0, ..., n - 1")
    dag_id = tuple(sorted((int(left), int(right)) for left, right in dag.edges))
    return CandidateDAG(
        id=dag_id,
        dag=nx.freeze(nx.DiGraph(dag)),
        parent_sets=tuple(
            tuple(sorted(int(parent) for parent in dag.predecessors(node)))
            for node in nodes
        ),
        colliders=unshielded_colliders(dag),
    )


def enumerate_tree_root_class(tree: nx.Graph) -> TreeCandidateClass:
    """Enumerate the complete collider-free tree MEC, one DAG per root."""
    if not nx.is_tree(tree):
        raise ValueError("candidate root representation requires a tree")
    nodes = tuple(sorted(int(node) for node in tree.nodes))
    if nodes != tuple(range(len(nodes))):
        raise ValueError("tree nodes must be labeled 0, ..., n - 1")
    candidates_tuple = tuple(_candidate(outward_tree_dag(tree, root)) for root in nodes)
    candidates = candidate_map(candidates_tuple)
    if len(candidates) != len(nodes):
        raise RuntimeError("different tree roots did not produce distinct DAGs")
    roots = {dag_id: infer_outward_tree_root(item) for dag_id, item in candidates.items()}
    ids_by_root = {root: dag_id for dag_id, root in roots.items()}
    if set(ids_by_root) != set(nodes):
        raise RuntimeError("tree root enumeration is incomplete")
    return TreeCandidateClass(
        tree=nx.freeze(nx.Graph(tree)),
        candidates=candidates,
        roots_by_id=roots,
        ids_by_root=ids_by_root,
    )


class CertificateCache:
    """Instance-local capped cache for direct-KL common-command searches."""

    def __init__(
        self,
        action_sets: Mapping[DagId, OptimalActionSet],
        *,
        maximum_uncached_checks: int = 2_000,
        maximum_tied_family_combinations: int = 10_000,
    ) -> None:
        if maximum_uncached_checks < 0 or maximum_tied_family_combinations < 0:
            raise ValueError("certificate caps must be nonnegative")
        self._action_sets = action_sets
        self.maximum_uncached_checks = maximum_uncached_checks
        self.maximum_tied_family_combinations = maximum_tied_family_combinations
        self._cache: dict[frozenset[DagId], CertificateSearchResult] = {}
        self.uncached_checks = 0
        self.tied_family_combinations = 0
        self.structural_candidates_visited = 0
        self.direct_kl_evaluations = 0
        self.cache_hits = 0
        self.outcome_counts = {"certified": 0, "absent": 0, "inconclusive": 0}
        self.seconds = 0.0
        self.cap_hits: list[dict[str, object]] = []

    @property
    def cache(self) -> Mapping[frozenset[DagId], CertificateSearchResult]:
        return self._cache

    def check(self, state: frozenset[DagId]) -> CertificateSearchResult:
        if not state:
            raise ValueError("cannot certify an empty candidate state")
        cached = self._cache.get(state)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.uncached_checks >= self.maximum_uncached_checks:
            result = CertificateSearchResult(
                status="inconclusive",
                certificate=None,
                tied_family_combinations=0,
                seconds=0.0,
                reason="uncached_certificate_check_cap",
            )
            self._cache[state] = result
            self.outcome_counts[result.status] += 1
            self.cap_hits.append({"kind": result.reason, "state_size": len(state)})
            return result

        family_count = prod(
            len(self._action_sets[dag_id].command_families) for dag_id in state
        )
        remaining = (
            self.maximum_tied_family_combinations - self.tied_family_combinations
        )
        if family_count > remaining:
            result = CertificateSearchResult(
                status="inconclusive",
                certificate=None,
                tied_family_combinations=0,
                seconds=0.0,
                reason="tied_family_combination_cap",
            )
            self._cache[state] = result
            self.outcome_counts[result.status] += 1
            self.cap_hits.append(
                {
                    "kind": result.reason,
                    "state_size": len(state),
                    "required_upper_bound": int(family_count),
                    "remaining": int(remaining),
                }
            )
            return result

        started = perf_counter()
        instrumentation: dict[str, int] = {}
        certificate = find_common_optimal_command(
            state, self._action_sets, instrumentation=instrumentation
        )
        elapsed = perf_counter() - started
        self.uncached_checks += 1
        self.tied_family_combinations += int(family_count)
        visited = instrumentation.get("structural_candidates_visited", 0)
        direct_kl = instrumentation.get("direct_kl_evaluations", 0)
        self.structural_candidates_visited += visited
        self.direct_kl_evaluations += direct_kl
        self.seconds += elapsed
        result = CertificateSearchResult(
            status="certified" if certificate is not None else "absent",
            certificate=certificate,
            tied_family_combinations=int(family_count),
            seconds=elapsed,
            structural_candidates_visited=visited,
            direct_kl_evaluations=direct_kl,
        )
        self._cache[state] = result
        self.outcome_counts[result.status] += 1
        return result


def response_partition(
    root_class: TreeCandidateClass,
    state: frozenset[DagId],
    node: int,
) -> tuple[frozenset[DagId], ...]:
    """Exact singleton-query response cells, cross-checked by parent signature."""
    specialized = tree_root_partition(
        root_class.tree, root_class.roots_by_id, state, int(node)
    )
    generic = singleton_query_partition(state, root_class.candidates, int(node))
    normalized = lambda cells: {frozenset(cell) for cell in cells}
    if normalized(specialized) != normalized(generic):
        raise RuntimeError("tree root components disagree with parent signatures")
    return specialized


def _eligible_partitions(
    root_class: TreeCandidateClass,
    state: frozenset[DagId],
    queried_nodes: frozenset[int],
) -> list[tuple[int, tuple[frozenset[DagId], ...]]]:
    output = []
    for node in sorted(root_class.tree.nodes):
        if node in queried_nodes:
            continue
        partition = response_partition(root_class, state, int(node))
        if len(partition) >= 2:
            output.append((int(node), partition))
    if not output:
        raise RuntimeError("non-singleton root state has no informative unqueried node")
    return output


def select_one_step(
    root_class: TreeCandidateClass,
    state: frozenset[DagId],
    queried_nodes: frozenset[int],
    certificates: CertificateCache,
) -> SelectionResult:
    """Probe ordering: nonterminal size, raw size, certified cells, node ID.

    Inconclusive children are conservatively nonterminal.  This function is
    defined for every non-singleton state, even when ``state`` itself is
    already a certified MATCH terminal.
    """
    scored: list[tuple[tuple[int, ...], int, tuple[frozenset[DagId], ...], tuple[CertificateStatus, ...]]] = []
    for node, partition in _eligible_partitions(root_class, state, queried_nodes):
        statuses = tuple(certificates.check(child).status for child in partition)
        worst_nonterminal = max(
            (
                len(child)
                for child, status in zip(partition, statuses, strict=True)
                if status != "certified"
            ),
            default=0,
        )
        score = (
            worst_nonterminal,
            max(len(child) for child in partition),
            -sum(status == "certified" for status in statuses),
            node,
        )
        scored.append((score, node, partition, statuses))
    score, node, partition, statuses = min(scored, key=lambda item: item[0])
    return SelectionResult(node, score, partition, statuses)


def select_balance(
    root_class: TreeCandidateClass,
    state: frozenset[DagId],
    queried_nodes: frozenset[int],
) -> SelectionResult:
    """Minimize the largest raw response cell, breaking ties by node ID."""
    scored = []
    for node, partition in _eligible_partitions(root_class, state, queried_nodes):
        score = (max(len(child) for child in partition), node)
        scored.append((score, node, partition))
    score, node, partition = min(scored, key=lambda item: item[0])
    return SelectionResult(node, score, partition)


def _variable_parent_nodes(
    root_class: TreeCandidateClass, state: frozenset[DagId]
) -> frozenset[int]:
    return frozenset(
        node
        for node in root_class.tree.nodes
        if len(
            {
                root_class.candidates[dag_id].parent_sets[int(node)]
                for dag_id in state
            }
        )
        > 1
    )


def select_first_unresolved(
    root_class: TreeCandidateClass,
    state: frozenset[DagId],
    queried_nodes: frozenset[int],
) -> SelectionResult:
    eligible = {
        node: partition
        for node, partition in _eligible_partitions(root_class, state, queried_nodes)
    }
    variable = _variable_parent_nodes(root_class, state)
    node = min(set(eligible) & set(variable))
    return SelectionResult(node, (node,), eligible[node])


def select_max_undirected_degree(
    root_class: TreeCandidateClass,
    state: frozenset[DagId],
    queried_nodes: frozenset[int],
) -> SelectionResult:
    eligible = {
        node: partition
        for node, partition in _eligible_partitions(root_class, state, queried_nodes)
    }
    variable = _variable_parent_nodes(root_class, state)
    degrees = {node: 0 for node in root_class.tree.nodes}
    for left, right in root_class.tree.edges:
        directions = {
            (int(left), int(right)) in root_class.candidates[dag_id].id
            for dag_id in state
        }
        if len(directions) > 1:
            degrees[int(left)] += 1
            degrees[int(right)] += 1
    nodes = set(eligible) & set(variable)
    node = min(nodes, key=lambda item: (-degrees[item], item))
    return SelectionResult(node, (-degrees[node], node), eligible[node])


def select_query(
    policy: PolicyName,
    root_class: TreeCandidateClass,
    state: frozenset[DagId],
    queried_nodes: frozenset[int],
    certificates: CertificateCache,
) -> SelectionResult:
    if len(state) <= 1:
        raise ValueError("selection is defined on non-singleton states")
    if policy == "one_step_matching_aware":
        return select_one_step(root_class, state, queried_nodes, certificates)
    if policy == "balance":
        return select_balance(root_class, state, queried_nodes)
    if policy == "first_unresolved":
        return select_first_unresolved(root_class, state, queried_nodes)
    if policy == "max_undirected_degree":
        return select_max_undirected_degree(root_class, state, queried_nodes)
    raise ValueError(f"unknown policy: {policy}")


def paired_policy_path(
    *,
    policy: PolicyName,
    root_class: TreeCandidateClass,
    true_root: int,
    certificates: CertificateCache,
) -> dict[str, object]:
    """Return a full DAG path and its conservatively certified MATCH prefix."""
    true_id = root_class.ids_by_root[int(true_root)]
    state = root_class.initial_state
    queried: frozenset[int] = frozenset()
    states = [state]
    steps: list[dict[str, object]] = []
    terminal_results = [certificates.check(state)]
    while len(state) > 1:
        selected = select_query(policy, root_class, state, queried, certificates)
        observed_index, observed = next(
            (index, child)
            for index, child in enumerate(selected.partition)
            if true_id in child
        )
        steps.append(
            {
                "state_size_before": len(state),
                "selected_node": selected.node,
                "score": list(selected.score),
                "cell_sizes": [len(child) for child in selected.partition],
                "child_certificate_statuses": list(selected.child_statuses),
                "observed_cell_index": observed_index,
                "observed_cell_size": len(observed),
                "surviving_roots": sorted(
                    root_class.roots_by_id[dag_id] for dag_id in observed
                ),
            }
        )
        state = observed
        queried = queried | {selected.node}
        states.append(state)
        terminal_results.append(certificates.check(state))

    matching_index = next(
        (
            index
            for index, result in enumerate(terminal_results)
            if result.status == "certified"
        ),
        None,
    )
    fallback_used = matching_index is None
    if matching_index is None:
        matching_index = len(steps)
    matching_result = terminal_results[matching_index]
    dag_result = terminal_results[-1]
    return {
        "policy": policy,
        "true_root_evaluation_only": int(true_root),
        "dag_query_count": len(steps),
        "matching_query_count": int(matching_index),
        "matching_is_prefix": True,
        "matching_terminal_candidate_count": len(states[matching_index]),
        "matching_terminal_roots": sorted(
            root_class.roots_by_id[dag_id] for dag_id in states[matching_index]
        ),
        "matching_certificate_status": matching_result.status,
        "matching_certificate": matching_result.certificate,
        "dag_certificate_status": dag_result.status,
        "dag_certificate": dag_result.certificate,
        "fallback_to_full_recovery": fallback_used,
        "queried_nodes_full": [step["selected_node"] for step in steps],
        "queried_nodes_matching": [
            step["selected_node"] for step in steps[:matching_index]
        ],
        "steps": steps,
    }


__all__ = [
    "CertificateCache",
    "CertificateSearchResult",
    "PolicyName",
    "SelectionResult",
    "TreeCandidateClass",
    "enumerate_tree_root_class",
    "paired_policy_path",
    "response_partition",
    "select_balance",
    "select_first_unresolved",
    "select_max_undirected_degree",
    "select_one_step",
    "select_query",
]
