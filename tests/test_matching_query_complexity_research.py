"""Deterministic tests for the isolated small-n matching-query prototype."""

from __future__ import annotations

from math import floor, log2

import networkx as nx
import numpy as np
import pytest

from graphs.pdag import PartiallyDirectedGraph
from research.matching_query_complexity import (
    CanonicalAction,
    actions_close,
    audit_singleton_query_signatures,
    candidate_map,
    common_optimal_actions,
    compute_all_optimal_action_sets,
    compute_optimal_action_set,
    direct_action_kl,
    enumerate_consistent_dags,
    infer_outward_tree_root,
    moments_from_parameters,
    outward_tree_dag,
    pdag_from_undirected_graph,
    reconstruct_all_candidate_scms,
    reconstruct_candidate_scm,
    singleton_query_partition,
    solve_dag_complexity,
    solve_matching_complexity,
    solve_specialized_tree_root_complexity,
    tree_root_partition,
    verify_matching_leq_dag_for_all_subsets,
)

SHIFT_BOUNDS = (-2.0, 2.0)
LOG_SCALE_BOUNDS = (-1.0, 1.0)


def _generic_tree_source(
    tree: nx.Graph, root: int = 0
) -> tuple[nx.DiGraph, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dag = outward_tree_dag(tree, root)
    n = tree.number_of_nodes()
    coefficients = np.zeros((n, n))
    for edge_index, (parent, child) in enumerate(sorted(dag.edges)):
        coefficients[parent, child] = 0.28 + 0.037 * (edge_index + 1)
    intercepts = np.linspace(-0.15, 0.2, n)
    noise = np.linspace(0.8, 1.3, n)
    mean, covariance = moments_from_parameters(
        coefficients=coefficients,
        intercepts=intercepts,
        noise_variances=noise,
    )
    return dag, coefficients, intercepts, noise, mean, covariance


def _path_problem(n: int = 5):
    tree = nx.path_graph(n)
    pdag = pdag_from_undirected_graph(tree)
    candidates = enumerate_consistent_dags(pdag)
    candidate_by_id = candidate_map(candidates)
    _, _, _, _, mean, covariance = _generic_tree_source(tree)
    scms = reconstruct_all_candidate_scms(candidates, mean, covariance)
    return tree, candidate_by_id, scms, mean, covariance


def _compelled_sink_problem(path_nodes: int):
    endpoint = path_nodes - 1
    sink = path_nodes
    other_parent = path_nodes + 1
    pdag = PartiallyDirectedGraph(
        nodes=frozenset(range(path_nodes + 2)),
        directed_edges=frozenset({(endpoint, sink), (other_parent, sink)}),
        undirected_edges=frozenset((node, node + 1) for node in range(endpoint)),
    )
    candidates = enumerate_consistent_dags(pdag)
    candidate_by_id = candidate_map(candidates)

    true_dag = nx.DiGraph()
    true_dag.add_nodes_from(range(path_nodes + 2))
    true_dag.add_edges_from((node, node + 1) for node in range(endpoint))
    true_dag.add_edges_from(((endpoint, sink), (other_parent, sink)))
    n = path_nodes + 2
    coefficients = np.zeros((n, n))
    for edge_index, (parent, child) in enumerate(sorted(true_dag.edges)):
        sign = -1.0 if (parent, child) == (other_parent, sink) else 1.0
        coefficients[parent, child] = sign * (0.31 + 0.029 * edge_index)
    intercepts = np.linspace(-0.12, 0.17, n)
    noise = np.linspace(0.75, 1.35, n)
    source_mean, source_covariance = moments_from_parameters(
        coefficients=coefficients,
        intercepts=intercepts,
        noise_variances=noise,
    )
    scms = reconstruct_all_candidate_scms(
        candidates, source_mean, source_covariance
    )
    true_id = tuple(sorted(true_dag.edges))
    true_scm = scms[true_id]
    shift = np.zeros(n)
    scale = np.ones(n)
    scale[sink] = 1.4
    target_mean, target_covariance = true_scm.interventional_moments(shift, scale)
    action = CanonicalAction.from_arrays(shift, scale)
    return (
        candidate_by_id,
        scms,
        source_mean,
        source_covariance,
        target_mean,
        target_covariance,
        action,
        sink,
    )


def test_candidate_dag_enumeration_validates_mec_and_stable_ids() -> None:
    path = nx.path_graph(4)
    candidates = enumerate_consistent_dags(pdag_from_undirected_graph(path))
    assert len(candidates) == 4
    assert len({candidate.id for candidate in candidates}) == 4
    assert all(not candidate.colliders for candidate in candidates)
    assert [infer_outward_tree_root(candidate) for candidate in candidates] == [
        infer_outward_tree_root(candidate)
        for candidate in sorted(candidates, key=lambda item: item.id)
    ]

    collider = PartiallyDirectedGraph(
        nodes=frozenset({0, 1, 2}),
        directed_edges=frozenset({(0, 1), (2, 1)}),
        undirected_edges=frozenset(),
    )
    only = enumerate_consistent_dags(collider)
    assert len(only) == 1
    assert only[0].colliders == frozenset({(0, 1, 2)})


def test_candidate_scm_reconstruction_reproduces_public_distribution() -> None:
    _, candidates, scms, mean, covariance = _path_problem(5)
    assert set(scms) == set(candidates)
    for scm in scms.values():
        reconstructed_mean, reconstructed_covariance = scm.observational_moments()
        np.testing.assert_allclose(reconstructed_mean, mean, atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(
            reconstructed_covariance, covariance, atol=1e-11, rtol=1e-10
        )
        assert scm.reproduction_mean_error < 1e-11
        assert scm.reproduction_covariance_error < 1e-11


def test_parent_partitions_equal_tree_root_component_partitions() -> None:
    tree, candidates, scms, _, _ = _path_problem(6)
    roots = {
        dag_id: infer_outward_tree_root(candidate)
        for dag_id, candidate in candidates.items()
    }
    model_ids = frozenset(candidates)
    for node in tree.nodes:
        assert singleton_query_partition(model_ids, candidates, node) == (
            tree_root_partition(tree, roots, model_ids, node)
        )
    audit = audit_singleton_query_signatures(scms)
    assert audit.passed, audit.failures
    assert audit.same_signature_max_error < 1e-10
    assert audit.different_signature_min_separation > 1e-8


def test_tie_preserving_action_enumeration_keeps_both_global_actions() -> None:
    pdag = PartiallyDirectedGraph(
        nodes=frozenset({0, 1}),
        directed_edges=frozenset(),
        undirected_edges=frozenset(),
    )
    candidate = enumerate_consistent_dags(pdag)[0]
    scm = reconstruct_candidate_scm(candidate, np.zeros(2), np.eye(2))
    action_set = compute_optimal_action_set(
        scm,
        target_mean=np.ones(2),
        target_covariance=np.eye(2),
        budget=1,
        shift_bounds=SHIFT_BOUNDS,
        log_scale_bounds=LOG_SCALE_BOUNDS,
    )
    assert len(action_set.actions) == 2
    assert {action.shift for action in action_set.actions} == {
        (0.0, 1.0),
        (1.0, 0.0),
    }
    assert "global_optimal_action_tie_count=2" in action_set.numerical_diagnostics


def test_bellman_base_cases_and_arbitrary_initial_model_set() -> None:
    _, candidates, scms, source_mean, source_covariance = _path_problem(5)
    action_sets = compute_all_optimal_action_sets(
        scms,
        target_mean=source_mean,
        target_covariance=source_covariance,
        budget=1,
        shift_bounds=SHIFT_BOUNDS,
        log_scale_bounds=LOG_SCALE_BOUNDS,
    )
    matching = solve_matching_complexity(
        candidates=candidates, action_sets=action_sets
    )
    dag = solve_dag_complexity(candidates=candidates)
    assert matching.complexity == 0
    assert matching.initial_state.terminal_reason == "common_optimal_action"
    assert dag.complexity == floor(log2(5))

    singleton = {next(iter(candidates))}
    singleton_dag = solve_dag_complexity(
        candidates=candidates, initial_model_ids=singleton
    )
    assert singleton_dag.complexity == 0
    assert singleton_dag.initial_model_ids == frozenset(singleton)


def test_matching_complexity_never_exceeds_dag_for_every_path_subset() -> None:
    _, candidates, scms, mean, covariance = _path_problem(5)
    target_mean = mean + np.array([0.25, -0.15, 0.1, 0.05, -0.2])
    target_covariance = covariance + np.diag([0.12, 0.08, 0.09, 0.11, 0.07])
    action_sets = compute_all_optimal_action_sets(
        scms,
        target_mean=target_mean,
        target_covariance=target_covariance,
        budget=1,
        shift_bounds=SHIFT_BOUNDS,
        log_scale_bounds=LOG_SCALE_BOUNDS,
    )
    checked, maximum_gap = verify_matching_leq_dag_for_all_subsets(
        candidates=candidates, action_sets=action_sets
    )
    assert checked == 2 ** len(candidates) - 1
    assert maximum_gap >= 0


@pytest.mark.parametrize(
    "tree",
    [
        nx.path_graph(5),
        nx.star_graph(4),
        nx.from_prufer_sequence([0, 2, 2, 4]),
    ],
)
def test_general_tree_recursion_agrees_with_specialized_root_recursion(
    tree: nx.Graph,
) -> None:
    pdag = pdag_from_undirected_graph(tree)
    candidate_sequence = enumerate_consistent_dags(pdag)
    candidates = candidate_map(candidate_sequence)
    _, _, _, _, mean, covariance = _generic_tree_source(tree)
    scms = reconstruct_all_candidate_scms(candidate_sequence, mean, covariance)
    target_mean = mean + np.linspace(-0.17, 0.23, len(mean))
    target_covariance = covariance + 0.09 * np.eye(len(mean))
    action_sets = compute_all_optimal_action_sets(
        scms,
        target_mean=target_mean,
        target_covariance=target_covariance,
        budget=1,
        shift_bounds=SHIFT_BOUNDS,
        log_scale_bounds=LOG_SCALE_BOUNDS,
    )
    assert solve_matching_complexity(
        candidates=candidates, action_sets=action_sets
    ).complexity == solve_specialized_tree_root_complexity(
        tree=tree, candidates=candidates, action_sets=action_sets
    )
    assert solve_dag_complexity(
        candidates=candidates
    ).complexity == solve_specialized_tree_root_complexity(
        tree=tree, candidates=candidates
    )


def test_realisable_path_subset_stops_before_dag_identification() -> None:
    tree, candidates, scms, _, _ = _path_problem(6)
    true_id = next(
        dag_id
        for dag_id, candidate in candidates.items()
        if infer_outward_tree_root(candidate) == 5
    )
    action = CanonicalAction.from_arrays(
        np.zeros(6), np.array([1.35, 1.0, 1.0, 1.0, 1.0, 1.0])
    )
    target_mean, target_covariance = scms[true_id].interventional_moments(
        action.shift, action.scale
    )
    action_sets = compute_all_optimal_action_sets(
        scms,
        target_mean=target_mean,
        target_covariance=target_covariance,
        budget=1,
        shift_bounds=SHIFT_BOUNDS,
        log_scale_bounds=LOG_SCALE_BOUNDS,
    )
    compatible = frozenset(
        dag_id
        for dag_id, candidate in candidates.items()
        if infer_outward_tree_root(candidate) != 0
    )
    common = common_optimal_actions(compatible, action_sets)
    assert any(actions_close(action, item) for item in common)
    matching = solve_matching_complexity(
        candidates=candidates,
        action_sets=action_sets,
        initial_model_ids=compatible,
    )
    dag = solve_dag_complexity(
        candidates=candidates, initial_model_ids=compatible
    )
    assert matching.initial_model_ids == compatible
    assert matching.complexity == 0 < dag.complexity
    assert solve_specialized_tree_root_complexity(
        tree=tree,
        candidates=candidates,
        initial_model_ids=compatible,
        action_sets=action_sets,
    ) == matching.complexity


@pytest.mark.parametrize("path_nodes", [2, 3, 4, 5, 6, 7])
def test_compelled_sink_full_mec_strict_separation(path_nodes: int) -> None:
    (
        candidates,
        scms,
        _,
        _,
        target_mean,
        target_covariance,
        action,
        sink,
    ) = _compelled_sink_problem(path_nodes)
    assert len(candidates) == path_nodes > 1
    assert {candidate.parent_sets[sink] for candidate in candidates.values()} == {
        (path_nodes - 1, path_nodes + 1)
    }
    audit = audit_singleton_query_signatures(scms)
    assert audit.passed, audit.failures

    action_sets = compute_all_optimal_action_sets(
        scms,
        target_mean=target_mean,
        target_covariance=target_covariance,
        budget=1,
        shift_bounds=SHIFT_BOUNDS,
        log_scale_bounds=LOG_SCALE_BOUNDS,
    )
    for dag_id, scm in scms.items():
        assert scm.candidate.parent_sets[sink]
        assert direct_action_kl(
            scm=scm,
            action=action,
            target_mean=target_mean,
            target_covariance=target_covariance,
        ) < 2e-10
        assert action_sets[dag_id].optimal_kl < 2e-10
        assert any(actions_close(action, item) for item in action_sets[dag_id].actions)

    matching = solve_matching_complexity(
        candidates=candidates, action_sets=action_sets
    )
    dag = solve_dag_complexity(candidates=candidates)
    assert matching.initial_model_ids == frozenset(candidates)
    assert matching.complexity == 0
    assert dag.complexity == floor(log2(path_nodes))
    assert matching.complexity < dag.complexity

