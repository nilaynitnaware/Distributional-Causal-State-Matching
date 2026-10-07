"""Focused validation for the restricted S01-B population tree policy."""

from __future__ import annotations

from math import log

import networkx as nx
import numpy as np

from graphs.support import parameter_support_cost
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from research.matching_query_complexity import (
    compute_all_optimal_action_sets,
    moments_from_parameters,
    reconstruct_all_candidate_scms,
    singleton_query_partition,
    solve_matching_complexity,
    solve_specialized_tree_root_complexity,
)
from research.s01_tree_population import (
    CertificateCache,
    enumerate_tree_root_class,
    paired_policy_path,
    response_partition,
    select_balance,
    select_one_step,
)

SHIFT_BOUNDS = (-2.0, 2.0)
LOG_SCALE_BOUNDS = (-1.0, 1.0)


def _problem(
    tree: nx.Graph,
    *,
    root: int = 0,
    budget: int = 1,
    target_mean: np.ndarray | None = None,
    target_covariance: np.ndarray | None = None,
):
    root_class = enumerate_tree_root_class(tree)
    true_id = root_class.ids_by_root[root]
    dag = root_class.candidates[true_id].dag
    n = tree.number_of_nodes()
    coefficients = np.zeros((n, n))
    for index, (parent, child) in enumerate(sorted(dag.edges)):
        coefficients[parent, child] = 0.3 + 0.03 * index
    source_mean, source_covariance = moments_from_parameters(
        coefficients=coefficients,
        intercepts=np.linspace(-0.1, 0.15, n),
        noise_variances=np.linspace(0.8, 1.2, n),
    )
    scms = reconstruct_all_candidate_scms(
        tuple(root_class.candidates.values()), source_mean, source_covariance
    )
    action_sets = compute_all_optimal_action_sets(
        scms,
        target_mean=source_mean if target_mean is None else target_mean,
        target_covariance=(
            source_covariance if target_covariance is None else target_covariance
        ),
        budget=budget,
        shift_bounds=SHIFT_BOUNDS,
        log_scale_bounds=LOG_SCALE_BOUNDS,
    )
    return root_class, scms, source_mean, source_covariance, action_sets


def _two_node_covariance_problem(target: np.ndarray, budget: int):
    tree = nx.path_graph(2)
    roots = enumerate_tree_root_class(tree)
    source = np.array([[1.0, 0.5], [0.5, 1.0]])
    scms = reconstruct_all_candidate_scms(
        tuple(roots.candidates.values()), np.zeros(2), source
    )
    action_sets = compute_all_optimal_action_sets(
        scms,
        target_mean=np.zeros(2),
        target_covariance=target,
        budget=budget,
        shift_bounds=SHIFT_BOUNDS,
        log_scale_bounds=(-log(3.0), log(3.0)),
    )
    return roots, action_sets


def test_complete_roots_reconstruct_observational_moments() -> None:
    root_class, scms, mean, covariance, _ = _problem(nx.path_graph(6))
    assert len(root_class.candidates) == 6
    assert set(root_class.ids_by_root) == set(range(6))
    for scm in scms.values():
        actual_mean, actual_covariance = scm.observational_moments()
        np.testing.assert_allclose(actual_mean, mean, atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(
            actual_covariance, covariance, atol=1e-11, rtol=1e-10
        )


def test_root_components_equal_parent_signatures_and_exact_responses() -> None:
    root_class, scms, _, _, _ = _problem(nx.balanced_tree(2, 2))
    state = root_class.initial_state
    for node in root_class.tree.nodes:
        assert {frozenset(cell) for cell in response_partition(root_class, state, node)} == {
            frozenset(cell)
            for cell in singleton_query_partition(
                state, root_class.candidates, int(node)
            )
        }
        columns_by_cell = []
        for cell in response_partition(root_class, state, node):
            columns = [
                scms[dag_id].total_effect_matrix[:, node] for dag_id in cell
            ]
            for column in columns[1:]:
                np.testing.assert_allclose(column, columns[0], atol=1e-11, rtol=1e-10)
            columns_by_cell.append(columns[0])
        for left in range(len(columns_by_cell)):
            for right in range(left):
                assert not np.allclose(
                    columns_by_cell[left], columns_by_cell[right], atol=1e-10, rtol=1e-9
                )


def test_known_root_scale_common_command_stops_at_non_singleton() -> None:
    target = np.array([[1.0, 1.5], [1.5, 3.0]])
    roots, action_sets = _two_node_covariance_problem(target, budget=1)
    cache = CertificateCache(action_sets)
    result = cache.check(roots.initial_state)
    assert result.status == "certified"
    assert result.certificate is not None
    assert result.certificate.command.scale == (1.0, 3.0)
    path = paired_policy_path(
        policy="balance", root_class=roots, true_root=0, certificates=cache
    )
    assert path["matching_query_count"] == 0
    assert path["dag_query_count"] == 1
    assert path["matching_terminal_candidate_count"] == 2


def test_no_matching_termination_when_commands_cannot_be_shared() -> None:
    target = np.array([[1.0, 1.0], [1.0, 1.75]])
    roots, action_sets = _two_node_covariance_problem(target, budget=1)
    cache = CertificateCache(action_sets)
    assert cache.check(roots.initial_state).status == "absent"
    path = paired_policy_path(
        policy="balance", root_class=roots, true_root=1, certificates=cache
    )
    assert path["matching_query_count"] == path["dag_query_count"] == 1


def test_inconclusive_search_never_authorizes_stopping_or_terminal_scoring() -> None:
    roots, _, _, _, action_sets = _problem(nx.path_graph(4))
    cache = CertificateCache(action_sets, maximum_tied_family_combinations=0)
    assert cache.check(roots.initial_state).status == "inconclusive"
    selected = select_one_step(roots, roots.initial_state, frozenset(), cache)
    assert selected.child_statuses
    assert set(selected.child_statuses) == {"inconclusive"}
    assert selected.score[0] > 0
    path = paired_policy_path(
        policy="one_step_matching_aware",
        root_class=roots,
        true_root=2,
        certificates=cache,
    )
    assert path["matching_query_count"] == path["dag_query_count"]
    assert path["fallback_to_full_recovery"] is True


def test_selection_is_independent_of_stop_rule_and_match_path_is_prefix() -> None:
    roots, _, _, _, action_sets = _problem(nx.path_graph(5))
    for policy in ("one_step_matching_aware", "balance"):
        cache = CertificateCache(action_sets)
        terminal = cache.check(roots.initial_state)
        assert terminal.status == "certified"
        if policy == "one_step_matching_aware":
            selected = select_one_step(roots, roots.initial_state, frozenset(), cache)
        else:
            selected = select_balance(roots, roots.initial_state, frozenset())
        assert selected.node in roots.tree
        path = paired_policy_path(
            policy=policy, root_class=roots, true_root=4, certificates=cache
        )
        assert path["matching_query_count"] == 0
        assert path["queried_nodes_matching"] == path["queried_nodes_full"][:0]
        assert path["matching_is_prefix"] is True


def test_small_tree_policy_respects_bellman_lower_bound_and_certifies_command() -> None:
    roots, _, mean, covariance, action_sets = _problem(
        nx.path_graph(5),
        target_mean=np.linspace(-0.2, 0.3, 5),
        target_covariance=None,
    )
    # Override the default source covariance with a nontrivial but valid target.
    # The supplied mean alone is enough to induce root-dependent optima here.
    exact = solve_specialized_tree_root_complexity(
        tree=roots.tree,
        candidates=roots.candidates,
        action_sets=action_sets,
    )
    generic_exact = solve_matching_complexity(
        candidates=roots.candidates, action_sets=action_sets
    ).complexity
    assert exact == generic_exact
    cache = CertificateCache(action_sets)
    paths = [
        paired_policy_path(
            policy="balance",
            root_class=roots,
            true_root=root,
            certificates=cache,
        )
        for root in sorted(roots.tree.nodes)
    ]
    assert max(int(path["matching_query_count"]) for path in paths) >= exact
    for path in paths:
        certificate = path["matching_certificate"]
        assert certificate is not None
        command = certificate.command
        assert parameter_support_cost(
            np.asarray(command.shift), np.asarray(command.scale)
        ) <= 1
        for item in certificate.candidate_losses:
            action_set = action_sets[item.dag_id]
            candidate_mean, candidate_covariance = action_set.scm.interventional_moments(
                np.asarray(command.shift), np.asarray(command.scale)
            )
            direct = gaussian_kl_target_to_candidate(
                action_set.target_mean,
                action_set.target_covariance,
                candidate_mean,
                candidate_covariance,
            )
            assert abs(direct - action_set.optimal_kl) <= item.tolerance
