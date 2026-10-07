"""Regression tests for DAG-independent physical-command matching."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from math import log, sqrt

import networkx as nx
import numpy as np
import pytest

from graphs.support import parameter_support_cost
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from research.matching_query_complexity import (
    CanonicalAction,
    CandidateDAG,
    CommandCertificationError,
    OptimalCommandFamily,
    PhysicalCommand,
    candidate_map,
    compute_all_optimal_action_sets,
    find_common_optimal_command,
    intersect_optimal_command_families,
    reconstruct_all_candidate_scms,
    solve_dag_complexity,
    solve_matching_complexity,
    solve_specialized_tree_root_complexity,
    unshielded_colliders,
)


H1_ID = ((0, 1),)
H2_ID = ((1, 0),)
ARCHIVED_SHIFT_BOUNDS = (-2.0, 2.0)
ARCHIVED_LOG_SCALE_BOUNDS = (-1.0, 1.0)


def _candidate(edges: tuple[tuple[int, int], ...]) -> CandidateDAG:
    dag = nx.DiGraph()
    dag.add_nodes_from((0, 1))
    dag.add_edges_from(edges)
    return CandidateDAG(
        id=tuple(sorted(edges)),
        dag=nx.freeze(dag),
        parent_sets=tuple(
            tuple(sorted(int(parent) for parent in dag.predecessors(node)))
            for node in (0, 1)
        ),
        colliders=unshielded_colliders(dag),
    )


def _two_node_problem(
    source_covariance: np.ndarray,
    target_covariance: np.ndarray,
    *,
    budget: int,
    log_scale_bounds: tuple[float, float],
):
    candidates = (_candidate(H1_ID), _candidate(H2_ID))
    candidate_by_id = candidate_map(candidates)
    scms = reconstruct_all_candidate_scms(
        candidates, np.zeros(2), source_covariance
    )
    action_sets = compute_all_optimal_action_sets(
        scms,
        target_mean=np.zeros(2),
        target_covariance=target_covariance,
        budget=budget,
        shift_bounds=ARCHIVED_SHIFT_BOUNDS,
        log_scale_bounds=log_scale_bounds,
    )
    return candidate_by_id, scms, action_sets


def _direct_kl(action_set, command: PhysicalCommand) -> float:
    mean, covariance = action_set.scm.interventional_moments(
        np.asarray(command.shift), np.asarray(command.scale)
    )
    return gaussian_kl_target_to_candidate(
        action_set.target_mean,
        action_set.target_covariance,
        mean,
        covariance,
    )


def _exhaustive_tiny_reference(action_sets, budget: int) -> tuple[PhysicalCommand, ...]:
    """Independent fixed-support quadratic reference for the zero-mean toys.

    For every candidate's non-root coordinate, derive the one-dimensional
    quadratic scale minimizer directly from its sufficient moments. Exhaust the
    resulting global scale supports and verify every command by direct Gaussian
    KL. This does not inspect canonical actions or command families.
    """
    first = next(iter(action_sets.values()))
    n = first.actions[0].dimension
    shift_values = [(0.0,) for _ in range(n)]
    scale_values: list[list[float]] = []
    for node in range(n):
        values = {1.0}
        for action_set in action_sets.values():
            moment = action_set.oracle_result.node_moments[node]
            if moment.scale_eligible:
                values.add(
                    float(
                        np.clip(
                            moment.second_yz / moment.second_z,
                            action_set.scale_bounds[0],
                            action_set.scale_bounds[1],
                        )
                    )
                )
        scale_values.append(sorted(values))
    witnesses: list[PhysicalCommand] = []
    for shift in product(*shift_values):
        for scale in product(*scale_values):
            shift_array = np.asarray(shift, dtype=float)
            scale_array = np.asarray(scale, dtype=float)
            if parameter_support_cost(shift_array, scale_array) > budget:
                continue
            command = PhysicalCommand(tuple(shift), tuple(scale))
            valid = True
            for action_set in action_sets.values():
                value = _direct_kl(action_set, command)
                tolerance = action_set.loss_atol + action_set.loss_rtol * max(
                    1.0, abs(value), abs(action_set.optimal_kl)
                )
                if abs(value - action_set.optimal_kl) > tolerance:
                    valid = False
                    break
            if valid:
                witnesses.append(command)
    return tuple(witnesses)


def test_original_root_scale_counterexample_has_zero_matching_queries() -> None:
    source = np.array([[1.0, 0.5], [0.5, 1.0]])
    target = np.array([[1.0, 1.5], [1.5, 3.0]])
    candidates, _, action_sets = _two_node_problem(
        source,
        target,
        budget=1,
        log_scale_bounds=(-log(3.0), log(3.0)),
    )

    assert action_sets[H1_ID].optimal_kl == pytest.approx(0.0, abs=2e-15)
    assert action_sets[H2_ID].optimal_kl == pytest.approx(2.0 / 3.0, abs=2e-15)
    certificate = find_common_optimal_command(candidates, action_sets)
    assert certificate is not None
    assert certificate.command.shift == (0.0, 0.0)
    assert certificate.command.scale == pytest.approx((1.0, 3.0))
    assert certificate.commanded_support == 1
    assert certificate.maximum_loss_residual <= 2e-15
    assert all(
        item.absolute_residual <= item.tolerance
        for item in certificate.candidate_losses
    )
    assert solve_matching_complexity(
        candidates=candidates, action_sets=action_sets
    ).complexity == 0
    assert solve_dag_complexity(candidates=candidates).complexity == 1
    assert solve_specialized_tree_root_complexity(
        tree=nx.path_graph(2),
        candidates=candidates,
        action_sets=action_sets,
    ) == 0
    assert any(
        command.scale == pytest.approx((1.0, 3.0))
        for command in _exhaustive_tiny_reference(action_sets, 1)
    )


def test_root_scale_witness_inside_archived_bounds() -> None:
    rho = 1.0 / sqrt(3.0)
    source = np.array([[1.0, rho], [rho, 1.0]])
    target = np.array([[1.0, 2.0 * rho], [2.0 * rho, 2.0]])
    candidates, _, action_sets = _two_node_problem(
        source,
        target,
        budget=1,
        log_scale_bounds=ARCHIVED_LOG_SCALE_BOUNDS,
    )

    certificate = find_common_optimal_command(candidates, action_sets)
    assert certificate is not None
    assert certificate.command.scale == pytest.approx((1.0, 2.0))
    assert certificate.commanded_support == 1
    for action_set in action_sets.values():
        assert _direct_kl(action_set, certificate.command) == pytest.approx(
            action_set.optimal_kl, abs=2e-15
        )
    assert solve_matching_complexity(
        candidates=candidates, action_sets=action_sets
    ).complexity == 0
    assert solve_dag_complexity(candidates=candidates).complexity == 1
    assert _exhaustive_tiny_reference(action_sets, 1)


def test_global_budget_blocks_two_required_scales_at_b1_and_allows_b2() -> None:
    source = np.array([[1.0, 0.5], [0.5, 1.0]])
    target = np.array([[1.0, 1.0], [1.0, 1.75]])
    candidates, _, action_sets_b1 = _two_node_problem(
        source,
        target,
        budget=1,
        log_scale_bounds=ARCHIVED_LOG_SCALE_BOUNDS,
    )
    assert action_sets_b1[H1_ID].actions[0].scale == pytest.approx((1.0, 2.0))
    assert action_sets_b1[H2_ID].actions[0].scale == pytest.approx((8.0 / 7.0, 1.0))
    for action_set in action_sets_b1.values():
        canonical = action_set.actions[0]
        assert _direct_kl(
            action_set,
            PhysicalCommand(canonical.shift, canonical.scale),
        ) == pytest.approx(action_set.optimal_kl, abs=2e-15)
    assert find_common_optimal_command(candidates, action_sets_b1) is None
    assert not _exhaustive_tiny_reference(action_sets_b1, 1)
    assert solve_matching_complexity(
        candidates=candidates, action_sets=action_sets_b1
    ).complexity == 1

    _, _, action_sets_b2 = _two_node_problem(
        source,
        target,
        budget=2,
        log_scale_bounds=ARCHIVED_LOG_SCALE_BOUNDS,
    )
    certificate = find_common_optimal_command(candidates, action_sets_b2)
    assert certificate is not None
    assert certificate.command.scale == pytest.approx((8.0 / 7.0, 2.0))
    assert certificate.commanded_support == 2
    assert _exhaustive_tiny_reference(action_sets_b2, 2)


def _family(
    shift: float,
    *,
    scale: float = 1.0,
    roots: frozenset[int] = frozenset(),
    shift_bounds: tuple[float, float] = (-2.0, 2.0),
    scale_bounds: tuple[float, float] = (0.25, 3.0),
    budget: int = 1,
) -> OptimalCommandFamily:
    return OptimalCommandFamily.from_canonical(
        CanonicalAction((shift,), (scale,)),
        root_nodes=roots,
        shift_bounds=shift_bounds,
        scale_bounds=scale_bounds,
        budget=budget,
    )


def test_tied_alternatives_are_exhausted_instead_of_chosen_greedily() -> None:
    left = ((0, 1),)
    right = ((1, 0),)
    families = {
        left: (_family(0.0), _family(1.0)),
        right: (_family(1.0),),
    }
    witnesses = intersect_optimal_command_families(
        families, families, budget=1
    )
    assert tuple(item.shift for item in witnesses) == ((1.0,),)


def test_pairwise_compatibility_is_not_mistaken_for_three_way_witness() -> None:
    first = ((0, 1),)
    second = ((1, 0),)
    third = ((0, 2),)
    families = {
        first: (_family(0.0), _family(1.0)),
        second: (_family(1.0), _family(2.0)),
        third: (_family(0.0), _family(2.0)),
    }
    assert intersect_optimal_command_families(
        (first, second), families, budget=1
    )
    assert intersect_optimal_command_families(
        (first, third), families, budget=1
    )
    assert intersect_optimal_command_families(
        (second, third), families, budget=1
    )
    assert not intersect_optimal_command_families(
        (first, second, third), families, budget=1
    )


def test_root_flat_coordinates_and_common_bounds_are_respected() -> None:
    fixed = ((0, 1),)
    root_flat = ((1, 0),)
    compatible = {
        fixed: (_family(0.0, scale=2.0),),
        root_flat: (_family(0.0, roots=frozenset({0})),),
    }
    witnesses = intersect_optimal_command_families(
        compatible, compatible, budget=1
    )
    assert len(witnesses) == 1
    assert witnesses[0].scale == (2.0,)
    assert parameter_support_cost(
        np.asarray(witnesses[0].shift), np.asarray(witnesses[0].scale)
    ) == 1

    all_free = {
        fixed: (_family(0.0, roots=frozenset({0})),),
        root_flat: (_family(0.0, roots=frozenset({0})),),
    }
    identity = intersect_optimal_command_families(all_free, all_free, budget=1)
    assert identity == (PhysicalCommand((0.0,), (1.0,)),)

    incompatible_bounds = {
        fixed: (_family(0.0, scale=2.0),),
        root_flat: (
            _family(
                0.0,
                roots=frozenset({0}),
                scale_bounds=(0.5, 1.5),
            ),
        ),
    }
    assert not intersect_optimal_command_families(
        incompatible_bounds, incompatible_bounds, budget=1
    )
    with pytest.raises(ValueError, match="outside bounds"):
        _family(0.0, scale=3.0, scale_bounds=(0.5, 2.0))


def test_structural_witness_with_inconsistent_optimum_fails_loudly() -> None:
    source = np.array([[1.0, 0.5], [0.5, 1.0]])
    target = np.array([[1.0, 1.5], [1.5, 3.0]])
    candidates, _, action_sets = _two_node_problem(
        source,
        target,
        budget=1,
        log_scale_bounds=(-log(3.0), log(3.0)),
    )
    corrupted = dict(action_sets)
    corrupted[H1_ID] = replace(
        corrupted[H1_ID],
        optimal_kl=corrupted[H1_ID].optimal_kl + 0.01,
    )
    with pytest.raises(CommandCertificationError, match="uncertified witnesses"):
        find_common_optimal_command(candidates, corrupted)
