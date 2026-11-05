#!/usr/bin/env python3
"""Regenerate the two introductory DCSM paper figures from exact release code.

The script defines two deliberately small population instances, reconstructs
every candidate SCM from moments, computes complete tied optimum families, and
uses direct-KL common-command certificates before it writes any artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.matching_query_complexity import (  # noqa: E402
    CanonicalAction,
    CandidateDAG,
    DagId,
    OptimalActionSet,
    candidate_map,
    compute_all_optimal_action_sets,
    direct_action_kl,
    enumerate_consistent_dags,
    find_common_optimal_command,
    infer_outward_tree_root,
    moments_from_parameters,
    outward_tree_dag,
    pdag_from_undirected_graph,
    reconstruct_all_candidate_scms,
    singleton_query_partition,
    solve_dag_complexity,
    solve_matching_complexity,
)
from research.s01_tree_population import (  # noqa: E402
    CertificateCache,
    TreeCandidateClass,
    enumerate_tree_root_class,
    paired_policy_path,
    select_balance,
    select_one_step,
)
BUDGET = 1
SHIFT_BOUNDS = (-2.0, 2.0)
LOG_SCALE_BOUNDS = (-1.0, 1.0)
ATOL = 1e-6
APPROX_ATOL = 1e-3


def _close(actual: float, expected: float, name: str, tolerance: float = ATOL) -> None:
    if not math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0.0):
        raise AssertionError(f"{name}: expected {expected:.12g}, got {actual:.12g}")


def _array_close(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    if not np.allclose(actual, expected, atol=ATOL, rtol=0.0):
        raise AssertionError(f"{name}: maximum error {np.max(np.abs(actual - expected)):.3g}")


def _action_dict(action: CanonicalAction | Any) -> dict[str, list[float]]:
    return {"shift": [float(value) for value in action.shift], "scale": [float(value) for value in action.scale]}


def _dag_label(dag: CandidateDAG) -> str:
    return "; ".join(f"X{parent + 1}->X{child + 1}" for parent, child in dag.id)


def _certificate_dict(certificate: Any | None) -> dict[str, Any] | None:
    if certificate is None:
        return None
    return {
        "command": _action_dict(certificate.command),
        "commanded_support": int(certificate.commanded_support),
        "maximum_loss_residual": float(certificate.maximum_loss_residual),
        "candidate_losses": [
            {
                "dag": [list(edge) for edge in item.dag_id],
                "direct_kl": float(item.direct_kl),
                "optimal_kl": float(item.optimal_kl),
                "absolute_residual": float(item.absolute_residual),
                "tolerance": float(item.tolerance),
            }
            for item in certificate.candidate_losses
        ],
    }


def _state_key(state: Iterable[DagId], roots: dict[DagId, int]) -> str:
    return ",".join(str(roots[dag] + 1) for dag in sorted(state, key=lambda item: roots[item]))


@dataclass
class PolicyTreeNode:
    state: frozenset[DagId]
    certificate: Any | None
    selected_node: int | None
    skip_query_for_matching: bool
    children: list["PolicyTreeNode"]


def _select(policy: str, root_class: TreeCandidateClass, state: frozenset[DagId], queried: frozenset[int], cache: CertificateCache):
    if policy == "one_step_matching_aware":
        return select_one_step(root_class, state, queried, cache)
    if policy == "balance":
        return select_balance(root_class, state, queried)
    raise ValueError(f"unknown policy {policy}")


def _policy_tree(
    *,
    policy: str,
    root_class: TreeCandidateClass,
    action_sets: dict[DagId, OptimalActionSet],
    truncate_at_matching: bool,
) -> PolicyTreeNode:
    """Build the actual policy response tree, optionally stopping at MATCH."""
    cache = CertificateCache(action_sets)

    def visit(state: frozenset[DagId], queried: frozenset[int]) -> PolicyTreeNode:
        result = cache.check(state)
        certificate = result.certificate if result.status == "certified" else None
        node = PolicyTreeNode(
            state=state,
            certificate=certificate,
            selected_node=None,
            skip_query_for_matching=certificate is not None and len(state) > 1,
            children=[],
        )
        if len(state) == 1 or (truncate_at_matching and certificate is not None):
            return node
        selection = _select(policy, root_class, state, queried, cache)
        node.selected_node = int(selection.node)
        node.children = [
            visit(child, queried | {int(selection.node)})
            for child in sorted(selection.partition, key=lambda cell: tuple(sorted(cell)))
        ]
        return node

    return visit(root_class.initial_state, frozenset())


def _serialize_tree(node: PolicyTreeNode, roots: dict[DagId, int]) -> dict[str, Any]:
    return {
        "roots": [roots[dag] + 1 for dag in sorted(node.state, key=lambda item: roots[item])],
        "selected_query": None if node.selected_node is None else node.selected_node + 1,
        "matching_skips_query": node.skip_query_for_matching,
        "certificate": _certificate_dict(node.certificate),
        "children": [_serialize_tree(child, roots) for child in node.children],
    }


def _all_tree_certificates(node: PolicyTreeNode, roots: dict[DagId, int]) -> dict[str, Any]:
    output = {
        _state_key(node.state, roots): {
            "roots": [roots[dag] + 1 for dag in sorted(node.state, key=lambda item: roots[item])],
            "certificate": _certificate_dict(node.certificate),
        }
    }
    for child in node.children:
        output.update(_all_tree_certificates(child, roots))
    return output


def _instance_1() -> dict[str, Any]:
    correlation = 1.0 / math.sqrt(3.0)
    source_covariance = np.array([[1.0, correlation], [correlation, 1.0]])
    target_covariance = np.array([[1.0, 2 * correlation], [2 * correlation, 2.0]])
    candidates = enumerate_consistent_dags(pdag_from_undirected_graph(nx.path_graph(2)))
    candidate_by_id = candidate_map(candidates)
    scms = reconstruct_all_candidate_scms(candidates, np.zeros(2), source_covariance)
    actions = compute_all_optimal_action_sets(scms, target_mean=np.zeros(2), target_covariance=target_covariance, budget=BUDGET, shift_bounds=SHIFT_BOUNDS, log_scale_bounds=LOG_SCALE_BOUNDS)
    witness = find_common_optimal_command(candidate_by_id, actions)
    if witness is None:
        raise AssertionError("Instance 1 must have a common optimal physical command")
    matching = solve_matching_complexity(candidates=candidate_by_id, action_sets=actions)
    dag = solve_dag_complexity(candidates=candidate_by_id)
    expected_covariances = [target_covariance, source_covariance]
    records = []
    for index, candidate in enumerate(candidates):
        action_set = actions[candidate.id]
        actual_mean, actual_covariance = scms[candidate.id].interventional_moments((0.0, 0.0), (1.0, 2.0))
        action_kl = direct_action_kl(scm=scms[candidate.id], action=CanonicalAction((0.0, 0.0), (1.0, 2.0)), target_mean=np.zeros(2), target_covariance=target_covariance)
        _array_close(actual_mean, np.zeros(2), f"Instance 1 H{index + 1} action mean")
        _array_close(actual_covariance, expected_covariances[index], f"Instance 1 H{index + 1} covariance")
        _close(action_kl, (0.0, 0.25)[index], f"Instance 1 H{index + 1} action KL")
        _close(action_set.optimal_kl, (0.0, 0.25)[index], f"Instance 1 H{index + 1} K*")
        records.append({
            "dag": _dag_label(candidate),
            "canonical_optima": [_action_dict(action) for action in action_set.actions],
            "root_scale_free_nodes": [[node + 1 for node in sorted(family.free_root_scale_nodes)] for family in action_set.command_families],
            "K_star": float(action_set.optimal_kl),
            "action_covariance": actual_covariance.tolist(),
            "action_kl": float(action_kl),
        })
    _close(witness.command.scale[0], 1.0, "Instance 1 common scale X1")
    _close(witness.command.scale[1], 2.0, "Instance 1 common scale X2")
    if witness.commanded_support != 1:
        raise AssertionError("Instance 1 common command must use one coordinate")
    if matching.complexity != 0 or dag.complexity != 1:
        raise AssertionError(f"Instance 1 complexities are {matching.complexity}, {dag.complexity}")
    return {
        "source_covariance": source_covariance.tolist(),
        "target_covariance": target_covariance.tolist(),
        "candidates": records,
        "common_certificate": _certificate_dict(witness),
        "T_match": matching.complexity,
        "T_DAG": dag.complexity,
        "assertions": "passed",
    }


def _instance_2() -> dict[str, Any]:
    tree = nx.path_graph(5)
    source_dag = outward_tree_dag(tree, 4)
    coefficients = np.zeros((5, 5))
    for parent, child in source_dag.edges:
        coefficients[parent, child] = 0.8
    source_mean, source_covariance = moments_from_parameters(coefficients=coefficients, intercepts=np.zeros(5), noise_variances=np.ones(5))
    target_coefficients = coefficients.copy()
    target_coefficients[1, 0] = 1.6
    target_mean, target_covariance = moments_from_parameters(coefficients=target_coefficients, intercepts=np.zeros(5), noise_variances=np.ones(5))
    root_class = enumerate_tree_root_class(tree)
    scms = reconstruct_all_candidate_scms(tuple(root_class.candidates.values()), source_mean, source_covariance)
    action_sets = compute_all_optimal_action_sets(scms, target_mean=target_mean, target_covariance=target_covariance, budget=BUDGET, shift_bounds=SHIFT_BOUNDS, log_scale_bounds=LOG_SCALE_BOUNDS)
    roots = {dag_id: infer_outward_tree_root(candidate) for dag_id, candidate in root_class.candidates.items()}
    ordered = sorted(action_sets, key=lambda dag_id: roots[dag_id])
    root_records = []
    for dag_id in ordered:
        root = roots[dag_id]
        action_set = action_sets[dag_id]
        optimum = action_set.actions[0]
        if root == 0:
            _close(action_set.optimal_kl, 0.5742491767689968, "Instance 2 root X1 K*", APPROX_ATOL)
            _close(optimum.scale[1], 0.7168236462155982, "Instance 2 root X1 scale", APPROX_ATOL)
        else:
            _close(action_set.optimal_kl, 0.0, f"Instance 2 root X{root + 1} K*")
            _close(optimum.scale[0], 2.0, f"Instance 2 root X{root + 1} action")
        root_records.append({"root": root + 1, "K_star": float(action_set.optimal_kl), "canonical_optima": [_action_dict(action) for action in action_set.actions]})
    root_one = next(dag_id for dag_id, root in roots.items() if root == 0)
    scale_x1 = CanonicalAction((0.0,) * 5, (2.0, 1.0, 1.0, 1.0, 1.0))
    _close(direct_action_kl(scm=scms[root_one], action=scale_x1, target_mean=target_mean, target_covariance=target_covariance), 0.73975808, "Instance 2 root X1 scale-X1 KL", APPROX_ATOL)
    matching = solve_matching_complexity(candidates=root_class.candidates, action_sets=action_sets)
    dag = solve_dag_complexity(candidates=root_class.candidates)
    if matching.complexity != 1 or dag.complexity != 2:
        raise AssertionError(f"Instance 2 complexities are {matching.complexity}, {dag.complexity}")
    x3 = singleton_query_partition(matching.initial_model_ids, root_class.candidates, 2)
    x3_roots = [[roots[dag_id] + 1 for dag_id in sorted(cell, key=lambda item: roots[item])] for cell in x3]
    if x3_roots != [[1, 2], [3], [4, 5]]:
        raise AssertionError(f"Instance 2 X3 partition is {x3_roots}")
    x3_certified = [find_common_optimal_command(cell, action_sets) is not None for cell in x3]
    if x3_certified != [False, True, True]:
        raise AssertionError(f"Instance 2 X3 certifications are {x3_certified}")
    balance_tree = _policy_tree(policy="balance", root_class=root_class, action_sets=action_sets, truncate_at_matching=False)
    one_step_tree = _policy_tree(policy="one_step_matching_aware", root_class=root_class, action_sets=action_sets, truncate_at_matching=True)
    cache = CertificateCache(action_sets)
    one_selection = select_one_step(root_class, root_class.initial_state, frozenset(), cache)
    balance_selection = select_balance(root_class, root_class.initial_state, frozenset())
    if one_selection.node != 1 or balance_selection.node != 2:
        raise AssertionError(f"Initial policy queries are {one_selection.node + 1}, {balance_selection.node + 1}")
    trajectories: dict[str, Any] = {}
    for policy in ("one_step_matching_aware", "balance"):
        runs = [paired_policy_path(policy=policy, root_class=root_class, true_root=root, certificates=CertificateCache(action_sets)) for root in range(5)]
        matching_cost = max(int(run["matching_query_count"]) for run in runs)
        expected = 1 if policy == "one_step_matching_aware" else 2
        if matching_cost != expected:
            raise AssertionError(f"{policy} worst matching cost is {matching_cost}")
        trajectories[policy] = {
            "first_query": int(runs[0]["queried_nodes_full"][0]) + 1,
            "worst_case_matching_cost": matching_cost,
            "by_true_root": [
                {
                    "true_root": int(run["true_root_evaluation_only"]) + 1,
                    "matching_query_count": int(run["matching_query_count"]),
                    "dag_query_count": int(run["dag_query_count"]),
                    "queried_nodes_matching": [int(node) + 1 for node in run["queried_nodes_matching"]],
                    "queried_nodes_full": [int(node) + 1 for node in run["queried_nodes_full"]],
                    "matching_terminal_roots": [int(root) + 1 for root in run["matching_terminal_roots"]],
                }
                for run in runs
            ],
        }
    return {
        "source_covariance": source_covariance.tolist(),
        "target_covariance": target_covariance.tolist(),
        "candidate_optima": root_records,
        "root_1_scale": float(action_sets[root_one].actions[0].scale[1]),
        "root_1_scale_X1_by_2_kl": float(direct_action_kl(scm=scms[root_one], action=scale_x1, target_mean=target_mean, target_covariance=target_covariance)),
        "T_match": matching.complexity,
        "T_DAG": dag.complexity,
        "matching_optimal_first_queries": [node + 1 for node in matching.initial_state.optimal_queries],
        "dag_optimal_first_queries": [node + 1 for node in dag.initial_state.optimal_queries],
        "X3_partition": [{"roots": cell, "certified": certified} for cell, certified in zip(x3_roots, x3_certified, strict=True)],
        "policy_trajectories": trajectories,
        "roots": roots,
        "balance_tree": balance_tree,
        "one_step_tree": one_step_tree,
        "policy_tree_certificates": {
            "balance_full_recovery": _all_tree_certificates(balance_tree, roots),
            "one_step_matching_truncated": _all_tree_certificates(one_step_tree, roots),
        },
        "assertions": "passed",
    }


def _json_instance_2(instance: dict[str, Any]) -> dict[str, Any]:
    roots = instance["roots"]
    return {
        key: value
        for key, value in instance.items()
        if key not in {"roots", "balance_tree", "one_step_tree"}
    } | {
        "balance_full_recovery_tree": _serialize_tree(instance["balance_tree"], roots),
        "one_step_matching_truncated_tree": _serialize_tree(instance["one_step_tree"], roots),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT, help="repository-like output root (default: repository root)")
    args = parser.parse_args()
    output_root = args.output_root.resolve()

    # No output is written until every numerical assertion passes.
    instance_1 = _instance_1()
    instance_2 = _instance_2()
    payload = {
        "schema_version": "paper_intro_instances_v2",
        "release_commit": "874a1957a11488757b781910fbe1a541707691e2",
        "index_mapping": "Paper X1..X5 map to repository indices 0..4.",
        "conventions": {"B": BUDGET, "shift_bounds": list(SHIFT_BOUNDS), "log_scale_bounds": list(LOG_SCALE_BOUNDS), "means": "zero"},
        "assertion_summary": {"instance_1": "passed", "instance_2": "passed", "all": "passed"},
        "instance_1": instance_1,
        "instance_2": _json_instance_2(instance_2),
    }
    results = output_root / "results" / "paper_figures"
    results.mkdir(parents=True, exist_ok=True)
    (results / "intro_instances.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("All Part A assertions passed.")
    print(results / "intro_instances.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
