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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import patches
from matplotlib.gridspec import GridSpec

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
from scripts.figure_style import apply_publication_style  # noqa: E402


WIDTH = 6.25
BUDGET = 1
SHIFT_BOUNDS = (-2.0, 2.0)
LOG_SCALE_BOUNDS = (-1.0, 1.0)
ATOL = 1e-6
APPROX_ATOL = 1e-3
LIGHT = "#d9d9d9"
DARK = "#404040"
MID = "#9e9e9e"


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


def _action_kind(certificate: Any | None) -> str | None:
    if certificate is None:
        return None
    scale = np.asarray(certificate.command.scale, dtype=float)
    if not np.isclose(scale[0], 1.0) and np.allclose(scale[1:], 1.0, atol=ATOL):
        return "x1"
    if len(scale) > 1 and not np.isclose(scale[1], 1.0) and np.allclose(
        np.delete(scale, 1), 1.0, atol=ATOL
    ):
        return "x2"
    return "other"


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
            visit(child, queried | {int(selection.node)}) for child in selection.partition
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


def _ellipse(axis: plt.Axes, covariance: np.ndarray, *, facecolor: str, edgecolor: str, linewidth: float, alpha: float = 1.0) -> None:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    axis.add_patch(
        patches.Ellipse(
            (0, 0),
            2 * math.sqrt(eigenvalues[0]),
            2 * math.sqrt(eigenvalues[1]),
            angle=angle,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
    )


def _draw_two_node_dag(axis: plt.Axes, *, reverse: bool) -> None:
    axis.set_axis_off()
    positions = [(0.28, 0.52), (0.70, 0.52)]
    for index, (x, y) in enumerate(positions):
        axis.add_patch(patches.Circle((x, y), 0.105, facecolor="white", edgecolor="black", linewidth=1.0))
        axis.text(x, y, f"$X_{index + 1}$", ha="center", va="center", fontsize=7.4)
    start, end = (positions[1], positions[0]) if reverse else (positions[0], positions[1])
    axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.1, "shrinkA": 10, "shrinkB": 10})
    axis.text(0.70, 0.25, "$s_2=2$", ha="center", va="center", fontsize=6.9)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)


def _tree_positions(root: PolicyTreeNode) -> tuple[dict[int, tuple[float, float]], dict[int, int]]:
    positions: dict[int, tuple[float, float]] = {}
    depths: dict[int, int] = {}
    leaves: list[int] = []

    def assign(node: PolicyTreeNode, depth: int) -> float:
        identity = id(node)
        depths[identity] = depth
        if not node.children:
            x = float(len(leaves))
            leaves.append(identity)
        else:
            x = float(np.mean([assign(child, depth + 1) for child in node.children]))
        positions[identity] = (x, -float(depth))
        return x

    assign(root, 0)
    maximum = max(1, len(leaves) - 1)
    for identity, (x, y) in list(positions.items()):
        positions[identity] = (0.08 + 0.84 * x / maximum, 0.93 + 0.26 * y)
    return positions, depths


def _draw_policy_tree(axis: plt.Axes, root: PolicyTreeNode, roots: dict[DagId, int], *, title: str, caption: str | None) -> None:
    positions, _ = _tree_positions(root)
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.18, 1.08)
    axis.set_axis_off()
    axis.set_title(title, fontsize=7.1, pad=1.5)

    def edges(node: PolicyTreeNode) -> None:
        x0, y0 = positions[id(node)]
        for child in node.children:
            x1, y1 = positions[id(child)]
            axis.plot(
                [x0, x1], [y0 - 0.055, y1 + 0.055],
                color="black", linewidth=0.8,
                linestyle="--" if node.skip_query_for_matching else "-",
                zorder=1,
            )
            edges(child)

    def boxes(node: PolicyTreeNode) -> None:
        x, y = positions[id(node)]
        kind = _action_kind(node.certificate)
        face = LIGHT if kind == "x1" else DARK if kind == "x2" else "white"
        text_color = "white" if kind == "x2" else "black"
        roots_text = "{" + ",".join(str(roots[dag] + 1) for dag in sorted(node.state, key=lambda item: roots[item])) + "}"
        query_text = "" if node.selected_node is None else f"\nquery $X_{node.selected_node + 1}$"
        axis.add_patch(
            patches.FancyBboxPatch(
                (x - 0.115, y - 0.052), 0.23, 0.104,
                boxstyle="round,pad=0.006,rounding_size=0.01",
                facecolor=face, edgecolor="black", linewidth=0.7, zorder=2,
            )
        )
        axis.text(x, y, roots_text + query_text, ha="center", va="center", fontsize=5.3, color=text_color, zorder=3)
        for child in node.children:
            boxes(child)

    edges(root)
    boxes(root)
    if caption:
        axis.text(0.5, -0.105, caption, ha="center", va="top", fontsize=5.3, wrap=True)


def _plot_intro(instance_1: dict[str, Any], instance_2: dict[str, Any], figures: Path) -> None:
    apply_publication_style(7.0)
    figure = plt.figure(figsize=(WIDTH, 2.6))
    outer = GridSpec(1, 2, width_ratios=(1.14, 1.0), figure=figure, wspace=0.14)
    left = outer[0].subgridspec(2, 3, width_ratios=(0.55, 0.78, 1.16), hspace=0.30, wspace=0.15)
    target = np.asarray(instance_1["target_covariance"], dtype=float)
    for row, entry in enumerate(instance_1["candidates"]):
        graph_axis = figure.add_subplot(left[row, 0])
        _draw_two_node_dag(graph_axis, reverse=row == 1)
        ellipse_axis = figure.add_subplot(left[row, 1])
        _ellipse(ellipse_axis, target, facecolor=LIGHT, edgecolor="none", linewidth=0.0, alpha=0.85)
        _ellipse(ellipse_axis, np.asarray(entry["action_covariance"], dtype=float), facecolor="none", edgecolor="black", linewidth=1.35)
        ellipse_axis.axhline(0, color="#777777", linewidth=0.55)
        ellipse_axis.axvline(0, color="#777777", linewidth=0.55)
        ellipse_axis.set_aspect("equal")
        ellipse_axis.set_xticks([])
        ellipse_axis.set_yticks([])
        ellipse_axis.set_xlabel("$x_1$", labelpad=-3)
        ellipse_axis.set_ylabel("$x_2$", labelpad=-7)
        ellipse_axis.set_xlim(-2.0, 2.0)
        ellipse_axis.set_ylim(-2.0, 2.0)
        for spine in ellipse_axis.spines.values():
            spine.set_visible(False)
        text_axis = figure.add_subplot(left[row, 2])
        text_axis.set_axis_off()
        if row == 0:
            text = f"If $X_1\\to X_2$:\ntarget reached, KL {entry['action_kl']:.0f}"
        else:
            text = f"If $X_2\\to X_1$:\nno effect, still optimal,\nKL {entry['action_kl']:.2g}"
        text_axis.text(0.0, 0.52, text, ha="left", va="center", fontsize=6.6)
    figure.text(0.275, 0.022, "grey: target $Q$; outline: result of the same action, $s=(1,2)$", ha="center", va="bottom", fontsize=5.7)

    trees = outer[1].subgridspec(1, 2, wspace=0.38)
    balance_axis = figure.add_subplot(trees[0, 0])
    one_axis = figure.add_subplot(trees[0, 1])
    roots_by_dag = instance_2["roots"]
    _draw_policy_tree(
        balance_axis,
        instance_2["balance_tree"],
        roots_by_dag,
        title="Balance rule:\n2 queries",
        caption="stopping at a shared action removes only the dashed query",
    )
    _draw_policy_tree(
        one_axis,
        instance_2["one_step_tree"],
        roots_by_dag,
        title="One-step rule:\n1 query",
        caption=None,
    )
    handles = [
        patches.Patch(facecolor=LIGHT, edgecolor="black", label="scale $X_1$ by 2"),
        patches.Patch(facecolor=DARK, edgecolor="black", label=f"scale $X_2$ by {instance_2['root_1_scale']:.2f}"),
        patches.Patch(facecolor="white", edgecolor="black", label="no shared action"),
    ]
    figure.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, 0.005), ncol=1, fontsize=5.2, handlelength=1.0, labelspacing=0.25)
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "intro_example.pdf", metadata={"Creator": "make_paper_figures.py", "Title": "Introductory causal-state matching example"})
    plt.close(figure)


def _box(axis: plt.Axes, xy: tuple[float, float], size: tuple[float, float], text: str, *, shaded: bool = False, diamond: bool = False, font_size: float = 6.8) -> None:
    x, y = xy
    width, height = size
    if diamond:
        patch = patches.Polygon([(x, y + height / 2), (x + width / 2, y + height), (x + width, y + height / 2), (x + width / 2, y)], closed=True, facecolor="white", edgecolor="black", linewidth=0.9)
    else:
        patch = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.014", facecolor=LIGHT if shaded else "white", edgecolor="black", linewidth=0.9)
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=font_size, wrap=True)


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, label: str | None = None) -> None:
    axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "black", "shrinkA": 1, "shrinkB": 1})
    if label:
        axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.035, label, ha="center", va="bottom", fontsize=6.0)


def _plot_overview(figures: Path) -> None:
    apply_publication_style(7.2)
    figure, axis = plt.subplots(figsize=(WIDTH, 2.3))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()
    source = (0.025, 0.61); candidates = (0.245, 0.61); actions = (0.465, 0.61); decision = (0.710, 0.59)
    response = (0.215, 0.13); query = (0.54, 0.13); returned = (0.875, 0.61); target = (0.50, 0.88)
    _box(axis, source, (0.17, 0.16), "Source moments\nand valid PDAG", shaded=True)
    _box(axis, candidates, (0.17, 0.16), "Surviving candidate\nmodels $C$")
    _box(axis, actions, (0.19, 0.16), "Optimal actions $A_B(H)$\nof each $H\\in C$")
    _box(axis, decision, (0.13, 0.19), "Shared optimal\naction?", diamond=True)
    _box(axis, returned, (0.10, 0.16), "Return it", shaded=True)
    _box(axis, query, (0.19, 0.16), "Query one node $v$\n(minimax or one-step rule)", font_size=6.3)
    _box(axis, response, (0.24, 0.16), "Response reveals $v$'s\nmechanism; keep the candidates\nconsistent with it", font_size=5.8)
    _box(axis, target, (0.17, 0.11), "Target moments", shaded=True)
    _arrow(axis, (source[0] + .17, .69), (candidates[0], .69))
    _arrow(axis, (candidates[0] + .17, .69), (actions[0], .69))
    _arrow(axis, (actions[0] + .19, .69), (decision[0], .685))
    _arrow(axis, (target[0] + .085, target[1]), (actions[0] + .095, actions[1] + .16))
    _arrow(axis, (decision[0] + .13, .685), (returned[0], .69), label="yes")
    _arrow(axis, (decision[0] + .065, decision[1]), (query[0] + .095, query[1] + .16), label="no")
    _arrow(axis, (query[0], .21), (response[0] + .24, .21))
    _arrow(axis, (response[0] + .12, response[1] + .16), (candidates[0] + .085, candidates[1]))
    note = dict(fontsize=5.5, color="#666666", style="italic")
    axis.text(candidates[0] + .085, .80, "initially the DAGs consistent with the PDAG", ha="center", **note)
    axis.text(actions[0] + .095, .54, "Section 5, Prop. 5.1", ha="center", **note)
    axis.text(decision[0] + .065, .50, "Def. 7.1", ha="center", **note)
    axis.text(query[0] + .095, .055, "Thm. 7.2, Alg. 2", ha="center", **note)
    axis.text(response[0] + .12, .045, "Thm. 6.1", ha="center", **note)
    axis.text(.975, .29, "Full recovery (Alg. 1)\nstops only when $|C|=1$", ha="right", **note)
    axis.text(returned[0] + .05, .81, "matching identified", ha="center", **note)
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "overview.pdf", metadata={"Creator": "make_paper_figures.py", "Title": "Distributional causal-state matching overview"})
    plt.close(figure)


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

    # No directories or figure files are created until every numerical assertion passes.
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
    figures = output_root / "Figures"
    results.mkdir(parents=True, exist_ok=True)
    (results / "intro_instances.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_intro(instance_1, instance_2, figures)
    _plot_overview(figures)
    print("All Part A assertions passed.")
    print(results / "intro_instances.json")
    print(figures / "intro_example.pdf")
    print(figures / "overview.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
