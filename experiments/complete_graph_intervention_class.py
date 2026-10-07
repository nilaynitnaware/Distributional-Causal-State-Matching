"""Focused paired intervention-class comparison on complete DAGs.

This module is intentionally independent of the paper ablation campaign.  It
constructs one mixed target per seed and passes that same immutable problem to
each class-restricted exact-moment search.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from math import exp
from pathlib import Path
from time import perf_counter
from typing import Literal

import matplotlib
import networkx as nx
import numpy as np
from scipy.optimize import minimize_scalar

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environments.blackbox import (
    EnvironmentQueryLedger,
    ExactMomentBlackBoxEnvironment,
)
from graphs.interventions import INTERVENTION_SEMANTICS, ShiftScaleIntervention
from graphs.noise import (
    derive_noise_variance_seed,
    generate_noise_variances,
    validate_noise_variance_regime,
)
from graphs.scm import LinearGaussianSCM, sample_coefficients
from graphs.support import DEFAULT_SUPPORT_ATOL, coordinate_support_masks
from matching.types import InterventionBackground, ParameterCoordinate
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation

FAMILY = "complete_graph_intervention_class"
SCHEMA_VERSION = 1
N_NODES = 10
RHO = 0.4
K = 4
SEEDS = tuple(range(10))
SUCCESS_THRESHOLDS = (1e-8, 5e-8, 1e-7)
PLOT_FLOOR = 1e-14
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260729
COEFFICIENT_ABS_BOUNDS = (0.10, 0.25)
SHIFT_MAGNITUDE_BOUNDS = (0.60, 1.00)
LOG_SCALE_MAGNITUDE_BOUNDS = (0.20, 0.40)
SHIFT_SEARCH_BOUNDS = (-1.5, 1.5)
LOG_SCALE_SEARCH_BOUNDS = (-0.6, 0.6)

SearchClass = Literal["shift_only", "scale_only", "shift_scale"]


@dataclass(frozen=True)
class Condition:
    label: str
    search_class: SearchClass
    budget: int
    comparison: str
    physical_result_label: str


CONDITIONS = (
    Condition("shift_only_k", "shift_only", K, "natural_capacity", "shift_only_k"),
    Condition("scale_only_k", "scale_only", K, "natural_capacity", "scale_only_k"),
    Condition(
        "shift_scale_2k",
        "shift_scale",
        2 * K,
        "natural_capacity",
        "shift_scale_2k",
    ),
    Condition(
        "shift_only_2k",
        "shift_only",
        2 * K,
        "equal_budget",
        "shift_only_2k",
    ),
    Condition(
        "scale_only_2k",
        "scale_only",
        2 * K,
        "equal_budget",
        "scale_only_2k",
    ),
    Condition(
        "shift_scale_2k_equal",
        "shift_scale",
        2 * K,
        "equal_budget",
        "shift_scale_2k",
    ),
)
PHYSICAL_CONDITIONS = tuple(
    condition
    for condition in CONDITIONS
    if condition.label == condition.physical_result_label
)


@dataclass(frozen=True)
class FocusedProblem:
    seed: int
    paired_problem_id: str
    subsystem_seeds: dict[str, int]
    topological_order: tuple[int, ...]
    root_node: int
    scm: LinearGaussianSCM
    target_intervention: ShiftScaleIntervention
    target_nodes: tuple[int, ...]
    target_mean: np.ndarray
    target_covariance: np.ndarray
    cpdag_directed_edges: tuple[tuple[int, int], ...]
    cpdag_undirected_edges: tuple[tuple[int, int], ...]
    problem_fingerprint: str
    true_target_kl: float
    noise_variance_regime: str = "homoskedastic"
    noise_variance_seed: int | None = None
    cross_noise_regime_pair_id: str = ""


def _seed_streams(seed: int) -> dict[str, int]:
    labels = (
        "graph",
        "coefficients",
        "target_support",
        "target_values",
        "observational_samples",
        "environment",
        "coordinate_search",
        "bootstrap",
        "policy",
    )
    children = np.random.SeedSequence(seed).spawn(len(labels))
    return {
        label: int(child.generate_state(1, dtype=np.uint64)[0])
        for label, child in zip(labels, children, strict=True)
    }


def _complete_dag(order: tuple[int, ...]) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(order)))
    dag.add_edges_from(
        (order[left], order[right])
        for left in range(len(order))
        for right in range(left + 1, len(order))
    )
    return dag


def is_complete_dag(dag: nx.DiGraph) -> bool:
    """Return whether every node pair has exactly one acyclic direction."""
    n_nodes = dag.number_of_nodes()
    return (
        nx.is_directed_acyclic_graph(dag)
        and dag.number_of_edges() == n_nodes * (n_nodes - 1) // 2
        and all(
            dag.has_edge(left, right) ^ dag.has_edge(right, left)
            for left in dag.nodes
            for right in dag.nodes
            if left < right
        )
    )


def generate_problem(
    seed: int,
    *,
    noise_variance_regime: str = "homoskedastic",
    noise_variance_seed: int | None = None,
) -> FocusedProblem:
    """Generate one complete-DAG problem with an exactly realizable mixed target."""
    streams = _seed_streams(seed)
    validate_noise_variance_regime(noise_variance_regime)
    effective_noise_seed = (
        noise_variance_seed
        if noise_variance_seed is not None
        else derive_noise_variance_seed(seed)
    )
    graph_rng = np.random.default_rng(streams["graph"])
    order = tuple(int(node) for node in graph_rng.permutation(N_NODES))
    dag = _complete_dag(order)
    coefficients = sample_coefficients(
        dag,
        np.random.default_rng(streams["coefficients"]),
        *COEFFICIENT_ABS_BOUNDS,
    )
    scm = LinearGaussianSCM(
        dag,
        coefficients,
        noise_variances=generate_noise_variances(
            N_NODES,
            noise_variance_regime,
            effective_noise_seed,
        ),
    )
    eligible = tuple(sorted(scm.scale_eligible_nodes))
    target_rng = np.random.default_rng(streams["target_support"])
    target_nodes = tuple(
        sorted(int(node) for node in target_rng.choice(eligible, K, replace=False))
    )
    value_rng = np.random.default_rng(streams["target_values"])
    shifts = np.zeros(N_NODES)
    log_scales = np.zeros(N_NODES)
    shifts[list(target_nodes)] = value_rng.uniform(
        *SHIFT_MAGNITUDE_BOUNDS, size=K
    ) * value_rng.choice((-1.0, 1.0), size=K)
    log_scales[list(target_nodes)] = value_rng.uniform(
        *LOG_SCALE_MAGNITUDE_BOUNDS, size=K
    ) * value_rng.choice((-1.0, 1.0), size=K)
    target = ShiftScaleIntervention(shifts, np.exp(log_scales))
    scm.validate_target_intervention(target)
    target_mean, target_covariance = scm.interventional_moments(target)
    true_mean, true_covariance = scm.interventional_moments(target)
    true_target_kl = gaussian_kl_target_to_candidate(
        target_mean,
        target_covariance,
        true_mean,
        true_covariance,
    )
    cpdag = cpdag_from_true_dag_for_simulation(dag)
    cross_noise_regime_pair_id = f"complete-n10-rho0.4-seed-{seed:02d}"
    paired_problem_id = (
        cross_noise_regime_pair_id
        if noise_variance_regime == "homoskedastic"
        else f"{cross_noise_regime_pair_id}-noise-{noise_variance_regime}"
    )
    fingerprint = _fingerprint(
        {
            "paired_problem_id": paired_problem_id,
            "subsystem_seeds": streams,
            "order": order,
            "coefficients": coefficients.tolist(),
            "target_shift": target.shift.tolist(),
            "target_scale": target.scale.tolist(),
            "target_mean": target_mean.tolist(),
            "target_covariance": target_covariance.tolist(),
        }
    )
    return FocusedProblem(
        seed=seed,
        paired_problem_id=paired_problem_id,
        subsystem_seeds=streams,
        topological_order=order,
        root_node=order[0],
        scm=scm,
        target_intervention=target,
        target_nodes=target_nodes,
        target_mean=target_mean,
        target_covariance=target_covariance,
        cpdag_directed_edges=tuple(sorted(cpdag.directed_edges)),
        cpdag_undirected_edges=tuple(sorted(cpdag.undirected_edges)),
        problem_fingerprint=fingerprint,
        true_target_kl=true_target_kl,
        noise_variance_regime=noise_variance_regime,
        noise_variance_seed=effective_noise_seed,
        cross_noise_regime_pair_id=cross_noise_regime_pair_id,
    )


def candidate_coordinates(
    problem: FocusedProblem,
    search_class: SearchClass,
) -> tuple[ParameterCoordinate, ...]:
    """Build the production candidate list before any search evaluation."""
    shifts = tuple(ParameterCoordinate(node, "shift") for node in range(N_NODES))
    scales = tuple(
        ParameterCoordinate(node, "log_scale")
        for node in sorted(problem.scm.scale_eligible_nodes)
    )
    if search_class == "shift_only":
        return shifts
    if search_class == "scale_only":
        return scales
    if search_class == "shift_scale":
        return shifts + scales
    raise ValueError(f"unknown search class: {search_class}")


def _coordinate_value(
    intervention: ShiftScaleIntervention,
    coordinate: ParameterCoordinate,
) -> float:
    if coordinate.kind == "shift":
        return float(intervention.shift[coordinate.node])
    return float(np.log(intervention.scale[coordinate.node]))


def _with_coordinate(
    intervention: ShiftScaleIntervention,
    coordinate: ParameterCoordinate,
    value: float,
) -> ShiftScaleIntervention:
    shifts = intervention.shift.copy()
    scales = intervention.scale.copy()
    if coordinate.kind == "shift":
        shifts[coordinate.node] = value
    else:
        scales[coordinate.node] = exp(value)
    return ShiftScaleIntervention(shifts, scales)


def _support(intervention: ShiftScaleIntervention) -> tuple[str, ...]:
    shifted, scaled = coordinate_support_masks(
        intervention.shift,
        intervention.scale,
    )
    return tuple(
        [f"shift:{node}" for node in np.flatnonzero(shifted)]
        + [f"log_scale:{node}" for node in np.flatnonzero(scaled)]
    )


def _query_kl(
    environment: ExactMomentBlackBoxEnvironment,
    intervention: ShiftScaleIntervention,
    problem: FocusedProblem,
    *,
    background: ShiftScaleIntervention,
    kind: Literal["baseline", "candidate"],
) -> float:
    result = environment.query(
        intervention,
        kind=kind,
        background_id=InterventionBackground.from_intervention(background).identifier,
    )
    return gaussian_kl_target_to_candidate(
        problem.target_mean,
        problem.target_covariance,
        result.mean,
        result.covariance,
    )


def run_search(
    problem: FocusedProblem,
    condition: Condition,
    *,
    max_accepted_updates: int = 160,
    improvement_tolerance: float = 1e-12,
    convergence_tolerance: float = 1e-10,
    refinement_tolerance: float = 2e-8,
) -> dict[str, object]:
    """Run audited exact-moment greedy coordinate descent for one condition."""
    if condition.label != condition.physical_result_label:
        raise ValueError("alias conditions do not trigger duplicate computation.")
    start = perf_counter()
    coordinates = candidate_coordinates(problem, condition.search_class)
    budget = min(condition.budget, len(coordinates))
    ledger = EnvironmentQueryLedger(max_query_batches=None)
    environment = ExactMomentBlackBoxEnvironment(
        problem.scm,
        instance_id=f"{problem.paired_problem_id}:{condition.label}",
        ledger=ledger,
        environment_semantics=INTERVENTION_SEMANTICS,
        numerical_configuration={"backend": "numpy", "device": "cpu"},
    )
    current = ShiftScaleIntervention.identity(N_NODES)
    current_kl = _query_kl(
        environment,
        current,
        problem,
        background=current,
        kind="baseline",
    )
    initial_kl = current_kl
    accepted_updates = 0
    accepted_new = 0
    accepted_existing = 0
    evaluated_counts: Counter[str] = Counter()
    evaluated_nodes: dict[str, set[int]] = {"shift": set(), "log_scale": set()}
    accepted_trace: list[dict[str, object]] = []
    termination_reason = "maximum_accepted_updates"

    while accepted_updates < max_accepted_updates:
        if current_kl <= convergence_tolerance:
            termination_reason = "kl_tolerance"
            break
        current_support = set(_support(current))
        best: (
            tuple[float, ParameterCoordinate, float, ShiftScaleIntervention] | None
        ) = None
        for coordinate in coordinates:
            key = f"{coordinate.kind}:{coordinate.node}"
            active = key in current_support
            if not active and len(current_support) >= budget:
                continue
            lower, upper = (
                SHIFT_SEARCH_BOUNDS
                if coordinate.kind == "shift"
                else LOG_SCALE_SEARCH_BOUNDS
            )
            old_value = _coordinate_value(current, coordinate)

            def objective(
                value: float,
                *,
                coord: ParameterCoordinate = coordinate,
                coord_key: str = key,
                background: ShiftScaleIntervention = current,
            ) -> float:
                evaluated_counts[coord_key] += 1
                evaluated_nodes[coord.kind].add(coord.node)
                proposal = _with_coordinate(background, coord, float(value))
                if proposal.support_cost() > budget:
                    return float("inf")
                return _query_kl(
                    environment,
                    proposal,
                    problem,
                    background=background,
                    kind="candidate",
                )

            result = minimize_scalar(
                objective,
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": refinement_tolerance, "maxiter": 80},
            )
            proposals = [(float(result.fun), float(result.x))]
            for value in (old_value, 0.0):
                proposals.append((objective(value), value))
            coordinate_kl, selected_value = min(proposals, key=lambda item: item[0])
            proposal = _with_coordinate(current, coordinate, selected_value)
            if best is None or coordinate_kl < best[0]:
                best = (coordinate_kl, coordinate, selected_value, proposal)

        if best is None:
            termination_reason = "support_budget_exhausted"
            break
        best_kl, coordinate, selected_value, proposal = best
        improvement = current_kl - best_kl
        if improvement <= improvement_tolerance:
            termination_reason = "no_improving_coordinate"
            break
        was_active = f"{coordinate.kind}:{coordinate.node}" in current_support
        now_active = abs(selected_value) > DEFAULT_SUPPORT_ATOL
        if not was_active and now_active:
            accepted_new += 1
        else:
            accepted_existing += 1
        accepted_updates += 1
        accepted_trace.append(
            {
                "update": accepted_updates,
                "coordinate": f"{coordinate.kind}:{coordinate.node}",
                "new_coordinate": bool(not was_active and now_active),
                "selected_value": selected_value,
                "kl_before": current_kl,
                "kl_after": best_kl,
                "improvement": improvement,
            }
        )
        current = proposal
        current_kl = best_kl
    else:
        termination_reason = "maximum_accepted_updates"

    final_query = environment.query(
        current,
        kind="candidate",
        background_id=InterventionBackground.from_intervention(current).identifier,
    )
    final_kl = gaussian_kl_target_to_candidate(
        problem.target_mean,
        problem.target_covariance,
        final_query.mean,
        final_query.covariance,
    )
    mean_error = float(np.linalg.norm(problem.target_mean - final_query.mean))
    covariance_error = float(
        np.linalg.norm(problem.target_covariance - final_query.covariance, ord="fro")
    )
    final_support = _support(current)
    shift_support = tuple(item for item in final_support if item.startswith("shift:"))
    scale_support = tuple(
        item for item in final_support if item.startswith("log_scale:")
    )
    absolute_reduction = initial_kl - final_kl
    elapsed = perf_counter() - start
    run_id = f"{problem.paired_problem_id}-{condition.label}"
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_family": FAMILY,
        "run_id": run_id,
        "paired_problem_id": problem.paired_problem_id,
        "condition": condition.label,
        "search_class": condition.search_class,
        "comparison": condition.comparison,
        "physical_result_label": condition.physical_result_label,
        "problem_fingerprint": problem.problem_fingerprint,
        "seed": problem.seed,
        "subsystem_seeds": problem.subsystem_seeds,
        "intervention_semantics": INTERVENTION_SEMANTICS,
        "n_nodes": N_NODES,
        "rho": RHO,
        "k": K,
        "graph_family": "complete",
        "graph_edge_count": problem.scm.dag.number_of_edges(),
        "topological_order": list(problem.topological_order),
        "root_node": problem.root_node,
        "supplied_exact_cpdag": {
            "directed_edges": [list(edge) for edge in problem.cpdag_directed_edges],
            "undirected_edges": [list(edge) for edge in problem.cpdag_undirected_edges],
        },
        "target_nodes": list(problem.target_nodes),
        "target_support": list(_support(problem.target_intervention)),
        "target_shift": problem.target_intervention.shift.tolist(),
        "target_scale": problem.target_intervention.scale.tolist(),
        "target_mean": problem.target_mean.tolist(),
        "target_covariance": problem.target_covariance.tolist(),
        "true_target_kl": problem.true_target_kl,
        "candidate_coordinates": [
            f"{coordinate.kind}:{coordinate.node}" for coordinate in coordinates
        ],
        "evaluated_coordinate_counts": dict(sorted(evaluated_counts.items())),
        "evaluated_shift_nodes": sorted(evaluated_nodes["shift"]),
        "evaluated_scale_nodes": sorted(evaluated_nodes["log_scale"]),
        "initial_target_to_candidate_kl": initial_kl,
        "final_target_to_candidate_kl": final_kl,
        "absolute_kl_reduction": absolute_reduction,
        "relative_kl_reduction": (
            absolute_reduction / initial_kl if initial_kl > 0.0 else 0.0
        ),
        "success_1e-8": final_kl <= 1e-8,
        "success_5e-8": final_kl <= 5e-8,
        "success_1e-7": final_kl <= 1e-7,
        "mean_error": mean_error,
        "covariance_error": covariance_error,
        "irreducible_residual_components": {
            "mean_error": mean_error
            if condition.search_class != "shift_only"
            else None,
            "covariance_error": (
                covariance_error if condition.search_class != "scale_only" else None
            ),
            "reported_mean_error": mean_error,
            "reported_covariance_error": covariance_error,
        },
        "final_support_size": len(final_support),
        "final_coordinate_budget": budget,
        "shift_support_size": len(shift_support),
        "scale_support_size": len(scale_support),
        "accepted_update_count": accepted_updates,
        "accepted_new_coordinate_count": accepted_new,
        "accepted_existing_coordinate_update_count": accepted_existing,
        "candidate_evaluations": int(sum(evaluated_counts.values())),
        "environment_calls": ledger.algorithm_environment_calls,
        "executed_environment_queries": ledger.executed_environment_queries,
        "environment_ledger": ledger.snapshot(),
        "wall_time_seconds": elapsed,
        "termination_reason": termination_reason,
        "recovered_support": list(final_support),
        "recovered_shift": current.shift.tolist(),
        "recovered_scale": current.scale.tolist(),
        "accepted_trace": accepted_trace,
    }


def validate_problem(problem: FocusedProblem) -> None:
    """Apply all pre-execution scientific invariants to one problem."""
    if INTERVENTION_SEMANTICS != "mechanism_scale_noise_fixed_v1":
        raise AssertionError("incorrect intervention semantics")
    if not is_complete_dag(problem.scm.dag):
        raise AssertionError("graph is not a complete DAG")
    if problem.root_node in problem.target_nodes:
        raise AssertionError("mixed target includes a root node")
    if len(problem.target_nodes) != K:
        raise AssertionError("mixed target does not contain exactly k nodes")
    if len(_support(problem.target_intervention)) != 2 * K:
        raise AssertionError("mixed target does not contain exactly 2k coordinates")
    if problem.true_target_kl > 1e-12:
        raise AssertionError("true target intervention does not reproduce the target")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _analysis_rows(
    physical_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_key = {
        (str(row["paired_problem_id"]), str(row["condition"])): row
        for row in physical_results
    }
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            paired_id = f"complete-n10-rho0.4-seed-{seed:02d}"
            source = by_key[(paired_id, condition.physical_result_label)]
            row = dict(source)
            row["analysis_condition"] = condition.label
            row["comparison"] = condition.comparison
            row["reused_physical_result"] = (
                condition.label != condition.physical_result_label
            )
            rows.append(row)
    return rows


def _flat_row(row: dict[str, object]) -> dict[str, object]:
    keys = (
        "run_id",
        "paired_problem_id",
        "analysis_condition",
        "physical_result_label",
        "reused_physical_result",
        "comparison",
        "seed",
        "intervention_semantics",
        "problem_fingerprint",
        "initial_target_to_candidate_kl",
        "final_target_to_candidate_kl",
        "absolute_kl_reduction",
        "relative_kl_reduction",
        "success_1e-8",
        "success_5e-8",
        "success_1e-7",
        "mean_error",
        "covariance_error",
        "final_support_size",
        "final_coordinate_budget",
        "shift_support_size",
        "scale_support_size",
        "accepted_update_count",
        "accepted_new_coordinate_count",
        "accepted_existing_coordinate_update_count",
        "candidate_evaluations",
        "environment_calls",
        "executed_environment_queries",
        "wall_time_seconds",
        "termination_reason",
        "target_support",
        "recovered_support",
    )
    result = {key: row[key] for key in keys}
    for key in ("target_support", "recovered_support"):
        result[key] = json.dumps(result[key], separators=(",", ":"))
    return result


def _bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int,
    statistic: Literal["mean", "median"] = "mean",
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    samples = values[indices]
    estimates = (
        np.mean(samples, axis=1) if statistic == "mean" else np.median(samples, axis=1)
    )
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def analyze(
    physical_results: list[dict[str, object]],
    output_directory: Path,
) -> dict[str, object]:
    """Generate paired tables, bootstrap summaries, figures, and report."""
    rows = _analysis_rows(physical_results)
    flat_rows = [_flat_row(row) for row in rows]
    _write_csv(output_directory / "flat_results.csv", flat_rows)

    grouped: list[dict[str, object]] = []
    for index, condition in enumerate(CONDITIONS):
        condition_rows = [
            row for row in rows if row["analysis_condition"] == condition.label
        ]
        kl = np.array(
            [float(row["final_target_to_candidate_kl"]) for row in condition_rows]
        )
        ci_low, ci_high = _bootstrap_ci(kl, seed=BOOTSTRAP_SEED + index)
        median_low, median_high = _bootstrap_ci(
            kl,
            seed=BOOTSTRAP_SEED + 100 + index,
            statistic="median",
        )
        group = {
            "condition": condition.label,
            "comparison": condition.comparison,
            "run_count": len(condition_rows),
            "mean_final_kl": float(np.mean(kl)),
            "median_final_kl": float(np.median(kl)),
            "mean_final_kl_ci95_low": ci_low,
            "mean_final_kl_ci95_high": ci_high,
            "median_final_kl_ci95_low": median_low,
            "median_final_kl_ci95_high": median_high,
            "success_rate_1e-8": float(
                np.mean([bool(row["success_1e-8"]) for row in condition_rows])
            ),
            "success_rate_5e-8": float(
                np.mean([bool(row["success_5e-8"]) for row in condition_rows])
            ),
            "success_rate_1e-7": float(
                np.mean([bool(row["success_1e-7"]) for row in condition_rows])
            ),
            "mean_mean_error": float(
                np.mean([float(row["mean_error"]) for row in condition_rows])
            ),
            "mean_covariance_error": float(
                np.mean([float(row["covariance_error"]) for row in condition_rows])
            ),
            "lowest_final_kl_count": 0,
        }
        grouped.append(group)

    by_label = {
        label: {
            str(row["paired_problem_id"]): row
            for row in rows
            if row["analysis_condition"] == label
        }
        for label in (condition.label for condition in CONDITIONS)
    }
    comparison_labels = {
        "natural_capacity": ("shift_only_k", "scale_only_k", "shift_scale_2k"),
        "equal_budget": (
            "shift_only_2k",
            "scale_only_2k",
            "shift_scale_2k_equal",
        ),
    }
    lowest_counts: Counter[str] = Counter()
    for labels in comparison_labels.values():
        for paired_id in by_label[labels[0]]:
            values = {
                label: float(by_label[label][paired_id]["final_target_to_candidate_kl"])
                for label in labels
            }
            minimum = min(values.values())
            for label, value in values.items():
                if np.isclose(value, minimum, rtol=1e-10, atol=1e-15):
                    lowest_counts[label] += 1
    for group in grouped:
        group["lowest_final_kl_count"] = lowest_counts[str(group["condition"])]

    paired_rows: list[dict[str, object]] = []
    paired_summary: list[dict[str, object]] = []
    for comparison, labels in comparison_labels.items():
        shift_label, scale_label, mixed_label = labels
        shift_deltas = []
        scale_deltas = []
        for paired_id in sorted(by_label[shift_label]):
            shift_kl = float(
                by_label[shift_label][paired_id]["final_target_to_candidate_kl"]
            )
            scale_kl = float(
                by_label[scale_label][paired_id]["final_target_to_candidate_kl"]
            )
            mixed_kl = float(
                by_label[mixed_label][paired_id]["final_target_to_candidate_kl"]
            )
            shift_delta = shift_kl - mixed_kl
            scale_delta = scale_kl - mixed_kl
            shift_deltas.append(shift_delta)
            scale_deltas.append(scale_delta)
            paired_rows.append(
                {
                    "comparison": comparison,
                    "paired_problem_id": paired_id,
                    "shift_condition": shift_label,
                    "scale_condition": scale_label,
                    "mixed_condition": mixed_label,
                    "shift_final_kl": shift_kl,
                    "scale_final_kl": scale_kl,
                    "mixed_final_kl": mixed_kl,
                    "delta_shift": shift_delta,
                    "delta_scale": scale_delta,
                }
            )
        for delta_name, values in (
            ("delta_shift", shift_deltas),
            ("delta_scale", scale_deltas),
        ):
            array = np.asarray(values)
            low, high = _bootstrap_ci(
                array,
                seed=BOOTSTRAP_SEED + 1000 + len(paired_summary),
            )
            paired_summary.append(
                {
                    "comparison": comparison,
                    "paired_difference": delta_name,
                    "mean": float(np.mean(array)),
                    "median": float(np.median(array)),
                    "mean_ci95_low": low,
                    "mean_ci95_high": high,
                    "positive_count": int(np.count_nonzero(array > 0.0)),
                    "pair_count": int(array.size),
                }
            )

    _write_csv(output_directory / "grouped_results.csv", grouped)
    _write_json(
        output_directory / "grouped_results.json",
        {
            "schema_version": SCHEMA_VERSION,
            "intervention_semantics": INTERVENTION_SEMANTICS,
            "bootstrap": {
                "method": "paired_problem_resampling_with_replacement",
                "replicates": BOOTSTRAP_REPLICATES,
                "confidence_level": 0.95,
                "seed": BOOTSTRAP_SEED,
            },
            "groups": grouped,
            "paired_summaries": paired_summary,
        },
    )
    _write_csv(output_directory / "paired_comparisons.csv", paired_rows)
    _write_latex(output_directory / "intervention_class_table.tex", grouped)
    _plot_metric(
        rows,
        output_directory / "final_kl.png",
        "final_target_to_candidate_kl",
        "Final target-to-candidate KL",
        log_scale=True,
    )
    _plot_metric(
        rows,
        output_directory / "mean_error.png",
        "mean_error",
        "Mean error (L2)",
        log_scale=True,
    )
    _plot_metric(
        rows,
        output_directory / "covariance_error.png",
        "covariance_error",
        "Covariance error (Frobenius)",
        log_scale=True,
    )
    _plot_success(rows, output_directory / "success_rate.png")
    report = _interpretation_report(grouped, paired_summary)
    (output_directory / "interpretation.md").write_text(report, encoding="utf-8")
    return {
        "groups": grouped,
        "paired_summaries": paired_summary,
        "analysis_row_count": len(rows),
        "physical_result_count": len(physical_results),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(path: Path, groups: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Condition & Mean KL & Median KL & $<10^{-8}$ & Mean err. & Cov. err. \\",
        r"\midrule",
    ]
    for group in groups:
        label = str(group["condition"]).replace("_", r"\_")
        lines.append(
            f"{label} & {float(group['mean_final_kl']):.3e} & "
            f"{float(group['median_final_kl']):.3e} & "
            f"{float(group['success_rate_1e-8']):.2f} & "
            f"{float(group['mean_mean_error']):.3e} & "
            f"{float(group['mean_covariance_error']):.3e} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_metric(
    rows: list[dict[str, object]],
    path: Path,
    metric: str,
    ylabel: str,
    *,
    log_scale: bool,
) -> None:
    labels = [condition.label for condition in CONDITIONS]
    data = [
        [
            max(float(row[metric]), PLOT_FLOOR) if log_scale else float(row[metric])
            for row in rows
            if row["analysis_condition"] == label
        ]
        for label in labels
    ]
    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    axis.boxplot(data, tick_labels=labels, showmeans=True)
    for position, values in enumerate(data, start=1):
        axis.scatter(
            np.full(len(values), position),
            values,
            s=16,
            alpha=0.65,
            color="tab:blue",
        )
    if log_scale:
        axis.set_yscale("log")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    if metric == "final_target_to_candidate_kl":
        axis.text(
            0.01,
            0.01,
            f"Visualization floor = {PLOT_FLOOR:g}; raw zeros preserved in tables.",
            transform=axis.transAxes,
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_success(rows: list[dict[str, object]], path: Path) -> None:
    labels = [condition.label for condition in CONDITIONS]
    x = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    for offset, (key, threshold) in enumerate(
        (("success_1e-8", 1e-8), ("success_5e-8", 5e-8), ("success_1e-7", 1e-7))
    ):
        rates = [
            np.mean(
                [bool(row[key]) for row in rows if row["analysis_condition"] == label]
            )
            for label in labels
        ]
        axis.bar(x + (offset - 1) * width, rates, width, label=f"{threshold:g}")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Success rate")
    axis.legend(title="KL threshold")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _interpretation_report(
    groups: list[dict[str, object]],
    paired: list[dict[str, object]],
) -> str:
    by_group = {str(group["condition"]): group for group in groups}
    natural = {
        str(row["paired_difference"]): row
        for row in paired
        if row["comparison"] == "natural_capacity"
    }
    equal = {
        str(row["paired_difference"]): row
        for row in paired
        if row["comparison"] == "equal_budget"
    }
    mixed = by_group["shift_scale_2k"]
    return f"""# Complete-DAG intervention-class comparison

All conditions used the same 10 paired mixed targets under
`{INTERVENTION_SEMANTICS}`.  Each target changed both shift and mechanism scale
on exactly four non-root nodes.  The two mixed labels reuse the same physical
search results.

The mixed search had mean final KL {float(mixed["mean_final_kl"]):.3e} and
success rate {float(mixed["success_rate_1e-8"]):.0%} at 1e-8.  In the
natural-capacity comparison, the paired mean KL advantage over shift-only was
{float(natural["delta_shift"]["mean"]):.3e} (95% paired-bootstrap CI
[{float(natural["delta_shift"]["mean_ci95_low"]):.3e},
{float(natural["delta_shift"]["mean_ci95_high"]):.3e}]); the advantage over
scale-only was {float(natural["delta_scale"]["mean"]):.3e} (CI
[{float(natural["delta_scale"]["mean_ci95_low"]):.3e},
{float(natural["delta_scale"]["mean_ci95_high"]):.3e}]).

With equal 2k budgets, the paired advantages were
{float(equal["delta_shift"]["mean"]):.3e} over shift-only and
{float(equal["delta_scale"]["mean"]):.3e} over scale-only.  Shift-only residual
covariance error and scale-only residual mean error are reported per run and in
the grouped tables.  These empirical results compare intervention capability
on identical tasks; no expected outcome was encoded in the optimizer.

The final-KL figure uses a positive plotting floor of {PLOT_FLOOR:g} only for
log visualization.  CSV and JSON tables retain raw floating-point zeros.
"""


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def experiment_configuration() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_family": FAMILY,
        "intervention_semantics": INTERVENTION_SEMANTICS,
        "n_nodes": N_NODES,
        "rho": RHO,
        "k": K,
        "seeds": list(SEEDS),
        "graph": {
            "family": "complete_dag",
            "construction": "random_topological_order_all_forward_edges",
            "coefficient_abs_bounds": list(COEFFICIENT_ABS_BOUNDS),
            "initial_structure": "supplied_exact_cpdag",
        },
        "target": {
            "family": "realisable_mixed",
            "selected_nonroot_nodes": K,
            "coordinates": 2 * K,
            "shift_magnitude_bounds": list(SHIFT_MAGNITUDE_BOUNDS),
            "log_scale_magnitude_bounds": list(LOG_SCALE_MAGNITUDE_BOUNDS),
            "moments": "exact_population",
        },
        "search": {
            "objective": "target_to_candidate_gaussian_kl",
            "provider": "exact_moment_black_box_environment",
            "coordinate_rule": "best_improvement",
            "shift_bounds": list(SHIFT_SEARCH_BOUNDS),
            "log_scale_bounds": list(LOG_SCALE_SEARCH_BOUNDS),
            "existing_coordinates_can_be_repeatedly_updated": True,
            "max_accepted_updates": 160,
        },
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "success_thresholds": list(SUCCESS_THRESHOLDS),
        "analysis": {
            "pairing_key": "paired_problem_id",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "plot_floor": PLOT_FLOOR,
        },
    }


def execute(output_directory: Path) -> dict[str, object]:
    """Validate, execute five unique searches per seed, and analyze six labels."""
    configuration = experiment_configuration()
    config_fingerprint = _fingerprint(configuration)
    _write_json(output_directory / "configuration.json", configuration)
    physical_results: list[dict[str, object]] = []
    manifest_runs: list[dict[str, object]] = []
    for seed in SEEDS:
        problem = generate_problem(seed)
        validate_problem(problem)
        for condition in PHYSICAL_CONDITIONS:
            result = run_search(problem, condition)
            result["config_fingerprint"] = config_fingerprint
            path = output_directory / "results" / f"{result['run_id']}.json"
            _write_json(path, result)
            physical_results.append(result)
            manifest_runs.append(
                {
                    "run_id": result["run_id"],
                    "paired_problem_id": problem.paired_problem_id,
                    "condition": condition.label,
                    "problem_fingerprint": problem.problem_fingerprint,
                    "result_file": str(path.relative_to(output_directory)),
                    "status": "completed",
                }
            )
    analysis = analyze(physical_results, output_directory)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_family": FAMILY,
        "intervention_semantics": INTERVENTION_SEMANTICS,
        "config_fingerprint": config_fingerprint,
        "expected_physical_run_count": len(SEEDS) * len(PHYSICAL_CONDITIONS),
        "completed_physical_run_count": len(physical_results),
        "analysis_condition_run_count": len(SEEDS) * len(CONDITIONS),
        "mixed_alias": {
            "shift_scale_2k_equal": "shift_scale_2k",
            "duplicate_computation_avoided": True,
        },
        "runs": manifest_runs,
        "analysis": analysis,
    }
    _write_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "focused" / FAMILY,
    )
    args = parser.parse_args()
    manifest = execute(args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
