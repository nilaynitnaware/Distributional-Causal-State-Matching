"""Shared setup and exact solvers for the Small-MEC timing scripts."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

os.environ.update(
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)

import numpy as np

from research.matching_query_complexity import (
    candidate_map,
    compute_all_optimal_action_sets,
    enumerate_consistent_dags,
    reconstruct_all_candidate_scms,
    solve_dag_complexity,
    solve_matching_complexity,
)
from research.small_mec_matching_query.experiment import (
    CampaignConfig,
    SourceProblem,
    _graph,
    _stable_seed,
    _target,
    _unresolved_tie_count,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "outputs" / "small_mec_command_space_paired_v2"
INPUTS = [
    json.loads(line)
    for line in (DATA_ROOT / "frozen_inputs" / "instances.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
]
ARCHIVE = {
    row["instance_id"]: row
    for row in map(
        json.loads,
        (DATA_ROOT / "corrected_audit" / "per_instance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines(),
    )
}
CONFIG = CampaignConfig()
FIELDS = [
    "part",
    "instance_id",
    "n",
    "family",
    "target",
    "source_seed",
    "target_seed",
    "mec_size",
    "undirected_edges",
    "enumeration_s",
    "optimization_s",
    "match_s",
    "dag_s",
    "T_match",
    "T_DAG",
    "status",
    "load_start",
    "load_end",
    "details",
]

T = TypeVar("T")


def load() -> str:
    """Return the host's 1, 5, and 15 minute load averages."""
    return " ".join(f"{value:.2f}" for value in os.getloadavg())


def initialize_output(output: Path) -> None:
    """Create or replace a timing CSV and write its header."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writeheader()


def save(output: Path, record: dict[str, Any]) -> None:
    """Append one timing record, creating a header for a new CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        initialize_output(output)
    with output.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(
            handle, fieldnames=FIELDS, extrasaction="ignore"
        ).writerow(record)


def timed(function: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    value = function()
    return value, time.perf_counter() - start


def solve(
    record: dict[str, Any],
    cpdag: Any,
    source_mean: Any,
    source_covariance: Any,
    target_mean: Any,
    target_covariance: Any,
    expected: set[Any] | None = None,
) -> None:
    """Time enumeration, candidate optimization, and both exact recursions."""
    record["load_start"] = load()
    candidates, record["enumeration_s"] = timed(
        lambda: enumerate_consistent_dags(
            cpdag, max_undirected_edges=len(cpdag.undirected_edges)
        )
    )
    record["mec_size"] = len(candidates)
    record["undirected_edges"] = len(cpdag.undirected_edges)
    if expected is not None:
        assert {candidate.id for candidate in candidates} == expected

    candidates_by_id = candidate_map(candidates)
    candidate_scms = reconstruct_all_candidate_scms(
        candidates, np.asarray(source_mean), np.asarray(source_covariance)
    )
    optimization_arguments = dict(
        target_mean=np.asarray(target_mean),
        target_covariance=np.asarray(target_covariance),
        budget=CONFIG.budget,
        shift_bounds=CONFIG.shift_bounds,
        log_scale_bounds=CONFIG.log_scale_bounds,
        loss_atol=CONFIG.loss_atol,
        loss_rtol=CONFIG.loss_rtol,
        action_atol=CONFIG.action_atol,
        action_rtol=CONFIG.action_rtol,
    )
    action_sets, record["optimization_s"] = timed(
        lambda: compute_all_optimal_action_sets(
            candidate_scms, **optimization_arguments
        )
    )
    diagnostics = [
        diagnostic
        for action_set in action_sets.values()
        for diagnostic in action_set.oracle_result.degeneracy_diagnostics
        if not diagnostic.endswith(":zero_parental_predictor")
    ]
    if diagnostics or _unresolved_tie_count(action_sets):
        raise ValueError("uncertified optimizer geometry")

    matching, record["match_s"] = timed(
        lambda: solve_matching_complexity(
            candidates=candidates_by_id, action_sets=action_sets
        )
    )
    record["T_match"] = matching.complexity
    dag, record["dag_s"] = timed(
        lambda: solve_dag_complexity(candidates=candidates_by_id)
    )
    record["T_DAG"] = dag.complexity

    if record["part"] == "B":
        archived = ARCHIVE[record["instance_id"]]
        assert (matching.complexity, dag.complexity) == (
            archived["T_match_star_worst_case"],
            archived["T_DAG_star_worst_case"],
        )
    record["status"] = "complete"
    record["load_end"] = load()
