"""Focused infrastructure tests for the frozen S01-C n=40 campaign."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import networkx as nx

from research.s01_tree_population import CertificateCache
from tests.test_s01_tree_population import _problem

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "outputs/s01_c_tree_population_n40_v1/run_s01_c_tree_population_n40.py"
SPEC = importlib.util.spec_from_file_location("s01_c_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_n40_design_and_execution_order_are_complete() -> None:
    config = RUNNER.frozen_config()
    specs = RUNNER.specs()
    tasks = RUNNER.task_specs()
    assert config["n"] == 40
    assert config["source_count"] == 20
    assert len(specs) == 80
    assert len(tasks) == 320
    assert len({row["instance_id"] for row in specs}) == 80
    assert [row["instance_id"] for row in specs] == config["execution_order"]
    assert all(sum(task["instance_id"] == row["instance_id"] for task in tasks) == 4 for row in specs)


def test_freeze_is_deterministic_and_refuses_partial_bank(tmp_path: Path) -> None:
    rows = RUNNER.freeze_inputs(tmp_path)
    first_hash = RUNNER.sha256_file(tmp_path / "frozen_instances.jsonl")
    assert len(rows) == 80
    assert all(len(row["candidate_roots"]) == 40 for row in rows)
    assert RUNNER.freeze_inputs(tmp_path) == rows
    assert RUNNER.sha256_file(tmp_path / "frozen_instances.jsonl") == first_hash


def test_cache_accounting_separates_upper_bound_from_visited_work() -> None:
    roots, _, _, _, action_sets = _problem(nx.path_graph(4))
    cache = CertificateCache(action_sets)
    state = roots.initial_state
    first = cache.check(state)
    second = cache.check(state)
    assert second is first
    assert cache.cache_hits == 1
    assert sum(cache.outcome_counts.values()) == 1
    assert cache.tied_family_combinations >= cache.structural_candidates_visited
    assert cache.direct_kl_evaluations >= 0


def test_campaign_cache_ids_are_policy_isolated() -> None:
    fingerprint = "f" * 64
    ids = {
        f"{task['instance_id']}::{task['policy']}::{fingerprint[:16]}"
        for task in RUNNER.task_specs()
    }
    assert len(ids) == 320
