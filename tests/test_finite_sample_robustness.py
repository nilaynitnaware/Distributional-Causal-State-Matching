"""Deterministic design, graph, schema, and pipeline tests for robustness v1."""

from __future__ import annotations

import inspect
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from experiments.finite_sample_robustness.design import (
    derive_seed,
    load_config,
    task_specs,
)
from experiments.finite_sample_robustness.execution import (
    _failure_row,
    _sample_source,
    run_task,
)
from experiments.finite_sample_robustness.graph_conditions import (
    corrupt_graph,
    estimate_pc_cpdag,
    graph_metrics,
    skeleton,
)
from graphs.pdag import PartiallyDirectedGraph
from graphs.scm import LinearGaussianSCM
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation
from scripts.run_query_complexity_local_mechanism_campaign import _source


@pytest.fixture(scope="module")
def config():
    return load_config()


def _chain():
    dag = nx.DiGraph([(0, 1), (1, 2)])
    coefficients = np.zeros((3, 3))
    coefficients[0, 1] = 0.5
    coefficients[1, 2] = -0.4
    return dag, LinearGaussianSCM(dag, coefficients)


def test_full_condition_generation_and_one_factor_axes(config) -> None:
    tasks = task_specs(config, mode="full", seed=config["default_seed"])
    counts = {}
    for task in tasks:
        counts[task.campaign] = counts.get(task.campaign, 0) + 1
    assert len(tasks) == config["expected_full_tasks"] == 3630
    assert counts == {
        "campaign_a_n_obs": 900,
        "campaign_b_n_target": 900,
        "campaign_c_estimated_cpdag": 900,
        "campaign_d_controlled_corruption": 450,
        "campaign_e_nonrealisable": 450,
        "campaign_f_large_n": 30,
    }
    assert {task.N_obs for task in tasks if task.campaign == "campaign_a_n_obs"} == {250, 500, 1000, 2000, 5000, 10000}
    assert {task.N_target for task in tasks if task.campaign == "campaign_b_n_target"} == {250, 500, 1000, 2000, 5000, 10000}


def test_pilot_covers_every_campaign_two_seeds_and_large_n(config) -> None:
    tasks = task_specs(config, mode="pilot", seed=config["default_seed"])
    assert len(tasks) == 16
    assert {task.campaign for task in tasks} == {
        "campaign_a_n_obs", "campaign_b_n_target", "campaign_c_estimated_cpdag",
        "campaign_d_controlled_corruption", "campaign_e_nonrealisable",
        "campaign_f_large_n",
    }
    assert {task.replicate for task in tasks} == {0, 1}
    assert {task.n for task in tasks if task.campaign == "campaign_f_large_n"} == {80, 100}


def test_source_problem_pairing_and_seed_derivation_are_stable(config) -> None:
    first = task_specs(config, mode="full", seed=config["default_seed"])
    second = task_specs(config, mode="full", seed=config["default_seed"])
    assert first == second
    grouped = {}
    for task in first:
        coordinate = (task.n, task.graph_index, task.replicate, task.seed)
        grouped.setdefault(coordinate, set()).add(task.source_problem_id)
    assert all(len(values) == 1 for values in grouped.values())
    assert derive_seed(11, "a", 3) == derive_seed(11, "a", 3)
    assert derive_seed(11, "a", 3) != derive_seed(11, "a", 4)


def test_observational_sample_prefix_pairing_matches_stage_b(config) -> None:
    tasks = task_specs(config, mode="full", seed=config["default_seed"])
    high = next(task for task in tasks if task.campaign == "campaign_a_n_obs" and task.n == 10 and task.graph_index == 0 and task.replicate == 0 and task.N_obs == 10000)
    low = replace(high, N_obs=250)
    regime = config["graph_regimes"][0]
    source_config = dict(config); source_config["base_seed"] = high.seed
    scm, _ = _source(source_config, regime, high.n, high.replicate)
    np.testing.assert_array_equal(_sample_source(low, scm), _sample_source(high, scm)[:250])


def test_graph_metrics_detect_false_positive_and_false_negative() -> None:
    exact = PartiallyDirectedGraph({0, 1, 2}, set(), {(0, 1), (1, 2)})
    supplied = PartiallyDirectedGraph({0, 1, 2}, set(), {(0, 1), (0, 2)})
    dag = nx.DiGraph([(0, 1), (1, 2)])
    metrics = graph_metrics(supplied, exact, dag)
    assert metrics["skeleton_shd"] == 2
    assert metrics["false_positive_adjacencies"] == 1
    assert metrics["false_negative_adjacencies"] == 1
    assert metrics["adjacency_precision"] == metrics["adjacency_recall"] == 0.5
    assert not metrics["retains_all_true_adjacencies"]


def test_edge_deletion_and_addition_are_exact_and_deterministic() -> None:
    dag, _ = _chain()
    exact = cpdag_from_true_dag_for_simulation(dag)
    deletion = corrupt_graph(exact, dag, corruption_type="edge_deletion", severity="one", seed=4)
    assert deletion.status == "SUCCESS"
    assert len(skeleton(exact) - skeleton(deletion.graph)) == 1
    assert deletion == corrupt_graph(exact, dag, corruption_type="edge_deletion", severity="one", seed=4)
    addition = corrupt_graph(exact, dag, corruption_type="edge_addition", severity="one", seed=8)
    assert addition.status == "SUCCESS"
    assert skeleton(addition.graph) - skeleton(exact) == {(0, 2)}


def test_orientation_corruption_is_wrong_and_valid_pdag() -> None:
    dag, _ = _chain()
    exact = cpdag_from_true_dag_for_simulation(dag)
    corruption = corrupt_graph(exact, dag, corruption_type="orientation_error", severity="one", seed=9)
    assert corruption.status == "SUCCESS"
    assert isinstance(corruption.graph, PartiallyDirectedGraph)
    assert corruption.construction_used_hidden_truth
    assert any(not dag.has_edge(*edge) for edge in corruption.graph.directed_edges)


def test_estimated_cpdag_is_learned_from_samples_and_not_repaired() -> None:
    dag, _ = _chain()
    exact = cpdag_from_true_dag_for_simulation(dag)
    independent = np.random.default_rng(33).normal(size=(4000, 3))
    settings = {
        "significance_level": 0.01,
        "maximum_conditioning_set_size": 2,
        "minimum_effective_sample_size": 20,
        "covariance_estimator": "oas",
        "diagonal_loading": 1e-10,
        "minimum_eigenvalue": 1e-10,
    }
    estimated, metadata = estimate_pc_cpdag(independent, settings)
    assert metadata.provider == "estimated_pc_fisher_z"
    assert estimated != exact
    assert not skeleton(estimated)
    assert tuple(inspect.signature(estimate_pc_cpdag).parameters) == ("samples", "settings")


def test_failure_serialization_retains_identity(config) -> None:
    task = task_specs(config, mode="pilot", seed=config["default_seed"])[0]
    row = _failure_row(
        task,
        status="GRAPH_ESTIMATION_FAILED",
        stage="graph_preparation",
        error="deliberate",
        stage_times={"graph_preparation": 0.1},
    )
    assert row["source_problem_id"] == task.source_problem_id
    assert row["task_id"] == task.task_id
    assert row["status"] == "GRAPH_ESTIMATION_FAILED"
    assert row["catastrophic_failure"]


@pytest.mark.parametrize(
    "family",
    ["nonproportional_structural_coefficients", "external_spd_gaussian"],
)
def test_nonrealisable_evaluation_is_relative_to_exact_k_b(config, family) -> None:
    task = next(
        item for item in task_specs(config, mode="pilot", seed=config["default_seed"])
        if item.target_family == family
    )
    row = run_task(config, task, bootstrap_replicates=2)
    assert row["status"] == "SUCCESS"
    assert row["excess_KL"] == pytest.approx(row["final_true_KL"] - row["K_B_star"])
    assert row["K_B_star"] >= -1e-10


def test_small_full_pipeline_schema_and_no_production_semantic_change(config) -> None:
    task = next(
        item for item in task_specs(config, mode="pilot", seed=config["default_seed"])
        if item.campaign == "campaign_a_n_obs" and item.N_obs == 250
    )
    row = run_task(config, task, bootstrap_replicates=2)
    assert row["status"] == "SUCCESS"
    assert row["schema_version"] == "finite_sample_robustness_row_v1"
    assert row["method"] == "finite_sample_algorithm1"
    assert row["policy"] == "first_unresolved"
    assert row["total_observations"] == task.N_obs + task.N_target + row["T_q"] * task.N_int
    assert "scm" not in inspect.signature(__import__("query_complexity.finite_sample", fromlist=["run_finite_sample_algorithm1"]).run_finite_sample_algorithm1).parameters
