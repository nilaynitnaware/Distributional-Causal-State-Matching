"""Paired-axis invariants that specifically protect target-sample isolation."""

from collections import defaultdict

from experiments.finite_sample_robustness.design import load_config, task_specs
from experiments.finite_sample_robustness.execution import run_task


def test_pilot_low_high_axes_reuse_the_same_source_problem() -> None:
    config = load_config()
    tasks = task_specs(config, mode="pilot", seed=int(config["default_seed"]))
    by_campaign = defaultdict(list)
    for task in tasks:
        by_campaign[task.campaign].append(task)
    for campaign in (
        "campaign_a_n_obs",
        "campaign_b_n_target",
        "campaign_c_estimated_cpdag",
    ):
        assert len({task.source_problem_id for task in by_campaign[campaign]}) == 1
    nonproportional = [
        task for task in by_campaign["campaign_e_nonrealisable"]
        if task.target_family == "nonproportional_structural_coefficients"
    ]
    external = [
        task for task in by_campaign["campaign_e_nonrealisable"]
        if task.target_family == "external_spd_gaussian"
    ]
    assert len({task.source_problem_id for task in nonproportional}) == 1
    assert len({task.source_problem_id for task in external}) == 1


def test_target_sample_axis_cannot_change_structural_rng_or_recovery() -> None:
    config = load_config()
    tasks = [
        task for task in task_specs(config, mode="pilot", seed=int(config["default_seed"]))
        if task.campaign == "campaign_b_n_target"
    ]
    assert len(tasks) == 2
    rows = [run_task(config, task, bootstrap_replicates=2) for task in tasks]
    assert all(row["status"] == "SUCCESS" for row in rows)
    structural_fields = (
        "bootstrap_seed",
        "T_q",
        "queried_nodes",
        "queried_parent_precision",
        "queried_parent_recall",
        "queried_parent_f1",
        "direct_orientation_errors",
        "meek_propagated_orientation_errors",
        "final_directed_graph_error",
    )
    assert all(rows[0][field] == rows[1][field] for field in structural_fields)
