"""Focused gates for the publication experiment wrapper."""

from __future__ import annotations

from dataclasses import replace

import pytest

from research.matching_query_complexity import (
    CanonicalAction,
    candidate_map,
    solve_dag_complexity,
    solve_matching_complexity,
)
from research.small_mec_matching_query.experiment import (
    CampaignConfig,
    _source_problem,
    aggregate,
    minimum_vertex_cover_size,
    solve_instance,
)


def _tiny_config() -> CampaignConfig:
    return CampaignConfig(
        node_counts=(5,),
        graph_families=("path",),
        replicates_per_cell=1,
        maximum_mec_size=8,
    )


def test_source_problem_is_nontrivial_exact_mec() -> None:
    config = _tiny_config()
    source, failures = _source_problem(config, "path", 5, 0)
    assert source is not None
    assert not failures
    assert len(source.candidates) == 5
    assert minimum_vertex_cover_size(source.cpdag) == 2


def test_singleton_and_no_advantage_terminal_conditions() -> None:
    config = _tiny_config()
    source, _ = _source_problem(config, "path", 5, 0)
    assert source is not None
    candidates = candidate_map(source.candidates)
    action_sets = {
        dag_id: (
            CanonicalAction(
                shift=(float(index + 1), 0.0, 0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0, 1.0, 1.0),
            ),
        )
        for index, dag_id in enumerate(sorted(candidates))
    }
    singleton = {next(iter(candidates))}
    with pytest.raises(TypeError, match="lifted OptimalActionSet"):
        solve_matching_complexity(
            candidates=candidates,
            action_sets=action_sets,
            initial_model_ids=singleton,
        )
    assert (
        solve_dag_complexity(
            candidates=candidates, initial_model_ids=singleton
        ).complexity
        == 0
    )
    with pytest.raises(TypeError, match="lifted OptimalActionSet"):
        solve_matching_complexity(candidates=candidates, action_sets=action_sets)


def test_both_target_regimes_obey_exact_query_invariants() -> None:
    config = _tiny_config()
    source, _ = _source_problem(config, "path", 5, 0)
    assert source is not None
    rows = [solve_instance(source, regime, config) for regime in config.target_regimes]
    for row in rows:
        assert 0 <= row["T_match_star"] <= row["T_DAG_star"] <= row["mec_size"] - 1
        assert row["optimizer_degeneracy_count"] == 0
        assert row["unresolved_within_mode_tie_count"] == 0
        assert row["matching_partition_count_audited"] > 0
        assert row["DAG_partition_count_audited"] > 0
        assert row["T_Alg1_first"] >= 0
        assert row["T_Alg1_maxdeg"] >= 0
    summary = aggregate(rows)
    assert summary["validation"]["all_matching_leq_DAG"]
    assert summary["overall"]["instances"] == 2


def test_config_validation_requires_two_target_regimes() -> None:
    config = replace(_tiny_config(), target_regimes=("realisable",))
    try:
        config.validate()
    except ValueError as error:
        assert "both target regimes" in str(error)
    else:
        raise AssertionError("invalid target grid was accepted")
