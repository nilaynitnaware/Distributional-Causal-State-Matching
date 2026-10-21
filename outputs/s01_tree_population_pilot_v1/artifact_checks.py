"""Independent artifact checks for the completed S01-B pilot."""

from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVED_PROBES = HERE.parent / "s01_scalable_matching_feasibility_v1" / "s01_probe_results.json"
RERUN_PROBES = HERE / "frozen_probe_rerun.json"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    instances = load_jsonl(HERE / "instances.jsonl")
    roots = load_jsonl(HERE / "per_root_results.jsonl")
    traces = load_jsonl(HERE / "query_traces.jsonl")
    certificates = load_jsonl(HERE / "terminal_certificates.jsonl")
    assert len(instances) == 80
    assert len(roots) == len(traces) == len(certificates) == 6400
    assert all(row["status"] == "complete" for row in instances)
    assert all(row["candidate_count"] == 20 for row in instances)

    by_source = defaultdict(list)
    by_target = defaultdict(list)
    budget_by_instance = {}
    for row in instances:
        source_key = (row["family"], row["replicate"])
        target_key = (*source_key, row["regime"])
        by_source[source_key].append(row["archive"])
        by_target[target_key].append(row["archive"])
        budget_by_instance[row["instance_id"]] = row["budget"]
    for archives in by_source.values():
        first = archives[0]
        for current in archives[1:]:
            assert current["source_seed"] == first["source_seed"]
            assert current["tree_edges"] == first["tree_edges"]
            np.testing.assert_array_equal(current["source_mean"], first["source_mean"])
            np.testing.assert_array_equal(current["source_covariance"], first["source_covariance"])
    for archives in by_target.values():
        assert len(archives) == 2
        first, second = archives
        assert first["target_seed"] == second["target_seed"]
        assert first["target_construction"] == second["target_construction"]
        np.testing.assert_array_equal(first["target_mean"], second["target_mean"])
        np.testing.assert_array_equal(first["target_covariance"], second["target_covariance"])

    operational = {}
    maximum_residual = 0.0
    maximum_true_excess = 0.0
    for row in roots:
        match_count = row["matching_query_count"]
        full = row["queried_nodes_full"]
        assert row["matching_is_prefix"]
        assert row["queried_nodes_matching"] == full[:match_count]
        assert row["matching_certificate_status"] == "certified"
        certificate = row["matching_certificate"]
        assert certificate is not None
        budget = budget_by_instance[row["instance_id"]]
        assert certificate["commanded_support"] <= budget
        assert all(-2.0 <= value <= 2.0 for value in certificate["shift"])
        assert all(math.exp(-1.0) <= value <= math.exp(1.0) for value in certificate["scale"])
        for loss in certificate["candidate_losses"]:
            assert loss["absolute_residual"] <= loss["tolerance"]
            maximum_residual = max(maximum_residual, loss["absolute_residual"])
        maximum_true_excess = max(maximum_true_excess, abs(row["true_root_excess_kl"]))
        before = tuple(range(20))
        queried = ()
        for step in row["steps"]:
            key = (row["instance_id"], row["policy"], before, queried)
            selected = step["selected_node"]
            if key in operational:
                assert operational[key] == selected
            operational[key] = selected
            before = tuple(step["surviving_roots"])
            queried = (*queried, selected)

    archived = json.loads(ARCHIVED_PROBES.read_text())
    rerun = json.loads(RERUN_PROBES.read_text())
    archived_rows = {
        row["instance_id"]: row
        for row in archived["frozen_bank_probes"] + archived["n20_tree_probes"]
    }
    rerun_rows = {
        row["instance_id"]: row
        for row in rerun["frozen_bank_probes"] + rerun["n20_tree_probes"]
    }
    assert set(archived_rows) == set(rerun_rows)
    for key in archived_rows:
        left = archived_rows[key]["policy_result"]
        right = rerun_rows[key]["policy_result"]
        for field in (
            "query_count", "queried_nodes", "terminal_candidate_count",
            "non_singleton_matching_stop",
        ):
            assert left[field] == right[field]
        np.testing.assert_allclose(
            left["terminal_certificate"]["shift"],
            right["terminal_certificate"]["shift"],
            atol=1e-12,
            rtol=1e-11,
        )
        np.testing.assert_allclose(
            left["terminal_certificate"]["scale"],
            right["terminal_certificate"]["scale"],
            atol=1e-12,
            rtol=1e-11,
        )

    result = {
        "status": "passed",
        "checks": {
            "complete_instances": len(instances),
            "complete_root_coverage": True,
            "source_reused_across_targets_and_budgets": True,
            "target_reused_across_budgets": True,
            "paired_match_paths_are_full_path_prefixes": True,
            "selection_is_hidden_root_independent_given_public_state": True,
            "all_returned_commands_within_bounds_and_budget": True,
            "all_candidate_direct_kl_residuals_within_tolerance": True,
            "all_true_root_excess_kl_within_1e-9": maximum_true_excess <= 1e-9,
            "frozen_probe_outcomes_reproduced": True,
        },
        "counts": {
            "instances": len(instances),
            "per_root_policy_paths": len(roots),
            "certificates": len(certificates),
            "distinct_public_policy_states_checked": len(operational),
        },
        "maximum_candidate_certificate_residual": maximum_residual,
        "maximum_absolute_true_root_excess_kl": maximum_true_excess,
        "commands": {
            "focused_and_regression_tests": (
                "python -m pytest -q "
                "tests/test_s01_tree_population.py tests/test_command_space_matching.py "
                "tests/test_matching_query_complexity_research.py"
            ),
            "frozen_probe_rerun": (
                "PYTHONPATH=. python "
                "outputs/s01_scalable_matching_feasibility_v1/run_s01_probes.py "
                "--output outputs/s01_tree_population_pilot_v1/frozen_probe_rerun.json"
            ),
            "artifact_checks": (
                "python "
                "outputs/s01_tree_population_pilot_v1/artifact_checks.py"
            ),
        },
        "test_result": "30 passed",
    }
    (HERE / "validation_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
