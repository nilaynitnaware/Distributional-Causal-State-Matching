"""Independent validation for the frozen S01-C n=40 campaign artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.matching_query_complexity import reconstruct_all_candidate_scms
from research.s01_tree_population import enumerate_tree_root_class


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_kl(target_mean, target_covariance, candidate_mean, candidate_covariance):
    target_mean = np.asarray(target_mean, dtype=float)
    target_covariance = np.asarray(target_covariance, dtype=float)
    candidate_mean = np.asarray(candidate_mean, dtype=float)
    candidate_covariance = np.asarray(candidate_covariance, dtype=float)
    chol = np.linalg.cholesky(candidate_covariance)
    solved_covariance = np.linalg.solve(chol, target_covariance)
    trace_term = float(np.trace(np.linalg.solve(chol.T, solved_covariance)))
    difference = candidate_mean - target_mean
    solved_mean = np.linalg.solve(chol, difference)
    quadratic = float(solved_mean @ solved_mean)
    logdet_candidate = 2.0 * float(np.log(np.diag(chol)).sum())
    target_chol = np.linalg.cholesky(target_covariance)
    logdet_target = 2.0 * float(np.log(np.diag(target_chol)).sum())
    return 0.5 * (trace_term + quadratic - len(difference) + logdet_candidate - logdet_target)


def main():
    config = json.loads((HERE / "frozen_config.json").read_text())
    proposal = json.loads((HERE / "proposal_and_eligibility_manifest.json").read_text())
    manifest = json.loads((HERE / "implementation_manifest.json").read_text())
    frozen = load_jsonl(HERE / "frozen_instances.jsonl")
    instances = load_jsonl(HERE / "instances.jsonl")
    roots = load_jsonl(HERE / "per_root_results.jsonl")
    traces = load_jsonl(HERE / "query_traces.jsonl")
    certificates = load_jsonl(HERE / "terminal_certificates.jsonl")
    work = load_jsonl(HERE / "computational_work.jsonl")
    summary = json.loads((HERE / "pilot_summary.json").read_text())

    assert config["n"] == 40 and config["instance_count"] == 80
    assert len(frozen) == len(instances) == 80
    assert proposal["valid_sources"] == 20 and proposal["valid_instances"] == 80
    assert all(any(item["accepted"] for item in source["proposals"]) for source in proposal["source_proposals"])
    assert all(row["status"] in ("complete", "complete_capped") for row in instances)
    assert len(work) == 320
    assert len(roots) == len(traces) == 80 * 4 * 40
    assert len({row["cache_id"] for row in work}) == len(work)
    assert all(row["cache_configuration"]["fresh_per_instance_policy"] for row in work)
    assert all(row["candidate_count"] == 40 and row["candidate_optimizer"]["calls"] == 40 for row in work)
    assert all(sum(row["certificate_metrics"]["outcomes_uncached_states"].values()) == row["certificate_metrics"]["cache_entries"] for row in work)
    assert all(row["certificate_metrics"]["tied_family_product_upper_bounds_charged"] >= row["certificate_metrics"]["structural_combinations_actually_visited"] for row in work)

    expected_frozen_hashes = manifest["frozen_artifact_sha256_before_outcomes"]
    assert expected_frozen_hashes == {name: sha256(HERE / name) for name in expected_frozen_hashes}
    assert manifest["protected_prior_artifact_sha256_before"] == {
        name: sha256(ROOT / name) for name in manifest["protected_prior_artifact_sha256_before"]
    }

    by_instance_policy = defaultdict(list)
    operational = {}
    for row in roots:
        key = (row["instance_id"], row["policy"])
        by_instance_policy[key].append(row)
        assert row["matching_is_prefix"]
        assert row["queried_nodes_matching"] == row["queried_nodes_full"][: row["matching_query_count"]]
        assert row["matching_certificate_status"] == "certified"
        assert row["matching_certificate"] is not None
        public_state = tuple(range(40))
        queried = ()
        for step in row["steps"]:
            public_key = (row["instance_id"], row["policy"], public_state, queried)
            if public_key in operational:
                assert operational[public_key] == step["selected_node"]
            operational[public_key] = step["selected_node"]
            public_state = tuple(step["surviving_roots"])
            queried = (*queried, step["selected_node"])
    assert all(len(rows_for_task) == 40 for rows_for_task in by_instance_policy.values())

    recomputed = {}
    for key, values in by_instance_policy.items():
        recomputed[key] = {
            "matching": max(row["matching_query_count"] for row in values),
            "dag": max(row["dag_query_count"] for row in values),
        }
    for instance in instances:
        for policy, values in instance["policy_summary"].items():
            actual = recomputed[(instance["instance_id"], policy)]
            assert actual["matching"] == values["matching_worst_case"]
            assert actual["dag"] == values["dag_worst_case"]
            assert actual["dag"] - actual["matching"] == values["worst_case_savings"]

    frozen_by_id = {row["instance_id"]: row for row in frozen}
    reconstructed = {}
    maximum_residual = 0.0
    maximum_true_excess = 0.0
    checked_candidate_losses = 0
    for record in certificates:
        instance_id = record["instance_id"]
        frozen_row = frozen_by_id[instance_id]
        if instance_id not in reconstructed:
            tree = nx.Graph()
            tree.add_nodes_from(range(40))
            tree.add_edges_from(tuple(edge) for edge in frozen_row["tree_edges"])
            root_class = enumerate_tree_root_class(tree)
            scms = reconstruct_all_candidate_scms(tuple(root_class.candidates.values()), np.asarray(frozen_row["source_mean"]), np.asarray(frozen_row["source_covariance"]))
            reconstructed[instance_id] = (root_class, scms)
        root_class, scms = reconstructed[instance_id]
        command = record["certificate"]
        shift = np.asarray(command["shift"], dtype=float)
        scale = np.asarray(command["scale"], dtype=float)
        assert np.all(shift >= -2.0) and np.all(shift <= 2.0)
        assert np.all(scale >= math.exp(-1.0)) and np.all(scale <= math.exp(1.0))
        support = int(np.count_nonzero(np.abs(shift) > 1e-10) + np.count_nonzero(np.abs(scale - 1.0) > 1e-10))
        assert support == command["commanded_support"] <= frozen_row["budget"]
        losses = {item["root"]: item for item in command["candidate_losses"]}
        assert set(losses) == set(record["terminal_roots"])
        for root in record["terminal_roots"]:
            scm = scms[root_class.ids_by_root[root]]
            mean, covariance = scm.interventional_moments(shift, scale)
            direct = independent_kl(frozen_row["target_mean"], frozen_row["target_covariance"], mean, covariance)
            loss = losses[root]
            residual = abs(direct - loss["optimal_kl"])
            assert residual <= loss["tolerance"] + 1e-12
            assert abs(direct - loss["direct_kl"]) <= 1e-10
            maximum_residual = max(maximum_residual, residual)
            checked_candidate_losses += 1
        true_root = record["true_root_evaluation_only"]
        scm = scms[root_class.ids_by_root[true_root]]
        mean, covariance = scm.interventional_moments(shift, scale)
        direct = independent_kl(frozen_row["target_mean"], frozen_row["target_covariance"], mean, covariance)
        excess = abs(direct - record["true_root_optimal_kl"])
        assert excess <= 1e-9
        maximum_true_excess = max(maximum_true_excess, excess)

    test_command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_s01_c_tree_population.py", "tests/test_s01_tree_population.py",
        "tests/test_command_space_matching.py", "tests/test_matching_query_complexity_research.py",
        "tests/test_s01_tied_optimum_validation.py", "tests/test_small_mec_matching_query.py",
    ]
    completed = subprocess.run(test_command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    test_output = (completed.stdout + completed.stderr).strip()

    result = {
        "status": "passed",
        "checks": {
            "frozen_instance_and_proposal_accounting": True,
            "complete_40_root_coverage": True,
            "cache_isolation": True,
            "aggregate_costs_recomputed_from_unaggregated_records": True,
            "policy_selection_hidden_root_independence": True,
            "all_match_paths_are_full_recovery_prefixes": True,
            "all_commands_within_bounds_and_global_budget": True,
            "all_serialized_terminal_candidate_losses_recomputed_with_independent_kl": True,
            "all_true_root_losses_recomputed_with_independent_kl": True,
            "frozen_manifest_hashes_match": True,
            "protected_prior_artifacts_unchanged": True,
        },
        "counts": {
            "instances": len(instances), "instance_policy_tasks": len(work),
            "root_policy_paths": len(roots), "terminal_certificates": len(certificates),
            "candidate_losses_independently_checked": checked_candidate_losses,
            "distinct_public_policy_states": len(operational),
        },
        "maximum_independent_candidate_residual": maximum_residual,
        "maximum_independent_true_root_excess": maximum_true_excess,
        "test_command": " ".join(test_command), "test_result": test_output,
    }
    temporary = HERE / "validation_results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(HERE / "validation_results.json")

    manifest["protected_prior_artifact_sha256_after"] = {
        name: sha256(ROOT / name) for name in manifest["protected_prior_artifact_sha256_before"]
    }
    manifest["git_status_short_after_campaign"] = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    manifest["validation"] = {"command": " ".join(test_command), "result": test_output, "artifact_checker": str(Path(__file__).resolve())}
    temporary = HERE / "implementation_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(HERE / "implementation_manifest.json")

    runner = HERE / "run_s01_c_tree_population_n40.py"
    subprocess.run([sys.executable, str(runner), "--aggregate-only"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
