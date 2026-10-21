"""Execute the frozen S01-B3 cases and emit machine-readable evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from graphs.support import parameter_support_cost
from research.matching_query_complexity import infer_outward_tree_root
from research.s01_tree_population import CertificateCache
from outputs.s01_b3_tied_optimum_validation_v1.tied_optimum_reference import (
    HERE,
    action_record,
    evaluate_case,
    load_frozen,
)
from scripts._output_safety import prepare_output_directory



def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _command(args: list[str]) -> dict[str, object]:
    started = perf_counter()
    completed = subprocess.run(
        args, cwd=REPO, text=True, capture_output=True, check=False,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    return {
        "argv": args,
        "exit_code": completed.returncode,
        "seconds": perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    output = prepare_output_directory(
        arguments.output, overwrite=arguments.overwrite, protected=(HERE,)
    )
    frozen = load_frozen()
    conventions = frozen["numerical_conventions"]
    records = []
    cached_production = None
    for case in frozen["cases"]:
        started = perf_counter()
        candidates, _, reference, production, certificate = evaluate_case(
            case, conventions
        )
        by_root = {}
        passed = True
        for candidate in candidates:
            root = infer_outward_tree_root(candidate)
            expected = reference[root]
            actual = production[candidate.id]
            action_match = len(expected.actions) == len(actual.actions) and all(
                any(
                    np.allclose(wanted.shift, got.shift, atol=1e-12, rtol=0.0)
                    and np.allclose(wanted.scale, got.scale, atol=1e-12, rtol=0.0)
                    for got in actual.actions
                )
                for wanted in expected.actions
            )
            kl_match = abs(expected.optimal_kl - actual.optimal_kl) <= 2e-12
            passed &= action_match and kl_match
            by_root[str(root)] = {
                "dag_id": [list(edge) for edge in candidate.id],
                "reference_optimal_kl": expected.optimal_kl,
                "production_optimal_kl": actual.optimal_kl,
                "reference_actions": [action_record(x) for x in expected.actions],
                "production_actions": [action_record(x) for x in actual.actions],
                "actions_match": action_match,
                "kl_matches": kl_match,
                "family_count": len(actual.command_families),
            }
        expected_common = bool(case["expected_common"])
        passed &= (certificate is not None) is expected_common
        certificate_record = None
        if certificate is not None:
            support = parameter_support_cost(
                np.asarray(certificate.command.shift),
                np.asarray(certificate.command.scale),
            )
            residuals_valid = all(
                loss.absolute_residual <= loss.tolerance
                for loss in certificate.candidate_losses
            )
            passed &= support <= int(conventions["budget"]) and residuals_valid
            certificate_record = {
                "command": action_record(certificate.command),
                "commanded_support": support,
                "selected_family_indices": [
                    {"dag_id": [list(edge) for edge in dag_id], "family_index": index}
                    for dag_id, index in certificate.selected_family_indices
                ],
                "candidate_losses": [asdict(item) for item in certificate.candidate_losses],
                "all_direct_kl_residuals_valid": residuals_valid,
            }
        records.append({
            "id": case["id"],
            "kind": case["kind"],
            "passed": passed,
            "seconds": perf_counter() - started,
            "by_root": by_root,
            "certificate": certificate_record,
            "independent_negative_justification": (
                case.get("analytic_construction") if certificate is None else None
            ),
        })
        if case["id"] == frozen["cap_condition"]["base_case"]:
            cached_production = production

    assert cached_production is not None
    state = frozenset(cached_production)
    cap = frozen["cap_condition"]
    low = CertificateCache(
        cached_production,
        maximum_tied_family_combinations=int(cap["low_family_cap"]),
    )
    low_result = low.check(state)
    recovered = CertificateCache(
        cached_production,
        maximum_tied_family_combinations=int(cap["recovery_family_cap"]),
    )
    recovered_result = recovered.check(state)
    no_checks = CertificateCache(
        cached_production,
        maximum_uncached_checks=int(cap["uncached_check_cap"]),
    )
    no_checks_result = no_checks.check(state)
    cap_record = {
        "id": cap["id"],
        "family_product_upper_bound": cap["family_product_upper_bound"],
        "low_status": low_result.status,
        "low_accounted_combinations": low.tied_family_combinations,
        "low_cap_hits": low.cap_hits,
        "zero_uncached_status": no_checks_result.status,
        "zero_uncached_checks": no_checks.uncached_checks,
        "recovered_status": recovered_result.status,
        "recovered_accounted_combinations": recovered.tied_family_combinations,
        "accounting_note": "family_product_upper_bound and tied_family_combinations are conservative accounting bounds, not measured visited combinations",
        "passed": (
            low_result.status == "inconclusive"
            and low.tied_family_combinations == 0
            and no_checks_result.status == "inconclusive"
            and no_checks.uncached_checks == 0
            and recovered_result.status == "certified"
        ),
    }
    _write_json(output / "case_results.json", {"cases": records, "cap_condition": cap_record})

    test_command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_s01_tied_optimum_validation.py",
        "tests/test_command_space_matching.py",
        "tests/test_s01_tree_population.py",
        "tests/test_matching_query_complexity_research.py",
        "tests/test_small_mec_matching_query.py",
    ]
    test_result = _command(test_command)
    all_cases_passed = all(record["passed"] for record in records) and cap_record["passed"]
    validation = {
        "schema_version": "s01_b3_validation_results_v1",
        "status": "passed" if all_cases_passed and test_result["exit_code"] == 0 else "failed",
        "one_numerical_thread": True,
        "case_count": len(records),
        "realisable_case_count": len(records),
        "new_geometry_only_case_count": 0,
        "all_independent_references_match_production": all(record["passed"] for record in records),
        "all_positive_witnesses_directly_kl_certified": all(
            record["certificate"] is None
            or record["certificate"]["all_direct_kl_residuals_valid"]
            for record in records
        ),
        "all_positive_witnesses_within_global_support_budget": all(
            record["certificate"] is None
            or record["certificate"]["commanded_support"] <= int(conventions["budget"])
            for record in records
        ),
        "cap_condition_passed": cap_record["passed"],
        "focused_regression_command": test_result,
    }
    _write_json(output / "validation_results.json", validation)

    relevant = [
        "research/matching_query_complexity.py",
        "research/s01_tree_population.py",
        "oracle/sparse_shift_scale_kl.py",
        "graphs/support.py",
        "tests/test_command_space_matching.py",
        "tests/test_matching_query_complexity_research.py",
        "tests/test_s01_tree_population.py",
        "tests/test_small_mec_matching_query.py",
        "tests/test_s01_tied_optimum_validation.py",
        "outputs/s01_b3_tied_optimum_validation_v1/frozen_cases.json",
        "outputs/s01_b3_tied_optimum_validation_v1/tied_optimum_reference.py",
        "outputs/s01_b3_tied_optimum_validation_v1/run_s01_b3_tied_optimum_validation.py",
        "outputs/s01_b3_tied_optimum_validation_v1/case_results.json",
        "outputs/s01_b3_tied_optimum_validation_v1/validation_results.json",
        "outputs/s01_b3_tied_optimum_validation_v1/coverage_matrix.md",
        "outputs/s01_b3_tied_optimum_validation_v1/exploratory_constructions.md",
        "outputs/s01_b3_tied_optimum_validation_v1/s01_b3_tied_optimum_validation_report.md",
    ]
    manifest = {
        "schema_version": "s01_b3_implementation_manifest_v1",
        "head_before_and_after": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "preexisting_worktree_note": "Dirty before S01-B3; preserved. See report and preexisting hashes.",
        "preexisting_relevant_hashes": {
            "research/matching_query_complexity.py": "341f708520bb27292dd9b6f2ada89d7ac38cf8710add2e17fdd0dafc6edcaa4c",
            "research/s01_tree_population.py": "020ce63dc036f969b65e7f2927b65768a9edfd87816eb2cdf1a79929733af64a",
            "oracle/sparse_shift_scale_kl.py": "126f3aa9debffd4f3ad2ea32c4f9deb0ae416d24d174f6e4580a2ae100e8b65d",
            "graphs/support.py": "250e190f36f42eb752d8d1450b6904362fbc3e5f6317b46355c6e93c969ca9bb",
            "tests/test_command_space_matching.py": "f08862792a8dd8e35b0e21c4977888cbfad13e5e6f0faf56d45c37c5140ad7a0",
            "tests/test_matching_query_complexity_research.py": "b23e68705648ef1826be45fd96b523e24849e5defc99c3836aa85ce4f820a068",
            "tests/test_s01_tree_population.py": "f82b74c8ae7162cefe233ac9b3eecd77f895cbd2a401ef45a14bc77172ba8d80",
            "tests/test_small_mec_matching_query.py": "03322238e34994c05bde4c3c26a288160d421c9d7e5367ee4d39b306961f516e"
        },
        "preexisting_relevant_diff_stat": {
            "research/matching_query_complexity.py": {"insertions": 564, "deletions": 13},
            "tests/test_small_mec_matching_query.py": {"insertions": 6, "deletions": 10}
        },
        "files": {
            path: _hash(REPO / path) for path in relevant if (REPO / path).exists()
        },
        "exact_runner_command": "python -m outputs.s01_b3_tied_optimum_validation_v1.run_s01_b3_tied_optimum_validation --output NEW_OUTPUT"
    }
    _write_json(output / "implementation_manifest.json", manifest)
    return 0 if validation["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
