from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from research.small_mec_matching_query.stopping_decomposition import (
    DEFAULT_INPUT,
    POLICIES,
    _certificate_valid,
    _first_matching_terminal,
    run_diagnostic,
)


def test_first_terminal_is_a_prefix_depth() -> None:
    certificate = {
        "candidate_losses": [],
        "commanded_support": 0,
        "maximum_loss_residual": 0.0,
    }
    depth, state, found = _first_matching_terminal(
        ["0,1,2", "0,1", "0"], {"0,1": certificate, "0": certificate}
    )
    assert (depth, state, found) == (1, "0,1", certificate)


def test_certificate_validation_checks_all_survivors() -> None:
    certificate = {
        "candidate_losses": [
            {
                "candidate_dag_identifier": "a",
                "direct_kl": 1.0,
                "optimal_kl": 1.0,
                "absolute_residual": 0.0,
                "tolerance": 1e-10,
            }
        ],
        "commanded_support": 2,
        "maximum_loss_residual": 0.0,
    }
    assert _certificate_valid(certificate, ["a"], budget=2)
    assert not _certificate_valid(certificate, ["a", "b"], budget=2)
    assert not _certificate_valid(certificate, ["a"], budget=1)


def test_frozen_bank_end_to_end(tmp_path: Path) -> None:
    if not (DEFAULT_INPUT / "per_instance.csv").is_file():
        pytest.skip("complete corrected-audit trace bank is intentionally omitted")
    result = run_diagnostic(DEFAULT_INPUT, tmp_path)
    validation = result["validation"]
    assert validation["instance_count"] == 180
    assert validation["source_count"] == 90
    assert validation["all_decompositions_exact"]
    assert validation["query_schedules_target_independent"]
    assert set(result["summary"]) == set(POLICIES)

    with (tmp_path / "per_instance_results.csv").open(newline="") as handle:
        per_instance = list(csv.DictReader(handle))
    with (tmp_path / "candidate_path_results.csv").open(newline="") as handle:
        paths = list(csv.DictReader(handle))
    assert len(per_instance) == 180 * len(POLICIES)
    assert len(paths) == 986 * len(POLICIES)
    assert all(row["matching_is_exact_prefix"] == "True" for row in paths)
    assert all(row["matching_terminal_certificate_valid"] == "True" for row in paths)
    assert json.loads((tmp_path / "validation.json").read_text())["full_costs_reproduced"]
