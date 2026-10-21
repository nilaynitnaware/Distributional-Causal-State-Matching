"""Leakage, schema, selector, and numerical tests for reduced Mk1 matching."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("mk1_reduced_matching", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)


def test_frozen_blocks_are_disjoint_and_exact() -> None:
    EXPERIMENT.validate_protocol()
    assert (EXPERIMENT.MODE_FIT.start, EXPERIMENT.MODE_FIT.stop) == (0, 400)
    assert (EXPERIMENT.TARGET_SELECTION.start, EXPERIMENT.TARGET_SELECTION.stop) == (400, 600)
    assert (EXPERIMENT.CANDIDATE_EVALUATION.start, EXPERIMENT.CANDIDATE_EVALUATION.stop) == (600, 800)
    assert (EXPERIMENT.TARGET_EVALUATION.start, EXPERIMENT.TARGET_EVALUATION.stop) == (800, 1000)
    assert (EXPERIMENT.IDENTITY_EVALUATION.start, EXPERIMENT.IDENTITY_EVALUATION.stop) == (8000, 8200)
    assert not EXPERIMENT.MODE_FIT.overlaps(EXPERIMENT.CANDIDATE_EVALUATION)
    assert not EXPERIMENT.TARGET_SELECTION.overlaps(EXPERIMENT.TARGET_EVALUATION)


def test_protocol_fails_when_evaluation_enters_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EXPERIMENT, "TARGET_EVALUATION", EXPERIMENT.RowBlock("target", 550, 750))
    with pytest.raises(ValueError, match="target evaluation"):
        EXPERIMENT.validate_protocol()


def test_protocol_fails_for_candidate_fit_evaluation_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EXPERIMENT, "CANDIDATE_EVALUATION", EXPERIMENT.RowBlock("action", 350, 550))
    with pytest.raises(ValueError, match="candidate evaluation"):
        EXPERIMENT.validate_protocol()


def test_protocol_fails_for_candidate_target_block_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EXPERIMENT, "CANDIDATE_EVALUATION", EXPERIMENT.RowBlock("action", 550, 750))
    with pytest.raises(ValueError, match="candidate physical evaluation"):
        EXPERIMENT.validate_protocol()


def test_protocol_fails_for_identity_source_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EXPERIMENT, "IDENTITY_EVALUATION", EXPERIMENT.RowBlock("reference", 4900, 5100))
    with pytest.raises(ValueError, match="identity evaluation"):
        EXPERIMENT.validate_protocol()


def test_selector_has_no_label_or_oracle_channel() -> None:
    assert tuple(inspect.signature(EXPERIMENT.select_action).parameters) == ("predicted_scores",)
    first = EXPERIMENT.select_action({"opaque-z": 2.0, "opaque-a": 1.0})
    renamed = EXPERIMENT.select_action({"generating-looking-label": 2.0, "other": 1.0})
    assert first == "opaque-a"
    assert renamed == "other"


def test_leave_out_catalog_removes_generating_action() -> None:
    action_ids = ["A_ID", "A000", "A001"]
    action_ids.remove("A001")
    assert "A001" not in action_ids


def test_oracle_uses_only_supplied_physical_evaluation_values() -> None:
    rng = np.random.default_rng(4)
    target = rng.normal(size=(200, 2))
    candidate = target.copy()
    identity = target + 2.0
    metrics, _ = EXPERIMENT.physical_metrics(
        target,
        {"A_ID": identity, "A000": candidate},
        "A000",
        np.random.default_rng(5),
        20,
    )
    assert metrics["oracle_id"] == "A000"
    assert metrics["top1_oracle_agreement"]


def test_runners_feed_reserved_rows_to_oracle() -> None:
    source_a = inspect.getsource(EXPERIMENT.run_experiment_a)
    source_b = inspect.getsource(EXPERIMENT.run_experiment_b_variant)
    for source in (source_a, source_b):
        assert "TARGET_EVALUATION.start:TARGET_EVALUATION.stop" in source
        assert "CANDIDATE_EVALUATION.start:CANDIDATE_EVALUATION.stop" in source
        assert "physical_metrics(target_eval, candidate_values" in source
    assert "MODE_FIT.start:MODE_FIT.stop" in source_a
    assert "MODE_FIT.start:MODE_FIT.stop" not in inspect.getsource(EXPERIMENT.physical_metrics)


def test_gaussian_kl_components_sum_and_identity() -> None:
    mean = np.array([0.2, -0.4])
    covariance = np.array([[1.2, 0.1], [0.1, 0.7]])
    total, mean_part, covariance_part = EXPERIMENT.gaussian_kl_components(mean, covariance, mean, covariance)
    assert abs(total) < 1e-12
    assert abs(total - mean_part - covariance_part) < 1e-12


def test_common_state_is_predeclared_full_causal_union() -> None:
    assert len(EXPERIMENT.COMMON_STATE) == 17
    assert set(EXPERIMENT.OUTCOMES).issubset(EXPERIMENT.COMMON_STATE)
    for parents in EXPERIMENT.SIGNAL_PARENTS.values():
        assert set(parents).issubset(EXPERIMENT.COMMON_STATE)


def test_incompatible_schema_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / EXPERIMENT.DATASET
    dataset.mkdir()
    columns = list(EXPERIMENT.COMMON_STATE)
    pd.DataFrame(columns=columns).to_csv(dataset / f"{EXPERIMENT.REFERENCE}.csv", index=False)
    pd.DataFrame(columns=columns[:-1]).to_csv(dataset / "bad.csv", index=False)
    with pytest.raises(ValueError, match="incompatible schema"):
        EXPERIMENT.validate_schemas(tmp_path, ["bad"])
