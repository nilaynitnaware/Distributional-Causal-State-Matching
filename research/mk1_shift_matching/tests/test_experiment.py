"""Integrity and numerical tests for Mk1 shift matching."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("mk1_shift_matching", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)


def test_frozen_blocks_are_exact_and_disjoint() -> None:
    EXPERIMENT.validate_protocol()
    assert (EXPERIMENT.SOURCE_FIT.start, EXPERIMENT.SOURCE_FIT.stop) == (0, 5000)
    assert (EXPERIMENT.ACTION_FIT.start, EXPERIMENT.ACTION_FIT.stop) == (0, 400)
    assert (EXPERIMENT.TARGET_SELECTION.start, EXPERIMENT.TARGET_SELECTION.stop) == (400, 600)
    assert (EXPERIMENT.CANDIDATE_EVALUATION.start, EXPERIMENT.CANDIDATE_EVALUATION.stop) == (600, 800)
    assert (EXPERIMENT.TARGET_EVALUATION.start, EXPERIMENT.TARGET_EVALUATION.stop) == (800, 1000)
    assert (EXPERIMENT.IDENTITY_EVALUATION.start, EXPERIMENT.IDENTITY_EVALUATION.stop) == (8000, 8200)


def test_target_evaluation_cannot_enter_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EXPERIMENT, "TARGET_EVALUATION", EXPERIMENT.RowBlock("target", 550, 750))
    with pytest.raises(ValueError, match="target evaluation"):
        EXPERIMENT.validate_protocol()


def test_candidate_evaluation_cannot_enter_fitting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EXPERIMENT, "CANDIDATE_EVALUATION", EXPERIMENT.RowBlock("action", 350, 550))
    with pytest.raises(ValueError, match="candidate evaluation"):
        EXPERIMENT.validate_protocol()


def test_identity_evaluation_is_disjoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EXPERIMENT, "IDENTITY_EVALUATION", EXPERIMENT.RowBlock("reference", 4900, 5100))
    with pytest.raises(ValueError, match="identity evaluation"):
        EXPERIMENT.validate_protocol()


def test_selector_has_no_target_label_or_oracle_channel() -> None:
    assert tuple(inspect.signature(EXPERIMENT.select_action).parameters) == ("predicted_scores",)
    assert EXPERIMENT.select_action({"opaque-z": 3.0, "opaque-a": 1.0}) == "opaque-a"
    assert EXPERIMENT.select_action({"looks-generating": 3.0, "other": 1.0}) == "other"


def test_s2_removes_generating_action() -> None:
    candidates = EXPERIMENT.candidate_ids_for_experiment(["S000", "S001"], "S001", True)
    assert candidates == ["A_ID", "S000"]
    assert "S001" not in candidates


def test_oracle_accepts_only_supplied_physical_arrays() -> None:
    rng = np.random.default_rng(4)
    target = rng.normal(size=(200, 2))
    observed, action_ids, boot, oracle = EXPERIMENT.bootstrap_physical(
        target,
        {"A_ID": target + 2.0, "S000": target.copy()},
        np.random.default_rng(5),
        10,
    )
    metrics = EXPERIMENT.physical_metrics("S000", observed, action_ids, boot, oracle)
    assert oracle == "S000"
    assert metrics["top1_oracle_agreement"]


def test_runner_uses_only_reserved_physical_rows() -> None:
    source = inspect.getsource(EXPERIMENT.run_matching)
    assert "TARGET_EVALUATION.start:TARGET_EVALUATION.stop" in source
    assert "CANDIDATE_EVALUATION.start:CANDIDATE_EVALUATION.stop" in source
    assert "bootstrap_physical" in source
    assert "ACTION_FIT.start:ACTION_FIT.stop" not in inspect.getsource(EXPERIMENT.bootstrap_physical)


def test_primary_shift_model_uses_source_covariance() -> None:
    source = inspect.getsource(EXPERIMENT.candidate_models)
    assert 'models["shift_model"][action.action_id] = GaussianModel(shift_mean, source_model.covariance)' in source
    assert "ACTION_FIT.start:ACTION_FIT.stop" in source
    assert "TARGET_EVALUATION" not in source


def test_gaussian_kl_components_sum_and_vanish_at_identity() -> None:
    mean = np.array([0.2, -0.4])
    covariance = np.array([[1.2, 0.1], [0.1, 0.7]])
    total, mean_part, covariance_part = EXPERIMENT.gaussian_kl_components(mean, covariance, mean, covariance)
    assert abs(total) < 1e-12
    assert abs(total - mean_part - covariance_part) < 1e-12


def test_states_are_predeclared_and_target_independent() -> None:
    assert len(EXPERIMENT.COMMON_STATE) == 17
    assert len(EXPERIMENT.DESCENDANT_STATE) == 12
    assert set(EXPERIMENT.ACTUATORS).isdisjoint(EXPERIMENT.DESCENDANT_STATE)
    assert set(EXPERIMENT.OUTCOMES).issubset(EXPERIMENT.DESCENDANT_STATE)
    assert "target" not in inspect.signature(EXPERIMENT.state_diagnostics).parameters


def test_incompatible_action_schema_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / EXPERIMENT.DATASET
    dataset.mkdir()
    columns = list(EXPERIMENT.COMMON_STATE)
    pd.DataFrame(columns=columns).to_csv(dataset / f"{EXPERIMENT.REFERENCE}.csv", index=False)
    pd.DataFrame(columns=columns[:-1]).to_csv(dataset / "bad.csv", index=False)
    with pytest.raises(ValueError, match="incompatible schema"):
        EXPERIMENT.validate_schemas(tmp_path, ["bad"])


def test_invariance_tolerance_is_explicit_and_fixed() -> None:
    assert EXPERIMENT.INVARIANCE_TOLERANCE == {
        "root_variance_ratio_low": 0.8,
        "root_variance_ratio_high": 1.25,
        "full_state_relative_covariance_frobenius_max": 0.25,
        "descendant_relative_covariance_frobenius_max": 0.25,
    }
