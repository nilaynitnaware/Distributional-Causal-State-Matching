"""Algebra and protocol tests for the isolated cross-archive audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import sys
import numpy as np
import pandas as pd


MODULE = Path(__file__).resolve().parents[1] / "run_audit.py"
SPEC = importlib.util.spec_from_file_location("lt_mk1_cross_archive", MODULE)
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def test_mode_catalog_is_b1_and_settings_are_unique() -> None:
    settings = {(mode.diode, mode.exposure) for mode in AUDIT.MODES}
    assert len(settings) == len(AUDIT.MODES) == 6
    assert sum(mode.name == "identity" for mode in AUDIT.MODES) == 1
    assert all(mode.changed_coordinate in {"none", "diode_ir", "t_ir"} for mode in AUDIT.MODES)


def test_gaussian_kl_identity_and_positive_shift() -> None:
    mean = np.array([0.0, 1.0])
    covariance = np.array([[1.0, 0.2], [0.2, 2.0]])
    assert AUDIT.gaussian_kl(mean, covariance, mean, covariance) < 1e-12
    shifted = AUDIT.gaussian_kl(mean, covariance, mean + 1.0, covariance)
    assert shifted > 0.0


def test_m2_recovers_proportional_coefficients() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 2))
    source = AUDIT.Fit(1.0, np.array([2.0, -1.0]), 0.5)
    y = -2.0 + x @ (0.4 * source.beta) + rng.normal(scale=0.2, size=len(x))
    frame = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y})
    fit, scale = AUDIT.fit_m2(frame, "y", ("x1", "x2"), source)
    assert abs(scale - 0.4) < 0.02
    assert np.allclose(fit.beta, scale * source.beta)


def test_test_splits_are_disjoint_and_bootstrap_compatible() -> None:
    selection = set(range(0, 200))
    target = set(range(200, 350))
    candidate = set(range(350, 500))
    assert selection.isdisjoint(target)
    assert selection.isdisjoint(candidate)
    assert target.isdisjoint(candidate)
    assert len(target) % AUDIT.BOOTSTRAP_BLOCK == 0
    assert len(candidate) % AUDIT.BOOTSTRAP_BLOCK == 0


def test_protocol_prohibits_test_fitted_candidates() -> None:
    assert AUDIT.PROTOCOL["calibration_archive"] == AUDIT.TRAIN_ARCHIVE
    assert AUDIT.PROTOCOL["test_archive"] == AUDIT.TEST_ARCHIVE
    assert AUDIT.PROTOCOL["no_test_fitted_candidate_parameters"] is True
