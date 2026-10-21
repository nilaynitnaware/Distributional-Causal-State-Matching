"""Small algebraic tests for the isolated Mk1 feasibility audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_audit.py"
SPEC = importlib.util.spec_from_file_location("mk1_e2e_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_changed_noise_union_survives_both_cancellations() -> None:
    frame = AUDIT.support_theory_validation()
    nonroots = frame[frame["parent_count"] > 0]
    assert nonroots["union_support_exact_symbolic"].all()
    assert (~nonroots["union_support_exact_at_1e_12"]).any()
    assert (~nonroots["row_support_exact_at_1e_12"]).any()
    assert (~nonroots["diagonal_support_exact_at_1e_12"]).any()


def test_gaussian_kl_identity_and_asymmetry() -> None:
    mean = np.array([0.2, -0.4])
    covariance = np.array([[1.2, 0.1], [0.1, 0.7]])
    shifted = mean + np.array([0.5, 0.0])
    assert abs(AUDIT.gaussian_kl(mean, covariance, mean, covariance)) < 1e-12
    assert AUDIT.gaussian_kl(mean, covariance, shifted, covariance) > 0


def test_model_families_keep_normalizing_variance_semantics() -> None:
    import pandas as pd

    frame = pd.DataFrame({"x": np.arange(10.0), "y": 2.0 + 3.0 * np.arange(10.0)})
    source = AUDIT.Fit(alpha=0.0, beta=np.array([1.0]), omega=4.0)
    m1, _ = AUDIT.fit_family(frame, "y", ("x",), source, "M1")
    m2, _ = AUDIT.fit_family(frame, "y", ("x",), source, "M2")
    assert m1.omega == source.omega
    assert 0 < m2.omega < source.omega
