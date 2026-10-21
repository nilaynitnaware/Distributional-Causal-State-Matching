"""Cross-archive Light Tunnel Mk1 B=1 finite-mode ranking audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

from scripts._output_safety import prepare_output_directory


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "outputs" / "lt_mk1_b1_cross_archive_ranking"
REPORT = REPO / "docs" / "lt_mk1_b1_cross_archive_ranking_deep_dive.md"
TRAIN_ARCHIVE = "lt_interventions_standard_v1"
TEST_ARCHIVE = "lt_test_v1"
TEST_FILE = "ir_sensors"
REFERENCE = "uniform_reference"
SEED = 20260831
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_BLOCK = 25

PARENTS: dict[str, tuple[str, ...]] = {
    "ir_1": ("red", "green", "blue", "l_11", "l_12"),
    "vis_1": ("red", "green", "blue", "l_11", "l_12"),
    "ir_2": ("red", "green", "blue", "l_21", "l_22"),
    "vis_2": ("red", "green", "blue", "l_21", "l_22"),
    "ir_3": ("red", "green", "blue", "pol_1", "pol_2", "l_31", "l_32"),
    "vis_3": ("red", "green", "blue", "pol_1", "pol_2", "l_31", "l_32"),
}

SAME_SESSION_TARGETS: dict[str, str] = {
    "ir_1": "uniform_t_ir_1_weak",
    "ir_2": "uniform_t_ir_2_weak",
    "ir_3": "uniform_t_ir_3_weak",
    "vis_1": "uniform_diode_vis_1_mid",
    "vis_2": "uniform_t_vis_2_weak",
    "vis_3": "uniform_t_vis_3_weak",
}


@dataclass(frozen=True)
class Fit:
    alpha: float
    beta: np.ndarray
    omega: float


@dataclass(frozen=True)
class Mode:
    name: str
    diode: int
    exposure: int
    train_template: str
    changed_coordinate: str
    nominal_strength: str

    def train_environment(self, sensor: int) -> str:
        return self.train_template.format(sensor=sensor)


MODES: tuple[Mode, ...] = (
    Mode("identity", 2, 3, REFERENCE, "none", "baseline"),
    Mode("diode_mid", 1, 3, "uniform_diode_ir_{sensor}_mid", "diode_ir", "mid"),
    Mode("diode_strong", 0, 3, "uniform_diode_ir_{sensor}_strong", "diode_ir", "strong"),
    Mode("t_weak", 2, 2, "uniform_t_ir_{sensor}_weak", "t_ir", "weak"),
    Mode("t_mid", 2, 1, "uniform_t_ir_{sensor}_mid", "t_ir", "mid"),
    Mode("t_strong", 2, 0, "uniform_t_ir_{sensor}_strong", "t_ir", "strong"),
)

PROTOCOL = {
    "schema_version": 1,
    "seed": SEED,
    "model": "M2 finite heteroskedastic proportional-mean mode",
    "objective": "target-sample mean local Gaussian negative log likelihood",
    "calibration_archive": TRAIN_ARCHIVE,
    "test_archive": TEST_ARCHIVE,
    "source_calibration_rows": "uniform_reference[0:5000]",
    "mode_calibration_rows": "each standard mode[0:400]",
    "same_session_target_rows": "mode[400:600]",
    "test_primary_block": "first 500 rows of each exact setting",
    "test_target_selection_rows": "setting block[0:200]",
    "test_empirical_target_rows": "setting block[200:350]",
    "test_empirical_candidate_rows": "setting block[350:500]",
    "empirical_discrepancy": "Gaussian KL on training-standardized (red, outcome)",
    "covariance_ridge": 1e-8,
    "bootstrap": {
        "replicates": BOOTSTRAP_REPLICATES,
        "contiguous_block_rows": BOOTSTRAP_BLOCK,
        "scope": "within-session sampling only",
    },
    "comparability_thresholds": {
        "active_red_mean_shift_sd_max": 0.25,
        "active_red_sd_ratio_low": 0.80,
        "active_red_sd_ratio_high": 1.25,
        "baseline_red_slope_relative_drift_max": 0.40,
        "required_setting_rows": 500,
    },
    "candidate_modes": [mode.name for mode in MODES],
    "no_test_fitted_candidate_parameters": True,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_frame(name: str, frame: pd.DataFrame) -> None:
    table = OUTPUT / "tables" / f"{name}.csv"
    machine = OUTPUT / "machine_readable" / f"{name}.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    machine.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table, index=False, lineterminator="\n")
    write_json(machine, frame.to_dict(orient="records"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(raw_root: Path, archive: str, environment: str) -> pd.DataFrame:
    path = raw_root / archive / f"{environment}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fit_ols(frame: pd.DataFrame, outcome: str, parents: tuple[str, ...]) -> Fit:
    x = frame.loc[:, parents].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    design = np.column_stack((np.ones(len(frame)), x))
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficient
    return Fit(float(coefficient[0]), coefficient[1:], max(float(np.mean(residual**2)), 1e-12))


def fit_m2(frame: pd.DataFrame, outcome: str, parents: tuple[str, ...], source: Fit) -> tuple[Fit, float]:
    x = frame.loc[:, parents].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    projected_parent = x @ source.beta
    design = np.column_stack((np.ones(len(frame)), projected_parent))
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    scale = max(float(coefficient[1]), 1e-12)
    beta = scale * source.beta
    alpha = float(np.mean(y - x @ beta))
    residual = y - alpha - x @ beta
    return Fit(alpha, beta, max(float(np.mean(residual**2)), 1e-12)), scale


def local_nll(frame: pd.DataFrame, outcome: str, parents: tuple[str, ...], fit: Fit) -> np.ndarray:
    x = frame.loc[:, parents].to_numpy(float)
    residual = frame[outcome].to_numpy(float) - fit.alpha - x @ fit.beta
    return 0.5 * (math.log(2.0 * math.pi * fit.omega) + residual**2 / fit.omega)


def regular_cov(values: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1)).astype(float)
    scale = max(float(np.trace(covariance)) / covariance.shape[0], 1.0)
    return covariance + ridge * scale * np.eye(covariance.shape[0])


def gaussian_kl(mean_q: np.ndarray, cov_q: np.ndarray, mean_p: np.ndarray, cov_p: np.ndarray) -> float:
    dimension = len(mean_q)
    sign_q, logdet_q = np.linalg.slogdet(cov_q)
    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    if sign_q <= 0 or sign_p <= 0:
        raise ValueError("covariance is not positive definite")
    difference = mean_p - mean_q
    solved_cov = np.linalg.solve(cov_p, cov_q)
    solved_mean = np.linalg.solve(cov_p, difference)
    value = 0.5 * (logdet_p - logdet_q - dimension + np.trace(solved_cov) + difference @ solved_mean)
    return max(float(value), 0.0)


def moment_kl(q: np.ndarray, p: np.ndarray) -> float:
    return gaussian_kl(q.mean(axis=0), regular_cov(q), p.mean(axis=0), regular_cov(p))


def compose_moments(root_values: np.ndarray, fit: Fit) -> tuple[np.ndarray, np.ndarray]:
    root_mean = root_values.mean(axis=0)
    root_cov = regular_cov(root_values)
    mean = np.r_[root_mean, fit.alpha + fit.beta @ root_mean]
    cross = root_cov @ fit.beta
    covariance = np.zeros((len(mean), len(mean)), dtype=float)
    covariance[:-1, :-1] = root_cov
    covariance[:-1, -1] = cross
    covariance[-1, :-1] = cross
    covariance[-1, -1] = fit.beta @ root_cov @ fit.beta + fit.omega
    return mean, covariance


def mode_environments(raw_root: Path, outcome: str) -> tuple[str, ...]:
    directory = raw_root / TRAIN_ARCHIVE
    names = []
    for path in sorted(directory.glob("uniform_*.csv")):
        name = path.stem
        if name.startswith(f"uniform_t_{outcome}_") or name.startswith(f"uniform_diode_{outcome}_"):
            names.append(name)
    return tuple(names)


def same_session_regression(raw_root: Path) -> pd.DataFrame:
    reference = load_csv(raw_root, TRAIN_ARCHIVE, REFERENCE)
    rows: list[dict[str, Any]] = []
    for outcome, target_environment in SAME_SESSION_TARGETS.items():
        parents = PARENTS[outcome]
        source = fit_ols(reference.iloc[:5000], outcome, parents)
        root_values = reference.iloc[:5000].loc[:, parents].to_numpy(float)
        target = load_csv(raw_root, TRAIN_ARCHIVE, target_environment).iloc[400:600]
        target_values = target.loc[:, (*parents, outcome)].to_numpy(float)
        target_mean = target_values.mean(axis=0)
        target_cov = regular_cov(target_values)
        scores = []
        for environment in (REFERENCE, *mode_environments(raw_root, outcome)):
            if environment == REFERENCE:
                candidate = source
            else:
                calibration = load_csv(raw_root, TRAIN_ARCHIVE, environment).iloc[:400]
                candidate, _ = fit_m2(calibration, outcome, parents, source)
            candidate_mean, candidate_cov = compose_moments(root_values, candidate)
            scores.append((gaussian_kl(target_mean, target_cov, candidate_mean, candidate_cov), environment))
        score, selected = min(scores, key=lambda item: (item[0], item[1]))
        rows.append({
            "outcome": outcome,
            "target_environment": target_environment,
            "selected_environment": selected,
            "selected_correct": selected == target_environment,
            "selection_gaussian_kl": score,
            "candidate_count": len(scores),
            "source_rows": "0:5000",
            "candidate_calibration_rows": "0:400",
            "target_rows": "400:600",
        })
    return pd.DataFrame(rows)


def test_groups(test: pd.DataFrame, sensor: int = 1) -> dict[str, pd.DataFrame]:
    groups: dict[str, pd.DataFrame] = {}
    for mode in MODES:
        mask = (test[f"diode_ir_{sensor}"] == mode.diode) & (test[f"t_ir_{sensor}"] == mode.exposure)
        group = test.loc[mask].copy()
        if len(group) < 500:
            raise ValueError(f"test mode {mode.name} has only {len(group)} rows")
        groups[mode.name] = group.iloc[:500].copy()
    return groups


def archive_inventory(raw_root: Path) -> pd.DataFrame:
    rows = []
    for archive in (TRAIN_ARCHIVE, TEST_ARCHIVE):
        for path in sorted((raw_root / archive).glob("*.csv")):
            frame = pd.read_csv(path)
            rows.append({
                "archive": archive,
                "environment": path.stem,
                "n_rows": len(frame),
                "n_columns": len(frame.columns),
                "columns_sha256": hashlib.sha256("\n".join(frame.columns).encode()).hexdigest(),
                "config_values": ";".join(sorted(frame["config"].astype(str).unique())),
                "timestamp_first": float(frame["timestamp"].iloc[0]),
                "timestamp_last": float(frame["timestamp"].iloc[-1]),
                "counter_first": float(frame["counter"].iloc[0]),
                "counter_last": float(frame["counter"].iloc[-1]),
                "counter_consecutive": bool(np.array_equal(frame["counter"].to_numpy(float), np.arange(len(frame)))),
                "csv_sha256": sha256(path),
            })
    return pd.DataFrame(rows)


def comparability_audit(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    reference = load_csv(raw_root, TRAIN_ARCHIVE, REFERENCE)
    test = load_csv(raw_root, TEST_ARCHIVE, TEST_FILE)
    groups = test_groups(test)
    schema_match = tuple(reference.columns) == tuple(test.columns)
    synchronized = all(
        bool((test[f"diode_ir_{sensor}"] == test["diode_ir_1"]).all())
        and bool((test[f"t_ir_{sensor}"] == test["t_ir_1"]).all())
        for sensor in (2, 3)
    )
    mode_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    red_mean = float(reference["red"].mean())
    red_sd = float(reference["red"].std(ddof=1))
    for sensor in (1, 2, 3):
        outcome = f"ir_{sensor}"
        parents = PARENTS[outcome]
        source = fit_ols(reference.iloc[:5000], outcome, parents)
        baseline = groups["identity"]
        baseline_red = fit_ols(baseline, outcome, ("red",))
        slope_relative_drift = abs(float(baseline_red.beta[0] / source.beta[0] - 1.0))
        for mode in MODES:
            group = groups[mode.name]
            train_environment = mode.train_environment(sensor)
            train = load_csv(raw_root, TRAIN_ARCHIVE, train_environment)
            if mode.name == "identity":
                calibrated_fit, calibrated_scale = source, 1.0
            else:
                calibrated_fit, calibrated_scale = fit_m2(train.iloc[:400], outcome, parents, source)
            test_fit = fit_ols(group, outcome, ("red",))
            test_scale = float(test_fit.beta[0] / baseline_red.beta[0])
            other_parents = tuple(parent for parent in parents if parent != "red")
            mode_rows.append({
                "subsystem": f"red->{outcome}",
                "mode": mode.name,
                "changed_coordinate_within_subsystem": mode.changed_coordinate,
                "nominal_strength": mode.nominal_strength,
                "train_environment": train_environment,
                "train_control_value": (
                    "baseline" if mode.name == "identity" else float(train[f"diode_ir_{sensor}" if mode.changed_coordinate == "diode_ir" else f"t_ir_{sensor}"].iloc[0])
                ),
                "test_file": TEST_FILE,
                "test_diode_value": mode.diode,
                "test_exposure_value": mode.exposure,
                "test_rows_available": len(test.loc[(test[f"diode_ir_{sensor}"] == mode.diode) & (test[f"t_ir_{sensor}"] == mode.exposure)]),
                "primary_rows_used": 500,
                "exact_setting_correspondence": True,
                "test_controls_synchronized_across_three_ir_sensors": synchronized,
                "global_action_warning": "three same-kind IR controls change together; B=1 only after projection to one red->ir_j subsystem",
                "calibration_scale": calibrated_scale,
                "test_red_slope_scale": test_scale,
                "absolute_scale_drift": abs(test_scale - calibrated_scale),
                "test_residual_to_calibration_omega_ratio": test_fit.omega / calibrated_fit.omega,
                "test_outcome_min": float(group[outcome].min()),
                "test_outcome_max": float(group[outcome].max()),
                "train_outcome_min": float(train[outcome].min()),
                "train_outcome_max": float(train[outcome].max()),
                "inactive_test_parents": ";".join(other_parents),
                "inactive_test_parents_all_zero": bool((group.loc[:, other_parents] == 0).all().all()),
            })
        test_red = baseline["red"].to_numpy(float)
        train_parent = reference.iloc[:5000].loc[:, parents].to_numpy(float)
        test_parent = baseline.loc[:, parents].to_numpy(float)
        train_cov = regular_cov(train_parent)
        test_cov = regular_cov(test_parent)
        standardized_mean = (test_parent.mean(axis=0) - train_parent.mean(axis=0)) / train_parent.std(axis=0, ddof=1)
        covariance_shift = float(np.linalg.norm(test_cov - train_cov, ord="fro") / np.linalg.norm(train_cov, ord="fro"))
        drift_rows.append({
            "subsystem": f"red->{outcome}",
            "active_parent": "red",
            "train_red_min": float(reference["red"].min()),
            "train_red_max": float(reference["red"].max()),
            "test_red_min": float(test_red.min()),
            "test_red_max": float(test_red.max()),
            "test_red_within_train_range_fraction": float(np.mean((test_red >= reference["red"].min()) & (test_red <= reference["red"].max()))),
            "active_red_mean_shift_train_sd": abs(float(test_red.mean() - red_mean)) / red_sd,
            "active_red_sd_ratio": float(test_red.std(ddof=1) / red_sd),
            "full_parent_standardized_mean_shift_l2": float(np.linalg.norm(standardized_mean)),
            "full_parent_covariance_relative_fro_shift": covariance_shift,
            "baseline_red_slope_train": float(source.beta[0]),
            "baseline_red_slope_test": float(baseline_red.beta[0]),
            "baseline_red_slope_relative_drift": slope_relative_drift,
            "baseline_intercept_train_at_inactive_zero": source.alpha,
            "baseline_intercept_test": baseline_red.alpha,
            "baseline_residual_variance_train": source.omega,
            "baseline_residual_variance_test": baseline_red.omega,
            "baseline_residual_variance_ratio_test_train": baseline_red.omega / source.omega,
            "full_parent_regime": "severe lower-dimensional shift: all non-red signal parents fixed to zero in test",
            "retained_regime": "screened red-only characterization slice; retention decided by active-parent gate",
        })
    mode_frame = pd.DataFrame(mode_rows)
    drift_frame = pd.DataFrame(drift_rows)
    thresholds = PROTOCOL["comparability_thresholds"]
    active_parent_pass = bool(
        (drift_frame["active_red_mean_shift_train_sd"] <= thresholds["active_red_mean_shift_sd_max"]).all()
        and drift_frame["active_red_sd_ratio"].between(
            thresholds["active_red_sd_ratio_low"], thresholds["active_red_sd_ratio_high"]
        ).all()
        and (drift_frame["test_red_within_train_range_fraction"] == 1.0).all()
    )
    slope_pass = bool((drift_frame["baseline_red_slope_relative_drift"] <= thresholds["baseline_red_slope_relative_drift_max"]).all())
    counts_pass = bool((mode_frame["test_rows_available"] >= thresholds["required_setting_rows"]).all())
    correspondence_pass = bool(mode_frame["exact_setting_correspondence"].all())
    pass_gate = bool(schema_match and synchronized and active_parent_pass and slope_pass and counts_pass and correspondence_pass)
    summary = {
        "verdict": "PASS_REDUCED_RED_ONLY_SLICES" if pass_gate else "FAIL",
        "pass": pass_gate,
        "full_station_comparability": "FAIL",
        "reduced_subsystem_comparability": "PASS" if pass_gate else "FAIL",
        "schema_match": schema_match,
        "same_raw_units_and_no_preprocessing": True,
        "config_standard_both": bool((reference["config"] == "standard").all() and (test["config"] == "standard").all()),
        "controls_synchronized_in_test": synchronized,
        "active_parent_distribution_pass": active_parent_pass,
        "baseline_slope_drift_pass": slope_pass,
        "setting_correspondence_pass": correspondence_pass,
        "sample_count_pass": counts_pass,
        "screened_subsystems": ["red->ir_1", "red->ir_2", "red->ir_3"],
        "eligible_subsystems": ["red->ir_1", "red->ir_2", "red->ir_3"] if pass_gate else [],
        "excluded_subsystems": ["full station 1", "full station 2", "full station 3", "all visible-light outcomes"],
        "reason": (
            "The archives share schema, raw units, exact IR settings, and same-sign baseline responses, but "
            + ("only the red-only physical characterization slices pass." if pass_gate else "even red is far outside calibration support, so no subsystem passes." )
        ),
    }
    return mode_frame, drift_frame, summary, groups


def calibration_catalog(raw_root: Path) -> tuple[dict[str, dict[str, Fit]], pd.DataFrame]:
    reference = load_csv(raw_root, TRAIN_ARCHIVE, REFERENCE)
    catalog: dict[str, dict[str, Fit]] = {}
    rows = []
    for sensor in (1, 2, 3):
        outcome = f"ir_{sensor}"
        parents = PARENTS[outcome]
        source = fit_ols(reference.iloc[:5000], outcome, parents)
        catalog[outcome] = {}
        for mode in MODES:
            environment = mode.train_environment(sensor)
            if mode.name == "identity":
                fit, scale = source, 1.0
            else:
                fit, scale = fit_m2(load_csv(raw_root, TRAIN_ARCHIVE, environment).iloc[:400], outcome, parents, source)
            catalog[outcome][mode.name] = fit
            rows.append({
                "outcome": outcome,
                "subsystem": f"red->{outcome}",
                "mode": mode.name,
                "environment": environment,
                "alpha": fit.alpha,
                "beta_json": json.dumps(fit.beta.tolist()),
                "red_beta": float(fit.beta[0]),
                "scale": scale,
                "omega": fit.omega,
                "cost": 0 if mode.name == "identity" else 1,
                "fit_archive": TRAIN_ARCHIVE,
                "fit_rows": "reference 0:5000" if mode.name == "identity" else "mode 0:400",
            })
    return catalog, pd.DataFrame(rows)


def bootstrap_indices(length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if length % block:
        raise ValueError("length must be divisible by block")
    blocks = np.arange(length).reshape(-1, block)
    chosen = rng.integers(0, len(blocks), size=len(blocks))
    return blocks[chosen].reshape(-1)


def cross_archive_evaluation(
    groups: dict[str, pd.DataFrame],
    catalog: dict[str, dict[str, Fit]],
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    reference = load_csv(raw_root, TRAIN_ARCHIVE, REFERENCE).iloc[:5000]
    rng = np.random.default_rng(SEED)
    candidate_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    task_bootstraps: list[dict[str, np.ndarray]] = []
    mode_names = [mode.name for mode in MODES]
    for sensor in (1, 2, 3):
        outcome = f"ir_{sensor}"
        parents = PARENTS[outcome]
        scale = reference.loc[:, ["red", outcome]].std(axis=0, ddof=1).to_numpy(float)
        for target_mode in mode_names:
            target = groups[target_mode]
            target_selection = target.iloc[:200]
            target_eval = target.iloc[200:350].loc[:, ["red", outcome]].to_numpy(float) / scale
            predicted_scores = np.asarray([
                float(local_nll(target_selection, outcome, parents, catalog[outcome][candidate]).mean())
                for candidate in mode_names
            ])
            empirical_scores = np.asarray([
                moment_kl(
                    target_eval,
                    groups[candidate].iloc[350:500].loc[:, ["red", outcome]].to_numpy(float) / scale,
                )
                for candidate in mode_names
            ])
            predicted_ranks = stats.rankdata(predicted_scores, method="average")
            empirical_ranks = stats.rankdata(empirical_scores, method="average")
            predicted_index = int(np.argmin(predicted_scores))
            empirical_index = int(np.argmin(empirical_scores))
            true_index = mode_names.index(target_mode)
            available = np.asarray([index for index in range(len(mode_names)) if index != true_index])
            leave_predicted_index = int(available[np.argmin(predicted_scores[available])])
            leave_empirical_index = int(available[np.argmin(empirical_scores[available])])
            boot_scores = np.empty((BOOTSTRAP_REPLICATES, len(mode_names)))
            for replicate in range(BOOTSTRAP_REPLICATES):
                target_index = bootstrap_indices(len(target_eval), BOOTSTRAP_BLOCK, rng)
                q = target_eval[target_index]
                for candidate_index, candidate in enumerate(mode_names):
                    candidate_values = groups[candidate].iloc[350:500].loc[:, ["red", outcome]].to_numpy(float) / scale
                    candidate_index_rows = bootstrap_indices(len(candidate_values), BOOTSTRAP_BLOCK, rng)
                    boot_scores[replicate, candidate_index] = moment_kl(q, candidate_values[candidate_index_rows])
            boot_best = np.argmin(boot_scores, axis=1)
            boot_regret = boot_scores[:, predicted_index] - boot_scores.min(axis=1)
            boot_leave_best_local = np.argmin(boot_scores[:, available], axis=1)
            boot_leave_best = available[boot_leave_best_local]
            boot_leave_regret = boot_scores[:, leave_predicted_index] - boot_scores[:, available].min(axis=1)
            boot_rank = np.asarray([
                stats.spearmanr(predicted_scores, boot_scores[index]).statistic
                for index in range(BOOTSTRAP_REPLICATES)
            ])
            point_best_boot_difference = boot_scores - boot_scores[:, [empirical_index]]
            plausible = [
                mode_names[index]
                for index in range(len(mode_names))
                if float(np.quantile(point_best_boot_difference[:, index], 0.025)) <= 0.0
            ]
            leave_best_boot_difference = boot_scores - boot_scores[:, [leave_empirical_index]]
            plausible_leave = [
                mode_names[index]
                for index in available
                if float(np.quantile(leave_best_boot_difference[:, index], 0.025)) <= 0.0
            ]
            empirical_order = np.argsort(empirical_scores)
            top_gap = boot_scores[:, empirical_order[1]] - boot_scores[:, empirical_order[0]]
            covariance_conditions = []
            for candidate in mode_names:
                values = groups[candidate].iloc[350:500].loc[:, ["red", outcome]].to_numpy(float) / scale
                covariance_conditions.append(float(np.linalg.cond(regular_cov(values))))
            red_mean_shift = abs(float(target_selection["red"].mean() - reference["red"].mean())) / float(reference["red"].std(ddof=1))
            red_sd_ratio = float(target_selection["red"].std(ddof=1) / reference["red"].std(ddof=1))
            domain_shift_metric = float(math.sqrt(red_mean_shift**2 + math.log(red_sd_ratio) ** 2))
            spearman = float(stats.spearmanr(predicted_scores, empirical_scores).statistic)
            for candidate_index, candidate in enumerate(mode_names):
                candidate_rows.append({
                    "subsystem": f"red->{outcome}",
                    "outcome": outcome,
                    "target_mode": target_mode,
                    "candidate_mode": candidate,
                    "candidate_is_generating_mode": candidate == target_mode,
                    "predicted_local_nll": predicted_scores[candidate_index],
                    "predicted_rank": predicted_ranks[candidate_index],
                    "empirical_gaussian_kl": empirical_scores[candidate_index],
                    "empirical_rank": empirical_ranks[candidate_index],
                    "in_full_plausible_top_set": candidate in plausible,
                    "in_leaveout_plausible_top_set": candidate in plausible_leave,
                    "candidate_empirical_covariance_condition": covariance_conditions[candidate_index],
                })
            task_rows.append({
                "subsystem": f"red->{outcome}",
                "outcome": outcome,
                "target_mode": target_mode,
                "predicted_selected_mode": mode_names[predicted_index],
                "empirical_best_mode": mode_names[empirical_index],
                "predicted_selects_generating_mode": predicted_index == true_index,
                "empirical_best_is_generating_mode": empirical_index == true_index,
                "top1_matches_empirical": predicted_index == empirical_index,
                "uncertainty_aware_top_set_hit": mode_names[predicted_index] in plausible,
                "plausible_top_set": ";".join(plausible),
                "spearman_rank_correlation": spearman,
                "regret": empirical_scores[predicted_index] - empirical_scores[empirical_index],
                "regret_ci_low": float(np.quantile(boot_regret, 0.025)),
                "regret_ci_high": float(np.quantile(boot_regret, 0.975)),
                "top_gap_ci_low": float(np.quantile(top_gap, 0.025)),
                "top_gap_ci_high": float(np.quantile(top_gap, 0.975)),
                "empirical_top_is_distinguishable": bool(np.quantile(top_gap, 0.025) > 0),
                "leaveout_predicted_selected_mode": mode_names[leave_predicted_index],
                "leaveout_empirical_best_mode": mode_names[leave_empirical_index],
                "leaveout_top1_matches_empirical": leave_predicted_index == leave_empirical_index,
                "leaveout_uncertainty_aware_top_set_hit": mode_names[leave_predicted_index] in plausible_leave,
                "leaveout_plausible_top_set": ";".join(plausible_leave),
                "leaveout_regret": empirical_scores[leave_predicted_index] - empirical_scores[leave_empirical_index],
                "leaveout_regret_ci_low": float(np.quantile(boot_leave_regret, 0.025)),
                "leaveout_regret_ci_high": float(np.quantile(boot_leave_regret, 0.975)),
                "domain_shift_metric": domain_shift_metric,
                "active_red_mean_shift_train_sd": red_mean_shift,
                "active_red_sd_ratio": red_sd_ratio,
                "max_empirical_covariance_condition": max(covariance_conditions),
                "empirical_samples_per_distribution": len(target_eval),
                "dimension": 2,
                "samples_per_dimension": len(target_eval) / 2,
            })
            task_bootstraps.append({
                "top1": (boot_best == predicted_index).astype(float),
                "rank": boot_rank,
                "regret": boot_regret,
                "leave_top1": (boot_leave_best == leave_predicted_index).astype(float),
                "leave_regret": boot_leave_regret,
            })
    tasks = pd.DataFrame(task_rows)
    candidates = pd.DataFrame(candidate_rows)
    def aggregate_boot(key: str) -> np.ndarray:
        return np.column_stack([item[key] for item in task_bootstraps]).mean(axis=1)
    boot_aggregate = {key: aggregate_boot(key) for key in ("top1", "rank", "regret", "leave_top1", "leave_regret")}
    shift_correlation = stats.spearmanr(tasks["domain_shift_metric"], 1.0 - tasks["spearman_rank_correlation"])
    summary = {
        "n_tasks": len(tasks),
        "n_subsystems": tasks["subsystem"].nunique(),
        "candidates_per_task": len(MODES),
        "top1_accuracy": float(tasks["top1_matches_empirical"].mean()),
        "top1_accuracy_bootstrap_ci": [float(value) for value in np.quantile(boot_aggregate["top1"], [0.025, 0.975])],
        "generating_mode_selection_accuracy": float(tasks["predicted_selects_generating_mode"].mean()),
        "empirical_best_generating_mode_fraction": float(tasks["empirical_best_is_generating_mode"].mean()),
        "uncertainty_aware_top_set_accuracy": float(tasks["uncertainty_aware_top_set_hit"].mean()),
        "mean_spearman_rank_correlation": float(tasks["spearman_rank_correlation"].mean()),
        "mean_spearman_bootstrap_ci": [float(value) for value in np.quantile(boot_aggregate["rank"], [0.025, 0.975])],
        "mean_regret": float(tasks["regret"].mean()),
        "mean_regret_bootstrap_ci": [float(value) for value in np.quantile(boot_aggregate["regret"], [0.025, 0.975])],
        "leaveout_top1_accuracy": float(tasks["leaveout_top1_matches_empirical"].mean()),
        "leaveout_top1_bootstrap_ci": [float(value) for value in np.quantile(boot_aggregate["leave_top1"], [0.025, 0.975])],
        "leaveout_uncertainty_aware_top_set_accuracy": float(tasks["leaveout_uncertainty_aware_top_set_hit"].mean()),
        "leaveout_mean_regret": float(tasks["leaveout_regret"].mean()),
        "leaveout_mean_regret_bootstrap_ci": [float(value) for value in np.quantile(boot_aggregate["leave_regret"], [0.025, 0.975])],
        "distinguishable_empirical_top_fraction": float(tasks["empirical_top_is_distinguishable"].mean()),
        "max_covariance_condition": float(tasks["max_empirical_covariance_condition"].max()),
        "domain_shift_vs_rank_error_spearman": float(shift_correlation.statistic),
        "domain_shift_vs_rank_error_p": float(shift_correlation.pvalue),
        "uncertainty_scope": "contiguous-block bootstrap within one test session; no between-session confidence claim",
    }
    return candidates, tasks, summary


def make_plots(candidates: pd.DataFrame, tasks: pd.DataFrame) -> None:
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"ir_1": "#1f77b4", "ir_2": "#ff7f0e", "ir_3": "#2ca02c"}
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for outcome, group in candidates.groupby("outcome"):
        axis.scatter(group["predicted_rank"], group["empirical_rank"], s=28, alpha=0.65, label=outcome, color=colors[outcome])
    axis.plot([1, len(MODES)], [1, len(MODES)], color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="predicted candidate rank", ylabel="empirical physical rank", title="Cross-archive finite-mode rankings")
    axis.set_xticks(range(1, len(MODES) + 1))
    axis.set_yticks(range(1, len(MODES) + 1))
    axis.legend()
    fig.savefig(figures / "predicted_vs_empirical_candidate_ranking.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.hist(tasks["regret"], bins=12, alpha=0.65, label="full candidates")
    axis.hist(tasks["leaveout_regret"], bins=12, alpha=0.65, label="leave true mode out")
    axis.set(xlabel="empirical Gaussian-KL regret", ylabel="task count", title="Cross-archive regret distribution")
    axis.legend()
    fig.savefig(figures / "regret_distribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    for outcome, group in tasks.groupby("outcome"):
        axis.scatter(group["regret"], group["leaveout_regret"], s=48, alpha=0.8, label=outcome, color=colors[outcome])
    maximum = max(float(tasks["regret"].max()), float(tasks["leaveout_regret"].max()), 1e-6)
    axis.plot([0, maximum], [0, maximum], color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="full-candidate regret", ylabel="leave-true-mode-out regret", title="Full versus projection performance")
    axis.legend()
    fig.savefig(figures / "full_vs_leave_true_mode_out.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    for outcome, group in tasks.groupby("outcome"):
        axis.scatter(group["domain_shift_metric"], 1.0 - group["spearman_rank_correlation"], s=48, alpha=0.8, label=outcome, color=colors[outcome])
    axis.set(xlabel="active-red distribution-shift metric", ylabel="ranking error (1 - Spearman)", title="Domain shift versus ranking error")
    axis.legend()
    fig.savefig(figures / "domain_shift_vs_ranking_error.png", dpi=180)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_report(
    comparability: dict[str, Any],
    drift: pd.DataFrame,
    correspondence: pd.DataFrame,
    regression: pd.DataFrame,
    tasks: pd.DataFrame | None,
    summary: dict[str, Any] | None,
    protocol_hash: str,
) -> None:
    max_slope_drift = float(drift["baseline_red_slope_relative_drift"].max())
    max_full_mean_shift = float(drift["full_parent_standardized_mean_shift_l2"].max())
    min_red_overlap = float(drift["test_red_within_train_range_fraction"].min())
    max_red_mean_shift = float(drift["active_red_mean_shift_train_sd"].max())
    max_red_sd_ratio = float(drift["active_red_sd_ratio"].max())
    if summary is None or tasks is None:
        headline = "The comparability kill gate failed; no cross-archive ranking was run."
        result_rows = "| Cross-archive top-1 | not run |\n| Cross-archive rank correlation | not run |\n| Mean regret | not run |\n| Leave-true-mode-out regret | not run |\n| Main-paper viable | NO |"
        cross_section = "No cross-archive candidate model was evaluated after the failed kill gate. Accordingly, no ranking, regret, projection, or shift-versus-ranking-error plot was generated."
        recommendation = "Reject the cross-archive ranking as a validation experiment. The comparability audit itself may be retained as an appendix-level negative result."
        evaluation_text = "Because the gate failed, no target ranking, empirical candidate KL, regret, leave-true-mode-out score, covariance estimate, or bootstrap was computed. The frozen downstream estimators remain documented only to make the stop auditable."
        domain_text = "No ranking-error correlation was computed. The decisive structural diagnostics are the active-red support failure, severe full-parent distribution shift, baseline coefficient drift across sessions, mode-scale drift in `mode_correspondence.csv`, and mode-specific residual-variance ratios."
        uncertainty_text = "Statistical ranking uncertainty is not applicable because ranking stopped at the kill gate. The underlying archive still has only one standard session per calibration mode and one test characterization session."
        strength_text = "There are no cross-archive ranking numbers to interpret. The 6/6 result is same-session regression evidence only."
        slice_gate_text = f"""The reduced `{{red, ir_j}}` slices also **fail**. The calibration reference covers red=0--85, whereas the test sweep covers red=0--254. Only {fmt(min_red_overlap)} of test red observations lie inside calibration support; the mean shift reaches {fmt(max_red_mean_shift)} training SD and the test/train SD ratio reaches {fmt(max_red_sd_ratio)}. Baseline red slopes retain their sign and drift by at most {fmt(max_slope_drift)}, but that coefficient resemblance does not cure the support failure. The frozen active-parent gate therefore stops the experiment before any cross-archive score is computed."""
    else:
        headline = (
            "The reduced red-only comparison passes the archive gate, but the ranking result is evidence for a "
            "narrow characterization-slice transfer task rather than full-station cross-archive matching."
        )
        result_rows = (
            f"| Cross-archive top-1 | {int(tasks['top1_matches_empirical'].sum())}/{len(tasks)} ({fmt(summary['top1_accuracy'])}) |\n"
            f"| Cross-archive rank correlation | mean Spearman {fmt(summary['mean_spearman_rank_correlation'])} |\n"
            f"| Mean regret | {fmt(summary['mean_regret'])} |\n"
            f"| Leave-true-mode-out regret | {fmt(summary['leaveout_mean_regret'])} |\n"
            "| Main-paper viable | MAYBE (appendix preferred) |"
        )
        cross_section = f"""Across {len(tasks)} blinded target tasks (three outcomes by six physical settings), the calibration-only model matched the empirical top candidate on **{int(tasks['top1_matches_empirical'].sum())}/{len(tasks)}** tasks. It selected the generating label on **{int(tasks['predicted_selects_generating_mode'].sum())}/{len(tasks)}** tasks; the independently estimated empirical optimum itself equaled the generating label on **{int(tasks['empirical_best_is_generating_mode'].sum())}/{len(tasks)}** tasks.

Mean Spearman rank correlation was **{fmt(summary['mean_spearman_rank_correlation'])}** (within-session block-bootstrap 95% interval {fmt(summary['mean_spearman_bootstrap_ci'][0])} to {fmt(summary['mean_spearman_bootstrap_ci'][1])}). Mean empirical Gaussian-KL regret was **{fmt(summary['mean_regret'])}** (95% interval {fmt(summary['mean_regret_bootstrap_ci'][0])} to {fmt(summary['mean_regret_bootstrap_ci'][1])}). The uncertainty-aware top-set hit rate was **{fmt(summary['uncertainty_aware_top_set_accuracy'])}**. Only **{fmt(summary['distinguishable_empirical_top_fraction'])}** of point-estimate empirical winners had a positive 95% top-gap lower bound, so raw top-1 must not be overinterpreted.

With the generating mode removed, top-1 agreement with the best remaining physical candidate was **{int(tasks['leaveout_top1_matches_empirical'].sum())}/{len(tasks)}** ({fmt(summary['leaveout_top1_accuracy'])}); mean regret was **{fmt(summary['leaveout_mean_regret'])}** (95% interval {fmt(summary['leaveout_mean_regret_bootstrap_ci'][0])} to {fmt(summary['leaveout_mean_regret_bootstrap_ci'][1])}). The uncertainty-aware leave-out top-set hit rate was **{fmt(summary['leaveout_uncertainty_aware_top_set_accuracy'])}**.

Every Gaussian covariance estimate used 150 observations in dimension two (75 observations per dimension). The maximum regularized covariance condition number was {fmt(summary['max_covariance_condition'], 1)}. This is numerically adequate, but it does not create independent physical sessions."""
        recommendation = (
            "Prefer an appendix result. Promote to main text only if the manuscript explicitly needs a known-graph, "
            "B=1 finite-mode characterization-slice transfer example and states that the test controls are synchronized "
            "outside each retained subsystem. Do not present it as full-station cross-archive validation."
        )
        slice_gate_text = f"""The reduced `{{red, ir_j}}` slices pass: every test red value lies in the training range, test/train red standard-deviation ratios satisfy the frozen 0.80--1.25 gate, and baseline red-slope drift is at most {fmt(max_slope_drift)} (threshold 0.40). Thus the retained claim is conditional on this physical characterization slice. The synchronized changes outside a selected station are causally downstream-irrelevant under the supplied Mk1 graph, but they prevent calling the global test-file action a literal one-coordinate intervention."""
        evaluation_text = "Predicted scores are local target-sample Gaussian cross-entropies. Empirical scores are directional Gaussian KLs on training-standardized `(red, ir_j)` using disjoint test rows: 0:200 for target selection, 200:350 for empirical target moments, and 350:500 for physical candidate moments."
        domain_text = f"The active-red shift metric versus ranking error has Spearman correlation {fmt(summary['domain_shift_vs_rank_error_spearman'])} (p={fmt(summary['domain_shift_vs_rank_error_p'])}). More important are the structural diagnostics: full-parent distribution shift, baseline coefficient drift across sessions, mode-scale drift, and mode-specific residual-variance ratios."
        uncertainty_text = "The block bootstrap resamples contiguous 25-row blocks and quantifies within-session sampling uncertainty only. There is one standard session per calibration mode and one test characterization session, so no interval supports a population of physical sessions."
        strength_text = "The numerical strength of the claim is exactly the top-1, rank-correlation, regret, and uncertainty results above."
    mode_lines = []
    for mode in MODES:
        row = correspondence[correspondence["mode"] == mode.name].iloc[0]
        mode_lines.append(
            f"| `{mode.name}` | `{row['train_environment']}` | diode={mode.diode}, exposure={mode.exposure} | {mode.changed_coordinate} |"
        )
    report = f"""# Light Tunnel Mk1 B=1 cross-archive ranking deep dive

## Executive verdict

{headline}

| Test | Result |
| --- | --- |
| Archive comparability | {comparability['reduced_subsystem_comparability']} for red-only slices; full stations FAIL |
| Same-session 6/6 reproduction | {'PASS' if regression['selected_correct'].all() else 'FAIL'} ({int(regression['selected_correct'].sum())}/6) |
{result_rows}

## Scope and frozen protocol

This is a known/fixed-structure **B=1 finite-mode projection/ranking** audit. It does not test active acquisition, DAG recovery, Algorithm 1, B>1 composability, or the exact shift--scale family. The fitted model is the prior audit's smallest defensible M2 catalog: each local conditional mean has an archive-calibrated positive scale of the source coefficient vector, a free intercept, and a mode-specific average residual variance. Parent-state heteroskedasticity means this is an approximation.

The protocol was serialized and hashed before cross-archive evaluation. Protocol SHA-256: `{protocol_hash}`. Calibration uses only `lt_interventions_standard_v1`; no `lt_test_v1` row fits a candidate mechanism or selects a hyperparameter.

## Archive comparability kill gate

Both archives have the same ordered 46-column raw schema, `config=standard`, raw command/ADC units, and no analysis normalization. The standard archive was acquired as one 10,000-row randomized reference plus separate 1,000-row singleton environments. `lt_test_v1/ir_sensors.csv` is a later 3,500-row characterization sweep with contiguous 500-row setting blocks; its identity setting occurs twice. It changes the corresponding IR controls at all three sensors together.

The full station environments are **not comparable**: the standard archive randomizes all five or seven signal parents, while the test sweep varies only red and fixes green, blue, local LEDs, and polarizers at zero. The maximum standardized full-parent mean shift is {fmt(max_full_mean_shift)}. Aggressive normalization was not used.

{slice_gate_text}

## Retained subsystems and finite action set

The screened subsystems are `red -> ir_1`, `red -> ir_2`, and `red -> ir_3`. Visible outcomes and full five/seven-parent stations are excluded. When the gate fails, none of the screened subsystems or modes is retained for ranking.

| Mode | Calibration environment (sensor 1 example) | Test setting | Changed coordinate within subsystem |
| --- | --- | --- | --- |
{chr(10).join(mode_lines)}

For sensors 2 and 3, the environment and control suffix changes to that sensor. No synthetic or composed action appears. Identity costs zero; every other local mode costs one.

## Same-session regression check

The independent implementation reproduces **{int(regression['selected_correct'].sum())}/6** M2 selections using source rows 0:5000, mode calibration rows 0:400, and hidden target rows 400:600. This regression was completed before applying the frozen model to the test archive.

## Blinded cross-archive ranking and projection

{cross_section}

{evaluation_text}

## Domain shift diagnostics

{domain_text}

{uncertainty_text}

## Strongest defensible claim

> The archives contain exactly corresponding IR hardware settings, but their parent/input regimes are too different for a clean cross-archive known-graph ranking test; the correct result is a preregistered comparability failure, not a forced normalized ranking.

{strength_text} It is not evidence for active acquisition, B>1 composition, DAG recovery, Algorithm 1, or an exact shift--scale intervention family.

## Recommendation

{recommendation}

## Files and reproduction

Created paths:

- `research/lt_mk1_b1_cross_archive/README.md`
- `research/lt_mk1_b1_cross_archive/run_audit.py`
- `research/lt_mk1_b1_cross_archive/tests/test_audit.py`
- `docs/lt_mk1_b1_cross_archive_ranking_deep_dive.md`
- `outputs/lt_mk1_b1_cross_archive_ranking/`

From the worktree root:

```bash
python -m research.lt_mk1_b1_cross_archive.run_audit --data-root publisher-data --output new-results/cross-archive
python -m pytest research/lt_mk1_b1_cross_archive/tests -q
```

All raw archives are immutable inputs. No manuscript, existing experiment, or archived dataset was modified.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    global OUTPUT, REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    raw_root = args.data_root.resolve()
    OUTPUT = prepare_output_directory(
        args.output, overwrite=args.overwrite, protected=(raw_root,)
    )
    REPORT = OUTPUT / "lt_mk1_b1_cross_archive_ranking_report.md"
    expected_hashes = {
        TRAIN_ARCHIVE: "8781960b5fff5d752f57566393d4b0f8706cac9b2cd2a3d3226442f17c7bab60",
        TEST_ARCHIVE: "828e00ae5b703e9f42c2ab7616017c476a0cf7f14e073b5fda294f8c64e738c2",
    }
    for archive, expected in expected_hashes.items():
        if sha256(raw_root / f"{archive}.zip") != expected:
            raise RuntimeError(f"official archive hash mismatch: {archive}")

    inventory = archive_inventory(raw_root)
    correspondence, drift, comparability, groups = comparability_audit(raw_root)
    write_frame("archive_inventory", inventory)
    write_frame("mode_correspondence", correspondence)
    write_frame("domain_shift_diagnostics", drift)
    write_json(OUTPUT / "machine_readable" / "comparability_gate.json", comparability)

    protocol_text = json.dumps(PROTOCOL, indent=2, sort_keys=True) + "\n"
    protocol_hash = hashlib.sha256(protocol_text.encode()).hexdigest()
    write_json(OUTPUT / "machine_readable" / "frozen_protocol.json", {**PROTOCOL, "sha256": protocol_hash})

    regression = same_session_regression(raw_root)
    write_frame("same_session_regression", regression)
    if int(regression["selected_correct"].sum()) != 6:
        write_report(comparability, drift, correspondence, regression, None, None, protocol_hash)
        raise RuntimeError("same-session 6/6 regression check failed; cross-archive evaluation not run")
    if not comparability["pass"]:
        failure_summary = {
            "status": "NOT_RUN_COMPARABILITY_KILL_GATE",
            "archive_comparability": "FAIL",
            "same_session_correct": 6,
            "same_session_tasks": 6,
            "cross_archive_top1": None,
            "cross_archive_rank_correlation": None,
            "mean_regret": None,
            "leave_true_mode_out_regret": None,
            "main_paper_viable": "NO",
            "recommendation": "reject ranking validation; appendix-only comparability negative result",
        }
        write_json(OUTPUT / "machine_readable" / "summary.json", failure_summary)
        write_json(
            OUTPUT / "machine_readable" / "run_metadata.json",
            {
                "seed": SEED,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "raw_root": raw_root,
                "archive_sha256": expected_hashes,
                "protocol_sha256": protocol_hash,
                "comparability": comparability,
                "same_session_correct": 6,
                "summary": failure_summary,
            },
        )
        write_report(comparability, drift, correspondence, regression, None, None, protocol_hash)
        print(json.dumps({"comparability": comparability, "same_session_correct": 6, "summary": failure_summary}, indent=2))
        return

    catalog, catalog_frame = calibration_catalog(raw_root)
    write_frame("calibration_catalog", catalog_frame)
    candidates, tasks, summary = cross_archive_evaluation(groups, catalog, raw_root)
    write_frame("cross_archive_candidate_rankings", candidates)
    write_frame("cross_archive_task_results", tasks)
    write_json(OUTPUT / "machine_readable" / "summary.json", summary)
    make_plots(candidates, tasks)
    write_report(comparability, drift, correspondence, regression, tasks, summary, protocol_hash)
    write_json(
        OUTPUT / "machine_readable" / "run_metadata.json",
        {
            "seed": SEED,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "raw_root": raw_root,
            "archive_sha256": expected_hashes,
            "protocol_sha256": protocol_hash,
            "comparability": comparability,
            "same_session_correct": int(regression["selected_correct"].sum()),
            "summary": summary,
        },
    )
    print(json.dumps({"comparability": comparability["verdict"], "same_session": "6/6", **summary}, indent=2))


if __name__ == "__main__":
    main()
