"""Leakage-safe Mk1 empirical finite gain-mode matching extensions."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import linalg, stats

from scripts._output_safety import prepare_output_directory


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "outputs" / "mk1_reduced_real_matching"
DATASET = "lt_interventions_standard_v1"
REFERENCE = "uniform_reference"
SEED = 20260831
N_BOOTSTRAP = 500
BLOCK_SIZE = 50

SIGNAL_PARENTS: dict[str, tuple[str, ...]] = {
    "ir_1": ("red", "green", "blue", "l_11", "l_12"),
    "vis_1": ("red", "green", "blue", "l_11", "l_12"),
    "ir_2": ("red", "green", "blue", "l_21", "l_22"),
    "vis_2": ("red", "green", "blue", "l_21", "l_22"),
    "ir_3": ("red", "green", "blue", "pol_1", "pol_2", "l_31", "l_32"),
    "vis_3": ("red", "green", "blue", "pol_1", "pol_2", "l_31", "l_32"),
}
ROOTS = (
    "red", "green", "blue", "pol_1", "pol_2",
    "l_11", "l_12", "l_21", "l_22", "l_31", "l_32",
)
OUTCOMES = ("ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3")
COMMON_STATE = (*ROOTS, *OUTCOMES)


@dataclass(frozen=True)
class RowBlock:
    environment_role: str
    start: int
    stop: int

    def overlaps(self, other: "RowBlock") -> bool:
        return self.environment_role == other.environment_role and max(self.start, other.start) < min(self.stop, other.stop)

    def row_range_overlaps(self, other: "RowBlock") -> bool:
        return max(self.start, other.start) < min(self.stop, other.stop)


SOURCE_FIT = RowBlock("reference", 0, 5000)
MODE_FIT = RowBlock("action", 0, 400)
TARGET_SELECTION = RowBlock("target", 400, 600)
CANDIDATE_EVALUATION = RowBlock("action", 600, 800)
TARGET_EVALUATION = RowBlock("target", 800, 1000)
IDENTITY_EVALUATION = RowBlock("reference", 8000, 8200)


@dataclass(frozen=True)
class Fit:
    alpha: float
    beta: np.ndarray
    omega: float


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
    table_path = OUTPUT / "tables" / f"{name}.csv"
    json_path = OUTPUT / "machine_readable" / f"{name}.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_path, index=False, lineterminator="\n")
    write_json(json_path, frame.to_dict(orient="records"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol() -> None:
    if MODE_FIT.stop > TARGET_SELECTION.start:
        raise ValueError("target selection rows enter candidate fitting")
    if TARGET_SELECTION.overlaps(TARGET_EVALUATION):
        raise ValueError("target evaluation rows enter target selection")
    if CANDIDATE_EVALUATION.overlaps(MODE_FIT):
        raise ValueError("candidate evaluation rows enter candidate fitting")
    if CANDIDATE_EVALUATION.row_range_overlaps(TARGET_SELECTION):
        raise ValueError("candidate physical evaluation overlaps target selection")
    if TARGET_EVALUATION.row_range_overlaps(TARGET_SELECTION):
        raise ValueError("target evaluation overlaps calibration")
    if IDENTITY_EVALUATION.overlaps(SOURCE_FIT):
        raise ValueError("identity evaluation overlaps source calibration")


def load(raw_root: Path, environment: str) -> pd.DataFrame:
    path = raw_root / DATASET / f"{environment}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def infer_action(environment: str) -> tuple[str, str, str]:
    body = environment.removeprefix("uniform_")
    for level in ("weak", "mid", "strong"):
        suffix = f"_{level}"
        if body.endswith(suffix):
            control = body[: -len(suffix)]
            if control.startswith("t_"):
                node = control.removeprefix("t_")
            elif control.startswith("diode_"):
                node = control.removeprefix("diode_")
            else:
                raise ValueError(environment)
            return node, control, level
    raise ValueError(environment)


def action_environments(raw_root: Path) -> list[str]:
    paths = sorted((raw_root / DATASET).glob("uniform_*.csv"))
    return [
        path.stem for path in paths
        if path.stem.startswith("uniform_t_") or path.stem.startswith("uniform_diode_")
    ]


def validate_schemas(raw_root: Path, environments: Iterable[str]) -> pd.DataFrame:
    reference_path = raw_root / DATASET / f"{REFERENCE}.csv"
    reference_columns = list(pd.read_csv(reference_path, nrows=0).columns)
    required = set(COMMON_STATE)
    if not required.issubset(reference_columns):
        raise ValueError("common state missing from reference schema")
    rows = []
    for environment in environments:
        path = raw_root / DATASET / f"{environment}.csv"
        columns = list(pd.read_csv(path, nrows=0).columns)
        exact = columns == reference_columns
        has_common = required.issubset(columns)
        rows.append({
            "environment": environment,
            "n_columns": len(columns),
            "exact_reference_schema": exact,
            "contains_common_state": has_common,
            "common_state_order_matches": [column for column in columns if column in required] == [column for column in reference_columns if column in required],
        })
        if not exact or not has_common:
            raise ValueError(f"incompatible schema: {environment}")
    return pd.DataFrame(rows)


def regular_cov(values: np.ndarray, ridge_fraction: float = 1e-6) -> np.ndarray:
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1)).astype(float)
    dimension = covariance.shape[0]
    ridge = max(float(np.trace(covariance)) / dimension * ridge_fraction, ridge_fraction)
    return covariance + ridge * np.eye(dimension)


def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), regular_cov(values)


def gaussian_kl_components(
    mean_q: np.ndarray,
    cov_q: np.ndarray,
    mean_p: np.ndarray,
    cov_p: np.ndarray,
) -> tuple[float, float, float]:
    dimension = len(mean_q)
    chol_p = np.linalg.cholesky(cov_p)
    solved_cov = linalg.cho_solve((chol_p, True), cov_q)
    difference = mean_p - mean_q
    solved_mean = linalg.cho_solve((chol_p, True), difference)
    logdet_p = 2.0 * np.log(np.diag(chol_p)).sum()
    chol_q = np.linalg.cholesky(cov_q)
    logdet_q = 2.0 * np.log(np.diag(chol_q)).sum()
    mean_component = float(0.5 * difference @ solved_mean)
    covariance_component = float(0.5 * (np.trace(solved_cov) - dimension + logdet_p - logdet_q))
    return mean_component + covariance_component, mean_component, covariance_component


def standardize(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - center) / scale


def fit_ols(frame: pd.DataFrame, outcome: str, parents: tuple[str, ...]) -> Fit:
    x = frame.loc[:, parents].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    design = np.column_stack((np.ones(len(frame)), x))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    return Fit(float(coefficients[0]), coefficients[1:].astype(float), max(float(np.mean(residual**2)), np.finfo(float).tiny))


def fit_m2(frame: pd.DataFrame, outcome: str, source: Fit) -> Fit:
    x = frame.loc[:, SIGNAL_PARENTS[outcome]].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    projected = x @ source.beta
    coefficients, *_ = np.linalg.lstsq(np.column_stack((np.ones(len(frame)), projected)), y, rcond=None)
    action_scale = max(float(coefficients[1]), 1e-12)
    beta = action_scale * source.beta
    alpha = float(np.mean(y - x @ beta))
    residual = y - alpha - x @ beta
    return Fit(alpha, beta, max(float(np.mean(residual**2)), np.finfo(float).tiny))


def residual_matrix(frame: pd.DataFrame, fits: dict[str, Fit]) -> np.ndarray:
    columns = []
    for outcome in OUTCOMES:
        fit = fits[outcome]
        x = frame.loc[:, SIGNAL_PARENTS[outcome]].to_numpy(float)
        columns.append(frame[outcome].to_numpy(float) - fit.alpha - x @ fit.beta)
    return np.column_stack(columns)


def compose_common_moments(
    root_mean: np.ndarray,
    root_cov: np.ndarray,
    fits: dict[str, Fit],
    residual_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coefficient_matrix = np.zeros((len(OUTCOMES), len(ROOTS)))
    intercept = np.zeros(len(OUTCOMES))
    root_lookup = {name: index for index, name in enumerate(ROOTS)}
    for row, outcome in enumerate(OUTCOMES):
        fit = fits[outcome]
        intercept[row] = fit.alpha
        for parent, coefficient in zip(SIGNAL_PARENTS[outcome], fit.beta):
            coefficient_matrix[row, root_lookup[parent]] = coefficient
    outcome_mean = intercept + coefficient_matrix @ root_mean
    cross = root_cov @ coefficient_matrix.T
    outcome_covariance = coefficient_matrix @ root_cov @ coefficient_matrix.T + residual_covariance
    mean = np.r_[root_mean, outcome_mean]
    covariance = np.block([[root_cov, cross], [cross.T, outcome_covariance]])
    ridge = 1e-9 * np.eye(len(mean))
    return mean, covariance + ridge


def transform_moments(mean: np.ndarray, covariance: np.ndarray, center: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = 1.0 / scale
    return (mean - center) * inv, covariance * np.outer(inv, inv)


def select_action(predicted_scores: dict[str, float]) -> str:
    """Select using opaque action IDs and scores only."""
    if not predicted_scores:
        raise ValueError("empty candidate catalog")
    return min(predicted_scores, key=lambda action_id: (predicted_scores[action_id], action_id))


def block_bootstrap(values: np.ndarray, rng: np.random.Generator, block_size: int = BLOCK_SIZE) -> np.ndarray:
    if len(values) % block_size:
        raise ValueError("evaluation block must divide into contiguous bootstrap blocks")
    blocks = values.reshape(len(values) // block_size, block_size, values.shape[1])
    indices = rng.integers(0, len(blocks), size=len(blocks))
    return blocks[indices].reshape(values.shape)


def ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def physical_metrics(
    target_values: np.ndarray,
    candidate_values: dict[str, np.ndarray],
    selected_id: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    target_mean, target_cov = moments(target_values)
    observed: dict[str, dict[str, float]] = {}
    for action_id, values in candidate_values.items():
        candidate_mean, candidate_cov = moments(values)
        total, mean_part, cov_part = gaussian_kl_components(target_mean, target_cov, candidate_mean, candidate_cov)
        observed[action_id] = {"kl": total, "mean_kl": mean_part, "covariance_kl": cov_part}
    oracle_id = min(observed, key=lambda key: (observed[key]["kl"], key))
    action_ids = sorted(candidate_values)
    boot = np.zeros((n_bootstrap, len(action_ids)))
    for replicate in range(n_bootstrap):
        boot_target = block_bootstrap(target_values, rng)
        target_boot_moments = moments(boot_target)
        for column, action_id in enumerate(action_ids):
            candidate_boot_moments = moments(block_bootstrap(candidate_values[action_id], rng))
            boot[replicate, column] = gaussian_kl_components(*target_boot_moments, *candidate_boot_moments)[0]
    lookup = {action_id: index for index, action_id in enumerate(action_ids)}
    selected_boot = boot[:, lookup[selected_id]]
    identity_boot = boot[:, lookup["A_ID"]]
    oracle_boot = np.min(boot, axis=1)
    selected_low, selected_high = ci(selected_boot)
    identity_low, identity_high = ci(identity_boot)
    regret_boot = selected_boot - oracle_boot
    regret_low, regret_high = ci(regret_boot)
    improvement_boot = identity_boot - selected_boot
    improvement_low, improvement_high = ci(improvement_boot)
    oracle_fixed = boot[:, lookup[oracle_id]]
    physical_top_set = []
    for action_id in action_ids:
        difference = boot[:, lookup[action_id]] - oracle_fixed
        if ci(difference)[0] <= 0.0:
            physical_top_set.append(action_id)
        low, high = ci(boot[:, lookup[action_id]])
        observed[action_id]["bootstrap_kl_ci_low"] = low
        observed[action_id]["bootstrap_kl_ci_high"] = high
    metrics = {
        "selected_physical_kl": observed[selected_id]["kl"],
        "selected_mean_kl_component": observed[selected_id]["mean_kl"],
        "selected_covariance_kl_component": observed[selected_id]["covariance_kl"],
        "selected_physical_kl_ci_low": selected_low,
        "selected_physical_kl_ci_high": selected_high,
        "identity_kl": observed["A_ID"]["kl"],
        "identity_kl_ci_low": identity_low,
        "identity_kl_ci_high": identity_high,
        "delta_identity": observed["A_ID"]["kl"] - observed[selected_id]["kl"],
        "delta_identity_ci_low": improvement_low,
        "delta_identity_ci_high": improvement_high,
        "oracle_id": oracle_id,
        "oracle_kl": observed[oracle_id]["kl"],
        "regret": observed[selected_id]["kl"] - observed[oracle_id]["kl"],
        "regret_ci_low": regret_low,
        "regret_ci_high": regret_high,
        "top1_oracle_agreement": selected_id == oracle_id,
        "physical_top_set_ids": ";".join(physical_top_set),
        "physical_top_set_size": len(physical_top_set),
        "selected_in_physical_top_set": selected_id in physical_top_set,
    }
    return metrics, observed


def rank_metrics(predicted: dict[str, float], physical: dict[str, dict[str, float]], selected_id: str) -> dict[str, Any]:
    action_ids = sorted(predicted)
    predicted_values = np.array([predicted[action_id] for action_id in action_ids])
    physical_values = np.array([physical[action_id]["kl"] for action_id in action_ids])
    correlation = stats.spearmanr(predicted_values, physical_values).statistic
    physical_order = sorted(action_ids, key=lambda action_id: (physical[action_id]["kl"], action_id))
    return {
        "spearman_rank_correlation": float(correlation),
        "selected_physical_rank": physical_order.index(selected_id) + 1,
        "selected_in_physical_top3": selected_id in physical_order[:3],
    }


def prepare_models(raw_root: Path, catalog: pd.DataFrame) -> tuple[dict[str, Fit], dict[str, tuple[np.ndarray, np.ndarray]], dict[str, pd.DataFrame]]:
    frames = {"A_ID": load(raw_root, REFERENCE)}
    for row in catalog.itertuples(index=False):
        frames[row.action_id] = load(raw_root, row.environment)
    reference_fit = frames["A_ID"].iloc[SOURCE_FIT.start:SOURCE_FIT.stop]
    source_fits = {outcome: fit_ols(reference_fit, outcome, SIGNAL_PARENTS[outcome]) for outcome in OUTCOMES}
    root_values = reference_fit.loc[:, ROOTS].to_numpy(float)
    root_mean, root_cov = moments(root_values)
    common_models: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    source_residual_cov = regular_cov(residual_matrix(reference_fit, source_fits))
    common_models["A_ID"] = compose_common_moments(root_mean, root_cov, source_fits, source_residual_cov)
    for row in catalog.itertuples(index=False):
        calibration = frames[row.action_id].iloc[MODE_FIT.start:MODE_FIT.stop]
        fits = dict(source_fits)
        fits[row.node] = fit_m2(calibration, row.node, source_fits[row.node])
        residual_cov = regular_cov(residual_matrix(calibration, fits))
        common_models[row.action_id] = compose_common_moments(root_mean, root_cov, fits, residual_cov)
    return source_fits, common_models, frames


def common_state_diagnostics(
    catalog: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    common_models: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    reference_fit = frames["A_ID"].iloc[SOURCE_FIT.start:SOURCE_FIT.stop].loc[:, COMMON_STATE]
    center = reference_fit.mean().to_numpy(float)
    scale = reference_fit.std(ddof=1).to_numpy(float)
    environment_lookup = catalog.set_index("action_id")["environment"].to_dict()
    rows = []
    for action_id in ["A_ID", *catalog["action_id"].tolist()]:
        model_covariance = transform_moments(*common_models[action_id], center, scale)[1]
        model_eigenvalues = np.linalg.eigvalsh(model_covariance)
        if action_id == "A_ID":
            physical = frames[action_id].iloc[IDENTITY_EVALUATION.start:IDENTITY_EVALUATION.stop].loc[:, COMMON_STATE].to_numpy(float)
        else:
            physical = frames[action_id].iloc[CANDIDATE_EVALUATION.start:CANDIDATE_EVALUATION.stop].loc[:, COMMON_STATE].to_numpy(float)
        physical_covariance = regular_cov(standardize(physical, center, scale))
        physical_eigenvalues = np.linalg.eigvalsh(physical_covariance)
        model_condition = float(model_eigenvalues.max() / model_eigenvalues.min())
        physical_condition = float(physical_eigenvalues.max() / physical_eigenvalues.min())
        rows.append({
            "action_id": action_id,
            "environment": REFERENCE if action_id == "A_ID" else environment_lookup[action_id],
            "dimension": len(COMMON_STATE),
            "evaluation_rows": len(physical),
            "model_min_eigenvalue": float(model_eigenvalues.min()),
            "model_condition_number": model_condition,
            "physical_min_eigenvalue": float(physical_eigenvalues.min()),
            "physical_condition_number": physical_condition,
            "stable": bool(model_eigenvalues.min() > 0 and physical_eigenvalues.min() > 0 and max(model_condition, physical_condition) < 1e8),
        })
    return pd.DataFrame(rows)


def local_model_moments(
    reference_fit: pd.DataFrame,
    source_fit: Fit,
    action_fit: Fit,
    nodes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    parents = nodes[:-1]
    root_values = reference_fit.loc[:, parents].to_numpy(float)
    root_mean, root_cov = moments(root_values)
    mean_y = action_fit.alpha + action_fit.beta @ root_mean
    cross = root_cov @ action_fit.beta
    mean = np.r_[root_mean, mean_y]
    covariance = np.block([
        [root_cov, cross[:, None]],
        [cross[None, :], np.array([[action_fit.beta @ root_cov @ action_fit.beta + action_fit.omega]])],
    ])
    return mean, covariance + 1e-9 * np.eye(len(nodes))


def decode(action_id: str, decode_lookup: dict[str, dict[str, Any]], prefix: str) -> dict[str, Any]:
    decoded = decode_lookup[action_id]
    return {
        f"{prefix}_action_id": action_id,
        f"{prefix}_environment": decoded["environment"],
        f"{prefix}_node": decoded["node"],
        f"{prefix}_mode": decoded["mode"],
    }


def run_experiment_a(
    catalog: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    source_fits: dict[str, Fit],
    decode_lookup: dict[str, dict[str, Any]],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = frames["A_ID"]
    reference_fit = reference.iloc[SOURCE_FIT.start:SOURCE_FIT.stop]
    results, rankings = [], []
    for target_row in catalog.itertuples(index=False):
        node = target_row.node
        nodes = (*SIGNAL_PARENTS[node], node)
        center = reference_fit.loc[:, nodes].mean().to_numpy(float)
        scale = reference_fit.loc[:, nodes].std(ddof=1).to_numpy(float)
        target_selection = standardize(frames[target_row.action_id].iloc[TARGET_SELECTION.start:TARGET_SELECTION.stop].loc[:, nodes].to_numpy(float), center, scale)
        target_eval = standardize(frames[target_row.action_id].iloc[TARGET_EVALUATION.start:TARGET_EVALUATION.stop].loc[:, nodes].to_numpy(float), center, scale)
        target_selection_moments = moments(target_selection)
        candidate_rows = catalog[(catalog["node"] == node) & (catalog["action_id"] != target_row.action_id)]
        candidate_ids = ["A_ID", *candidate_rows["action_id"].tolist()]
        if target_row.action_id in candidate_ids:
            raise AssertionError("Experiment A generating action remains in catalog")
        predicted: dict[str, float] = {}
        identity_model = local_model_moments(reference_fit, source_fits[node], source_fits[node], nodes)
        identity_model_z = transform_moments(*identity_model, center, scale)
        predicted["A_ID"] = gaussian_kl_components(*target_selection_moments, *identity_model_z)[0]
        for action_id in candidate_rows["action_id"]:
            calibration = frames[action_id].iloc[MODE_FIT.start:MODE_FIT.stop]
            action_fit = fit_m2(calibration, node, source_fits[node])
            model_z = transform_moments(*local_model_moments(reference_fit, source_fits[node], action_fit, nodes), center, scale)
            predicted[action_id] = gaussian_kl_components(*target_selection_moments, *model_z)[0]
        selected_id = select_action(predicted)
        candidate_values = {"A_ID": standardize(reference.iloc[IDENTITY_EVALUATION.start:IDENTITY_EVALUATION.stop].loc[:, nodes].to_numpy(float), center, scale)}
        for action_id in candidate_rows["action_id"]:
            candidate_values[action_id] = standardize(frames[action_id].iloc[CANDIDATE_EVALUATION.start:CANDIDATE_EVALUATION.stop].loc[:, nodes].to_numpy(float), center, scale)
        metrics, physical = physical_metrics(target_eval, candidate_values, selected_id, rng, n_bootstrap)
        row = {
            "experiment": "A_M5_per_node_leave_generating_mode_out",
            "target_action_id": target_row.action_id,
            "target_environment": target_row.environment,
            "target_node": node,
            "target_mode": target_row.mode,
            "target_gain_classification": target_row.gain_classification,
            "n_candidate_actions_including_identity": len(candidate_ids),
            "common_state": ";".join(nodes),
            "selected_predicted_kl": predicted[selected_id],
            **decode(selected_id, decode_lookup, "selected"),
            **metrics,
            **decode(metrics["oracle_id"], decode_lookup, "oracle"),
            **rank_metrics(predicted, physical, selected_id),
        }
        results.append(row)
        predicted_order = sorted(predicted, key=lambda key: (predicted[key], key))
        physical_order = sorted(physical, key=lambda key: (physical[key]["kl"], key))
        for action_id in candidate_ids:
            rankings.append({
                "experiment": "A_M5_per_node_leave_generating_mode_out",
                "target_action_id": target_row.action_id,
                "candidate_action_id": action_id,
                "candidate_environment": decode_lookup[action_id]["environment"],
                "candidate_node": decode_lookup[action_id]["node"],
                "candidate_mode": decode_lookup[action_id]["mode"],
                "predicted_kl": predicted[action_id],
                "predicted_rank": predicted_order.index(action_id) + 1,
                "physical_kl": physical[action_id]["kl"],
                "physical_mean_kl_component": physical[action_id]["mean_kl"],
                "physical_covariance_kl_component": physical[action_id]["covariance_kl"],
                "physical_rank": physical_order.index(action_id) + 1,
                "physical_kl_ci_low": physical[action_id]["bootstrap_kl_ci_low"],
                "physical_kl_ci_high": physical[action_id]["bootstrap_kl_ci_high"],
            })
    return pd.DataFrame(results), pd.DataFrame(rankings)


def run_experiment_b_variant(
    variant: str,
    leave_out: bool,
    catalog: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    common_models: dict[str, tuple[np.ndarray, np.ndarray]],
    decode_lookup: dict[str, dict[str, Any]],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = frames["A_ID"]
    source = reference.iloc[SOURCE_FIT.start:SOURCE_FIT.stop].loc[:, COMMON_STATE]
    center = source.mean().to_numpy(float)
    scale = source.std(ddof=1).to_numpy(float)
    model_z = {action_id: transform_moments(*model, center, scale) for action_id, model in common_models.items()}
    results, rankings = [], []
    all_action_ids = catalog["action_id"].tolist()
    for target_row in catalog.itertuples(index=False):
        candidate_ids = ["A_ID", *all_action_ids]
        if leave_out:
            candidate_ids.remove(target_row.action_id)
        if leave_out and target_row.action_id in candidate_ids:
            raise AssertionError("leave-out generating action remains in catalog")
        target_selection = standardize(frames[target_row.action_id].iloc[TARGET_SELECTION.start:TARGET_SELECTION.stop].loc[:, COMMON_STATE].to_numpy(float), center, scale)
        target_eval = standardize(frames[target_row.action_id].iloc[TARGET_EVALUATION.start:TARGET_EVALUATION.stop].loc[:, COMMON_STATE].to_numpy(float), center, scale)
        target_selection_moments = moments(target_selection)
        predicted = {action_id: gaussian_kl_components(*target_selection_moments, *model_z[action_id])[0] for action_id in candidate_ids}
        selected_id = select_action(predicted)
        candidate_values = {"A_ID": standardize(reference.iloc[IDENTITY_EVALUATION.start:IDENTITY_EVALUATION.stop].loc[:, COMMON_STATE].to_numpy(float), center, scale)}
        for action_id in candidate_ids:
            if action_id != "A_ID":
                candidate_values[action_id] = standardize(frames[action_id].iloc[CANDIDATE_EVALUATION.start:CANDIDATE_EVALUATION.stop].loc[:, COMMON_STATE].to_numpy(float), center, scale)
        metrics, physical = physical_metrics(target_eval, candidate_values, selected_id, rng, n_bootstrap)
        selected = decode_lookup[selected_id]
        exact = selected_id == target_row.action_id
        node_correct = selected["node"] == target_row.node
        row = {
            "experiment": variant,
            "generating_mode_available": not leave_out,
            "target_action_id": target_row.action_id,
            "target_environment": target_row.environment,
            "target_node": target_row.node,
            "target_mode": target_row.mode,
            "target_gain_classification": target_row.gain_classification,
            "n_candidate_actions_including_identity": len(candidate_ids),
            "common_state": ";".join(COMMON_STATE),
            "selected_predicted_kl": predicted[selected_id],
            **decode(selected_id, decode_lookup, "selected"),
            "node_selection_correct": node_correct if not leave_out else None,
            "mode_correct_conditional_on_node": (exact if node_correct else False) if not leave_out else None,
            "joint_node_mode_top1": exact if not leave_out else None,
            **metrics,
            **decode(metrics["oracle_id"], decode_lookup, "oracle"),
            **rank_metrics(predicted, physical, selected_id),
        }
        results.append(row)
        predicted_order = sorted(predicted, key=lambda key: (predicted[key], key))
        physical_order = sorted(physical, key=lambda key: (physical[key]["kl"], key))
        for action_id in candidate_ids:
            rankings.append({
                "experiment": variant,
                "target_action_id": target_row.action_id,
                "candidate_action_id": action_id,
                "candidate_environment": decode_lookup[action_id]["environment"],
                "candidate_node": decode_lookup[action_id]["node"],
                "candidate_mode": decode_lookup[action_id]["mode"],
                "predicted_kl": predicted[action_id],
                "predicted_rank": predicted_order.index(action_id) + 1,
                "physical_kl": physical[action_id]["kl"],
                "physical_mean_kl_component": physical[action_id]["mean_kl"],
                "physical_covariance_kl_component": physical[action_id]["covariance_kl"],
                "physical_rank": physical_order.index(action_id) + 1,
                "physical_kl_ci_low": physical[action_id]["bootstrap_kl_ci_low"],
                "physical_kl_ci_high": physical[action_id]["bootstrap_kl_ci_high"],
            })
    return pd.DataFrame(results), pd.DataFrame(rankings)


def aggregate(experiment: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_targets": len(experiment),
        "median_selected_kl": float(experiment["selected_physical_kl"].median()),
        "mean_selected_kl": float(experiment["selected_physical_kl"].mean()),
        "median_identity_kl": float(experiment["identity_kl"].median()),
        "mean_identity_improvement": float(experiment["delta_identity"].mean()),
        "identity_improved_fraction": float((experiment["delta_identity"] > 0).mean()),
        "median_oracle_kl": float(experiment["oracle_kl"].median()),
        "mean_regret": float(experiment["regret"].mean()),
        "median_regret": float(experiment["regret"].median()),
        "top1_oracle_agreement_fraction": float(experiment["top1_oracle_agreement"].mean()),
        "selected_in_physical_top_set_fraction": float(experiment["selected_in_physical_top_set"].mean()),
        "selected_in_physical_top3_fraction": float(experiment["selected_in_physical_top3"].mean()),
        "median_spearman": float(experiment["spearman_rank_correlation"].median()),
    }
    if "node_selection_correct" in experiment and experiment["node_selection_correct"].notna().any():
        valid = experiment[experiment["node_selection_correct"].notna()]
        result.update({
            "node_selection_accuracy": float(valid["node_selection_correct"].mean()),
            "mode_accuracy_conditional_on_selected_node": float(valid.loc[valid["node_selection_correct"], "mode_correct_conditional_on_node"].mean()) if valid["node_selection_correct"].any() else None,
            "joint_node_mode_top1_accuracy": float(valid["joint_node_mode_top1"].mean()),
        })
    return result


def stratify(frame: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    local = frame.copy()
    local["effect_tertile"] = pd.qcut(local["identity_kl"].rank(method="first"), 3, labels=["small", "medium", "large"])
    rows = []
    group_specs = [
        ("target_node", local["target_node"]),
        ("effect_tertile", local["effect_tertile"]),
        ("gain_semantics", local["target_gain_classification"].replace({"PASS": "common-gain-compatible", "BORDERLINE": "failed-or-bordered", "FAIL": "failed-or-bordered"})),
    ]
    for stratum, values in group_specs:
        local["_group"] = values
        for group, part in local.groupby("_group", observed=True):
            rows.append({
                "experiment": experiment_name,
                "stratum": stratum,
                "group": str(group),
                "n_targets": len(part),
                "median_selected_kl": float(part["selected_physical_kl"].median()),
                "mean_identity_improvement": float(part["delta_identity"].mean()),
                "mean_regret": float(part["regret"].mean()),
                "top1_oracle_agreement_fraction": float(part["top1_oracle_agreement"].mean()),
                "median_spearman": float(part["spearman_rank_correlation"].median()),
            })
    return pd.DataFrame(rows)


def make_figures(a: pd.DataFrame, b: pd.DataFrame, b_leave: pd.DataFrame) -> None:
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, frame, title in zip(axes, (a, b, b_leave), ("A: per-node M5", "B: full catalog", "B+M5: full catalog")):
        ordered = frame.sort_values(["target_node", "target_environment"]).reset_index(drop=True)
        axis.scatter(np.arange(len(ordered)), ordered["regret"], c=ordered["delta_identity"], cmap="coolwarm", s=24)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("target (node-grouped)")
    axes[0].set_ylabel("held-out physical oracle regret")
    fig.tight_layout()
    fig.savefig(figure_dir / "oracle_regret_by_target.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(b["oracle_kl"], b["selected_physical_kl"], c=b["top1_oracle_agreement"], cmap="coolwarm", s=35)
    limit = float(max(b["oracle_kl"].max(), b["selected_physical_kl"].max()))
    axis.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("physical oracle KL")
    axis.set_ylabel("selected physical KL")
    axis.set_title("Experiment B selected versus oracle")
    fig.tight_layout()
    fig.savefig(figure_dir / "experiment_b_selected_vs_oracle.png", dpi=180)
    plt.close(fig)


def build_catalog(raw_root: Path) -> pd.DataFrame:
    classifications = pd.read_csv(raw_root / "support" / "mode_classifications.csv")
    classification_lookup = classifications.set_index("environment")["classification"].to_dict()
    rows = []
    for index, environment in enumerate(action_environments(raw_root)):
        node, control, level = infer_action(environment)
        rows.append({
            "action_id": f"A{index:03d}",
            "environment": environment,
            "node": node,
            "physical_control": control,
            "level": level,
            "mode": f"{control}:{level}",
            "gain_classification": classification_lookup[environment],
            "common_gain_compatible": classification_lookup[environment] == "PASS",
        })
    catalog = pd.DataFrame(rows)
    if len(catalog) != 27 or set(catalog["node"]) != set(OUTCOMES):
        raise ValueError("expected 27 modes on six nodes")
    return catalog


def main(raw_root: Path, n_bootstrap: int = N_BOOTSTRAP) -> None:
    validate_protocol()
    catalog = build_catalog(raw_root)
    schemas = validate_schemas(raw_root, [REFERENCE, *catalog["environment"]])
    decode_lookup = {
        "A_ID": {"environment": REFERENCE, "node": "identity", "mode": "identity"},
        **{row.action_id: {"environment": row.environment, "node": row.node, "mode": row.mode} for row in catalog.itertuples(index=False)},
    }
    source_fits, common_models, frames = prepare_models(raw_root, catalog)
    state_diagnostics = common_state_diagnostics(catalog, frames, common_models)
    if not state_diagnostics["stable"].all():
        raise ValueError("full common state failed numerical stability gate")
    rng = np.random.default_rng(SEED)
    experiment_a, ranking_a = run_experiment_a(catalog, frames, source_fits, decode_lookup, rng, n_bootstrap)
    experiment_b, ranking_b = run_experiment_b_variant("B_M3_full_27_action_catalog", False, catalog, frames, common_models, decode_lookup, rng, n_bootstrap)
    experiment_b_leave, ranking_b_leave = run_experiment_b_variant("B_M3_plus_M5_leave_generating_mode_out", True, catalog, frames, common_models, decode_lookup, rng, n_bootstrap)

    write_frame("action_catalog", catalog)
    write_frame("schema_compatibility", schemas)
    write_frame("common_state_diagnostics", state_diagnostics)
    write_frame("experiment_a_per_target", experiment_a)
    write_frame("experiment_a_rankings", ranking_a)
    write_frame("experiment_b_per_target", experiment_b)
    write_frame("experiment_b_rankings", ranking_b)
    write_frame("experiment_b_leave_out_per_target", experiment_b_leave)
    write_frame("experiment_b_leave_out_rankings", ranking_b_leave)
    stratified = pd.concat([
        stratify(experiment_a, "A_M5_per_node_leave_generating_mode_out"),
        stratify(experiment_b, "B_M3_full_27_action_catalog"),
        stratify(experiment_b_leave, "B_M3_plus_M5_leave_generating_mode_out"),
    ], ignore_index=True)
    write_frame("stratified_results", stratified)
    summary = {
        "scientific_label": "empirical finite gain-mode / mode-specific-noise real-hardware B=1 target matching",
        "claim_exclusions": ["exact invariant-noise shift-scale validation", "independent-session validation", "active acquisition", "B>1 composition"],
        "experiment_a": aggregate(experiment_a),
        "experiment_b": aggregate(experiment_b),
        "experiment_b_leave_out": aggregate(experiment_b_leave),
        "common_gain_modes": int(catalog["common_gain_compatible"].sum()),
        "failed_or_bordered_gain_modes": int((~catalog["common_gain_compatible"]).sum()),
    }
    write_json(OUTPUT / "machine_readable" / "summary.json", summary)
    protocol = {
        "source_reference_calibration": {"environment": REFERENCE, "rows": [SOURCE_FIT.start, SOURCE_FIT.stop]},
        "candidate_mode_calibration": {"environment": "each action", "rows": [MODE_FIT.start, MODE_FIT.stop]},
        "target_selection": {"environment": "hidden target", "rows": [TARGET_SELECTION.start, TARGET_SELECTION.stop]},
        "candidate_physical_evaluation": {"environment": "each candidate", "rows": [CANDIDATE_EVALUATION.start, CANDIDATE_EVALUATION.stop]},
        "target_physical_evaluation": {"environment": "hidden target", "rows": [TARGET_EVALUATION.start, TARGET_EVALUATION.stop]},
        "identity_physical_evaluation": {"environment": REFERENCE, "rows": [IDENTITY_EVALUATION.start, IDENTITY_EVALUATION.stop]},
        "bootstrap": {"method": "resample contiguous non-overlapping blocks with replacement", "block_rows": BLOCK_SIZE, "replicates": n_bootstrap, "scope": "within-session sampling uncertainty, not physical replication"},
        "selector_signature": str(inspect.signature(select_action)),
        "opaque_ids_used_for_selection": True,
    }
    write_json(OUTPUT / "machine_readable" / "frozen_protocol.json", protocol)
    common_state = {
        "gate": "PASS",
        "variables": list(COMMON_STATE),
        "dimension": len(COMMON_STATE),
        "definition": "predeclared causally closed union of all signal parents and six optical responses",
        "selection_model": "known-graph linear conditional means with one M2 mode-specific-noise action fit and full six-response residual covariance from calibration rows",
        "comparability_reason": "identical ordered 46-column schema and shared randomized root/input regime in every same-session action and reference file",
        "numerical_preprocessing": "source-calibration z-scoring; invertible scaling used for conditioning; Gaussian covariance ridge 1e-6",
        "maximum_model_condition_number": float(state_diagnostics["model_condition_number"].max()),
        "maximum_physical_condition_number": float(state_diagnostics["physical_condition_number"].max()),
        "minimum_covariance_eigenvalue": float(min(state_diagnostics["model_min_eigenvalue"].min(), state_diagnostics["physical_min_eigenvalue"].min())),
        "all_28_identity_or_action_states_numerically_stable": bool(state_diagnostics["stable"].all()),
        "target_specific": False,
    }
    write_json(OUTPUT / "machine_readable" / "common_state_gate.json", common_state)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unavailable"
    metadata = {
        "seed": SEED,
        "repository_commit_before_run": commit,
        "dataset": DATASET,
        "reference_sha256": sha256(raw_root / DATASET / f"{REFERENCE}.csv"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "prior_artifacts_reused": [
            "outputs/mk1_heteroskedastic_e2e/machine_readable/b1_matching.json",
            "outputs/mk1_heteroskedastic_e2e/machine_readable/hardware_mode_catalog.json",
            "outputs/lt_mk1_multiparent_scaling/tables/mode_classifications.csv",
            "outputs/lt_mk1_b1_cross_archive_ranking/machine_readable/comparability_gate.json",
        ],
        "prohibited_prior_experiments_rerun": [],
    }
    write_json(OUTPUT / "machine_readable" / "run_metadata.json", metadata)
    make_figures(experiment_a, experiment_b, experiment_b_leave)
    completion = {
        "status": "complete",
        "bootstrap_replicates": n_bootstrap,
        "expected_target_tasks_per_experiment": 27,
        "experiment_a_rows": len(experiment_a),
        "experiment_b_rows": len(experiment_b),
        "experiment_b_leave_out_rows": len(experiment_b_leave),
    }
    write_json(OUTPUT / "machine_readable" / "completion.json", completion)
    print(json.dumps(completion, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    OUTPUT = prepare_output_directory(
        arguments.output,
        overwrite=arguments.overwrite,
        protected=(arguments.data_root,),
    )
    main(arguments.data_root.resolve(), arguments.bootstrap)
