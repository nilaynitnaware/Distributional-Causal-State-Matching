"""Leakage-safe Light Tunnel Mk1 B=1 root shift matching.

The primary selector is deliberately constrained to the original linear-Gaussian
shift semantics: action calibration estimates only the shift magnitude at the
manipulated root; the direction and covariance come from the source SCM.
"""

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

import numpy as np
import pandas as pd
import scipy
from scipy import linalg, stats


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "outputs" / "mk1_shift_matching"
DATASET = "lt_interventions_standard_v1"
REFERENCE = "uniform_reference"
SEED = 20260831
N_BOOTSTRAP = 500
BLOCK_SIZE = 50

ACTUATORS = ("red", "green", "blue", "pol_1", "pol_2")
LOCAL_ROOTS = ("l_11", "l_12", "l_21", "l_22", "l_31", "l_32")
ROOTS = (*ACTUATORS, *LOCAL_ROOTS)
OUTCOMES = ("ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3")
COMMON_STATE = (*ROOTS, *OUTCOMES)
DESCENDANT_STATE = (*LOCAL_ROOTS, *OUTCOMES)
STATE_DEFINITIONS = {
    "full_state": COMMON_STATE,
    "descendant_only": DESCENDANT_STATE,
}

SIGNAL_PARENTS: dict[str, tuple[str, ...]] = {
    "ir_1": ("red", "green", "blue", "l_11", "l_12"),
    "vis_1": ("red", "green", "blue", "l_11", "l_12"),
    "ir_2": ("red", "green", "blue", "l_21", "l_22"),
    "vis_2": ("red", "green", "blue", "l_21", "l_22"),
    "ir_3": ("red", "green", "blue", "pol_1", "pol_2", "l_31", "l_32"),
    "vis_3": ("red", "green", "blue", "pol_1", "pol_2", "l_31", "l_32"),
}

# Frozen before matching results are computed. These are descriptive within-session
# tolerances, not hypothesis-test thresholds or between-session guarantees.
INVARIANCE_TOLERANCE = {
    "root_variance_ratio_low": 0.8,
    "root_variance_ratio_high": 1.25,
    "full_state_relative_covariance_frobenius_max": 0.25,
    "descendant_relative_covariance_frobenius_max": 0.25,
}


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
ACTION_FIT = RowBlock("action", 0, 400)
TARGET_SELECTION = RowBlock("target", 400, 600)
CANDIDATE_EVALUATION = RowBlock("action", 600, 800)
TARGET_EVALUATION = RowBlock("target", 800, 1000)
IDENTITY_EVALUATION = RowBlock("reference", 8000, 8200)


@dataclass(frozen=True)
class Fit:
    alpha: float
    beta: np.ndarray


@dataclass(frozen=True)
class GaussianModel:
    mean: np.ndarray
    covariance: np.ndarray


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
    frame.to_csv(table, index=False, lineterminator="\n")
    write_json(machine, frame.to_dict(orient="records"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol() -> None:
    if ACTION_FIT.stop > TARGET_SELECTION.start:
        raise ValueError("target selection rows enter action calibration")
    if ACTION_FIT.overlaps(CANDIDATE_EVALUATION):
        raise ValueError("candidate evaluation rows enter candidate fitting")
    if TARGET_SELECTION.overlaps(TARGET_EVALUATION):
        raise ValueError("target evaluation rows enter selection")
    if CANDIDATE_EVALUATION.row_range_overlaps(TARGET_SELECTION):
        raise ValueError("candidate evaluation overlaps target selection row range")
    if TARGET_EVALUATION.row_range_overlaps(TARGET_SELECTION):
        raise ValueError("target evaluation overlaps target selection row range")
    if IDENTITY_EVALUATION.overlaps(SOURCE_FIT):
        raise ValueError("identity evaluation overlaps source calibration")
    if IDENTITY_EVALUATION.row_range_overlaps(TARGET_EVALUATION):
        raise ValueError("identity evaluation overlaps target evaluation row range")


def load(raw_root: Path, environment: str) -> pd.DataFrame:
    path = raw_root / DATASET / f"{environment}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_catalog(raw_root: Path) -> pd.DataFrame:
    expected = sorted(f"uniform_{node}_{level}" for node in ACTUATORS for level in ("mid", "strong"))
    missing = [environment for environment in expected if not (raw_root / DATASET / f"{environment}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"missing root action files: {missing}")
    rows = []
    for index, environment in enumerate(expected):
        body = environment.removeprefix("uniform_")
        level = "strong" if body.endswith("_strong") else "mid"
        node = body.removesuffix(f"_{level}")
        rows.append({
            "action_id": f"S{index:03d}",
            "environment": environment,
            "node": node,
            "strength": level,
            "mode": f"{node}:{level}",
            "unit_cost": 1,
            "cached_semantics": "APPROXIMATE SHIFT-LIKE",
        })
    catalog = pd.DataFrame(rows)
    if len(catalog) != 10 or set(catalog["node"]) != set(ACTUATORS):
        raise ValueError("expected exactly ten mid/strong actions on five roots")
    return catalog


def validate_schemas(raw_root: Path, environments: Iterable[str]) -> pd.DataFrame:
    reference_columns = list(pd.read_csv(raw_root / DATASET / f"{REFERENCE}.csv", nrows=0).columns)
    required = set(COMMON_STATE)
    if not required.issubset(reference_columns):
        raise ValueError("predeclared common state missing from reference")
    rows = []
    for environment in environments:
        columns = list(pd.read_csv(raw_root / DATASET / f"{environment}.csv", nrows=0).columns)
        exact = columns == reference_columns
        contains = required.issubset(columns)
        ordered = [item for item in columns if item in required] == [item for item in reference_columns if item in required]
        rows.append({
            "environment": environment,
            "n_columns": len(columns),
            "exact_reference_schema": exact,
            "contains_common_state": contains,
            "common_state_order_matches": ordered,
        })
        if not exact or not contains or not ordered:
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
    mean_part = float(0.5 * difference @ solved_mean)
    covariance_part = float(0.5 * (np.trace(solved_cov) - dimension + logdet_p - logdet_q))
    return mean_part + covariance_part, mean_part, covariance_part


def standardize(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - center) / scale


def transform_model(model: GaussianModel, center: np.ndarray, scale: np.ndarray) -> GaussianModel:
    inverse = 1.0 / scale
    return GaussianModel((model.mean - center) * inverse, model.covariance * np.outer(inverse, inverse))


def subset_model(model: GaussianModel, indices: np.ndarray) -> GaussianModel:
    return GaussianModel(model.mean[indices], model.covariance[np.ix_(indices, indices)])


def fit_ols(frame: pd.DataFrame, outcome: str) -> Fit:
    parents = SIGNAL_PARENTS[outcome]
    x = frame.loc[:, parents].to_numpy(float)
    design = np.column_stack((np.ones(len(frame)), x))
    coefficients, *_ = np.linalg.lstsq(design, frame[outcome].to_numpy(float), rcond=None)
    return Fit(float(coefficients[0]), coefficients[1:].astype(float))


def fit_source_scm(frame: pd.DataFrame) -> tuple[GaussianModel, dict[str, Fit], dict[str, np.ndarray]]:
    fits = {outcome: fit_ols(frame, outcome) for outcome in OUTCOMES}
    root_values = frame.loc[:, ROOTS].to_numpy(float)
    root_mean, root_cov = moments(root_values)
    root_lookup = {name: index for index, name in enumerate(ROOTS)}
    coefficient_matrix = np.zeros((len(OUTCOMES), len(ROOTS)))
    intercept = np.zeros(len(OUTCOMES))
    residuals = []
    for row, outcome in enumerate(OUTCOMES):
        fit = fits[outcome]
        intercept[row] = fit.alpha
        for parent, coefficient in zip(SIGNAL_PARENTS[outcome], fit.beta):
            coefficient_matrix[row, root_lookup[parent]] = coefficient
        x = frame.loc[:, SIGNAL_PARENTS[outcome]].to_numpy(float)
        residuals.append(frame[outcome].to_numpy(float) - fit.alpha - x @ fit.beta)
    residual_cov = regular_cov(np.column_stack(residuals))
    outcome_mean = intercept + coefficient_matrix @ root_mean
    cross = root_cov @ coefficient_matrix.T
    outcome_cov = coefficient_matrix @ root_cov @ coefficient_matrix.T + residual_cov
    mean = np.r_[root_mean, outcome_mean]
    covariance = np.block([[root_cov, cross], [cross.T, outcome_cov]]) + 1e-9 * np.eye(len(COMMON_STATE))
    effects: dict[str, np.ndarray] = {}
    for actuator in ACTUATORS:
        direction = np.zeros(len(COMMON_STATE))
        root_index = root_lookup[actuator]
        direction[root_index] = 1.0
        direction[len(ROOTS):] = coefficient_matrix[:, root_index]
        effects[actuator] = direction
    return GaussianModel(mean, covariance), fits, effects


def candidate_models(
    catalog: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    source_model: GaussianModel,
    effects: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, GaussianModel]], pd.DataFrame]:
    models: dict[str, dict[str, GaussianModel]] = {
        "shift_model": {"A_ID": source_model},
        "empirical_finite_mode": {"A_ID": source_model},
        "shift_mean_action_covariance": {"A_ID": source_model},
    }
    rows = []
    root_lookup = {name: index for index, name in enumerate(COMMON_STATE)}
    for action in catalog.itertuples(index=False):
        calibration = frames[action.action_id].iloc[ACTION_FIT.start:ACTION_FIT.stop]
        root_index = root_lookup[action.node]
        shift = float(calibration[action.node].mean() - source_model.mean[root_index])
        shift_mean = source_model.mean + shift * effects[action.node]
        empirical_mean, empirical_cov = moments(calibration.loc[:, COMMON_STATE].to_numpy(float))
        models["shift_model"][action.action_id] = GaussianModel(shift_mean, source_model.covariance)
        models["empirical_finite_mode"][action.action_id] = GaussianModel(empirical_mean, empirical_cov)
        models["shift_mean_action_covariance"][action.action_id] = GaussianModel(shift_mean, empirical_cov)
        rows.append({
            "action_id": action.action_id,
            "environment": action.environment,
            "node": action.node,
            "strength": action.strength,
            "estimated_shift_magnitude": shift,
            "magnitude_estimation_rows": f"[{ACTION_FIT.start}:{ACTION_FIT.stop}]",
            "direction_source": "source SCM total effect",
            "primary_covariance_source": "source SCM covariance",
        })
    return models, pd.DataFrame(rows)


def select_action(predicted_scores: dict[str, float]) -> str:
    """Select an opaque action ID using scores only."""
    if not predicted_scores:
        raise ValueError("empty candidate catalog")
    return min(predicted_scores, key=lambda action_id: (predicted_scores[action_id], action_id))


def candidate_ids_for_experiment(all_action_ids: list[str], target_id: str, leave_out: bool) -> list[str]:
    candidate_ids = ["A_ID", *all_action_ids]
    if leave_out:
        candidate_ids.remove(target_id)
    if leave_out and target_id in candidate_ids:
        raise AssertionError("S2 generating action remains in catalog")
    return candidate_ids


def block_bootstrap(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(values) % BLOCK_SIZE:
        raise ValueError("evaluation block does not divide into contiguous 50-row blocks")
    blocks = values.reshape(len(values) // BLOCK_SIZE, BLOCK_SIZE, values.shape[1])
    indices = rng.integers(0, len(blocks), size=len(blocks))
    return blocks[indices].reshape(values.shape)


def ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_physical(
    target_values: np.ndarray,
    candidate_values: dict[str, np.ndarray],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[dict[str, dict[str, float]], list[str], np.ndarray, str]:
    target_mean, target_cov = moments(target_values)
    observed: dict[str, dict[str, float]] = {}
    for action_id, values in candidate_values.items():
        candidate_mean, candidate_cov = moments(values)
        total, mean_part, cov_part = gaussian_kl_components(target_mean, target_cov, candidate_mean, candidate_cov)
        observed[action_id] = {"kl": total, "mean_kl": mean_part, "covariance_kl": cov_part}
    action_ids = sorted(candidate_values)
    oracle_id = min(action_ids, key=lambda item: (observed[item]["kl"], item))
    boot = np.zeros((n_bootstrap, len(action_ids)))
    for replicate in range(n_bootstrap):
        target_moments = moments(block_bootstrap(target_values, rng))
        for column, action_id in enumerate(action_ids):
            candidate_moments = moments(block_bootstrap(candidate_values[action_id], rng))
            boot[replicate, column] = gaussian_kl_components(*target_moments, *candidate_moments)[0]
    for column, action_id in enumerate(action_ids):
        low, high = ci(boot[:, column])
        observed[action_id]["bootstrap_kl_ci_low"] = low
        observed[action_id]["bootstrap_kl_ci_high"] = high
    return observed, action_ids, boot, oracle_id


def physical_metrics(
    selected_id: str,
    observed: dict[str, dict[str, float]],
    action_ids: list[str],
    boot: np.ndarray,
    oracle_id: str,
) -> dict[str, Any]:
    lookup = {action_id: index for index, action_id in enumerate(action_ids)}
    selected_boot = boot[:, lookup[selected_id]]
    identity_boot = boot[:, lookup["A_ID"]]
    adaptive_oracle_boot = np.min(boot, axis=1)
    fixed_oracle_boot = boot[:, lookup[oracle_id]]
    selected_low, selected_high = ci(selected_boot)
    improvement_low, improvement_high = ci(identity_boot - selected_boot)
    regret_low, regret_high = ci(selected_boot - adaptive_oracle_boot)
    fixed_low, fixed_high = ci(selected_boot - fixed_oracle_boot)
    top_set = []
    for action_id in action_ids:
        if ci(boot[:, lookup[action_id]] - fixed_oracle_boot)[0] <= 0:
            top_set.append(action_id)
    physical_order = sorted(action_ids, key=lambda item: (observed[item]["kl"], item))
    return {
        "selected_physical_kl": observed[selected_id]["kl"],
        "selected_mean_kl_component": observed[selected_id]["mean_kl"],
        "selected_covariance_kl_component": observed[selected_id]["covariance_kl"],
        "selected_physical_kl_ci_low": selected_low,
        "selected_physical_kl_ci_high": selected_high,
        "identity_kl": observed["A_ID"]["kl"],
        "delta_identity": observed["A_ID"]["kl"] - observed[selected_id]["kl"],
        "delta_identity_ci_low": improvement_low,
        "delta_identity_ci_high": improvement_high,
        "oracle_id": oracle_id,
        "oracle_kl": observed[oracle_id]["kl"],
        "regret": observed[selected_id]["kl"] - observed[oracle_id]["kl"],
        "regret_ci_low": regret_low,
        "regret_ci_high": regret_high,
        "selected_vs_fixed_oracle_ci_low": fixed_low,
        "selected_vs_fixed_oracle_ci_high": fixed_high,
        "top1_oracle_agreement": selected_id == oracle_id,
        "selected_physical_rank": physical_order.index(selected_id) + 1,
        "selected_in_physical_top3": selected_id in physical_order[:3],
        "physical_top_set_ids": ";".join(top_set),
        "physical_top_set_size": len(top_set),
        "selected_in_physical_top_set": selected_id in top_set,
    }


def decode(action_id: str, lookup: dict[str, dict[str, str]], prefix: str) -> dict[str, str]:
    item = lookup[action_id]
    return {
        f"{prefix}_action_id": action_id,
        f"{prefix}_environment": item["environment"],
        f"{prefix}_node": item["node"],
        f"{prefix}_strength": item["strength"],
    }


def run_matching(
    experiment: str,
    leave_out: bool,
    state_name: str,
    state: tuple[str, ...],
    catalog: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    models: dict[str, dict[str, GaussianModel]],
    decode_lookup: dict[str, dict[str, str]],
    source_center: np.ndarray,
    source_scale: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_indices = np.array([COMMON_STATE.index(variable) for variable in state])
    center = source_center[state_indices]
    scale = source_scale[state_indices]
    standardized_models: dict[str, dict[str, GaussianModel]] = {}
    for model_name, entries in models.items():
        standardized_models[model_name] = {
            action_id: transform_model(subset_model(model, state_indices), center, scale)
            for action_id, model in entries.items()
        }
    all_action_ids = catalog["action_id"].tolist()
    results: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    reference = frames["A_ID"]
    for target in catalog.itertuples(index=False):
        candidate_ids = candidate_ids_for_experiment(all_action_ids, target.action_id, leave_out)
        target_selection = standardize(
            frames[target.action_id].iloc[TARGET_SELECTION.start:TARGET_SELECTION.stop].loc[:, state].to_numpy(float), center, scale
        )
        target_evaluation = standardize(
            frames[target.action_id].iloc[TARGET_EVALUATION.start:TARGET_EVALUATION.stop].loc[:, state].to_numpy(float), center, scale
        )
        target_selection_moments = moments(target_selection)
        candidate_values = {
            "A_ID": standardize(
                reference.iloc[IDENTITY_EVALUATION.start:IDENTITY_EVALUATION.stop].loc[:, state].to_numpy(float), center, scale
            )
        }
        for action_id in candidate_ids:
            if action_id != "A_ID":
                candidate_values[action_id] = standardize(
                    frames[action_id].iloc[CANDIDATE_EVALUATION.start:CANDIDATE_EVALUATION.stop].loc[:, state].to_numpy(float), center, scale
                )
        observed, physical_ids, boot, oracle_id = bootstrap_physical(
            target_evaluation, candidate_values, rng, n_bootstrap
        )
        physical_order = sorted(physical_ids, key=lambda item: (observed[item]["kl"], item))
        for model_name in ("shift_model", "empirical_finite_mode", "shift_mean_action_covariance"):
            predicted: dict[str, float] = {}
            mean_only: dict[str, float] = {}
            for action_id in candidate_ids:
                total, mean_part, _ = gaussian_kl_components(
                    *target_selection_moments,
                    standardized_models[model_name][action_id].mean,
                    standardized_models[model_name][action_id].covariance,
                )
                predicted[action_id] = total
                mean_only[action_id] = mean_part
            selected_id = select_action(predicted)
            mean_only_selected = select_action(mean_only)
            metrics = physical_metrics(selected_id, observed, physical_ids, boot, oracle_id)
            predicted_order = sorted(candidate_ids, key=lambda item: (predicted[item], item))
            predicted_values = np.array([predicted[item] for item in sorted(candidate_ids)])
            physical_values = np.array([observed[item]["kl"] for item in sorted(candidate_ids)])
            rho = float(stats.spearmanr(predicted_values, physical_values).statistic)
            selected = decode_lookup[selected_id]
            node_correct = selected["node"] == target.node
            exact = selected_id == target.action_id
            results.append({
                "experiment": experiment,
                "evaluation_state": state_name,
                "model": model_name,
                "generating_action_available": not leave_out,
                "target_action_id": target.action_id,
                "target_environment": target.environment,
                "target_node": target.node,
                "target_strength": target.strength,
                "n_candidates_including_identity": len(candidate_ids),
                "state_variables": ";".join(state),
                "selected_predicted_kl": predicted[selected_id],
                "mean_only_selected_action_id": mean_only_selected,
                "mean_only_selection_agrees": mean_only_selected == selected_id,
                **decode(selected_id, decode_lookup, "selected"),
                "node_selection_correct": node_correct if not leave_out else None,
                "strength_correct_conditional_on_node": exact if (not leave_out and node_correct) else (False if not leave_out else None),
                "joint_node_strength_top1": exact if not leave_out else None,
                **metrics,
                **decode(oracle_id, decode_lookup, "oracle"),
                "spearman_rank_correlation": rho,
            })
            for action_id in candidate_ids:
                rankings.append({
                    "experiment": experiment,
                    "evaluation_state": state_name,
                    "model": model_name,
                    "target_action_id": target.action_id,
                    "target_environment": target.environment,
                    "candidate_action_id": action_id,
                    "candidate_environment": decode_lookup[action_id]["environment"],
                    "candidate_node": decode_lookup[action_id]["node"],
                    "candidate_strength": decode_lookup[action_id]["strength"],
                    "predicted_kl": predicted[action_id],
                    "predicted_mean_component": mean_only[action_id],
                    "predicted_rank": predicted_order.index(action_id) + 1,
                    "physical_kl": observed[action_id]["kl"],
                    "physical_mean_kl_component": observed[action_id]["mean_kl"],
                    "physical_covariance_kl_component": observed[action_id]["covariance_kl"],
                    "physical_rank": physical_order.index(action_id) + 1,
                    "physical_kl_ci_low": observed[action_id]["bootstrap_kl_ci_low"],
                    "physical_kl_ci_high": observed[action_id]["bootstrap_kl_ci_high"],
                })
    return pd.DataFrame(results), pd.DataFrame(rankings)


def relative_covariance_frobenius(source_cov: np.ndarray, action_cov: np.ndarray) -> float:
    return float(np.linalg.norm(action_cov - source_cov, ord="fro") / np.linalg.norm(source_cov, ord="fro"))


def invariance_diagnostics(
    catalog: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    source_center: np.ndarray,
    source_scale: np.ndarray,
) -> pd.DataFrame:
    cached_path = REPO / "outputs" / "mk1_heteroskedastic_e2e" / "machine_readable" / "shift_environment_classification.json"
    cached = {row["environment"]: row for row in json.loads(cached_path.read_text(encoding="utf-8"))}
    full_reference = standardize(
        frames["A_ID"].iloc[IDENTITY_EVALUATION.start:IDENTITY_EVALUATION.stop].loc[:, COMMON_STATE].to_numpy(float),
        source_center,
        source_scale,
    )
    desc_indices = np.array([COMMON_STATE.index(variable) for variable in DESCENDANT_STATE])
    full_source_cov = regular_cov(full_reference)
    desc_source_cov = regular_cov(full_reference[:, desc_indices])
    rows = []
    for action in catalog.itertuples(index=False):
        full_action = standardize(
            frames[action.action_id].iloc[CANDIDATE_EVALUATION.start:CANDIDATE_EVALUATION.stop].loc[:, COMMON_STATE].to_numpy(float),
            source_center,
            source_scale,
        )
        action_cov = regular_cov(full_action)
        desc_action_cov = regular_cov(full_action[:, desc_indices])
        root_index = COMMON_STATE.index(action.node)
        root_ratio = float(np.var(full_action[:, root_index], ddof=1) / np.var(full_reference[:, root_index], ddof=1))
        full_discrepancy = relative_covariance_frobenius(full_source_cov, action_cov)
        desc_discrepancy = relative_covariance_frobenius(desc_source_cov, desc_action_cov)
        _, _, full_covariance_kl = gaussian_kl_components(
            np.zeros(len(COMMON_STATE)), action_cov, np.zeros(len(COMMON_STATE)), full_source_cov
        )
        _, _, desc_covariance_kl = gaussian_kl_components(
            np.zeros(len(DESCENDANT_STATE)), desc_action_cov, np.zeros(len(DESCENDANT_STATE)), desc_source_cov
        )
        root_pass = INVARIANCE_TOLERANCE["root_variance_ratio_low"] <= root_ratio <= INVARIANCE_TOLERANCE["root_variance_ratio_high"]
        full_pass = full_discrepancy <= INVARIANCE_TOLERANCE["full_state_relative_covariance_frobenius_max"]
        desc_pass = desc_discrepancy <= INVARIANCE_TOLERANCE["descendant_relative_covariance_frobenius_max"]
        cached_row = cached[action.environment]
        rows.append({
            "action_id": action.action_id,
            "environment": action.environment,
            "node": action.node,
            "strength": action.strength,
            "cached_classification": cached_row["classification"],
            "cached_full_file_root_std_ratio": cached_row["std_ratio"],
            "cached_full_file_root_variance_ratio": cached_row["std_ratio"] ** 2,
            "heldout_root_variance_ratio": root_ratio,
            "root_variance_tolerance_pass": root_pass,
            "full_state_relative_covariance_frobenius": full_discrepancy,
            "full_state_covariance_tolerance_pass": full_pass,
            "full_state_covariance_only_gaussian_kl": full_covariance_kl,
            "descendant_relative_covariance_frobenius": desc_discrepancy,
            "descendant_covariance_tolerance_pass": desc_pass,
            "descendant_covariance_only_gaussian_kl": desc_covariance_kl,
            "joint_invariance_tolerance_pass": root_pass and full_pass and desc_pass,
            "source_rows": f"[{IDENTITY_EVALUATION.start}:{IDENTITY_EVALUATION.stop}]",
            "action_rows": f"[{CANDIDATE_EVALUATION.start}:{CANDIDATE_EVALUATION.stop}]",
        })
    return pd.DataFrame(rows)


def state_diagnostics(
    catalog: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    models: dict[str, dict[str, GaussianModel]],
    source_center: np.ndarray,
    source_scale: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for state_name, state in STATE_DEFINITIONS.items():
        indices = np.array([COMMON_STATE.index(variable) for variable in state])
        center, scale = source_center[indices], source_scale[indices]
        for model_name, entries in models.items():
            for action_id, model in entries.items():
                covariance = transform_model(subset_model(model, indices), center, scale).covariance
                eigenvalues = np.linalg.eigvalsh(covariance)
                rows.append({
                    "diagnostic_type": "predicted_model",
                    "evaluation_state": state_name,
                    "model": model_name,
                    "action_id": action_id,
                    "dimension": len(state),
                    "minimum_eigenvalue": float(eigenvalues.min()),
                    "condition_number": float(eigenvalues.max() / eigenvalues.min()),
                    "stable": bool(eigenvalues.min() > 0 and eigenvalues.max() / eigenvalues.min() < 1e8),
                })
        candidate_ids = ["A_ID", *catalog["action_id"].tolist()]
        for action_id in candidate_ids:
            block = IDENTITY_EVALUATION if action_id == "A_ID" else CANDIDATE_EVALUATION
            values = standardize(frames[action_id].iloc[block.start:block.stop].loc[:, state].to_numpy(float), center, scale)
            covariance = regular_cov(values)
            eigenvalues = np.linalg.eigvalsh(covariance)
            rows.append({
                "diagnostic_type": "heldout_physical",
                "evaluation_state": state_name,
                "model": "physical",
                "action_id": action_id,
                "dimension": len(state),
                "minimum_eigenvalue": float(eigenvalues.min()),
                "condition_number": float(eigenvalues.max() / eigenvalues.min()),
                "stable": bool(eigenvalues.min() > 0 and eigenvalues.max() / eigenvalues.min() < 1e8),
            })
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(["experiment", "evaluation_state", "model"], sort=True):
        experiment, state, model = keys
        row: dict[str, Any] = {
            "experiment": experiment,
            "evaluation_state": state,
            "model": model,
            "n_targets": len(part),
            "mean_selected_kl": float(part["selected_physical_kl"].mean()),
            "median_selected_kl": float(part["selected_physical_kl"].median()),
            "median_identity_kl": float(part["identity_kl"].median()),
            "mean_identity_improvement": float(part["delta_identity"].mean()),
            "improved_over_identity_fraction": float((part["delta_identity"] > 0).mean()),
            "improvement_ci_positive_fraction": float((part["delta_identity_ci_low"] > 0).mean()),
            "median_oracle_kl": float(part["oracle_kl"].median()),
            "mean_regret": float(part["regret"].mean()),
            "median_regret": float(part["regret"].median()),
            "maximum_regret": float(part["regret"].max()),
            "oracle_top1_fraction": float(part["top1_oracle_agreement"].mean()),
            "physical_top3_fraction": float(part["selected_in_physical_top3"].mean()),
            "selected_in_uncertainty_top_set_fraction": float(part["selected_in_physical_top_set"].mean()),
            "median_spearman": float(part["spearman_rank_correlation"].median()),
            "mean_only_selection_agreement_fraction": float(part["mean_only_selection_agrees"].mean()),
        }
        if part["joint_node_strength_top1"].notna().any():
            eligible = part[part["joint_node_strength_top1"].notna()]
            row.update({
                "node_accuracy": float(eligible["node_selection_correct"].mean()),
                "strength_accuracy_conditional_on_correct_node": float(
                    eligible.loc[eligible["node_selection_correct"], "strength_correct_conditional_on_node"].mean()
                ) if eligible["node_selection_correct"].any() else None,
                "joint_node_strength_accuracy": float(eligible["joint_node_strength_top1"].mean()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def model_comparison(results: pd.DataFrame) -> pd.DataFrame:
    key = ["experiment", "evaluation_state", "target_action_id", "target_environment"]
    columns = [
        "selected_action_id", "selected_predicted_kl", "selected_physical_kl",
        "regret", "spearman_rank_correlation",
    ]
    wide = results.pivot(index=key, columns="model", values=columns)
    rows = []
    for index, row in wide.iterrows():
        item = dict(zip(key, index))
        for model_name in ("shift_model", "empirical_finite_mode", "shift_mean_action_covariance"):
            for column in columns:
                item[f"{model_name}_{column}"] = row[(column, model_name)]
        item["shift_vs_empirical_selected_action_agreement"] = (
            item["shift_model_selected_action_id"] == item["empirical_finite_mode_selected_action_id"]
        )
        item["shift_minus_empirical_selected_physical_kl"] = (
            item["shift_model_selected_physical_kl"] - item["empirical_finite_mode_selected_physical_kl"]
        )
        item["source_covariance_vs_action_covariance_selection_agreement"] = (
            item["shift_model_selected_action_id"] == item["shift_mean_action_covariance_selected_action_id"]
        )
        item["source_covariance_enforcement_physical_kl_difference"] = (
            item["shift_model_selected_physical_kl"] - item["shift_mean_action_covariance_selected_physical_kl"]
        )
        item["source_covariance_enforcement_selected_predicted_kl_difference"] = (
            item["shift_model_selected_predicted_kl"] - item["shift_mean_action_covariance_selected_predicted_kl"]
        )
        rows.append(item)
    return pd.DataFrame(rows)


def main(raw_root: Path, n_bootstrap: int = N_BOOTSTRAP) -> None:
    validate_protocol()
    catalog = build_catalog(raw_root)
    schemas = validate_schemas(raw_root, [REFERENCE, *catalog["environment"]])
    frames = {"A_ID": load(raw_root, REFERENCE)}
    for action in catalog.itertuples(index=False):
        frames[action.action_id] = load(raw_root, action.environment)
    source_frame = frames["A_ID"].iloc[SOURCE_FIT.start:SOURCE_FIT.stop]
    source_center = source_frame.loc[:, COMMON_STATE].mean().to_numpy(float)
    source_scale = source_frame.loc[:, COMMON_STATE].std(ddof=1).to_numpy(float)
    if np.any(source_scale <= 0):
        raise ValueError("zero source scale in common state")
    source_model, _, effects = fit_source_scm(source_frame)
    models, action_parameters = candidate_models(catalog, frames, source_model, effects)
    diagnostics = state_diagnostics(catalog, frames, models, source_center, source_scale)
    if not diagnostics["stable"].all():
        raise ValueError("predeclared state failed numerical stability gate")
    invariance = invariance_diagnostics(catalog, frames, source_center, source_scale)
    decode_lookup = {
        "A_ID": {"environment": REFERENCE, "node": "identity", "strength": "identity"},
        **{
            action.action_id: {"environment": action.environment, "node": action.node, "strength": action.strength}
            for action in catalog.itertuples(index=False)
        },
    }
    rng = np.random.default_rng(SEED)
    result_frames = []
    ranking_frames = []
    for state_name, state in STATE_DEFINITIONS.items():
        for experiment, leave_out in (("S1_realizable", False), ("S2_leave_true_action_out", True)):
            result, ranking = run_matching(
                experiment, leave_out, state_name, state, catalog, frames, models, decode_lookup,
                source_center, source_scale, rng, n_bootstrap,
            )
            result_frames.append(result)
            ranking_frames.append(ranking)
    results = pd.concat(result_frames, ignore_index=True)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    aggregates = aggregate(results)
    comparisons = model_comparison(results)
    bootstrap_columns = [
        "experiment", "evaluation_state", "model", "target_action_id", "target_environment",
        "selected_action_id", "selected_environment", "oracle_action_id", "oracle_environment",
        "selected_physical_kl", "selected_physical_kl_ci_low", "selected_physical_kl_ci_high",
        "identity_kl", "delta_identity", "delta_identity_ci_low", "delta_identity_ci_high",
        "oracle_kl", "regret", "regret_ci_low", "regret_ci_high",
        "selected_vs_fixed_oracle_ci_low", "selected_vs_fixed_oracle_ci_high",
        "physical_top_set_ids", "physical_top_set_size", "selected_in_physical_top_set",
    ]
    bootstrap_summary = results.loc[:, bootstrap_columns].copy()

    write_frame("action_catalog", catalog)
    write_frame("action_shift_parameters", action_parameters)
    write_frame("schema_compatibility", schemas)
    write_frame("state_numerical_diagnostics", diagnostics)
    write_frame("variance_covariance_invariance", invariance)
    write_frame("s1_full_state_per_target", results[(results.experiment == "S1_realizable") & (results.evaluation_state == "full_state")])
    write_frame("s2_full_state_per_target", results[(results.experiment == "S2_leave_true_action_out") & (results.evaluation_state == "full_state")])
    write_frame("s1_descendant_only_per_target", results[(results.experiment == "S1_realizable") & (results.evaluation_state == "descendant_only")])
    write_frame("s2_descendant_only_per_target", results[(results.experiment == "S2_leave_true_action_out") & (results.evaluation_state == "descendant_only")])
    write_frame("all_candidate_rankings", rankings)
    write_frame("shift_vs_empirical_comparison", comparisons)
    write_frame("bootstrap_summaries", bootstrap_summary)
    write_frame("aggregate_results", aggregates)

    protocol = {
        "source_reference_calibration": {"environment": REFERENCE, "rows": [SOURCE_FIT.start, SOURCE_FIT.stop]},
        "action_calibration": {"environment": "each root action", "rows": [ACTION_FIT.start, ACTION_FIT.stop]},
        "target_selection": {"environment": "hidden target", "rows": [TARGET_SELECTION.start, TARGET_SELECTION.stop]},
        "candidate_physical_evaluation": {"environment": "each eligible candidate", "rows": [CANDIDATE_EVALUATION.start, CANDIDATE_EVALUATION.stop]},
        "target_physical_evaluation": {"environment": "hidden target", "rows": [TARGET_EVALUATION.start, TARGET_EVALUATION.stop]},
        "identity_physical_evaluation": {"environment": REFERENCE, "rows": [IDENTITY_EVALUATION.start, IDENTITY_EVALUATION.stop]},
        "bootstrap": {"replicates": n_bootstrap, "block_rows": BLOCK_SIZE, "method": "independent resampling of four non-overlapping contiguous 50-row target and candidate blocks", "scope": "within-session sampling variation only"},
        "selector_signature": str(inspect.signature(select_action)),
        "opaque_ids_used_for_selection": True,
        "target_labels_available_to_selector": False,
    }
    write_json(OUTPUT / "machine_readable" / "frozen_protocol.json", protocol)
    state_gate = {
        "gate": "PASS",
        "primary_state": list(COMMON_STATE),
        "primary_dimension": len(COMMON_STATE),
        "primary_definition": "predeclared 17-variable causally closed union reused from reduced gain matching",
        "diagnostic_state": list(DESCENDANT_STATE),
        "diagnostic_dimension": len(DESCENDANT_STATE),
        "diagnostic_definition": "symmetric removal of all five directly manipulated actuator coordinates",
        "target_specific": False,
        "all_schemas_identical": bool(schemas["exact_reference_schema"].all()),
        "all_models_and_physical_covariances_stable": bool(diagnostics["stable"].all()),
        "maximum_condition_number": float(diagnostics["condition_number"].max()),
        "minimum_covariance_eigenvalue": float(diagnostics["minimum_eigenvalue"].min()),
        "numerical_stop_boundary": 1e8,
    }
    write_json(OUTPUT / "machine_readable" / "state_schema_gate.json", state_gate)
    invariance_gate = {
        "gate": "PASS" if bool(invariance["joint_invariance_tolerance_pass"].all()) else "FAIL",
        "tolerance_frozen_before_matching": INVARIANCE_TOLERANCE,
        "actions_passing": int(invariance["joint_invariance_tolerance_pass"].sum()),
        "actions_total": len(invariance),
        "interpretation_if_fail": "retain original constrained shift-model results, but do not claim exact original shift-family validation",
        "cached_diagnostics_reused": [
            "outputs/mk1_heteroskedastic_e2e/machine_readable/shift_environment_classification.json",
            "outputs/mk1_heteroskedastic_e2e/machine_readable/shift_downstream_invariance.json",
        ],
    }
    write_json(OUTPUT / "machine_readable" / "covariance_invariance_gate.json", invariance_gate)
    summary = {
        "scientific_question": "source-calibrated B=1 shift-only root and strength selection",
        "primary_model": "source SCM total-effect mean shift with action-calibrated magnitude and invariant source covariance",
        "secondary_model": "empirical finite-mode Gaussian fitted on the same action calibration block",
        "invariance_gate": invariance_gate,
        "aggregate_results": aggregates.to_dict(orient="records"),
        "claim_exclusions": ["independent-session replication", "non-root shift acquisition", "causal discovery", "B>1 composition"],
    }
    write_json(OUTPUT / "machine_readable" / "aggregate_summary.json", summary)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unavailable"
    metadata = {
        "seed": SEED,
        "dataset": DATASET,
        "reference_sha256": sha256(raw_root / DATASET / f"{REFERENCE}.csv"),
        "repository_commit_before_run": commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "prior_artifacts_reused": [
            "docs/lt_mk1_root_shift_response_deep_dive.md",
            "outputs/mk1_heteroskedastic_e2e/machine_readable/shift_environment_classification.json",
            "outputs/mk1_heteroskedastic_e2e/machine_readable/shift_downstream_invariance.json",
            "outputs/mk1_reduced_real_matching/machine_readable/frozen_protocol.json",
        ],
        "prohibited_prior_experiments_rerun": [],
    }
    write_json(OUTPUT / "machine_readable" / "run_metadata.json", metadata)
    completion = {
        "status": "complete",
        "bootstrap_replicates": n_bootstrap,
        "expected_targets_per_experiment_state_model": 10,
        "result_rows": len(results),
        "ranking_rows": len(rankings),
        "invariance_gate": invariance_gate["gate"],
    }
    write_json(OUTPUT / "machine_readable" / "completion.json", completion)
    print(json.dumps(completion, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    arguments = parser.parse_args()
    main(arguments.raw_root, arguments.bootstrap)
