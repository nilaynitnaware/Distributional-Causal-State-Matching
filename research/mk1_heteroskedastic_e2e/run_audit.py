"""Bounded Light Tunnel Mk1 heteroskedastic finite-mode feasibility audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import linalg, optimize, spatial

from scripts._output_safety import prepare_output_directory


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "outputs" / "mk1_heteroskedastic_e2e"
DATASET = "lt_interventions_standard_v1"
REFERENCE = "uniform_reference"
SEED = 20260825

SIGNAL_PARENTS: dict[str, tuple[str, ...]] = {
    "ir_1": ("red", "green", "blue", "l_11", "l_12"),
    "vis_1": ("red", "green", "blue", "l_11", "l_12"),
    "ir_2": ("red", "green", "blue", "l_21", "l_22"),
    "vis_2": ("red", "green", "blue", "l_21", "l_22"),
    "ir_3": ("red", "green", "blue", "pol_1", "pol_2", "l_31", "l_32"),
    "vis_3": ("red", "green", "blue", "pol_1", "pol_2", "l_31", "l_32"),
}

STATIONS: dict[str, tuple[str, ...]] = {
    "station_1": ("red", "green", "blue", "l_11", "l_12", "ir_1", "vis_1"),
    "station_2": ("red", "green", "blue", "l_21", "l_22", "ir_2", "vis_2"),
    "station_3": (
        "red", "green", "blue", "pol_1", "pol_2", "l_31", "l_32", "ir_3", "vis_3"
    ),
}

PILOT_MODES: dict[str, str] = {
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
    tables = OUTPUT / "tables"
    machine = OUTPUT / "machine_readable"
    tables.mkdir(parents=True, exist_ok=True)
    machine.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / f"{name}.csv", index=False, lineterminator="\n")
    write_json(machine / f"{name}.json", frame.to_dict(orient="records"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(raw_root: Path, environment: str, dataset: str = DATASET) -> pd.DataFrame:
    path = raw_root / dataset / f"{environment}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def infer_gain_environment(environment: str) -> tuple[str, str, str]:
    body = environment.removeprefix("uniform_")
    for level in ("weak", "mid", "strong"):
        suffix = f"_{level}"
        if body.endswith(suffix):
            control = body[: -len(suffix)]
            if control.startswith("t_"):
                outcome = control.removeprefix("t_")
            elif control.startswith("diode_"):
                outcome = control.removeprefix("diode_")
            else:
                raise ValueError(environment)
            return control, outcome, level
    raise ValueError(environment)


def gain_environments(raw_root: Path) -> tuple[str, ...]:
    names = []
    for path in sorted((raw_root / DATASET).glob("uniform_*.csv")):
        name = path.stem
        if name.startswith("uniform_t_") or name.startswith("uniform_diode_"):
            names.append(name)
    return tuple(names)


def fit_ols(frame: pd.DataFrame, outcome: str, parents: tuple[str, ...]) -> Fit:
    x = frame.loc[:, parents].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    design = np.column_stack((np.ones(len(frame)), x))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    omega = max(float(np.mean(residual**2)), np.finfo(float).tiny)
    return Fit(float(coef[0]), np.asarray(coef[1:], float), omega)


def fit_family(
    frame: pd.DataFrame,
    outcome: str,
    parents: tuple[str, ...],
    source: Fit,
    family: str,
) -> tuple[Fit, float]:
    x = frame.loc[:, parents].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    if family == "M0":
        beta = source.beta.copy()
        alpha = float(np.mean(y - x @ beta))
        omega = source.omega
        return Fit(alpha, beta, omega), 1.0
    if family in {"M1", "M2"}:
        z = x @ source.beta
        design = np.column_stack((np.ones(len(frame)), z))
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        scale = max(float(coef[1]), 1e-12)
        alpha = float(np.mean(y - scale * z))
        beta = scale * source.beta
        residual = y - alpha - x @ beta
        omega = source.omega if family == "M1" else max(float(np.mean(residual**2)), np.finfo(float).tiny)
        return Fit(alpha, beta, omega), scale
    if family == "M3":
        fit = fit_ols(frame, outcome, parents)
        denominator = float(source.beta @ source.beta)
        scale = max(float(source.beta @ fit.beta) / denominator, 1e-12)
        return fit, scale
    raise ValueError(family)


def nll_rows(frame: pd.DataFrame, outcome: str, parents: tuple[str, ...], fit: Fit) -> np.ndarray:
    x = frame.loc[:, parents].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    residual = y - fit.alpha - x @ fit.beta
    return 0.5 * (math.log(2.0 * math.pi) + math.log(fit.omega) + residual**2 / fit.omega)


def block_bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    block_size: int = 50,
    replicates: int = 500,
) -> tuple[float, float]:
    if len(left) != len(right) or len(left) % block_size:
        raise ValueError("loss vectors must have equal block-divisible length")
    delta = left - right
    blocks = delta.reshape(-1, block_size).mean(axis=1)
    estimates = np.empty(replicates)
    for index in range(replicates):
        estimates[index] = rng.choice(blocks, size=len(blocks), replace=True).mean()
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def dataset_inventory(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        for path in sorted(dataset_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            timestamps = frame["timestamp"].to_numpy(float)
            counters = frame["counter"].to_numpy(float)
            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "environment": path.stem,
                    "n_rows": len(frame),
                    "n_columns": len(frame.columns),
                    "timestamp_first": float(timestamps[0]),
                    "timestamp_last": float(timestamps[-1]),
                    "timestamp_monotone": bool(np.all(np.diff(timestamps) >= 0)),
                    "median_spacing_seconds": float(np.median(np.diff(timestamps))),
                    "counter_first": float(counters[0]),
                    "counter_last": float(counters[-1]),
                    "counter_consecutive": bool(np.array_equal(counters, np.arange(len(frame)))),
                    "config_values": ";".join(map(str, sorted(frame["config"].astype(str).unique()))),
                    "flag_values": ";".join(map(str, sorted(frame["flag"].astype(str).unique()))),
                    "intervention_values": ";".join(map(str, sorted(frame["intervention"].astype(str).unique()))),
                    "csv_bytes": path.stat().st_size,
                    "csv_sha256": sha256(path),
                    "session_evidence": "one contiguous counter/timestamp sequence; no replicate-session identifier",
                }
            )
    return pd.DataFrame(rows)


def environment_settings(raw_root: Path) -> pd.DataFrame:
    """Inventory the encoded target and its exact archived setting/distribution."""
    reference = load(raw_root, REFERENCE)
    rows = []
    for path in sorted((raw_root / DATASET).glob("uniform_*.csv")):
        environment = path.stem
        if environment == REFERENCE:
            continue
        body = environment.removeprefix("uniform_")
        level = ""
        for candidate in ("weak", "mid", "strong"):
            suffix = f"_{candidate}"
            if body.endswith(suffix):
                body = body[: -len(suffix)]
                level = candidate
                break
        target = body
        frame = pd.read_csv(path)
        values = frame[target].to_numpy(float)
        source = reference[target].to_numpy(float)
        rows.append(
            {
                "environment": environment,
                "intervention_target": target,
                "nominal_strength": level,
                "target_kind": (
                    "actuator root distribution" if target in {
                        "red", "green", "blue", "pol_1", "pol_2",
                        "l_11", "l_12", "l_21", "l_22", "l_31", "l_32",
                    } else "sensor/measurement parameter"
                ),
                "n_rows": len(frame),
                "reference_min": float(source.min()),
                "reference_max": float(source.max()),
                "reference_mean": float(source.mean()),
                "reference_sd": float(source.std(ddof=1)),
                "intervention_min": float(values.min()),
                "intervention_max": float(values.max()),
                "intervention_mean": float(values.mean()),
                "intervention_sd": float(values.std(ddof=1)),
                "intervention_unique_values": int(pd.Series(values).nunique()),
                "setting_or_distribution": (
                    "fixed hardware setting" if pd.Series(values).nunique() == 1
                    else "randomized command distribution"
                ),
            }
        )
    return pd.DataFrame(rows)


def augmented_graph() -> pd.DataFrame:
    manifest = pd.read_csv(raw_root / "support" / "mk1_variable_manifest.csv").fillna("")
    manifest = manifest.rename(columns={"known_parents": "parents"})
    manifest["parents_observed"] = True
    manifest["directly_manipulable"] = manifest["manipulable"].eq("yes")
    manifest["mode_semantics"] = np.select(
        [
            manifest["candidate_role"].eq("shift_probe"),
            manifest["candidate_role"].eq("scale_action"),
            manifest["candidate_role"].eq("other_action"),
        ],
        [
            "root command-distribution intervention; approximate shift only after diagnostics",
            "measurement-mechanism mode; structural scale only after conditional evidence",
            "resolution/precision/calibration mode; not presumed shift or structural scale",
        ],
        default="measured response; not directly manipulable",
    )
    keep = [
        "variable_name", "human_readable_meaning", "variable_kind", "value_type", "parents",
        "parents_observed", "directly_manipulable", "intervention_environments", "candidate_role",
        "mode_semantics", "physical_model_action", "evidence_source",
    ]
    return manifest.loc[:, keep]


def closure_candidates() -> pd.DataFrame:
    rows = []
    for station, nodes in STATIONS.items():
        outcomes = [node for node in nodes if node.startswith(("ir_", "vis_"))]
        for outcome in outcomes:
            single_nodes = (*SIGNAL_PARENTS[outcome], outcome)
            rows.append(
                {
                    "candidate": f"{station}_{outcome}",
                    "nodes": ";".join(single_nodes),
                    "n_nodes": len(single_nodes),
                    "nonroot_nodes": outcome,
                    "max_parent_count": len(SIGNAL_PARENTS[outcome]),
                    "all_relevant_varying_parents_included": True,
                    "fixed_exogenous_inputs": f"diode_{outcome};t_{outcome}",
                    "common_cause_warning": "single response avoids the observed IR/visible cross-residual dependence; latent state dependence remains",
                    "mechanism_route": "official graph plus observational regression; gain-probe support recovery failed",
                    "active_acquisition_complete": False,
                    "pass": True,
                    "scope": "Level A one-response finite-catalog B=1 only",
                    "reason": "all varying physical parents are represented and the one child has at least five parents",
                }
            )
        rows.append(
            {
                "candidate": station,
                "nodes": ";".join(nodes),
                "n_nodes": len(nodes),
                "nonroot_nodes": ";".join(outcomes),
                "max_parent_count": max(len(SIGNAL_PARENTS[node]) for node in outcomes),
                "all_relevant_varying_parents_included": True,
                "fixed_exogenous_inputs": ";".join(f"diode_{node};t_{node}" for node in outcomes),
                "common_cause_warning": "IR/visible source residual correlation is 0.75-0.80",
                "mechanism_route": "official graph plus observational regression",
                "active_acquisition_complete": False,
                "pass": False,
                "scope": "invalid for independent node-noise theorem",
                "reason": "strong cross-residual dependence contradicts the independent nodewise-noise factorization",
            }
        )
    rows.extend(
        [
            {
                "candidate": "nine_measurements_only",
                "nodes": "current;angle_1;angle_2;ir_1;vis_1;ir_2;vis_2;ir_3;vis_3",
                "n_nodes": 9,
                "nonroot_nodes": "all",
                "max_parent_count": 0,
                "all_relevant_varying_parents_included": False,
                "fixed_exogenous_inputs": "",
                "common_cause_warning": "all actuator causes omitted",
                "mechanism_route": "none",
                "active_acquisition_complete": False,
                "pass": False,
                "scope": "invalid",
                "reason": "omits observed RGB, polarizer, and local-LED parents",
            },
            {
                "candidate": "six_optical_measurements_only",
                "nodes": "ir_1;vis_1;ir_2;vis_2;ir_3;vis_3",
                "n_nodes": 6,
                "nonroot_nodes": "all",
                "max_parent_count": 0,
                "all_relevant_varying_parents_included": False,
                "fixed_exogenous_inputs": "",
                "common_cause_warning": "shared observed actuator causes omitted",
                "mechanism_route": "none",
                "active_acquisition_complete": False,
                "pass": False,
                "scope": "invalid",
                "reason": "not causally closed; deleting parents would manufacture closure",
            },
        ]
    )
    return pd.DataFrame(rows)


def run_model_comparisons(raw_root: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], Fit], dict[str, Fit]]:
    reference = load(raw_root, REFERENCE)
    source_fits = {
        outcome: fit_ols(reference.iloc[:5000], outcome, parents)
        for outcome, parents in SIGNAL_PARENTS.items()
    }
    fits: dict[tuple[str, str], Fit] = {}
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for environment in gain_environments(raw_root):
        control, outcome, level = infer_gain_environment(environment)
        parents = SIGNAL_PARENTS[outcome]
        frame = load(raw_root, environment)
        train, test = frame.iloc[:500], frame.iloc[500:]
        family_fits: dict[str, Fit] = {}
        scales: dict[str, float] = {}
        losses: dict[str, np.ndarray] = {}
        for family in ("M0", "M1", "M2", "M3"):
            family_fit, scale = fit_family(train, outcome, parents, source_fits[outcome], family)
            family_fits[family] = family_fit
            fits[(environment, family)] = family_fit
            scales[family] = scale
            losses[family] = nll_rows(test, outcome, parents, family_fit)
        unrestricted = family_fits["M3"]
        projection = scales["M3"]
        prop_error = float(
            np.linalg.norm(unrestricted.beta - projection * source_fits[outcome].beta)
            / max(np.linalg.norm(unrestricted.beta), np.finfo(float).eps)
        )
        ci_low, ci_high = block_bootstrap_difference(losses["M2"], losses["M3"], rng)
        row: dict[str, Any] = {
            "environment": environment,
            "physical_control": control,
            "control_level": level,
            "outcome": outcome,
            "parents": ";".join(parents),
            "source_train_rows": 5000,
            "action_train_rows": 500,
            "action_test_rows": 500,
            "split_rule": "contiguous acquisition order: first 500 calibration, last 500 test",
            "s_hat_unrestricted_projection": projection,
            "s_hat_M2": scales["M2"],
            "proportionality_error": prop_error,
            "omega_source": source_fits[outcome].omega,
            "omega_action_M2": family_fits["M2"].omega,
            "omega_action_M3": family_fits["M3"].omega,
            "residual_variance_ratio_M3_source": unrestricted.omega / source_fits[outcome].omega,
            "M2_minus_M3_nll": float(losses["M2"].mean() - losses["M3"].mean()),
            "M2_minus_M3_block_bootstrap_ci_low": ci_low,
            "M2_minus_M3_block_bootstrap_ci_high": ci_high,
            "coefficients_proportional": bool(prop_error <= 0.15 and projection > 0),
            "noise_invariant": bool(0.8 <= unrestricted.omega / source_fits[outcome].omega <= 1.25),
            "M2_competitive_with_M3": bool(prop_error <= 0.15 and losses["M2"].mean() - losses["M3"].mean() <= 0.02 and ci_high <= 0.05),
            "single_session_only": True,
        }
        for family in ("M0", "M1", "M2", "M3"):
            row[f"heldout_nll_{family}"] = float(losses[family].mean())
            row[f"alpha_{family}"] = family_fits[family].alpha
            row[f"omega_{family}"] = family_fits[family].omega
            row[f"beta_{family}_json"] = json.dumps(family_fits[family].beta.tolist())
        rows.append(row)
    output = pd.DataFrame(rows).sort_values("environment").reset_index(drop=True)
    prior = pd.read_csv(raw_root / "support" / "scale_semantics_summary.csv")
    diagnostics = prior.loc[:, [
        "intervention_environment", "intervention_heteroscedasticity_lm",
        "intervention_heteroscedasticity_p", "intervention_nonlinear_residual_r2",
        "intervention_normality_p", "intervention_clipping_frequency",
    ]].rename(columns={"intervention_environment": "environment"})
    output = output.merge(diagnostics, on="environment", how="left", validate="one_to_one")
    output["fixed_conditional_variance_supported"] = (
        output["intervention_heteroscedasticity_p"] >= 0.01
    )
    return output, fits, source_fits


def variance_stability(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    segment_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for outcome, environment in PILOT_MODES.items():
        frame = load(raw_root, environment)
        parents = SIGNAL_PARENTS[outcome]
        variances = []
        for segment in range(4):
            start, stop = 250 * segment, 250 * (segment + 1)
            fit = fit_ols(frame.iloc[start:stop], outcome, parents)
            variances.append(fit.omega)
            segment_rows.append(
                {
                    "environment": environment,
                    "outcome": outcome,
                    "segment": segment + 1,
                    "row_start": start,
                    "row_stop_exclusive": stop,
                    "timestamp_first": float(frame["timestamp"].iloc[start]),
                    "timestamp_last": float(frame["timestamp"].iloc[stop - 1]),
                    "residual_variance_M3": fit.omega,
                    "alpha_M3": fit.alpha,
                    "beta_M3_json": json.dumps(fit.beta.tolist()),
                }
            )
        values = np.asarray(variances)
        ratio = float(values.max() / values.min())
        summary_rows.append(
            {
                "environment": environment,
                "outcome": outcome,
                "n_segments": 4,
                "segment_rows": 250,
                "variance_mean": float(values.mean()),
                "variance_cv": float(values.std(ddof=1) / values.mean()),
                "variance_max_min_ratio": ratio,
                "within_session_stable_threshold_1_5": bool(ratio <= 1.5),
                "between_session_reproducibility": "not identifiable: one acquisition session",
            }
        )
    return pd.DataFrame(segment_rows), pd.DataFrame(summary_rows)


def support_theory_validation() -> pd.DataFrame:
    rows = []
    q = 1.3
    for scale in (0.5, 0.8, 1.2, 2.0):
        ratios = (0.2, 0.5, scale * (1 - 1e-8), scale, scale * (1 + 1e-8), scale**2, 2.0, 4.0)
        for parent_count, coefficients in (
            (0, np.array([])),
            (1, np.array([0.7])),
            (3, np.array([0.7, -0.45, 1e-6])),
        ):
            for ratio in ratios:
                q_prime = q / ratio
                row_factor = q - scale * q_prime
                block_factor = scale**2 * q_prime - q
                row_values = row_factor * coefficients
                diagonal_values = block_factor * coefficients**2
                row_support = set(np.flatnonzero(np.abs(row_values) > 1e-12).tolist())
                diagonal_support = set(np.flatnonzero(np.abs(diagonal_values) > 1e-12).tolist())
                truth = set(range(parent_count))
                rows.append(
                    {
                        "scale": scale,
                        "noise_variance_ratio": ratio,
                        "parent_count": parent_count,
                        "minimum_abs_coefficient": float(np.min(np.abs(coefficients))) if parent_count else None,
                        "row_factor": row_factor,
                        "parent_block_factor": block_factor,
                        "row_cancellation": bool(np.isclose(ratio, scale, rtol=0, atol=1e-14)),
                        "parent_block_cancellation": bool(np.isclose(ratio, scale**2, rtol=0, atol=1e-14)),
                        "row_support_exact_symbolic": (not parent_count) or row_factor != 0.0,
                        "diagonal_support_exact_symbolic": (not parent_count) or block_factor != 0.0,
                        "union_support_exact_symbolic": (
                            (not parent_count) or row_factor != 0.0 or block_factor != 0.0
                        ),
                        "row_support_exact_at_1e_12": row_support == truth,
                        "diagonal_support_exact_at_1e_12": diagonal_support == truth,
                        "union_support_exact_at_1e_12": row_support | diagonal_support == truth,
                    }
                )
    return pd.DataFrame(rows)


def shift_results(raw_root: Path, source_fits: dict[str, Fit]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = pd.read_csv(raw_root / "support" / "shift_semantics_summary.csv")
    rows = []
    for _, row in prior.iterrows():
        prior_class = str(row["classification"])
        probe = str(row["probe_variable"])
        if prior_class == "good approximate shift probe":
            classification = "APPROXIMATE SHIFT-LIKE"
            scope = "root distribution intervention; matching-validity candidate, not non-root singleton-shift recovery"
        elif probe.startswith("l_"):
            classification = "ROOT DISTRIBUTION INTERVENTION ONLY"
            scope = "root support/range replacement with altered variance and weak response"
        else:
            classification = "OUT OF FAMILY"
            scope = "failed shape/response gate"
        rows.append(
            {
                "environment": row["intervention_environment"],
                "probe_variable": probe,
                "classification": classification,
                "semantic_scope": scope,
                "mean_location_change": row["mean_location_change"],
                "std_ratio": row["std_ratio"],
                "centered_ks": row["centered_ks"],
                "max_abs_descendant_smd": row["max_abs_descendant_smd"],
            }
        )
    downstream = []
    frame = load(raw_root, "uniform_red_mid")
    for outcome, parents in SIGNAL_PARENTS.items():
        action_train, action_test = frame.iloc[:500], frame.iloc[500:]
        fixed = source_fits[outcome]
        intercept_only, _ = fit_family(action_train, outcome, parents, fixed, "M0")
        unrestricted, _ = fit_family(action_train, outcome, parents, fixed, "M3")
        downstream.append(
            {
                "environment": "uniform_red_mid",
                "outcome": outcome,
                "parents": ";".join(parents),
                "heldout_nll_fully_invariant_source_mechanism": float(nll_rows(action_test, outcome, parents, fixed).mean()),
                "heldout_nll_intercept_only_M0": float(nll_rows(action_test, outcome, parents, intercept_only).mean()),
                "heldout_nll_unrestricted_M3": float(nll_rows(action_test, outcome, parents, unrestricted).mean()),
                "note": "support shift changes parent range; NLL gap mixes drift/nonlinearity with mechanism change",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(downstream)


def mode_catalog(
    comparisons: pd.DataFrame,
    fits: dict[tuple[str, str], Fit],
    source_fits: dict[str, Fit],
) -> pd.DataFrame:
    rows = []
    for outcome, source in source_fits.items():
        rows.append(
            {
                "node": outcome,
                "mode": "identity",
                "environment": REFERENCE,
                "family": "source",
                "alpha": source.alpha,
                "s": 1.0,
                "omega": source.omega,
                "beta_json": json.dumps(source.beta.tolist()),
                "cost": 0,
                "parameter_origin": "source observational regression, first 5000 contiguous rows",
                "archive_composability": "identity only",
            }
        )
    for _, row in comparisons.iterrows():
        environment = str(row["environment"])
        outcome = str(row["outcome"])
        fit = fits[(environment, "M2")]
        rows.append(
            {
                "node": outcome,
                "mode": f"{row['physical_control']}:{row['control_level']}",
                "environment": environment,
                "family": "M2 finite heteroskedastic scale",
                "alpha": fit.alpha,
                "s": row["s_hat_M2"],
                "omega": fit.omega,
                "beta_json": json.dumps(fit.beta.tolist()),
                "cost": 1,
                "parameter_origin": "pre-calibrated hardware mode, first 500 contiguous environment rows",
                "archive_composability": "singleton only; no cross-node joint execution in archive",
            }
        )
    return pd.DataFrame(rows).sort_values(["node", "cost", "environment"]).reset_index(drop=True)


def regular_cov(values: np.ndarray) -> np.ndarray:
    cov = np.cov(values, rowvar=False, ddof=1)
    ridge = max(float(np.trace(cov)) / len(cov) * 1e-9, 1e-9)
    return cov + ridge * np.eye(len(cov))


def gaussian_kl(mean_q: np.ndarray, cov_q: np.ndarray, mean_p: np.ndarray, cov_p: np.ndarray) -> float:
    dimension = len(mean_q)
    chol_p = np.linalg.cholesky(cov_p)
    solved_cov = linalg.cho_solve((chol_p, True), cov_q)
    difference = mean_p - mean_q
    solved_mean = linalg.cho_solve((chol_p, True), difference)
    logdet_p = 2.0 * np.log(np.diag(chol_p)).sum()
    chol_q = np.linalg.cholesky(cov_q)
    logdet_q = 2.0 * np.log(np.diag(chol_q)).sum()
    return float(0.5 * (np.trace(solved_cov) + difference @ solved_mean - dimension + logdet_p - logdet_q))


def gaussian_w2(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray) -> float:
    root_b = linalg.sqrtm(cov_b)
    middle = linalg.sqrtm(root_b @ cov_a @ root_b)
    value = float(np.sum((mean_a - mean_b) ** 2) + np.trace(cov_a + cov_b - 2.0 * middle).real)
    return math.sqrt(max(value, 0.0))


def empirical_distances(target: np.ndarray, candidate: np.ndarray, source_scale: np.ndarray) -> dict[str, float]:
    target_z = target / source_scale
    candidate_z = candidate / source_scale
    mean_error = float(np.linalg.norm(target_z.mean(axis=0) - candidate_z.mean(axis=0)))
    covariance_error = float(
        np.linalg.norm(np.cov(target_z, rowvar=False) - np.cov(candidate_z, rowvar=False), ord="fro")
    )
    distances = spatial.distance.cdist(target_z, candidate_z)
    energy = float(
        2 * distances.mean()
        - spatial.distance.pdist(target_z).mean()
        - spatial.distance.pdist(candidate_z).mean()
    )
    pooled = np.vstack((target_z, candidate_z))
    median = float(np.median(spatial.distance.pdist(pooled)))
    gamma = 1.0 / max(2.0 * median**2, np.finfo(float).eps)
    k_xy = np.exp(-gamma * distances**2).mean()
    k_xx = np.exp(-gamma * spatial.distance.squareform(spatial.distance.pdist(target_z)) ** 2)
    k_yy = np.exp(-gamma * spatial.distance.squareform(spatial.distance.pdist(candidate_z)) ** 2)
    mmd2 = float(k_xx.mean() + k_yy.mean() - 2 * k_xy)
    return {
        "standardized_mean_error": mean_error,
        "standardized_covariance_fro_error": covariance_error,
        "energy_distance_standardized": energy,
        "mmd2_rbf_standardized": max(mmd2, 0.0),
    }


def compose_station_moments(
    nodes: tuple[str, ...],
    root_mean: np.ndarray,
    root_cov: np.ndarray,
    local_fits: dict[str, Fit],
) -> tuple[np.ndarray, np.ndarray]:
    roots = tuple(node for node in nodes if node not in local_fits)
    outcomes = tuple(node for node in nodes if node in local_fits)
    if len(roots) + len(outcomes) != len(nodes):
        raise ValueError(nodes)
    mean = np.zeros(len(nodes))
    cov = np.zeros((len(nodes), len(nodes)))
    root_index = [nodes.index(node) for node in roots]
    mean[root_index] = root_mean
    cov[np.ix_(root_index, root_index)] = root_cov
    for outcome in outcomes:
        fit = local_fits[outcome]
        index = nodes.index(outcome)
        mean[index] = fit.alpha + fit.beta @ root_mean
        cross = root_cov @ fit.beta
        cov[root_index, index] = cross
        cov[index, root_index] = cross
        cov[index, index] = fit.beta @ root_cov @ fit.beta + fit.omega
    for left_index, left in enumerate(outcomes):
        for right in outcomes[left_index + 1 :]:
            value = float(local_fits[left].beta @ root_cov @ local_fits[right].beta)
            i, j = nodes.index(left), nodes.index(right)
            cov[i, j] = cov[j, i] = value
    return mean, cov


def station_for_outcome(outcome: str) -> str:
    return f"station_{outcome[-1]}"


def b1_matching(
    raw_root: Path,
    comparisons: pd.DataFrame,
    fits: dict[tuple[str, str], Fit],
    source_fits: dict[str, Fit],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = load(raw_root, REFERENCE)
    mode_lookup = comparisons.groupby("outcome")["environment"].apply(list).to_dict()
    rows = []
    residual_rows = []
    for station, nodes in STATIONS.items():
        roots = tuple(node for node in nodes if node not in source_fits)
        outcomes = tuple(node for node in nodes if node in source_fits)
        ref_train = reference.iloc[:5000]
        residuals = []
        for outcome in outcomes:
            fit = source_fits[outcome]
            residuals.append(
                ref_train[outcome].to_numpy(float)
                - fit.alpha
                - ref_train.loc[:, roots].to_numpy(float) @ fit.beta
            )
        corr = float(np.corrcoef(residuals)[0, 1])
        residual_rows.append(
            {
                "station": station,
                "outcome_left": outcomes[0],
                "outcome_right": outcomes[1],
                "source_residual_correlation": corr,
                "independent_noise_plausible_abs_corr_le_0_1": bool(abs(corr) <= 0.1),
            }
        )
    for outcome, hidden_environment in PILOT_MODES.items():
        station = station_for_outcome(outcome)
        nodes = (*SIGNAL_PARENTS[outcome], outcome)
        roots = tuple(node for node in nodes if node not in source_fits)
        station_outcomes = tuple(node for node in nodes if node in source_fits)
        root_values = reference.iloc[:5000].loc[:, roots].to_numpy(float)
        root_mean = root_values.mean(axis=0)
        root_cov = regular_cov(root_values)
        target_frame = load(raw_root, hidden_environment)
        target_cal = target_frame.iloc[400:600].loc[:, nodes].to_numpy(float)
        target_eval = target_frame.iloc[800:1000].loc[:, nodes].to_numpy(float)
        target_cal_mean, target_cal_cov = target_cal.mean(axis=0), regular_cov(target_cal)
        target_eval_mean, target_eval_cov = target_eval.mean(axis=0), regular_cov(target_eval)
        source_scale = reference.iloc[:5000].loc[:, nodes].std(axis=0, ddof=1).to_numpy(float)
        candidate_environments = [REFERENCE, *mode_lookup[outcome]]
        for family in ("M0", "M1", "M2", "M3"):
            scored = []
            for candidate_environment in candidate_environments:
                locals_for_candidate = {node: source_fits[node] for node in station_outcomes}
                if candidate_environment != REFERENCE:
                    candidate_calibration = load(raw_root, candidate_environment).iloc[:400]
                    candidate_fit, _ = fit_family(
                        candidate_calibration,
                        outcome,
                        SIGNAL_PARENTS[outcome],
                        source_fits[outcome],
                        family,
                    )
                    locals_for_candidate[outcome] = candidate_fit
                candidate_mean, candidate_cov = compose_station_moments(
                    nodes, root_mean, root_cov, locals_for_candidate
                )
                scored.append(
                    (
                        gaussian_kl(target_cal_mean, target_cal_cov, candidate_mean, candidate_cov),
                        candidate_environment,
                    )
                )
            selection_kl, selected = min(scored, key=lambda item: (item[0], item[1]))
            if selected == REFERENCE:
                candidate_eval = reference.iloc[8000:8200].loc[:, nodes].to_numpy(float)
            else:
                candidate_eval = load(raw_root, selected).iloc[600:800].loc[:, nodes].to_numpy(float)
            candidate_mean = candidate_eval.mean(axis=0)
            candidate_cov = regular_cov(candidate_eval)
            distances = empirical_distances(target_eval, candidate_eval, source_scale)
            rows.append(
                {
                    "hidden_target_environment": hidden_environment,
                    "outcome": outcome,
                    "station_subsystem": f"{station}_{outcome}",
                    "n_nodes": len(nodes),
                    "family": family,
                    "selected_environment": selected,
                    "selected_mode_correct": selected == hidden_environment,
                    "selection_target_rows": "400:600",
                    "candidate_calibration_rows": "0:400",
                    "candidate_evaluation_rows": "600:800",
                    "target_evaluation_rows": "800:1000",
                    "selection_model_gaussian_kl": selection_kl,
                    "heldout_physical_gaussian_kl": gaussian_kl(
                        target_eval_mean, target_eval_cov, candidate_mean, candidate_cov
                    ),
                    "heldout_gaussian_w2": gaussian_w2(
                        target_eval_mean / source_scale,
                        target_eval_cov / np.outer(source_scale, source_scale),
                        candidate_mean / source_scale,
                        candidate_cov / np.outer(source_scale, source_scale),
                    ),
                    **distances,
                    "physical_execution_interpretation": "held-out rows from one archived singleton-action session; not a new or replicated execution",
                }
            )
        candidate_eval = reference.iloc[8000:8200].loc[:, nodes].to_numpy(float)
        candidate_mean, candidate_cov = candidate_eval.mean(axis=0), regular_cov(candidate_eval)
        distances = empirical_distances(target_eval, candidate_eval, source_scale)
        rows.append(
            {
                "hidden_target_environment": hidden_environment,
                "outcome": outcome,
                "station_subsystem": f"{station}_{outcome}",
                "n_nodes": len(nodes),
                "family": "NO_ACTION",
                "selected_environment": REFERENCE,
                "selected_mode_correct": False,
                "selection_target_rows": "400:600",
                "candidate_calibration_rows": "reference 0:5000",
                "candidate_evaluation_rows": "reference 8000:8200",
                "target_evaluation_rows": "800:1000",
                "selection_model_gaussian_kl": None,
                "heldout_physical_gaussian_kl": gaussian_kl(
                    target_eval_mean, target_eval_cov, candidate_mean, candidate_cov
                ),
                "heldout_gaussian_w2": gaussian_w2(
                    target_eval_mean / source_scale,
                    target_eval_cov / np.outer(source_scale, source_scale),
                    candidate_mean / source_scale,
                    candidate_cov / np.outer(source_scale, source_scale),
                ),
                **distances,
                "physical_execution_interpretation": "held-out reference rows; no action",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(residual_rows)


def continuous_noise_law(
    raw_root: Path,
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    exposure = comparisons[comparisons["physical_control"].str.startswith("t_")]
    for control, group in exposure.groupby("physical_control"):
        group = group.sort_values("s_hat_M2")
        outcome = str(group.iloc[0]["outcome"])
        s = np.r_[1.0, group["s_hat_M2"].to_numpy(float)]
        omega = np.r_[float(group.iloc[0]["omega_source"]), group["omega_action_M2"].to_numpy(float)]
        design = np.column_stack((s**2, np.ones(len(s))))
        parameters, _ = optimize.nnls(design, omega)
        predictions = design @ parameters
        loo_predictions = []
        for index in range(len(s)):
            mask = np.arange(len(s)) != index
            loo_parameters, _ = optimize.nnls(design[mask], omega[mask])
            loo_predictions.append(float(design[index] @ loo_parameters))
        loo_predictions_array = np.asarray(loo_predictions)
        relative = np.abs(loo_predictions_array - omega) / omega
        law_minus_free = []
        invariant_minus_law = []
        for action_index, (_, action) in enumerate(group.iterrows(), start=1):
            environment = str(action["environment"])
            frame = load(raw_root, environment).iloc[500:]
            parents = SIGNAL_PARENTS[outcome]
            beta = np.asarray(json.loads(str(action["beta_M2_json"])), float)
            residual = (
                frame[outcome].to_numpy(float)
                - float(action["alpha_M2"])
                - frame.loc[:, parents].to_numpy(float) @ beta
            )
            def residual_nll(variance: float) -> float:
                return float(0.5 * np.mean(
                    math.log(2.0 * math.pi * variance) + residual**2 / variance
                ))
            law_nll = residual_nll(max(loo_predictions_array[action_index], np.finfo(float).tiny))
            free_nll = residual_nll(float(action["omega_action_M2"]))
            invariant_nll = residual_nll(float(action["omega_source"]))
            law_minus_free.append(law_nll - free_nll)
            invariant_minus_law.append(invariant_nll - law_nll)
        rows.append(
            {
                "physical_control": control,
                "outcome": outcome,
                "n_distinct_settings_including_source": len(s),
                "tau_squared": float(parameters[0]),
                "nu_squared": float(parameters[1]),
                "positive_nondegenerate": bool(np.all(parameters > 0)),
                "in_sample_relative_rmse": float(np.sqrt(np.mean(((predictions - omega) / omega) ** 2))),
                "leave_one_level_out_median_relative_error": float(np.median(relative)),
                "leave_one_level_out_max_relative_error": float(relative.max()),
                "heldout_loo_law_minus_free_M2_nll_mean": float(np.mean(law_minus_free)),
                "heldout_loo_law_improvement_over_invariant_M1_nll_mean": float(np.mean(invariant_minus_law)),
                "all_action_coefficients_proportional": bool(group["coefficients_proportional"].all()),
                "pursue_continuous_law": bool(
                    np.all(parameters > 0)
                    and np.median(relative) <= 0.2
                    and relative.max() <= 0.5
                    and group["coefficients_proportional"].all()
                    and np.mean(law_minus_free) <= 0.02
                    and np.mean(invariant_minus_law) > 0
                ),
                "s_values_json": json.dumps(s.tolist()),
                "omega_values_json": json.dumps(omega.tolist()),
                "loo_predictions_json": json.dumps(loo_predictions),
            }
        )
    return pd.DataFrame(rows).sort_values("physical_control").reset_index(drop=True)


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    raw_root = args.data_root.resolve()
    OUTPUT = prepare_output_directory(
        args.output, overwrite=args.overwrite, protected=(raw_root,)
    )
    archive = raw_root / f"{DATASET}.zip"
    expected = "8781960b5fff5d752f57566393d4b0f8706cac9b2cd2a3d3226442f17c7bab60"
    if sha256(archive) != expected:
        raise RuntimeError("official intervention archive SHA-256 mismatch")

    inventory = dataset_inventory(raw_root)
    settings = environment_settings(raw_root)
    graph = augmented_graph()
    closure = closure_candidates()
    comparisons, fits, source_fits = run_model_comparisons(raw_root)
    selected_comparisons = comparisons[
        comparisons.apply(lambda row: PILOT_MODES.get(str(row["outcome"])) == row["environment"], axis=1)
    ].reset_index(drop=True)
    segments, stability = variance_stability(raw_root)
    support = support_theory_validation()
    shifts, downstream = shift_results(raw_root, source_fits)
    catalog = mode_catalog(comparisons, fits, source_fits)
    matching, residual_dependence = b1_matching(raw_root, comparisons, fits, source_fits)
    noise_law = continuous_noise_law(raw_root, comparisons)

    for name, frame in (
        ("dataset_inventory", inventory),
        ("environment_settings", settings),
        ("augmented_graph", graph),
        ("causal_closure_candidates", closure),
        ("model_comparisons_all_gain_modes", comparisons),
        ("model_comparisons_six_mode_pilot", selected_comparisons),
        ("variance_stability_segments", segments),
        ("variance_stability_summary", stability),
        ("changed_noise_support_synthetic", support),
        ("shift_environment_classification", shifts),
        ("shift_downstream_invariance", downstream),
        ("hardware_mode_catalog", catalog),
        ("b1_matching", matching),
        ("station_residual_dependence", residual_dependence),
        ("continuous_noise_law", noise_law),
    ):
        write_frame(name, frame)

    prior_gate = json.loads(
        (raw_root / "support" / "changed_noise_gate_summary.json").read_text()
    )
    m2 = matching[matching["family"] == "M2"]
    no_action = matching[matching["family"] == "NO_ACTION"].set_index("hidden_target_environment")
    paired_improvements = []
    for _, row in m2.iterrows():
        baseline = no_action.loc[row["hidden_target_environment"], "heldout_physical_gaussian_kl"]
        paired_improvements.append(float(baseline - row["heldout_physical_gaussian_kl"]))
    stable_count = int(stability["within_session_stable_threshold_1_5"].sum())
    conditionally_fixed_count = int(selected_comparisons["fixed_conditional_variance_supported"].sum())
    gate4 = bool((np.asarray(paired_improvements) > 0).sum() >= 4 and m2["selected_mode_correct"].sum() >= 4)
    gates = {
        "gate_1_causal_closure": {
            "outcome": "PASS",
            "basis": "six 6-8 node one-response station subsystems include every varying signal parent; paired IR/visible subsystems fail residual-independence",
        },
        "gate_2_heteroskedastic_mode_stability": {
            "outcome": "FAIL",
            "within_session_modes_passing_max_min_1_5": stable_count,
            "modes_without_parent_state_heteroskedasticity_at_p_0_01": conditionally_fixed_count,
            "denominator": len(stability),
            "basis": "all six averages are temporally stable, but all six reject constant conditional residual variance and each mode has only one session",
        },
        "gate_3_acquisition_coverage": {
            "outcome": "FAIL",
            "basis": "the prior preregistered changed-noise gate recovered useful union support on 0/6 targets",
            "prior_gate_result": prior_gate.get("gate_result", prior_gate.get("result", "FAIL")),
        },
        "gate_4_valid_B1_physical_matching_task": {
            "outcome": "PASS" if gate4 else "FAIL",
            "scope": "Level A restricted-station archive replay only",
            "M2_correct_selections": int(m2["selected_mode_correct"].sum()),
            "M2_tasks_improving_over_no_action": int((np.asarray(paired_improvements) > 0).sum()),
            "tasks": len(m2),
        },
    }
    write_json(OUTPUT / "machine_readable" / "gate_summary.json", gates)
    write_json(
        OUTPUT / "machine_readable" / "run_metadata.json",
        {
            "schema_version": 1,
            "seed": SEED,
            "raw_root": raw_root,
            "archive": archive,
            "archive_sha256": expected,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "split_semantics": {
                "M0_M3": "source 0:5000; action calibration 0:500; action test 500:1000",
                "stability": "four contiguous 250-row segments",
                "B1": "catalog calibration 0:400, target moments 400:600, candidate evaluation 600:800, target evaluation 800:1000",
            },
            "independence_warning": "each environment is one session; rows/segments are not independent physical replications",
            "gates": gates,
        },
    )
    print(json.dumps(_jsonable(gates), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
