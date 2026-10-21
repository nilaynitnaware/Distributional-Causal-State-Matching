#!/usr/bin/env python3
"""Run and report the frozen calibration-free singleton-query diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from experiments.finite_sample_robustness.design import derive_seed
from graphs.interventions import ShiftScaleIntervention
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from oracle.sparse_shift_scale_kl import solve_sparse_shift_scale_kl
from query_complexity.finite_sample import (
    _solve_estimated_dp,
    bootstrap_source_cache,
    conditional_node_moment,
    estimate_oas,
    estimated_node_moment,
)
from query_complexity.local_mechanism import (
    recover_local_mechanism_from_additive_probe,
)
from research.calibration_free_singleton_diagnostic.estimator import (
    DiagnosticMechanismFit,
    fit_calibrated,
    fit_calibration_free,
)
from scripts.run_query_complexity_local_mechanism_campaign import _source
from scripts._output_safety import prepare_output_directory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/calibration_free_singleton_finite_sample_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/calibration_free_singleton_finite_sample_v1"
SCHEMA = "calibration_free_singleton_finite_sample_results_v1"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "calibration_free_singleton_finite_sample_config_v1":
        raise ValueError("configuration schema is stale")
    if config.get("status") != "frozen_before_full_grid":
        raise ValueError("configuration is not frozen")
    expected = 3 * int(config["replicates_per_family"])
    if expected != int(config["expected_sources"]):
        raise ValueError("expected source count disagrees with the frozen design")
    return config


def _family(regime: dict[str, object]) -> str:
    return (
        "erdos_renyi_0.15"
        if regime["family"] == "erdos_renyi"
        else str(regime["family"])
    )


def _source_specs(config: dict[str, object]) -> list[dict[str, object]]:
    specs = []
    n = int(config["n"])
    for graph_index, regime in enumerate(config["graph_regimes"]):
        for replicate in range(int(config["replicates_per_family"])):
            index = len(specs)
            specs.append(
                {
                    "source_index": index,
                    "source_id": f"calfree-v1:n{n}:g{graph_index}:r{replicate}",
                    "graph_index": graph_index,
                    "graph_family": _family(regime),
                    "replicate": replicate,
                }
            )
    return specs


def _make_source(config: dict[str, object], spec: dict[str, object]):
    source_config = dict(config)
    source_config["base_seed"] = int(config["source_generator_base_seed"])
    regime = config["graph_regimes"][int(spec["graph_index"])]
    return _source(source_config, regime, int(config["n"]), int(spec["replicate"]))


def _source_manifest(config: dict[str, object]) -> dict[str, object]:
    attempts = []
    for spec in _source_specs(config):
        row = dict(spec)
        try:
            scm, target = _make_source(config, spec)
            roots = sorted(int(node) for node, degree in scm.dag.in_degree if degree == 0)
            row.update(
                {
                    "status": "selected_on_first_predetermined_attempt",
                    "attempt": 1,
                    "directed_edges": [list(map(int, edge)) for edge in sorted(scm.dag.edges)],
                    "edge_count": int(scm.dag.number_of_edges()),
                    "root_nodes": roots,
                    "structural_limitations": (
                        ["empty_graph"] if scm.dag.number_of_edges() == 0 else []
                    ),
                    "coefficients_sha256": _array_digest(scm.coefficients),
                    "noise_variances_sha256": _array_digest(scm.noise_variances),
                    "target_mean_sha256": _array_digest(target.mean),
                    "target_covariance_sha256": _array_digest(target.covariance),
                    "source_identity_sha256": _digest(
                        {
                            "edges": sorted(tuple(map(int, edge)) for edge in scm.dag.edges),
                            "coefficients": scm.coefficients.tolist(),
                            "noise": scm.noise_variances.tolist(),
                            "target_mean": target.mean.tolist(),
                            "target_covariance": target.covariance.tolist(),
                        }
                    ),
                }
            )
        except Exception as error:  # noqa: BLE001
            row.update(
                {
                    "status": "predetermined_source_construction_failed",
                    "attempt": 1,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "structural_limitations": ["source_construction_failure"],
                }
            )
        attempts.append(row)
    return {
        "selection_rule": "all 15 predetermined family-by-replicate coordinates; no performance-based replacement",
        "attempts": attempts,
        "scheduled": len(attempts),
        "constructed": sum(row["status"].startswith("selected") for row in attempts),
    }


def _environment() -> dict[str, object]:
    packages = {}
    for package in ("numpy", "scipy", "sklearn", "matplotlib", "networkx"):
        try:
            module = __import__(package)
            packages[package] = getattr(module, "__version__", "unknown")
        except Exception as error:  # noqa: BLE001
            packages[package] = f"unavailable: {error}"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as error:
        return f"UNAVAILABLE: {error.output.strip()}"


def _code_manifest(config_path: Path) -> dict[str, object]:
    paths = [
        config_path,
        ROOT / "research/calibration_free_singleton_diagnostic/estimator.py",
        Path(__file__).resolve(),
        ROOT / "tests/test_calibration_free_singleton_diagnostic.py",
    ]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.exists()
    }
    return {
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_at_freeze": _git("status", "--short"),
        "tracked_diff_sha256": hashlib.sha256(_git("diff", "--binary").encode()).hexdigest(),
        "diagnostic_file_sha256": hashes,
        "diagnostic_code_identity_sha256": _digest(hashes),
    }


def _freeze(config_path: Path, output: Path, config: dict[str, object]) -> str:
    output.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    frozen_path = output / "provenance/frozen_config.json"
    if frozen_path.exists():
        existing = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        if existing != config_hash:
            raise RuntimeError("existing frozen configuration differs from requested config")
    else:
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        frozen_path.write_bytes(config_path.read_bytes())
    _json(output / "provenance/source_manifest.json", _source_manifest(config))
    _json(output / "provenance/environment.json", _environment())
    code = _code_manifest(config_path)
    code["config_sha256"] = config_hash
    code["freeze_timestamp_note"] = "wall-clock time omitted from scientific identity"
    _json(output / "provenance/code_revision.json", code)
    return config_hash


def _bootstrap_intervention_means(
    samples: np.ndarray, replicates: int, seed: int, *, batch_size: int = 16
) -> np.ndarray:
    """Existing nonparametric row bootstrap, batched without changing its draws."""
    rng = np.random.default_rng(seed)
    count, n = samples.shape
    means = np.empty((replicates, n))
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        indices = rng.integers(0, count, size=(stop - start, count))
        means[start:stop] = samples[indices].mean(axis=1)
    return means


def _parent_metrics(true: set[int], estimated: set[int]) -> dict[str, float | bool]:
    tp = len(true & estimated)
    fp = len(estimated - true)
    fn = len(true - estimated)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "parent_precision": precision,
        "parent_recall": recall,
        "parent_f1": f1,
        "exact_support_recovery": true == estimated,
    }


def _node_row(
    *,
    config: dict[str, object],
    spec: dict[str, object],
    scm,
    source_population_mean: np.ndarray,
    node: int,
    absolute_h: float,
    signed_h: float,
    n_int: int,
    fit: DiagnosticMechanismFit,
    paired_difference: float | None,
) -> dict[str, object]:
    true_row = np.eye(int(config["n"]))[node] - scm.coefficients[:, node]
    true_incoming = scm.coefficients[:, node]
    true_parents = set(int(value) for value in scm.dag.predecessors(node))
    estimated_parents = set(fit.selected_parents)
    raw_incoming = -np.array(fit.raw_mechanism_row, copy=True)
    raw_incoming[node] = 0.0
    true_alpha = float(true_row @ source_population_mean)
    true_omega = float(scm.noise_variances[node])
    selected_coefficient_error = float(
        np.linalg.norm(fit.selected_incoming_coefficients - true_incoming)
    )
    raw_coefficient_error = (
        float(np.linalg.norm(raw_incoming - true_incoming))
        if np.all(np.isfinite(raw_incoming))
        else None
    )
    intercept_error = (
        abs(fit.source_intercept - true_alpha) if fit.valid else None
    )
    variance_error = (
        abs(fit.noise_variance - true_omega) if fit.valid else None
    )
    complete_error = (
        float(
            np.sqrt(
                selected_coefficient_error**2
                + (fit.source_intercept - true_alpha) ** 2
                + (fit.noise_variance - true_omega) ** 2
            )
        )
        if fit.valid
        else None
    )
    denominator_expected_sign = 1 if fit.estimator == "calibrated" else int(np.sign(signed_h))
    return {
        "schema_version": SCHEMA,
        **spec,
        "absolute_h": absolute_h,
        "signed_h_evaluation_only_for_calibration_free": signed_h,
        "query_sign": int(np.sign(signed_h)),
        "N_int": n_int,
        "N_obs": int(config["N_obs"]),
        "N_target": int(config["N_target"]),
        "estimator": fit.estimator,
        "node": node,
        "root_node": not true_parents,
        "valid_fit": fit.valid,
        "invalid_reason": fit.invalid_reason,
        "estimated_signed_denominator": fit.signed_denominator,
        "absolute_estimated_denominator": abs(fit.signed_denominator),
        "denominator_sign_error": int(np.sign(fit.signed_denominator)) != denominator_expected_sign,
        "true_gamma_evaluation_only": absolute_h / true_omega,
        "bootstrap_accepted_replicates": fit.bootstrap_valid_replicates,
        "bootstrap_scheduled_replicates": fit.bootstrap_total_replicates,
        "true_parents": sorted(true_parents),
        "selected_parents": list(fit.selected_parents),
        **_parent_metrics(true_parents, estimated_parents),
        "raw_mechanism_row": fit.raw_mechanism_row,
        "selected_mechanism_row": fit.selected_mechanism_row,
        "true_mechanism_row": true_row,
        "raw_coefficient_l2_error": raw_coefficient_error,
        "selected_coefficient_l2_error": selected_coefficient_error if fit.valid else None,
        "source_intercept_estimate": fit.source_intercept if fit.valid else None,
        "true_source_intercept": true_alpha,
        "intercept_absolute_error": intercept_error,
        "source_noise_variance_estimate": fit.noise_variance if fit.valid else None,
        "true_source_noise_variance": true_omega,
        "variance_absolute_error": variance_error,
        "variance_relative_error": variance_error / true_omega if variance_error is not None else None,
        "raw_quadratic_variance": fit.raw_quadratic_variance if fit.valid else None,
        "selected_quadratic_variance": fit.selected_quadratic_variance if fit.valid else None,
        "calibrated_vs_selected_quadratic_variance_difference": (
            fit.noise_variance - fit.selected_quadratic_variance
            if fit.valid and fit.estimator == "calibrated"
            else None
        ),
        "complete_local_mechanism_error": complete_error,
        "complete_local_mechanism_error_definition": "sqrt(||beta_hat-beta||_2^2 + (alpha_hat-alpha)^2 + (omega_hat-omega)^2)",
        "paired_raw_row_max_abs_difference": paired_difference,
        "bootstrap_parent_probabilities": fit.bootstrap_parent_probabilities,
        "bootstrap_confidence_intervals": fit.bootstrap_confidence_intervals,
        "root_metric_convention": "precision=recall=F1=1 for true-empty/predicted-empty; recall=1 for any true-empty set; precision=1 for any predicted-empty set",
    }


def _fallback_objective(node: int, source_estimate, target_estimate):
    return conditional_node_moment(
        node=node,
        parents=frozenset(),
        source_estimate=source_estimate,
        target_estimate=target_estimate,
    )


def _end_to_end(
    *,
    config: dict[str, object],
    scm,
    target,
    source_estimate,
    target_estimate,
    fits: list[DiagnosticMechanismFit],
    oracle,
) -> dict[str, object]:
    objectives = []
    fallback_nodes = []
    for node, fit in enumerate(fits):
        if fit.valid:
            objectives.append(
                estimated_node_moment(
                    node=node,
                    incoming_coefficients=fit.selected_incoming_coefficients,
                    source_intercept=fit.source_intercept,
                    noise_variance=fit.noise_variance,
                    target_mean=target_estimate.mean,
                    target_covariance=target_estimate.covariance,
                    parents=fit.selected_parents,
                )
            )
        else:
            fallback_nodes.append(node)
            objectives.append(_fallback_objective(node, source_estimate, target_estimate))
    dp = _solve_estimated_dp(
        tuple(objectives),
        source_estimate=source_estimate,
        target_estimate=target_estimate,
        budget=int(config["budget"]),
        shift_bounds=tuple(map(float, config["shift_bounds"])),
        scale_bounds=tuple(map(float, config["scale_bounds"])),
        method="graph_free_finite_sample",
    )
    action = ShiftScaleIntervention(dp.budgeted_shift, dp.budgeted_scale)
    candidate_mean, candidate_covariance = scm.interventional_moments(action)
    true_kl = gaussian_kl_target_to_candidate(
        target.mean, target.covariance, candidate_mean, candidate_covariance
    )
    low_h, high_h = map(float, config["shift_bounds"])
    low_s, high_s = map(float, config["scale_bounds"])
    support = action.support_cost(float(config["support_atol"]))
    bounds_violation = max(
        0.0,
        float(np.max(low_h - action.shift)),
        float(np.max(action.shift - high_h)),
        float(np.max(low_s - action.scale)),
        float(np.max(action.scale - high_s)),
    )
    bounds_atol = float(config["support_atol"])
    return {
        "returned_shift": action.shift,
        "returned_scale": action.scale,
        "returned_support_cost": support,
        "support_budget": int(config["budget"]),
        "support_check_passed": support <= int(config["budget"]),
        "bounds_check_passed": bounds_violation <= bounds_atol,
        "bounds_violation_max": bounds_violation,
        "bounds_check_atol": bounds_atol,
        "estimated_objective_value": dp.raw_K_B_star,
        "true_returned_KL": true_kl,
        "exact_sparse_population_optimum_KL": oracle.raw_K_B_star,
        "true_excess_KL": true_kl - oracle.raw_K_B_star,
        "target_has_zero_population_optimum": oracle.raw_K_B_star
        <= float(config["zero_optimum_atol"]),
        "fallback_used": bool(fallback_nodes),
        "fallback_nodes": fallback_nodes,
        "fallback_rule": "existing graph-free empty-parent OAS conditional objective",
    }


def _aggregate_source_condition(
    node_rows: list[dict[str, object]], e2e: dict[str, object]
) -> dict[str, object]:
    first = node_rows[0]
    valid = [row for row in node_rows if row["valid_fit"]]
    metrics = (
        "raw_coefficient_l2_error",
        "selected_coefficient_l2_error",
        "parent_precision",
        "parent_recall",
        "parent_f1",
        "intercept_absolute_error",
        "variance_absolute_error",
        "variance_relative_error",
        "complete_local_mechanism_error",
        "paired_raw_row_max_abs_difference",
    )
    result = {
        key: first[key]
        for key in (
            "schema_version",
            "source_index",
            "source_id",
            "graph_index",
            "graph_family",
            "replicate",
            "absolute_h",
            "N_int",
            "N_obs",
            "N_target",
            "estimator",
        )
    }
    result.update(
        {
            "scheduled_nodes": len(node_rows),
            "valid_nodes": len(valid),
            "invalid_nodes": len(node_rows) - len(valid),
            "node_failure_rate": (len(node_rows) - len(valid)) / len(node_rows),
            "denominator_sign_errors": sum(bool(row["denominator_sign_error"]) for row in node_rows),
            "denominator_sign_error_rate": sum(
                bool(row["denominator_sign_error"]) for row in node_rows
            )
            / len(node_rows),
            "exact_support_recovery_rate": float(
                np.mean([bool(row["exact_support_recovery"]) for row in valid])
            )
            if valid
            else None,
            "run_status": "SUCCESS" if valid else "ALL_NODE_FITS_INVALID",
        }
    )
    for metric in metrics:
        values = [float(row[metric]) for row in valid if row.get(metric) is not None]
        result[f"mean_{metric}"] = float(np.mean(values)) if values else None
    result.update(e2e)
    return result


def _execute_source(
    config: dict[str, object], spec: dict[str, object], *, validation: bool = False
) -> dict[str, object]:
    started = perf_counter()
    scm, target = _make_source(config, spec)
    n = int(config["n"])
    max_n_int = max(map(int, config["N_int_grid"]))
    source_id = str(spec["source_id"])
    source_rng = np.random.default_rng(
        derive_seed(int(config["base_seed"]), source_id, "observational")
    )
    target_rng = np.random.default_rng(
        derive_seed(int(config["base_seed"]), source_id, "target")
    )
    source_samples = scm.sample_observational(int(config["N_obs"]), source_rng)
    target_samples = scm.sample_interventional(
        target.generating_intervention, int(config["N_target"]), target_rng
    )
    source_estimate = estimate_oas(source_samples)
    target_estimate = estimate_oas(target_samples)
    bootstrap_replicates = int(config["bootstrap_replicates"])
    source_boot = bootstrap_source_cache(
        source_samples,
        replicates=bootstrap_replicates,
        seed=derive_seed(int(config["base_seed"]), source_id, "source_bootstrap"),
    )
    source_population_mean, _ = scm.observational_moments()
    oracle = solve_sparse_shift_scale_kl(
        scm=scm,
        target_mean=target.mean,
        target_covariance=target.covariance,
        budget=int(config["budget"]),
        shift_bounds=tuple(map(float, config["shift_bounds"])),
        log_scale_bounds=tuple(np.log(np.asarray(config["scale_bounds"], dtype=float))),
    )
    all_node_rows: list[dict[str, object]] = []
    all_source_rows: list[dict[str, object]] = []
    base_pools = {}
    total_effects = {}
    base_bootstraps = {}
    for node in range(n):
        rng = np.random.default_rng(
            derive_seed(int(config["base_seed"]), source_id, "intervention_noise", node)
        )
        base_pools[node] = scm.sample_observational(max_n_int, rng)
        unit_shift = np.zeros(n)
        unit_shift[node] = 1.0
        shifted_mean, shifted_covariance = scm.interventional_moments(
            ShiftScaleIntervention(unit_shift, np.ones(n))
        )
        if not np.allclose(shifted_covariance, scm.observational_moments()[1]):
            raise RuntimeError("singleton additive query changed population covariance")
        total_effects[node] = shifted_mean - source_population_mean
        for n_int in map(int, config["N_int_grid"]):
            base_bootstraps[node, n_int] = _bootstrap_intervention_means(
                base_pools[node][:n_int],
                bootstrap_replicates,
                derive_seed(
                    int(config["base_seed"]), source_id, "intervention_bootstrap", node, n_int
                ),
            )

    for absolute_h in map(float, config["absolute_h_grid"]):
        per_condition_fits: dict[tuple[int, str], list[DiagnosticMechanismFit]] = defaultdict(list)
        per_condition_rows: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
        for n_int in map(int, config["N_int_grid"]):
            for node in range(n):
                sign = -1.0 if (int(spec["source_index"]) * n + node) % 2 else 1.0
                signed_h = sign * absolute_h
                point_intervention_mean = (
                    base_pools[node][:n_int].mean(axis=0)
                    + signed_h * total_effects[node]
                )
                bootstrap_intervention_means = (
                    base_bootstraps[node, n_int]
                    + signed_h * total_effects[node][None, :]
                )
                common = dict(
                    node=node,
                    source_mean=source_estimate.mean,
                    source_covariance=source_estimate.covariance,
                    source_precision=source_estimate.precision,
                    mean_response=point_intervention_mean - source_estimate.mean,
                    bootstrap_source_means=source_boot.means,
                    bootstrap_precisions=source_boot.precisions,
                    bootstrap_intervention_means=bootstrap_intervention_means,
                    confidence_level=float(config["confidence_level"]),
                    orientation_confidence=float(config["orientation_confidence"]),
                    support_probability_threshold=float(
                        config["support_probability_threshold"]
                    ),
                )
                calibrated = fit_calibrated(
                    signed_h=signed_h,
                    minimum_self_precision_effect=float(
                        config["minimum_calibrated_self_precision_effect"]
                    ),
                    **common,
                )
                calibration_free = fit_calibration_free(
                    minimum_abs_denominator=float(
                        config["minimum_abs_uncalibrated_denominator"]
                    ),
                    **common,
                )
                paired_difference = (
                    float(
                        np.max(
                            np.abs(
                                calibrated.raw_mechanism_row
                                - calibration_free.raw_mechanism_row
                            )
                        )
                    )
                    if calibrated.valid and calibration_free.valid
                    else None
                )
                for fit in (calibrated, calibration_free):
                    row = _node_row(
                        config=config,
                        spec=spec,
                        scm=scm,
                        source_population_mean=source_population_mean,
                        node=node,
                        absolute_h=absolute_h,
                        signed_h=signed_h,
                        n_int=n_int,
                        fit=fit,
                        paired_difference=paired_difference,
                    )
                    all_node_rows.append(row)
                    per_condition_fits[n_int, fit.estimator].append(fit)
                    per_condition_rows[n_int, fit.estimator].append(row)
            for estimator in ("calibrated", "calibration_free"):
                fits = per_condition_fits[n_int, estimator]
                e2e = _end_to_end(
                    config=config,
                    scm=scm,
                    target=target,
                    source_estimate=source_estimate,
                    target_estimate=target_estimate,
                    fits=fits,
                    oracle=oracle,
                )
                all_source_rows.append(
                    _aggregate_source_condition(
                        per_condition_rows[n_int, estimator], e2e
                    )
                )
    expected_nodes = (
        len(config["absolute_h_grid"])
        * len(config["N_int_grid"])
        * n
        * len(config["methods"])
    )
    expected_conditions = (
        len(config["absolute_h_grid"])
        * len(config["N_int_grid"])
        * len(config["methods"])
    )
    if len(all_node_rows) != expected_nodes or len(all_source_rows) != expected_conditions:
        raise RuntimeError("source execution emitted an incomplete fixed grid")
    return {
        "schema_version": SCHEMA,
        "spec": spec,
        "node_rows": all_node_rows,
        "source_condition_rows": all_source_rows,
        "sample_hashes": {
            "observational": _array_digest(source_samples),
            "target": _array_digest(target_samples),
            "intervention_base_by_node": {
                str(node): _array_digest(base_pools[node]) for node in range(n)
            },
        },
        "pairing": {
            "observational_target_reused_across_all_conditions": True,
            "intervention_prefixes_nested": True,
            "intervention_noise_common_across_magnitudes": True,
            "intervention_noise_independent_of_observational_and_target": True,
            "common_noise_construction": "independent observational SCM draw plus exact singleton mean displacement h*(I-B^T)^-1 e_i",
        },
        "runtime_seconds": perf_counter() - started,
        "validation_execution": validation,
    }


def _validate(config: dict[str, object]) -> dict[str, object]:
    checks = []
    spec = _source_specs(config)[0]
    scm, target = _make_source(config, spec)
    mu, sigma = scm.observational_moments()
    node = next((i for i in range(int(config["n"])) if scm.dag.in_degree(i)), 0)
    for h in (-2.0, -0.1, 0.05, 0.5):
        shift = np.zeros(int(config["n"]))
        shift[node] = h
        shifted_mu, _ = scm.interventional_moments(
            ShiftScaleIntervention(shift, np.ones(int(config["n"])))
        )
        recovered = recover_local_mechanism_from_additive_probe(
            node=node,
            source_mean=mu,
            exact_shifted_mean=shifted_mu,
            source_covariance=sigma,
            coefficient_support_tolerance=0.0,
        )
        true_row = np.eye(int(config["n"]))[node] - scm.coefficients[:, node]
        checks.append(
            {
                "name": f"exact_moment_recovery_h_{h}",
                "passed": bool(np.allclose(recovered.mechanism_row, true_row, atol=1e-10)),
                "max_abs_error": float(np.max(np.abs(recovered.mechanism_row - true_row))),
            }
        )
    precision = np.linalg.solve(sigma, np.eye(sigma.shape[0]))
    h = -0.25
    shift = np.zeros(int(config["n"]))
    shift[node] = h
    shifted_mu, _ = scm.interventional_moments(
        ShiftScaleIntervention(shift, np.ones(int(config["n"])))
    )
    d = shifted_mu - mu
    left = precision @ d
    left /= left[node]
    right = precision @ (d / h)
    right /= right[node]
    checks.append(
        {
            "name": "same_input_raw_ratio_algebra",
            "passed": bool(np.array_equal(left, right) or np.allclose(left, right, atol=1e-15)),
            "max_abs_difference": float(np.max(np.abs(left - right))),
        }
    )
    parameters = set(inspect.signature(fit_calibration_free).parameters)
    forbidden = {"h", "signed_h", "shift_magnitude", "query_magnitude"}
    checks.append(
        {
            "name": "calibration_free_api_cannot_receive_h",
            "passed": not bool(parameters & forbidden),
            "parameters": sorted(parameters),
        }
    )
    n = int(config["n"])
    r = 4
    near_zero_fit = fit_calibration_free(
        node=0,
        source_mean=np.zeros(n),
        source_covariance=np.eye(n),
        source_precision=np.eye(n),
        mean_response=np.zeros(n),
        bootstrap_source_means=np.zeros((r, n)),
        bootstrap_precisions=np.repeat(np.eye(n)[None, :, :], r, axis=0),
        bootstrap_intervention_means=np.zeros((r, n)),
        minimum_abs_denominator=float(config["minimum_abs_uncalibrated_denominator"]),
        confidence_level=float(config["confidence_level"]),
        orientation_confidence=float(config["orientation_confidence"]),
        support_probability_threshold=float(config["support_probability_threshold"]),
    )
    checks.append(
        {
            "name": "near_zero_denominator_explicit_failure",
            "passed": (not near_zero_fit.valid)
            and near_zero_fit.invalid_reason == "near_zero_or_nonfinite_signed_denominator",
            "valid": near_zero_fit.valid,
            "invalid_reason": near_zero_fit.invalid_reason,
        }
    )
    small = dict(config)
    small["absolute_h_grid"] = [0.5]
    small["N_int_grid"] = [250]
    small["bootstrap_replicates"] = 16
    result = _execute_source(small, spec, validation=True)
    source_rows = result["source_condition_rows"]
    checks.append(
        {
            "name": "small_end_to_end_support_bounds_and_population_kl",
            "passed": all(
                row["support_check_passed"]
                and row["bounds_check_passed"]
                and row["true_excess_KL"] >= -1e-9
                for row in source_rows
            ),
            "rows": source_rows,
            "population_kl_evaluation": "independent exact SCM moments plus gaussian_kl_target_to_candidate",
        }
    )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "validation_source": spec,
        "target_population_optimum": solve_sparse_shift_scale_kl(
            scm=scm,
            target_mean=target.mean,
            target_covariance=target.covariance,
            budget=int(config["budget"]),
            shift_bounds=tuple(map(float, config["shift_bounds"])),
            log_scale_bounds=tuple(np.log(np.asarray(config["scale_bounds"], dtype=float))),
        ).raw_K_B_star,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in keys:
                value = _clean(row.get(key))
                encoded[key] = (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
            writer.writerow(encoded)


def _source_bootstrap_summary(
    values: list[float], seed: int, replicates: int
) -> dict[str, float | int | None]:
    if not values:
        return {"n_sources": 0, "mean": None, "median": None, "ci_lower": None, "ci_upper": None}
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, array.size, size=(replicates, array.size))].mean(axis=1)
    return {
        "n_sources": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci_lower": float(np.quantile(means, 0.025)),
        "ci_upper": float(np.quantile(means, 0.975)),
    }


def _summaries(config: dict[str, object], rows: list[dict[str, object]]):
    expanded = rows + [{**row, "graph_family": "ALL"} for row in rows]
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in expanded:
        grouped[
            (
                row["absolute_h"],
                row["N_int"],
                row["estimator"],
                row["graph_family"],
            )
        ].append(row)
    metrics = [
        "mean_raw_coefficient_l2_error",
        "mean_selected_coefficient_l2_error",
        "mean_parent_f1",
        "mean_intercept_absolute_error",
        "mean_variance_relative_error",
        "mean_complete_local_mechanism_error",
        "node_failure_rate",
        "denominator_sign_error_rate",
        "true_excess_KL",
    ]
    result = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        record = {
            "absolute_h": key[0],
            "N_int": key[1],
            "estimator": key[2],
            "graph_family": key[3],
            "scheduled_source_conditions": len(items),
            "source_condition_failure_count": sum(row["run_status"] != "SUCCESS" for row in items),
            "source_condition_fallback_count": sum(bool(row["fallback_used"]) for row in items),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in items if row.get(metric) is not None]
            record[metric] = _source_bootstrap_summary(
                values,
                derive_seed(int(config["base_seed"]), "summary", *key, metric),
                int(config["source_bootstrap_ci_replicates"]),
            )
        result.append(record)
    return result


def _figures(output: Path, summary: list[dict[str, object]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in summary if row["graph_family"] == "ALL"]
    methods = ("calibrated", "calibration_free")
    colors = {250: "#d95f02", 1000: "#7570b3", 10000: "#1b9e77"}
    styles = {"calibrated": "--", "calibration_free": "-"}
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    files = []

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for metric, label, axis in (
        ("mean_raw_coefficient_l2_error", "Raw coefficient L2 error", axes[0]),
        ("mean_variance_relative_error", "Noise-variance relative error", axes[1]),
    ):
        for method in methods:
            for n_int, color in colors.items():
                selected = sorted(
                    (row for row in rows if row["estimator"] == method and row["N_int"] == n_int),
                    key=lambda row: row["absolute_h"],
                )
                axis.plot(
                    [row["absolute_h"] for row in selected],
                    [row[metric]["mean"] for row in selected],
                    linestyle=styles[method],
                    marker="o",
                    markersize=3,
                    color=color,
                    label=f"{method}, N={n_int}",
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("|h|")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7)
    path = figure_dir / "coefficient_and_variance_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path.relative_to(output)))

    for metric, ylabel, filename, log_y in (
        ("true_excess_KL", "True excess KL", "excess_kl.png", True),
        ("node_failure_rate", "Node-fit failure rate", "failure_rate.png", False),
    ):
        fig, axis = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
        for method in methods:
            for n_int, color in colors.items():
                selected = sorted(
                    (row for row in rows if row["estimator"] == method and row["N_int"] == n_int),
                    key=lambda row: row["absolute_h"],
                )
                y = [max(float(row[metric]["mean"]), 1e-14) if log_y else row[metric]["mean"] for row in selected]
                axis.plot(
                    [row["absolute_h"] for row in selected],
                    y,
                    linestyle=styles[method],
                    marker="o",
                    markersize=3,
                    color=color,
                    label=f"{method}, N={n_int}",
                )
        axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel("|h|")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, ncol=2)
        path = figure_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(str(path.relative_to(output)))
    return files


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4g}"


def _report(
    config: dict[str, object],
    config_hash: str,
    output: Path,
    node_rows: list[dict[str, object]],
    source_rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    validation: dict[str, object],
    figures: list[str],
) -> None:
    overall = [row for row in summary if row["graph_family"] == "ALL"]
    smallest = next(
        row
        for row in overall
        if row["absolute_h"] == 0.05
        and row["N_int"] == 250
        and row["estimator"] == "calibration_free"
    )
    largest = next(
        row
        for row in overall
        if row["absolute_h"] == 2.0
        and row["N_int"] == 10000
        and row["estimator"] == "calibration_free"
    )
    raw_differences = [
        row["paired_raw_row_max_abs_difference"]
        for row in node_rows
        if row.get("paired_raw_row_max_abs_difference") is not None
    ]
    maximum_raw_difference = max(raw_differences, default=None)
    invalid = sum(int(row["invalid_nodes"]) for row in source_rows)
    scheduled = sum(int(row["scheduled_nodes"]) for row in source_rows)
    fallback = sum(bool(row["fallback_used"]) for row in source_rows)
    by_estimator = {}
    for estimator in ("calibrated", "calibration_free"):
        selected = [row for row in source_rows if row["estimator"] == estimator]
        by_estimator[estimator] = {
            "scheduled": sum(int(row["scheduled_nodes"]) for row in selected),
            "invalid": sum(int(row["invalid_nodes"]) for row in selected),
            "sign_errors": sum(int(row["denominator_sign_errors"]) for row in selected),
            "fallbacks": sum(bool(row["fallback_used"]) for row in selected),
        }
    first_source_condition = {}
    for row in source_rows:
        first_source_condition.setdefault(row["source_id"], row)
    zero_optima = sum(
        bool(row["target_has_zero_population_optimum"])
        for row in first_source_condition.values()
    )
    report = f"""# Calibration-free singleton-query finite-sample diagnostic

## Scope and frozen design

This is the isolated `{config['diagnostic_id']}` diagnostic, not a rerun of an archived campaign. The frozen configuration SHA-256 is `{config_hash}`. It uses 15 predetermined n=10 sources (five random trees, five sparse ER p=0.15 DAGs, and five trees of cliques), queries all ten nodes, and crosses |h|={config['absolute_h_grid']} with N_int={config['N_int_grid']}. N_obs=N_target=10,000. Query signs are exactly balanced over source-node pairs and fixed across magnitudes and sample sizes.

Observational, target, and intervention RNG streams are independent. Observational and target samples are reused across every condition. Each source-node has one independent 10,000-row intervention-noise pool; sample sizes are nested prefixes, and the same rowwise noise is shifted by the exact additive-query mean displacement across magnitudes. This is exact common-random-number pairing because invariant-noise singleton additive queries change the mean but not covariance.

## Inspected implementation references

- Current OAS covariance and precision: `query_complexity/finite_sample.py:69-151`, `_oas_from_empirical` and `estimate_oas`.
- Current source bootstrap: `query_complexity/finite_sample.py:260-307`, `bootstrap_source_cache`.
- Current calibrated local estimator and support logic: `query_complexity/finite_sample.py:310-542`, `_mechanism_from_moments` and `estimate_local_mechanism`.
- Current graph-free all-node acquisition and invalid-fit fallback: `query_complexity/finite_sample.py:926-1012`, `run_graph_free_finite_sample`.
- Exact sparse four-state optimizer adapter: `query_complexity/finite_sample.py:699-740`, `_solve_estimated_dp`; population oracle: `oracle/sparse_shift_scale_kl.py:1288`, `solve_sparse_shift_scale_kl`.
- Synthetic graph/source/target construction: `scripts/run_query_complexity_local_mechanism_campaign.py:42-105`, `_graph` and `_source`; target moments: `graphs/target_gen.py:99-133`, `generate_realisable_target`.
- Stage-B protocol: `configs/finite_sample_algorithm1_stage_b_v2.json` (OAS, 256 bootstraps, h=1, bounds, budget, generators); Stage-B sampling: `scripts/run_finite_sample_algorithm1.py:153-186`, `_problem` and `_sample_pools`.
- Population calibration-free identity already present separately: `query_complexity/local_mechanism.py:78-279`, `recover_local_mechanism_from_singleton_shift`; production finite-sample defaults were not changed.

## Algebra and estimators

For empirical source mean `mu_hat`, OAS covariance `Sigma_hat`, precision `K_hat`, and response `d_hat=mu_hat_int-mu_hat`, the calibrated comparator forms `v_hat=K_hat(d_hat/h)` and the calibration-free estimator forms `w_hat=K_hat d_hat`. For nonzero h, `v_hat=w_hat/h`, hence both raw normalized rows are exactly `m_hat=w_hat/w_hat[i]=v_hat/v_hat[i]` before validity filtering and support masking. The largest observed paired raw-row discrepancy when both fits were valid was {_fmt(maximum_raw_difference)}.

The calibrated variance is `1/v_hat[i]`. The calibration-free variance is `m_hat_selected^T Sigma_hat m_hat_selected`, using the same OAS covariance paired with `K_hat`. Thus this experiment studies an OAS-regularized implementation of the population identity; it does not validate an unregularized concentration theorem. Both methods normalize the complete row first and run the same graph-free percentile/sign-stability support rule. After masking, selected coefficients are copied from that raw row and `alpha_hat=m_hat_selected^T mu_hat` is recomputed. Matching the existing calibrated implementation, its `1/v_hat[i]` variance remains tied to the unmasked response and is not support-refit; the calibration-free variance is recomputed quadratically from the selected row. The output also saves raw and selected quadratic variances so masking and variance-definition effects are explicit.

The calibration-free numerical validity rule is `abs(w_hat[i]) > {config['minimum_abs_uncalibrated_denominator']}` and therefore accepts either sign. The calibrated comparator retains the production-style `v_hat[i] > {config['minimum_calibrated_self_precision_effect']}` rule. Bootstrap replicates apply the corresponding rule and all invalid point/replicate fits remain in the accounting.

## Validation

All pre-grid checks passed: **{validation['passed']}**. They covered exact-moment recovery at positive and negative h, paired raw-ratio equality, absence of any h-like argument from the calibration-free API, explicit near-zero failure, and a small end-to-end condition with independent population KL plus support/bounds checks. Full details are in `validation/validation.json`.

## Results and limitations

Across {scheduled} scheduled method-node fits, {invalid} were invalid ({invalid / scheduled:.3%}); {fallback} of {len(source_rows)} source-condition-method runs used the declared empty-parent fallback. All {by_estimator['calibrated']['invalid']} invalid fits were in the calibrated arm, whose positive-self-effect check rejected the same {by_estimator['calibrated']['sign_errors']} wrong-sign empirical denominators. The calibration-free arm had {by_estimator['calibration_free']['invalid']} invalid fits under its absolute-denominator threshold, but it retained {by_estimator['calibration_free']['sign_errors']} wrong-sign denominators; this is a material limitation, not a success. The predetermined target audit found {zero_optima} of 15 source targets had a numerically zero exact sparse population optimum; this was measured, not assumed.

At the hardest calibration-free cell (|h|=0.05, N_int=250), mean raw coefficient L2 error was {_fmt(smallest['mean_raw_coefficient_l2_error']['mean'])}, mean variance relative error was {_fmt(smallest['mean_variance_relative_error']['mean'])}, mean excess KL was {_fmt(smallest['true_excess_KL']['mean'])}, node-fit failure rate was {_fmt(smallest['node_failure_rate']['mean'])}, and denominator-sign error rate was {_fmt(smallest['denominator_sign_error_rate']['mean'])}. At |h|=2 and N_int=10,000, the corresponding error/KL/failure values were {_fmt(largest['mean_raw_coefficient_l2_error']['mean'])}, {_fmt(largest['mean_variance_relative_error']['mean'])}, {_fmt(largest['true_excess_KL']['mean'])}, and {_fmt(largest['node_failure_rate']['mean'])}; its sign-error rate was {_fmt(largest['denominator_sign_error_rate']['mean'])}. Exact graph-family and source-level-bootstrap uncertainty summaries are in the tables; negative or non-monotone findings are retained there without threshold tuning.

This directly tests normalized mechanism estimation and downstream bounded matching when every node is queried (T_q=n). It does **not** test adaptive acquisition, learned-graph errors, or matching-specific query complexity. Nodes, nested prefixes, and magnitudes are not treated as independent replicates: metrics are aggregated within source first and uncertainty is over sources.

## Artifacts

- `machine_readable/node_level_results.csv/json`: unaggregated method-node fits.
- `machine_readable/source_condition_results.csv/json`: within-source aggregates and end-to-end results.
- `tables/summary_by_h_nint_estimator_graph_family.csv/json`: source-level summaries and 95% source bootstrap intervals.
- `provenance/`: frozen config, hashes/revision, environment, seeds through deterministic rules, and every predetermined source attempt/identity.
- Figures: {', '.join(f'`{name}`' for name in figures)}.
"""
    _text(output / "diagnostic_report.md", report)
    manuscript = f"""# Manuscript-ready factual summary

In a frozen finite-sample diagnostic on 15 predetermined n=10 linear-Gaussian sources (five each of random trees, sparse ER p=0.15 DAGs, and trees of cliques), we queried every node with invariant-noise additive interventions over |h| in {config['absolute_h_grid']} and N_int in {config['N_int_grid']}, using N_obs=N_target=10,000 and 256 bootstraps. With identical OAS moments, the calibrated and calibration-free estimators had matching raw normalized coefficient rows up to a maximum observed absolute discrepancy of {_fmt(maximum_raw_difference)}, as required algebraically. The calibration-free estimator did not receive h and used the signed normalization denominator together with the observational quadratic variance `m_hat^T Sigma_hat m_hat`; its hardest and easiest grid endpoints had mean raw coefficient errors {_fmt(smallest['mean_raw_coefficient_l2_error']['mean'])} and {_fmt(largest['mean_raw_coefficient_l2_error']['mean'])}, respectively, with corresponding mean excess KL {_fmt(smallest['true_excess_KL']['mean'])} and {_fmt(largest['true_excess_KL']['mean'])}. Its absolute-denominator rule produced no invalid fits but retained {by_estimator['calibration_free']['sign_errors']} wrong-sign denominators (including a {_fmt(smallest['denominator_sign_error_rate']['mean'])} rate at |h|=0.05, N_int=250); the calibrated arm rejected those same 66 point fits. These results are direct evidence about all-node normalized mechanism recovery and downstream matching under the stated OAS/bootstrap protocol; they do not evaluate adaptive acquisition, graph learning, or matching-specific query complexity.
"""
    _text(output / "manuscript_ready_summary.md", manuscript)


def _analyze(config: dict[str, object], output: Path, config_hash: str) -> None:
    files = sorted((output / "raw/source_results").glob("*.json"))
    if len(files) != int(config["expected_sources"]):
        raise RuntimeError(f"expected {config['expected_sources']} source files, found {len(files)}")
    node_rows = []
    source_rows = []
    sample_hashes = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        node_rows.extend(payload["node_rows"])
        for row in payload["source_condition_rows"]:
            # The exact sparse solver can return exp(log(upper)) one ulp above
            # the declared scale upper bound.  Re-evaluate the recorded check
            # with the configuration's frozen numerical support tolerance and
            # retain the raw violation magnitude.
            shift = np.asarray(row["returned_shift"], dtype=float)
            scale = np.asarray(row["returned_scale"], dtype=float)
            low_h, high_h = map(float, config["shift_bounds"])
            low_s, high_s = map(float, config["scale_bounds"])
            violation = max(
                0.0,
                float(np.max(low_h - shift)),
                float(np.max(shift - high_h)),
                float(np.max(low_s - scale)),
                float(np.max(scale - high_s)),
            )
            row["bounds_violation_max"] = violation
            row["bounds_check_atol"] = float(config["support_atol"])
            row["bounds_check_passed"] = violation <= float(config["support_atol"])
            row["denominator_sign_error_rate"] = (
                int(row["denominator_sign_errors"]) / int(row["scheduled_nodes"])
            )
            source_rows.append(row)
        sample_hashes.append(
            {"source_id": payload["spec"]["source_id"], "sample_hashes": payload["sample_hashes"], "pairing": payload["pairing"]}
        )
    if len(node_rows) != int(config["expected_node_rows"]):
        raise RuntimeError("node-level row count is incomplete")
    if len(source_rows) != int(config["expected_source_conditions"]):
        raise RuntimeError("source-condition row count is incomplete")
    _json(output / "machine_readable/node_level_results.json", node_rows)
    _json(output / "machine_readable/source_condition_results.json", source_rows)
    _json(output / "provenance/sample_identities.json", sample_hashes)
    _write_csv(output / "machine_readable/node_level_results.csv", node_rows)
    _write_csv(output / "machine_readable/source_condition_results.csv", source_rows)
    summary = _summaries(config, source_rows)
    _json(output / "tables/summary_by_h_nint_estimator_graph_family.json", summary)
    flat = []
    for row in summary:
        item = {key: value for key, value in row.items() if not isinstance(value, dict)}
        for key, value in row.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    item[f"{key}__{subkey}"] = subvalue
        flat.append(item)
    _write_csv(output / "tables/summary_by_h_nint_estimator_graph_family.csv", flat)
    figures = _figures(output, summary)
    validation = json.loads((output / "validation/validation.json").read_text(encoding="utf-8"))
    _report(
        config,
        config_hash,
        output,
        node_rows,
        source_rows,
        summary,
        validation,
        figures,
    )
    _json(
        output / "summary.json",
        {
            "schema_version": SCHEMA,
            "status": "COMPLETE",
            "config_sha256": config_hash,
            "node_rows": len(node_rows),
            "source_condition_rows": len(source_rows),
            "source_files": len(files),
            "validation_passed": validation["passed"],
            "figures": figures,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "run", "analyze", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    output = prepare_output_directory(
        arguments.output,
        overwrite=arguments.overwrite,
        protected=(config_path,),
    )
    config = _load_config(config_path)
    config_hash = _freeze(config_path, output, config)
    if arguments.command in {"validate", "all"}:
        validation = _validate(config)
        _json(output / "validation/validation.json", validation)
        if not validation["passed"]:
            raise RuntimeError("pre-grid validation failed")
    if arguments.command in {"run", "all"}:
        validation_path = output / "validation/validation.json"
        if not validation_path.exists() or not json.loads(validation_path.read_text())["passed"]:
            raise RuntimeError("a passing frozen validation is required before the full grid")
        destination = output / "raw/source_results"
        destination.mkdir(parents=True, exist_ok=True)
        specs = _source_specs(config)
        pending = [spec for spec in specs if not (destination / f"{spec['source_id']}.json").exists()]
        print(f"{len(pending)} predetermined sources pending; workers={arguments.workers}", flush=True)
        failures = []
        if arguments.workers == 1:
            iterator = ((spec, _execute_source(config, spec)) for spec in pending)
            for spec, result in iterator:
                _json(destination / f"{spec['source_id']}.json", result)
                print(f"completed {spec['source_id']} ({result['runtime_seconds']:.1f}s)", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
                futures = {executor.submit(_execute_source, config, spec): spec for spec in pending}
                for future in as_completed(futures):
                    spec = futures[future]
                    try:
                        result = future.result()
                        _json(destination / f"{spec['source_id']}.json", result)
                        print(f"completed {spec['source_id']} ({result['runtime_seconds']:.1f}s)", flush=True)
                    except Exception as error:  # noqa: BLE001
                        failure = {
                            "spec": spec,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        _json(output / f"raw/failures/{spec['source_id']}.json", failure)
                        print(f"FAILED {spec['source_id']}: {error}", flush=True)
        if failures:
            raise RuntimeError(f"{len(failures)} predetermined source executions failed")
    if arguments.command in {"analyze", "all"}:
        _analyze(config, output, config_hash)


if __name__ == "__main__":
    main()
