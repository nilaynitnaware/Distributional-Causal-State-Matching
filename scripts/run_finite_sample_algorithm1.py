#!/usr/bin/env python3
"""Resumable Stage-A campaign for finite-sample Algorithm 1.

Commands: ``run``, ``resume``, ``validate``, ``analyze``, and ``benchmark``.
Every task writes one atomic JSON result; completed stable task IDs are never
duplicated.  Hidden SCM state enters only the simulation/evaluation functions
in this file and is never passed to the finite-sample main algorithm.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from environments.blackbox import EnvironmentQueryLedger, ExactMomentBlackBoxEnvironment
from graphs.interventions import ShiftScaleIntervention
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation
from oracle.sparse_shift_scale_kl import solve_sparse_shift_scale_kl
from query_complexity import FirstUnresolvedPolicy, run_query_method
from query_complexity.finite_sample import (
    FINITE_SAMPLE_PROTOCOL_VERSION,
    CachedProbeSamples,
    FiniteLocalMechanism,
    FiniteSampleAlgorithmResult,
    FiniteSampleSettings,
    estimate_oas,
    run_finite_sample_algorithm1,
    run_graph_free_finite_sample,
    run_known_dag_finite_sample,
)
from scripts.run_query_complexity_local_mechanism_campaign import _source

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/finite_sample_algorithm1_pilot_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/finite_sample_algorithm1_v1"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _digest(payload: object) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def _git(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime_provenance(config_hash: str) -> dict[str, object]:
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        torch_version = cuda_version = gpu = None
    return {
        "protocol_version": FINITE_SAMPLE_PROTOCOL_VERSION,
        "config_sha256": config_hash,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch_version,
        "cuda": cuda_version,
        "gpu": gpu,
        "cpu": platform.processor() or platform.machine(),
    }


def _task_specs(config: dict[str, object]) -> tuple[dict[str, object], ...]:
    specs = []
    for n in config["node_counts"]:
        for graph_index, regime in enumerate(config["graph_regimes"]):
            for replicate in range(int(config["replicate_seeds"])):
                for n_int in config["N_int_grid"]:
                    coordinate = {
                        "n": int(n),
                        "graph_index": graph_index,
                        "graph_family": (
                            f"{regime['family']}"
                            + (
                                f"_{regime['edge_probability']}"
                                if "edge_probability" in regime
                                else ""
                            )
                        ),
                        "replicate": replicate,
                        "N_int": int(n_int),
                    }
                    coordinate["condition_id"] = "condition-" + _digest(coordinate)[:16]
                    specs.append(coordinate)
    return tuple(specs)


def _problem(config: dict[str, object], spec: dict[str, object]):
    regime = config["graph_regimes"][int(spec["graph_index"])]
    return _source(config, regime, int(spec["n"]), int(spec["replicate"]))


def _sample_pools(config, spec, scm, target):
    problem_key = (
        int(spec["n"]), int(spec["graph_index"]), int(spec["replicate"])
    )
    source_rng = np.random.default_rng(_seed(config["base_seed"], *problem_key, "source_samples"))
    target_rng = np.random.default_rng(_seed(config["base_seed"], *problem_key, "target_samples"))
    source_samples = scm.sample_observational(int(config["N_obs"]), source_rng)
    target_samples = scm.sample_interventional(
        target.generating_intervention, int(config["N_target"]), target_rng
    )
    h = float(config["probe_shift_magnitude"])
    n = int(spec["n"])
    n_int = int(spec["N_int"])
    probes = {}
    for node in range(n):
        shift = np.zeros(n)
        shift[node] = h
        rng = np.random.default_rng(
            _seed(config["base_seed"], *problem_key, "probe", node)
        )
        probes[node] = scm.sample_interventional(
            ShiftScaleIntervention(shift, np.ones(n)), n_int, rng
        )
    fingerprints = {
        "source": _digest(source_samples),
        "target": _digest(target_samples),
        "probes": _digest({node: _digest(value) for node, value in probes.items()}),
    }
    return source_samples, target_samples, probes, fingerprints


def _coordinate_support(intervention: ShiftScaleIntervention) -> frozenset[str]:
    return frozenset(
        [f"shift:{node}" for node in np.flatnonzero(intervention.shift != 0.0)]
        + [f"scale:{node}" for node in np.flatnonzero(intervention.scale != 1.0)]
    )


def _support_jaccard(left: ShiftScaleIntervention, right: ShiftScaleIntervention) -> float:
    a, b = _coordinate_support(left), _coordinate_support(right)
    return 1.0 if not (a | b) else len(a & b) / len(a | b)


def _state_labels(intervention: ShiftScaleIntervention) -> tuple[str, ...]:
    labels = []
    for shift, scale in zip(intervention.shift, intervention.scale, strict=True):
        active_shift, active_scale = shift != 0.0, scale != 1.0
        labels.append(
            "shift+scale" if active_shift and active_scale else
            "shift-only" if active_shift else
            "scale-only" if active_scale else "none"
        )
    return tuple(labels)


def _mechanism_metrics(
    mechanisms: Iterable[FiniteLocalMechanism], scm
) -> dict[str, object]:
    mechanisms = tuple(mechanisms)
    if not mechanisms:
        return {
            "parent_precision": None, "parent_recall": None, "parent_f1": None,
            "exact_parent_support": None, "coefficient_rmse": None,
            "coefficient_relative_error": None, "coefficient_sign_accuracy": None,
            "omega_absolute_error": None, "omega_relative_error": None,
            "alpha_error": None, "mechanism_l2_error": None,
        }
    tp = fp = fn = exact = 0
    coefficient_errors = []
    coefficient_relative = []
    sign_correct = sign_total = 0
    omega_abs = []
    omega_rel = []
    alpha_abs = []
    mechanism_l2 = []
    for estimate in mechanisms:
        node = estimate.node
        true_parents = frozenset(int(item) for item in scm.dag.predecessors(node))
        estimated = frozenset(estimate.parents)
        tp += len(true_parents & estimated)
        fp += len(estimated - true_parents)
        fn += len(true_parents - estimated)
        exact += int(true_parents == estimated)
        true_beta = scm.coefficients[:, node]
        coefficient_errors.extend((estimate.incoming_coefficients - true_beta).tolist())
        denominator = float(np.linalg.norm(true_beta))
        if denominator > 0.0:
            coefficient_relative.append(
                float(np.linalg.norm(estimate.incoming_coefficients - true_beta) / denominator)
            )
        for parent in true_parents:
            sign_total += 1
            sign_correct += int(
                np.sign(estimate.incoming_coefficients[parent]) == np.sign(true_beta[parent])
            )
        true_omega = float(scm.noise_variances[node])
        if np.isfinite(estimate.noise_variance):
            omega_abs.append(abs(estimate.noise_variance - true_omega))
            omega_rel.append(abs(estimate.noise_variance - true_omega) / true_omega)
        alpha_abs.append(abs(estimate.source_intercept))
        true_m = np.eye(scm.dag.number_of_nodes())[node] - true_beta
        mechanism_l2.append(float(np.linalg.norm(estimate.mechanism_row - true_m)))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "parent_precision": precision,
        "parent_recall": recall,
        "parent_f1": f1,
        "exact_parent_support": exact / len(mechanisms),
        "coefficient_rmse": float(np.sqrt(np.mean(np.square(coefficient_errors)))),
        "coefficient_relative_error": float(np.mean(coefficient_relative)) if coefficient_relative else None,
        "coefficient_sign_accuracy": sign_correct / sign_total if sign_total else 1.0,
        "omega_absolute_error": float(np.mean(omega_abs)) if omega_abs else None,
        "omega_relative_error": float(np.mean(omega_rel)) if omega_rel else None,
        "alpha_error": float(np.mean(alpha_abs)),
        "mechanism_l2_error": float(np.mean(mechanism_l2)),
    }


def _orientation_metrics(result: FiniteSampleAlgorithmResult, dag) -> dict[str, object]:
    direct = [event for event in result.orientation_events if event.route == "direct"]
    meek = [event for event in result.orientation_events if event.route == "meek"]
    correct = lambda event: dag.has_edge(event.source, event.target)
    wrong_direct = sum(not correct(event) for event in direct)
    wrong_meek = sum(not correct(event) for event in meek)
    ordered = list(result.orientation_events)
    first_wrong = next((index for index, event in enumerate(ordered) if not correct(event)), None)
    downstream = (
        sum(not correct(event) for event in ordered[first_wrong + 1 :])
        if first_wrong is not None else 0
    )
    final_edges = result.final_graph.directed_edges if result.final_graph is not None else frozenset()
    wrong_final = sum(not dag.has_edge(*edge) for edge in final_edges)
    return {
        "direct_orientations": len(direct),
        "direct_orientation_accuracy": (
            (len(direct) - wrong_direct) / len(direct) if direct else 1.0
        ),
        "wrong_direct_orientations": wrong_direct,
        "meek_orientations": len(meek),
        "meek_orientation_accuracy": (
            (len(meek) - wrong_meek) / len(meek) if meek else 1.0
        ),
        "wrong_meek_orientations": wrong_meek,
        "first_wrong_orientation_index": first_wrong,
        "downstream_wrong_orientations": downstream,
        "directed_shd": wrong_final + result.unresolved_edges,
    }


def _mechanism_details(mechanisms: Iterable[FiniteLocalMechanism]) -> list[dict[str, object]]:
    details = []
    for item in mechanisms:
        details.append({
            "node": item.node,
            "parents": item.parents,
            "incoming_coefficients": item.incoming_coefficients,
            "mechanism_row": item.mechanism_row,
            "total_effect_column": item.total_effect_column,
            "noise_variance": item.noise_variance,
            "source_intercept": item.source_intercept,
            "valid": item.valid,
            "invalid_reason": item.invalid_reason,
            "null_threshold": item.null_threshold,
            "bootstrap_valid_replicates": item.bootstrap_valid_replicates,
            "evidence": [asdict(evidence) for evidence in item.evidence],
        })
    return details


def _result_row(
    *, config, spec, method, result, scm, target, oracle, population_t,
    runtime, fingerprints, problem_id, target_id,
) -> dict[str, object]:
    initial_mean, initial_covariance = scm.observational_moments()
    final_mean, final_covariance = scm.interventional_moments(result.intervention)
    initial_true = gaussian_kl_target_to_candidate(
        target.mean, target.covariance, initial_mean, initial_covariance
    )
    final_true = gaussian_kl_target_to_candidate(
        target.mean, target.covariance, final_mean, final_covariance
    )
    states = _state_labels(result.intervention)
    optimal = ShiftScaleIntervention(oracle.budgeted_shift, oracle.budgeted_scale)
    row = {
        **spec,
        "protocol_version": FINITE_SAMPLE_PROTOCOL_VERSION,
        "problem_id": problem_id,
        "target_id": target_id,
        "information_regime": method,
        "method": method,
        "backend": config["backend"],
        "device": "cpu",
        "N_obs": int(config["N_obs"]),
        "N_target": int(config["N_target"]),
        "T_q": result.physical_query_count,
        "T_q_over_n": result.physical_query_count / int(spec["n"]),
        "population_T_q": population_t,
        "T_q_minus_population": result.physical_query_count - population_t,
        "total_intervention_observations": result.total_intervention_observations,
        "total_observations": result.total_observations,
        "initial_true_KL": initial_true,
        "final_true_KL": final_true,
        "K_B_star": oracle.raw_K_B_star,
        "excess_KL": final_true - oracle.raw_K_B_star,
        "support_jaccard": _support_jaccard(result.intervention, optimal),
        "coordinate_state_accuracy": float(np.mean(np.array(states) == np.array(_state_labels(optimal)))),
        "runtime_seconds": runtime,
        "termination_reason": result.termination_reason,
        "terminated": result.termination_reason in {"all_local_objectives_estimated", "all_nodes_probed", "known_parent_sets"},
        "unresolved_edges": result.unresolved_edges,
        "invalid_estimates": result.invalid_estimates,
        "oas_shrinkage": result.source_estimate.shrinkage,
        "covariance_condition_number": result.source_estimate.covariance_condition_number,
        "precision_condition_number": result.source_estimate.precision_condition_number,
        "minimum_covariance_eigenvalue": result.source_estimate.minimum_eigenvalue,
        "numerical_inversion_failure": result.source_estimate.inversion_failed,
        "target_oas_shrinkage": result.target_estimate.shrinkage,
        "sample_fingerprints": fingerprints,
        "queried_nodes": result.queried_nodes,
        "returned_shift": result.intervention.shift,
        "returned_scale": result.intervention.scale,
        "coordinate_states": states,
        "mechanism_details": _mechanism_details(result.mechanisms),
        **_mechanism_metrics(result.mechanisms, scm),
        **_orientation_metrics(result, scm.dag),
    }
    row["task_id"] = f"{spec['condition_id']}:{method}"
    return row


def _run_condition(config, spec) -> list[dict[str, object]]:
    scm, target = _problem(config, spec)
    source_samples, target_samples, probes, fingerprints = _sample_pools(
        config, spec, scm, target
    )
    n = int(spec["n"])
    budget = min(6, n)
    shift_bounds = tuple(float(value) for value in config["shift_bounds"])
    scale_bounds = tuple(float(value) for value in config["scale_bounds"])
    oracle = solve_sparse_shift_scale_kl(
        scm=scm,
        target_mean=target.mean,
        target_covariance=target.covariance,
        budget=budget,
        shift_bounds=shift_bounds,
        log_scale_bounds=tuple(float(np.log(value)) for value in scale_bounds),
    )
    exact_mean, exact_covariance = scm.observational_moments()
    cpdag = cpdag_from_true_dag_for_simulation(scm.dag)
    population = run_query_method(
        method="public_adaptive_local_mechanism_dp",
        environment=ExactMomentBlackBoxEnvironment(
            scm, instance_id=str(spec["condition_id"]), ledger=EnvironmentQueryLedger(None)
        ),
        source_mean=exact_mean,
        source_covariance=exact_covariance,
        target_mean=target.mean,
        target_covariance=target.covariance,
        budget_B=budget,
        experimental_sparsity_S=1,
        shift_bounds=shift_bounds,
        scale_bounds=scale_bounds,
        initial_graph=cpdag,
        singleton_shift_policy=FirstUnresolvedPolicy(),
    )
    settings = FiniteSampleSettings(
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        confidence_level=float(config["confidence_level"]),
        orientation_confidence=float(config["orientation_confidence"]),
        null_quantile=float(config["null_quantile"]),
        support_probability_threshold=float(config["support_probability_threshold"]),
        backend=str(config["backend"]),
        device=str(config.get("device", "cpu")),
    )
    problem_id = f"population:n{n}:g{spec['graph_index']}:r{spec['replicate']}"
    target_id = problem_id + ":canonical_target"
    common = {
        "config": config, "spec": spec, "scm": scm, "target": target, "oracle": oracle,
        "population_t": population.counts.total_query_count, "fingerprints": fingerprints,
        "problem_id": problem_id, "target_id": target_id,
    }
    rows = []
    start = perf_counter()
    adaptive = run_finite_sample_algorithm1(
        source_samples=source_samples,
        target_samples=target_samples,
        probe_provider=CachedProbeSamples(probes, int(spec["N_int"])),
        initial_graph=cpdag,
        shift_magnitude=float(config["probe_shift_magnitude"]),
        budget=budget,
        shift_bounds=shift_bounds,
        scale_bounds=scale_bounds,
        settings=settings,
        bootstrap_seed=_seed(spec["condition_id"], "bootstrap"),
        singleton_shift_policy=FirstUnresolvedPolicy(),
    )
    rows.append(_result_row(method="finite_sample_algorithm1", result=adaptive, runtime=perf_counter() - start, **common))

    start = perf_counter()
    graph_free = run_graph_free_finite_sample(
        source_samples=source_samples,
        target_samples=target_samples,
        probe_provider=CachedProbeSamples(probes, int(spec["N_int"])),
        shift_magnitude=float(config["probe_shift_magnitude"]),
        budget=budget,
        shift_bounds=shift_bounds,
        scale_bounds=scale_bounds,
        settings=settings,
        bootstrap_seed=_seed(spec["condition_id"], "bootstrap"),
    )
    rows.append(_result_row(method="graph_free_finite_sample", result=graph_free, runtime=perf_counter() - start, **common))

    start = perf_counter()
    known = run_known_dag_finite_sample(
        source_samples=source_samples,
        target_samples=target_samples,
        parent_sets={node: frozenset(scm.dag.predecessors(node)) for node in range(n)},
        budget=budget,
        shift_bounds=shift_bounds,
        scale_bounds=scale_bounds,
    )
    rows.append(_result_row(method="known_dag_finite_sample", result=known, runtime=perf_counter() - start, **common))

    # Population row uses an API-compatible zero-sample accounting shell while
    # retaining the authoritative exact intervention and query count.
    population_shell = FiniteSampleAlgorithmResult(
        method="finite_sample_algorithm1",
        intervention=ShiftScaleIntervention(population.shift, population.scale),
        dp_result=population.dp_result,
        source_estimate=estimate_oas(source_samples),
        target_estimate=estimate_oas(target_samples),
        mechanisms=(), final_graph=population.final_graph, orientation_events=(),
        queried_nodes=population.queried_nodes,
        physical_query_count=population.counts.total_query_count,
        total_intervention_observations=0, total_observations=0,
        unresolved_edges=len(population.final_graph.undirected_edges), invalid_estimates=0,
        termination_reason="population_reference", estimated_initial_kl=population.initial_kl,
    )
    row = _result_row(method="population_algorithm1", result=population_shell, runtime=population.runtime_seconds, **common)
    row.update(total_intervention_observations=0, total_observations=0, terminated=True)
    rows.append(row)
    return rows


def _layout(output: Path) -> None:
    for directory in (
        "config", "pilot/results", "pilot/failed", "campaign", "tables",
        "figures", "logs", "machine_readable", "reports"
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)


def preregister(config_path: Path, output: Path) -> dict[str, object]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["protocol_version"] != FINITE_SAMPLE_PROTOCOL_VERSION:
        raise ValueError("configuration protocol is stale")
    _layout(output)
    frozen = output / "config/pilot_config.json"
    if frozen.exists() and frozen.read_bytes() != config_bytes:
        raise RuntimeError("refusing to replace a different frozen pilot config")
    if not frozen.exists():
        frozen.write_bytes(config_bytes)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    manifest = {
        "campaign_id": config["campaign_id"],
        "protocol_version": FINITE_SAMPLE_PROTOCOL_VERSION,
        "config_sha256": config_hash,
        "status": "preregistered",
        "task_conditions": len(_task_specs(config)),
        "expected_rows": len(_task_specs(config)) * len(config["methods"]),
        "resume_semantics": "one atomic JSON per stable condition; skip complete files",
        "provenance": _runtime_provenance(config_hash),
    }
    manifest_path = output / "machine_readable/manifest.json"
    if not manifest_path.exists():
        _atomic_json(manifest_path, manifest)
    prereg = f"""# Finite-sample Algorithm 1 Stage-A preregistration

Frozen before execution. Configuration SHA-256: `{config_hash}`.

The pilot contains {len(_task_specs(config))} paired sample conditions and {manifest['expected_rows']} result rows: n in {config['node_counts']}, all five population graph regimes, {config['replicate_seeds']} seeds per n/family, N_obs=N_target=10,000, and N_int in {config['N_int_grid']}. OAS is fit once per observational dataset and reused. Each queried mechanism uses {config['bootstrap_replicates']} deterministic nonparametric bootstrap replicates, graph-neighbor candidate support, known non-neighbors as an empirical null, a {config['confidence_level']:.0%} percentile interval, and a predeclared {config['orientation_confidence']:.0%} orientation gate. Low-confidence edges remain unoriented.

The gate is exactly the JSON `gate` object in the frozen configuration. Stage B is not authorized by this runner and is not launched automatically.
"""
    prereg_path = output / "pilot_preregistration.md"
    if not prereg_path.exists():
        prereg_path.write_text(prereg, encoding="utf-8")
    return config


def run(config_path: Path, output: Path, *, resume: bool) -> None:
    config = preregister(config_path, output)
    results_dir = output / "pilot/results"
    for index, spec in enumerate(_task_specs(config), start=1):
        path = results_dir / f"{spec['condition_id']}.json"
        if path.exists():
            if resume:
                continue
            raise FileExistsError(f"result already exists: {path}; use resume")
        try:
            rows = _run_condition(config, spec)
            _atomic_json(path, {"spec": spec, "rows": rows, "status": "complete"})
            print(f"[{index}/{len(_task_specs(config))}] {spec['condition_id']}", flush=True)
        except BaseException as error:
            _atomic_json(
                output / "pilot/failed" / f"{spec['condition_id']}.json",
                {"spec": spec, "error_type": type(error).__name__, "error": str(error)},
            )
            raise


def _rows(output: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted((output / "pilot/results").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete":
            rows.extend(payload["rows"])
    return rows


def _validation_report(config: dict[str, object], output: Path) -> dict[str, object]:
    expected = {spec["condition_id"] for spec in _task_specs(config)}
    observed = {path.stem for path in (output / "pilot/results").glob("*.json")}
    rows = _rows(output)
    task_ids = [row["task_id"] for row in rows]
    report = {
        "expected_conditions": len(expected), "complete_conditions": len(observed & expected),
        "missing_conditions": sorted(expected - observed),
        "unexpected_conditions": sorted(observed - expected),
        "expected_rows": int(config["expected_result_rows"]), "observed_rows": len(rows),
        "duplicate_task_ids": len(task_ids) - len(set(task_ids)),
        "failed_jobs": len(list((output / "pilot/failed").glob("*.json"))),
        "complete": observed == expected and len(rows) == int(config["expected_result_rows"]) and len(task_ids) == len(set(task_ids)),
    }
    return report


def validation_report(config_path: Path, output: Path) -> dict[str, object]:
    """Read-only completeness check used by separate-root Stage-B analysis."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["protocol_version"] != FINITE_SAMPLE_PROTOCOL_VERSION:
        raise ValueError("configuration protocol is stale")
    return _validation_report(config, output)


def _write_results_csv(rows: list[dict[str, object]], output: Path) -> None:
    """Materialize the existing row serializer's authoritative CSV interface."""
    keys = sorted({key for row in rows for key in row if key not in {"mechanism_details", "sample_fingerprints"}})
    with (output / "machine_readable/results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key)) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in keys})


def validate(config_path: Path, output: Path) -> dict[str, object]:
    config = preregister(config_path, output)
    report = _validation_report(config, output)
    _atomic_json(output / "machine_readable/validation.json", report)
    if report["complete"] and config.get("campaign_id") == "finite_sample_algorithm1_v1_stage_b_v2":
        _write_results_csv(_rows(output), output)
    print(json.dumps(report, indent=2))
    return report


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(20260815)
    means = [float(np.mean(rng.choice(values, len(values), replace=True))) for _ in range(2000)]
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def analyze(
    config_path: Path,
    output: Path,
    *,
    analysis_output: Path | None = None,
    legacy_comparison_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    if config.get("campaign_id") == "finite_sample_algorithm1_v1_stage_b_v2":
        from scripts.stage_b_analysis import analyze_stage_b

        if analysis_output is None:
            raise ValueError("Stage-B analysis requires --analysis-output distinct from --output")
        validation = validation_report(config_path, output)
        if not validation["complete"]:
            raise RuntimeError("Stage-B analysis requires a complete validated pilot")
        return analyze_stage_b(
            config_path,
            analysis_output,
            input_dir=output,
            legacy_comparison_dir=legacy_comparison_dir,
            overwrite=overwrite,
        )
    validation = validate(config_path, output)
    if not validation["complete"]:
        raise RuntimeError("analysis requires a complete validated pilot")
    rows = _rows(output)
    _write_results_csv(rows, output)
    registry_keys = ["problem_id", "replicate", "graph_family", "n", "target_id", "information_regime", "N_obs", "N_target", "N_int", "method", "backend", "task_id"]
    with (output / "machine_readable/registry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=registry_keys + ["result_file", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in registry_keys}, "result_file": f"pilot/results/{row['condition_id']}.json", "status": "complete"})
    main = [row for row in rows if row["method"] == "finite_sample_algorithm1"]
    grouped = []
    for n_int in config["N_int_grid"]:
        subset = [row for row in main if row["N_int"] == n_int]
        record = {"N_int": n_int, "count": len(subset)}
        for key in ("parent_f1", "mechanism_l2_error", "final_true_KL", "excess_KL", "T_q_over_n", "wrong_direct_orientations", "wrong_meek_orientations", "total_observations"):
            values = [float(row[key]) for row in subset if row.get(key) is not None]
            record[key] = float(np.mean(values))
            record[key + "_median"] = float(np.median(values))
            record[key + "_ci95"] = _ci(values)
        grouped.append(record)
    low, high = grouped[0], grouped[-1]
    total_orientations_high = sum(
        row["direct_orientations"] + row["meek_orientations"]
        for row in main if row["N_int"] == config["N_int_grid"][-1]
    )
    wrong_high = sum(
        row["wrong_direct_orientations"] + row["wrong_meek_orientations"]
        for row in main if row["N_int"] == config["N_int_grid"][-1]
    )
    high_wrong_rate = wrong_high / total_orientations_high if total_orientations_high else 0.0
    scientific = {
        "parent_f1": high["parent_f1"] - low["parent_f1"] >= config["gate"]["parent_f1_high_minus_low_min"],
        "mechanism_error": high["mechanism_l2_error"] < low["mechanism_l2_error"],
        "excess_kl": high["excess_KL"] < low["excess_KL"],
        "query_gap": _mean([row for row in main if row["N_int"] == config["N_int_grid"][-1]], "T_q_minus_population") <= _mean([row for row in main if row["N_int"] == config["N_int_grid"][0]], "T_q_minus_population"),
        "orientation_safety": high_wrong_rate <= config["gate"]["high_nint_wrong_orientation_rate_max"],
    }
    termination = float(np.mean([row["terminated"] for row in main]))
    n40_runtime = float(np.median([row["runtime_seconds"] for row in main if row["n"] == 40]))
    passed = termination >= config["gate"]["termination_rate"] and n40_runtime <= config["gate"]["n40_median_runtime_seconds_max"] and sum(scientific.values()) >= 4
    summary = {
        "stage_a_verdict": "PASS" if passed else "FAIL",
        "grouped_main_method": grouped,
        "termination_rate": termination,
        "n40_median_runtime_seconds": n40_runtime,
        "high_nint_wrong_orientation_rate": high_wrong_rate,
        "scientific_gate_components": scientific,
        "baselines": {
            method: {str(n_int): {"final_true_KL": _mean([row for row in rows if row["method"] == method and row["N_int"] == n_int], "final_true_KL"), "excess_KL": _mean([row for row in rows if row["method"] == method and row["N_int"] == n_int], "excess_KL"), "T_q_over_n": _mean([row for row in rows if row["method"] == method and row["N_int"] == n_int], "T_q_over_n")} for n_int in config["N_int_grid"]}
            for method in config["methods"]
        },
    }
    _atomic_json(output / "tables/pilot_summary.json", summary)
    _write_figures(grouped, output)
    _write_handoff(config, summary, rows, output)
    manifest_path = output / "machine_readable/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status="stage_a_complete", stage_a_verdict=summary["stage_a_verdict"])
    _atomic_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2))
    return summary


def _write_figures(grouped, output):
    import matplotlib.pyplot as plt

    plots = (
        ("parent_f1", "Parent F1"), ("mechanism_l2_error", "Mechanism L2 error"),
        ("final_true_KL", "Final true KL divergence"), ("excess_KL", "Excess KL"),
        ("T_q_over_n", "T_q / n"), ("total_observations", "Total observations"),
    )
    x = [row["N_int"] for row in grouped]
    for key, label in plots:
        figure, axis = plt.subplots(figsize=(5, 3.5))
        axis.plot(x, [row[key] for row in grouped], marker="o")
        axis.set_xscale("log")
        axis.set_xlabel("N_int")
        axis.set_ylabel(label)
        figure.tight_layout()
        figure.savefig(output / "figures" / f"pilot__nint_vs_{key}.png", dpi=160)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(5, 3.5))
    axis.plot(x, [row["wrong_direct_orientations"] for row in grouped], marker="o", label="direct")
    axis.plot(x, [row["wrong_meek_orientations"] for row in grouped], marker="o", label="Meek")
    axis.set_xscale("log"); axis.set_xlabel("N_int"); axis.set_ylabel("Mean wrong orientations"); axis.legend()
    figure.tight_layout(); figure.savefig(output / "figures/pilot__nint_vs_wrong_orientations.png", dpi=160); plt.close(figure)


def _write_handoff(config, summary, rows, output):
    grouped = summary["grouped_main_method"]
    table = "\n".join(
        f"| {item['N_int']} | {item['parent_f1']:.4f} | {item['mechanism_l2_error']:.4f} | {item['final_true_KL']:.4f} | {item['excess_KL']:.4f} | {item['T_q_over_n']:.4f} |"
        for item in grouped
    )
    handoff = f"""# Pilot handoff for manuscript: finite-sample Algorithm 1

## Scientific question and exact procedure

This Stage-A pilot asks whether the manuscript's primary adaptive local-mechanism Algorithm 1 remains useful with sample-estimated source/target moments and singleton-shift responses. The algorithm fits OAS once to source observations; estimates `t_i=(mean_int-mean_obs)/h`, `v_i=K_hat t_i`, `omega_i=1/v_i[i]`, and `m_i=omega_i v_i`; bootstraps source and intervention observations; restricts possible parents to the public CPDAG neighborhood; gates direct orientations at {config['orientation_confidence']:.0%}; applies Meek closure; estimates newly graph-resolved conditionals analytically; constructs estimated local shift-scale KL objectives; and calls the unchanged exact four-state O(nB) DP. Hidden SCM information and exact target moments are used only after return for evaluation.

## Frozen design

- Verdict: **Stage A {summary['stage_a_verdict']}**.
- n: {config['node_counts']}; graph families: five population campaign families; paired source problems: {config['expected_source_problems']}.
- N_obs=N_target={config['N_obs']}; N_int={config['N_int_grid']}; total rows={len(rows)}.
- Bootstrap replicates={config['bootstrap_replicates']}; CI={config['confidence_level']}; empirical-null quantile={config['null_quantile']}.
- Baselines: population Algorithm 1, graph-free finite-sample identification, and known-DAG finite-sample reference.

## Main finite-sample Algorithm 1 results

| N_int | parent F1 | mechanism L2 | final true KL | excess KL | T_q/n |
|---:|---:|---:|---:|---:|---:|
{table}

Termination rate was {summary['termination_rate']:.3f}; high-N_int wrong-orientation rate was {summary['high_nint_wrong_orientation_rate']:.4f}; median n=40 runtime was {summary['n40_median_runtime_seconds']:.3f} seconds. Gate components: `{json.dumps(summary['scientific_gate_components'], sort_keys=True)}`.

## Authoritative artifacts and figures

- `machine_readable/results.csv`: all row-level metrics and sample/query accounting.
- `machine_readable/registry.csv`: immutable problem/result registry.
- `tables/pilot_summary.json`: grouped means, medians, bootstrap CIs, baselines, and gate.
- Figures: `figures/pilot__nint_vs_parent_f1.png`, `pilot__nint_vs_mechanism_l2_error.png`, `pilot__nint_vs_final_true_KL.png`, `pilot__nint_vs_excess_KL.png`, `pilot__nint_vs_T_q_over_n.png`, `pilot__nint_vs_total_observations.png`, and `pilot__nint_vs_wrong_orientations.png`.

## Claims and limitations

Supported claims are limited to the frozen paired simulation distribution and trends recorded above. This pilot does not prove finite-sample consistency, does not license calling target-to-candidate KL “reverse KL,” and does not establish superiority to Public Greedy. Stage B should be launched only if the verdict is PASS; if FAIL, inspect row-level support/orientation failures and revise the estimator in a newly versioned preregistration.

The exact Results/Discussion wording should be generated from `tables/pilot_summary.json`; no rounded number in this handoff supersedes that JSON.
"""
    (output / "reports/pilot_handoff_for_manuscript.md").write_text(handoff, encoding="utf-8")


def benchmark(output: Path) -> None:
    import torch

    _layout(output)
    records = []
    for sample_count in (2000, 10000):
        rng = np.random.default_rng(8145 + sample_count)
        samples = rng.normal(size=(sample_count, 40))
        indices = rng.integers(0, sample_count, size=(16, sample_count))
        start = perf_counter()
        numpy_means = samples[indices].mean(axis=1)
        numpy_cov = np.stack([np.cov(samples[index], rowvar=False, bias=True) for index in indices[:8]])
        records.append({"backend": "numpy", "device": "cpu", "N": sample_count, "runtime_seconds": perf_counter() - start, "peak_gpu_memory_bytes": 0, "agreement_max_abs": 0.0})
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            tensor = torch.as_tensor(samples, dtype=torch.float64, device=device)
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            start = perf_counter()
            selected = tensor[index_tensor]
            means = selected.mean(dim=1)
            centered = selected[:8] - selected[:8].mean(dim=1, keepdim=True)
            covariances = centered.transpose(1, 2) @ centered / sample_count
            if device == "cuda": torch.cuda.synchronize()
            elapsed = perf_counter() - start
            agreement = max(float(np.max(np.abs(means.cpu().numpy() - numpy_means))), float(np.max(np.abs(covariances.cpu().numpy() - numpy_cov))))
            records.append({"backend": "torch", "device": device, "N": sample_count, "runtime_seconds": elapsed, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0, "agreement_max_abs": agreement})
    _atomic_json(output / "machine_readable/backend_benchmark.json", {"records": records, "selected_backend": "numpy", "reason": "tiny n=40 OAS/DP algebra remains CPU NumPy; CUDA is selected only if end-to-end resident bootstrap evidence is materially faster"})
    print(json.dumps(records, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run", "resume", "validate", "analyze", "benchmark"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analysis-output", type=Path)
    parser.add_argument("--legacy-comparison-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "preregister": preregister(arguments.config, arguments.output)
    elif arguments.command == "run": run(arguments.config, arguments.output, resume=False)
    elif arguments.command == "resume": run(arguments.config, arguments.output, resume=True)
    elif arguments.command == "validate": validate(arguments.config, arguments.output)
    elif arguments.command == "analyze": analyze(
        arguments.config, arguments.output,
        analysis_output=arguments.analysis_output,
        legacy_comparison_dir=arguments.legacy_comparison_dir,
        overwrite=arguments.overwrite,
    )
    else: benchmark(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
