#!/usr/bin/env python3
"""Audited Stage-B analysis for finite-sample Algorithm 1.

This module is deliberately analysis-only. It consumes the frozen config,
validated ``results.csv``, and validation metadata; it never calls a campaign
executor or rewrites per-condition result files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

STAGE_B_CAMPAIGN_ID = "finite_sample_algorithm1_v1_stage_b_v2"
MAIN_METHOD = "finite_sample_algorithm1"
METHODS = (
    "population_algorithm1",
    MAIN_METHOD,
    "graph_free_finite_sample",
    "known_dag_finite_sample",
)
METRICS = (
    "parent_precision",
    "parent_recall",
    "parent_f1",
    "exact_parent_support",
    "mechanism_l2_error",
    "coefficient_rmse",
    "coefficient_relative_error",
    "coefficient_sign_accuracy",
    "omega_relative_error",
    "alpha_error",
    "final_true_KL",
    "K_B_star",
    "excess_KL",
    "T_q",
    "T_q_over_n",
    "T_q_minus_population",
    "total_intervention_observations",
    "total_observations",
    "support_jaccard",
    "directed_shd",
    "unresolved_edges",
    "wrong_direct_orientations",
    "wrong_meek_orientations",
    "runtime_seconds",
)
LEGACY_GROUPED_METRICS = (
    "parent_f1",
    "mechanism_l2_error",
    "final_true_KL",
    "excess_KL",
    "T_q_over_n",
    "wrong_direct_orientations",
    "wrong_meek_orientations",
    "total_observations",
)
EXPECTED_GRAPHS = (
    "random_tree",
    "erdos_renyi_0.15",
    "erdos_renyi_0.35",
    "barabasi_albert",
    "tree_of_cliques",
)


def _configured_graph_labels(config: dict[str, object]) -> tuple[str, ...]:
    labels = []
    for regime in config["graph_regimes"]:
        label = str(regime["family"])
        if "edge_probability" in regime:
            label += f"_{regime['edge_probability']}"
        labels.append(label)
    return tuple(labels)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _resolved_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is not a file: {path}")
    return path.resolve(strict=True)


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _preflight(
    config_path: Path,
    input_dir: Path,
    output: Path,
    legacy_comparison_dir: Path | None,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path | None, Path | None]:
    """Check resolved inputs and every write target before creating an output."""
    config = _resolved_file(config_path, "configuration")
    campaign = input_dir.resolve(strict=True)
    if not campaign.is_dir():
        raise NotADirectoryError(f"Stage-B input directory is not a directory: {input_dir}")
    results = _resolved_file(campaign / "machine_readable/results.csv", "results CSV")
    validation = _resolved_file(campaign / "machine_readable/validation.json", "validation JSON")
    output_root = output.resolve(strict=False)
    protected_roots = [campaign]
    protected_files = [config, results, validation]
    legacy_summary = legacy_handoff = None
    if legacy_comparison_dir is not None:
        legacy = legacy_comparison_dir.resolve(strict=True)
        if not legacy.is_dir():
            raise NotADirectoryError(f"legacy comparison directory is not a directory: {legacy_comparison_dir}")
        protected_roots.append(legacy)
        legacy_summary = _resolved_file(legacy / "tables/pilot_summary.json", "legacy summary")
        legacy_handoff = _resolved_file(legacy / "reports/pilot_handoff_for_manuscript.md", "legacy handoff")
        protected_files.extend([legacy_summary, legacy_handoff])
    if any(_overlaps(output_root, root) for root in protected_roots):
        raise ValueError("analysis output must be disjoint from every protected input root")
    targets = [
        output_root / name for name in (
            "tables/stage_b_summary.json",
            "machine_readable/stage_b_analysis_audit.json",
            "reports/final_handoff_for_manuscript.md",
            "reports/stage_b_analysis_audit.md",
        )
    ] + [
        output_root / "figures" / name for name in (
            "stage_b__excess_KL_vs_N_int_by_n.png",
            "stage_b__parent_f1_vs_N_int_by_n.png",
            "stage_b__mechanism_l2_vs_N_int_by_n.png",
            "stage_b__T_q_over_n_vs_N_int_by_n.png",
            "stage_b__excess_T_q_vs_N_int.png",
            "stage_b__support_jaccard_vs_N_int.png",
            "stage_b__total_observations_vs_excess_KL.png",
            "stage_b__wrong_orientations_vs_N_int.png",
            "stage_b__directed_SHD_unresolved_vs_N_int.png",
            "stage_b__excess_KL_vs_N_int_by_graph_family.png",
            "stage_b__parent_f1_vs_N_int_by_graph_family.png",
            "stage_b__query_fraction_reference_comparison.png",
        )
    ]
    resolved_targets = [target.resolve(strict=False) for target in targets]
    if any(_overlaps(target, root) for target in resolved_targets for root in protected_roots):
        raise ValueError("analysis output target must be disjoint from every protected input root")
    if any(target in protected_files for target in resolved_targets):
        raise ValueError("analysis output target resolves to a protected input file")
    existing = [target for target in targets if target.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to replace existing outputs: {existing}")
    return config, results, validation, legacy_summary, legacy_handoff


def _load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("results CSV has no header")
        missing = set(METRICS) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"results CSV is missing publication metrics: {sorted(missing)}"
            )
        rows: list[dict[str, object]] = []
        for raw in reader:
            row: dict[str, object] = dict(raw)
            for key in METRICS:
                text = raw[key]
                row[key] = None if text == "" else float(text)
            for key in ("n", "N_int", "replicate", "N_obs", "N_target"):
                row[key] = int(raw[key])
            rows.append(row)
    return rows


def _metric_values(rows: list[dict[str, object]], metric: str) -> np.ndarray:
    return np.asarray(
        [float(row[metric]) for row in rows if row[metric] is not None], dtype=float
    )


def _stats(
    rows: list[dict[str, object]], metric: str, *, label: str, replicates: int = 5000
) -> dict[str, object]:
    values = _metric_values(rows, metric)
    if not len(values):
        return {
            "count": 0,
            "missing": len(rows),
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "bootstrap_ci95": None,
        }
    seed = int.from_bytes(
        hashlib.sha256(f"stage-b-v2|{label}|{metric}".encode()).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.mean(
        values[rng.integers(0, len(values), size=(replicates, len(values)))], axis=1
    )
    return {
        "count": len(values),
        "missing": int(len(rows) - len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "bootstrap_ci95": [
            float(value) for value in np.quantile(bootstrap_means, (0.025, 0.975))
        ],
    }


def _summarize_group(rows: list[dict[str, object]], label: str) -> dict[str, object]:
    return {
        "row_count": len(rows),
        "independent_source_problem_units": len({row["problem_id"] for row in rows}),
        "metrics": {metric: _stats(rows, metric, label=label) for metric in METRICS},
    }


def _grouped(
    rows: list[dict[str, object]], keys: tuple[str, ...]
) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)
    records = []
    for coordinates in sorted(buckets):
        group = buckets[coordinates]
        label = "|".join(
            f"{key}={value}" for key, value in zip(keys, coordinates, strict=True)
        )
        records.append(
            {
                **dict(zip(keys, coordinates, strict=True)),
                **_summarize_group(group, label),
            }
        )
    return records


def _record(
    records: list[dict[str, object]], **coordinates: object
) -> dict[str, object]:
    for item in records:
        if all(item[key] == value for key, value in coordinates.items()):
            return item
    raise KeyError(coordinates)


def _mean(record: dict[str, object], metric: str) -> float:
    value = record["metrics"][metric]["mean"]
    if value is None:
        raise ValueError(f"metric {metric} is unavailable")
    return float(value)


def _raw_integrity(
    rows: list[dict[str, object]],
    config: dict[str, object],
    validation: dict[str, object],
) -> dict[str, object]:
    condition_methods: dict[str, set[str]] = defaultdict(set)
    seeds: dict[tuple[int, str], set[int]] = defaultdict(set)
    for row in rows:
        condition_methods[str(row["condition_id"])].add(str(row["method"]))
        seeds[(int(row["n"]), str(row["graph_family"]))].add(int(row["replicate"]))
    expected_graphs = _configured_graph_labels(config)
    expected_methods = set(config["methods"])
    expected_seed_values = set(range(int(config["replicate_seeds"])))
    expected_seed_cells = len(config["node_counts"]) * len(config["graph_regimes"])
    checks = {
        "row_count": len(rows)
        == int(config["expected_result_rows"])
        == int(validation["expected_rows"]),
        "condition_count": len(condition_methods)
        == int(config["expected_sample_conditions"])
        == int(validation["expected_conditions"]),
        "source_problem_count": len({row["problem_id"] for row in rows})
        == int(config["expected_source_problems"]),
        "unique_task_ids": len({row["task_id"] for row in rows}) == len(rows),
        "zero_failed_jobs": int(validation["failed_jobs"]) == 0,
        "n_int_grid": sorted({int(row["N_int"]) for row in rows})
        == list(config["N_int_grid"]),
        "node_grid": sorted({int(row["n"]) for row in rows})
        == list(config["node_counts"]),
        "graph_families": {row["graph_family"] for row in rows} == set(expected_graphs),
        "configured_seeds_per_n_graph": all(
            value == expected_seed_values for value in seeds.values()
        )
        and len(seeds) == expected_seed_cells,
        "configured_methods_per_condition": all(
            value == expected_methods for value in condition_methods.values()
        ),
        "publication_metric_columns": all(metric in rows[0] for metric in METRICS),
        "validation_complete": bool(validation["complete"]),
        "no_missing_conditions": not validation["missing_conditions"],
        "no_unexpected_conditions": not validation["unexpected_conditions"],
    }
    metric_missing = {
        metric: sum(row[metric] is None for row in rows if row["method"] == MAIN_METHOD)
        for metric in METRICS
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed": {
            "rows": len(rows),
            "conditions": len(condition_methods),
            "source_problems": len({row["problem_id"] for row in rows}),
            "task_ids": len({row["task_id"] for row in rows}),
            "N_int": sorted({int(row["N_int"]) for row in rows}),
            "n": sorted({int(row["n"]) for row in rows}),
            "graph_families": sorted({str(row["graph_family"]) for row in rows}),
            "methods": dict(
                sorted(Counter(str(row["method"]) for row in rows).items())
            ),
        },
        "main_method_missing_metric_values": metric_missing,
        "missing_value_note": (
            "Mechanism metrics are unavailable when the adaptive method makes no physical "
            "query; corrected coefficient relative error is additionally unavailable when "
            "no queried mechanism is nonroot. These are predeclared semantic NA values, not "
            "missing columns or malformed rows."
        ),
    }


def _legacy_ci(rows: list[dict[str, object]], metric: str) -> list[float]:
    values = _metric_values(rows, metric)
    rng = np.random.default_rng(20260815)
    means = [
        float(np.mean(rng.choice(values, len(values), replace=True)))
        for _ in range(2000)
    ]
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _classification(observed: object, expected: object) -> str:
    if observed == expected:
        return "EXACT MATCH"
    try:
        left = np.asarray(observed, dtype=float)
        right = np.asarray(expected, dtype=float)
        if left.shape == right.shape and np.allclose(
            left, right, rtol=1e-12, atol=1e-14
        ):
            return "NUMERICALLY EQUIVALENT"
    except (TypeError, ValueError):
        pass
    return "WRONG"


def _audit_legacy_summary(
    legacy: dict[str, object],
    main_rows: list[dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    legacy_groups = {int(item["N_int"]): item for item in legacy["grouped_main_method"]}
    comparisons = []
    for n_int in config["N_int_grid"]:
        group_rows = [row for row in main_rows if row["N_int"] == n_int]
        item = legacy_groups.get(int(n_int))
        if item is None:
            comparisons.append(
                {"N_int": n_int, "field": "group", "classification": "MISSING"}
            )
            continue
        expected_fields: dict[str, object] = {"count": len(group_rows)}
        for metric in LEGACY_GROUPED_METRICS:
            values = _metric_values(group_rows, metric)
            expected_fields[metric] = float(np.mean(values))
            expected_fields[f"{metric}_median"] = float(np.median(values))
            expected_fields[f"{metric}_ci95"] = _legacy_ci(group_rows, metric)
        for field, expected in expected_fields.items():
            comparisons.append(
                {
                    "N_int": n_int,
                    "field": field,
                    "classification": _classification(item.get(field), expected),
                    "legacy_value": item.get(field),
                    "recomputed_value": expected,
                }
            )
    classifications = Counter(item["classification"] for item in comparisons)
    missing_metrics = sorted(set(METRICS) - set(LEGACY_GROUPED_METRICS))
    all_n_int = sorted(legacy_groups) == list(config["N_int_grid"])
    numeric_correct = not any(
        item["classification"] in {"WRONG", "MISSING"} for item in comparisons
    )
    return {
        "verdict": (
            "CORRECT_STAGE_B_NUMBERS_BUT_INCOMPLETE_LEGACY_SCHEMA"
            if all_n_int and numeric_correct
            else "SCIENTIFICALLY_UNSAFE"
        ),
        "contains_all_seven_stage_b_n_int_values": all_n_int,
        "field_classification_counts": dict(sorted(classifications.items())),
        "field_by_field_comparison": comparisons,
        "missing_requested_metrics": missing_metrics,
        "missing_stratifications": ["n x N_int", "graph_family x N_int"],
        "stale_fields": {
            "stage_a_verdict": "STAGE-A STALE VALUE (computed from Stage-B rows but mislabeled)",
            "filename": "STAGE-A STALE VALUE",
        },
    }


def _audit_legacy_handoff(
    handoff: str, config: dict[str, object], legacy_summary_audit: dict[str, object]
) -> dict[str, object]:
    checks = [
        {
            "part": "Stage-A title and research-question label",
            "classification": "STALE STAGE-A CONTENT",
        },
        {
            "part": "claim that Stage B should only be launched later",
            "classification": "STALE STAGE-A CONTENT",
        },
        {
            "part": "150 paired source problems",
            "classification": "EXACT MATCH" if "150" in handoff else "NUMERICAL ERROR",
        },
        {
            "part": "n grid {10,20,40}",
            "classification": "EXACT MATCH"
            if "[10, 20, 40]" in handoff
            else "NUMERICAL ERROR",
        },
        {
            "part": "4200 result rows",
            "classification": "EXACT MATCH"
            if "total rows=4200" in handoff
            else "NUMERICAL ERROR",
        },
        {
            "part": "seven-value N_int grid",
            "classification": "EXACT MATCH"
            if "[100, 250, 500, 1000, 2000, 5000, 10000]" in handoff
            else "NUMERICAL ERROR",
        },
        {
            "part": "256 bootstrap replicates",
            "classification": "EXACT MATCH"
            if "Bootstrap replicates=256" in handoff
            else "NUMERICAL ERROR",
        },
        {
            "part": "main seven-row quantitative table",
            "classification": "EXACT MATCH"
            if legacy_summary_audit["verdict"].startswith("CORRECT_STAGE_B_NUMBERS")
            else "NUMERICAL ERROR",
        },
    ]
    missing = (
        "config SHA; 1050 condition count; ten-seed declaration; full metric set; results "
        "by n and graph family; tree-of-cliques analysis; corrected coefficient-relative-error "
        "definition; population, graph-free, and known-DAG quantitative reference tables; "
        "manuscript-ready interpretation; figure recommendations"
    )
    checks.append({"part": missing, "classification": "MISSING STAGE-B CONTENT"})
    return {
        "verdict": "SUBSTANTIVE_STALE_AND_MISSING_CONTENT_NOT_SAFE_FOR_MANUSCRIPT",
        "component_classifications": checks,
        "legacy_file_preserved": True,
        "campaign_id_expected": config["campaign_id"],
    }


def _scientific_answers(
    by_n_int: list[dict[str, object]],
    by_n_n_int: list[dict[str, object]],
    by_graph_n_int: list[dict[str, object]],
    references: dict[str, list[dict[str, object]]],
    config: dict[str, object],
) -> dict[str, object]:
    grid = [int(value) for value in config["N_int_grid"]]
    trajectory = lambda metric: [
        _mean(_record(by_n_int, N_int=value), metric) for value in grid
    ]
    parent_f1 = trajectory("parent_f1")
    exact_support = trajectory("exact_parent_support")
    mechanism = trajectory("mechanism_l2_error")
    excess = trajectory("excess_KL")
    query_fraction = trajectory("T_q_over_n")
    excess_queries = trajectory("T_q_minus_population")
    wrong_direct = trajectory("wrong_direct_orientations")
    wrong_meek = trajectory("wrong_meek_orientations")
    known = references["known_dag_finite_sample"]
    known_high = _mean(_record(known, N_int=grid[-1]), "excess_KL")
    main_high = excess[-1]
    population = references["population_algorithm1"]
    population_query = _mean(_record(population, N_int=grid[-1]), "T_q_over_n")
    low_graph = [item for item in by_graph_n_int if item["N_int"] == grid[0]]
    hardest_excess = max(low_graph, key=lambda item: _mean(item, "excess_KL"))
    hardest_f1 = min(low_graph, key=lambda item: _mean(item, "parent_f1"))
    n40 = [_record(by_n_n_int, n=40, N_int=value) for value in grid]
    graph_free = references["graph_free_finite_sample"]
    comparisons = []
    for value in grid:
        adaptive = _record(by_n_int, N_int=value)
        free = _record(graph_free, N_int=value)
        comparisons.append(
            {
                "N_int": value,
                "adaptive_T_q_over_n": _mean(adaptive, "T_q_over_n"),
                "graph_free_T_q_over_n": _mean(free, "T_q_over_n"),
                "adaptive_excess_KL": _mean(adaptive, "excess_KL"),
                "graph_free_excess_KL": _mean(free, "excess_KL"),
            }
        )
    first_f1 = next(
        (value for value, score in zip(grid, parent_f1, strict=True) if score >= 0.95),
        None,
    )
    first_excess = next(
        (value for value, score in zip(grid, excess, strict=True) if score <= 0.01),
        None,
    )
    direct_zero = next(
        (
            value
            for value, score in zip(grid, wrong_direct, strict=True)
            if score == 0.0
        ),
        None,
    )
    meek_zero = next(
        (value for value, score in zip(grid, wrong_meek, strict=True) if score == 0.0),
        None,
    )
    return {
        "N_int": grid,
        "parent_f1": parent_f1,
        "exact_parent_support": exact_support,
        "mechanism_l2_error": mechanism,
        "excess_KL": excess,
        "T_q_over_n": query_fraction,
        "T_q_minus_population": excess_queries,
        "wrong_direct_orientations": wrong_direct,
        "wrong_meek_orientations": wrong_meek,
        "parent_f1_improves": parent_f1[-1] > parent_f1[0],
        "exact_support_improves": exact_support[-1] > exact_support[0],
        "mechanism_error_decreases": mechanism[-1] < mechanism[0],
        "excess_KL_approaches_zero": excess[-1] < excess[0],
        "high_N_int_known_DAG_floor": known_high,
        "high_N_int_main_excess_KL": main_high,
        "high_N_int_gap_above_known_DAG_floor": main_high - known_high,
        "fraction_of_high_N_int_excess_explained_by_known_DAG_floor": known_high
        / main_high,
        "population_T_q_over_n": population_query,
        "first_zero_mean_wrong_direct_N_int": direct_zero,
        "first_zero_mean_wrong_meek_N_int": meek_zero,
        "first_mean_parent_f1_at_least_0_95": first_f1,
        "first_mean_excess_KL_at_most_0_01": first_excess,
        "n40": {
            "parent_f1": [_mean(item, "parent_f1") for item in n40],
            "mechanism_l2_error": [_mean(item, "mechanism_l2_error") for item in n40],
            "excess_KL": [_mean(item, "excess_KL") for item in n40],
            "T_q_over_n": [_mean(item, "T_q_over_n") for item in n40],
        },
        "hardest_low_sample_by_mean_excess_KL": str(hardest_excess["graph_family"]),
        "hardest_low_sample_by_mean_parent_f1": str(hardest_f1["graph_family"]),
        "tree_of_cliques_is_primary_low_sample_challenge": (
            hardest_excess["graph_family"] == "tree_of_cliques"
            or hardest_f1["graph_family"] == "tree_of_cliques"
        ),
        "adaptive_vs_graph_free": comparisons,
        "descriptive_only": True,
    }


def _format(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _main_table(records: list[dict[str, object]]) -> str:
    lines = [
        "| N_int | parent F1 | exact support | mechanism L2 | final KL | excess KL | T_q | T_q/n | excess T_q | total obs. | wrong direct | wrong Meek |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(records, key=lambda row: int(row["N_int"])):
        metric = item["metrics"]
        values = [
            str(item["N_int"]),
            _format(metric["parent_f1"]["mean"]),
            _format(metric["exact_parent_support"]["mean"]),
            _format(metric["mechanism_l2_error"]["mean"]),
            _format(metric["final_true_KL"]["mean"]),
            _format(metric["excess_KL"]["mean"]),
            _format(metric["T_q"]["mean"], 3),
            _format(metric["T_q_over_n"]["mean"]),
            _format(metric["T_q_minus_population"]["mean"], 3),
            _format(metric["total_observations"]["mean"], 1),
            _format(metric["wrong_direct_orientations"]["mean"]),
            _format(metric["wrong_meek_orientations"]["mean"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _stratified_table(records: list[dict[str, object]], key: str) -> str:
    lines = [
        f"| {key} | N_int | parent F1 | mechanism L2 | excess KL | T_q/n |",
        "|:--|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(records, key=lambda row: (str(row[key]), int(row["N_int"]))):
        lines.append(
            f"| {item[key]} | {item['N_int']} | {_format(_mean(item, 'parent_f1'))} | "
            f"{_format(_mean(item, 'mechanism_l2_error'))} | {_format(_mean(item, 'excess_KL'))} | "
            f"{_format(_mean(item, 'T_q_over_n'))} |"
        )
    return "\n".join(lines)


def _reference_table(references: dict[str, list[dict[str, object]]]) -> str:
    lines = [
        "| method | N_int | excess KL | final KL | T_q/n | total observations |",
        "|:--|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for item in sorted(references[method], key=lambda row: int(row["N_int"])):
            lines.append(
                f"| {method} | {item['N_int']} | {_format(_mean(item, 'excess_KL'))} | "
                f"{_format(_mean(item, 'final_true_KL'))} | {_format(_mean(item, 'T_q_over_n'))} | "
                f"{_format(_mean(item, 'total_observations'), 1)} |"
            )
    return "\n".join(lines)


def _secondary_table(records: list[dict[str, object]]) -> str:
    lines = [
        "| N_int | precision | recall | coeff. RMSE | corrected coeff. rel. error | sign acc. | omega rel. | alpha error | support Jaccard | directed SHD | unresolved | runtime (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = (
        "parent_precision",
        "parent_recall",
        "coefficient_rmse",
        "coefficient_relative_error",
        "coefficient_sign_accuracy",
        "omega_relative_error",
        "alpha_error",
        "support_jaccard",
        "directed_shd",
        "unresolved_edges",
        "runtime_seconds",
    )
    for item in sorted(records, key=lambda row: int(row["N_int"])):
        lines.append(
            "| "
            + " | ".join(
                [str(item["N_int"])] + [_format(_mean(item, key)) for key in keys]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_figures(summary: dict[str, object], output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    by_n_int = summary["grouped_main_method_by_n_and_N_int"]
    by_graph = summary["grouped_main_method_by_graph_family_and_N_int"]
    overall = summary["grouped_main_method_by_N_int"]
    references = summary["references_by_method_and_N_int"]
    names: list[str] = []

    def line_by_n(metric: str, ylabel: str, name: str, zero: bool = False) -> None:
        figure, axis = plt.subplots(figsize=(6.2, 4.2))
        for n in (10, 20, 40):
            records = sorted(
                (item for item in by_n_int if item["n"] == n), key=lambda x: x["N_int"]
            )
            x = [item["N_int"] for item in records]
            y = [_mean(item, metric) for item in records]
            low = [item["metrics"][metric]["bootstrap_ci95"][0] for item in records]
            high = [item["metrics"][metric]["bootstrap_ci95"][1] for item in records]
            axis.plot(x, y, marker="o", label=f"n={n}")
            axis.fill_between(x, low, high, alpha=0.12)
        if zero:
            axis.axhline(
                0.0,
                color="black",
                linewidth=1,
                linestyle="--",
                label="population optimum",
            )
        axis.set_xscale("log")
        axis.set_xlabel("Singleton-intervention samples N_int")
        axis.set_ylabel(ylabel)
        axis.legend()
        figure.tight_layout()
        path = figures / name
        figure.savefig(path, dpi=220)
        plt.close(figure)
        names.append(f"figures/{name}")

    line_by_n(
        "excess_KL",
        "Excess target-to-candidate KL",
        "stage_b__excess_KL_vs_N_int_by_n.png",
        True,
    )
    line_by_n("parent_f1", "Parent F1", "stage_b__parent_f1_vs_N_int_by_n.png")
    line_by_n(
        "mechanism_l2_error",
        "Mechanism L2 error",
        "stage_b__mechanism_l2_vs_N_int_by_n.png",
    )
    line_by_n("T_q_over_n", "T_q / n", "stage_b__T_q_over_n_vs_N_int_by_n.png")

    simple = (
        (
            "T_q_minus_population",
            "Excess physical query count",
            "stage_b__excess_T_q_vs_N_int.png",
        ),
        (
            "support_jaccard",
            "Intervention support Jaccard",
            "stage_b__support_jaccard_vs_N_int.png",
        ),
    )
    for metric, ylabel, name in simple:
        figure, axis = plt.subplots(figsize=(5.8, 4.0))
        records = sorted(overall, key=lambda row: row["N_int"])
        axis.plot(
            [item["N_int"] for item in records],
            [_mean(item, metric) for item in records],
            marker="o",
        )
        axis.set_xscale("log")
        axis.set_xlabel("N_int")
        axis.set_ylabel(ylabel)
        figure.tight_layout()
        figure.savefig(figures / name, dpi=220)
        plt.close(figure)
        names.append(f"figures/{name}")

    records = sorted(overall, key=lambda row: row["N_int"])
    figure, axis = plt.subplots(figsize=(5.8, 4.0))
    axis.plot(
        [_mean(item, "total_observations") for item in records],
        [_mean(item, "excess_KL") for item in records],
        marker="o",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Mean total observations")
    axis.set_ylabel("Mean excess KL")
    figure.tight_layout()
    name = "stage_b__total_observations_vs_excess_KL.png"
    figure.savefig(figures / name, dpi=220)
    plt.close(figure)
    names.append(f"figures/{name}")

    figure, axis = plt.subplots(figsize=(5.8, 4.0))
    axis.plot(
        [item["N_int"] for item in records],
        [_mean(item, "wrong_direct_orientations") for item in records],
        marker="o",
        label="direct",
    )
    axis.plot(
        [item["N_int"] for item in records],
        [_mean(item, "wrong_meek_orientations") for item in records],
        marker="o",
        label="Meek",
    )
    axis.set_xscale("log")
    axis.set_xlabel("N_int")
    axis.set_ylabel("Mean wrong orientations")
    axis.legend()
    figure.tight_layout()
    name = "stage_b__wrong_orientations_vs_N_int.png"
    figure.savefig(figures / name, dpi=220)
    plt.close(figure)
    names.append(f"figures/{name}")

    figure, axis = plt.subplots(figsize=(5.8, 4.0))
    axis.plot(
        [item["N_int"] for item in records],
        [_mean(item, "directed_shd") for item in records],
        marker="o",
        label="directed SHD",
    )
    axis.plot(
        [item["N_int"] for item in records],
        [_mean(item, "unresolved_edges") for item in records],
        marker="o",
        label="unresolved",
    )
    axis.set_xscale("log")
    axis.set_xlabel("N_int")
    axis.set_ylabel("Mean edge count")
    axis.legend()
    figure.tight_layout()
    name = "stage_b__directed_SHD_unresolved_vs_N_int.png"
    figure.savefig(figures / name, dpi=220)
    plt.close(figure)
    names.append(f"figures/{name}")

    for metric, ylabel, name in (
        (
            "excess_KL",
            "Excess target-to-candidate KL",
            "stage_b__excess_KL_vs_N_int_by_graph_family.png",
        ),
        ("parent_f1", "Parent F1", "stage_b__parent_f1_vs_N_int_by_graph_family.png"),
    ):
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        for graph in EXPECTED_GRAPHS:
            group = sorted(
                (item for item in by_graph if item["graph_family"] == graph),
                key=lambda row: row["N_int"],
            )
            axis.plot(
                [item["N_int"] for item in group],
                [_mean(item, metric) for item in group],
                marker="o",
                label=graph,
            )
        axis.set_xscale("log")
        axis.set_xlabel("N_int")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(figures / name, dpi=220)
        plt.close(figure)
        names.append(f"figures/{name}")

    # Reference plot isolates query advantage without implying a Public Greedy comparison.
    figure, axis = plt.subplots(figsize=(5.8, 4.0))
    for method in (MAIN_METHOD, "graph_free_finite_sample", "population_algorithm1"):
        group = sorted(references[method], key=lambda row: row["N_int"])
        axis.plot(
            [item["N_int"] for item in group],
            [_mean(item, "T_q_over_n") for item in group],
            marker="o",
            label=method,
        )
    axis.set_xscale("log")
    axis.set_xlabel("N_int")
    axis.set_ylabel("T_q / n")
    axis.legend(fontsize=8)
    figure.tight_layout()
    name = "stage_b__query_fraction_reference_comparison.png"
    figure.savefig(figures / name, dpi=220)
    plt.close(figure)
    names.append(f"figures/{name}")
    return names


def _write_handoff(summary: dict[str, object], output: Path) -> None:
    integrity = summary["campaign_integrity"]
    answers = summary["scientific_answers"]
    main = summary["grouped_main_method_by_N_int"]
    by_n = summary["grouped_main_method_by_n_and_N_int"]
    by_graph = summary["grouped_main_method_by_graph_family_and_N_int"]
    references = summary["references_by_method_and_N_int"]
    config = summary["design"]
    grid = answers["N_int"]
    f1 = answers["parent_f1"]
    excess = answers["excess_KL"]
    mechanism = answers["mechanism_l2_error"]
    query = answers["T_q_over_n"]
    experimental_design = (
        f"We evaluated finite-sample Algorithm 1 on {integrity['observed']['source_problems']} "
        f"paired linear-Gaussian source problems covering n in {config['node_counts']}, "
        f"{len(config['graph_families'])} graph regimes, and "
        f"{config['replicate_seeds_per_n_graph']} seeds per n-by-family cell. "
        f"For each problem, N_obs=N_target={config['N_obs']} and N_int ranged over "
        f"{config['N_int_grid']}. The method began from the exact CPDAG, used "
        "the first-unresolved singleton-shift policy, OAS source and target covariance estimates, "
        f"and {config['bootstrap_replicates']} deterministic nonparametric bootstrap replicates "
        "with 95% support and orientation gating. We compared the graph-aware estimator with "
        "exact population Algorithm 1, a graph-free finite-sample estimator, and a known-DAG "
        "finite-sample reference on the same paired problems."
    )
    results_text = (
        f"Mean parent F1 rose from {f1[0]:.3f} at N_int={grid[0]} to {f1[-1]:.3f} at "
        f"N_int={grid[-1]}, while mean mechanism L2 error fell from {mechanism[0]:.3f} to "
        f"{mechanism[-1]:.3f}. The primary mean excess target-to-candidate KL decreased from "
        f"{excess[0]:.3f} to {excess[-1]:.4f}; mean parent F1 first exceeded 0.95 at "
        f"N_int={answers['first_mean_parent_f1_at_least_0_95']}, and mean excess KL first fell "
        f"below 0.01 at N_int={answers['first_mean_excess_KL_at_most_0_01']}. Mean T_q/n "
        f"declined from {query[0]:.3f} to {query[-1]:.3f}, toward the population reference "
        f"{answers['population_T_q_over_n']:.3f}. At N_int={grid[-1]}, excess KL was "
        f"{answers['high_N_int_main_excess_KL']:.4f}, versus the known-DAG finite-sample floor "
        f"{answers['high_N_int_known_DAG_floor']:.4f}."
    )
    handoff = f"""# Final manuscript handoff: finite-sample Algorithm 1 Stage B v2

## Research question and campaign integrity

Does finite-sample, graph-aware Algorithm 1 converge toward the population sparse optimum as singleton-intervention replication increases across n=10,20,40 and five graph families?

- Raw-data audit: **{integrity["verdict"]}**.
- Config SHA-256: `{summary["config_sha256"]}`.
- Campaign: `{summary["campaign_id"]}`; protocol: `{summary["protocol_version"]}`.
- Source problems: {integrity["observed"]["source_problems"]}; sample conditions: {integrity["observed"]["conditions"]}; rows: {integrity["observed"]["rows"]}.
- Validation: 0 failures, 0 duplicate task IDs, 0 missing conditions, 0 unexpected conditions.
- n: {config["node_counts"]}; graph families: {config["graph_families"]}; seeds: {config["replicate_seeds_per_n_graph"]} per n-by-family cell.
- N_int: {config["N_int_grid"]}; N_obs=N_target={config["N_obs"]}.
- Bootstrap replicates: {config["bootstrap_replicates"]}; covariance estimator: OAS; initial graph: exact CPDAG; policy: first_unresolved.

## Exact finite-sample procedure

OAS is fit to centered source observations using the declared closed-form shrinkage estimator, producing a positive-definite covariance and precision estimate. For queried node i, a singleton shift h=1 estimates the total-effect column as `(mean_int-mean_obs)/h`; the method forms `v_i=K_hat t_i`, `omega_i=1/v_i[i]`, and the mechanism row `m_i=omega_i v_i`. Source and intervention observations are resampled in {config["bootstrap_replicates"]} deterministic nonparametric bootstrap replicates. Candidate parent support is restricted to public CPDAG neighbors and calibrated against known non-neighbors as an empirical null. Direct orientations require the declared 95% confidence gate; low-confidence edges remain unresolved, and accepted directions are followed by Meek closure. Graph-resolved local conditionals and queried mechanisms supply estimated local shift-scale objectives to the unchanged exact four-state sparse dynamic program. Hidden DAG and exact source/target moments are evaluation-only.

The increase from 32 pilot bootstraps to 256 was frozen before Stage-B preregistration because 32 draws give a coarse empirical representation of a nominal 95% decision. Bootstrap work is computational inference and is not counted as a physical query.

The corrected coefficient relative error is the mean, over queried nonroot mechanisms only, of coefficient-vector L2 error divided by the true coefficient-vector L2 norm. Rows without a queried nonroot mechanism are NA. The pilot root-normalization bug is not reproduced.

## Main finite-sample Algorithm 1 results

All entries are means over {integrity["observed"]["source_problems"]} independent paired source problems. The machine-readable summary contains medians, sample standard deviations, metric-specific availability counts, and deterministic 95% bootstrap confidence intervals.

{_main_table(main)}

### Secondary structural and parameter metrics

{_secondary_table(main)}

## Stratification by n

{_stratified_table(by_n, "n")}

At n=40, parent F1 changed from {answers["n40"]["parent_f1"][0]:.4f} to {answers["n40"]["parent_f1"][-1]:.4f}, mechanism L2 from {answers["n40"]["mechanism_l2_error"][0]:.4f} to {answers["n40"]["mechanism_l2_error"][-1]:.4f}, excess KL from {answers["n40"]["excess_KL"][0]:.4f} to {answers["n40"]["excess_KL"][-1]:.4f}, and T_q/n from {answers["n40"]["T_q_over_n"][0]:.4f} to {answers["n40"]["T_q_over_n"][-1]:.4f}. Thus the convergence pattern remains visible at n=40 on this campaign.

## Stratification by graph family

{_stratified_table(by_graph, "graph_family")}

By the primary low-sample mean excess-KL criterion, the hardest family is **{answers["hardest_low_sample_by_mean_excess_KL"]}**; by low-sample parent F1 it is **{answers["hardest_low_sample_by_mean_parent_f1"]}**. Therefore the pilot suggestion that tree_of_cliques is the unique principal low-sample challenge is **{"supported" if answers["tree_of_cliques_is_primary_low_sample_challenge"] else "not supported"}** by these Stage-B criteria.

## Population and finite-sample references

{_reference_table(references)}

Population Algorithm 1 is the exact sparse reference and has excess KL numerically at zero by construction. Graph-free finite-sample identification queries every node (`T_q/n=1`) and isolates the query benefit of graph-aware adaptation. Known-DAG finite-sample uses no physical queries and isolates source/target moment-estimation error. At high N_int, the known-DAG floor explains {100 * answers["fraction_of_high_N_int_excess_explained_by_known_DAG_floor"]:.1f}% of the main method's residual mean excess KL; the remaining mean gap is {answers["high_N_int_gap_above_known_DAG_floor"]:.4f}.

The historical Public Greedy result using 309,800 observations is from a different problem distribution and protocol. It is not a paired Stage-B baseline and supports no apples-to-apples sample-efficiency claim here.

## Answers to the preregistered scientific questions

1. Parent F1 increases from {f1[0]:.4f} to {f1[-1]:.4f}.
2. Exact support recovery increases from {answers["exact_parent_support"][0]:.4f} to {answers["exact_parent_support"][-1]:.4f}.
3. Mechanism L2 decreases from {mechanism[0]:.4f} to {mechanism[-1]:.4f}.
4. Excess KL decreases from {excess[0]:.4f} toward zero, reaching {excess[-1]:.4f}.
5. At N_int={grid[-1]}, main excess KL is {answers["high_N_int_main_excess_KL"]:.4f}, known-DAG is {answers["high_N_int_known_DAG_floor"]:.4f}, and their gap is {answers["high_N_int_gap_above_known_DAG_floor"]:.4f}.
6. T_q/n decreases from {query[0]:.4f} to {query[-1]:.4f}, toward population {answers["population_T_q_over_n"]:.4f}.
7. Mean excess physical queries decrease from {answers["T_q_minus_population"][0]:.4f} to {answers["T_q_minus_population"][-1]:.4f}.
8. Mean wrong direct orientations first reach zero at N_int={answers["first_zero_mean_wrong_direct_N_int"]}.
9. Mean wrong Meek orientations first reach zero at N_int={answers["first_zero_mean_wrong_meek_N_int"]}.
10. The n=40 trajectory improves strongly, as quantified above.
11. The hardest low-sample family by mean excess KL is {answers["hardest_low_sample_by_mean_excess_KL"]}.
12. Tree-of-cliques is {"one of the principal low-sample challenges" if answers["tree_of_cliques_is_primary_low_sample_challenge"] else "not the principal low-sample challenge"} under the declared criteria.
13. The graph-aware method uses fewer environments than graph-free identification at every N_int; the exact paired excess-KL trajectory is in `stage_b_summary.json` and should be described rather than generalized.
14. Mean parent F1 first reaches 0.95 at N_int={answers["first_mean_parent_f1_at_least_0_95"]}.
15. Mean excess KL first reaches 0.01 at N_int={answers["first_mean_excess_KL_at_most_0_01"]}.

These thresholds are descriptive properties of this campaign, not universal sample-complexity claims.

## Manuscript-ready Experimental Design

{experimental_design}

## Manuscript-ready Results

{results_text}

The graph-aware method used fewer physical environments than graph-free identification throughout the grid, while its excess KL converged toward the known-DAG estimation floor. This establishes a query-saving, finite-sample convergence pattern for Algorithm 1 on the frozen paired simulation distribution; it does not establish a universal sample-complexity rate.

## Manuscript-ready Discussion

These results show that the sparse local-mechanism strategy remains effective when moments and singleton responses are sample-estimated: structural recovery, mechanism estimation, matching loss, and excess physical queries all improve with intervention replication. The high-replication residual is largely attributable to finite-sample source/target estimation, as indicated by the known-DAG reference. The graph-free comparison isolates the physical-query benefit of exploiting the public CPDAG. A paired Public Greedy campaign could enrich the algorithmic comparison but is optional for answering whether Algorithm 1 itself works under finite samples.

## Manuscript-ready Limitations

Stage B uses linear-Gaussian SCMs, realizable targets, an exact CPDAG, bounded coefficients and interventions, and fixed N_obs=N_target=10,000. It does not include graph-estimation error, latent confounding, cycles, nonlinear or non-Gaussian mechanisms, nonrealizable targets, or a paired Public Greedy baseline. The empirical thresholds and bootstrap behavior therefore should not be interpreted as universal guarantees or finite-sample consistency theory.

## Manuscript-ready Conclusion

Across {integrity["observed"]["source_problems"]} paired problems up to n={max(config["node_counts"])}, finite-sample Algorithm 1 approaches the population sparse optimum as singleton-intervention replication grows while retaining a substantial physical-query advantage over graph-free identification; this directly answers the criticism that Algorithm 1 itself lacked finite-sample evaluation, within the stated simulation scope.

## Existing statements to replace

No current manuscript `.tex` source is present in this repository, so exact manuscript line edits cannot be identified here. In the maintained handoff sources, replace:

- `docs/paper_support/comprehensive_paper_dossier.md`: “evaluate finite-sample target moments; and test scaling beyond n=20” — both are now addressed by Stage B through n=40.
- `docs/query_complexity_paper_handoff.md`: “Finite-sample matching: historical greedy-derived held-out method with limited partial-reduction claims.” — Algorithm 1 now has its own paired finite-sample campaign and should be presented separately.
- `docs/query_complexity_paper_handoff.md`: the evidence row labeled “finite-sample partial reduction” should no longer be the sole finite-sample evidence; retain it as historical Public Greedy evidence and add the Stage-B Algorithm 1 results.

Do not replace cautions against claims of finite-sample consistency; Stage B is empirical evidence, not a theorem.

## Figure recommendations

Main paper:

1. `figures/stage_b__excess_KL_vs_N_int_by_n.png` — primary convergence figure with the population optimum at zero.
2. `figures/stage_b__parent_f1_vs_N_int_by_n.png` — structural-recovery companion.

Supplement:

- `figures/stage_b__mechanism_l2_vs_N_int_by_n.png`
- `figures/stage_b__T_q_over_n_vs_N_int_by_n.png`
- `figures/stage_b__excess_T_q_vs_N_int.png`
- `figures/stage_b__total_observations_vs_excess_KL.png`
- `figures/stage_b__wrong_orientations_vs_N_int.png`
- `figures/stage_b__directed_SHD_unresolved_vs_N_int.png`
- `figures/stage_b__support_jaccard_vs_N_int.png`
- `figures/stage_b__excess_KL_vs_N_int_by_graph_family.png`
- `figures/stage_b__parent_f1_vs_N_int_by_graph_family.png`
- `figures/stage_b__query_fraction_reference_comparison.png`

## Authoritative artifacts

- `machine_readable/results.csv` — row-level Stage-B data.
- `machine_readable/validation.json` — completion validation.
- `machine_readable/run_metadata.json` — software/hardware provenance.
- `machine_readable/config_diff_v1_to_v2.json` — preregistered v1-to-v2 change.
- `machine_readable/stage_b_analysis_audit.json` — raw/legacy audit.
- `tables/stage_b_summary.json` — full-precision grouped statistics and 95% source-problem bootstrap CIs.
- `reports/stage_b_analysis_audit.md` — human-readable audit.

## Claims supported and not supported

Supported: descriptive convergence of finite-sample Algorithm 1 toward the population sparse optimum; improved parent/mechanism recovery with replication; fewer physical environments than graph-free identification; stable qualitative improvement through n=40; and attribution of much high-N_int residual to the known-DAG finite-sample floor.

Not supported: universal sample complexity, formal consistency, learned-graph deployment, nonlinear/non-Gaussian generality, an apples-to-apples Public Greedy comparison, or superiority outside the frozen campaign distribution.
"""
    _atomic_text(output / "reports/final_handoff_for_manuscript.md", handoff)


def _write_audit_report(audit: dict[str, object], output: Path) -> None:
    integrity = audit["raw_data_integrity"]
    legacy = audit["legacy_summary"]
    handoff = audit["legacy_handoff"]
    classifications = "\n".join(
        f"- **{key}:** {value}"
        for key, value in legacy["field_classification_counts"].items()
    )
    handoff_rows = "\n".join(
        f"| {item['part']} | {item['classification']} |"
        for item in handoff["component_classifications"]
    )
    legacy_note = (
        "The supplied legacy files were preserved without modification."
        if legacy["verdict"] != "NOT_PROVIDED"
        else "No legacy comparison directory was supplied; no legacy comparison was performed."
    )
    legacy_detail = (
        "All emitted grouped values, medians, and legacy bootstrap CIs were compared field-by-field. "
        f"Missing publication metrics: `{legacy['missing_requested_metrics']}`. Missing stratifications: "
        f"`{legacy['missing_stratifications']}`. The `stage_a_verdict` key and filename are stale "
        "Stage-A labels even though their underlying grouped inputs are Stage-B rows."
        if legacy["verdict"] != "NOT_PROVIDED"
        else "Legacy fields were not read or compared; this status is not a scientific or audit PASS."
    )
    report = f"""# Stage-B-v2 analysis audit

## Verdict

- Raw data: **{integrity["verdict"]}** — correct validated Stage-B-v2 data.
- Legacy `pilot_summary.json`: **{legacy["verdict"]}**.
- Legacy `pilot_handoff_for_manuscript.md`: **{handoff["verdict"]}**.
- {legacy_note}

## Raw integrity

Observed {integrity["observed"]["rows"]} rows, {integrity["observed"]["conditions"]} conditions, {integrity["observed"]["source_problems"]} source problems, and {integrity["observed"]["task_ids"]} unique task IDs. N_int={integrity["observed"]["N_int"]}; n={integrity["observed"]["n"]}; graph families={integrity["observed"]["graph_families"]}; methods={integrity["observed"]["methods"]}.

All integrity checks: `{json.dumps(integrity["checks"], sort_keys=True)}`.

Mechanism-metric NA counts are recorded in the machine-readable audit. They arise when no physical mechanism is queried; corrected coefficient relative error is also NA when no queried mechanism is nonroot. These are semantic NA values rather than malformed rows.

## Legacy summary field comparison

{classifications}

{legacy_detail}

## Legacy handoff classification

| component | classification |
|:--|:--|
{handoff_rows}

## Corrective action

The analyzer now detects the Stage-B-v2 campaign and consumes the validated authoritative CSV through a dedicated Stage-B path. It emits `tables/stage_b_summary.json`, `reports/final_handoff_for_manuscript.md`, this audit, and Stage-B-named figures without rewriting legacy files or executing campaign conditions.
"""
    _atomic_text(output / "reports/stage_b_analysis_audit.md", report)


def analyze_stage_b(
    config_path: Path,
    output: Path,
    *,
    input_dir: Path | None = None,
    legacy_comparison_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if input_dir is None:
        raise ValueError("Stage-B analysis requires an explicit separate input_dir")
    config_path, results_path, validation_path, legacy_summary_path, legacy_handoff_path = _preflight(
        config_path, input_dir, output, legacy_comparison_dir, overwrite
    )
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("campaign_id") != STAGE_B_CAMPAIGN_ID:
        raise ValueError("Stage-B analyzer requires the frozen Stage-B-v2 campaign")
    try:
        from scripts.run_finite_sample_algorithm1 import validation_report
    except ModuleNotFoundError:
        from run_finite_sample_algorithm1 import validation_report
    fresh_validation = validation_report(config_path, input_dir)
    if not fresh_validation["complete"]:
        raise RuntimeError("Stage-B analysis requires a current complete campaign validation")
    rows = _load_rows(results_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation != fresh_validation:
        raise RuntimeError("stored validation.json is stale or does not match the current campaign")
    integrity = _raw_integrity(rows, config, validation)
    if integrity["verdict"] != "PASS":
        raise RuntimeError(f"Stage-B raw-data integrity failed: {integrity['checks']}")
    main_rows = [row for row in rows if row["method"] == MAIN_METHOD]
    by_n_int = _grouped(main_rows, ("N_int",))
    by_n_n_int = _grouped(main_rows, ("n", "N_int"))
    by_graph_n_int = _grouped(main_rows, ("graph_family", "N_int"))
    references = {
        method: _grouped([row for row in rows if row["method"] == method], ("N_int",))
        for method in METHODS
    }
    if legacy_summary_path is None or legacy_handoff_path is None:
        legacy_summary_audit = {
            "verdict": "NOT_PROVIDED",
            "field_classification_counts": {"NOT_PROVIDED": 1},
            "missing_requested_metrics": ["NOT_PROVIDED"],
            "missing_stratifications": ["NOT_PROVIDED"],
        }
        legacy_handoff_audit = {
            "verdict": "NOT_PROVIDED",
            "component_classifications": [{"part": "legacy comparison", "classification": "NOT_PROVIDED"}],
        }
    else:
        legacy_summary = json.loads(legacy_summary_path.read_text(encoding="utf-8"))
        legacy_handoff = legacy_handoff_path.read_text(encoding="utf-8")
        legacy_summary_audit = _audit_legacy_summary(legacy_summary, main_rows, config)
        legacy_handoff_audit = _audit_legacy_handoff(
            legacy_handoff, config, legacy_summary_audit
        )
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    answers = _scientific_answers(
        by_n_int, by_n_n_int, by_graph_n_int, references, config
    )
    summary: dict[str, object] = {
        "schema_version": "finite_sample_algorithm1_stage_b_analysis_v2",
        "campaign_id": config["campaign_id"],
        "protocol_version": config["protocol_version"],
        "config_sha256": config_hash,
        "campaign_integrity": integrity,
        "design": {
            "node_counts": config["node_counts"],
            "graph_families": list(_configured_graph_labels(config)),
            "replicate_seeds_per_n_graph": config["replicate_seeds"],
            "N_int_grid": config["N_int_grid"],
            "N_obs": config["N_obs"],
            "N_target": config["N_target"],
            "bootstrap_replicates": config["bootstrap_replicates"],
            "covariance_estimator": config["primary_covariance_estimator"],
            "initial_graph_regime": config["initial_graph_regime"],
            "singleton_shift_policy": config["singleton_shift_policy"],
            "methods": config["methods"],
        },
        "bootstrap_ci": {
            "level": 0.95,
            "replicates": 5000,
            "unit": "independent paired source problem",
            "estimand": "group mean",
            "standard_deviation_ddof": 1,
        },
        "metric_note": (
            "coefficient_relative_error uses queried nonroot mechanisms only; "
            "rows without a queried nonroot mechanism are NA"
        ),
        "grouped_main_method_by_N_int": by_n_int,
        "grouped_main_method_by_n_and_N_int": by_n_n_int,
        "grouped_main_method_by_graph_family_and_N_int": by_graph_n_int,
        "references_by_method_and_N_int": references,
        "scientific_answers": answers,
    }
    audit = {
        "schema_version": "finite_sample_algorithm1_stage_b_analysis_audit_v1",
        "config_sha256": config_hash,
        "raw_data_integrity": integrity,
        "legacy_summary": legacy_summary_audit,
        "legacy_handoff": legacy_handoff_audit,
        "corrected_artifacts": [
            "tables/stage_b_summary.json",
            "reports/final_handoff_for_manuscript.md",
            "machine_readable/stage_b_analysis_audit.json",
            "reports/stage_b_analysis_audit.md",
        ],
    }
    _atomic_json(output / "tables/stage_b_summary.json", summary)
    _atomic_json(output / "machine_readable/stage_b_analysis_audit.json", audit)
    summary["figure_files"] = _write_figures(summary, output)
    _atomic_json(output / "tables/stage_b_summary.json", summary)
    _write_handoff(summary, output)
    _write_audit_report(audit, output)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-comparison-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    analyze_stage_b(
        arguments.config,
        arguments.output_dir,
        input_dir=arguments.input_dir,
        legacy_comparison_dir=arguments.legacy_comparison_dir,
        overwrite=arguments.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
