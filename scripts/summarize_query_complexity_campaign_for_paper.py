#!/usr/bin/env python3
"""Audit and summarize the completed local-mechanism query campaign.

This script is intentionally derivative-only: it reads the immutable campaign
configuration/results and writes compact paper-facing summaries under the
campaign's ``paper_summary`` directory.  It never regenerates source problems
or changes historical outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


MAIN_TASK = re.compile(r"^main:n(?P<n>\d+):g(?P<g>\d+):r(?P<r>\d+):(?P<tail>.+)$")
ADAPTIVE = "public_adaptive_local_mechanism_dp"
FIRST = "first_unresolved"
DEGREE = "max_undirected_degree"
CONTROL = "fully_directed_control"
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_DRAWS = 10_000


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    malformed: list[dict] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, TypeError) as error:
            malformed.append({"line": line_number, "error": str(error)})
            continue
        if not isinstance(row, dict):
            malformed.append({"line": line_number, "error": "JSON value is not an object"})
            continue
        rows.append(row)
    return rows, malformed


def _graph_label(config: dict, index: int) -> str:
    return f"{config['graph_regimes'][index]['family']}:{index}"


def _expected_ids(config: dict) -> tuple[set[str], set[str]]:
    main: set[str] = set()
    for n in config["node_counts"]:
        for graph_index, _ in enumerate(config["graph_regimes"]):
            for replicate in range(int(config["replicate_seeds"])):
                source = f"n{int(n)}:g{graph_index}:r{replicate}"
                main.add(f"main:{source}:known_dag_exact_dp")
                main.add(f"main:{source}:graph_free_total_effect_dp")
                for information in config["initial_graph_regimes"]:
                    for policy in config["policies"]:
                        main.add(f"main:{source}:{information['name']}:{policy}")
    panel = {
        f"target_robustness:{family}:{policy}"
        for family in config["target_robustness_panel"]
        for policy in config["policies"]
    }
    return main, panel


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _stats(values: Sequence[float], prefix: str) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {f"{prefix}_count": 0}
    q25, q75 = np.quantile(array, [0.25, 0.75])
    return {
        f"{prefix}_count": int(len(array)),
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
    }


def _aggregate(rows: Sequence[dict], keys: Sequence[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group_key in sorted(groups, key=lambda item: tuple(str(x) for x in item)):
        group = groups[group_key]
        output = dict(zip(keys, group_key))
        output["rows"] = len(group)
        for metric in ("T", "T_over_n", "n_minus_T"):
            output.update(_stats([float(row[metric]) for row in group], metric))
        output.update(
            _stats([1.0 - float(row["T_over_n"]) for row in group], "one_minus_T_over_n")
        )
        for metric in (
            "direct_mechanism_recoveries",
            "graph_resolved_without_query_nodes",
            "direct_orientations",
            "meek_orientations",
            "initial_resolved_nodes",
            "initial_undirected_edges",
        ):
            values = [float(row[metric]) for row in group if _finite_number(row.get(metric))]
            if values:
                output.update(_stats(values, metric))
        result.append(output)
    return result


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pearson(rows: Sequence[dict], x: str, y: str) -> dict:
    pairs = [
        (float(row[x]), float(row[y]))
        for row in rows
        if _finite_number(row.get(x)) and _finite_number(row.get(y))
    ]
    if len(pairs) < 2:
        return {"x": x, "y": y, "n": len(pairs), "pearson_r": None}
    xs, ys = (np.asarray(values, dtype=float) for values in zip(*pairs))
    if np.std(xs) == 0 or np.std(ys) == 0:
        coefficient = None
    else:
        coefficient = float(np.corrcoef(xs, ys)[0, 1])
    return {"x": x, "y": y, "n": len(pairs), "pearson_r": coefficient}


def _bootstrap_mean_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    chunk = 500
    means = []
    for start in range(0, BOOTSTRAP_DRAWS, chunk):
        draws = min(chunk, BOOTSTRAP_DRAWS - start)
        sample_indices = rng.integers(0, len(array), size=(draws, len(array)))
        means.append(np.mean(array[sample_indices], axis=1))
    bootstrap_means = np.concatenate(means)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def _policy_pairs(adaptive_main: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in adaptive_main:
        match = MAIN_TASK.match(row["task_id"])
        assert match is not None
        key = (
            int(match["n"]),
            int(match["g"]),
            int(match["r"]),
            row["graph_family"],
            row["initial_graph_regime"],
        )
        grouped[key][row["policy"]] = row
    differences: list[dict] = []
    for key in sorted(grouped, key=lambda item: tuple(str(x) for x in item)):
        pair = grouped[key]
        if FIRST not in pair or DEGREE not in pair:
            continue
        first, degree = pair[FIRST], pair[DEGREE]
        delta_t = int(first["T"]) - int(degree["T"])
        delta_fraction = float(first["T_over_n"]) - float(degree["T_over_n"])
        differences.append(
            {
                "n": key[0],
                "graph_index": key[1],
                "replicate": key[2],
                "graph_family": key[3],
                "initial_graph_regime": key[4],
                "T_first_unresolved": int(first["T"]),
                "T_max_undirected_degree": int(degree["T"]),
                "delta_T_first_minus_degree": delta_t,
                "delta_T_over_n_first_minus_degree": delta_fraction,
                "winner": DEGREE if delta_t > 0 else FIRST if delta_t < 0 else "tie",
                "first_task_id": first["task_id"],
                "degree_task_id": degree["task_id"],
            }
        )

    summary: list[dict] = []

    def add_summary(scope: str, values: Sequence[dict], stratum: str = "all") -> None:
        deltas = [float(row["delta_T_first_minus_degree"]) for row in values]
        normalized = [float(row["delta_T_over_n_first_minus_degree"]) for row in values]
        low, high = _bootstrap_mean_ci(deltas)
        summary.append(
            {
                "scope": scope,
                "stratum": stratum,
                "pairs": len(values),
                "mean_delta_T_first_minus_degree": float(np.mean(deltas)),
                "median_delta_T_first_minus_degree": float(np.median(deltas)),
                "mean_delta_T_over_n_first_minus_degree": float(np.mean(normalized)),
                "degree_wins": sum(x > 0 for x in deltas),
                "ties": sum(x == 0 for x in deltas),
                "first_wins": sum(x < 0 for x in deltas),
                "degree_win_fraction": sum(x > 0 for x in deltas) / len(deltas),
                "tie_fraction": sum(x == 0 for x in deltas) / len(deltas),
                "first_win_fraction": sum(x < 0 for x in deltas) / len(deltas),
                "maximum_degree_saving": max(deltas),
                "maximum_degree_disadvantage": min(deltas),
                "bootstrap_mean_delta_T_ci95_low": low,
                "bootstrap_mean_delta_T_ci95_high": high,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
            }
        )

    meaningful = [row for row in differences if row["initial_graph_regime"] != CONTROL]
    add_summary("overall_excluding_fully_directed", meaningful)
    add_summary("fully_directed_control", [row for row in differences if row["initial_graph_regime"] == CONTROL])
    for field in ("graph_family", "initial_graph_regime", "n"):
        values = sorted({row[field] for row in meaningful}, key=str)
        for value in values:
            add_summary(field, [row for row in meaningful if row[field] == value], str(value))
    return differences, summary


def _exactness(rows: Sequence[dict]) -> tuple[list[dict], list[dict], dict]:
    def summarize(group: Sequence[dict], label: dict) -> dict:
        gaps = np.asarray([abs(float(row["final_KL"]) - float(row["K_B_star"])) for row in group])
        worst_index = int(np.argmax(gaps))
        return {
            **label,
            "rows": len(group),
            "max_abs_KL_gap": float(np.max(gaps)),
            "mean_abs_KL_gap": float(np.mean(gaps)),
            "median_abs_KL_gap": float(np.median(gaps)),
            "p95_abs_KL_gap": float(np.quantile(gaps, 0.95)),
            "p99_abs_KL_gap": float(np.quantile(gaps, 0.99)),
            "worst_task_id": group[worst_index]["task_id"],
            "count_abs_gap_gt_1e-12": int(np.sum(gaps > 1e-12)),
            "count_abs_gap_gt_1e-11": int(np.sum(gaps > 1e-11)),
        }

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["panel"], row["method"])].append(row)
    summary = [summarize(grouped[key], {"panel": key[0], "method": key[1]}) for key in sorted(grouped)]
    summary.insert(0, summarize(rows, {"panel": "all", "method": "all_exact_methods"}))
    panel_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["panel"] == "target_robustness":
            panel_groups[row["target_family"]].append(row)
    panel = [summarize(panel_groups[key], {"target_family": key}) for key in sorted(panel_groups)]
    overall = summary[0]
    return summary, panel, overall


def _audit(config: dict, manifest: dict, rows: Sequence[dict], malformed: Sequence[dict], config_path: Path) -> dict:
    expected_main, expected_panel = _expected_ids(config)
    ids = [row.get("task_id") for row in rows]
    counts = Counter(ids)
    duplicates = sorted(str(task_id) for task_id, count in counts.items() if count > 1)
    actual_ids = {task_id for task_id in ids if isinstance(task_id, str)}
    actual_main = {row["task_id"] for row in rows if row.get("panel") == "main" and isinstance(row.get("task_id"), str)}
    actual_panel = {row["task_id"] for row in rows if row.get("panel") == "target_robustness" and isinstance(row.get("task_id"), str)}

    required = {
        "task_id", "panel", "protocol_version", "n", "graph_family", "target_family",
        "K_B_star", "method", "policy", "initial_graph_regime", "T", "T_over_n",
        "n_minus_T", "final_KL", "KL_gap",
    }
    malformed_schema = []
    inconsistent_gap = []
    for line_index, row in enumerate(rows, 1):
        missing = sorted(required - row.keys())
        numeric_bad = [key for key in ("n", "K_B_star", "T", "T_over_n", "n_minus_T", "final_KL", "KL_gap") if key in row and not _finite_number(row[key])]
        if missing or numeric_bad:
            malformed_schema.append({"record": line_index, "task_id": row.get("task_id"), "missing": missing, "nonfinite_or_nonnumeric": numeric_bad})
        if all(_finite_number(row.get(key)) for key in ("final_KL", "K_B_star", "KL_gap")):
            recomputed = float(row["final_KL"]) - float(row["K_B_star"])
            if recomputed != float(row["KL_gap"]):
                inconsistent_gap.append(row.get("task_id"))

    pairing_errors = []
    source_groups: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("panel") != "main" or not isinstance(row.get("task_id"), str):
            continue
        match = MAIN_TASK.match(row["task_id"])
        if not match:
            pairing_errors.append({"task_id": row["task_id"], "error": "unparseable main task ID"})
            continue
        key = (int(match["n"]), int(match["g"]), int(match["r"]))
        source_groups[key].append(row)
    expected_rows_per_source = 2 + len(config["initial_graph_regimes"]) * len(config["policies"])
    for key, group in sorted(source_groups.items()):
        n, graph_index, replicate = key
        expected_graph = _graph_label(config, graph_index)
        issues = []
        if len(group) != expected_rows_per_source:
            issues.append(f"row_count={len(group)} expected={expected_rows_per_source}")
        if {int(row["n"]) for row in group} != {n}:
            issues.append("n mismatch")
        if {row["graph_family"] for row in group} != {expected_graph}:
            issues.append("graph label mismatch")
        if {row["protocol_version"] for row in group} != {config["protocol_version"]}:
            issues.append("protocol mismatch")
        if len({float(row["K_B_star"]) for row in group}) != 1:
            issues.append("K_B_star mismatch within source")
        adaptive = [row for row in group if row["method"] == ADAPTIVE]
        if {int(row.get("replicate", -1)) for row in adaptive} != {replicate}:
            issues.append("adaptive replicate mismatch")
        if issues:
            pairing_errors.append({"source": f"n{n}:g{graph_index}:r{replicate}", "errors": issues})

    known = [row for row in rows if row.get("method") == "known_dag_exact_dp"]
    graph_free = [row for row in rows if row.get("method") == "graph_free_total_effect_dp"]
    adaptive = [row for row in rows if row.get("method") == ADAPTIVE]
    directed = [row for row in adaptive if row.get("initial_graph_regime") == CONTROL]
    failed = [row.get("task_id") for row in rows if row.get("status") in {"failed", "incomplete", "error"}]
    unexpected_terminations = [row["task_id"] for row in adaptive if row.get("termination_reason") != "all_local_mechanisms_known"]

    gate = float(config["exactness_tolerance"])
    gaps = [abs(float(row["final_KL"]) - float(row["K_B_star"])) for row in rows if _finite_number(row.get("final_KL")) and _finite_number(row.get("K_B_star"))]
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    integrity_failures = {
        "malformed_json_rows": len(malformed),
        "malformed_schema_rows": len(malformed_schema),
        "duplicate_task_ids": len(duplicates),
        "missing_expected_task_ids": len((expected_main | expected_panel) - actual_ids),
        "unexpected_task_ids": len(actual_ids - (expected_main | expected_panel)),
        "pairing_errors": len(pairing_errors),
        "failed_or_incomplete_rows": len(failed),
        "known_DAG_control_violations": sum(int(row["T"]) != 0 or float(row["T_over_n"]) != 0.0 for row in known),
        "graph_free_control_violations": sum(int(row["T"]) != int(row["n"]) or float(row["T_over_n"]) != 1.0 for row in graph_free),
        "fully_directed_control_violations": sum(int(row["T"]) != 0 or float(row["T_over_n"]) != 0.0 for row in directed),
        "unexpected_termination_rows": len(unexpected_terminations),
        "stored_KL_gap_inconsistencies": len(inconsistent_gap),
        "rows_exceeding_predeclared_gate": sum(gap > gate for gap in gaps),
        "config_hash_mismatch": int(config_hash != manifest.get("config_sha256")),
        "manifest_not_complete": int(manifest.get("status") != "complete"),
    }
    return {
        "verdict": "PASS" if not any(integrity_failures.values()) else "FAIL",
        "integrity_failure_counts": integrity_failures,
        "counts": {
            "source_problems_expected": len(config["node_counts"]) * len(config["graph_regimes"]) * int(config["replicate_seeds"]),
            "source_problems_actual": len(source_groups),
            "rows_per_main_source": expected_rows_per_source,
            "expected_main_rows": len(expected_main),
            "actual_main_rows": sum(row.get("panel") == "main" for row in rows),
            "expected_exactness_panel_rows": len(expected_panel),
            "actual_exactness_panel_rows": sum(row.get("panel") == "target_robustness" for row in rows),
            "total_rows": len(rows),
            "duplicates": len(duplicates),
            "missing_task_ids": len((expected_main | expected_panel) - actual_ids),
            "malformed_rows": len(malformed) + len(malformed_schema),
            "failed_or_incomplete_rows": len(failed),
        },
        "details": {
            "duplicate_task_ids": duplicates,
            "missing_task_ids": sorted((expected_main | expected_panel) - actual_ids),
            "unexpected_task_ids": sorted(actual_ids - (expected_main | expected_panel)),
            "malformed_json": list(malformed),
            "malformed_schema": malformed_schema,
            "pairing_errors": pairing_errors,
            "failed_or_incomplete_task_ids": failed,
            "unexpected_termination_task_ids": unexpected_terminations,
        },
        "pairing": {
            "validated_keys": ["n", "graph_family", "replicate", "protocol_version", "K_B_star"],
            "source_fingerprint_available": False,
            "limitation": "Rows do not serialize a source fingerprint; stable task IDs plus deterministic seed construction and equality of source-level K_B_star provide the available pairing audit.",
        },
        "controls": {
            "known_dag_rows": len(known),
            "graph_free_rows": len(graph_free),
            "fully_directed_adaptive_rows": len(directed),
            "termination_reasons": dict(sorted(Counter(row.get("termination_reason") for row in adaptive).items(), key=lambda x: str(x[0]))),
        },
        "exactness": {
            "predeclared_gate": gate,
            "rows_exceeding_gate": sum(gap > gate for gap in gaps),
        },
        "provenance": {
            "campaign_id": manifest.get("campaign_id"),
            "protocol_version": manifest.get("protocol_version"),
            "branch": manifest.get("branch"),
            "head": manifest.get("head"),
            "manifest_status": manifest.get("status"),
            "manifest_config_sha256": manifest.get("config_sha256"),
            "recomputed_config_sha256": config_hash,
            "base_seed": config.get("base_seed"),
            "seed_semantics": "SeedSequence([base_seed, n, replicate, graph_regime_index]), spawned into graph, coefficient, noise, and target RNG streams.",
            "resume_semantics": manifest.get("resume_semantics"),
            "launch_worktree_cleanliness": "not recorded in manifest; cannot be inferred retrospectively from campaign artifacts",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/query_complexity_local_mechanism_campaign_v1.json"))
    parser.add_argument("--campaign", type=Path, default=Path("outputs/kl_query_complexity_v1_local_mechanism_full"))
    args = parser.parse_args()
    config = _load_json(args.config)
    manifest = _load_json(args.campaign / "manifest.json")
    rows, malformed = _load_jsonl(args.campaign / "results.jsonl")
    output = args.campaign / "paper_summary"
    output.mkdir(parents=True, exist_ok=True)

    audit = _audit(config, manifest, rows, malformed, args.config)
    exactness_summary, exactness_panel, exactness_overall = _exactness(rows)
    audit["exactness"].update(exactness_overall)

    adaptive_main = [row for row in rows if row.get("panel") == "main" and row.get("method") == ADAPTIVE]
    meaningful = [row for row in adaptive_main if row["initial_graph_regime"] != CONTROL]
    controls = [row for row in adaptive_main if row["initial_graph_regime"] == CONTROL]

    correlations = [
        _pearson(meaningful, "initial_undirected_edges", "T_over_n"),
        _pearson(meaningful, "initial_resolved_nodes", "T_over_n"),
        _pearson(meaningful, "graph_resolved_without_query_nodes", "T_over_n"),
        _pearson(meaningful, "meek_orientations", "T_over_n"),
    ]
    audit["structural_correlations"] = correlations
    (output / "audit_summary.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    _write_csv(output / "aggregate_by_n.csv", _aggregate(meaningful, ["n", "policy"]))
    _write_csv(output / "aggregate_by_graph_family.csv", _aggregate(meaningful, ["graph_family", "policy"]))
    _write_csv(output / "aggregate_by_information_regime.csv", _aggregate(adaptive_main, ["initial_graph_regime", "policy"]))
    policy_rows = []
    for scope, group in (("excluding_fully_directed", meaningful), ("fully_directed_control", controls)):
        for row in _aggregate(group, ["policy"]):
            row = {"scope": scope, **row}
            policy_rows.append(row)
    _write_csv(output / "aggregate_by_policy.csv", policy_rows)
    _write_csv(output / "aggregate_by_n_graph.csv", _aggregate(meaningful, ["n", "graph_family", "policy"]))
    _write_csv(output / "aggregate_by_n_information.csv", _aggregate(adaptive_main, ["n", "initial_graph_regime", "policy"]))
    _write_csv(output / "aggregate_by_graph_information.csv", _aggregate(adaptive_main, ["graph_family", "initial_graph_regime"]))
    _write_csv(output / "aggregate_by_graph_information_policy.csv", _aggregate(adaptive_main, ["graph_family", "initial_graph_regime", "policy"]))

    differences, paired_summary = _policy_pairs(adaptive_main)
    _write_csv(output / "paired_policy_differences.csv", differences)
    _write_csv(output / "paired_policy_summary.csv", paired_summary)
    _write_csv(output / "exactness_summary.csv", exactness_summary)
    _write_csv(output / "exactness_panel_by_target_family.csv", exactness_panel)

    mechanism = []
    for policy in config["policies"]:
        group = [row for row in meaningful if row["policy"] == policy]
        total_queries = sum(int(row["T"]) for row in group)
        total_meek = sum(int(row["meek_orientations"]) for row in group)
        total_graph_resolved = sum(int(row["graph_resolved_without_query_nodes"]) for row in group)
        mechanism.append(
            {
                "policy": policy,
                "rows": len(group),
                "total_queries": total_queries,
                "mean_queried_fraction": float(np.mean([row["T_over_n"] for row in group])),
                "mean_fraction_mechanisms_resolved_without_direct_query": float(np.mean([row["graph_resolved_without_query_nodes"] / row["n"] for row in group])),
                "aggregate_meek_orientations_per_query": total_meek / total_queries if total_queries else 0.0,
                "aggregate_graph_resolved_nodes_per_query": total_graph_resolved / total_queries if total_queries else 0.0,
            }
        )
    _write_csv(output / "structural_mechanism_summary.csv", mechanism)
    _write_csv(output / "structural_correlations.csv", correlations)

    if audit["verdict"] != "PASS":
        raise RuntimeError(f"campaign audit failed; inspect {output / 'audit_summary.json'}")
    print(f"PASS: audited {len(rows)} rows; summaries written to {output}")


if __name__ == "__main__":
    main()
