#!/usr/bin/env python3
"""Validate and summarize finite-sample robustness v1 without manual CSV edits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from scripts._output_safety import prepare_output_directory

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "docs/finite_sample_robustness_manuscript_handoff.md"
PRIMARY_METRICS = (
    "excess_KL",
    "final_true_KL",
    "K_B_star",
    "queried_parent_precision",
    "queried_parent_recall",
    "queried_parent_f1",
    "queried_mechanism_l2_error",
    "graph_resolved_mechanism_l2_error",
    "T_q",
    "T_q_over_n",
    "direct_orientation_errors",
    "meek_propagated_orientation_errors",
    "final_directed_graph_error",
    "total_observations",
    "support_jaccard",
    "local_mode_disagreement_fraction",
    "oas_shrinkage",
    "covariance_minimum_eigenvalue",
    "covariance_condition_number",
    "runtime_seconds",
    "process_peak_rss_mb",
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    from experiments.finite_sample_robustness.campaign import atomic_json

    atomic_json(path, payload)


def validate(
    output: Path, *, allow_incomplete: bool, analysis_output: Path | None = None
) -> tuple[dict, list[dict], dict]:
    analysis_output = output / "analysis" if analysis_output is None else analysis_output
    analysis_output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = _json(manifest_path)
    expected_tasks = manifest.get("tasks", [])
    expected_ids = [item["task_id"] for item in expected_tasks]
    paths = sorted((output / "results").glob("robust-*.json"))
    rows = [_json(path) for path in paths]
    observed_ids = [row.get("task_id") for row in rows]
    expected_set, observed_set = set(expected_ids), set(observed_ids)
    duplicate_ids = sorted(
        task_id for task_id, count in Counter(observed_ids).items() if count > 1
    )
    success_required = {
        "source_problem_id", "seed", "n", "graph_family", "target_family",
        "graph_condition", "N_obs", "N_target", "N_int", "method", "policy",
        "K_B_star", "final_true_KL", "excess_KL", "T_q", "oas_shrinkage",
        "status", "stage",
    }
    invalid_schema = [
        row.get("task_id")
        for row in rows
        if row.get("schema_version") != "finite_sample_robustness_row_v1"
        or (row.get("status") == "SUCCESS" and not success_required <= set(row))
        or (row.get("status") != "SUCCESS" and not {"error", "stage"} <= set(row))
    ]
    unknown_status = [
        row.get("task_id")
        for row in rows
        if row.get("status") not in {
            "SUCCESS", "GRAPH_ESTIMATION_FAILED", "INVALID_ESTIMATED_GRAPH",
            "ALGORITHM_FAILED", "NUMERICAL_FAILURE",
            "NO_VALID_CORRUPTION_AVAILABLE", "TARGET_CONSTRUCTION_FAILED",
        }
    ]
    source_ids_by_coordinate: dict[tuple, set[str]] = defaultdict(set)
    for row in rows:
        coordinate = (
            row.get("n"), row.get("graph_index"), row.get("replicate"), row.get("seed")
        )
        source_ids_by_coordinate[coordinate].add(str(row.get("source_problem_id")))
    pairing_errors = [
        list(coordinate)
        for coordinate, source_ids in source_ids_by_coordinate.items()
        if len(source_ids) != 1
    ]
    report = {
        "schema_version": "finite_sample_robustness_validation_v1",
        "mode": manifest.get("mode"),
        "expected_task_count": len(expected_ids),
        "observed_task_count": len(rows),
        "missing_task_ids": sorted(expected_set - observed_set),
        "unexpected_task_ids": sorted(observed_set - expected_set),
        "duplicate_task_ids": duplicate_ids,
        "invalid_schema_task_ids": invalid_schema,
        "unknown_status_task_ids": unknown_status,
        "pairing_errors": pairing_errors,
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
    }
    report["complete"] = not any(
        report[key]
        for key in (
            "missing_task_ids", "unexpected_task_ids", "duplicate_task_ids",
            "invalid_schema_task_ids", "unknown_status_task_ids", "pairing_errors",
        )
    ) and len(rows) == len(expected_ids)
    _write_json(analysis_output / "validation.json", report)
    if not report["complete"] and not allow_incomplete:
        raise RuntimeError(
            "refusing to summarize an incomplete campaign; inspect "
            f"{analysis_output / 'validation.json'} or pass --allow-incomplete"
        )
    return manifest, rows, report


def _source_statistics(rows: list[dict], metric: str, seed: int = 20260820) -> dict:
    by_source: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if row.get("status") == "SUCCESS" and value is not None and np.isfinite(value):
            by_source[str(row["source_problem_id"])].append(float(value))
    units = np.asarray(
        [float(np.mean(values)) for values in by_source.values()], dtype=float
    )
    if not units.size:
        return {"n_source_units": 0, "mean": None, "median": None, "ci95": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, units.size, size=(2000, units.size))
    bootstrap_means = units[indices].mean(axis=1)
    return {
        "n_source_units": int(units.size),
        "mean": float(np.mean(units)),
        "median": float(np.median(units)),
        "ci95": [float(value) for value in np.quantile(bootstrap_means, (0.025, 0.975))],
    }


def _summarize(
    rows: list[dict], group_fields: tuple[str, ...]
) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    records = []
    for key, subset in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        record = {field: value for field, value in zip(group_fields, key, strict=True)}
        record["task_count"] = len(subset)
        record["status_counts"] = dict(Counter(str(row.get("status")) for row in subset))
        record["catastrophic_failure_frequency"] = float(
            np.mean([bool(row.get("catastrophic_failure", True)) for row in subset])
        )
        record["metrics"] = {
            metric: _source_statistics(subset, metric) for metric in PRIMARY_METRICS
        }
        records.append(record)
    return records


def _paired_endpoint(rows: list[dict], campaign: str, axis: str) -> dict:
    subset = [
        row for row in rows
        if row.get("campaign") == campaign and row.get("status") == "SUCCESS"
    ]
    levels = sorted({int(row[axis]) for row in subset})
    if len(levels) < 2:
        return {"campaign": campaign, "axis": axis, "available": False}
    low, high = levels[0], levels[-1]
    by_level = {
        level: {
            str(row["source_problem_id"]): float(row["excess_KL"])
            for row in subset if int(row[axis]) == level
        }
        for level in (low, high)
    }
    paired = sorted(set(by_level[low]) & set(by_level[high]))
    differences = np.asarray(
        [by_level[high][source] - by_level[low][source] for source in paired]
    )
    if not differences.size:
        return {"campaign": campaign, "axis": axis, "available": False}
    rng = np.random.default_rng(20260820)
    bootstrap = differences[
        rng.integers(0, differences.size, size=(2000, differences.size))
    ].mean(axis=1)
    return {
        "campaign": campaign,
        "axis": axis,
        "available": True,
        "low": low,
        "high": high,
        "n_paired_source_units": len(paired),
        "mean_high_minus_low_excess_KL": float(np.mean(differences)),
        "median_high_minus_low_excess_KL": float(np.median(differences)),
        "ci95": [float(value) for value in np.quantile(bootstrap, (0.025, 0.975))],
    }


def _flatten(row: dict) -> dict:
    flat = {}
    for key, value in row.items():
        if key == "graph_metrics_before_algorithm" and isinstance(value, dict):
            flat.update({f"graph_{name}": item for name, item in value.items()})
        elif isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, sort_keys=True)
        else:
            flat[key] = value
    return flat


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    records = list(rows)
    keys = sorted({key for row in records for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _estimated_relationship(rows: list[dict]) -> dict:
    estimated = [
        row for row in rows
        if row.get("campaign") == "campaign_c_estimated_cpdag"
        and row.get("status") == "SUCCESS"
    ]
    output = {}
    for metric in (
        "skeleton_shd", "adjacency_recall", "false_negative_adjacencies",
        "false_positive_adjacencies",
    ):
        pairs = [
            (
                float(row["graph_metrics_before_algorithm"][metric]),
                float(row["excess_KL"]),
            )
            for row in estimated
            if row.get("graph_metrics_before_algorithm", {}).get(metric) is not None
        ]
        if len(pairs) < 2:
            output[metric] = {"pearson_correlation_with_excess_KL": None, "n": len(pairs)}
            continue
        values = np.asarray(pairs)
        correlation = (
            float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
            if np.std(values[:, 0]) > 0 and np.std(values[:, 1]) > 0
            else None
        )
        output[metric] = {"pearson_correlation_with_excess_KL": correlation, "n": len(pairs)}
    return output


def _figures(output: Path, rows: list[dict]) -> list[str]:
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def line(campaign: str, axis: str, group: str | None, name: str) -> None:
        subset = [row for row in rows if row.get("campaign") == campaign and row.get("status") == "SUCCESS"]
        figure, plot = plt.subplots(figsize=(5.5, 3.8))
        groups = sorted({str(row.get(group)) for row in subset}) if group else ["all"]
        for label in groups:
            chosen = subset if group is None else [row for row in subset if str(row.get(group)) == label]
            levels = sorted({int(row[axis]) for row in chosen})
            means = [np.mean([row["excess_KL"] for row in chosen if int(row[axis]) == value]) for value in levels]
            plot.plot(levels, means, marker="o", label=label)
        plot.set_xscale("log")
        plot.set_xlabel(axis)
        plot.set_ylabel("Mean excess KL above $K_B^*$")
        if group:
            plot.legend(fontsize=7)
        figure.tight_layout()
        path = figure_dir / name
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(str(path))

    line("campaign_a_n_obs", "N_obs", None, "campaign_a_excess_kl_vs_n_obs.png")
    line("campaign_b_n_target", "N_target", None, "campaign_b_excess_kl_vs_n_target.png")
    line("campaign_e_nonrealisable", "N_target", "target_family", "campaign_e_excess_kl_vs_n_target.png")

    estimated = [row for row in rows if row.get("campaign") == "campaign_c_estimated_cpdag" and row.get("status") == "SUCCESS"]
    figure, plot = plt.subplots(figsize=(5.5, 3.8))
    for retained, marker, label in ((True, "o", "all true adjacencies retained"), (False, "x", ">=1 true adjacency missing")):
        chosen = [row for row in estimated if row["graph_metrics_before_algorithm"]["retains_all_true_adjacencies"] is retained]
        plot.scatter(
            [row["graph_metrics_before_algorithm"]["skeleton_shd"] for row in chosen],
            [row["excess_KL"] for row in chosen], marker=marker, label=label, alpha=0.75,
        )
    plot.set_xlabel("Skeleton SHD")
    plot.set_ylabel("Excess KL above $K_B^*$")
    plot.legend(fontsize=7)
    figure.tight_layout()
    path = figure_dir / "campaign_c_excess_kl_vs_skeleton_shd.png"
    figure.savefig(path, dpi=180); plt.close(figure); paths.append(str(path))

    corrupted = [row for row in rows if row.get("campaign") == "campaign_d_controlled_corruption" and row.get("status") == "SUCCESS"]
    labels, values = [], []
    for key in sorted({(row["corruption_type"], row["corruption_severity"]) for row in corrupted}):
        chosen = [row for row in corrupted if (row["corruption_type"], row["corruption_severity"]) == key]
        labels.append(f"{key[0]}\n{key[1]}")
        values.append(float(np.mean([row["excess_KL"] for row in chosen])))
    figure, plot = plt.subplots(figsize=(8, 4))
    plot.bar(np.arange(len(labels)), values)
    plot.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right", fontsize=7)
    plot.set_ylabel("Mean excess KL above $K_B^*$")
    figure.tight_layout()
    path = figure_dir / "campaign_d_excess_kl_by_corruption.png"
    figure.savefig(path, dpi=180); plt.close(figure); paths.append(str(path))

    large = [row for row in rows if row.get("campaign") == "campaign_f_large_n" and row.get("status") == "SUCCESS"]
    figure, plot = plt.subplots(figsize=(5.5, 3.8))
    for family in sorted({row["graph_family"] for row in large}):
        chosen = [row for row in large if row["graph_family"] == family]
        levels = sorted({row["n"] for row in chosen})
        plot.plot(levels, [np.median([row["runtime_seconds"] for row in chosen if row["n"] == n]) for n in levels], marker="o", label=family)
    plot.set_xlabel("n"); plot.set_ylabel("Median task runtime (seconds)"); plot.legend(fontsize=7)
    figure.tight_layout()
    path = figure_dir / "campaign_f_runtime_by_n.png"
    figure.savefig(path, dpi=180); plt.close(figure); paths.append(str(path))
    return paths


def _handoff(manifest: dict, validation: dict, summary: dict, output: Path) -> str:
    full = manifest.get("mode") == "full" and validation["complete"]
    status = "FULL CAMPAIGN COMPLETE" if full else "PILOT ONLY"
    manifest["frozen_configuration"]
    failures = validation["status_counts"]
    lines = [
        "# Finite-sample robustness manuscript handoff",
        "",
        "## A. EXPERIMENT STATUS",
        "",
        f"**{status}**",
        "",
        f"The analyzed artifact contains {validation['observed_task_count']} of {validation['expected_task_count']} expected tasks. Pilot rows must not be cited as full-campaign evidence.",
        "",
        "## B. CRITICISMS ADDRESSED",
        "",
        "1. The prior finite-sample centerpiece was favorable: exact CPDAG, realisable targets, N_obs=N_target=10,000, n in {10,20,40}, and primarily an N_int sweep. Campaigns A, B, E, and F independently vary observational samples, target samples/target realizability, and dimension while the validated Stage-B N_int campaign remains unchanged.",
        "2. Algorithm 1 had not been tested with estimated or misspecified CPDAGs. Campaign C uses the repository Fisher-z PC learner without truth repair; Campaign D separately studies deletion, addition, and wrong-orientation failure modes.",
        "",
        "## C. FINAL EXPERIMENTAL DESIGN",
        "",
        "The exact frozen design is serialized in `configs/finite_sample_robustness_v1.json` and the run manifest. Full design: A/B/C use n={10,20,40}, all five Stage-B graph regimes, 10 paired source replicates, one six-level sample axis, and N_int=10,000. D uses n=20, all five regimes, 10 replicates, three corruption types, and one/5%/10% severities. E uses n=20, ER(0.35), BA(2), and tree-of-cliques, 10 replicates, three target families, and N_target={500,1000,2000,5000,10000}. F uses n={80,100}, random tree, ER(0.15), and BA(2), with five replicates. Every row uses finite-sample Algorithm 1 and the first-unresolved policy; full rows use 256 bootstraps.",
        "",
        "## D. ALGORITHM SEMANTICS",
        "",
        "**NO production Algorithm 1 semantic change.** The campaign imports the validated implementation. New code is isolated under `experiments/finite_sample_robustness/`; hidden state is used only for construction/evaluation, except explicitly labeled controlled wrong-orientation construction.",
        "",
        "## E. PRIMARY RESULTS",
        "",
    ]
    if full:
        lines.append("Exact source-level means, medians, and 95% bootstrap intervals are in `analysis/summary.json` and `analysis/summary.csv`. Use those full-precision values for manuscript text.")
    else:
        lines.append("Withheld: this is a pipeline pilot, not the full campaign. The pilot validates execution and failure retention only.")
    lines += [
        "",
        "## F. ESTIMATED-CPDAG RESULTS",
        "",
        "The analysis separates rows retaining every true adjacency from rows missing at least one, and relates excess KL to skeleton SHD, adjacency recall, false negatives, and false positives. Full numerical claims are withheld until FULL CAMPAIGN COMPLETE." if not full else "See `analysis/summary.json` campaign C strata and `analysis/estimated_graph_relationships.json`; catastrophic missing-adjacency cases are included.",
        "",
        "## G. CONTROLLED ERROR RESULTS",
        "",
        "Deletion, addition, and orientation-error rows are kept separate by severity. No ranking is claimed from pilot data." if not full else "See Campaign D groups in `analysis/summary.json`, including catastrophic-failure frequency and deletion relevance labels.",
        "",
        "## H. NON-REALISABLE RESULTS",
        "",
        "The convergence endpoint is final true KL minus exact K_B^*, never raw KL. Strictly positive K_B^* is permitted and recorded.",
        "",
        "## I. LARGE-n RESULTS",
        "",
        "n=80 and n=100 are included in the frozen design. Accuracy, runtime, peak process RSS, OAS shrinkage, and conditioning are recorded. The separate hardware benchmark determines the launch backend.",
        "",
        "## J. NEGATIVE RESULTS / FAILURES",
        "",
        f"Status counts: `{json.dumps(failures, sort_keys=True)}`. Failures remain as explicit task rows and are never silently dropped.",
        "",
        "## K. FIGURES AND TABLES",
        "",
        "Generated paths are listed in `analysis/artifacts.json`. The principal figures cover N_obs, N_target, estimated-graph SHD, controlled corruptions, non-realisable targets, and large-n runtime.",
        "",
        "## L. MANUSCRIPT UPDATE RECOMMENDATION",
        "",
    ]
    if full:
        lines += [
            "Use the full-precision aggregate file to fill the following pasteable locations; do not round from plots.",
            "",
            "### Abstract",
            "",
            "INSERT AFTER the finite-sample contribution: `We further separate observational-, target-, intervention-, and graph-estimation error, including controlled CPDAG misspecification and non-realisable Gaussian targets.`",
            "",
            "### Introduction / Contributions",
            "",
            "INSERT AFTER the finite-sample contribution: `A paired robustness campaign varies source and target sample sizes independently, learns CPDAGs from observational data, isolates edge-deletion/addition/orientation errors, and evaluates excess divergence above the exact bounded sparse population optimum.`",
            "",
            "### Experimental Design",
            "",
            "INSERT AFTER the Stage-B design: `We additionally run the frozen one-factor robustness panels documented in the campaign manifest; estimated graphs use Fisher-z PC (alpha=0.05, maximum conditioning-set size 2) on the same observational sample used for moment estimation, with no oracle repair.`",
            "",
            "### Results",
            "",
            "REPLACE WITH manuscript-ready numerical sentences generated from `analysis/summary.json`; this handoff intentionally does not duplicate rounded values.",
            "",
            "### Discussion",
            "",
            "INSERT AFTER graph-robustness discussion: `Controlled corruptions distinguish exclusion caused by missing adjacencies from the additional-query and ambiguity effects of false-positive adjacencies; all catastrophic cases are retained.`",
            "",
            "### Limitations",
            "",
            "INSERT AFTER synthetic-data limitations: `Graph learning is limited to the repository Fisher-z PC procedure and the tested sparse linear-Gaussian regimes; robustness to latent confounding, nonlinear mechanisms, measurement error, and real data remains open.`",
            "",
            "### Conclusion",
            "",
            "INSERT BEFORE the final sentence: `Within the tested synthetic regimes, finite-sample matching is assessed relative to the exact sparse population floor under independent moment-sample axes and explicit graph uncertainty.`",
        ]
    else:
        lines.append("FIND/REPLACE and numerical LaTeX are intentionally withheld until the full campaign passes completeness validation. Do not update the manuscript from pilot results.")
    lines += [
        "",
        "## M. CLAIMS NOW SUPPORTED",
        "",
        ("For the completed full campaign: empirical robustness across independent N_obs/N_target axes; measured behavior under repository-PC graph uncertainty; controlled sensitivity to adjacency deletion/addition/orientation errors; convergence toward K_B^* for tested non-realisable targets; and operation at n=80/100." if full else "Pilot only: deterministic construction, execution, failure serialization, and metric plumbing across every campaign type."),
        "",
        "## N. CLAIMS STILL NOT SUPPORTED",
        "",
        "Latent confounding; nonlinear SCMs; measurement error; failed or imperfect interventions; non-Gaussian conditional mechanisms; universal finite-sample consistency; real-world robustness; or superiority to methods not included in the paired campaign.",
        "",
        "## Reproducibility",
        "",
        f"Analyzed output: `{output}`. Backend: `{manifest.get('backend')}`; tasks: {manifest.get('expected_task_count')}; config hash: `{manifest.get('config_sha256')}`.",
    ]
    return "\n".join(lines) + "\n"


def analyze(
    output: Path, *, allow_incomplete: bool, analysis_output: Path
) -> dict:
    manifest, rows, validation = validate(
        output, allow_incomplete=allow_incomplete, analysis_output=analysis_output
    )
    success = [row for row in rows if row.get("status") == "SUCCESS"]
    campaign_groups = {
        "campaign_a_n_obs": ("N_obs",),
        "campaign_b_n_target": ("N_target",),
        "campaign_c_estimated_cpdag": ("N_graph", "graph_condition"),
        "campaign_d_controlled_corruption": ("corruption_type", "corruption_severity"),
        "campaign_e_nonrealisable": ("target_family", "N_target"),
        "campaign_f_large_n": ("n", "graph_family"),
    }
    summary = {
        "schema_version": "finite_sample_robustness_analysis_v1",
        "experiment_status": (
            "FULL CAMPAIGN COMPLETE"
            if manifest.get("mode") == "full" and validation["complete"]
            else "PILOT ONLY"
        ),
        "validation": validation,
        "campaigns": {
            campaign: _summarize(
                [row for row in rows if row.get("campaign") == campaign], fields
            )
            for campaign, fields in campaign_groups.items()
        },
        "estimated_cpdag_retention_strata": _summarize(
            [
                {**row, "retention_stratum": (
                    "all_true_adjacencies_retained"
                    if row.get("graph_metrics_before_algorithm", {}).get("retains_all_true_adjacencies")
                    else "one_or_more_true_adjacencies_missing"
                )}
                for row in rows if row.get("campaign") == "campaign_c_estimated_cpdag"
            ],
            ("N_graph", "retention_stratum"),
        ),
        "paired_endpoint_differences": [
            _paired_endpoint(rows, "campaign_a_n_obs", "N_obs"),
            _paired_endpoint(rows, "campaign_b_n_target", "N_target"),
            _paired_endpoint(rows, "campaign_c_estimated_cpdag", "N_graph"),
        ],
        "success_count": len(success),
        "failure_count": len(rows) - len(success),
    }
    _write_json(analysis_output / "summary.json", summary)
    relationships = _estimated_relationship(rows)
    _write_json(analysis_output / "estimated_graph_relationships.json", relationships)
    _write_csv(analysis_output / "results.csv", (_flatten(row) for row in rows))
    summary_rows = []
    for campaign, groups in summary["campaigns"].items():
        for group in groups:
            base = {key: value for key, value in group.items() if key != "metrics"}
            for metric, stats in group["metrics"].items():
                summary_rows.append({"campaign": campaign, **base, "metric": metric, **stats})
    _write_csv(analysis_output / "summary.csv", summary_rows)
    figures = _figures(analysis_output, rows)
    handoff = analysis_output / "finite_sample_robustness_manuscript_handoff.md"
    artifacts = {
        "validation": str(analysis_output / "validation.json"),
        "row_table": str(analysis_output / "results.csv"),
        "aggregate_json": str(analysis_output / "summary.json"),
        "aggregate_csv": str(analysis_output / "summary.csv"),
        "estimated_graph_relationships": str(analysis_output / "estimated_graph_relationships.json"),
        "figures": figures,
        "manuscript_handoff": str(handoff),
    }
    _write_json(analysis_output / "artifacts.json", artifacts)
    handoff.write_text(_handoff(manifest, validation, summary, output), encoding="utf-8")
    print(json.dumps({
        "experiment_status": summary["experiment_status"],
        "expected": validation["expected_task_count"],
        "observed": validation["observed_task_count"],
        "status_counts": validation["status_counts"],
        "handoff": str(handoff),
    }, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    analysis_output = prepare_output_directory(
        arguments.output_dir,
        overwrite=arguments.overwrite,
        protected=(arguments.input_dir,),
    )
    analyze(
        arguments.input_dir.resolve(),
        allow_incomplete=arguments.allow_incomplete,
        analysis_output=analysis_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
