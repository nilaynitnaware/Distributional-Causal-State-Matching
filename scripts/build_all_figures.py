#!/usr/bin/env python3
"""Build all reviewed manuscript graphs from archived result artifacts only.

This is analysis/plotting code. It does not import or run experiment modules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

from scripts._output_safety import prepare_output_directory
from scripts.figure_style import PALETTE, apply_publication_style, save_figure, style_axis


N_INT_GRID = (100, 250, 500, 1000, 2000, 5000, 10000)
N_GRAPH_GRID = (250, 500, 1000, 2000, 5000, 10000)
N_VALUES = (10, 20, 40)
N_INT_LABELS = ("100", "250", "500", "1k", "2k", "5k", "10k")
N_GRAPH_LABELS = ("250", "500", "1k", "2k", "5k", "10k")

N_STYLES = {
    10: (PALETTE["blue_main"], "o", "-"),
    20: (PALETTE["red_strong"], "s", "--"),
    40: (PALETTE["teal"], "^", "-."),
}

FAMILY_STYLES = {
    "random_tree:0": (PALETTE["blue_main"], "o", "Random tree"),
    "erdos_renyi:1": (PALETTE["green"], "s", "ER, p=0.15"),
    "erdos_renyi:2": (PALETTE["red_strong"], "^", "ER, p=0.35"),
    "barabasi_albert:3": (PALETTE["teal"], "D", "Barabasi-Albert"),
    "tree_of_cliques:4": (PALETTE["violet"], "P", "Trees of cliques"),
}

FINITE_EXPECTED = {
    "parent_f1": [0.8379, 0.9324, 0.9733, 0.9880, 0.9906, 0.9928, 0.9957],
    "mechanism_l2_error": [
        0.1761,
        0.1031,
        0.0678,
        0.0483,
        0.0348,
        0.0260,
        0.0216,
    ],
    "excess_KL": [
        0.26813,
        0.14654,
        0.05971,
        0.01049,
        0.00513,
        0.00312,
        0.00233,
    ],
    "T_q_over_n": [0.3862, 0.3563, 0.3372, 0.3305, 0.3280, 0.3252, 0.3217],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, required=True,
        help="Root containing the required Stage-B, robustness, and exact-population output trees",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _metric(record: dict[str, object], name: str) -> dict[str, object]:
    return record["metrics"][name]


def build_finite_sample(
    stage_summary_path: Path,
    stage_results_path: Path,
    data_dir: Path,
    figure_dir: Path,
) -> dict[str, object]:
    summary = json.loads(stage_summary_path.read_text(encoding="utf-8"))
    grouped = summary["grouped_main_method_by_n_and_N_int"]
    pooled = summary["grouped_main_method_by_N_int"]
    require(len(grouped) == 21, "expected 21 n x N_int aggregate records")
    require(len(pooled) == 7, "expected seven pooled N_int aggregate records")

    checks: dict[str, object] = {"pooled": {}, "passed": True}
    for metric, expected in FINITE_EXPECTED.items():
        actual = [float(_metric(row, metric)["mean"]) for row in pooled]
        differences = [abs(a - e) for a, e in zip(actual, expected, strict=True)]
        # Manuscript checks are rounded to 4 or 5 decimals.
        tolerance = 5.1e-5 if metric == "excess_KL" else 5.1e-5
        require(max(differences) <= tolerance, f"finite-sample manuscript check failed: {metric}")
        checks["pooled"][metric] = {
            "actual": actual,
            "manuscript_rounded_values": expected,
            "maximum_absolute_difference": max(differences),
            "tolerance": tolerance,
        }

    raw = read_csv(stage_results_path)
    main = [row for row in raw if row["method"] == "finite_sample_algorithm1"]
    require(len(main) == 1050, "expected 1050 main finite-sample rows")
    references: dict[int, float] = {}
    for n in N_VALUES:
        rows = [row for row in main if int(row["n"]) == n and int(row["N_int"]) == 100]
        require(len(rows) == 50, f"expected 50 population-reference rows for n={n}")
        require(len({row["problem_id"] for row in rows}) == 50, "population reference units not unique")
        references[n] = statistics.fmean(float(row["population_T_q"]) / n for row in rows)
    pooled_reference = statistics.fmean(
        float(row["population_T_q"]) / int(row["n"])
        for row in main
        if int(row["N_int"]) == 100
    )
    require(abs(pooled_reference - 0.2385) <= 5e-13, "population query-fraction check failed")
    checks["population_reference"] = {
        "by_n": references,
        "pooled": pooled_reference,
        "manuscript_value": 0.2385,
    }

    derived: list[dict[str, object]] = []
    for record in sorted(grouped, key=lambda item: (int(item["n"]), int(item["N_int"]))):
        row: dict[str, object] = {
            "n": int(record["n"]),
            "N_int": int(record["N_int"]),
            "independent_source_problems": int(record["independent_source_problem_units"]),
            "population_T_q_over_n_mean": references[int(record["n"])],
        }
        for metric in ("excess_KL", "parent_f1", "mechanism_l2_error", "T_q_over_n"):
            values = _metric(record, metric)
            row[f"{metric}_mean"] = float(values["mean"])
            row[f"{metric}_ci95_low"] = float(values["bootstrap_ci95"][0])
            row[f"{metric}_ci95_high"] = float(values["bootstrap_ci95"][1])
            row[f"{metric}_available"] = int(values["count"])
            row[f"{metric}_missing"] = int(values["missing"])
        derived.append(row)
    fields = list(derived[0])
    write_csv(data_dir / "finite_sample_main.csv", derived, fields)

    apply_publication_style(8.5)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.45), sharex=True)
    specs = (
        ("excess_KL", "Excess KL above $K_B^*$", "(a) Matching error"),
        ("parent_f1", "Queried-parent F1", "(b) Parent recovery"),
        ("mechanism_l2_error", "Local-mechanism $L_2$ error", "(c) Mechanism error"),
        ("T_q_over_n", "Acquisition query fraction $T_q/n$", "(d) Query burden"),
    )
    for axis, (metric, ylabel, title) in zip(axes.flat, specs, strict=True):
        for n in N_VALUES:
            rows = [row for row in derived if row["n"] == n]
            color, marker, linestyle = N_STYLES[n]
            x = np.asarray([row["N_int"] for row in rows], dtype=float)
            mean = np.asarray([row[f"{metric}_mean"] for row in rows], dtype=float)
            low = np.asarray([row[f"{metric}_ci95_low"] for row in rows], dtype=float)
            high = np.asarray([row[f"{metric}_ci95_high"] for row in rows], dtype=float)
            axis.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0, zorder=1)
            axis.plot(
                x,
                mean,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                label=f"n={n}",
                zorder=3,
            )
            if metric == "T_q_over_n":
                axis.axhline(
                    references[n],
                    color=color,
                    linewidth=1.15,
                    linestyle=":",
                    alpha=0.95,
                    zorder=2,
                )
        axis.set_xscale("log")
        axis.set_xlim(82, 12200)
        axis.set_xticks(N_INT_GRID, N_INT_LABELS)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", pad=4)
        style_axis(axis)
    axes[0, 0].axhline(0.0, color=PALETTE["neutral"], linewidth=0.8, zorder=0)
    axes[0, 0].set_ylim(-0.015, 0.80)
    axes[0, 1].set_ylim(0.70, 1.01)
    axes[1, 0].set_ylim(0.0, 0.245)
    axes[1, 1].set_ylim(0.15, 0.57)
    for axis in axes[1, :]:
        axis.set_xlabel("Samples per queried singleton intervention, $N_{int}$")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=N_STYLES[n][0],
            marker=N_STYLES[n][1],
            linestyle=N_STYLES[n][2],
            markerfacecolor="white",
            markeredgecolor=N_STYLES[n][0],
            label=f"n={n}",
        )
        for n in N_VALUES
    ]
    legend_handles.append(
        Line2D([0], [0], color=PALETTE["neutral"], linestyle=":", label="Exact-population mean")
    )
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.945), pad=0.55, w_pad=1.05, h_pad=1.0)
    save_figure(fig, figure_dir / "finite_sample_main_4panel")
    plt.close(fig)
    return checks


def deterministic_jitter(key: str, width: float) -> float:
    value = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return width * (2.0 * (value / (2**64 - 1)) - 1.0)


def build_learned_graph(
    robustness_manifest_path: Path,
    robustness_results_path: Path,
    data_dir: Path,
    figure_dir: Path,
) -> dict[str, object]:
    manifest = json.loads(robustness_manifest_path.read_text(encoding="utf-8"))
    estimator = manifest["frozen_configuration"]["pc_estimator"]
    require(estimator["name"] == "repository_fisher_z_pc", "unexpected frozen PC estimator")
    require(estimator["significance_level"] == 0.05, "unexpected frozen PC alpha")
    require(
        estimator["maximum_conditioning_set_size"] == 2,
        "unexpected frozen PC conditioning-set limit",
    )
    raw = read_csv(robustness_results_path)
    selected = [row for row in raw if row["campaign"] == "campaign_c_estimated_cpdag"]
    require(len(selected) == 900, "expected 900 learned-graph rows")
    require(all(row["status"] == "SUCCESS" for row in selected), "learned-graph rows include failures")
    require(
        Counter(int(row["N_graph"]) for row in selected) == Counter({x: 150 for x in N_GRAPH_GRID}),
        "learned-graph N_graph cells are incomplete",
    )
    require(len({row["source_problem_id"] for row in selected}) == 150, "expected 150 paired source problems")
    for row in selected:
        metadata = json.loads(row["structure_estimation_metadata"])
        require(metadata["provider"] == "estimated_pc_fisher_z", "unexpected graph estimator")

    points: list[dict[str, object]] = []
    for row in selected:
        retained = row["graph_retains_all_true_adjacencies"] == "True"
        points.append(
            {
                "source_problem_id": row["source_problem_id"],
                "N_graph": int(row["N_graph"]),
                "retention_stratum": "all_true_adjacencies_retained" if retained else "missing_true_adjacency",
                "excess_KL": float(row["excess_KL"]),
                "queried_parent_f1": float(row["queried_parent_f1"]),
                "catastrophic_failure": row["catastrophic_failure"] == "True",
                "n": int(row["n"]),
                "graph_family": row["graph_family"],
            }
        )
    write_csv(data_dir / "learned_graph_retention_points.csv", points, list(points[0]))

    summaries: list[dict[str, object]] = []
    for n_graph in N_GRAPH_GRID:
        for stratum in ("all_true_adjacencies_retained", "missing_true_adjacency"):
            rows = [
                row
                for row in points
                if row["N_graph"] == n_graph and row["retention_stratum"] == stratum
            ]
            values = np.asarray([row["excess_KL"] for row in rows], dtype=float)
            summaries.append(
                {
                    "N_graph": n_graph,
                    "retention_stratum": stratum,
                    "source_problems": len(rows),
                    "mean_excess_KL": float(np.mean(values)),
                    "median_excess_KL": float(np.median(values)),
                    "q25_excess_KL": float(np.quantile(values, 0.25)),
                    "q75_excess_KL": float(np.quantile(values, 0.75)),
                    "q90_excess_KL": float(np.quantile(values, 0.90)),
                    "q95_excess_KL": float(np.quantile(values, 0.95)),
                    "maximum_excess_KL": float(np.max(values)),
                    "catastrophic_failures": sum(bool(row["catastrophic_failure"]) for row in rows),
                }
            )
    write_csv(data_dir / "learned_graph_retention_summary.csv", summaries, list(summaries[0]))

    def summary_at(n_graph: int, stratum: str) -> dict[str, object]:
        return next(
            row
            for row in summaries
            if row["N_graph"] == n_graph and row["retention_stratum"] == stratum
        )

    checks: dict[str, object] = {"passed": True, "by_N_graph": {}}
    for n_graph in N_GRAPH_GRID:
        overall = [row for row in points if row["N_graph"] == n_graph]
        checks["by_N_graph"][str(n_graph)] = {
            "overall_parent_f1_mean": statistics.fmean(row["queried_parent_f1"] for row in overall),
            "overall_excess_KL_mean": statistics.fmean(row["excess_KL"] for row in overall),
            "retained": summary_at(n_graph, "all_true_adjacencies_retained"),
            "missing": summary_at(n_graph, "missing_true_adjacency"),
        }
    require(abs(checks["by_N_graph"]["250"]["overall_parent_f1_mean"] - 0.640) < 0.001, "N_graph=250 F1 check failed")
    require(abs(checks["by_N_graph"]["10000"]["overall_parent_f1_mean"] - 0.801) < 0.001, "N_graph=10000 F1 check failed")
    require(abs(checks["by_N_graph"]["250"]["overall_excess_KL_mean"] - 21.75) < 0.01, "N_graph=250 excess-KL check failed")
    require(abs(checks["by_N_graph"]["10000"]["overall_excess_KL_mean"] - 7.88) < 0.01, "N_graph=10000 excess-KL check failed")
    require(summary_at(10000, "all_true_adjacencies_retained")["source_problems"] == 47, "N_graph=10000 retained count failed")
    require(summary_at(10000, "missing_true_adjacency")["source_problems"] == 103, "N_graph=10000 missing count failed")
    require(abs(summary_at(10000, "all_true_adjacencies_retained")["mean_excess_KL"] - 0.00957) < 5e-6, "N_graph=10000 retained mean failed")
    require(abs(summary_at(10000, "missing_true_adjacency")["mean_excess_KL"] - 11.47) < 0.01, "N_graph=10000 missing mean failed")
    require(abs(summary_at(1000, "all_true_adjacencies_retained")["mean_excess_KL"] - 0.0339) < 5e-5, "N_graph=1000 retained mean failed")
    require(abs(summary_at(1000, "missing_true_adjacency")["mean_excess_KL"] - 20.67) < 0.01, "N_graph=1000 missing mean failed")

    apply_publication_style(9.0)
    fig, axis = plt.subplots(figsize=(7.15, 4.35))
    styles = {
        "all_true_adjacencies_retained": (PALETTE["blue_main"], "o", -0.17, "All true adjacencies retained"),
        "missing_true_adjacency": (PALETTE["red_strong"], "x", 0.17, "At least one true adjacency missing"),
    }
    x_positions = np.arange(len(N_GRAPH_GRID), dtype=float)
    for stratum, (color, marker, offset, _) in styles.items():
        for index, n_graph in enumerate(N_GRAPH_GRID):
            rows = [
                row
                for row in points
                if row["N_graph"] == n_graph and row["retention_stratum"] == stratum
            ]
            x = [
                index
                + offset
                + deterministic_jitter(f"{row['source_problem_id']}:{n_graph}:{stratum}", 0.095)
                for row in rows
            ]
            axis.scatter(
                x,
                [row["excess_KL"] for row in rows],
                s=13 if marker == "o" else 17,
                marker=marker,
                facecolors="none" if marker == "o" else color,
                edgecolors=color if marker == "o" else None,
                color=color if marker != "o" else None,
                linewidths=0.65,
                alpha=0.38,
                rasterized=False,
                zorder=2,
            )
            summary_row = summary_at(n_graph, stratum)
            center = index + offset
            median = float(summary_row["median_excess_KL"])
            mean = float(summary_row["mean_excess_KL"])
            axis.plot([center - 0.105, center + 0.105], [median, median], color=color, linewidth=2.5, solid_capstyle="round", zorder=5)
            axis.scatter([center], [mean], marker="D", s=30, color=color, edgecolor="white", linewidth=0.65, zorder=6)

    axis.axhline(0.0, color=PALETTE["dark"], linewidth=0.8, zorder=1)
    axis.set_yscale("symlog", linthresh=0.01, linscale=0.8, base=10)
    axis.set_ylim(-0.003, 1800)
    axis.yaxis.set_major_locator(FixedLocator([0, 0.01, 0.1, 1, 10, 100, 1000]))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if value == 0 else f"{value:g}"))
    count_labels = []
    for n_graph, label in zip(N_GRAPH_GRID, N_GRAPH_LABELS, strict=True):
        retained_n = summary_at(n_graph, "all_true_adjacencies_retained")["source_problems"]
        missing_n = summary_at(n_graph, "missing_true_adjacency")["source_problems"]
        count_labels.append(f"{label}\n{retained_n} / {missing_n}")
    axis.set_xticks(x_positions, count_labels)
    axis.set_xlim(-0.55, len(N_GRAPH_GRID) - 0.45)
    axis.set_xlabel("Graph-learning sample size, $N_{graph}$\n(number retained / missing)")
    axis.set_ylabel("Excess KL above $K_B^*$ (symmetric-log scale)")
    fig.suptitle(
        "Learned-graph failure mode: all 900 runs",
        x=0.09,
        y=0.988,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    style_axis(axis)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=PALETTE["blue_main"], label="All adjacencies retained"),
        Line2D([0], [0], marker="x", linestyle="none", color=PALETTE["red_strong"], label="Missing adjacency"),
        Line2D([0], [0], color=PALETTE["dark"], linewidth=2.5, label="Median"),
        Line2D([0], [0], marker="D", linestyle="none", color=PALETTE["dark"], markeredgecolor="white", label="Arithmetic mean"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.94),
        ncol=4,
        columnspacing=1.0,
        handletextpad=0.45,
        fontsize=7.7,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84), pad=0.7)
    save_figure(fig, figure_dir / "learned_graph_retention_failure")
    plt.close(fig)
    return checks


def build_query_burden(
    acquisition_results_path: Path,
    data_dir: Path,
    figure_dir: Path,
) -> dict[str, object]:
    raw = read_csv(acquisition_results_path)
    regimes = {"exact_cpdag", "cpdag_revealed_25", "cpdag_revealed_50"}
    selected = [
        row
        for row in raw
        if row["panel"] == "main"
        and row["method"] == "public_adaptive_local_mechanism_dp"
        and row["initial_graph_regime"] in regimes
    ]
    require(len(selected) == 2700, "expected 2700 non-control acquisition rows")
    require(Counter(row["policy"] for row in selected) == Counter({"first_unresolved": 1350, "max_undirected_degree": 1350}), "policy rows are incomplete")
    require(len({(row["n"], row["graph_family"], row["replicate"]) for row in selected}) == 450, "expected 450 source problems")

    points: list[dict[str, object]] = []
    for row in selected:
        n = int(row["n"])
        points.append(
            {
                "source_problem_key": f"n{n}:{row['graph_family']}:r{row['replicate']}",
                "n": n,
                "graph_family": row["graph_family"],
                "initial_graph_regime": row["initial_graph_regime"],
                "policy": row["policy"],
                "initial_undirected_edges": int(row["initial_undirected_edges"]),
                "initial_undirected_edges_per_node": int(row["initial_undirected_edges"]) / n,
                "T_q": int(row["T"]),
                "T_q_over_n": float(row["T_over_n"]),
            }
        )
    write_csv(data_dir / "query_burden_points.csv", points, list(points[0]))

    x_all = np.asarray([row["initial_undirected_edges_per_node"] for row in points], dtype=float)
    y_all = np.asarray([row["T_q_over_n"] for row in points], dtype=float)
    pooled_r = float(np.corrcoef(x_all, y_all)[0, 1])
    require(abs(pooled_r - 0.8808929273688348) < 1e-12, "normalized ambiguity correlation check failed")

    family_means: dict[str, dict[str, float]] = defaultdict(dict)
    for family in FAMILY_STYLES:
        for policy in ("first_unresolved", "max_undirected_degree"):
            values = [
                row["T_q_over_n"]
                for row in points
                if row["graph_family"] == family and row["policy"] == policy
            ]
            require(len(values) == 270, f"unexpected family-policy count: {family}, {policy}")
            family_means[family][policy] = statistics.fmean(values)
    require(abs(family_means["tree_of_cliques:4"]["first_unresolved"] - 0.4323) < 5e-5, "tree-of-cliques first-policy check failed")
    require(abs(family_means["tree_of_cliques:4"]["max_undirected_degree"] - 0.4070) < 5e-5, "tree-of-cliques max-degree check failed")
    require(abs(family_means["erdos_renyi:2"]["first_unresolved"] - 0.0817) < 5e-5, "dense-ER first-policy check failed")
    require(abs(family_means["erdos_renyi:2"]["max_undirected_degree"] - 0.0681) < 5e-5, "dense-ER max-degree check failed")

    table_expectations = {
        "exact_cpdag": {"first_unresolved": 0.2445, "max_undirected_degree": 0.2131},
        "cpdag_revealed_25": {"first_unresolved": 0.1713, "max_undirected_degree": 0.1529},
        "cpdag_revealed_50": {"first_unresolved": 0.1171, "max_undirected_degree": 0.1086},
    }
    table_means: dict[str, dict[str, float]] = {}
    for regime, expected_by_policy in table_expectations.items():
        table_means[regime] = {}
        for policy, expected in expected_by_policy.items():
            values = [
                row["T_q_over_n"]
                for row in points
                if row["initial_graph_regime"] == regime and row["policy"] == policy
            ]
            require(len(values) == 450, f"unexpected Table 1 cell count: {regime}, {policy}")
            observed = statistics.fmean(values)
            require(abs(observed - expected) < 5e-5, f"Table 1 check failed: {regime}, {policy}")
            table_means[regime][policy] = observed
    controls = [
        row
        for row in raw
        if row["panel"] == "main"
        and row["method"] == "public_adaptive_local_mechanism_dp"
        and row["initial_graph_regime"] == "fully_directed_control"
    ]
    require(len(controls) == 900 and all(float(row["T_over_n"]) == 0.0 for row in controls), "fully directed control check failed")
    apply_publication_style(8.4)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True, sharey=True)
    policies = (
        ("first_unresolved", "(a) First unresolved"),
        ("max_undirected_degree", "(b) Max undirected degree"),
    )
    for axis, (policy, title) in zip(axes, policies, strict=True):
        policy_rows = [row for row in points if row["policy"] == policy]
        for family, (color, marker, _) in FAMILY_STYLES.items():
            family_rows = [row for row in policy_rows if row["graph_family"] == family]
            overlap: Counter[tuple[float, float]] = Counter(
                (row["initial_undirected_edges_per_node"], row["T_q_over_n"])
                for row in family_rows
            )
            coordinates = sorted(overlap)
            axis.scatter(
                [item[0] for item in coordinates],
                [item[1] for item in coordinates],
                s=[8 + 3.2 * overlap[item] for item in coordinates],
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.56,
                label=FAMILY_STYLES[family][2],
                zorder=2,
            )
        x = np.asarray([row["initial_undirected_edges_per_node"] for row in policy_rows], dtype=float)
        y = np.asarray([row["T_q_over_n"] for row in policy_rows], dtype=float)
        r_value = float(np.corrcoef(x, y)[0, 1])
        axis.text(
            0.98,
            0.05,
            f"pooled descriptive r = {r_value:.2f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.4,
            color=PALETTE["dark"],
        )
        axis.set_title(title, loc="left", pad=4)
        axis.set_xlabel("Initial undirected edges per node")
        axis.set_xlim(-0.05, 3.3)
        axis.set_ylim(-0.02, 0.84)
        style_axis(axis)
    axes[0].set_ylabel("Acquisition query fraction $T_q/n$")
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            color=color,
            markeredgecolor="white",
            label=label,
        )
        for color, marker, label in FAMILY_STYLES.values()
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=5, columnspacing=0.9, handletextpad=0.3)
    fig.text(0.995, 0.005, "Marker area reflects overlapping observations", ha="right", va="bottom", fontsize=7.0, color=PALETTE["neutral"])
    fig.tight_layout(rect=(0, 0.025, 1, 0.91), pad=0.6, w_pad=0.8)
    save_figure(fig, figure_dir / "query_burden_orientation_ambiguity")
    plt.close(fig)
    return {
        "passed": True,
        "pooled_descriptive_pearson_r_edges_per_node": pooled_r,
        "table_1_means": table_means,
        "fully_directed_control_rows": len(controls),
        "family_policy_means": family_means,
    }


def main() -> None:
    args = parse_args()
    source_root = args.input_root.resolve()
    output_root = prepare_output_directory(
        args.output_root, overwrite=args.overwrite, protected=(source_root,)
    )
    data_dir = output_root / "data"
    figure_dir = output_root / "figures"
    inputs = {
        "finite_sample_summary": source_root / "outputs/finite_sample_algorithm1_stage_b_v2/tables/stage_b_summary.json",
        "finite_sample_results": source_root / "outputs/finite_sample_algorithm1_stage_b_v2/machine_readable/results.csv",
        "learned_graph_manifest": source_root / "outputs/finite_sample_robustness_v1/full/manifest.json",
        "learned_graph_results": source_root / "outputs/finite_sample_robustness_v1/full/analysis/results.csv",
        "acquisition_results": source_root / "outputs/kl_query_complexity_v1_local_mechanism_full/results.csv",
    }
    for name, path in inputs.items():
        require(path.is_file(), f"missing input {name}: {path}")

    checks = {
        "finite_sample": build_finite_sample(
            inputs["finite_sample_summary"],
            inputs["finite_sample_results"],
            data_dir,
            figure_dir,
        ),
        "learned_graph": build_learned_graph(
            inputs["learned_graph_manifest"],
            inputs["learned_graph_results"],
            data_dir,
            figure_dir,
        ),
        "query_burden": build_query_burden(
            inputs["acquisition_results"],
            data_dir,
            figure_dir,
        ),
    }
    write_json(data_dir / "numeric_checks.json", checks)
    write_json(
        data_dir / "input_checksums.json",
        {
            name: {
                "path_relative_to_input_root": str(path.relative_to(source_root)),
                "sha256": sha256(path),
            }
            for name, path in inputs.items()
        },
    )
    print(f"Built 3 figures (PDF + PNG) under {figure_dir}")
    print(f"All numeric checks passed; audit: {data_dir / 'numeric_checks.json'}")


if __name__ == "__main__":
    main()
