#!/usr/bin/env python3
"""Validate and plot the completed query-complexity campaign.

This script is intentionally derivative-only. It reads the completed campaign,
checks its paper summaries against the raw rows and audited headline values, and
writes figure source tables plus new paper figures. It never runs campaign code,
changes raw campaign artifacts, or edits manuscript sources.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/kl_query_complexity_v1_local_mechanism_full"
SUMMARY = CAMPAIGN / "paper_summary"
FIGURE_DATA = SUMMARY / "figures"
FIGURES = ROOT / "Figures"
REQUIRED_SUMMARY_FILES = (
    "audit_summary.json",
    "aggregate_by_information_regime.csv",
    "aggregate_by_graph_family.csv",
    "aggregate_by_n.csv",
    "paired_policy_differences.csv",
    "exactness_panel_by_target_family.csv",
    "aggregate_by_graph_information_policy.csv",
    "aggregate_by_n_information.csv",
    "aggregate_by_policy.csv",
    "paired_policy_summary.csv",
)

METHOD = "public_adaptive_local_mechanism_dp"
FIRST = "first_unresolved"
DEGREE = "max_undirected_degree"
CONTROL = "fully_directed_control"
NONCONTROL = ("exact_cpdag", "cpdag_revealed_25", "cpdag_revealed_50")
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_DRAWS = 10_000
ROUNDING_TOLERANCE = 5.1e-6
PNG_DPI = 360

POLICY_LABEL = {
    FIRST: "First unresolved",
    DEGREE: "Max undirected degree",
}
POLICY_STYLE = {
    FIRST: {"color": "#0072B2", "marker": "o"},
    DEGREE: {"color": "#D55E00", "marker": "s"},
}
INFO_LABEL = {
    "exact_cpdag": "Exact CPDAG",
    "cpdag_revealed_25": "+25% seeded",
    "cpdag_revealed_50": "+50% seeded",
    CONTROL: "Fully directed",
}
INFO_ORDER = tuple(INFO_LABEL)
GRAPH_LABEL = {
    "erdos_renyi:2": "ER p=.35",
    "barabasi_albert:3": "Barabasi-Albert",
    "erdos_renyi:1": "ER p=.15",
    "random_tree:0": "Random tree",
    "tree_of_cliques:4": "Tree of cliques",
}
GRAPH_TICK_LABEL = {
    "erdos_renyi:2": "ER\np=.35",
    "barabasi_albert:3": "Barabasi-\nAlbert",
    "erdos_renyi:1": "ER\np=.15",
    "random_tree:0": "Random\ntree",
    "tree_of_cliques:4": "Tree of\ncliques",
}
GRAPH_ORDER = tuple(GRAPH_LABEL)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise AssertionError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resolved_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is not a file: {path}")
    return path.resolve(strict=True)


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _preflight(campaign_arg: Path, output_arg: Path, overwrite: bool) -> tuple[Path, Path]:
    """Protect resolved campaign inputs before a table or figure directory exists."""
    campaign = campaign_arg.resolve(strict=True)
    if not campaign.is_dir():
        raise NotADirectoryError(f"campaign is not a directory: {campaign_arg}")
    inputs = [_resolved_file(campaign / "results.csv", "campaign results")]
    inputs.extend(_resolved_file(campaign / "paper_summary" / name, name) for name in REQUIRED_SUMMARY_FILES)
    output = output_arg.resolve(strict=False)
    if _overlaps(output, campaign):
        raise ValueError("figure output must be disjoint from the campaign input root")
    targets = [
        output / "paper_summary/figures" / name for name in (
            "figure1_information_regime.csv",
            "figure2_graph_information_policy.csv",
            "figure3_size_trend.csv",
            "figureA_policy_difference.csv",
        )
    ] + [
        output / "Figures" / f"{stem}.{suffix}"
        for stem in (
            "query_complexity__query_fraction_by_information_regime",
            "query_complexity__graph_family_by_information_and_policy",
            "query_complexity__empirical_size_trend",
            "query_complexity__paired_policy_query_difference",
        )
        for suffix in ("png", "pdf")
    ]
    resolved_targets = [target.resolve(strict=False) for target in targets]
    if any(_overlaps(target, campaign) for target in resolved_targets):
        raise ValueError("figure output target must be disjoint from the campaign input root")
    if any(target in inputs for target in resolved_targets):
        raise ValueError("figure output target resolves to a protected campaign input")
    existing = [target for target in targets if target.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to replace existing outputs: {existing}")
    return campaign, output


def assert_close(actual: float, expected: float, label: str, atol: float = ROUNDING_TOLERANCE) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
        raise AssertionError(f"{label}: expected {expected!r}, found {actual!r}")


def mean(rows: Sequence[dict[str, str]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def median(rows: Sequence[dict[str, str]], field: str) -> float:
    return float(np.median([float(row[field]) for row in rows]))


def bootstrap_mean_ci(values: Iterable[float]) -> tuple[float, float]:
    """Percentile CI, resampling the displayed cell's campaign rows."""
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        raise AssertionError("Cannot bootstrap an empty group")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means: list[np.ndarray] = []
    chunk = 500
    for start in range(0, BOOTSTRAP_DRAWS, chunk):
        draws = min(chunk, BOOTSTRAP_DRAWS - start)
        indices = rng.integers(0, len(array), size=(draws, len(array)))
        means.append(np.mean(array[indices], axis=1))
    low, high = np.quantile(np.concatenate(means), [0.025, 0.975])
    return float(low), float(high)


def select(rows: Sequence[dict[str, str]], **criteria: object) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if all(str(row.get(field)) == str(value) for field, value in criteria.items())
    ]


def summary_value(
    rows: Sequence[dict[str, str]], keys: dict[str, object], field: str
) -> float:
    matches = select(rows, **keys)
    if len(matches) != 1:
        raise AssertionError(f"Expected one summary row for {keys}, found {len(matches)}")
    return float(matches[0][field])


def validate_inputs() -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    missing = [name for name in REQUIRED_SUMMARY_FILES if not (SUMMARY / name).is_file()]
    if missing:
        raise AssertionError(f"Missing authoritative inputs: {missing}")

    audit = json.loads((SUMMARY / "audit_summary.json").read_text())
    if audit.get("verdict") != "PASS":
        raise AssertionError(f"Campaign audit verdict is not PASS: {audit.get('verdict')!r}")
    if audit["counts"]["actual_main_rows"] != 4500:
        raise AssertionError("Campaign does not contain the audited 4,500 main rows")
    if audit["counts"]["failed_or_incomplete_rows"] != 0:
        raise AssertionError("Campaign contains failed or incomplete rows")
    assert_close(audit["exactness"]["max_abs_KL_gap"], 2.1280754936015e-12,
                 "campaign-wide maximum absolute KL gap", atol=1e-25)

    tables = {
        name.removesuffix(".csv"): read_csv(SUMMARY / name)
        for name in REQUIRED_SUMMARY_FILES
        if name.endswith(".csv")
    }
    if any(not table for table in tables.values()):
        raise AssertionError("At least one required paper-summary table is empty")

    raw = read_csv(CAMPAIGN / "results.csv")
    if len(raw) != 4512:
        raise AssertionError(f"Expected 4,512 total rows, found {len(raw)}")
    main = [row for row in raw if row["panel"] == "main"]
    adaptive = [row for row in main if row["method"] == METHOD]
    if len(main) != 4500 or len(adaptive) != 3600:
        raise AssertionError(f"Unexpected row counts: main={len(main)}, adaptive={len(adaptive)}")
    for regime in INFO_ORDER:
        for policy in POLICY_LABEL:
            group = select(adaptive, initial_graph_regime=regime, policy=policy)
            if len(group) != 450:
                raise AssertionError(f"Expected 450 rows for {regime}/{policy}, found {len(group)}")

    noncontrol = [row for row in adaptive if row["initial_graph_regime"] in NONCONTROL]
    if len(noncontrol) != 2700:
        raise AssertionError(f"Expected 2,700 non-control adaptive rows, found {len(noncontrol)}")

    information_expected = {
        ("exact_cpdag", FIRST): 0.24450,
        ("exact_cpdag", DEGREE): 0.21306,
        ("cpdag_revealed_25", FIRST): 0.17128,
        ("cpdag_revealed_25", DEGREE): 0.15294,
        ("cpdag_revealed_50", FIRST): 0.11706,
        ("cpdag_revealed_50", DEGREE): 0.10856,
        (CONTROL, FIRST): 0.0,
        (CONTROL, DEGREE): 0.0,
    }
    information_table = tables["aggregate_by_information_regime"]
    for (regime, policy), expected in information_expected.items():
        group = select(adaptive, initial_graph_regime=regime, policy=policy)
        raw_value = mean(group, "T_over_n")
        summary_mean = summary_value(
            information_table,
            {"initial_graph_regime": regime, "policy": policy},
            "T_over_n_mean",
        )
        assert_close(raw_value, summary_mean, f"raw/table information mean {regime}/{policy}", atol=1e-14)
        assert_close(raw_value, expected, f"information headline {regime}/{policy}")

    overall_expected = {
        FIRST: (0.17761, 0.10),
        DEGREE: (0.15819, 0.10),
    }
    policy_table = tables["aggregate_by_policy"]
    for policy, (expected_mean, expected_median) in overall_expected.items():
        group = select(noncontrol, policy=policy)
        raw_mean = mean(group, "T_over_n")
        raw_median = median(group, "T_over_n")
        assert_close(raw_mean, expected_mean, f"overall mean {policy}")
        assert_close(raw_median, expected_median, f"overall median {policy}")
        assert_close(
            raw_mean,
            summary_value(
                policy_table,
                {"scope": "excluding_fully_directed", "policy": policy},
                "T_over_n_mean",
            ),
            f"raw/table overall mean {policy}",
            atol=1e-14,
        )

    family_expected = {
        ("erdos_renyi:2", FIRST): 0.08167,
        ("erdos_renyi:2", DEGREE): 0.06815,
        ("barabasi_albert:3", FIRST): 0.09741,
        ("barabasi_albert:3", DEGREE): 0.08731,
        ("erdos_renyi:1", FIRST): 0.11306,
        ("erdos_renyi:1", DEGREE): 0.09139,
        ("random_tree:0", FIRST): 0.16361,
        ("random_tree:0", DEGREE): 0.13704,
        ("tree_of_cliques:4", FIRST): 0.43231,
        ("tree_of_cliques:4", DEGREE): 0.40704,
    }
    family_table = tables["aggregate_by_graph_family"]
    for (family, policy), expected in family_expected.items():
        group = select(noncontrol, graph_family=family, policy=policy)
        raw_value = mean(group, "T_over_n")
        table_value = summary_value(
            family_table, {"graph_family": family, "policy": policy}, "T_over_n_mean"
        )
        assert_close(raw_value, table_value, f"raw/table graph mean {family}/{policy}", atol=1e-14)
        assert_close(raw_value, expected, f"graph-family headline {family}/{policy}")

    size_expected = {
        (10, FIRST): 0.20444,
        (10, DEGREE): 0.18044,
        (20, FIRST): 0.17467,
        (20, DEGREE): 0.15411,
        (40, FIRST): 0.15372,
        (40, DEGREE): 0.14000,
    }
    size_table = tables["aggregate_by_n"]
    for (n, policy), expected in size_expected.items():
        group = select(noncontrol, n=n, policy=policy)
        raw_value = mean(group, "T_over_n")
        table_value = summary_value(size_table, {"n": n, "policy": policy}, "T_over_n_mean")
        assert_close(raw_value, table_value, f"raw/table n mean {n}/{policy}", atol=1e-14)
        assert_close(raw_value, expected, f"size headline {n}/{policy}")

    graph_information_table = tables["aggregate_by_graph_information_policy"]
    for regime in NONCONTROL:
        for family in GRAPH_ORDER:
            for policy in POLICY_LABEL:
                group = select(
                    noncontrol,
                    initial_graph_regime=regime,
                    graph_family=family,
                    policy=policy,
                )
                if len(group) != 90:
                    raise AssertionError(
                        f"Expected 90 rows for {regime}/{family}/{policy}, found {len(group)}"
                    )
                table_value = summary_value(
                    graph_information_table,
                    {
                        "graph_family": family,
                        "initial_graph_regime": regime,
                        "policy": policy,
                    },
                    "T_over_n_mean",
                )
                assert_close(
                    mean(group, "T_over_n"),
                    table_value,
                    f"raw/table graph-information mean {regime}/{family}/{policy}",
                    atol=1e-14,
                )

    pairs = tables["paired_policy_differences"]
    paired_noncontrol = [row for row in pairs if row["initial_graph_regime"] != CONTROL]
    if len(paired_noncontrol) != 1350:
        raise AssertionError(f"Expected 1,350 non-control pairs, found {len(paired_noncontrol)}")
    deltas = [float(row["delta_T_first_minus_degree"]) for row in paired_noncontrol]
    delta_counts = Counter(deltas)
    paired_low, paired_high = bootstrap_mean_ci(deltas)
    assert_close(float(np.mean(deltas)), 0.400, "paired mean delta T", atol=1e-14)
    assert_close(paired_low, 0.360, "paired bootstrap lower CI", atol=1e-14)
    assert_close(paired_high, 0.44296296296296295, "paired bootstrap upper CI", atol=1e-14)
    if sum(count for value, count in delta_counts.items() if value > 0) != 407:
        raise AssertionError("Paired degree-win count does not equal 407")
    if delta_counts[0.0] != 913:
        raise AssertionError("Paired tie count does not equal 913")
    if sum(count for value, count in delta_counts.items() if value < 0) != 30:
        raise AssertionError("Paired degree-loss count does not equal 30")
    assert_close(max(deltas), 5.0, "maximum degree-policy saving", atol=0.0)
    assert_close(min(deltas), -2.0, "maximum degree-policy disadvantage", atol=0.0)

    pair_summary = select(
        tables["paired_policy_summary"], scope="overall_excluding_fully_directed", stratum="all"
    )
    if len(pair_summary) != 1:
        raise AssertionError("Missing overall paired-policy summary")
    for field, actual in (
        ("mean_delta_T_first_minus_degree", np.mean(deltas)),
        ("bootstrap_mean_delta_T_ci95_low", paired_low),
        ("bootstrap_mean_delta_T_ci95_high", paired_high),
    ):
        assert_close(float(pair_summary[0][field]), float(actual), f"raw/table paired {field}", atol=1e-14)

    exactness_panel = tables["exactness_panel_by_target_family"]
    panel_max = max(float(row["max_abs_KL_gap"]) for row in exactness_panel)
    assert_close(panel_max, 8.881784197001252e-16, "six-family exactness maximum", atol=1e-29)

    # The remaining requested tables are explicitly loaded and checked for expected coverage.
    if len(tables["aggregate_by_n_information"]) != 24:
        raise AssertionError("aggregate_by_n_information.csv has unexpected coverage")
    return tables, adaptive


def build_figure_tables(
    tables: dict[str, list[dict[str, str]]], adaptive: list[dict[str, str]]
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    figure1: list[dict] = []
    for regime in INFO_ORDER:
        for policy in POLICY_LABEL:
            group = select(adaptive, initial_graph_regime=regime, policy=policy)
            values = [float(row["T_over_n"]) for row in group]
            low, high = bootstrap_mean_ci(values)
            figure1.append(
                {
                    "information_regime": regime,
                    "information_label": INFO_LABEL[regime],
                    "policy": policy,
                    "policy_label": POLICY_LABEL[policy],
                    "n_rows": len(group),
                    "mean_T_over_n": float(np.mean(values)),
                    "median_T_over_n": float(np.median(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                }
            )

    figure2: list[dict] = []
    noncontrol = [row for row in adaptive if row["initial_graph_regime"] in NONCONTROL]
    for regime in NONCONTROL:
        for family in GRAPH_ORDER:
            for policy in POLICY_LABEL:
                group = select(
                    noncontrol,
                    initial_graph_regime=regime,
                    graph_family=family,
                    policy=policy,
                )
                values = [float(row["T_over_n"]) for row in group]
                low, high = bootstrap_mean_ci(values)
                figure2.append(
                    {
                        "information_regime": regime,
                        "information_label": INFO_LABEL[regime],
                        "graph_family": family,
                        "graph_family_label": GRAPH_LABEL[family],
                        "policy": policy,
                        "policy_label": POLICY_LABEL[policy],
                        "n_rows": len(group),
                        "mean_T_over_n": float(np.mean(values)),
                        "median_T_over_n": float(np.median(values)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "bootstrap_draws": BOOTSTRAP_DRAWS,
                    }
                )

    figure3: list[dict] = []
    for n in (10, 20, 40):
        for policy in POLICY_LABEL:
            group = select(noncontrol, n=n, policy=policy)
            raw_values = [float(row["T"]) for row in group]
            normalized_values = [float(row["T_over_n"]) for row in group]
            raw_low, raw_high = bootstrap_mean_ci(raw_values)
            normalized_low, normalized_high = bootstrap_mean_ci(normalized_values)
            figure3.append(
                {
                    "n": n,
                    "policy": policy,
                    "policy_label": POLICY_LABEL[policy],
                    "n_rows": len(group),
                    "mean_T": float(np.mean(raw_values)),
                    "median_T": float(np.median(raw_values)),
                    "T_ci95_low": raw_low,
                    "T_ci95_high": raw_high,
                    "mean_T_over_n": float(np.mean(normalized_values)),
                    "median_T_over_n": float(np.median(normalized_values)),
                    "T_over_n_ci95_low": normalized_low,
                    "T_over_n_ci95_high": normalized_high,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                }
            )

    pair_rows = [
        row
        for row in tables["paired_policy_differences"]
        if row["initial_graph_regime"] != CONTROL
    ]
    deltas = [int(float(row["delta_T_first_minus_degree"])) for row in pair_rows]
    low, high = bootstrap_mean_ci(deltas)
    counts = Counter(deltas)
    figure_a = [
        {
            "delta_T_first_minus_degree": delta,
            "outcome": "degree loses" if delta < 0 else "tie" if delta == 0 else "degree wins",
            "count": counts[delta],
            "total_pairs": len(deltas),
            "mean_delta_T": float(np.mean(deltas)),
            "median_delta_T": float(np.median(deltas)),
            "mean_delta_T_ci95_low": low,
            "mean_delta_T_ci95_high": high,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        }
        for delta in range(min(deltas), max(deltas) + 1)
    ]
    return figure1, figure2, figure3, figure_a


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "generate_query_complexity_paper_figures.py"}
    figure.savefig(FIGURES / f"{stem}.png", dpi=PNG_DPI, metadata=metadata)
    figure.savefig(FIGURES / f"{stem}.pdf", metadata=metadata)
    plt.close(figure)


def plot_figure1(rows: Sequence[dict]) -> None:
    figure, axis = plt.subplots(figsize=(6.9, 3.55))
    x = np.arange(len(INFO_ORDER), dtype=float)
    for policy in POLICY_LABEL:
        group = [row for regime in INFO_ORDER for row in rows
                 if row["information_regime"] == regime and row["policy"] == policy]
        values = np.asarray([row["mean_T_over_n"] for row in group])
        low = np.asarray([row["ci95_low"] for row in group])
        high = np.asarray([row["ci95_high"] for row in group])
        style = POLICY_STYLE[policy]
        axis.errorbar(
            x,
            values,
            yerr=np.vstack([values - low, high - values]),
            label=POLICY_LABEL[policy],
            color=style["color"],
            marker=style["marker"],
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            capsize=3.0,
            elinewidth=1.1,
        )
    axis.axhline(1.0, color="#666666", linewidth=1.0, linestyle=(0, (3, 2)), zorder=0)
    axis.text(
        0.015,
        0.940,
        "Graph-free n-shift reference",
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="#555555",
        fontsize=8.2,
    )
    axis.set_xticks(x, [INFO_LABEL[key] for key in INFO_ORDER])
    axis.set_ylabel(r"Query fraction $T/n$")
    axis.set_ylim(-0.025, 1.045)
    axis.set_xlim(-0.18, 3.18)
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.legend(
        loc="center right",
        bbox_to_anchor=(0.99, 0.70),
        frameon=False,
        handlelength=2.4,
    )
    clean_axis(axis)
    figure.subplots_adjust(left=0.105, right=0.98, bottom=0.18, top=0.96)
    save_figure(figure, "query_complexity__query_fraction_by_information_regime")


def plot_figure2(rows: Sequence[dict]) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(7.1, 5.75), sharex=True, sharey=True)
    x = np.arange(len(GRAPH_ORDER), dtype=float)
    offsets = {FIRST: -0.10, DEGREE: 0.10}
    for panel_index, (axis, regime) in enumerate(zip(axes, NONCONTROL)):
        for policy in POLICY_LABEL:
            group = [row for family in GRAPH_ORDER for row in rows
                     if row["information_regime"] == regime
                     and row["graph_family"] == family
                     and row["policy"] == policy]
            values = np.asarray([row["mean_T_over_n"] for row in group])
            low = np.asarray([row["ci95_low"] for row in group])
            high = np.asarray([row["ci95_high"] for row in group])
            style = POLICY_STYLE[policy]
            axis.errorbar(
                x + offsets[policy],
                values,
                yerr=np.vstack([values - low, high - values]),
                linestyle="none",
                label=POLICY_LABEL[policy],
                color=style["color"],
                marker=style["marker"],
                markersize=4.7,
                markerfacecolor="white",
                markeredgewidth=1.2,
                capsize=2.2,
                elinewidth=1.0,
            )
        axis.set_title(
            f"{chr(ord('A') + panel_index)}   {INFO_LABEL[regime]}",
            loc="left",
            pad=3,
            fontweight="bold",
        )
        axis.set_xticks(x, [GRAPH_TICK_LABEL[key] for key in GRAPH_ORDER])
        axis.tick_params(axis="x", pad=3)
        axis.set_xlim(-0.42, 4.42)
        axis.set_ylim(0.0, 0.68)
        axis.set_yticks([0.0, 0.2, 0.4, 0.6])
        clean_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                  bbox_to_anchor=(0.53, 0.995), handlelength=1.6)
    figure.text(0.018, 0.48, r"Mean query fraction $T/n$", rotation="vertical",
                va="center", ha="left", fontsize=9.5)
    figure.subplots_adjust(left=0.105, right=0.99, bottom=0.13, top=0.89, hspace=0.20)
    save_figure(figure, "query_complexity__graph_family_by_information_and_policy")


def plot_figure3(rows: Sequence[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(6.9, 3.2))
    specifications = (
        (axes[0], "mean_T", "T_ci95_low", "T_ci95_high", r"Mean queries $T$", (0, 7.3)),
        (
            axes[1],
            "mean_T_over_n",
            "T_over_n_ci95_low",
            "T_over_n_ci95_high",
            r"Mean query fraction $T/n$",
            (0, 0.235),
        ),
    )
    ns = np.asarray([10, 20, 40])
    for panel_index, (axis, mean_field, low_field, high_field, ylabel, ylim) in enumerate(specifications):
        for policy in POLICY_LABEL:
            group = [row for n in ns for row in rows if row["n"] == n and row["policy"] == policy]
            values = np.asarray([row[mean_field] for row in group])
            low = np.asarray([row[low_field] for row in group])
            high = np.asarray([row[high_field] for row in group])
            style = POLICY_STYLE[policy]
            axis.errorbar(
                ns,
                values,
                yerr=np.vstack([values - low, high - values]),
                label=POLICY_LABEL[policy],
                color=style["color"],
                marker=style["marker"],
                markersize=5.0,
                markerfacecolor="white",
                markeredgewidth=1.2,
                capsize=3.0,
                elinewidth=1.0,
            )
        axis.set_xlabel("Number of nodes, n")
        axis.set_ylabel(ylabel)
        axis.set_xticks(ns)
        axis.set_xlim(8, 42)
        axis.set_ylim(*ylim)
        clean_axis(axis)
        axis.text(-0.14, 1.02, chr(ord("A") + panel_index), transform=axis.transAxes,
                  fontweight="bold", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                  bbox_to_anchor=(0.53, 1.01), handlelength=2.2)
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.80, wspace=0.29)
    save_figure(figure, "query_complexity__empirical_size_trend")


def plot_figure_a(rows: Sequence[dict]) -> None:
    figure, axis = plt.subplots(figsize=(3.35, 2.75))
    deltas = np.asarray([row["delta_T_first_minus_degree"] for row in rows])
    counts = np.asarray([row["count"] for row in rows])
    colors = ["#B2182B" if value < 0 else "#888888" if value == 0 else "#2166AC" for value in deltas]
    bars = axis.bar(deltas, counts, width=0.72, color=colors, edgecolor="white", linewidth=0.6)
    axis.axvline(0, color="#555555", linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    axis.bar_label(bars, labels=[str(value) for value in counts], padding=2, fontsize=7.8)
    axis.set_xlabel(
        "Paired query difference\n" + r"$\Delta_T=T_{\mathrm{first}}-T_{\mathrm{degree}}$"
    )
    axis.set_ylabel("Paired problems")
    axis.set_xticks(deltas)
    axis.set_ylim(0, max(counts) * 1.13)
    axis.text(-1.45, max(counts) * 0.64, "Degree loses", color="#B2182B",
              ha="center", va="center", fontsize=7.8)
    axis.text(3.25, max(counts) * 0.64, "Degree wins", color="#2166AC",
              ha="center", va="center", fontsize=7.8)
    clean_axis(axis)
    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.245, top=0.96)
    save_figure(figure, "query_complexity__paired_policy_query_difference")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    campaign, output = _preflight(arguments.campaign, arguments.output_dir, arguments.overwrite)
    global CAMPAIGN, SUMMARY, FIGURE_DATA, FIGURES
    CAMPAIGN = campaign
    SUMMARY = CAMPAIGN / "paper_summary"
    FIGURE_DATA = output / "paper_summary/figures"
    FIGURES = output / "Figures"
    tables, adaptive = validate_inputs()
    figure1, figure2, figure3, figure_a = build_figure_tables(tables, adaptive)

    write_csv(FIGURE_DATA / "figure1_information_regime.csv", figure1)
    write_csv(FIGURE_DATA / "figure2_graph_information_policy.csv", figure2)
    write_csv(FIGURE_DATA / "figure3_size_trend.csv", figure3)
    write_csv(FIGURE_DATA / "figureA_policy_difference.csv", figure_a)

    configure_matplotlib()
    plot_figure1(figure1)
    plot_figure2(figure2)
    plot_figure3(figure3)
    plot_figure_a(figure_a)

    print("PASS: audited 4,500-row campaign; all numerical assertions passed")
    print(f"Wrote 4 source tables to {FIGURE_DATA}")
    print(f"Wrote 4 PNG/PDF figure pairs to {FIGURES} at {PNG_DPI} dpi")


if __name__ == "__main__":
    main()
