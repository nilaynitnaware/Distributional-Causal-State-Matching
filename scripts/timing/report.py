"""Build the Small-MEC distribution and exact-runtime report."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "outputs" / "small_mec_command_space_paired_v2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="timing CSV")
    parser.add_argument("--output", type=Path, required=True, help="report Markdown")
    return parser


def _format_seconds(row: dict[str, str], key: str) -> str:
    return f"{float(row[key]):.6f}" if row[key] else "—"


def main() -> None:
    args = _parser().parse_args()
    released = [
        json.loads(line)
        for line in (DATA_ROOT / "corrected_audit" / "per_instance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    with args.input.open(newline="", encoding="utf-8") as handle:
        timings = list(csv.DictReader(handle))
    frozen = [row for row in timings if row["part"] == "B"]
    generated = [row for row in timings if row["part"] == "C"]
    gap_values = sorted(
        {row["DAG_minus_match_minimax_gap"] for row in released}
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    description = subprocess.check_output(
        ["git", "describe", "--tags", "--always"], cwd=ROOT, text=True
    ).strip()
    uptime = subprocess.check_output(["uptime"], text=True).strip()

    lines = [
        "# Small-MEC distribution and exact runtime",
        "",
        f"Repository state `{description}` at commit `{commit}`. Single process, "
        "OMP/MKL/OpenBLAS threads set to 1; wall-clock seconds. End uptime: "
        f"{uptime}. Each timing row records start/end 1, 5, and 15 minute load "
        "averages; shared-host load is a limitation.",
        "",
        "## Part A: released corrected results (no reruns)",
        "",
        f"Instances: {len(released)}. Mean T_match = "
        f"{statistics.mean(row['T_match_star_worst_case'] for row in released):.6f}; "
        "mean T_DAG = "
        f"{statistics.mean(row['T_DAG_star_worst_case'] for row in released):.6f}; "
        "strict savings = "
        f"{sum(row['T_match_star_worst_case'] < row['T_DAG_star_worst_case'] for row in released)}.",
        "",
        "| Stratum | Group | "
        + " | ".join(f"gap {value}" for value in gap_values)
        + " | total |",
        "|---|---|" + "---:|" * (len(gap_values) + 1),
    ]
    for field in [None, "target_regime", "graph_family", "n"]:
        if field is None:
            groups = [("all", released)]
        else:
            groups = [
                (str(key), [row for row in released if row[field] == key])
                for key in sorted({row[field] for row in released})
            ]
        for key, items in groups:
            counts = collections.Counter(
                row["DAG_minus_match_minimax_gap"] for row in items
            )
            lines.append(
                "| "
                + ("overall" if field is None else field)
                + " | "
                + key
                + " | "
                + " | ".join(str(counts[value]) for value in gap_values)
                + f" | {len(items)} |"
            )

    lines += [
        "",
        "## Part B: retiming frozen instances",
        "",
        f"{len(frozen)} rows; "
        f"{sum(row['status'] == 'complete' for row in frozen)} completed. Frozen "
        "moments and true DAG were used; enumeration was checked against every "
        "archived candidate, and both computed complexities against the corrected "
        "archive. Reconstructing SCMs and diagnostic checks are outside the four "
        "stage timings. Optimization includes tied optima; matching includes "
        "command certification. The CSV contains all instance timings and metadata.",
        "",
        "| Stage | Median s | Maximum s | Maximum instance | Load at start → end (1, 5, 15 min) |",
        "|---|---:|---:|---|---|",
    ]
    for key, name in [
        ("enumeration_s", "MEC enumeration"),
        ("optimization_s", "Candidate optimization"),
        ("match_s", "T_match and certification"),
        ("dag_s", "T_DAG"),
    ]:
        complete_values = [row for row in frozen if row[key]]
        if not complete_values:
            lines.append(f"| {name} | — | — | — | — |")
            continue
        worst = max(complete_values, key=lambda row: float(row[key]))
        lines.append(
            f"| {name} | "
            f"{statistics.median(float(row[key]) for row in complete_values):.6f} | "
            f"{float(worst[key]):.6f} | {worst['instance_id']} | "
            f"{worst['load_start']} → {worst['load_end']} |"
        )
    if frozen:
        lines += [
            "",
            "The Part B per-instance loads span "
            + ", ".join(
                f"{min(float(row['load_start'].split()[index]) for row in frozen):.2f}–"
                f"{max(float(row['load_end'].split()[index]) for row in frozen):.2f}"
                for index in range(3)
            )
            + " (1, 5, 15 min); medians summarize runs across that range.",
        ]

    lines += [
        "",
        "## Part C: generated realizable instances",
        "",
        "Release generator and target, seed base 20261104, source seed "
        "`_stable_seed(20261104, \"source\", family, n, replicate, 0)` and "
        "target seed `_stable_seed(source_seed, \"target\", \"realisable\")`; "
        "coefficient magnitudes U(0.25, 0.65) with random signs, variances "
        "U(0.7, 1.4), zero intercept; B=2 and release bounds. Enumeration has "
        "no fixed edge cap and each accepted MEC includes the true DAG. Each "
        "selected instance has a 1,800-second wall-clock deadline for enumeration, "
        "optimization and both recursions, within a four-hour probe budget. Stop "
        "a bin after its first timeout. A timed-out stage is left blank, not treated "
        "as a completed timing.",
        "",
        "| Instance | n | family | MEC | undirected | enum s | optimization s | match s | DAG s | T_match | T_DAG | status | load start → end (1, 5, 15 min) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in generated:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["instance_id"],
                    row["n"],
                    row["family"],
                    row["mec_size"],
                    row["undirected_edges"],
                    *(
                        _format_seconds(row, key)
                        for key in (
                            "enumeration_s",
                            "optimization_s",
                            "match_s",
                            "dag_s",
                        )
                    ),
                    row["T_match"] or "—",
                    row["T_DAG"] or "—",
                    row["status"],
                    row["load_start"] + " → " + row["load_end"],
                ]
            )
            + " |"
        )
    completed_sizes = [
        int(row["mec_size"])
        for row in generated
        if row["status"] == "complete"
    ]
    lines += [
        "",
        f"Largest completed probe MEC: {max(completed_sizes) if completed_sizes else 'none'}. "
        "The full search limit is 10,000 draws per Erdős–Rényi size (the first "
        "250 in the first script and the remainder in the continuation), plus one "
        "path draw per size; no claim is made that all possible sources were "
        "covered. Each bin contains at most three selected sources. Empty bins "
        "mean no selected source was found, not a recursion timeout. The path "
        "generator produces an oriented collider-free path, whose MEC has exactly "
        "n DAGs, so all requested path sizes are below the smallest bin.",
        "",
        "Scripts: `scripts/timing/run_small_mec_timing.py`, "
        "`scripts/timing/common.py`, `scripts/timing/probe_generated_classes.py`, "
        "`scripts/timing/probe_generated_classes_more.py`, and "
        "`scripts/timing/report.py`. Timing CSV: `" + str(args.input) + "`.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
