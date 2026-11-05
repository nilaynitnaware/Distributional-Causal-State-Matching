"""Summarize the released Small-MEC gap T_DAG - T_match."""

from __future__ import annotations

import argparse
import collections
import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "compact" / "c_small_mec" / "paired_results.csv"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="released per-instance Small-MEC CSV",
    )
    return parser


def _counts(rows: list[dict[str, object]]) -> str:
    counts = collections.Counter(int(row["gap"]) for row in rows)
    return ", ".join(f"{gap}:{counts[gap]}" for gap in sorted(counts))


def main() -> None:
    args = _parser().parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    rows: list[dict[str, object]] = []
    for source in source_rows:
        t_match = int(source["corrected_T_match_star"])
        t_dag = int(source["T_DAG_star"])
        gap = t_dag - t_match
        if gap != int(source["corrected_DAG_minus_matching_gap"]):
            raise ValueError(f"inconsistent gap for {source['instance_id']}")
        rows.append(
            {
                "T_match": t_match,
                "T_DAG": t_dag,
                "gap": gap,
                "target type": source["target_regime"],
                "graph family": source["graph_family"],
                "n": int(source["n"]),
            }
        )

    print(f"overall: {_counts(rows)}")
    for field in ("target type", "graph family", "n"):
        print(f"{field}:")
        for value in sorted({row[field] for row in rows}):
            group = [row for row in rows if row[field] == value]
            print(f"  {value}: {_counts(group)}")
    print(
        "mean T_match: "
        f"{statistics.mean(int(row['T_match']) for row in rows):.6f}"
    )
    print(
        "mean T_DAG: "
        f"{statistics.mean(int(row['T_DAG']) for row in rows):.6f}"
    )


if __name__ == "__main__":
    main()
