"""Retiming runner for the released frozen Small-MEC instances."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import networkx as nx

from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation
from scripts.timing.common import INPUTS, initialize_output, load, save, solve


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="timing CSV")
    parser.add_argument(
        "--limit",
        type=int,
        help="time only the first N frozen instances (for smoke tests)",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    inputs = INPUTS if args.limit is None else INPUTS[: args.limit]
    initialize_output(args.output)

    for index, frozen in enumerate(inputs, start=1):
        record = dict(
            part="B",
            instance_id=frozen["instance_id"],
            n=frozen["n"],
            family=frozen["graph_family"],
            target=frozen["target_regime"],
            source_seed=frozen["source_seed"],
            target_seed=frozen["target_seed"],
        )
        try:
            dag = nx.DiGraph()
            dag.add_nodes_from(range(frozen["n"]))
            dag.add_edges_from(
                tuple(map(int, edge.split(">")))
                for edge in frozen["sampled_true_dag_identifier"].split(";")
                if edge
            )
            cpdag = cpdag_from_true_dag_for_simulation(dag)
            expected = {
                tuple(
                    sorted(
                        tuple(map(int, edge.split(">")))
                        for edge in candidate["candidate_dag_identifier"].split(
                            ";"
                        )
                        if edge
                    )
                )
                for candidate in frozen["candidate_dags"]
            }
            solve(
                record,
                cpdag,
                frozen["source_mean"],
                frozen["source_covariance"],
                frozen["target_mean"],
                frozen["target_covariance"],
                expected,
            )
        except Exception as error:  # keep a complete timing ledger
            record.update(
                status="error", details=repr(error), load_end=load()
            )
            traceback.print_exc()
        save(args.output, record)
        print(
            "B",
            index,
            record["status"],
            record.get("match_s"),
            flush=True,
        )


if __name__ == "__main__":
    main()
