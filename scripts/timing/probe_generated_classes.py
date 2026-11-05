"""Probe generated graph classes for exact Small-MEC timing bins."""

from __future__ import annotations

import argparse
import signal
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

from graphs.scm import LinearGaussianSCM, sample_coefficients
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation
from research.matching_query_complexity import enumerate_consistent_dags
from scripts.timing.common import (
    CONFIG,
    SourceProblem,
    _graph,
    _stable_seed,
    _target,
    load,
    save,
    solve,
)


SEED_BASE = 20261104
T = TypeVar("T")


class Expired(Exception):
    """Raised when a probe exceeds its wall-clock allocation."""


def _alarm(_signum: int, _frame: object) -> None:
    raise Expired()


def _within(seconds: float, function: Callable[[], T]) -> T:
    signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    try:
        return function()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _bin(mec_size: int) -> str:
    for lower, upper in [
        (33, 64),
        (65, 128),
        (129, 256),
        (257, 512),
        (513, 1024),
    ]:
        if lower <= mec_size <= upper:
            return f"{lower}-{upper}"
    return ""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="timing CSV")
    parser.add_argument(
        "--draw-limit",
        type=int,
        default=250,
        help="ER draws per n (default: 250; lower values are for smoke tests)",
    )
    parser.add_argument(
        "--time-budget-seconds", type=float, default=4 * 3600
    )
    parser.add_argument(
        "--instance-timeout-seconds", type=float, default=1800
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.draw_limit < 1:
        raise SystemExit("--draw-limit must be positive")
    signal.signal(signal.SIGALRM, _alarm)
    start = time.monotonic()
    deadline = start + args.time_budget_seconds
    counts: Counter[str] = Counter()
    stopped: set[str] = set()
    cells = [("path", n, 1) for n in (8, 10, 12, 16)] + [
        ("erdos_renyi", n, args.draw_limit) for n in (8, 9, 10)
    ]

    for family, n, limit in cells:
        for replicate in range(limit):
            filled = {key for key, count in counts.items() if count >= 3}
            if time.monotonic() >= deadline or len(stopped | filled) == 5:
                break
            seed = _stable_seed(
                SEED_BASE, "source", family, n, replicate, 0
            )
            rng = np.random.default_rng(seed)
            dag = _graph(CONFIG, family, n, rng)
            if not dag.edges:
                continue
            cpdag = cpdag_from_true_dag_for_simulation(dag)
            undirected = len(cpdag.undirected_edges)
            enumeration_start = time.perf_counter()
            before = load()
            try:
                candidates = _within(
                    deadline - time.monotonic(),
                    lambda: enumerate_consistent_dags(
                        cpdag, max_undirected_edges=undirected
                    ),
                )
            except Expired:
                print(
                    "global deadline during source enumeration",
                    family,
                    n,
                    replicate,
                    seed,
                    flush=True,
                )
                break
            enumeration_s = time.perf_counter() - enumeration_start
            mec_size = len(candidates)
            timing_bin = _bin(mec_size)
            print(
                "draw",
                family,
                n,
                replicate,
                seed,
                mec_size,
                undirected,
                round(enumeration_s, 4),
                timing_bin,
                flush=True,
            )
            if (
                not timing_bin
                or timing_bin in stopped
                or counts[timing_bin] >= 3
            ):
                continue

            true_identifier = tuple(sorted(dag.edges))
            assert true_identifier in {candidate.id for candidate in candidates}
            coefficients = sample_coefficients(
                dag,
                rng,
                CONFIG.coefficient_min_abs,
                CONFIG.coefficient_max_abs,
            )
            noise = rng.uniform(
                *CONFIG.noise_variance_bounds, size=n
            )
            scm = LinearGaussianSCM(
                dag, coefficients, noise_variances=noise
            )
            source = SourceProblem(
                seed,
                n,
                family,
                replicate,
                0,
                scm,
                cpdag,
                candidates,
                enumeration_s,
            )
            target, target_seed = _target(source, "realisable", CONFIG.budget)
            source_mean, source_covariance = scm.observational_moments()
            record = dict(
                part="C",
                instance_id=f"{family}:n{n}:r{replicate}",
                n=n,
                family=family,
                target="realisable",
                source_seed=seed,
                target_seed=target_seed,
                mec_size=mec_size,
                undirected_edges=undirected,
                enumeration_s=enumeration_s,
                load_start=before,
                details=f"bin={timing_bin}; draw={replicate}",
            )
            try:
                _within(
                    min(
                        args.instance_timeout_seconds,
                        deadline - time.monotonic(),
                    ),
                    lambda: solve(
                        record,
                        cpdag,
                        source_mean,
                        source_covariance,
                        target.mean,
                        target.covariance,
                        {candidate.id for candidate in candidates},
                    ),
                )
            except Expired:
                record["status"] = (
                    "timeout"
                    if time.monotonic() < deadline
                    else "global_budget_exhausted"
                )
                record["load_end"] = load()
                stopped.add(timing_bin)
            except Exception as error:  # retain failed probes in the ledger
                record.update(
                    status="error",
                    details=record["details"] + "; " + repr(error),
                    load_end=load(),
                )
                traceback.print_exc()
            counts[timing_bin] += 1
            save(args.output, record)
            print("result", record, flush=True)

    print(
        "finished counts",
        counts,
        "stopped",
        stopped,
        "elapsed",
        time.monotonic() - start,
        "load",
        load(),
        flush=True,
    )


if __name__ == "__main__":
    main()
