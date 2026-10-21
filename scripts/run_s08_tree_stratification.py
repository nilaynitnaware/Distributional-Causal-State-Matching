#!/usr/bin/env python3
"""S08: deterministic stratification audit of archived S01-B/S01-C results.

This program only reads archived campaign artifacts.  It does not import or
invoke campaign, policy, optimizer, certificate, or simulation code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts._output_safety import prepare_output_directory


PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 20261004
BOOTSTRAP_REPLICATES = 10_000
POLICIES = (
    "one_step_matching_aware",
    "balance",
    "first_unresolved",
    "max_undirected_degree",
)
FAMILIES = ("path", "uniform_random_labelled_tree")
FAMILY_LABELS = {
    "path": "path",
    "uniform_random_labelled_tree": "random-tree",
}
POLICY_LABELS = {
    "one_step_matching_aware": "One-step matching-aware",
    "balance": "Balance",
    "first_unresolved": "Algorithm 1 first-unresolved",
    "max_undirected_degree": "Algorithm 1 max-undirected-degree",
}
REGIME_LABELS = {
    "shift_scale_generated": "shift--scale-generated",
    "correlated_noise": "correlated-noise",
}
BANKS = {
    20: {
        "name": "S01-B",
        "directory": PROJECT / "outputs/s01_tree_population_pilot_v1",
        "n": 20,
        "source_id_field": False,
        "definition_file": "instances.jsonl",
        "proposal_file": None,
    },
    40: {
        "name": "S01-C",
        "directory": PROJECT / "outputs/s01_c_tree_population_n40_v1",
        "n": 40,
        "source_id_field": True,
        "definition_file": "frozen_instances.jsonl",
        "proposal_file": "proposal_and_eligibility_manifest.json",
    },
}
PROVENANCE_FILES = {
    20: (
        "frozen_config.json",
        "instances.jsonl",
        "per_root_results.jsonl",
        "query_traces.jsonl",
        "terminal_certificates.jsonl",
        "pilot_summary.json",
        "validation_results.json",
        "implementation_manifest.json",
        "frozen_probe_rerun.json",
        "s01_tree_population_pilot_report.md",
    ),
    40: (
        "frozen_config.json",
        "frozen_instances.jsonl",
        "proposal_and_eligibility_manifest.json",
        "instances.jsonl",
        "per_root_results.jsonl",
        "query_traces.jsonl",
        "terminal_certificates.jsonl",
        "pilot_summary.json",
        "validation_results.json",
        "implementation_manifest.json",
        "computational_work.jsonl",
        "execution_log.jsonl",
        "failures.jsonl",
        "incomplete.jsonl",
        "s01_c_tree_population_n40_report.md",
    ),
}
KNOWN_MEANS = {
    20: {
        "one_step_matching_aware": (1.2875, 4.1125),
        "balance": (2.7250, 3.7500),
        "first_unresolved": (4.1125, 7.7000),
        "max_undirected_degree": (3.1250, 5.1000),
    },
    40: {
        "one_step_matching_aware": (1.4375, 5.2250),
        "balance": (3.6500, 4.7500),
        "first_unresolved": (5.5500, 10.1000),
        "max_undirected_degree": (4.3125, 6.8000),
    },
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_number(value: float) -> int | float:
    value = float(value)
    if value.is_integer():
        return int(value)
    return value


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarize an empty distribution")
    frequency = Counter(clean_number(value) for value in array)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "minimum": clean_number(array.min()),
        "maximum": clean_number(array.max()),
        "positive": int(np.sum(array > 0)),
        "ties": int(np.sum(array == 0)),
        "negative": int(np.sum(array < 0)),
        "frequency": {str(key): frequency[key] for key in sorted(frequency)},
    }


def source_id_for(size: int, row: dict[str, Any]) -> str:
    if BANKS[size]["source_id_field"]:
        return str(row["source_id"])
    # The S01-B instance identifier itself explicitly carries the source prefix.
    # Remove only its final target-regime and budget components.
    return str(row["instance_id"]).rsplit(":", 2)[0]


def target_id_for(source_id: str, regime: str, target_seed: int) -> str:
    return f"{source_id}:{regime}:seed{target_seed}"


def add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, details: Any) -> None:
    checks.append({"id": check_id, "status": "PASS" if passed else "FAIL", "details": details})


def read_bank(size: int, checks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = BANKS[size]
    directory: Path = spec["directory"]
    instances = load_jsonl(directory / "instances.jsonl")
    roots = load_jsonl(directory / "per_root_results.jsonl")
    n = int(spec["n"])

    by_instance = {row["instance_id"]: row for row in instances}
    add_check(checks, f"n{size}_unique_instance_ids", len(by_instance) == len(instances) == 80,
              {"unique": len(by_instance), "records": len(instances), "expected": 80})
    add_check(checks, f"n{size}_no_exclusions", all(row.get("status") == "complete" for row in instances),
              dict(Counter(row.get("status") for row in instances)))

    root_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in roots:
        root_groups[(row["instance_id"], row["policy"])].append(row)
    expected_group_keys = {(instance_id, policy) for instance_id in by_instance for policy in POLICIES}
    add_check(checks, f"n{size}_unique_instance_policy_cells",
              set(root_groups) == expected_group_keys and len(root_groups) == 320,
              {"observed": len(root_groups), "expected": 320,
               "missing": sorted([list(key) for key in expected_group_keys - set(root_groups)])[:10],
               "unexpected": sorted([list(key) for key in set(root_groups) - expected_group_keys])[:10]})

    complete_root_coverage = True
    root_coverage_failures = []
    raw_summary_matches_archive = True
    summary_mismatches = []
    joined = []
    for instance_id, instance in sorted(by_instance.items()):
        archive = instance.get("archive", instance)
        source_id = source_id_for(size, instance)
        target_seed = int(archive["target_seed"])
        source_seed = int(archive["source_seed"])
        family = str(instance["family"])
        regime = str(instance["regime"])
        budget = int(instance["budget"])
        record: dict[str, Any] = {
            "size": size,
            "bank": spec["name"],
            "instance_id": instance_id,
            "source_id": source_id,
            "target_id": target_id_for(source_id, regime, target_seed),
            "source_seed": source_seed,
            "target_seed": target_seed,
            "source_replicate": int(instance.get("source_replicate", instance.get("replicate"))),
            "family": family,
            "regime": regime,
            "budget": budget,
        }
        for policy in POLICIES:
            group = root_groups.get((instance_id, policy), [])
            observed_roots = [int(row["true_root_evaluation_only"]) for row in group]
            if len(group) != n or sorted(observed_roots) != list(range(n)) or len(set(observed_roots)) != n:
                complete_root_coverage = False
                root_coverage_failures.append({
                    "instance_id": instance_id,
                    "policy": policy,
                    "record_count": len(group),
                    "unique_roots": len(set(observed_roots)),
                })
                continue
            matching_max = max(int(row["matching_query_count"]) for row in group)
            full_max = max(int(row["dag_query_count"]) for row in group)
            matching_roots = sorted(int(row["true_root_evaluation_only"]) for row in group
                                    if int(row["matching_query_count"]) == matching_max)
            full_roots = sorted(int(row["true_root_evaluation_only"]) for row in group
                                if int(row["dag_query_count"]) == full_max)
            archived = instance["policy_summary"][policy]
            if (matching_max != int(archived["matching_worst_case"]) or
                    full_max != int(archived["dag_worst_case"])):
                raw_summary_matches_archive = False
                summary_mismatches.append({
                    "instance_id": instance_id,
                    "policy": policy,
                    "raw": [matching_max, full_max],
                    "archived": [archived["matching_worst_case"], archived["dag_worst_case"]],
                })
            record[f"{policy}_match"] = matching_max
            record[f"{policy}_full"] = full_max
            record[f"{policy}_stopping_saving"] = full_max - matching_max
            record[f"{policy}_match_max_roots"] = matching_roots
            record[f"{policy}_full_max_roots"] = full_roots
        record["d_match_balance_minus_one_step"] = (
            record["balance_match"] - record["one_step_matching_aware_match"]
        )
        record["d_full_balance_minus_one_step"] = (
            record["balance_full"] - record["one_step_matching_aware_full"]
        )
        joined.append(record)

    add_check(checks, f"n{size}_complete_root_coverage", complete_root_coverage,
              {"expected_roots_per_instance_policy": n, "failures": root_coverage_failures[:20]})
    add_check(checks, f"n{size}_raw_maxima_equal_archived_instance_summaries", raw_summary_matches_archive,
              {"mismatches": summary_mismatches[:20]})

    sources = defaultdict(list)
    cells = defaultdict(list)
    source_metadata = {}
    for row in joined:
        sources[row["source_id"]].append(row)
        cells[(row["budget"], row["regime"])].append(row)
        source_metadata.setdefault(row["source_id"], (row["family"], row["source_seed"]))
        if source_metadata[row["source_id"]] != (row["family"], row["source_seed"]):
            raise ValueError(f"inconsistent source metadata for {row['source_id']}")

    expected_cells = {(budget, regime) for budget in (1, 2) for regime in REGIME_LABELS}
    cell_details = {f"B{budget}:{regime}": {"instances": len(cells[(budget, regime)]),
                                                    "sources": len({r['source_id'] for r in cells[(budget, regime)]})}
                    for budget, regime in sorted(expected_cells)}
    cell_accounting_ok = (set(cells) == expected_cells and
                          all(len(rows) == 20 and len({row["source_id"] for row in rows}) == 20
                              for rows in cells.values()))
    add_check(checks, f"n{size}_four_complete_budget_target_cells", cell_accounting_ok, cell_details)
    source_blocks_ok = (len(sources) == 20 and
                        all(len(rows) == 4 and
                            {(row["budget"], row["regime"]) for row in rows} == expected_cells
                            for rows in sources.values()))
    add_check(checks, f"n{size}_twenty_complete_source_blocks", source_blocks_ok,
              {"sources": len(sources), "block_sizes": dict(Counter(len(v) for v in sources.values()))})
    family_counts = Counter(meta[0] for meta in source_metadata.values())
    add_check(checks, f"n{size}_fixed_10_plus_10_family_design",
              family_counts == Counter({"path": 10, "uniform_random_labelled_tree": 10}), dict(family_counts))
    identifier_details = {
        "unique_sources": len({row["source_id"] for row in joined}),
        "unique_targets": len({row["target_id"] for row in joined}),
        "budgets": sorted({row["budget"] for row in joined}),
        "policies": sorted({row["policy"] for row in roots}),
    }
    identifier_ok = (
        identifier_details["unique_sources"] == 20
        and identifier_details["unique_targets"] == 40
        and identifier_details["budgets"] == [1, 2]
        and identifier_details["policies"] == sorted(POLICIES)
    )
    add_check(checks, f"n{size}_source_target_budget_policy_identifiers", identifier_ok, identifier_details)

    return joined, {
        "instance_records": len(instances),
        "root_records": len(roots),
        "source_count": len(sources),
        "family_source_counts": dict(sorted(family_counts.items())),
        "cell_details": cell_details,
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policy_costs = {}
    stopping = {}
    for policy in POLICIES:
        match_values = [row[f"{policy}_match"] for row in rows]
        full_values = [row[f"{policy}_full"] for row in rows]
        saving_values = [row[f"{policy}_stopping_saving"] for row in rows]
        policy_costs[policy] = {
            "match_mean": float(np.mean(match_values)),
            "full_recovery_mean": float(np.mean(full_values)),
            "match_distribution": distribution(match_values),
            "full_recovery_distribution": distribution(full_values),
        }
        stopping[policy] = distribution(saving_values)
    return {
        "instances": len(rows),
        "sources": len({row["source_id"] for row in rows}),
        "policy_costs": policy_costs,
        "paired_match_balance_minus_one_step": distribution(
            row["d_match_balance_minus_one_step"] for row in rows
        ),
        "paired_full_balance_minus_one_step": distribution(
            row["d_full_balance_minus_one_step"] for row in rows
        ),
        "stopping_savings_full_minus_match": stopping,
    }


def make_summaries(joined: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    all_summaries: dict[str, Any] = {}
    cell_csv = []
    family_csv = []
    for size in sorted(BANKS):
        size_rows = [row for row in joined if row["size"] == size]
        overall = summarize_group(size_rows)
        cells = {}
        family_cells = {}
        for budget in (1, 2):
            for regime in REGIME_LABELS:
                key = f"B{budget}:{regime}"
                rows = [row for row in size_rows if row["budget"] == budget and row["regime"] == regime]
                summary = summarize_group(rows)
                cells[key] = summary
                csv_row: dict[str, Any] = {
                    "size": size,
                    "budget": budget,
                    "regime": regime,
                    "instances": summary["instances"],
                    "sources": summary["sources"],
                }
                for policy in POLICIES:
                    csv_row[f"{policy}_match_mean"] = summary["policy_costs"][policy]["match_mean"]
                    csv_row[f"{policy}_full_mean"] = summary["policy_costs"][policy]["full_recovery_mean"]
                    csv_row[f"{policy}_stopping_saving_mean"] = summary["stopping_savings_full_minus_match"][policy]["mean"]
                for prefix, field in (
                    ("d_match", "paired_match_balance_minus_one_step"),
                    ("d_full", "paired_full_balance_minus_one_step"),
                ):
                    dist = summary[field]
                    for metric in ("mean", "median", "q1", "q3", "minimum", "maximum", "positive", "ties", "negative"):
                        csv_row[f"{prefix}_{metric}"] = dist[metric]
                    csv_row[f"{prefix}_frequency_json"] = json.dumps(dist["frequency"], sort_keys=True)
                cell_csv.append(csv_row)
                for family in FAMILIES:
                    frows = [row for row in rows if row["family"] == family]
                    fsummary = summarize_group(frows)
                    family_cells[f"{key}:{family}"] = fsummary
                    family_csv.append({
                        "size": size,
                        "budget": budget,
                        "regime": regime,
                        "family": family,
                        "instances": fsummary["instances"],
                        "sources": fsummary["sources"],
                        "d_match_mean": fsummary["paired_match_balance_minus_one_step"]["mean"],
                        "d_match_median": fsummary["paired_match_balance_minus_one_step"]["median"],
                        "d_match_positive": fsummary["paired_match_balance_minus_one_step"]["positive"],
                        "d_match_ties": fsummary["paired_match_balance_minus_one_step"]["ties"],
                        "d_match_negative": fsummary["paired_match_balance_minus_one_step"]["negative"],
                        "d_full_mean": fsummary["paired_full_balance_minus_one_step"]["mean"],
                    })
        all_summaries[str(size)] = {"overall": overall, "cells": cells, "family_cells": family_cells}
    return all_summaries, cell_csv, family_csv


def source_summaries(joined: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = {}
    csv_rows = []
    for size in sorted(BANKS):
        source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in joined:
            if row["size"] == size:
                source_groups[row["source_id"]].append(row)
        values = []
        serialized_sources = []
        for source_id, rows in sorted(source_groups.items()):
            ordered = sorted(rows, key=lambda row: (row["budget"], row["regime"]))
            average_d = float(np.mean([row["d_match_balance_minus_one_step"] for row in ordered]))
            values.append(average_d)
            entry = {
                "size": size,
                "source_id": source_id,
                "family": ordered[0]["family"],
                "source_seed": ordered[0]["source_seed"],
                "average_d_over_four_instances": average_d,
            }
            for row in ordered:
                entry[f"B{row['budget']}:{row['regime']}"] = row["d_match_balance_minus_one_step"]
            serialized_sources.append(entry)
            csv_rows.append(entry)
        result[str(size)] = {
            "estimand": "mean across 20 source-level averages; each source average covers its two targets crossed with two budgets",
            "distribution": distribution(values),
            "sources": serialized_sources,
        }
    return result, csv_rows


def bootstrap_for_size(size: int, joined: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    rows = [row for row in joined if row["size"] == size]
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_rows[row["source_id"]].append(row)
    ids_by_family = {
        family: sorted(source_id for source_id, block in source_rows.items() if block[0]["family"] == family)
        for family in FAMILIES
    }
    source_overall = {
        source_id: float(np.mean([row["d_match_balance_minus_one_step"] for row in block]))
        for source_id, block in source_rows.items()
    }
    source_cell = {
        (source_id, row["budget"], row["regime"]): float(row["d_match_balance_minus_one_step"])
        for source_id, block in source_rows.items() for row in block
    }

    def generate() -> dict[str, np.ndarray]:
        rng = np.random.default_rng(SEED)
        draws = {
            family: rng.integers(0, len(ids), size=(BOOTSTRAP_REPLICATES, len(ids)))
            for family, ids in ids_by_family.items()
        }
        estimates = {"overall": np.empty(BOOTSTRAP_REPLICATES, dtype=float)}
        for budget in (1, 2):
            for regime in REGIME_LABELS:
                estimates[f"B{budget}:{regime}"] = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for replicate in range(BOOTSTRAP_REPLICATES):
            sampled_ids = []
            for family in FAMILIES:
                ids = ids_by_family[family]
                sampled_ids.extend(ids[index] for index in draws[family][replicate])
            estimates["overall"][replicate] = np.mean([source_overall[source_id] for source_id in sampled_ids])
            for budget in (1, 2):
                for regime in REGIME_LABELS:
                    key = f"B{budget}:{regime}"
                    estimates[key][replicate] = np.mean(
                        [source_cell[(source_id, budget, regime)] for source_id in sampled_ids]
                    )
        return estimates

    first = generate()
    second = generate()
    reproducible = all(np.array_equal(first[key], second[key]) for key in first)
    output = []
    for key in ("overall", "B1:shift_scale_generated", "B1:correlated_noise",
                "B2:shift_scale_generated", "B2:correlated_noise"):
        if key == "overall":
            observed = float(np.mean(list(source_overall.values())))
            budget: int | str = "all"
            regime = "all"
        else:
            budget_text, regime = key.split(":", 1)
            budget = int(budget_text[1:])
            observed = float(np.mean([value for (source_id, b, r), value in source_cell.items()
                                      if b == budget and r == regime]))
        low, high = np.quantile(first[key], [0.025, 0.975])
        output.append({
            "size": size,
            "stratum": key,
            "budget": budget,
            "regime": regime,
            "estimand": "mean d = W_balance,MATCH - W_one-step,MATCH",
            "observed_mean": observed,
            "percentile_95_low": float(low),
            "percentile_95_high": float(high),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "seed": SEED,
            "resampling_unit": "source four-instance block",
            "stratification": "10 path and 10 random-tree source draws with replacement",
        })
    return output, reproducible


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def freq_text(frequency: dict[str, int]) -> str:
    return "{" + ", ".join(f"{key}: {value}" for key, value in frequency.items()) + "}"


def f4(value: float) -> str:
    return f"{value:.4f}"


def report_text(analysis: dict[str, Any]) -> str:
    summaries = analysis["summaries"]
    boot_lookup = {(row["size"], row["stratum"]): row for row in analysis["bootstrap_intervals"]}
    lines = [
        "# S08 archived tree-bank stratification audit",
        "",
        "## Scope and finding",
        "",
        "This is a bounded re-analysis of the completed population-moment S01-B (n=20) and S01-C (n=40) archives. No simulations, policy replays, certificate searches, KL recomputations, or production tests were run.",
        "",
        "Positive paired difference $d=W_{\\mathrm{balance},\\mathrm{MATCH}}-W_{\\mathrm{one\\mbox{-}step},\\mathrm{MATCH}}$ favors one-step matching-aware stopping. The advantage is positive in every budget-by-target cell in both separately generated banks, but its magnitude varies descriptively across the frozen strata. At n=20 the cell means range from " +
        f"{min(summaries['20']['cells'][key]['paired_match_balance_minus_one_step']['mean'] for key in summaries['20']['cells']):.2f} to {max(summaries['20']['cells'][key]['paired_match_balance_minus_one_step']['mean'] for key in summaries['20']['cells']):.2f}; " +
        "at n=40 they range from " +
        f"{min(summaries['40']['cells'][key]['paired_match_balance_minus_one_step']['mean'] for key in summaries['40']['cells']):.2f} to {max(summaries['40']['cells'][key]['paired_match_balance_minus_one_step']['mean'] for key in summaries['40']['cells']):.2f}. " +
        "These are descriptive contrasts, not causal budget effects or cross-size paired comparisons.",
        "",
        "## Input provenance and accounting",
        "",
        "The exact SHA-256 hashes of all inputs are serialized in `s08_tree_stratification.json`; core evidence is listed here.",
        "",
        "| Bank | Role | Exact path | SHA-256 |",
        "|---|---|---|---|",
    ]
    core_roles = {
        20: {
            "frozen_config.json": "frozen configuration",
            "instances.jsonl": "instance definitions and archived summaries",
            "per_root_results.jsonl": "root-level trajectories",
            "pilot_summary.json": "pilot summary",
            "validation_results.json": "validation record",
            "implementation_manifest.json": "implementation provenance",
        },
        40: {
            "frozen_config.json": "frozen configuration",
            "frozen_instances.jsonl": "frozen instance definitions",
            "proposal_and_eligibility_manifest.json": "proposal/eligibility manifest",
            "instances.jsonl": "completed instance summaries",
            "per_root_results.jsonl": "root-level trajectories",
            "pilot_summary.json": "pilot summary",
            "validation_results.json": "validation record",
            "implementation_manifest.json": "implementation provenance",
        },
    }
    for size in (20, 40):
        by_name = {Path(item["path"]).name: item for item in analysis["provenance"][str(size)]["files"]}
        for filename, role in core_roles[size].items():
            item = by_name[filename]
            lines.append(f"| {BANKS[size]['name']} (n={size}) | {role} | `{item['path']}` | `{item['sha256']}` |")
    lines += [
        "",
        "S01-B has no standalone frozen-instance file or proposal/eligibility manifest: its definitions, source/target seeds, candidate roots, and target construction are embedded in `instances.jsonl`. S01-B source identifiers below are the explicit instance-ID prefix obtained by removing only the regime and budget suffix. S01-C supplies an explicit `source_id`, a separate `frozen_instances.jsonl`, and a proposal/eligibility manifest. This packaging difference is not an accounting blocker.",
        "",
        "Each bank contains 20 sources and 80 instances: two explicit target regimes crossed with B in {1,2}. Every one of the four cells contains 20 instances from 20 sources; path and random-tree families contribute 10 sources each. All four policies have complete coverage of all 20 roots (n=20) or all 40 roots (n=40), with no excluded instances. Raw root trajectories were independently grouped and maximized; every maximum equals its archived per-instance summary.",
        "",
        "The shift--scale-generated target is guaranteed reachable at B=2 by design; no B=1 reachability claim is made. The correlated-noise target is not labeled reachable under invariant-noise shift--scale actions.",
        "",
        "## Aggregate reconciliation",
        "",
        "| n | Policy | Mean worst-case MATCH | Mean full recovery |",
        "|---:|---|---:|---:|",
    ]
    for size in (20, 40):
        for policy in POLICIES:
            p = summaries[str(size)]["overall"]["policy_costs"][policy]
            lines.append(f"| {size} | {POLICY_LABELS[policy]} | {f4(p['match_mean'])} | {f4(p['full_recovery_mean'])} |")
    lines += [
        "",
        "These reproduce all supplied known aggregates. For n=40, MATCH gives one-step 72 wins and 8 ties against balance; full recovery gives one-step 37 losses and 43 ties. Weighted means of the four equal-size cells reproduce every overall mean.",
        "",
        "## Budget-by-target results",
        "",
        "Entries under each policy are mean worst-case MATCH / full-recovery query costs. `d CI` is the descriptive 95% source-cluster percentile bootstrap interval for mean MATCH d.",
        "",
        "| n | B | Target regime | One-step | Balance | First unresolved | Max degree | d mean | d 95% CI | d +/0/- |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in (20, 40):
        for budget in (1, 2):
            for regime in REGIME_LABELS:
                key = f"B{budget}:{regime}"
                cell = summaries[str(size)]["cells"][key]
                costs = []
                for policy in POLICIES:
                    p = cell["policy_costs"][policy]
                    costs.append(f"{p['match_mean']:.2f} / {p['full_recovery_mean']:.2f}")
                d = cell["paired_match_balance_minus_one_step"]
                ci = boot_lookup[(size, key)]
                lines.append(
                    f"| {size} | {budget} | {REGIME_LABELS[regime]} | " + " | ".join(costs) +
                    f" | {d['mean']:.2f} | [{ci['percentile_95_low']:.2f}, {ci['percentile_95_high']:.2f}] | {d['positive']}/{d['ties']}/{d['negative']} |"
                )
    lines += [
        "",
        "## Paired distributions",
        "",
        "Quartiles use the standard linear empirical quantile. For both d columns, positive values favor one-step; full-recovery d is $W_{balance,full}-W_{one-step,full}$.",
        "",
        "| n | B | Target | Quantity | Mean | Median | Q1 | Q3 | Min | Max | +/0/- | Exact frequency |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for size in (20, 40):
        for budget in (1, 2):
            for regime in REGIME_LABELS:
                cell = summaries[str(size)]["cells"][f"B{budget}:{regime}"]
                for label, field in (("MATCH d", "paired_match_balance_minus_one_step"),
                                     ("Full d", "paired_full_balance_minus_one_step")):
                    d = cell[field]
                    lines.append(
                        f"| {size} | {budget} | {REGIME_LABELS[regime]} | {label} | {d['mean']:.2f} | {d['median']:.2f} | {d['q1']:.2f} | {d['q3']:.2f} | {d['minimum']} | {d['maximum']} | {d['positive']}/{d['ties']}/{d['negative']} | `{freq_text(d['frequency'])}` |"
                    )
    lines += [
        "",
        "## Stopping savings",
        "",
        "Savings are $W_{policy,full}-W_{policy,MATCH}$. They are differences of two separately maximized costs; they are not necessarily savings along one common maximizing root trajectory. The underlying root-level record does show MATCH as a prefix of the same policy path for each fixed root, but the MATCH and full-recovery maxima may occur at different roots.",
        "",
        "| n | B | Target | Policy | Mean | Median | Q1 | Q3 | Min | Max | Exact frequency |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for size in (20, 40):
        for budget in (1, 2):
            for regime in REGIME_LABELS:
                cell = summaries[str(size)]["cells"][f"B{budget}:{regime}"]
                for policy in POLICIES:
                    d = cell["stopping_savings_full_minus_match"][policy]
                    lines.append(
                        f"| {size} | {budget} | {REGIME_LABELS[regime]} | {POLICY_LABELS[policy]} | {d['mean']:.2f} | {d['median']:.2f} | {d['q1']:.2f} | {d['q3']:.2f} | {d['minimum']} | {d['maximum']} | `{freq_text(d['frequency'])}` |"
                    )
    lines += [
        "",
        "## Tree-family splits",
        "",
        "Each entry reports mean MATCH d and +/0/- counts within the 10-source family stratum. These are descriptive family splits, not uncertainty-adjusted comparisons between families.",
        "",
        "| n | B | Target | Path d (+/0/-) | Random-tree d (+/0/-) |",
        "|---:|---:|---|---:|---:|",
    ]
    for size in (20, 40):
        for budget in (1, 2):
            for regime in REGIME_LABELS:
                key = f"B{budget}:{regime}"
                family_entries = []
                for family in FAMILIES:
                    d = summaries[str(size)]["family_cells"][f"{key}:{family}"]["paired_match_balance_minus_one_step"]
                    family_entries.append(f"{d['mean']:.2f} ({d['positive']}/{d['ties']}/{d['negative']})")
                lines.append(f"| {size} | {budget} | {REGIME_LABELS[regime]} | {family_entries[0]} | {family_entries[1]} |")
    lines += [
        "",
        "## Source dependence and descriptive uncertainty",
        "",
        "The primary source-level estimand is the mean of 20 source averages, where each source average d covers its entire four-instance block (two target regimes by two budgets). For each bank independently, 10,000 bootstrap replicates use seed 20261004 and resample sources with replacement within family, retaining the fixed 10 path + 10 random-tree composition. Each replicate's same selected source blocks are used for the overall and all four cell estimates.",
        "",
        "| n | Estimand | Observed mean d | Descriptive 95% percentile interval |",
        "|---:|---|---:|---:|",
    ]
    for size in (20, 40):
        row = boot_lookup[(size, "overall")]
        lines.append(f"| {size} | Mean of source-average d | {row['observed_mean']:.3f} | [{row['percentile_95_low']:.3f}, {row['percentile_95_high']:.3f}] |")
    lines += [
        "",
        "All 20 source-level values per bank follow:",
        "",
        "| n | Source | Family | Four-instance average d |",
        "|---:|---|---|---:|",
    ]
    for size in (20, 40):
        for source in analysis["source_summaries"][str(size)]["sources"]:
            lines.append(f"| {size} | `{source['source_id']}` | {FAMILY_LABELS[source['family']]} | {source['average_d_over_four_instances']:.2f} |")
    lines += [
        "",
        "Source-average distribution summaries:",
        "",
        "| n | Mean | Median | Q1 | Q3 | Min | Max | Exact frequency |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for size in (20, 40):
        d = analysis["source_summaries"][str(size)]["distribution"]
        lines.append(f"| {size} | {d['mean']:.3f} | {d['median']:.3f} | {d['q1']:.3f} | {d['q3']:.3f} | {d['minimum']} | {d['maximum']} | `{freq_text(d['frequency'])}` |")
    lines += [
        "",
        "The optional family splits are serialized in `family_cell_summaries.csv`. Their denominators are 10 sources per family per budget-target cell. They provide descriptive checks only; the small family samples do not support general tree-family guarantees.",
        "",
        "## Interpretation and limitations",
        "",
        "The frozen cells show that the one-step-versus-balance MATCH advantage is not numerically constant across budget, target regime, or family. The tables and CSVs quantify those variations without asserting that budget or regime caused them. The n=20 and n=40 banks were generated separately and are not paired, so their differences are not size effects.",
        "",
        "The bootstrap intervals condition on the fixed 10+10 family composition and summarize source-cluster variation in these archives. There are only 20 source clusters per bank, no multiplicity adjustment is applied across reported strata, and the intervals are descriptive: they do not establish a superiority test, a causal budget effect, or a general performance guarantee.",
        "",
        "Population observation counts are not applicable. Computational-work scaling is not assessed: S01-B used a cache shared across policies, whereas S01-C used fresh isolated caches per instance-policy evaluation, so raw cache-work counts are not comparable across sizes. Archived timing/work records remain in provenance but are not reinterpreted here.",
        "",
        "## Reproduction and deliverables",
        "",
        "Exact command (from the repository root):",
        "",
        "```bash",
        "python -m scripts.run_s08_tree_stratification --input-root TREE_BANK_ROOT --output NEW_OUTPUT",
        "```",
        "",
        "Dependency: Python 3 plus NumPy. The script and input hashes are in `s08_tree_stratification.json`; generated-file hashes are in `output_hashes.json`. `validation_results.json` records all PASS/FAIL checks. The candidate appendix table is deliberately unattached to any manuscript insertion anchor.",
        "",
    ]
    return "\n".join(lines)


def tex_text(analysis: dict[str, Any]) -> str:
    lines = [
        "% Candidate appendix table only; no manuscript insertion anchor is asserted.",
        "% Generated by the S08 archived-result audit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Worst-case population query costs stratified by bank size, budget, and target regime. Entries are mean MATCH/full-recovery costs over 20 frozen source instances. Positive $d=W_{\\mathrm{bal},M}-W_{\\mathrm{1step},M}$ favors one-step matching-aware stopping; brackets are descriptive 95\\% source-cluster percentile bootstrap intervals.}",
        "\\label{tab:candidate-s08-tree-strata}",
        "\\small",
        "\\begin{tabular}{rrlccccc}",
        "\\toprule",
        "$n$ & $B$ & Target & One-step & Balance & First unresolved & Max degree & $d$ [95\\% interval] \\\\",
        "\\midrule",
    ]
    boot_lookup = {(row["size"], row["stratum"]): row for row in analysis["bootstrap_intervals"]}
    for size in (20, 40):
        for budget in (1, 2):
            for regime in REGIME_LABELS:
                key = f"B{budget}:{regime}"
                cell = analysis["summaries"][str(size)]["cells"][key]
                entries = []
                for policy in POLICIES:
                    p = cell["policy_costs"][policy]
                    entries.append(f"{p['match_mean']:.2f}/{p['full_recovery_mean']:.2f}")
                d = cell["paired_match_balance_minus_one_step"]
                ci = boot_lookup[(size, key)]
                target = "shift--scale-generated" if regime == "shift_scale_generated" else "correlated-noise"
                lines.append(f"{size} & {budget} & {target} & " + " & ".join(entries) +
                             f" & {d['mean']:.2f} [{ci['percentile_95_low']:.2f}, {ci['percentile_95_high']:.2f}] \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{minipage}{0.98\\linewidth}\\footnotesize",
        "The two size banks were generated separately and are not paired. Intervals condition on the fixed 10 path plus 10 random-tree source composition, use 10,000 within-family source-cluster resamples (seed 20261004), have no multiplicity adjustment, and are descriptive rather than performance guarantees. The correlated-noise regime is not asserted reachable under invariant-noise shift--scale actions.",
        "\\end{minipage}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    global OUT, BANKS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, required=True,
        help="Directory containing the complete n=20 and n=40 tree-bank directories",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    input_root = arguments.input_root.resolve()
    BANKS[20]["directory"] = input_root / "s01_tree_population_pilot_v1"
    BANKS[40]["directory"] = input_root / "s01_c_tree_population_n40_v1"
    OUT = prepare_output_directory(
        arguments.output,
        overwrite=arguments.overwrite,
        protected=tuple(Path(BANKS[size]["directory"]) for size in BANKS),
    )
    checks: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    joined: list[dict[str, Any]] = []
    accounting = {}
    for size in sorted(BANKS):
        directory: Path = BANKS[size]["directory"]
        files = []
        for filename in PROVENANCE_FILES[size]:
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
        provenance[str(size)] = {
            "bank": BANKS[size]["name"],
            "directory": str(directory),
            "instance_definition_path": str(directory / BANKS[size]["definition_file"]),
            "proposal_and_eligibility_manifest_path": (
                str(directory / BANKS[size]["proposal_file"]) if BANKS[size]["proposal_file"] else None
            ),
            "proposal_manifest_note": (None if BANKS[size]["proposal_file"] else
                "No separate manifest exists; definitions and source/target seeds are embedded in instances.jsonl."),
            "files": files,
        }
        bank_joined, bank_accounting = read_bank(size, checks)
        joined.extend(bank_joined)
        accounting[str(size)] = bank_accounting

    summaries, cell_csv, family_csv = make_summaries(joined)
    source_summary, source_csv = source_summaries(joined)
    bootstrap_rows = []
    bootstrap_reproducible = True
    for size in sorted(BANKS):
        rows, reproducible = bootstrap_for_size(size, joined)
        bootstrap_rows.extend(rows)
        bootstrap_reproducible &= reproducible
    add_check(checks, "deterministic_bootstrap_reproducibility", bootstrap_reproducible,
              {"seed": SEED, "replicates": BOOTSTRAP_REPLICATES,
               "criterion": "two independent reruns produced bit-identical estimate arrays"})

    # Known aggregate reconciliation.
    for size in (20, 40):
        observed = summaries[str(size)]["overall"]["policy_costs"]
        for policy, (known_match, known_full) in KNOWN_MEANS[size].items():
            passed = (np.isclose(observed[policy]["match_mean"], known_match, atol=0, rtol=0) and
                      np.isclose(observed[policy]["full_recovery_mean"], known_full, atol=0, rtol=0))
            add_check(checks, f"n{size}_{policy}_known_aggregate_reconciliation", bool(passed), {
                "observed_match": observed[policy]["match_mean"], "known_match": known_match,
                "observed_full": observed[policy]["full_recovery_mean"], "known_full": known_full,
            })

    n40_match = summaries["40"]["overall"]["paired_match_balance_minus_one_step"]
    n40_full = summaries["40"]["overall"]["paired_full_balance_minus_one_step"]
    add_check(checks, "n40_known_match_win_tie_loss", (n40_match["positive"], n40_match["ties"], n40_match["negative"]) == (72, 8, 0), n40_match)
    add_check(checks, "n40_known_full_one_step_loss_tie", (n40_full["negative"], n40_full["ties"], n40_full["positive"]) == (37, 43, 0), n40_full)

    for size in (20, 40):
        weighted_ok = True
        details = {}
        cells = summaries[str(size)]["cells"]
        overall = summaries[str(size)]["overall"]
        for policy in POLICIES:
            for cost_key, overall_key in (("match_mean", "match_mean"), ("full_recovery_mean", "full_recovery_mean")):
                weighted = sum(cell["instances"] * cell["policy_costs"][policy][cost_key]
                               for cell in cells.values()) / sum(cell["instances"] for cell in cells.values())
                target = overall["policy_costs"][policy][overall_key]
                weighted_ok &= bool(np.isclose(weighted, target, atol=0, rtol=0))
                details[f"{policy}:{cost_key}"] = {"weighted_cells": weighted, "overall": target}
        weighted_d = sum(cell["instances"] * cell["paired_match_balance_minus_one_step"]["mean"]
                         for cell in cells.values()) / 80
        target_d = overall["paired_match_balance_minus_one_step"]["mean"]
        weighted_ok &= bool(np.isclose(weighted_d, target_d, atol=0, rtol=0))
        details["paired_match_d"] = {"weighted_cells": weighted_d, "overall": target_d}
        add_check(checks, f"n{size}_weighted_cell_to_overall_identities", weighted_ok, details)

    integer_ok = all(
        isinstance(row[f"{policy}_{metric}"], int)
        for row in joined for policy in POLICIES for metric in ("match", "full", "stopping_saving")
    )
    add_check(checks, "integer_query_costs_and_savings", integer_ok,
              {"joined_instances": len(joined), "policies": len(POLICIES)})
    add_check(checks, "combined_bank_denominators", len(joined) == 160,
              {"joined_instances": len(joined), "expected": 160})

    # Verify the explicit source identifier and target seed are stable over budgets.
    target_stability_ok = True
    target_stability_failures = []
    target_groups = defaultdict(list)
    for row in joined:
        target_groups[(row["size"], row["source_id"], row["regime"])].append(row)
    for key, rows in target_groups.items():
        if len(rows) != 2 or len({row["target_seed"] for row in rows}) != 1 or {row["budget"] for row in rows} != {1, 2}:
            target_stability_ok = False
            target_stability_failures.append({"key": key, "records": rows})
    add_check(checks, "target_identifiers_stable_across_budget", target_stability_ok,
              {"source_target_groups": len(target_groups), "failures": target_stability_failures[:10]})

    # Maxima are allowed to occur at distinct roots; record how often no root jointly maximizes both.
    root_maximum_diagnostics = {}
    for size in (20, 40):
        size_rows = [row for row in joined if row["size"] == size]
        per_policy = {}
        for policy in POLICIES:
            disjoint = sum(not (set(row[f"{policy}_match_max_roots"]) & set(row[f"{policy}_full_max_roots"]))
                           for row in size_rows)
            per_policy[policy] = {"instance_count": len(size_rows), "disjoint_match_full_maximizer_sets": disjoint}
        root_maximum_diagnostics[str(size)] = per_policy

    script_hash = sha256(Path(__file__))
    analysis = {
        "schema_version": "s08_tree_stratification_v1",
        "analysis_kind": "read-only archived-result stratification",
        "reproduction_command": "python -m scripts.run_s08_tree_stratification --input-root TREE_BANK_ROOT --output NEW_OUTPUT",
        "dependencies": {"python": "3.x", "numpy": np.__version__, "script_path": str(Path(__file__)), "script_sha256": script_hash},
        "provenance": provenance,
        "accounting": accounting,
        "joined_per_instance_results": joined,
        "summaries": summaries,
        "source_summaries": source_summary,
        "bootstrap": {
            "seed": SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "method": "source-cluster percentile bootstrap, resampled within family while preserving 10 path + 10 random-tree draws; whole four-instance source blocks retained; same resamples used for overall and cell estimates within each bank",
            "estimand": "mean d = W_balance,MATCH - W_one-step,MATCH, conditional on fixed family composition",
            "multiplicity_adjustment": False,
            "interpretation": "descriptive uncertainty only; not a general performance guarantee",
        },
        "bootstrap_intervals": bootstrap_rows,
        "root_maximum_diagnostics": root_maximum_diagnostics,
        "missing_or_unsupported": [
            "S01-B has no standalone proposal/eligibility manifest or frozen-instances file; its definitions are embedded in instances.jsonl.",
            "Cross-size cache-work scaling is unsupported because S01-B shared caches across policies and S01-C isolated caches per instance-policy.",
            "Population observation counts are not applicable.",
            "No causal budget effects, superiority p-values, cross-size paired effects, or general performance guarantees are supported.",
        ],
    }

    json_path = OUT / "s08_tree_stratification.json"
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    joined_csv_rows = []
    for row in joined:
        csv_row = {key: value for key, value in row.items() if not key.endswith("_max_roots")}
        for key, value in row.items():
            if key.endswith("_max_roots"):
                csv_row[key] = ";".join(str(root) for root in value)
        joined_csv_rows.append(csv_row)
    write_csv(OUT / "joined_per_instance_results.csv", joined_csv_rows)
    write_csv(OUT / "cell_summaries.csv", cell_csv)
    write_csv(OUT / "family_cell_summaries.csv", family_csv)
    write_csv(OUT / "per_source_paired_differences.csv", source_csv)
    write_csv(OUT / "bootstrap_intervals.csv", bootstrap_rows)

    report_path = OUT / "s08_tree_stratification_report.md"
    report_path.write_text(report_text(analysis), encoding="utf-8")
    (OUT / "candidate_appendix_table.tex").write_text(tex_text(analysis), encoding="utf-8")

    validation = {
        "schema_version": "s08_validation_v1",
        "status": "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL",
        "checks": checks,
        "check_counts": dict(Counter(check["status"] for check in checks)),
        "independent_aggregation_route": "per_root_results.jsonl grouped by (instance_id, policy), separately maximizing matching_query_count and dag_query_count; checked against instances.jsonl policy_summary",
        "bootstrap_reproducibility": {"seed": SEED, "replicates": BOOTSTRAP_REPLICATES, "bit_identical_rerun": bootstrap_reproducible},
    }
    (OUT / "validation_results.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )

    generated_names = (
        "s08_tree_stratification_report.md",
        "s08_tree_stratification.json",
        "joined_per_instance_results.csv",
        "cell_summaries.csv",
        "family_cell_summaries.csv",
        "per_source_paired_differences.csv",
        "bootstrap_intervals.csv",
        "candidate_appendix_table.tex",
        "validation_results.json",
    )
    output_manifest = {
        "schema_version": "s08_output_hashes_v1",
        "note": "This manifest intentionally excludes itself to avoid a self-referential hash.",
        "script": {"path": str(Path(__file__)), "sha256": script_hash},
        "outputs": [{"path": str(OUT / name), "sha256": sha256(OUT / name), "bytes": (OUT / name).stat().st_size}
                    for name in generated_names],
    }
    (OUT / "output_hashes.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if validation["status"] != "PASS":
        raise SystemExit("S08 validation failed; inspect validation_results.json")
    print(f"S08 complete: {len(joined)} joined instances; {len(checks)} validation checks PASS")


if __name__ == "__main__":
    main()
