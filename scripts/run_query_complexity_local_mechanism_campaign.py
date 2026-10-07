#!/usr/bin/env python3
"""Resumable production runner for the local-mechanism query campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from environments.blackbox import EnvironmentQueryLedger, ExactMomentBlackBoxEnvironment
from graphs.DAG_gen import (
    generate_barabasi_albert_dag,
    generate_erdos_renyi_dag,
    generate_randomly_oriented_tree_dag,
    generate_tree_of_cliques_dag,
)
from graphs.interventions import ShiftScaleIntervention
from graphs.meek import apply_meek_closure
from graphs.scm import LinearGaussianSCM, sample_coefficients
from graphs.target_gen import generate_realisable_target
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation
from oracle.sparse_shift_scale_kl import solve_sparse_shift_scale_kl
from query_complexity import (
    PROTOCOL_VERSION,
    FirstUnresolvedPolicy,
    MaxUndirectedDegreePolicy,
    run_query_method,
)
from scripts.audit_blackbox_quadratic_dp import make_problems


def _git(command: str) -> str:
    return subprocess.check_output(["git", *command.split()], text=True).strip()


def _graph(regime: dict[str, object], n: int, rng: np.random.Generator):
    family = regime["family"]
    if family == "random_tree":
        return generate_randomly_oriented_tree_dag(n, rng)
    if family == "erdos_renyi":
        return generate_erdos_renyi_dag(n, float(regime["edge_probability"]), rng)
    if family == "barabasi_albert":
        return generate_barabasi_albert_dag(n, int(regime["attachment_edges"]), rng)
    if family == "tree_of_cliques":
        return generate_tree_of_cliques_dag(
            n,
            int(regime["branching_degree"]),
            int(regime["min_clique_size"]),
            min(n, int(regime["max_clique_size"])),
            rng,
        )
    raise ValueError(f"unsupported graph family: {family}")


def _source(config, regime, n, replicate):
    seed_sequence = np.random.SeedSequence(
        [
            int(config["base_seed"]),
            n,
            int(replicate),
            int(config["graph_regimes"].index(regime)),
        ]
    )
    graph_rng, coefficient_rng, noise_rng, target_rng = (
        np.random.default_rng(seed) for seed in seed_sequence.spawn(4)
    )
    dag = _graph(regime, n, graph_rng)
    coefficients = sample_coefficients(
        dag,
        coefficient_rng,
        float(config["coefficient_min_abs"]),
        float(config["coefficient_max_abs"]),
    )
    low_noise, high_noise = config["noise_variance_bounds"]
    noise = noise_rng.uniform(float(low_noise), float(high_noise), size=n)
    scm = LinearGaussianSCM(dag, coefficients, noise_variances=noise)
    target_config = config["canonical_target"]
    shift = np.zeros(n)
    shift_count = max(1, round(float(target_config["active_shift_fraction"]) * n))
    shift_nodes = target_rng.choice(n, size=min(n, shift_count), replace=False)
    shift_low, shift_high = target_config["shift_magnitude_bounds"]
    shift_magnitudes = target_rng.uniform(
        float(shift_low), float(shift_high), shift_nodes.size
    )
    shift_signs = np.where(target_rng.integers(0, 2, shift_nodes.size) == 0, -1.0, 1.0)
    shift[shift_nodes] = shift_magnitudes * shift_signs
    scale = np.ones(n)
    nonroots = np.array([node for node in range(n) if dag.in_degree(node)], dtype=int)
    if nonroots.size:
        scale_count = max(1, round(float(target_config["active_scale_fraction"]) * n))
        scale_nodes = target_rng.choice(
            nonroots, size=min(nonroots.size, scale_count), replace=False
        )
        scale_low, scale_high = target_config["scale_value_bounds"]
        scale[scale_nodes] = target_rng.uniform(
            float(scale_low), float(scale_high), scale_nodes.size
        )
    target = generate_realisable_target(scm, ShiftScaleIntervention(shift, scale))
    return scm, target


def _initial_graph(dag, revealed_fraction: float):
    graph = cpdag_from_true_dag_for_simulation(dag)
    count = int(np.ceil(revealed_fraction * len(graph.undirected_edges)))
    for node_a, node_b in sorted(graph.undirected_edges)[:count]:
        graph = (
            graph.orient(node_a, node_b)
            if dag.has_edge(node_a, node_b)
            else graph.orient(node_b, node_a)
        )
    return apply_meek_closure(graph)


def _policy(name: str):
    if name == "first_unresolved":
        return FirstUnresolvedPolicy()
    if name == "max_undirected_degree":
        return MaxUndirectedDegreePolicy()
    raise ValueError(f"unsupported public policy: {name}")


def _append_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _public_result(*, task_id, scm, target, initial_graph, policy_name, budget, config):
    source_mean, source_covariance = scm.observational_moments()
    result = run_query_method(
        method="public_adaptive_local_mechanism_dp",
        environment=ExactMomentBlackBoxEnvironment(
            scm,
            instance_id=task_id,
            ledger=EnvironmentQueryLedger(None),
        ),
        source_mean=source_mean,
        source_covariance=source_covariance,
        target_mean=target.mean,
        target_covariance=target.covariance,
        budget_B=budget,
        experimental_sparsity_S=int(config["experimental_sparsity"]),
        shift_bounds=tuple(float(x) for x in config["shift_bounds"]),
        scale_bounds=tuple(float(x) for x in config["scale_bounds"]),
        initial_graph=initial_graph,
        singleton_shift_policy=_policy(policy_name),
    )
    return result


def _base_row(task_id, panel, n, graph_family, target_family, oracle):
    return {
        "task_id": task_id,
        "panel": panel,
        "protocol_version": PROTOCOL_VERSION,
        "n": n,
        "graph_family": graph_family,
        "target_family": target_family,
        "K_B_star": oracle.raw_K_B_star,
    }


def _finalize(output: Path, manifest: dict[str, object]) -> None:
    rows = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text().splitlines()
        if line
    ]
    keys = sorted({key for row in rows for key in row})
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )
    public_rows = [
        row for row in rows if row.get("method") == "public_adaptive_local_mechanism_dp"
    ]
    summary = {
        "row_count": len(rows),
        "public_row_count": len(public_rows),
        "max_absolute_KL_gap": max(abs(float(row["KL_gap"])) for row in rows),
        "mean_T_over_n": float(np.mean([row["T_over_n"] for row in public_rows])),
        "mean_n_minus_T": float(np.mean([row["n_minus_T"] for row in public_rows])),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest["status"] = "complete"
    manifest["summary"] = summary
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/query_complexity_local_mechanism_campaign_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/kl_query_complexity_v1_local_mechanism_full"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-full-campaign", action="store_true")
    args = parser.parse_args()
    if not args.allow_full_campaign:
        raise RuntimeError("full campaign requires explicit --allow-full-campaign")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    if config["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeError("campaign config protocol does not match implementation")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    manifest_path = args.output / "manifest.json"
    rows_path = args.output / "results.jsonl"
    if args.output.exists() and not args.resume:
        raise FileExistsError("output exists; use --resume only for the same manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config_sha256") != config_hash:
            raise RuntimeError("resume config hash does not match existing manifest")
    else:
        manifest = {
            "campaign_id": config["campaign_id"],
            "protocol_version": PROTOCOL_VERSION,
            "config_path": str(args.config),
            "config_sha256": config_hash,
            "branch": _git("branch --show-current"),
            "head": _git("rev-parse HEAD"),
            "status": "running",
            "resume_semantics": "append-only JSONL; skip completed stable task_id",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    completed = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line:
                completed.add(json.loads(line)["task_id"])
    tolerance = float(config["exactness_tolerance"])

    for n in config["node_counts"]:
        n = int(n)
        budget = min(6, n)
        for graph_index, regime in enumerate(config["graph_regimes"]):
            graph_label = f"{regime['family']}:{graph_index}"
            for replicate in range(int(config["replicate_seeds"])):
                scm, target = _source(config, regime, n, replicate)
                oracle = solve_sparse_shift_scale_kl(
                    scm=scm,
                    target_mean=target.mean,
                    target_covariance=target.covariance,
                    budget=budget,
                    shift_bounds=tuple(float(x) for x in config["shift_bounds"]),
                    log_scale_bounds=tuple(
                        float(np.log(x)) for x in config["scale_bounds"]
                    ),
                )
                source_key = f"n{n}:g{graph_index}:r{replicate}"
                known_id = f"main:{source_key}:known_dag_exact_dp"
                if known_id not in completed:
                    row = _base_row(
                        known_id, "main", n, graph_label, "realisable", oracle
                    )
                    row.update(
                        method="known_dag_exact_dp",
                        policy="not_applicable",
                        initial_graph_regime="known_dag_oracle",
                        T=0,
                        T_over_n=0.0,
                        n_minus_T=n,
                        final_KL=oracle.raw_K_B_star,
                        KL_gap=0.0,
                    )
                    _append_row(rows_path, row)
                    completed.add(known_id)
                reference_id = f"main:{source_key}:graph_free_total_effect_dp"
                if reference_id not in completed:
                    mean, covariance = scm.observational_moments()
                    reference = run_query_method(
                        method="graph_free_total_effect_dp",
                        environment=ExactMomentBlackBoxEnvironment(
                            scm,
                            instance_id=reference_id,
                            ledger=EnvironmentQueryLedger(None),
                        ),
                        source_mean=mean,
                        source_covariance=covariance,
                        target_mean=target.mean,
                        target_covariance=target.covariance,
                        budget_B=budget,
                        experimental_sparsity_S=1,
                        shift_bounds=tuple(float(x) for x in config["shift_bounds"]),
                        scale_bounds=tuple(float(x) for x in config["scale_bounds"]),
                    )
                    gap = reference.final_kl - oracle.raw_K_B_star
                    if abs(gap) > tolerance:
                        raise RuntimeError(f"exactness gate failed for {reference_id}")
                    row = _base_row(
                        reference_id, "main", n, graph_label, "realisable", oracle
                    )
                    row.update(
                        method=reference.method,
                        policy="not_applicable",
                        initial_graph_regime="graph_free",
                        T=reference.counts.total_query_count,
                        T_over_n=reference.counts.total_query_count / n,
                        n_minus_T=n - reference.counts.total_query_count,
                        final_KL=reference.final_kl,
                        KL_gap=gap,
                    )
                    _append_row(rows_path, row)
                    completed.add(reference_id)
                for initial_regime in config["initial_graph_regimes"]:
                    initial = _initial_graph(
                        scm.dag, float(initial_regime["revealed_fraction"])
                    )
                    for policy_name in config["policies"]:
                        task_id = (
                            f"main:{source_key}:{initial_regime['name']}:{policy_name}"
                        )
                        if task_id in completed:
                            continue
                        result = _public_result(
                            task_id=task_id,
                            scm=scm,
                            target=target,
                            initial_graph=initial,
                            policy_name=policy_name,
                            budget=budget,
                            config=config,
                        )
                        gap = result.final_kl - oracle.raw_K_B_star
                        if abs(gap) > tolerance:
                            raise RuntimeError(f"exactness gate failed for {task_id}")
                        row = _base_row(
                            task_id, "main", n, graph_label, "realisable", oracle
                        )
                        row.update(
                            method=result.method,
                            policy=result.singleton_shift_policy,
                            replicate=replicate,
                            initial_graph_regime=initial_regime["name"],
                            initial_undirected_edges=len(initial.undirected_edges),
                            initial_resolved_nodes=len(result.initially_resolved_nodes),
                            T=result.counts.total_query_count,
                            T_over_n=result.query_fraction,
                            n_minus_T=n - result.counts.total_query_count,
                            queried_nodes=list(result.queried_nodes),
                            direct_mechanism_recoveries=len(
                                result.directly_recovered_nodes
                            ),
                            graph_resolved_without_query_nodes=len(
                                result.graph_resolved_without_query_nodes
                            ),
                            direct_orientations=result.orientations_from_local_mechanism,
                            meek_orientations=result.orientations_from_meek,
                            termination_reason=result.termination_reason,
                            final_KL=result.final_kl,
                            KL_gap=gap,
                        )
                        _append_row(rows_path, row)
                        completed.add(task_id)
                        print(task_id, flush=True)

    for problem in make_problems():
        oracle = solve_sparse_shift_scale_kl(
            scm=problem.scm,
            target_mean=problem.target.mean,
            target_covariance=problem.target.covariance,
            budget=6,
            shift_bounds=tuple(float(x) for x in config["shift_bounds"]),
            log_scale_bounds=tuple(float(np.log(x)) for x in config["scale_bounds"]),
        )
        initial = cpdag_from_true_dag_for_simulation(problem.scm.dag)
        for policy_name in config["policies"]:
            task_id = f"target_robustness:{problem.label}:{policy_name}"
            if task_id in completed:
                continue
            result = _public_result(
                task_id=task_id,
                scm=problem.scm,
                target=problem.target,
                initial_graph=initial,
                policy_name=policy_name,
                budget=6,
                config=config,
            )
            gap = result.final_kl - oracle.raw_K_B_star
            if abs(gap) > tolerance:
                raise RuntimeError(f"exactness gate failed for {task_id}")
            row = _base_row(
                task_id,
                "target_robustness",
                5,
                "fixed_audit_dag",
                problem.label,
                oracle,
            )
            row.update(
                method=result.method,
                policy=policy_name,
                initial_graph_regime="exact_cpdag",
                initial_undirected_edges=len(initial.undirected_edges),
                initial_resolved_nodes=len(result.initially_resolved_nodes),
                T=result.counts.total_query_count,
                T_over_n=result.query_fraction,
                n_minus_T=5 - result.counts.total_query_count,
                final_KL=result.final_kl,
                KL_gap=gap,
                termination_reason=result.termination_reason,
            )
            _append_row(rows_path, row)
            completed.add(task_id)
    _finalize(args.output, manifest)


if __name__ == "__main__":
    main()
