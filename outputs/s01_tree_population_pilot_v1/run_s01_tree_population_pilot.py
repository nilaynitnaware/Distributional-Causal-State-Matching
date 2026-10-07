"""Frozen capped S01-B population pilot."""

from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import math
import platform
import resource
import signal
import subprocess
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.nonrealisable_targets import correlated_noise_target
from graphs.scm import LinearGaussianSCM, sample_coefficients
from graphs.target_gen import generate_realisable_target
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from research.matching_query_complexity import (
    compute_all_optimal_action_sets,
    reconstruct_all_candidate_scms,
)
from research.s01_tree_population import (
    CertificateCache,
    enumerate_tree_root_class,
    paired_policy_path,
)
from research.small_mec_matching_query.experiment import _stable_seed

OUTPUT = Path(__file__).resolve().parent
MASTER_SEED, N = 20261011, 20
FAMILIES = ("path", "uniform_random_labelled_tree")
REGIMES = ("shift_scale_generated", "correlated_noise")
BUDGETS = (1, 2)
POLICIES = ("one_step_matching_aware", "balance", "first_unresolved", "max_undirected_degree")
SHIFT_BOUNDS, LOG_SCALE_BOUNDS = (-2.0, 2.0), (-1.0, 1.0)
WORKER_SECONDS, WORKER_BYTES = 60, 512 * 1024 * 1024
PILOT_CPU_SECONDS, PILOT_BYTES = 7200, 4 * 1024 * 1024 * 1024
CERT_CAP, TIE_CAP = 2000, 10000


def all_specs():
    return [
        {
            "family": family,
            "replicate": replicate,
            "regime": regime,
            "budget": budget,
            "instance_id": f"s01b:n20:{family}:r{replicate}:{regime}:B{budget}",
        }
        for family in FAMILIES
        for replicate in range(10)
        for regime in REGIMES
        for budget in BUDGETS
    ]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def config():
    return {
        "schema_version": "s01_tree_population_pilot_v1",
        "frozen_before_execution": True,
        "master_seed": MASTER_SEED,
        "n": N,
        "families": list(FAMILIES),
        "source_replicates_per_family": 10,
        "target_regimes": list(REGIMES),
        "budgets": list(BUDGETS),
        "instance_count": 80,
        "source_seed": "stable_seed(20261011,'s01b','source',family,20,replicate)",
        "target_seed": "stable_seed(source_seed,'target',regime), independent of budget",
        "source_distributions": {
            "coefficient_abs_uniform": [0.25, 0.65],
            "coefficient_sign": "independent fair sign",
            "noise_variance_uniform": [0.7, 1.4],
            "root": "uniform node",
            "path": "random node permutation joined consecutively",
            "random_tree": "uniform random labelled tree",
        },
        "target_distributions": {
            "shift": "one node, Uniform(-1.25,1.25)",
            "scale": "one nonroot node, exp(Uniform(log(0.7),log(1.45)))",
            "correlation": "one node pair, signed magnitude Uniform(0.12,0.28)",
            "reachability_note": (
                "standard shift-scale target is B=2 reachable; at B=1 it is a "
                "projection unless independently established; correlated-noise "
                "targets retain the stated non-reachability interpretation"
            ),
        },
        "action_bounds": {"shift": list(SHIFT_BOUNDS), "log_scale": list(LOG_SCALE_BOUNDS)},
        "caps": {
            "pilot_cpu_seconds": PILOT_CPU_SECONDS,
            "pilot_rss_bytes": PILOT_BYTES,
            "worker_wall_seconds": WORKER_SECONDS,
            "worker_cpu_seconds": WORKER_SECONDS,
            "worker_rss_bytes": WORKER_BYTES,
            "uncached_certificate_checks": CERT_CAP,
            "tied_family_combinations": TIE_CAP,
        },
        "thread_environment": {
            key: os.environ[key]
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "stopping": {
            "MATCH": "direct-KL common-command certificate over every surviving root",
            "DAG": "one surviving root",
            "inconclusive": "continue conservatively; never a terminal child",
        },
    }


def make_source(family, replicate):
    seed = _stable_seed(MASTER_SEED, "s01b", "source", family, N, replicate)
    rng = np.random.default_rng(seed)
    if family == "path":
        order = [int(x) for x in rng.permutation(N)]
        tree = nx.Graph()
        tree.add_nodes_from(range(N))
        tree.add_edges_from(zip(order[:-1], order[1:], strict=True))
    else:
        order = None
        tree = nx.random_labeled_tree(N, seed=rng)
    true_root = int(rng.integers(N))
    root_class = enumerate_tree_root_class(tree)
    true_dag = root_class.candidates[root_class.ids_by_root[true_root]].dag
    coefficients = sample_coefficients(true_dag, rng, 0.25, 0.65)
    noise = rng.uniform(0.7, 1.4, size=N)
    scm = LinearGaussianSCM(nx.DiGraph(true_dag), coefficients, noise_variances=noise)
    source_mean, source_covariance = scm.observational_moments()
    return {
        "seed": seed, "tree": tree, "order": order, "true_root": true_root,
        "root_class": root_class, "coefficients": coefficients, "noise": noise,
        "scm": scm, "mean": source_mean, "covariance": source_covariance,
    }


def make_target(source, regime):
    seed = _stable_seed(source["seed"], "target", regime)
    rng = np.random.default_rng(seed)
    shift, scale = np.zeros(N), np.ones(N)
    shift_node = int(rng.integers(N))
    scale_node = int(rng.choice(sorted(source["scm"].scale_eligible_nodes)))
    shift[shift_node] = float(rng.uniform(-1.25, 1.25))
    scale[scale_node] = float(np.exp(rng.uniform(math.log(0.7), math.log(1.45))))
    intervention = ShiftScaleIntervention(shift, scale)
    metadata = {
        "target_seed": seed, "shift_node": shift_node, "scale_node": scale_node,
        "generating_shift": shift.tolist(), "generating_scale": scale.tolist(),
    }
    if regime == "shift_scale_generated":
        return generate_realisable_target(source["scm"], intervention), metadata
    noise_covariance = np.diag(np.asarray(source["noise"], dtype=float))
    left, right = (int(x) for x in rng.choice(N, size=2, replace=False))
    rho = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.12, 0.28))
    value = rho * math.sqrt(noise_covariance[left, left] * noise_covariance[right, right])
    noise_covariance[left, right] = noise_covariance[right, left] = value
    metadata.update({"correlated_noise_pair": [left, right], "correlated_noise_rho": rho})
    return correlated_noise_target(source["scm"], noise_covariance, intervention), metadata


def certificate_json(certificate, root_class):
    if certificate is None:
        return None
    return {
        "shift": list(certificate.command.shift),
        "scale": list(certificate.command.scale),
        "commanded_support": certificate.commanded_support,
        "maximum_loss_residual": certificate.maximum_loss_residual,
        "candidate_losses": [
            {
                "root": root_class.roots_by_id[item.dag_id],
                "direct_kl": item.direct_kl, "optimal_kl": item.optimal_kl,
                "absolute_residual": item.absolute_residual, "tolerance": item.tolerance,
            }
            for item in certificate.candidate_losses
        ],
    }


def serialize_path(path, root_class, action_sets):
    row = dict(path)
    match = row.pop("matching_certificate")
    dag = row.pop("dag_certificate")
    row["matching_certificate"] = certificate_json(match, root_class)
    row["dag_certificate"] = certificate_json(dag, root_class)
    true_id = root_class.ids_by_root[row["true_root_evaluation_only"]]
    row["true_root_optimal_kl"] = action_sets[true_id].optimal_kl
    row["true_root_kl"] = row["true_root_excess_kl"] = None
    if match is not None:
        action_set = action_sets[true_id]
        actual = action_set.scm.interventional_moments(
            np.asarray(match.command.shift), np.asarray(match.command.scale)
        )
        value = gaussian_kl_target_to_candidate(
            action_set.target_mean, action_set.target_covariance, *actual
        )
        row["true_root_kl"] = value
        row["true_root_excess_kl"] = value - action_set.optimal_kl
    return row


def set_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (WORKER_SECONDS, WORKER_SECONDS + 1))
    if hasattr(resource, "RLIMIT_AS"):
        resource.setrlimit(resource.RLIMIT_AS, (WORKER_BYTES, WORKER_BYTES))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("wall cap")))
    signal.alarm(WORKER_SECONDS)


def run_instance(spec):
    set_limits()
    total_started = perf_counter()
    source = make_source(spec["family"], spec["replicate"])
    target, target_metadata = make_target(source, spec["regime"])
    root_class = source["root_class"]
    tick = perf_counter()
    scms = reconstruct_all_candidate_scms(
        tuple(root_class.candidates.values()), source["mean"], source["covariance"]
    )
    reconstruction = perf_counter() - tick
    tick = perf_counter()
    action_sets = compute_all_optimal_action_sets(
        scms, target_mean=target.mean, target_covariance=target.covariance,
        budget=spec["budget"], shift_bounds=SHIFT_BOUNDS,
        log_scale_bounds=LOG_SCALE_BOUNDS,
    )
    optimization = perf_counter() - tick
    cache = CertificateCache(
        action_sets, maximum_uncached_checks=CERT_CAP,
        maximum_tied_family_combinations=TIE_CAP,
    )
    tick = perf_counter()
    policy_metrics = {}
    roots = []
    for policy in POLICIES:
        policy_started = perf_counter()
        checks_before = cache.uncached_checks
        combinations_before = cache.tied_family_combinations
        certificate_seconds_before = cache.seconds
        for true_root in range(N):
            roots.append(
                serialize_path(
                    paired_policy_path(
                        policy=policy, root_class=root_class, true_root=true_root,
                        certificates=cache,
                    ),
                    root_class, action_sets,
                )
            )
        policy_metrics[policy] = {
            "marginal_seconds_with_shared_cache": perf_counter() - policy_started,
            "marginal_uncached_certificate_checks": cache.uncached_checks - checks_before,
            "marginal_tied_family_combinations": cache.tied_family_combinations - combinations_before,
            "marginal_certificate_seconds": cache.seconds - certificate_seconds_before,
            "policy_order": list(POLICIES).index(policy),
        }
    policy_time = perf_counter() - tick
    signal.alarm(0)
    summaries = {}
    for policy in POLICIES:
        selected = [row for row in roots if row["policy"] == policy]
        generated = selected[source["true_root"]]
        match_worst = max(row["matching_query_count"] for row in selected)
        dag_worst = max(row["dag_query_count"] for row in selected)
        summaries[policy] = {
            "matching_worst_case": match_worst, "dag_worst_case": dag_worst,
            "worst_case_savings": dag_worst - match_worst,
            "uniform_root_descriptive_mean_matching": mean(row["matching_query_count"] for row in selected),
            "uniform_root_descriptive_mean_dag": mean(row["dag_query_count"] for row in selected),
            "non_singleton_stop_roots": sum(row["matching_terminal_candidate_count"] > 1 for row in selected),
            "generating_root_matching": generated["matching_query_count"],
            "generating_root_dag": generated["dag_query_count"],
        }
    archive = {
        "source_seed": source["seed"], "target_seed": target_metadata["target_seed"],
        "true_root_evaluation_only": source["true_root"], "path_order": source["order"],
        "tree_edges": [list(edge) for edge in sorted(tuple(sorted(edge)) for edge in source["tree"].edges)],
        "source_coefficients": source["coefficients"].tolist(),
        "source_noise_variances": source["noise"].tolist(),
        "source_mean": source["mean"].tolist(),
        "source_covariance": source["covariance"].tolist(),
        "target_mean": target.mean.tolist(), "target_covariance": target.covariance.tolist(),
        "target_construction": target_metadata,
        "candidate_roots": [
            {
                "root": root,
                "dag_edges": [list(edge) for edge in root_class.ids_by_root[root]],
                "parent_sets": [list(x) for x in root_class.candidates[root_class.ids_by_root[root]].parent_sets],
            }
            for root in range(N)
        ],
    }
    return {
        **spec, "status": "complete", "archive": archive,
        "policy_metrics": policy_metrics,
        "candidate_count": N, "per_root_results": roots, "policy_summary": summaries,
        "graph_free_all_node_acquisition_count": N,
        "timing_seconds": {
            "reconstruction": reconstruction, "optimization": optimization,
            "policy_total_including_certification": policy_time,
            "certification": cache.seconds,
            "selection_excluding_certification": max(0.0, policy_time - cache.seconds),
            "total": perf_counter() - total_started,
        },
        "certificate_metrics": {
            "uncached_checks": cache.uncached_checks, "cache_entries": len(cache.cache),
            "tied_family_combinations": cache.tied_family_combinations,
            "cap_hits": cache.cap_hits,
            "tied_family_counts_by_root": {
                str(root_class.roots_by_id[key]): len(value.command_families)
                for key, value in action_sets.items()
            },
        },
        "fallback_count": sum(row["fallback_to_full_recovery"] for row in roots),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def worker(index, output):
    spec = all_specs()[index]
    try:
        result = run_instance(spec)
    except BaseException as error:
        result = {
            **spec, "status": "failed",
            "failure": {"type": type(error).__name__, "message": str(error)},
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    write_json(output / "instance_parts" / f"{index:03d}.json", result)
    return int(result["status"] != "complete")


def manifest():
    dependencies = [
        ROOT / "research/matching_query_complexity.py",
        ROOT / "research/s01_tree_population.py",
        ROOT / "oracle/sparse_shift_scale_kl.py",
        ROOT / "research/small_mec_matching_query/experiment.py",
        ROOT / "tests/test_s01_tree_population.py",
        ROOT / "outputs/s01_tree_population_pilot_v1/artifact_checks.py",
        Path(__file__).resolve(),
    ]
    git = lambda *args: subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    return {
        "head": git("rev-parse", "HEAD").strip(),
        "branch": git("branch", "--show-current").strip(),
        "git_status_short_before_pilot": git("status", "--short").splitlines(),
        "relevant_dirty_diff": git(
            "diff", "--", "research/matching_query_complexity.py",
            "research/small_mec_matching_query/experiment.py",
            "tests/test_small_mec_matching_query.py",
        ),
        "dependency_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in dependencies
        },
        "python": sys.executable, "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": (
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            f"NUMEXPR_NUM_THREADS=1 {sys.executable} {Path(__file__).resolve()}"
        ),
    }


def aggregate(output, overall_status):
    rows = []
    for index, spec in enumerate(all_specs()):
        path = output / "instance_parts" / f"{index:03d}.json"
        rows.append(
            json.loads(path.read_text())
            if path.exists()
            else {**spec, "status": "not_run", "failure": {"type": "OverallCap", "message": overall_status}}
        )
    instances, per_root, traces, certificates = [], [], [], []
    for row in rows:
        instances.append(json.dumps({k: v for k, v in row.items() if k != "per_root_results"}, sort_keys=True))
        for result in row.get("per_root_results", []):
            tagged = {"instance_id": row["instance_id"], **result}
            per_root.append(json.dumps(tagged, sort_keys=True))
            traces.append(json.dumps({
                "instance_id": row["instance_id"], "policy": result["policy"],
                "true_root_evaluation_only": result["true_root_evaluation_only"],
                "queried_nodes_matching": result["queried_nodes_matching"],
                "queried_nodes_full": result["queried_nodes_full"], "steps": result["steps"],
            }, sort_keys=True))
            if result["matching_certificate"] is not None:
                certificates.append(json.dumps({
                    "instance_id": row["instance_id"], "policy": result["policy"],
                    "true_root_evaluation_only": result["true_root_evaluation_only"],
                    "terminal_roots": result["matching_terminal_roots"],
                    "certificate": result["matching_certificate"],
                    "true_root_excess_kl": result["true_root_excess_kl"],
                }, sort_keys=True))
    for name, values in (
        ("instances.jsonl", instances), ("per_root_results.jsonl", per_root),
        ("query_traces.jsonl", traces), ("terminal_certificates.jsonl", certificates),
    ):
        (output / name).write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")
    complete = [row for row in rows if row["status"] == "complete"]
    failures = [row for row in rows if row["status"] != "complete"]
    summary = {
        "overall_status": overall_status if failures else "complete",
        "instances_expected": 80, "instances_complete": len(complete),
        "instances_failed_or_not_run": len(failures),
        "failures": [
            {"instance_id": row["instance_id"], "status": row["status"], "failure": row.get("failure")}
            for row in failures
        ],
    }
    if complete:
        summary["policy_aggregate"] = {
            policy: {
                "mean_worst_case_matching": mean(row["policy_summary"][policy]["matching_worst_case"] for row in complete),
                "mean_worst_case_dag": mean(row["policy_summary"][policy]["dag_worst_case"] for row in complete),
                "mean_worst_case_savings": mean(row["policy_summary"][policy]["worst_case_savings"] for row in complete),
                "instances_with_worst_case_savings": sum(row["policy_summary"][policy]["worst_case_savings"] > 0 for row in complete),
                "non_singleton_stop_root_paths": sum(row["policy_summary"][policy]["non_singleton_stop_roots"] for row in complete),
                "marginal_seconds_with_shared_cache": sum(row["policy_metrics"][policy]["marginal_seconds_with_shared_cache"] for row in complete),
                "marginal_uncached_certificate_checks": sum(row["policy_metrics"][policy]["marginal_uncached_certificate_checks"] for row in complete),
                "marginal_tied_family_combinations": sum(row["policy_metrics"][policy]["marginal_tied_family_combinations"] for row in complete),
                "marginal_certificate_seconds": sum(row["policy_metrics"][policy]["marginal_certificate_seconds"] for row in complete),
            }
            for policy in POLICIES
        }
        summary["all_returned_commands_certified"] = all(
            result["matching_certificate_status"] == "certified"
            and result["matching_certificate"] is not None
            and abs(result["true_root_excess_kl"]) <= 1e-9
            for row in complete for result in row["per_root_results"]
        )
        summary["cap_hit_count"] = sum(len(row["certificate_metrics"]["cap_hits"]) for row in complete)
        summary["fallback_count"] = sum(row["fallback_count"] for row in complete)
        summary["maximum_peak_rss_kib"] = max(row["peak_rss_kib"] for row in complete)
        summary["runtime_seconds"] = {
            "sum": sum(row["timing_seconds"]["total"] for row in complete),
            "mean": mean(row["timing_seconds"]["total"] for row in complete),
            "maximum": max(row["timing_seconds"]["total"] for row in complete),
        }
    write_json(output / "pilot_summary.json", summary)
    write_report(output, summary)


def write_report(output, summary):
    policies = summary.get("policy_aggregate", {})
    labels = (
        ("one_step_matching_aware", "One-step matching-aware"),
        ("balance", "Balance"),
        ("first_unresolved", "Algorithm 1 first-unresolved"),
        ("max_undirected_degree", "Algorithm 1 max-undirected-degree"),
    )
    table = [
        "| Policy | Worst-case MATCH mean | Full-recovery mean | Saving mean | Instances saving | Marginal s | Uncached certs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in labels:
        value = policies.get(key, {})
        table.append(
            f"| {label} | {value.get('mean_worst_case_matching', 'NA')} | "
            f"{value.get('mean_worst_case_dag', 'NA')} | "
            f"{value.get('mean_worst_case_savings', 'NA')} | "
            f"{value.get('instances_with_worst_case_savings', 'NA')} | "
            f"{value.get('marginal_seconds_with_shared_cache', 0):.3f} | "
            f"{value.get('marginal_uncached_certificate_checks', 'NA')} |"
        )
    one, balance = policies.get("one_step_matching_aware", {}), policies.get("balance", {})
    justified = (
        summary.get("overall_status") == "complete"
        and summary.get("all_returned_commands_certified") is True
        and summary.get("cap_hit_count") == 0
        and one.get("mean_worst_case_matching", 999) < balance.get("mean_worst_case_matching", -1)
    )
    report = [
        "# S01-B population tree pilot", "",
        f"Completed {summary['instances_complete']}/80 frozen instances; status: "
        f"**{summary['overall_status']}**.", "",
        "This is a population-moment study. Observation/sample counts are not "
        "applicable, and are not presented as zero experimental cost.", "",
        "## Primary results", "", *table, "",
        "Graph-free all-node acquisition is the fixed 20-environment reference.", "",
        "## Answers", "",
        "1. Matching stopping savings are the within-row differences above. "
        f"One-step saved on {one.get('instances_with_worst_case_savings', 'NA')}/80 "
        f"instances and balance on {balance.get('instances_with_worst_case_savings', 'NA')}/80.", "",
        "2. MATCH and full recovery use the identical selection path, so their "
        "difference isolates stopping. Mean stopping gains were "
        f"{one.get('mean_worst_case_savings', 'NA')} queries for one-step and "
        f"{balance.get('mean_worst_case_savings', 'NA')} for balance. Under MATCH, "
        f"one-step improved on balance by "
        f"{balance.get('mean_worst_case_matching', 0) - one.get('mean_worst_case_matching', 0):.4f} "
        "queries; under DAG stopping it was "
        f"{one.get('mean_worst_case_dag', 0) - balance.get('mean_worst_case_dag', 0):.4f} "
        "queries worse.", "",
        "3. At n=20, one-step terminal scoring was worth it for the matching-query "
        f"objective: it used {one.get('marginal_seconds_with_shared_cache', 0):.3f} "
        f"marginal seconds and {one.get('marginal_uncached_certificate_checks', 'NA')} "
        "uncached checks while reducing worst-case MATCH count by the amount above. "
        "The timing is first-in-order; later policies benefit from the shared cache, "
        "so it is not an isolated benchmark.", "",
        "4. All returned commands directly KL-certified over the complete surviving "
        f"root class: **{summary.get('all_returned_commands_certified', False)}**.", "",
        "5. The remaining limitation is complete tied-family intersection: a linear "
        "root representation does not imply polynomial worst-case certification. A "
        "cap yields inconclusive status, conservative continuation, and an explicit "
        "fallback/failure. There is no general-CPDAG, minimax-policy, or finite-sample claim.", "",
        "## Expansion decision", "",
        (
            "A separately frozen n=40 pilot is conditionally justified; n=80-100 is not "
            "yet justified."
            if justified else
            "Population expansion to n=40-100 is not justified by this pilot."
        ),
        "No expansion or finite-sample campaign was run.", "",
        "Exact commands, configuration, implementation hashes/diff, validation, "
        "unaggregated results, traces, and certificates are adjacent to this report.",
    ]
    (output / "s01_tree_population_pilot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.worker is not None:
        return worker(args.worker, output)
    if hasattr(resource, "RLIMIT_AS"):
        _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
        pilot_limit = PILOT_BYTES if hard_limit == resource.RLIM_INFINITY else min(PILOT_BYTES, hard_limit)
        resource.setrlimit(resource.RLIMIT_AS, (pilot_limit, hard_limit))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen_config.json", config())
    write_json(output / "implementation_manifest.json", manifest())
    if args.aggregate_only:
        aggregate(output, "aggregate_only")
        return 0
    usage0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    self0 = resource.getrusage(resource.RUSAGE_SELF)
    status = "complete"
    for index, spec in enumerate(all_specs()):
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        self_usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu = (usage.ru_utime + usage.ru_stime - usage0.ru_utime - usage0.ru_stime
               + self_usage.ru_utime + self_usage.ru_stime
               - self0.ru_utime - self0.ru_stime)
        if cpu >= PILOT_CPU_SECONDS:
            status = "overall_cpu_cap"
            break
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(index), "--output-dir", str(output)]
        try:
            subprocess.run(command, cwd=ROOT, env=os.environ.copy(), timeout=WORKER_SECONDS + 5, check=False)
        except subprocess.TimeoutExpired:
            write_json(output / "instance_parts" / f"{index:03d}.json", {
                **spec, "status": "failed",
                "failure": {"type": "ParentWallTimeout", "message": "worker exceeded wall cap"},
            })
    aggregate(output, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
