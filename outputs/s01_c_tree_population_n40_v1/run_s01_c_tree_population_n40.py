"""Frozen, capped S01-C n=40 population tree campaign."""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import signal
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter, time

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
from research.matching_query_complexity import compute_all_optimal_action_sets, reconstruct_all_candidate_scms
from research.s01_tree_population import CertificateCache, enumerate_tree_root_class, paired_policy_path
from research.small_mec_matching_query.experiment import _stable_seed

OUTPUT = Path(__file__).resolve().parent
MASTER_SEED = 20261012
N = 40
FAMILIES = ("path", "uniform_random_labelled_tree")
REGIMES = ("shift_scale_generated", "correlated_noise")
BUDGETS = (1, 2)
POLICIES = ("one_step_matching_aware", "balance", "first_unresolved", "max_undirected_degree")
SHIFT_BOUNDS = (-2.0, 2.0)
LOG_SCALE_BOUNDS = (-1.0, 1.0)
WORKER_SECONDS = 60
WORKER_BYTES = 512 * 1024 * 1024
CAMPAIGN_WALL_SECONDS = 2 * 60 * 60
MAX_WORKERS = 4
CERT_CAP = 2_000
TIE_CAP = 10_000
SOURCE_RETRY_LIMIT = 5
SCHEMA = "s01_c_tree_population_n40_v1"


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def write_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def write_jsonl(path, rows):
    atomic_bytes(path, b"".join(canonical_bytes(row) for row in rows))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def specs():
    return [
        {
            "execution_index": index,
            "family": family,
            "source_replicate": replicate,
            "regime": regime,
            "budget": budget,
            "source_id": f"s01c:n40:{family}:r{replicate}",
            "instance_id": f"s01c:n40:{family}:r{replicate}:{regime}:B{budget}",
        }
        for index, (family, replicate, regime, budget) in enumerate(
            (family, replicate, regime, budget)
            for family in FAMILIES
            for replicate in range(10)
            for regime in REGIMES
            for budget in BUDGETS
        )
    ]


def task_specs():
    return [
        {**spec, "policy": policy, "task_index": spec["execution_index"] * len(POLICIES) + pidx}
        for spec in specs()
        for pidx, policy in enumerate(POLICIES)
    ]


def frozen_config():
    return {
        "schema_version": SCHEMA,
        "frozen_before_policy_evaluation": True,
        "master_seed": MASTER_SEED,
        "n": N,
        "families": list(FAMILIES),
        "source_replicates_per_family": 10,
        "source_count": 20,
        "targets_per_source": 2,
        "target_regimes": list(REGIMES),
        "budgets": list(BUDGETS),
        "instance_count": 80,
        "policy_count": 4,
        "execution_order": [row["instance_id"] for row in specs()],
        "policy_order": list(POLICIES),
        "seed_schedule": {
            "source_proposal": "stable_seed(20261012,'s01c','source',family,40,replicate,proposal_index)",
            "target": "stable_seed(accepted_source_seed,'target',regime), independent of budget",
            "disjointness": "new master/tag schedule; no S01-B source seed is reused",
        },
        "source_generation": {
            "retry_limit_proposals_per_source": SOURCE_RETRY_LIMIT,
            "acceptance_checks": [
                "tree has nodes 0..39 and exactly 39 edges",
                "complete outward-root class has exactly 40 distinct candidates",
                "source moments finite and covariance positive definite",
                "all 40 candidate SCM reconstructions reproduce source moments",
            ],
            "selection_prohibitions": [
                "query cost", "matching savings", "optimum value", "certificate difficulty", "runtime"
            ],
        },
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
            "reachability_note": "generated two-coordinate target is B=2 reachable; B=1 is not assumed reachable",
        },
        "action_bounds": {"shift": list(SHIFT_BOUNDS), "log_scale": list(LOG_SCALE_BOUNDS)},
        "caps": {
            "campaign_wall_seconds": CAMPAIGN_WALL_SECONDS,
            "worker_wall_seconds": WORKER_SECONDS,
            "worker_cpu_seconds": WORKER_SECONDS,
            "worker_rss_bytes": WORKER_BYTES,
            "uncached_certificate_searches_per_instance_policy": CERT_CAP,
            "cumulative_tied_family_product_upper_bound_per_instance_policy": TIE_CAP,
        },
        "concurrency": {"maximum_workers": MAX_WORKERS, "submission_order_frozen": True},
        "cache": {
            "scope": "fresh isolated cache per instance and policy; reused across all 40 true-root replays",
            "shared_across_policies": False,
        },
        "timing_scope": {
            "worker": "reconstruction, 40 candidate optimizations, one policy over every root, serialization",
            "certificate_seconds": "uncached calls to complete tied-family intersection and direct KL certification",
            "selection_excluding_certificate": "policy replay wall time minus certificate seconds",
        },
        "thread_environment": {key: os.environ[key] for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "stopping": {
            "MATCH": "first directly certified common physical command over every surviving root",
            "full_recovery": "one surviving root",
            "inconclusive": "continue conservatively and record fallback; never authorize stopping",
        },
        "historical_design_discrepancy": "S01-B did not serialize proposal/retry logs and shared caches across policies; S01-C preserves its distributions but separately freezes proposals and isolates caches.",
    }


def make_source(family, replicate, proposal_index):
    seed = _stable_seed(MASTER_SEED, "s01c", "source", family, N, replicate, proposal_index)
    rng = np.random.default_rng(seed)
    if family == "path":
        order = [int(value) for value in rng.permutation(N)]
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
    checks = {
        "tree_shape": set(tree.nodes) == set(range(N)) and tree.number_of_edges() == N - 1 and nx.is_tree(tree),
        "complete_root_class": len(root_class.candidates) == N and set(root_class.ids_by_root) == set(range(N)),
        "finite_moments": bool(np.all(np.isfinite(source_mean)) and np.all(np.isfinite(source_covariance))),
        "positive_definite_covariance": bool(np.linalg.eigvalsh(source_covariance).min() > 0.0),
    }
    reconstructed = reconstruct_all_candidate_scms(tuple(root_class.candidates.values()), source_mean, source_covariance)
    checks["candidate_reconstruction"] = all(
        np.allclose(candidate.observational_moments()[0], source_mean, atol=1e-11, rtol=1e-10)
        and np.allclose(candidate.observational_moments()[1], source_covariance, atol=1e-11, rtol=1e-10)
        for candidate in reconstructed.values()
    )
    return {
        "seed": seed, "tree": tree, "order": order, "true_root": true_root,
        "coefficients": coefficients, "noise": noise, "scm": scm,
        "mean": source_mean, "covariance": source_covariance, "checks": checks,
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
        target = generate_realisable_target(source["scm"], intervention)
    else:
        noise_covariance = np.diag(np.asarray(source["noise"], dtype=float))
        left, right = (int(value) for value in rng.choice(N, size=2, replace=False))
        rho = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.12, 0.28))
        covariance = rho * math.sqrt(noise_covariance[left, left] * noise_covariance[right, right])
        noise_covariance[left, right] = noise_covariance[right, left] = covariance
        metadata.update({"correlated_noise_pair": [left, right], "correlated_noise_rho": rho})
        target = correlated_noise_target(source["scm"], noise_covariance, intervention)
    checks = {
        "finite_moments": bool(np.all(np.isfinite(target.mean)) and np.all(np.isfinite(target.covariance))),
        "positive_definite_covariance": bool(np.linalg.eigvalsh(target.covariance).min() > 0.0),
        "generating_support": int(np.count_nonzero(np.abs(shift) > 1e-10) + np.count_nonzero(np.abs(scale - 1.0) > 1e-10)),
    }
    if not all(value is True or (key == "generating_support" and value == 2) for key, value in checks.items()):
        raise RuntimeError(f"predetermined target failed validity checks: {checks}")
    return target, metadata, checks


def freeze_inputs(output):
    config_path = output / "frozen_config.json"
    instances_path = output / "frozen_instances.jsonl"
    proposals_path = output / "proposal_and_eligibility_manifest.json"
    desired = frozen_config()
    if config_path.exists() or instances_path.exists() or proposals_path.exists():
        if not (config_path.exists() and instances_path.exists() and proposals_path.exists()):
            raise RuntimeError("partial frozen bank exists; refusing to regenerate")
        if json.loads(config_path.read_text()) != desired:
            raise RuntimeError("existing frozen configuration differs from this implementation")
        rows = read_jsonl(instances_path)
        if len(rows) != 80 or [row["instance_id"] for row in rows] != desired["execution_order"]:
            raise RuntimeError("existing frozen instance bank is incomplete or reordered")
        return rows

    write_json(config_path, desired)
    sources = {}
    source_logs = []
    for family in FAMILIES:
        for replicate in range(10):
            proposals = []
            accepted = None
            for proposal_index in range(SOURCE_RETRY_LIMIT):
                try:
                    candidate = make_source(family, replicate, proposal_index)
                    accepted_now = all(candidate["checks"].values())
                    proposals.append({
                        "proposal_index": proposal_index, "seed": candidate["seed"],
                        "checks": candidate["checks"], "accepted": accepted_now,
                        "rejection_reason": None if accepted_now else "numerical_or_structural_validity",
                    })
                    if accepted_now:
                        accepted = candidate
                        break
                except Exception as error:
                    proposals.append({
                        "proposal_index": proposal_index,
                        "seed": _stable_seed(MASTER_SEED, "s01c", "source", family, N, replicate, proposal_index),
                        "checks": {}, "accepted": False,
                        "rejection_reason": f"{type(error).__name__}: {error}",
                    })
            if accepted is None:
                raise RuntimeError(f"source retry cap exhausted for {family}/{replicate}")
            source_id = f"s01c:n40:{family}:r{replicate}"
            sources[(family, replicate)] = accepted
            source_logs.append({
                "source_id": source_id, "family": family, "replicate": replicate,
                "retry_limit": SOURCE_RETRY_LIMIT, "proposals": proposals,
                "accepted_proposal_index": len(proposals) - 1,
            })

    rows = []
    target_logs = []
    for spec in specs():
        source = sources[(spec["family"], spec["source_replicate"])]
        target, metadata, checks = make_target(source, spec["regime"])
        root_class = enumerate_tree_root_class(source["tree"])
        row = {
            **spec,
            "source_seed": source["seed"], "target_seed": metadata["target_seed"],
            "true_root_evaluation_only": source["true_root"], "path_order": source["order"],
            "tree_edges": [list(edge) for edge in sorted(tuple(sorted(edge)) for edge in source["tree"].edges)],
            "source_coefficients": source["coefficients"].tolist(),
            "source_noise_variances": source["noise"].tolist(),
            "source_mean": source["mean"].tolist(), "source_covariance": source["covariance"].tolist(),
            "target_mean": target.mean.tolist(), "target_covariance": target.covariance.tolist(),
            "target_construction": metadata,
            "candidate_roots": [
                {"root": root, "dag_edges": [list(edge) for edge in root_class.ids_by_root[root]]}
                for root in range(N)
            ],
            "source_validity_checks": source["checks"], "target_validity_checks": checks,
        }
        rows.append(row)
        target_logs.append({
            "instance_id": spec["instance_id"], "target_seed": metadata["target_seed"],
            "checks": checks, "accepted": True,
            "budget_independent_target_hash": sha256_bytes(canonical_bytes({"mean": row["target_mean"], "covariance": row["target_covariance"]})),
        })
    write_jsonl(instances_path, rows)
    write_json(proposals_path, {
        "schema_version": SCHEMA, "source_retry_limit": SOURCE_RETRY_LIMIT,
        "source_proposals": source_logs, "target_proposals": target_logs,
        "selection_used_outcomes": False,
        "valid_sources": len(source_logs), "valid_instances": len(rows), "intended_instances": 80,
    })
    return rows


def dependency_paths():
    return [
        ROOT / "research/matching_query_complexity.py",
        ROOT / "research/s01_tree_population.py",
        ROOT / "oracle/sparse_shift_scale_kl.py",
        ROOT / "research/small_mec_matching_query/experiment.py",
        ROOT / "tests/test_s01_tree_population.py",
        ROOT / "tests/test_s01_c_tree_population.py",
        ROOT / "tests/test_s01_tied_optimum_validation.py",
        Path(__file__).resolve(),
        OUTPUT / "artifact_checks.py",
    ]


def current_dependency_hashes():
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in dependency_paths()}


def git_output(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def protected_hashes():
    paths = []
    for dirname in ("s01_tree_population_pilot_v1", "s01_b2_exact_oracle_validation_v1", "s01_b3_tied_optimum_validation_v1"):
        base = ROOT / "outputs" / dirname
        for name in ("frozen_config.json", "implementation_manifest.json", "pilot_summary.json", "summary.json", "validation_results.json", "s01_tree_population_pilot_report.md", "s01_b2_exact_oracle_validation_report.md", "s01_b3_tied_optimum_validation_report.md"):
            path = base / name
            if path.exists():
                paths.append(path)
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def environment_info():
    packages = {}
    for package in ("numpy", "scipy", "networkx", "pytest"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    memory = next((line.strip() for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")), "unknown")
    return {
        "python": sys.executable, "python_version": platform.python_version(), "platform": platform.platform(),
        "packages": packages,
        "hardware": {"logical_cpu_count": os.cpu_count(), "machine": platform.machine(), "processor": platform.processor(), "memory_total": memory},
        "thread_environment": {key: os.environ[key] for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }


def initialize_manifest(output):
    path = output / "implementation_manifest.json"
    frozen_hashes = {
        "frozen_config.json": sha256_file(output / "frozen_config.json"),
        "frozen_instances.jsonl": sha256_file(output / "frozen_instances.jsonl"),
        "proposal_and_eligibility_manifest.json": sha256_file(output / "proposal_and_eligibility_manifest.json"),
    }
    dependencies = current_dependency_hashes()
    fingerprint = sha256_bytes(canonical_bytes({"frozen": frozen_hashes, "dependencies": dependencies}))
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["campaign_fingerprint"] != fingerprint:
            raise RuntimeError("implementation or frozen inputs changed; refusing silent resume")
        return manifest
    historical = {}
    for name in ("s01_tree_population_pilot_v1", "s01_b2_exact_oracle_validation_v1", "s01_b3_tied_optimum_validation_v1"):
        historical_path = ROOT / "outputs" / name / "implementation_manifest.json"
        if historical_path.exists():
            value = json.loads(historical_path.read_text())
            historical[name] = {"head": value.get("head"), "implementation_manifest_sha256": sha256_file(historical_path)}
    manifest = {
        "schema_version": SCHEMA,
        "campaign_fingerprint": fingerprint,
        "frozen_artifact_sha256_before_outcomes": frozen_hashes,
        "dependency_sha256": dependencies,
        "head": git_output("rev-parse", "HEAD").strip(),
        "branch": git_output("branch", "--show-current").strip(),
        "git_status_short_before_campaign": git_output("status", "--short").splitlines(),
        "relevant_dirty_diff_before_campaign": git_output("diff", "--", "research/matching_query_complexity.py", "research/s01_tree_population.py", "tests/test_s01_tree_population.py"),
        "protected_prior_artifact_sha256_before": protected_hashes(),
        "current_campaign_environment": environment_info(),
        "historical_campaign_provenance": historical,
        "commands": {
            "run_or_resume": f"{sys.executable} {Path(__file__).resolve()}",
            "aggregate_only": f"{sys.executable} {Path(__file__).resolve()} --aggregate-only",
            "artifact_validation": f"{sys.executable} {output / 'artifact_checks.py'}",
        },
        "limits_scope": "Each instance-policy worker includes candidate reconstruction/optimization, all-root replay, scoring/certification, and serialization; parent enforces wall timeout. Campaign cap is parent elapsed wall time.",
        "instrumentation_change": "Optional counters only: cache hits/outcomes, structural candidates reached, and direct KL calls. No policy, intervention, optimization, tie, bound, or tolerance semantics changed.",
    }
    write_json(path, manifest)
    return manifest


def verify_fingerprint(output):
    manifest = json.loads((output / "implementation_manifest.json").read_text())
    frozen = {name: sha256_file(output / name) for name in ("frozen_config.json", "frozen_instances.jsonl", "proposal_and_eligibility_manifest.json")}
    dependencies = current_dependency_hashes()
    fingerprint = sha256_bytes(canonical_bytes({"frozen": frozen, "dependencies": dependencies}))
    if fingerprint != manifest["campaign_fingerprint"]:
        raise RuntimeError("campaign fingerprint mismatch")
    return fingerprint


def set_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (WORKER_SECONDS, WORKER_SECONDS + 1))
    if hasattr(resource, "RLIMIT_AS"):
        resource.setrlimit(resource.RLIMIT_AS, (WORKER_BYTES, WORKER_BYTES))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("worker wall cap")))
    signal.alarm(WORKER_SECONDS)


def certificate_json(certificate, root_class):
    if certificate is None:
        return None
    return {
        "shift": list(certificate.command.shift), "scale": list(certificate.command.scale),
        "commanded_support": certificate.commanded_support,
        "maximum_loss_residual": certificate.maximum_loss_residual,
        "candidate_losses": [
            {
                "root": root_class.roots_by_id[item.dag_id], "direct_kl": item.direct_kl,
                "optimal_kl": item.optimal_kl, "absolute_residual": item.absolute_residual,
                "tolerance": item.tolerance,
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
        actual = action_set.scm.interventional_moments(np.asarray(match.command.shift), np.asarray(match.command.scale))
        value = gaussian_kl_target_to_candidate(action_set.target_mean, action_set.target_covariance, *actual)
        row["true_root_kl"] = value
        row["true_root_excess_kl"] = value - action_set.optimal_kl
    return row


def run_task(task, frozen, fingerprint):
    total_started = perf_counter()
    tree = nx.Graph()
    tree.add_nodes_from(range(N))
    tree.add_edges_from(tuple(edge) for edge in frozen["tree_edges"])
    root_class = enumerate_tree_root_class(tree)
    if len(root_class.candidates) != 40:
        raise RuntimeError("frozen tree did not enumerate exactly 40 roots")
    source_mean = np.asarray(frozen["source_mean"], dtype=float)
    source_covariance = np.asarray(frozen["source_covariance"], dtype=float)
    target_mean = np.asarray(frozen["target_mean"], dtype=float)
    target_covariance = np.asarray(frozen["target_covariance"], dtype=float)
    tick = perf_counter()
    scms = reconstruct_all_candidate_scms(tuple(root_class.candidates.values()), source_mean, source_covariance)
    reconstruction = perf_counter() - tick
    tick = perf_counter()
    action_sets = compute_all_optimal_action_sets(
        scms, target_mean=target_mean, target_covariance=target_covariance,
        budget=task["budget"], shift_bounds=SHIFT_BOUNDS, log_scale_bounds=LOG_SCALE_BOUNDS,
    )
    optimization = perf_counter() - tick
    cache = CertificateCache(action_sets, maximum_uncached_checks=CERT_CAP, maximum_tied_family_combinations=TIE_CAP)
    tick = perf_counter()
    roots = [
        serialize_path(paired_policy_path(policy=task["policy"], root_class=root_class, true_root=true_root, certificates=cache), root_class, action_sets)
        for true_root in range(N)
    ]
    replay = perf_counter() - tick
    signal.alarm(0)
    match_worst = max(row["matching_query_count"] for row in roots)
    dag_worst = max(row["dag_query_count"] for row in roots)
    residuals = [
        loss["absolute_residual"]
        for row in roots if row["matching_certificate"] is not None
        for loss in row["matching_certificate"]["candidate_losses"]
    ]
    return {
        **task, "status": "complete", "campaign_fingerprint": fingerprint,
        "candidate_count": len(action_sets), "cache_id": f"{task['instance_id']}::{task['policy']}::{fingerprint[:16]}",
        "cache_configuration": {"fresh_per_instance_policy": True, "reused_across_true_roots": True, "uncached_cap": CERT_CAP, "tied_product_cap": TIE_CAP},
        "policy_summary": {
            "matching_worst_case": match_worst, "dag_worst_case": dag_worst,
            "worst_case_savings": dag_worst - match_worst,
            "uniform_root_descriptive_mean_matching": mean(row["matching_query_count"] for row in roots),
            "uniform_root_descriptive_mean_dag": mean(row["dag_query_count"] for row in roots),
            "non_singleton_stop_roots": sum(row["matching_terminal_candidate_count"] > 1 for row in roots),
        },
        "per_root_results": roots,
        "timing_seconds": {
            "candidate_reconstruction": reconstruction, "candidate_optimization": optimization,
            "scoring_and_replay_including_certification": replay,
            "certification": cache.seconds,
            "selection_and_replay_excluding_certification": max(0.0, replay - cache.seconds),
            "total": perf_counter() - total_started,
        },
        "candidate_optimizer": {
            "calls": len(action_sets),
            "retained_optimum_multiplicity_by_root": {str(root_class.roots_by_id[key]): len(value.actions) for key, value in action_sets.items()},
            "total_retained_optima": sum(len(value.actions) for value in action_sets.values()),
        },
        "certificate_metrics": {
            "attempts_total": cache.uncached_checks + cache.cache_hits,
            "cache_hits": cache.cache_hits, "cache_entries": len(cache.cache),
            "uncached_searches": cache.uncached_checks,
            "outcomes_uncached_states": cache.outcome_counts,
            "tied_family_product_upper_bounds_charged": cache.tied_family_combinations,
            "structural_combinations_actually_visited": cache.structural_candidates_visited,
            "direct_kl_evaluations": cache.direct_kl_evaluations,
            "cap_hits": cache.cap_hits,
            "maximum_certificate_residual": max(residuals, default=0.0),
        },
        "fallback_count": sum(row["fallback_to_full_recovery"] for row in roots),
        "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def next_attempt_path(output, task):
    directory = output / "attempts" / f"{task['task_index']:03d}_{task['policy']}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"attempt_{len(list(directory.glob('attempt_*.json'))):03d}.json"


def worker(task_index, output):
    task = task_specs()[task_index]
    part = output / "task_parts" / f"{task_index:03d}.json"
    if part.exists():
        return 0
    attempt_path = next_attempt_path(output, task)
    started = time()
    try:
        fingerprint = verify_fingerprint(output)
        set_limits()
        frozen_rows = read_jsonl(output / "frozen_instances.jsonl")
        frozen = frozen_rows[task["execution_index"]]
        if frozen["instance_id"] != task["instance_id"]:
            raise RuntimeError("frozen execution order mismatch")
        result = run_task(task, frozen, fingerprint)
    except BaseException as error:
        signal.alarm(0)
        result = {
            **task, "status": "failed",
            "failure": {"type": type(error).__name__, "message": str(error)},
            "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    attempt = {"started_unix": started, "ended_unix": time(), "result": result}
    write_json(attempt_path, attempt)
    write_json(part, result)
    return int(result["status"] != "complete")


def run_subprocess(task_index, output):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(task_index), "--output-dir", str(output)]
    try:
        completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), timeout=WORKER_SECONDS + 5, capture_output=True, text=True)
        return {"task_index": task_index, "returncode": completed.returncode, "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]}
    except subprocess.TimeoutExpired as error:
        task = task_specs()[task_index]
        result = {**task, "status": "failed", "failure": {"type": "ParentWallTimeout", "message": "worker exceeded 65 second parent timeout"}}
        write_json(next_attempt_path(output, task), {"started_unix": None, "ended_unix": time(), "result": result})
        write_json(output / "task_parts" / f"{task_index:03d}.json", result)
        return {"task_index": task_index, "returncode": 124, "stdout": "", "stderr": str(error)}


def histogram(values):
    return {str(key): value for key, value in sorted(Counter(values).items())}


def aggregate(output, campaign_status):
    frozen = read_jsonl(output / "frozen_instances.jsonl")
    tasks = []
    for task in task_specs():
        path = output / "task_parts" / f"{task['task_index']:03d}.json"
        tasks.append(json.loads(path.read_text()) if path.exists() else {**task, "status": "not_run", "failure": {"type": "CampaignLimit", "message": campaign_status}})
    by_instance = defaultdict(dict)
    for row in tasks:
        by_instance[row["instance_id"]][row["policy"]] = row
    instance_rows = []
    per_root_rows = []
    trace_rows = []
    certificate_rows = []
    work_rows = []
    failures = []
    incompletes = []
    for frozen_row in frozen:
        policies = by_instance[frozen_row["instance_id"]]
        complete = [policies.get(policy, {}).get("status") == "complete" for policy in POLICIES]
        cap_hits = sum(len(policies[policy]["certificate_metrics"]["cap_hits"]) for policy in POLICIES if policies.get(policy, {}).get("status") == "complete")
        status = "complete" if all(complete) and cap_hits == 0 else "complete_capped" if all(complete) else "incomplete"
        instance_rows.append({
            **frozen_row, "status": status, "completed_policy_count": sum(complete), "cap_hit_count": cap_hits,
            "policy_summary": {policy: policies[policy]["policy_summary"] for policy in POLICIES if policies.get(policy, {}).get("status") == "complete"},
        })
        if status != "complete":
            incompletes.append({"instance_id": frozen_row["instance_id"], "status": status, "cap_hit_count": cap_hits})
    for task in tasks:
        if task["status"] != "complete":
            failures.append({"task_index": task["task_index"], "instance_id": task["instance_id"], "policy": task["policy"], "status": task["status"], "failure": task.get("failure")})
            continue
        work_rows.append({key: value for key, value in task.items() if key not in ("per_root_results",)})
        for root in task["per_root_results"]:
            tagged = {"instance_id": task["instance_id"], "source_id": task["source_id"], "family": task["family"], "regime": task["regime"], "budget": task["budget"], **root}
            per_root_rows.append(tagged)
            trace_rows.append({"instance_id": task["instance_id"], "policy": task["policy"], "true_root_evaluation_only": root["true_root_evaluation_only"], "queried_nodes_matching": root["queried_nodes_matching"], "queried_nodes_full": root["queried_nodes_full"], "steps": root["steps"]})
            if root["matching_certificate"] is not None:
                certificate_rows.append({"instance_id": task["instance_id"], "policy": task["policy"], "true_root_evaluation_only": root["true_root_evaluation_only"], "terminal_roots": root["matching_terminal_roots"], "certificate": root["matching_certificate"], "true_root_optimal_kl": root["true_root_optimal_kl"], "true_root_kl": root["true_root_kl"], "true_root_excess_kl": root["true_root_excess_kl"]})
    write_jsonl(output / "instances.jsonl", instance_rows)
    write_jsonl(output / "per_root_results.jsonl", per_root_rows)
    write_jsonl(output / "query_traces.jsonl", trace_rows)
    write_jsonl(output / "terminal_certificates.jsonl", certificate_rows)
    write_jsonl(output / "computational_work.jsonl", work_rows)
    write_jsonl(output / "failures.jsonl", failures)
    write_jsonl(output / "incomplete.jsonl", incompletes)
    complete_instances = [row for row in instance_rows if row["completed_policy_count"] == len(POLICIES)]
    summary = {
        "overall_status": "complete" if len(complete_instances) == 80 and not failures else campaign_status,
        "denominators": {
            "intended_instances": 80, "attempted_instances": sum(row["completed_policy_count"] > 0 for row in instance_rows),
            "valid_frozen_instances": 80, "completed_all_policy_instances": len(complete_instances),
            "uncapped_complete_instances": sum(row["status"] == "complete" for row in instance_rows),
            "capped_complete_instances": sum(row["status"] == "complete_capped" for row in instance_rows),
            "incomplete_instances": sum(row["status"] == "incomplete" for row in instance_rows),
            "failed_or_not_run_instance_policy_tasks": len(failures),
        },
        "failures": failures,
        "cap_hit_count": sum(len(row["certificate_metrics"]["cap_hits"]) for row in tasks if row["status"] == "complete"),
        "fallback_count": sum(row["fallback_count"] for row in tasks if row["status"] == "complete"),
        "cache_isolation": {"unique_cache_ids": len({row["cache_id"] for row in tasks if row["status"] == "complete"}), "complete_tasks": sum(row["status"] == "complete" for row in tasks)},
    }
    if complete_instances:
        policy_aggregate = {}
        for policy in POLICIES:
            values = [row["policy_summary"][policy] for row in complete_instances]
            matching = [value["matching_worst_case"] for value in values]
            dag = [value["dag_worst_case"] for value in values]
            savings = [value["worst_case_savings"] for value in values]
            policy_tasks = [row for row in tasks if row["status"] == "complete" and row["policy"] == policy]
            policy_aggregate[policy] = {
                "denominator": len(values),
                "mean_worst_case_matching": mean(matching), "median_worst_case_matching": median(matching), "distribution_worst_case_matching": histogram(matching),
                "mean_worst_case_full_recovery": mean(dag), "median_worst_case_full_recovery": median(dag), "distribution_worst_case_full_recovery": histogram(dag),
                "mean_worst_case_savings": mean(savings), "savings_sign_counts": {"positive": sum(x > 0 for x in savings), "zero": sum(x == 0 for x in savings), "negative": sum(x < 0 for x in savings)},
                "non_singleton_match_terminal_root_paths": sum(value["non_singleton_stop_roots"] for value in values),
                "work": {
                    "worker_seconds_sum": sum(row["timing_seconds"]["total"] for row in policy_tasks),
                    "certificate_seconds_sum": sum(row["timing_seconds"]["certification"] for row in policy_tasks),
                    "uncached_searches": sum(row["certificate_metrics"]["uncached_searches"] for row in policy_tasks),
                    "cache_hits": sum(row["certificate_metrics"]["cache_hits"] for row in policy_tasks),
                    "tied_product_upper_bounds": sum(row["certificate_metrics"]["tied_family_product_upper_bounds_charged"] for row in policy_tasks),
                    "structural_combinations_visited": sum(row["certificate_metrics"]["structural_combinations_actually_visited"] for row in policy_tasks),
                    "direct_kl_evaluations": sum(row["certificate_metrics"]["direct_kl_evaluations"] for row in policy_tasks),
                    "candidate_optimizer_calls": sum(row["candidate_optimizer"]["calls"] for row in policy_tasks),
                    "maximum_peak_worker_rss_kib": max((row["peak_worker_rss_kib"] for row in policy_tasks), default=None),
                },
            }
        summary["policy_aggregate"] = policy_aggregate
        one = [row["policy_summary"]["one_step_matching_aware"] for row in complete_instances]
        bal = [row["policy_summary"]["balance"] for row in complete_instances]
        summary["one_step_minus_balance"] = {}
        for label, field in (("matching", "matching_worst_case"), ("full_recovery", "dag_worst_case")):
            differences = [left[field] - right[field] for left, right in zip(one, bal, strict=True)]
            summary["one_step_minus_balance"][label] = {"mean": mean(differences), "distribution": histogram(differences), "positive": sum(x > 0 for x in differences), "zero": sum(x == 0 for x in differences), "negative": sum(x < 0 for x in differences)}
        source_groups = defaultdict(list)
        for row in complete_instances:
            source_groups[row["source_id"]].append(row["policy_summary"]["one_step_matching_aware"]["matching_worst_case"])
        summary["source_level_variation"] = {"source_count": len(source_groups), "one_step_matching_source_means": {key: mean(value) for key, value in sorted(source_groups.items())}, "minimum_source_mean": min(mean(value) for value in source_groups.values()), "maximum_source_mean": max(mean(value) for value in source_groups.values())}
        summary["all_returned_commands_serialized_as_certified"] = all(row["matching_certificate_status"] == "certified" and row["matching_certificate"] is not None and abs(row["true_root_excess_kl"]) <= 1e-9 for row in per_root_rows)
        summary["maximum_peak_worker_rss_kib"] = max((row["peak_worker_rss_kib"] for row in tasks if row["status"] == "complete"), default=None)
    write_json(output / "pilot_summary.json", summary)
    write_report(output, summary)


def write_report(output, summary):
    aggregates = summary.get("policy_aggregate", {})
    labels = (("one_step_matching_aware", "One-step matching-aware"), ("balance", "Balance"), ("first_unresolved", "Algorithm 1 first-unresolved"), ("max_undirected_degree", "Algorithm 1 max-undirected-degree"))
    table = ["| Policy | Complete-case n | Worst-case MATCH mean | Full-recovery mean | Saving mean | +/0/- savings | Uncached certs | Worker s |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label in labels:
        value = aggregates.get(key, {})
        work = value.get("work", {})
        signs = value.get("savings_sign_counts", {})
        table.append(f"| {label} | {value.get('denominator', 'NA')} | {value.get('mean_worst_case_matching', 'NA')} | {value.get('mean_worst_case_full_recovery', 'NA')} | {value.get('mean_worst_case_savings', 'NA')} | {signs.get('positive', 'NA')}/{signs.get('zero', 'NA')}/{signs.get('negative', 'NA')} | {work.get('uncached_searches', 'NA')} | {work.get('worker_seconds_sum', 'NA')} |")
    den = summary["denominators"]
    comparison = summary.get("one_step_minus_balance", {})
    validation = None
    if (output / "validation_results.json").exists():
        validation = json.loads((output / "validation_results.json").read_text()).get("status")
    report = [
        "# S01-C capped n=40 population tree pilot", "",
        f"Campaign outcome: **{summary['overall_status']}**. Frozen/valid/completed-all-policy instances: 80/80/{den['completed_all_policy_instances']}; uncapped complete: {den['uncapped_complete_instances']}; capped complete: {den['capped_complete_instances']}; incomplete: {den['incomplete_instances']}; failed/not-run policy tasks: {den['failed_or_not_run_instance_policy_tasks']}.", "",
        "This is a population-moment computation. Observation counts are not applicable; this is not a zero-sample practical method.", "",
        "## Paired query results", "", *table, "",
        f"One-step minus balance (negative favors one-step): MATCH {comparison.get('matching', {})}; full recovery {comparison.get('full_recovery', {})}.", "",
        f"Non-singleton MATCH terminals and full cost distributions are recorded in `pilot_summary.json`. Variation is summarized over 20 sources; the 80 target-budget instances are not treated as 80 independent sources.", "",
        "## Certificate and computational validity", "",
        f"All serialized returned commands passed in-run bounds/budget/candidate/true-root checks: **{summary.get('all_returned_commands_serialized_as_certified', False)}**. Independent artifact validation: **{validation or 'pending'}**.",
        f"Certificate cap hits: {summary.get('cap_hit_count')}; conservative fallbacks: {summary.get('fallback_count')}; maximum worker RSS: {summary.get('maximum_peak_worker_rss_kib')} KiB. Family-product upper bounds and combinations actually visited are reported separately in `computational_work.jsonl` and the summary.", "",
        "Each policy used a fresh instance-local cache, reused only across its 40 true-root replays. Consequently these work counts are isolated-policy measurements and are not directly comparable to S01-B's shared-cache marginal counts.", "",
        "## Evidence boundary and expansion decision", "",
        "This campaign measures capped n=40 feasibility and complete-root-class certificate behavior for the existing heuristic on separately generated banks. It does not establish minimax optimality, finite-sample performance, arbitrary-CPDAG coverage, asymptotic complexity, or a paired n=20/n=40 comparison.",
        "The archived n=20 means were 1.2875/4.1125 (one-step MATCH/full recovery) and 2.7250/3.7500 (balance); comparisons here are descriptive because the banks differ and cache-work protocols differ.",
        "Further expansion is justified only if all 80 instances complete without material cap/failure rates and the measured work remains comfortably within limits. No n=80 or n=100 campaign was launched.", "",
        "Exact frozen inputs, proposal logs, commands, hashes, unaggregated records, failures, and validation are adjacent to this report.",
    ]
    atomic_bytes(output / "s01_c_tree_population_n40_report.md", ("\n".join(report) + "\n").encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker is not None:
        return worker(args.worker, output)
    freeze_inputs(output)
    initialize_manifest(output)
    verify_fingerprint(output)
    if args.aggregate_only:
        aggregate(output, "aggregate_only")
        return 0
    started = perf_counter()
    pending = [task["task_index"] for task in task_specs() if not (output / "task_parts" / f"{task['task_index']:03d}.json").exists()]
    status = "complete"
    events = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.max_workers, MAX_WORKERS))) as executor:
        futures = {}
        for task_index in pending:
            if perf_counter() - started >= CAMPAIGN_WALL_SECONDS:
                status = "campaign_wall_cap"
                break
            futures[executor.submit(run_subprocess, task_index, output)] = task_index
        for future in concurrent.futures.as_completed(futures):
            events.append(future.result())
            write_jsonl(output / "execution_log.jsonl", sorted(events, key=lambda row: row["task_index"]))
            if perf_counter() - started >= CAMPAIGN_WALL_SECONDS:
                status = "campaign_wall_cap"
    aggregate(output, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
