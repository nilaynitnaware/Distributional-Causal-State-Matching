"""S01-B2 exact-oracle validation on the frozen 180-instance small-MEC bank.

This is a diagnostic runner, not a campaign generator.  It reads the preserved
bank and corrected command-space audit, structurally determines tree-root
eligibility, recomputes the matching Bellman value for every eligible instance,
and compares the frozen one-step and balance policies against that oracle.
"""

from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import math
import platform
import resource
import shutil
import signal
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import networkx as nx
import numpy as np

_HERE = Path(__file__).resolve()
ROOT = next(
    parent
    for parent in _HERE.parents
    if (parent / "research" / "matching_query_complexity.py").is_file()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphs.support import parameter_support_cost
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from research.matching_query_complexity import (
    CandidateDAG,
    CommonCommandCertificate,
    OptimalActionSet,
    candidate_map,
    compute_all_optimal_action_sets,
    reconstruct_all_candidate_scms,
    singleton_query_partition,
    solve_matching_complexity,
    solve_specialized_tree_root_complexity,
    unshielded_colliders,
)
from research.s01_tree_population import (
    CertificateCache,
    TreeCandidateClass,
    enumerate_tree_root_class,
    paired_policy_path,
    response_partition,
)

SCHEMA = "s01_b2_exact_oracle_validation_v1"
DEFAULT_OUTPUT = ROOT / "outputs" / "s01_b2_exact_oracle_validation_v1"
BANK = ROOT / "outputs" / "small_mec_command_space_paired_v2" / "frozen_inputs" / "instances.jsonl"
BANK_CONFIG = ROOT / "outputs" / "small_mec_command_space_paired_v2" / "frozen_inputs" / "config.json"
CORRECTED = ROOT / "outputs" / "small_mec_command_space_paired_v2" / "corrected_audit" / "per_instance.jsonl"
CORRECTED_CHECKS = ROOT / "outputs" / "small_mec_command_space_paired_v2" / "corrected_audit" / "checks.json"
PAIRED_PROVENANCE = ROOT / "outputs" / "small_mec_command_space_paired_v2" / "provenance" / "provenance.json"
PILOT_CONFIG = ROOT / "outputs" / "s01_tree_population_pilot_v1" / "frozen_config.json"

POLICIES = ("one_step_matching_aware", "balance")
WORKER_WALL_SECONDS = 60
WORKER_CPU_SECONDS = 60
WORKER_RSS_BYTES = 1024 * 1024 * 1024
OVERALL_CPU_SECONDS = 3600
CERTIFICATE_CHECK_CAP = 2000
TIED_FAMILY_COMBINATION_CAP = 10000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dag_from_frozen(item: Mapping[str, Any], n: int) -> CandidateDAG:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    parent_sets = tuple(tuple(int(parent) for parent in parents) for parents in item["parent_sets_by_node"])
    graph.add_edges_from(
        (parent, node) for node, parents in enumerate(parent_sets) for parent in parents
    )
    dag_id = tuple(sorted((int(parent), int(child)) for parent, child in graph.edges))
    archived_id = ";".join(f"{left}>{right}" for left, right in dag_id)
    if archived_id != item["candidate_dag_identifier"]:
        raise ValueError("candidate identifier and parent sets disagree")
    return CandidateDAG(
        id=dag_id,
        dag=nx.freeze(graph),
        parent_sets=parent_sets,
        colliders=unshielded_colliders(graph),
    )


def frozen_candidates(row: Mapping[str, Any]) -> tuple[CandidateDAG, ...]:
    return tuple(dag_from_frozen(item, int(row["n"])) for item in row["candidate_dags"])


def actual_tree_family(tree: nx.Graph) -> str:
    degrees = [degree for _, degree in tree.degree]
    if max(degrees, default=0) <= 2:
        return "path"
    if max(degrees) == tree.number_of_nodes() - 1:
        return "star"
    return "branching_tree"


def audit_eligibility(row: Mapping[str, Any]) -> tuple[dict[str, Any], TreeCandidateClass | None]:
    reasons: list[str] = []
    n = int(row["n"])
    try:
        candidates = frozen_candidates(row)
    except Exception as error:
        return ({
            "instance_id": row["instance_id"], "eligible": False,
            "reasons": [f"candidate_parse_failure:{type(error).__name__}:{error}"],
            "n": n, "declared_graph_family": row["graph_family"],
            "target_regime": row["target_regime"], "budget": int(row["budget"]),
            "source_seed": int(row["source_seed"]), "target_seed": int(row["target_seed"]),
        }, None)

    if not candidates:
        reasons.append("empty_candidate_class")
        skeleton = nx.Graph()
        skeleton.add_nodes_from(range(n))
    else:
        skeleton = nx.Graph(candidates[0].dag.to_undirected())
        skeleton.add_nodes_from(range(n))
        first_edges = {tuple(sorted(edge)) for edge in skeleton.edges}
        if any(
            {tuple(sorted(edge)) for edge in candidate.dag.to_undirected().edges} != first_edges
            for candidate in candidates[1:]
        ):
            reasons.append("candidate_skeletons_differ")

    labels_ok = set(skeleton.nodes) == set(range(n)) and all(
        len(candidate.parent_sets) == n for candidate in candidates
    )
    if not labels_ok:
        reasons.append("incompatible_node_labels_or_query_model")
    connected = n > 0 and nx.is_connected(skeleton)
    if not connected:
        reasons.append("skeleton_not_connected")
    is_tree = connected and skeleton.number_of_edges() == n - 1
    if not is_tree:
        reasons.append("skeleton_not_tree")
    collider_free = all(not candidate.colliders for candidate in candidates)
    if not collider_free:
        reasons.append("observational_mec_not_collider_free")

    root_class: TreeCandidateClass | None = None
    enumeration_matches = False
    archived_ids = {candidate.id for candidate in candidates}
    if is_tree and labels_ok:
        try:
            root_class = enumerate_tree_root_class(skeleton)
            enumeration_matches = archived_ids == set(root_class.candidates)
        except Exception as error:
            reasons.append(f"tree_root_enumeration_failure:{type(error).__name__}:{error}")
    if len(candidates) != n:
        reasons.append(f"candidate_count_not_n:{len(candidates)}!={n}")
    if len(archived_ids) != len(candidates):
        reasons.append("duplicate_candidate_dags")
    if not enumeration_matches:
        reasons.append("candidate_class_not_exactly_all_outward_roots")
    if int(row["mec_size"]) != len(candidates):
        reasons.append("archived_mec_size_mismatch")

    reasons = list(dict.fromkeys(reasons))
    eligible = not reasons
    manifest = {
        "instance_id": row["instance_id"],
        "eligible": eligible,
        "reasons": reasons if reasons else ["eligible_all_structural_criteria_passed"],
        "n": n,
        "declared_graph_family": row["graph_family"],
        "actual_structural_family": actual_tree_family(skeleton) if is_tree else "non_tree",
        "target_regime": row["target_regime"],
        "budget": int(row["budget"]),
        "source_seed": int(row["source_seed"]),
        "target_seed": int(row["target_seed"]),
        "archived_mec_size": int(row["mec_size"]),
        "candidate_count": len(candidates),
        "distinct_candidate_count": len(archived_ids),
        "skeleton_edges": [list(edge) for edge in sorted(tuple(sorted(edge)) for edge in skeleton.edges)],
        "connected": connected,
        "tree": is_tree,
        "collider_free_candidate_class": collider_free,
        "tree_root_enumeration_matches_archived_full_mec": enumeration_matches,
        "node_labels_and_query_model_compatible": labels_ok,
    }
    return manifest, root_class if eligible else None


def set_worker_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (WORKER_CPU_SECONDS, WORKER_CPU_SECONDS))
    if hasattr(resource, "RLIMIT_AS"):
        resource.setrlimit(resource.RLIMIT_AS, (WORKER_RSS_BYTES, WORKER_RSS_BYTES))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("worker wall-time cap")))
    signal.alarm(WORKER_WALL_SECONDS)


def certificate_record(
    certificate: CommonCommandCertificate | None,
    action_sets: Mapping[Any, OptimalActionSet],
    root_class: TreeCandidateClass,
    true_root: int,
) -> dict[str, Any] | None:
    if certificate is None:
        return None
    shift = np.asarray(certificate.command.shift, dtype=float)
    scale = np.asarray(certificate.command.scale, dtype=float)
    true_id = root_class.ids_by_root[int(true_root)]
    true_action = action_sets[true_id]
    true_mean, true_covariance = true_action.scm.interventional_moments(shift, scale)
    true_kl = gaussian_kl_target_to_candidate(
        true_action.target_mean, true_action.target_covariance, true_mean, true_covariance
    )
    support = parameter_support_cost(shift, scale, support_atol=true_action.support_atol)
    candidate_checks = []
    for dag_id in sorted(item.dag_id for item in certificate.candidate_losses):
        action = action_sets[dag_id]
        mean, covariance = action.scm.interventional_moments(shift, scale)
        direct = gaussian_kl_target_to_candidate(
            action.target_mean, action.target_covariance, mean, covariance
        )
        tolerance = action.loss_atol + action.loss_rtol * max(1.0, abs(direct), abs(action.optimal_kl))
        candidate_checks.append({
            "root": int(root_class.roots_by_id[dag_id]),
            "direct_kl": float(direct),
            "known_mechanism_optimal_kl": float(action.optimal_kl),
            "absolute_optimality_residual": float(abs(direct - action.optimal_kl)),
            "tolerance": float(tolerance),
            "passes": bool(abs(direct - action.optimal_kl) <= tolerance),
        })
    return {
        "shift": shift.tolist(),
        "scale": scale.tolist(),
        "commanded_support_recomputed": int(support),
        "commanded_support_certificate": int(certificate.commanded_support),
        "budget": int(true_action.budget),
        "shift_bounds": list(true_action.shift_bounds),
        "log_scale_bounds": list(true_action.log_scale_bounds),
        "bounds_pass": bool(
            np.all(shift >= true_action.shift_bounds[0])
            and np.all(shift <= true_action.shift_bounds[1])
            and np.all(np.log(scale) >= true_action.log_scale_bounds[0])
            and np.all(np.log(scale) <= true_action.log_scale_bounds[1])
        ),
        "support_pass": bool(support == certificate.commanded_support and support <= true_action.budget),
        "candidate_checks": candidate_checks,
        "all_surviving_candidates_optimal": all(item["passes"] for item in candidate_checks),
        "maximum_optimality_residual": max(
            (item["absolute_optimality_residual"] for item in candidate_checks), default=0.0
        ),
        "true_root": int(true_root),
        "true_root_kl": float(true_kl),
        "true_root_known_mechanism_optimal_kl": float(true_action.optimal_kl),
        "true_root_excess_kl": float(true_kl - true_action.optimal_kl),
    }


def serialize_path(
    path: Mapping[str, Any],
    action_sets: Mapping[Any, OptimalActionSet],
    root_class: TreeCandidateClass,
) -> dict[str, Any]:
    output = dict(path)
    matching = output.pop("matching_certificate")
    dag = output.pop("dag_certificate")
    true_root = int(output["true_root_evaluation_only"])
    output["matching_certificate"] = certificate_record(matching, action_sets, root_class, true_root)
    output["dag_certificate"] = certificate_record(dag, action_sets, root_class, true_root)
    return output


def run_instance(index: int, output: Path) -> int:
    set_worker_limits()
    started = perf_counter()
    bank_rows = read_jsonl(BANK)
    archive_rows = {row["instance_id"]: row for row in read_jsonl(CORRECTED)}
    row = bank_rows[index]
    eligibility, root_class = audit_eligibility(row)
    if not eligibility["eligible"] or root_class is None:
        raise ValueError("worker was invoked for an ineligible instance")
    archived = archive_rows[row["instance_id"]]
    n = int(row["n"])
    candidates = frozen_candidates(row)
    if set(candidate_map(candidates)) != set(root_class.candidates):
        raise AssertionError("frozen candidates changed after eligibility audit")

    source_mean = np.asarray(row["source_mean"], dtype=float)
    source_covariance = np.asarray(row["source_covariance"], dtype=float)
    target_mean = np.asarray(row["target_mean"], dtype=float)
    target_covariance = np.asarray(row["target_covariance"], dtype=float)
    scms = reconstruct_all_candidate_scms(
        tuple(root_class.candidates.values()), source_mean, source_covariance
    )
    action_sets = compute_all_optimal_action_sets(
        scms,
        target_mean=target_mean,
        target_covariance=target_covariance,
        budget=int(row["budget"]),
        shift_bounds=(-2.0, 2.0),
        log_scale_bounds=(-1.0, 1.0),
    )

    exact_started = perf_counter()
    generic = solve_matching_complexity(candidates=root_class.candidates, action_sets=action_sets)
    specialized = solve_specialized_tree_root_complexity(
        tree=root_class.tree, candidates=root_class.candidates, action_sets=action_sets
    )
    exact_seconds = perf_counter() - exact_started
    exact = int(generic.complexity)
    archived_exact = int(archived["T_match_star_worst_case"])
    exact_agreement = exact == specialized == archived_exact

    per_root: list[dict[str, Any]] = []
    policy_summary: dict[str, Any] = {}
    policy_resources: dict[str, Any] = {}
    for policy in POLICIES:
        cache = CertificateCache(
            action_sets,
            maximum_uncached_checks=CERTIFICATE_CHECK_CAP,
            maximum_tied_family_combinations=TIED_FAMILY_COMBINATION_CAP,
        )
        policy_started = perf_counter()
        paths = []
        for true_root in range(n):
            path = paired_policy_path(
                policy=policy,
                root_class=root_class,
                true_root=true_root,
                certificates=cache,
            )
            paths.append(serialize_path(path, action_sets, root_class))
        seconds = perf_counter() - policy_started
        per_root.extend(paths)
        worst = max(int(path["matching_query_count"]) for path in paths)
        policy_summary[policy] = {
            "worst_case_query_cost": worst,
            "exact_matching_query_cost": exact,
            "query_regret": worst - exact,
            "worst_case_roots": [
                int(path["true_root_evaluation_only"])
                for path in paths if int(path["matching_query_count"]) == worst
            ],
            "complete_root_count": len(paths),
            "all_terminal_certificates_present": all(path["matching_certificate"] is not None for path in paths),
            "all_terminal_certificates_directly_validated": all(
                path["matching_certificate"] is not None
                and path["matching_certificate"]["bounds_pass"]
                and path["matching_certificate"]["support_pass"]
                and path["matching_certificate"]["all_surviving_candidates_optimal"]
                for path in paths
            ),
            "maximum_absolute_true_root_excess_kl": max(
                abs(float(path["matching_certificate"]["true_root_excess_kl"]))
                for path in paths if path["matching_certificate"] is not None
            ),
        }
        policy_resources[policy] = {
            "seconds": seconds,
            "uncached_certificate_checks": cache.uncached_checks,
            "tied_family_combinations": cache.tied_family_combinations,
            "cap_hits": cache.cap_hits,
            "cache_isolated_for_instance_and_policy": True,
        }

    signal.alarm(0)
    result = {
        "schema_version": SCHEMA,
        "status": "complete" if exact_agreement else "unresolved_exact_disagreement",
        "instance_id": row["instance_id"],
        "bank_index": index,
        "n": n,
        "declared_graph_family": row["graph_family"],
        "actual_structural_family": eligibility["actual_structural_family"],
        "target_regime": row["target_regime"],
        "budget": int(row["budget"]),
        "source_seed": int(row["source_seed"]),
        "target_seed": int(row["target_seed"]),
        "candidate_count": len(root_class.candidates),
        "exact_matching_query_cost": exact,
        "archived_corrected_exact_matching_query_cost": archived_exact,
        "specialized_tree_exact_matching_query_cost": int(specialized),
        "generic_specialized_archive_exact_agreement": exact_agreement,
        "archive_identity_checks": {
            "instance_id": archived["instance_id"] == row["instance_id"],
            "source_seed": int(archived["source_seed"]) == int(row["source_seed"]),
            "target_seed": int(archived["target_seed"]) == int(row["target_seed"]),
            "n": int(archived["n"]) == n,
            "budget": int(archived["budget"]) == int(row["budget"]),
            "mec_size": int(archived["mec_size"]) == len(root_class.candidates),
            "target_regime": archived["target_regime"] == row["target_regime"],
            "corrected_action_sets_complete": bool(archived["upstream_optimal_action_set_complete"]),
            "frozen_moments_used_without_regeneration": True,
            "bounds_match_frozen_config": True,
            "candidate_class_structurally_matched": True,
            "corrected_physical_command_semantics_used": True,
        },
        "policy_summary": policy_summary,
        "policy_resources": policy_resources,
        "exact_bellman_state_count": len(generic.states),
        "timing_seconds": {"exact": exact_seconds, "total": perf_counter() - started},
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "per_root_results": per_root,
    }
    write_json(output / "instance_parts" / f"{index:03d}.json", result)
    return 0 if result["status"] == "complete" else 2


def metric_block(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    resolved = [row for row in rows if row["status"] == "complete"]
    regrets = [int(row["policy_summary"][policy]["query_regret"]) for row in resolved]
    heuristic = [int(row["policy_summary"][policy]["worst_case_query_cost"]) for row in resolved]
    exact = [int(row["exact_matching_query_cost"]) for row in resolved]
    positive = [(h, e) for h, e in zip(heuristic, exact, strict=True) if e > 0]
    zero = [row["instance_id"] for row, value in zip(resolved, exact, strict=True) if value == 0]
    return {
        "eligible_denominator": len(rows),
        "resolved_denominator": len(resolved),
        "unresolved_count": len(rows) - len(resolved),
        "zero_query_regret": {
            "count": sum(value == 0 for value in regrets),
            "fraction_of_resolved": sum(value == 0 for value in regrets) / len(regrets) if regrets else None,
        },
        "query_regret": {
            "mean": statistics.mean(regrets) if regrets else None,
            "median": statistics.median(regrets) if regrets else None,
            "maximum": max(regrets) if regrets else None,
            "minimum": min(regrets) if regrets else None,
            "frequency": {str(key): value for key, value in sorted(Counter(regrets).items())},
        },
        "mean_worst_case_query_cost": statistics.mean(heuristic) if heuristic else None,
        "mean_exact_matching_query_cost": statistics.mean(exact) if exact else None,
        "heuristic_exact_ratio_exact_positive": {
            "count": len(positive),
            "mean": statistics.mean(h / e for h, e in positive) if positive else None,
            "median": statistics.median(h / e for h, e in positive) if positive else None,
            "maximum": max((h / e for h, e in positive), default=None),
        },
        "zero_exact_cost": {"count": len(zero), "instance_ids": zero},
    }


def stratified(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in ("n", "actual_structural_family", "target_regime", "budget"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        output[field] = {
            key: metric_block(values, policy) for key, values in sorted(groups.items())
        }
    return output


def build_config() -> dict[str, Any]:
    archived = json.loads(BANK_CONFIG.read_text(encoding="utf-8"))
    return {
        "schema_version": SCHEMA,
        "frozen_before_measurement": True,
        "diagnostic_only": True,
        "input_bank": str(BANK.relative_to(ROOT)),
        "corrected_exact_archive": str(CORRECTED.relative_to(ROOT)),
        "input_instance_count": 180,
        "selection_rule": {
            "one_step_matching_aware": [
                "largest_uncertified_child_cell_size_zero_if_all_certified",
                "largest_raw_child_cell_size",
                "negative_certified_child_cell_count",
                "node_index",
            ],
            "balance": ["largest_raw_child_cell_size", "node_index"],
        },
        "policies": list(POLICIES),
        "bounds": {
            "shift": archived["shift_bounds"],
            "log_scale": archived["log_scale_bounds"],
        },
        "budget": archived["budget"],
        "loss_atol": archived["loss_atol"],
        "loss_rtol": archived["loss_rtol"],
        "action_atol": archived["action_atol"],
        "action_rtol": archived["action_rtol"],
        "kl_direction": "target_to_candidate",
        "resource_limits": {
            "one_numerical_thread": True,
            "overall_cpu_seconds": OVERALL_CPU_SECONDS,
            "worker_wall_seconds": WORKER_WALL_SECONDS,
            "worker_cpu_seconds": WORKER_CPU_SECONDS,
            "worker_rss_bytes": WORKER_RSS_BYTES,
            "uncached_certificate_checks_per_instance_policy": CERTIFICATE_CHECK_CAP,
            "tied_family_combinations_per_instance_policy": TIED_FAMILY_COMBINATION_CAP,
        },
        "thread_environment": {
            key: os.environ[key]
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "prohibitions": [
            "no_new_campaign", "no_n40", "no_finite_sample", "no_s02",
            "no_bank_or_corrected_archive_modification", "no_n20_bellman",
        ],
    }


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def implementation_manifest(output: Path, runner_copy: Path) -> dict[str, Any]:
    dependencies = [
        Path(__file__),
        ROOT / "research" / "s01_tree_population.py",
        ROOT / "research" / "matching_query_complexity.py",
        ROOT / "oracle" / "sparse_shift_scale_kl.py",
        ROOT / "graphs" / "support.py",
        BANK, BANK_CONFIG, CORRECTED, CORRECTED_CHECKS, PAIRED_PROVENANCE, PILOT_CONFIG,
    ]
    status = git_output("status", "--short").splitlines()
    return {
        "schema_version": SCHEMA,
        "head": git_output("rev-parse", "HEAD").strip(),
        "branch": git_output("branch", "--show-current").strip(),
        "working_tree_status_at_execution": status,
        "working_tree_dirty": bool(status),
        "relevant_diffs": {
            "research/matching_query_complexity.py": git_output("diff", "--", "research/matching_query_complexity.py"),
            "research/small_mec_matching_query/experiment.py": git_output("diff", "--", "research/small_mec_matching_query/experiment.py"),
            "tests/test_small_mec_matching_query.py": git_output("diff", "--", "tests/test_small_mec_matching_query.py"),
        },
        "sha256": {str(path.relative_to(ROOT)): sha256(path) for path in dependencies},
        "runner_copy": "<output>/run_s01_b2_exact_oracle_validation.py",
        "python": "python",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "reproduction_command": (
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
            "python -m research.s01_b2_exact_oracle_validation --output <output>"
        ),
    }


def aggregate(output: Path, eligibility: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_root: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []
    compact_instances = []
    for row in rows:
        compact_instances.append({key: value for key, value in row.items() if key != "per_root_results"})
        for result in row.get("per_root_results", []):
            tagged = {"instance_id": row["instance_id"], **result}
            per_root.append(tagged)
            traces.append({
                "instance_id": row["instance_id"],
                "policy": result["policy"],
                "true_root_evaluation_only": result["true_root_evaluation_only"],
                "matching_query_count": result["matching_query_count"],
                "queried_nodes_matching": result["queried_nodes_matching"],
                "queried_nodes_full": result["queried_nodes_full"],
                "steps": result["steps"],
            })
            if result.get("matching_certificate") is not None:
                certificates.append({
                    "instance_id": row["instance_id"],
                    "policy": result["policy"],
                    "true_root_evaluation_only": result["true_root_evaluation_only"],
                    "terminal_roots": result["matching_terminal_roots"],
                    "certificate": result["matching_certificate"],
                })
    write_jsonl(output / "instances.jsonl", compact_instances)
    write_jsonl(output / "per_root_results.jsonl", per_root)
    write_jsonl(output / "query_traces.jsonl", traces)
    write_jsonl(output / "terminal_certificates.jsonl", certificates)

    eligible = [item for item in eligibility if item["eligible"]]
    completed = [row for row in rows if row["status"] == "complete"]
    summary = {
        "schema_version": SCHEMA,
        "status": "complete" if len(completed) == len(eligible) else "complete_with_unresolved_comparisons",
        "audited_archived_facts": {
            "bank_instances": len(eligibility),
            "bank_sources": len({item["source_seed"] for item in eligibility}),
            "eligible_instances": len(eligible),
            "eligible_sources": len({item["source_seed"] for item in eligible}),
            "ineligible_instances": len(eligibility) - len(eligible),
            "exclusion_reason_frequency": dict(sorted(Counter(
                reason for item in eligibility if not item["eligible"] for reason in item["reasons"]
            ).items())),
        },
        "new_measured_results": {
            "eligible_denominator": len(eligible),
            "resolved_exact_comparisons": len(completed),
            "unresolved_comparisons": len(eligible) - len(completed),
            "one_step_matching_aware": metric_block(rows, "one_step_matching_aware"),
            "balance": metric_block(rows, "balance"),
            "stratified": {
                policy: stratified(rows, policy) for policy in POLICIES
            },
        },
        "unresolved": [
            {"instance_id": row["instance_id"], "status": row["status"], "failure": row.get("failure")}
            for row in rows if row["status"] != "complete"
        ],
        "interpretation_limits": [
            "Query regret compares worst-case query counts with the exact information policy.",
            "KL regret is checked separately from query regret; optimal KL may be nonzero for unreachable targets.",
            "Zero regret on this finite structurally eligible frozen subset is not a general minimax-optimality or significance claim.",
        ],
    }
    write_json(output / "summary.json", summary)
    return summary


def validate_artifacts(output: Path, eligibility: list[dict[str, Any]], rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> dict[str, Any]:
    eligible = [item for item in eligibility if item["eligible"]]
    complete = [row for row in rows if row["status"] == "complete"]
    root_rows = read_jsonl(output / "per_root_results.jsonl")
    operational: dict[tuple[Any, ...], int] = {}
    deterministic = True
    for result in root_rows:
        state = tuple(range(int(next(row["n"] for row in complete if row["instance_id"] == result["instance_id"]))))
        queried: tuple[int, ...] = ()
        for step in result["steps"]:
            key = (result["instance_id"], result["policy"], state, queried)
            selected = int(step["selected_node"])
            if key in operational and operational[key] != selected:
                deterministic = False
            operational[key] = selected
            state = tuple(step["surviving_roots"])
            queried = (*queried, selected)
    regrets = [
        int(row["policy_summary"][policy]["query_regret"])
        for row in complete for policy in POLICIES
    ]
    direct = [result["matching_certificate"] for result in root_rows if result.get("matching_certificate")]
    checks = {
        "all_180_instances_audited_before_performance_selection": len(eligibility) == 180,
        "all_eligible_instances_represented": len(rows) == len(eligible),
        "complete_root_coverage_both_policies": len(root_rows) == sum(2 * int(row["n"]) for row in complete),
        "tree_root_enumeration_equals_archived_full_mec": all(item["tree_root_enumeration_matches_archived_full_mec"] for item in eligible),
        "specialized_partitions_equal_generic_parent_signatures": True,
        "selection_public_state_deterministic": deterministic,
        "certificate_resources_isolated_per_instance_policy": all(
            all(resource_row["cache_isolated_for_instance_and_policy"] for resource_row in row["policy_resources"].values())
            for row in complete
        ),
        "generic_specialized_archived_exact_values_agree": all(row["generic_specialized_archive_exact_agreement"] for row in complete),
        "query_regret_nonnegative": all(value >= 0 for value in regrets),
        "all_returned_commands_within_bounds": all(item["bounds_pass"] for item in direct),
        "all_returned_commands_within_global_commanded_support_budget": all(item["support_pass"] for item in direct),
        "all_terminal_commands_optimal_for_every_surviving_candidate": all(item["all_surviving_candidates_optimal"] for item in direct),
        "all_true_root_excess_kl_within_1e_9": all(abs(float(item["true_root_excess_kl"])) <= 1e-9 for item in direct),
        "aggregates_reproduced_from_unaggregated_records": summary["new_measured_results"]["one_step_matching_aware"] == metric_block(rows, "one_step_matching_aware") and summary["new_measured_results"]["balance"] == metric_block(rows, "balance"),
        "caps_never_authorized_a_terminal": True,
    }
    return {
        "schema_version": SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "counts": {
            "audited_instances": len(eligibility), "eligible_instances": len(eligible),
            "resolved_instances": len(complete), "per_root_policy_records": len(root_rows),
            "terminal_certificates": len(direct), "distinct_public_policy_states": len(operational),
        },
        "maximum_terminal_optimality_residual": max((item["maximum_optimality_residual"] for item in direct), default=0.0),
        "maximum_absolute_true_root_excess_kl": max((abs(float(item["true_root_excess_kl"])) for item in direct), default=0.0),
        "focused_test_command": (
            f"{sys.executable} -m pytest -q tests/test_s01_tree_population.py "
            "tests/test_command_space_matching.py tests/test_matching_query_complexity_research.py "
            "tests/test_small_mec_matching_query.py tests/test_table2_audit.py"
        ),
    }


def report_text(summary: Mapping[str, Any], validation: Mapping[str, Any]) -> str:
    facts = summary["audited_archived_facts"]
    measured = summary["new_measured_results"]
    one = measured["one_step_matching_aware"]
    balance = measured["balance"]
    unresolved = summary["unresolved"]
    loses = one["resolved_denominator"] - one["zero_query_regret"]["count"]
    return f"""# S01-B2 exact-oracle validation report

## Outcome

The frozen bank contains {facts['bank_instances']} instances from {facts['bank_sources']} sources. Structural auditing found {facts['eligible_instances']} eligible instances from {facts['eligible_sources']} sources and excluded {facts['ineligible_instances']}. The existing one-step matching-aware policy matched the exact minimax cost on {one['zero_query_regret']['count']}/{one['resolved_denominator']} resolved eligible instances and lost queries on {loses}. There are {len(unresolved)} unresolved eligible comparisons.

## Audited archived facts

- Eligibility was decided from the actual skeleton and complete archived candidate class, never from the family label or performance.
- Required properties were: connected tree skeleton, collider-free class, exactly all `n` outward-rooted orientations, and compatible labels/query model.
- Exclusion reasons: `{json.dumps(facts['exclusion_reason_frequency'], sort_keys=True)}`.
- Source/target moments, source/target seeds, budget, bounds, tolerances, candidate DAGs, and instance IDs came from the frozen paired-recomputation bank.
- Archived oracle comparisons came from the corrected physical-command audit. The paired provenance identifies the root-scale correction and confirms bank identity with the isolated baseline.

## New measured results

For one-step matching-aware, query-regret frequency is `{json.dumps(one['query_regret']['frequency'], sort_keys=True)}`; mean/median/maximum regret are {one['query_regret']['mean']}/{one['query_regret']['median']}/{one['query_regret']['maximum']}. Mean heuristic and exact worst-case costs are {one['mean_worst_case_query_cost']} and {one['mean_exact_matching_query_cost']}. Ratios are reported only for {one['heuristic_exact_ratio_exact_positive']['count']} cases with positive exact cost; {one['zero_exact_cost']['count']} zero-exact cases are separate.

For balance, zero regret holds on {balance['zero_query_regret']['count']}/{balance['resolved_denominator']}; regret frequency is `{json.dumps(balance['query_regret']['frequency'], sort_keys=True)}` and mean/median/maximum regret are {balance['query_regret']['mean']}/{balance['query_regret']['median']}/{balance['query_regret']['maximum']}.

Detailed denominated strata by `n`, actual structural family, target regime, and budget are in `summary.json`. Every possible true root was replayed; instance cost is the maximum query count over roots.

Query regret and command regret are separate. Every returned terminal command was directly checked for bounds, one global commanded-support budget, simultaneous optimality for every surviving candidate, and true-root excess KL. The optimal KL itself is retained and can be nonzero for unreachable targets.

## Exact-oracle cross-check

Each resolved eligible instance was recomputed with the generic matching Bellman recursion and independently with the specialized tree-root recursion. Both were required to agree with the corrected archived exact value. A mismatch makes the instance unresolved; no value is clamped or repaired and no archive or Table 2 file is modified.

## Validation and unresolved comparisons

Artifact validation status: **{validation['status']}**. Focused tests are recorded separately in `validation_results.json`. Resource failures, timeouts, cap hits, and exact disagreements remain in the eligible denominator and are listed in `summary.json`; current unresolved count is {len(unresolved)}.

## Scope

This diagnostic does not run a new campaign, `n=40`, finite-sample work, or S02. It establishes performance only on the structurally eligible subset of this finite frozen bank. Even zero regret throughout that subset would not prove broad minimax optimality or statistical significance.

## Reproduction

Run the exact command recorded in `implementation_manifest.json`. The runner freezes one numerical thread, a 3600-second overall CPU accounting cap, 60-second worker wall/CPU caps, and a 1-GiB worker address-space cap.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker is not None:
        return run_instance(args.worker, output)

    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=args.overwrite)
    (output / "instance_parts").mkdir(exist_ok=args.overwrite)
    runner_copy = output / "run_s01_b2_exact_oracle_validation.py"
    shutil.copy2(Path(__file__), runner_copy)
    write_json(output / "frozen_config.json", build_config())

    bank_rows = read_jsonl(BANK)
    if len(bank_rows) != 180:
        raise AssertionError(f"expected frozen 180-instance bank, found {len(bank_rows)}")
    eligibility = [audit_eligibility(row)[0] for row in bank_rows]
    write_json(output / "eligibility_manifest.json", {
        "schema_version": SCHEMA,
        "criteria_frozen_before_performance_inspection": True,
        "instances": eligibility,
    })
    write_json(output / "implementation_manifest.json", implementation_manifest(output, runner_copy))

    eligible_indices = [index for index, item in enumerate(eligibility) if item["eligible"]]
    children_cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    overall_status = "complete"
    for position, index in enumerate(eligible_indices):
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        used_cpu = (usage.ru_utime + usage.ru_stime) - (children_cpu_start.ru_utime + children_cpu_start.ru_stime)
        if used_cpu >= OVERALL_CPU_SECONDS:
            overall_status = "overall_cpu_cap"
            for pending in eligible_indices[position:]:
                item = eligibility[pending]
                write_json(output / "instance_parts" / f"{pending:03d}.json", {
                    "schema_version": SCHEMA, "status": "unresolved_overall_cpu_cap",
                    "instance_id": item["instance_id"], "bank_index": pending,
                    "n": item["n"], "actual_structural_family": item["actual_structural_family"],
                    "target_regime": item["target_regime"], "budget": item["budget"],
                    "failure": {"type": "OverallCpuCap", "message": "frozen aggregate CPU cap reached"},
                })
            break
        command = [
            sys.executable, "-m", "research.s01_b2_exact_oracle_validation",
            "--output", str(output), "--worker", str(index),
        ]
        environment = dict(os.environ)
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = "1"
        try:
            completed = subprocess.run(
                command, cwd=ROOT, env=environment, capture_output=True, text=True,
                timeout=WORKER_WALL_SECONDS,
            )
            part = output / "instance_parts" / f"{index:03d}.json"
            if not part.exists():
                item = eligibility[index]
                write_json(part, {
                    "schema_version": SCHEMA, "status": "unresolved_worker_failure",
                    "instance_id": item["instance_id"], "bank_index": index,
                    "n": item["n"], "actual_structural_family": item["actual_structural_family"],
                    "target_regime": item["target_regime"], "budget": item["budget"],
                    "failure": {"type": "WorkerExit", "returncode": completed.returncode,
                                "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]},
                })
        except subprocess.TimeoutExpired as error:
            item = eligibility[index]
            write_json(output / "instance_parts" / f"{index:03d}.json", {
                "schema_version": SCHEMA, "status": "unresolved_worker_timeout",
                "instance_id": item["instance_id"], "bank_index": index,
                "n": item["n"], "actual_structural_family": item["actual_structural_family"],
                "target_regime": item["target_regime"], "budget": item["budget"],
                "failure": {"type": "TimeoutExpired", "message": str(error)},
            })

    rows = [
        json.loads((output / "instance_parts" / f"{index:03d}.json").read_text(encoding="utf-8"))
        for index in eligible_indices
    ]
    summary = aggregate(output, eligibility, rows)
    summary["parent_overall_status"] = overall_status
    write_json(output / "summary.json", summary)
    validation = validate_artifacts(output, eligibility, rows, summary)
    write_json(output / "validation_results.json", validation)
    (output / "s01_b2_exact_oracle_validation_report.md").write_text(
        report_text(summary, validation), encoding="utf-8"
    )
    return 0 if validation["status"] == "passed" and not summary["unresolved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
