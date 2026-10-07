"""Reproducible exact experiment comparing matching and DAG query complexity.

The implementation intentionally targets small graphs.  It reuses the tested
MEC/action/Bellman prototype in :mod:`research.matching_query_complexity`, the
production exact sparse shift--scale optimizer, and production Algorithm 1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx
import numpy as np

from environments.blackbox import EnvironmentQueryLedger, ExactMomentBlackBoxEnvironment
from graphs.DAG_gen import (
    generate_erdos_renyi_dag,
    generate_path_dag,
    generate_rooted_tree_dag,
)
from graphs.interventions import ShiftScaleIntervention
from graphs.nonrealisable_targets import correlated_noise_target
from graphs.scm import LinearGaussianSCM, sample_coefficients
from graphs.target_gen import GaussianTarget, generate_realisable_target
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation
from query_complexity.core import (
    FirstUnresolvedPolicy,
    MaxUndirectedDegreePolicy,
    run_query_method,
)
from research.matching_query_complexity import (
    BellmanResult,
    CandidateDAG,
    DagId,
    OptimalActionSet,
    audit_singleton_query_signatures,
    candidate_map,
    compute_all_optimal_action_sets,
    enumerate_consistent_dags,
    reconstruct_all_candidate_scms,
    solve_dag_complexity,
    solve_matching_complexity,
)

SCHEMA_VERSION = "small_mec_matching_query_v1"
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2] / "outputs" / "small_mec_matching_query"
)


@dataclass(frozen=True)
class CampaignConfig:
    schema_version: str = SCHEMA_VERSION
    seed: int = 20260826
    node_counts: tuple[int, ...] = (5, 6, 7)
    graph_families: tuple[str, ...] = ("path", "rooted_tree", "erdos_renyi")
    replicates_per_cell: int = 10
    target_regimes: tuple[str, ...] = ("realisable", "correlated_noise")
    budget: int = 2
    shift_bounds: tuple[float, float] = (-2.0, 2.0)
    log_scale_bounds: tuple[float, float] = (-1.0, 1.0)
    coefficient_min_abs: float = 0.25
    coefficient_max_abs: float = 0.65
    noise_variance_bounds: tuple[float, float] = (0.7, 1.4)
    erdos_renyi_edge_probability: float = 0.35
    minimum_mec_size: int = 2
    maximum_mec_size: int = 32
    maximum_undirected_edges: int = 12
    maximum_source_attempts: int = 200
    loss_atol: float = 1e-12
    loss_rtol: float = 1e-11
    action_atol: float = 1e-10
    action_rtol: float = 1e-9

    @classmethod
    def from_json(cls, path: Path) -> CampaignConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        tuple_fields = {
            "node_counts",
            "graph_families",
            "target_regimes",
            "shift_bounds",
            "log_scale_bounds",
            "noise_variance_bounds",
        }
        for name in tuple_fields:
            if name in payload:
                payload[name] = tuple(payload[name])
        known = {field.name for field in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported schema version")
        if not self.node_counts or min(self.node_counts) < 2:
            raise ValueError("node_counts must contain integers at least two")
        allowed_families = {"path", "rooted_tree", "erdos_renyi"}
        if not set(self.graph_families) <= allowed_families:
            raise ValueError("unsupported graph family")
        if set(self.target_regimes) != {"realisable", "correlated_noise"}:
            raise ValueError("campaign requires both target regimes")
        if self.replicates_per_cell < 1 or self.budget < 1:
            raise ValueError("replicates and budget must be positive")
        if self.minimum_mec_size < 2 or self.maximum_mec_size < self.minimum_mec_size:
            raise ValueError("invalid MEC-size bounds")
        if self.maximum_undirected_edges < 1 or self.maximum_source_attempts < 1:
            raise ValueError("enumeration guards must be positive")


@dataclass(frozen=True)
class SourceProblem:
    source_seed: int
    n: int
    graph_family: str
    replicate: int
    generation_attempt: int
    scm: LinearGaussianSCM
    cpdag: Any
    candidates: tuple[CandidateDAG, ...]
    mec_enumeration_seconds: float


def _stable_seed(base: int, *parts: object) -> int:
    message = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little")


def _dag_id_text(dag_id: DagId) -> str:
    return ";".join(f"{left}>{right}" for left, right in dag_id)


def _graph(
    config: CampaignConfig, family: str, n: int, rng: np.random.Generator
) -> nx.DiGraph:
    if family == "path":
        return generate_path_dag(n, rng, permute=True)
    if family == "rooted_tree":
        return generate_rooted_tree_dag(n, rng)
    return generate_erdos_renyi_dag(n, config.erdos_renyi_edge_probability, rng)


def _source_problem(
    config: CampaignConfig, family: str, n: int, replicate: int
) -> tuple[SourceProblem | None, list[dict[str, object]]]:
    failures: list[dict[str, object]] = []
    for attempt in range(config.maximum_source_attempts):
        seed = _stable_seed(config.seed, "source", family, n, replicate, attempt)
        rng = np.random.default_rng(seed)
        dag = _graph(config, family, n, rng)
        if dag.number_of_edges() == 0:
            failures.append({"stage": "source", "reason": "empty_graph", "seed": seed})
            continue
        cpdag = cpdag_from_true_dag_for_simulation(dag)
        if len(cpdag.undirected_edges) > config.maximum_undirected_edges:
            failures.append(
                {
                    "stage": "source",
                    "reason": "too_many_undirected_edges",
                    "seed": seed,
                    "count": len(cpdag.undirected_edges),
                }
            )
            continue
        started = perf_counter()
        candidates = enumerate_consistent_dags(
            cpdag, max_undirected_edges=config.maximum_undirected_edges
        )
        enumeration_seconds = perf_counter() - started
        if not config.minimum_mec_size <= len(candidates) <= config.maximum_mec_size:
            failures.append(
                {
                    "stage": "source",
                    "reason": "mec_size_outside_bounds",
                    "seed": seed,
                    "mec_size": len(candidates),
                }
            )
            continue
        coefficients = sample_coefficients(
            dag,
            rng,
            config.coefficient_min_abs,
            config.coefficient_max_abs,
        )
        noise = rng.uniform(*config.noise_variance_bounds, size=n)
        scm = LinearGaussianSCM(dag, coefficients, noise_variances=noise)
        true_id = tuple(sorted((int(a), int(b)) for a, b in dag.edges))
        if true_id not in {candidate.id for candidate in candidates}:
            raise RuntimeError("true DAG is absent from its enumerated MEC")
        return (
            SourceProblem(
                source_seed=seed,
                n=n,
                graph_family=family,
                replicate=replicate,
                generation_attempt=attempt,
                scm=scm,
                cpdag=cpdag,
                candidates=candidates,
                mec_enumeration_seconds=enumeration_seconds,
            ),
            failures,
        )
    return None, failures


def _legal_intervention(
    scm: LinearGaussianSCM, budget: int, rng: np.random.Generator
) -> ShiftScaleIntervention:
    n = scm.dag.number_of_nodes()
    shift = np.zeros(n)
    scale = np.ones(n)
    shift_node = int(rng.integers(n))
    shift[shift_node] = float(rng.uniform(-1.25, 1.25))
    if budget >= 2 and scm.scale_eligible_nodes:
        scale_node = int(rng.choice(sorted(scm.scale_eligible_nodes)))
        scale[scale_node] = float(np.exp(rng.uniform(math.log(0.7), math.log(1.45))))
    return ShiftScaleIntervention(shift, scale)


def _target(
    source: SourceProblem, regime: str, budget: int
) -> tuple[GaussianTarget, int]:
    seed = _stable_seed(source.source_seed, "target", regime)
    rng = np.random.default_rng(seed)
    intervention = _legal_intervention(source.scm, budget, rng)
    if regime == "realisable":
        return generate_realisable_target(source.scm, intervention), seed
    noise_covariance = np.diag(np.asarray(source.scm.noise_variances, dtype=float))
    left, right = rng.choice(source.n, size=2, replace=False)
    correlation = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.12, 0.28))
    covariance = correlation * math.sqrt(
        noise_covariance[left, left] * noise_covariance[right, right]
    )
    noise_covariance[left, right] = covariance
    noise_covariance[right, left] = covariance
    return correlated_noise_target(source.scm, noise_covariance, intervention), seed


def minimum_vertex_cover_size(cpdag: Any) -> int:
    """Compute the exact cover number of the initial undirected ambiguity graph."""
    edges = tuple(cpdag.undirected_edges)
    if not edges:
        return 0
    incident_nodes = tuple(sorted({node for edge in edges for node in edge}))
    for size in range(len(incident_nodes) + 1):
        for selected in combinations(incident_nodes, size):
            cover = frozenset(selected)
            if all(left in cover or right in cover for left, right in edges):
                return size
    raise RuntimeError("finite graph has no vertex cover")


def _audit_partitions(result: BellmanResult) -> int:
    checked = 0
    for state in result.states.values():
        for node, cells in state.partitions.items():
            if len(cells) < 2 or any(not cell for cell in cells):
                raise AssertionError(
                    "informative partition has fewer than two nonempty cells"
                )
            union = frozenset().union(*cells)
            if union != state.model_ids:
                raise AssertionError("partition union differs from candidate state")
            for left, right in combinations(cells, 2):
                if left & right:
                    raise AssertionError("query partition cells overlap")
            if node not in state.worst_children:
                raise AssertionError("partition lacks worst-child record")
            checked += 1
    return checked


def _terminal_survivors(result: BellmanResult) -> int:
    """Follow a deterministic optimal policy and its largest worst-value cell."""
    state = result.initial_state
    while state.terminal_reason is None:
        node = state.optimal_queries[0]
        children = state.worst_children[node]
        child = max(children, key=lambda ids: (len(ids), tuple(sorted(ids))))
        state = result.states[child]
    return len(state.model_ids)


def _unresolved_tie_count(action_sets: Mapping[DagId, OptimalActionSet]) -> int:
    """Count within-mode distinct ties not represented as separate local states."""
    count = 0
    for action_set in action_sets.values():
        result = action_set.oracle_result
        atol = float(result.tolerances["loss_atol"])
        rtol = float(result.tolerances["loss_rtol"])
        for states in result.node_local_candidate_states:
            for state in states:
                if not state.numerically_tied or not state.joint_candidates:
                    continue
                tolerance = atol + rtol * max(1.0, abs(state.raw_local_loss))
                distinct = {
                    (candidate.shift, candidate.scale)
                    for candidate in state.joint_candidates
                    if abs(candidate.raw_local_loss - state.raw_local_loss) <= tolerance
                    and not np.allclose(
                        (candidate.shift, candidate.scale),
                        (state.shift, state.scale),
                        atol=1e-10,
                        rtol=1e-9,
                    )
                }
                count += len(distinct)
    return count


def _run_algorithm1(
    source: SourceProblem,
    target: GaussianTarget,
    config: CampaignConfig,
    policy: Any,
) -> tuple[int, float, tuple[int, ...]]:
    environment = ExactMomentBlackBoxEnvironment(
        source.scm,
        instance_id=f"small-mec:{source.source_seed}:{target.public_metadata.get('target_family', 'realisable')}:{policy.name}",
        ledger=EnvironmentQueryLedger(None),
    )
    source_mean, source_covariance = source.scm.observational_moments()
    started = perf_counter()
    result = run_query_method(
        method="public_adaptive_local_mechanism_dp",
        environment=environment,
        source_mean=source_mean,
        source_covariance=source_covariance,
        target_mean=target.mean,
        target_covariance=target.covariance,
        budget_B=config.budget,
        experimental_sparsity_S=1,
        shift_bounds=config.shift_bounds,
        scale_bounds=tuple(math.exp(value) for value in config.log_scale_bounds),
        initial_graph=source.cpdag,
        singleton_shift_policy=policy,
        loss_atol=config.loss_atol,
        loss_rtol=config.loss_rtol,
    )
    runtime = perf_counter() - started
    if result.counts.singleton_shift_queries != result.counts.total_query_count:
        raise AssertionError("Algorithm 1 physical count is not singleton-query count")
    return result.counts.singleton_shift_queries, runtime, result.queried_nodes


def solve_instance(
    source: SourceProblem, regime: str, config: CampaignConfig
) -> dict[str, object]:
    total_started = perf_counter()
    target, target_seed = _target(source, regime, config.budget)
    source_mean, source_covariance = source.scm.observational_moments()
    candidates = candidate_map(source.candidates)
    scms = reconstruct_all_candidate_scms(
        source.candidates, source_mean, source_covariance
    )
    signature_audit = audit_singleton_query_signatures(scms)
    if not signature_audit.passed:
        raise RuntimeError(
            f"response-signature audit failed: {signature_audit.failures}"
        )

    action_started = perf_counter()
    action_sets = compute_all_optimal_action_sets(
        scms,
        target_mean=target.mean,
        target_covariance=target.covariance,
        budget=config.budget,
        shift_bounds=config.shift_bounds,
        log_scale_bounds=config.log_scale_bounds,
        loss_atol=config.loss_atol,
        loss_rtol=config.loss_rtol,
        action_atol=config.action_atol,
        action_rtol=config.action_rtol,
    )
    action_seconds = perf_counter() - action_started
    diagnostic_messages = [
        message
        for action_set in action_sets.values()
        for message in action_set.oracle_result.degeneracy_diagnostics
    ]
    # Every DAG root has a zero parental predictor, so scale is a canonical
    # no-op there rather than an uncertified continuous action degeneracy.
    canonical_root_degeneracies = sum(
        message.endswith(":zero_parental_predictor") for message in diagnostic_messages
    )
    degeneracies = len(diagnostic_messages) - canonical_root_degeneracies
    unresolved_ties = _unresolved_tie_count(action_sets)
    if degeneracies or unresolved_ties:
        raise ValueError(
            f"uncertified action geometry: degeneracies={degeneracies}, "
            f"unresolved_within_mode_ties={unresolved_ties}"
        )

    matching_started = perf_counter()
    matching = solve_matching_complexity(candidates=candidates, action_sets=action_sets)
    matching_seconds = perf_counter() - matching_started
    dag_started = perf_counter()
    dag = solve_dag_complexity(candidates=candidates)
    dag_seconds = perf_counter() - dag_started
    matching_partitions = _audit_partitions(matching)
    dag_partitions = _audit_partitions(dag)
    mec_size = len(candidates)
    if not 0 <= matching.complexity <= dag.complexity <= mec_size - 1:
        raise AssertionError("Bellman invariant failed")

    first_count, first_seconds, first_nodes = _run_algorithm1(
        source, target, config, FirstUnresolvedPolicy()
    )
    maxdeg_count, maxdeg_seconds, maxdeg_nodes = _run_algorithm1(
        source, target, config, MaxUndirectedDegreePolicy()
    )
    true_id = tuple(sorted((int(a), int(b)) for a, b in source.scm.dag.edges))
    unique_action_representatives = {
        action for action_set in action_sets.values() for action in action_set.actions
    }
    root_certificate = matching.initial_state.common_command_certificate
    multiple_optima = sum(len(item.actions) > 1 for item in action_sets.values())
    numerical_tie_messages = sum(
        len(item.numerical_diagnostics) for item in action_sets.values()
    )
    dag_savings = dag.complexity - matching.complexity
    first_savings = first_count - matching.complexity
    maxdeg_savings = maxdeg_count - matching.complexity
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": source.source_seed,
        "target_seed": target_seed,
        "n": source.n,
        "graph_family": source.graph_family,
        "replicate": source.replicate,
        "generation_attempt": source.generation_attempt,
        "number_of_edges": source.scm.dag.number_of_edges(),
        "cpdag_directed_edges": len(source.cpdag.directed_edges),
        "cpdag_undirected_edges": len(source.cpdag.undirected_edges),
        "mec_size": mec_size,
        "mec_size_bin": _mec_bin(mec_size),
        "budget": config.budget,
        "target_regime": regime,
        "target_is_realisable": target.is_realisable,
        "true_dag_identifier": _dag_id_text(true_id),
        "T_match_star": matching.complexity,
        "T_DAG_star": dag.complexity,
        "T_Alg1_first": first_count,
        "T_Alg1_maxdeg": maxdeg_count,
        "vertex_cover_bound": minimum_vertex_cover_size(source.cpdag),
        "strict_match_saving_vs_DAG": matching.complexity < dag.complexity,
        "strict_match_saving_vs_Alg1_first": matching.complexity < first_count,
        "strict_match_saving_vs_Alg1_maxdeg": matching.complexity < maxdeg_count,
        "DAG_absolute_savings": dag_savings,
        "DAG_relative_savings": dag_savings / dag.complexity if dag.complexity else 0.0,
        "Alg1_first_absolute_savings": first_savings,
        "Alg1_first_relative_savings": first_savings / first_count
        if first_count
        else 0.0,
        "Alg1_maxdeg_absolute_savings": maxdeg_savings,
        "Alg1_maxdeg_relative_savings": maxdeg_savings / maxdeg_count
        if maxdeg_count
        else 0.0,
        "matching_worst_path_terminal_surviving_DAGs": _terminal_survivors(matching),
        "matching_root_common_action_count": len(matching.initial_state.common_actions),
        "matching_root_common_command_shift": (
            json.dumps(root_certificate.command.shift) if root_certificate else ""
        ),
        "matching_root_common_command_scale": (
            json.dumps(root_certificate.command.scale) if root_certificate else ""
        ),
        "matching_root_commanded_support": (
            root_certificate.commanded_support if root_certificate else -1
        ),
        "matching_root_maximum_loss_residual": (
            root_certificate.maximum_loss_residual
            if root_certificate
            else float("nan")
        ),
        "distinct_candidate_optimal_action_representatives": len(
            unique_action_representatives
        ),
        "candidate_DAGs_with_multiple_optimal_actions": multiple_optima,
        "tie_diagnostic_message_count": numerical_tie_messages,
        "optimizer_degeneracy_count": degeneracies,
        "canonical_root_scale_noop_count": canonical_root_degeneracies,
        "unresolved_within_mode_tie_count": unresolved_ties,
        "action_intersection_method": "finite_local_mode_product_with_all_global_discrete_ties",
        "query_response_equivalence": "exact_parent_set_signature_audited_against_total_effect_response",
        "response_precision_identity_max_error": signature_audit.precision_identity_max_error,
        "response_same_signature_max_error": signature_audit.same_signature_max_error,
        "response_different_signature_min_separation": signature_audit.different_signature_min_separation,
        "matching_partition_count_audited": matching_partitions,
        "DAG_partition_count_audited": dag_partitions,
        "matching_bellman_states": len(matching.states),
        "DAG_bellman_states": len(dag.states),
        "Alg1_first_query_nodes": json.dumps(first_nodes),
        "Alg1_maxdeg_query_nodes": json.dumps(maxdeg_nodes),
        "runtime_mec_enumeration_seconds": source.mec_enumeration_seconds,
        "runtime_candidate_action_seconds": action_seconds,
        "runtime_matching_bellman_seconds": matching_seconds,
        "runtime_DAG_bellman_seconds": dag_seconds,
        "runtime_Alg1_first_seconds": first_seconds,
        "runtime_Alg1_maxdeg_seconds": maxdeg_seconds,
        "runtime_total_instance_seconds": perf_counter() - total_started,
    }


def _mec_bin(size: int) -> str:
    if size <= 4:
        return "2-4"
    if size <= 8:
        return "5-8"
    if size <= 16:
        return "9-16"
    return "17-32"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


COUNT_FIELDS = ("T_match_star", "T_DAG_star", "T_Alg1_first", "T_Alg1_maxdeg")
SAVING_FIELDS = (
    "DAG_absolute_savings",
    "Alg1_first_absolute_savings",
    "Alg1_maxdeg_absolute_savings",
)


def _stats(values: Iterable[float]) -> dict[str, float]:
    sequence = [float(value) for value in values]
    return {
        "mean": statistics.fmean(sequence),
        "median": statistics.median(sequence),
        "minimum": min(sequence),
        "maximum": max(sequence),
    }


def _group_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "instances": len(rows),
        "frequency_strict_savings_vs_DAG": statistics.fmean(
            bool(row["strict_match_saving_vs_DAG"]) for row in rows
        ),
        "frequency_strict_savings_vs_Alg1_first": statistics.fmean(
            bool(row["strict_match_saving_vs_Alg1_first"]) for row in rows
        ),
        "frequency_strict_savings_vs_Alg1_maxdeg": statistics.fmean(
            bool(row["strict_match_saving_vs_Alg1_maxdeg"]) for row in rows
        ),
        "query_counts": {
            field: _stats(float(row[field]) for row in rows) for field in COUNT_FIELDS
        },
        "savings": {
            field: _stats(float(row[field]) for row in rows) for field in SAVING_FIELDS
        },
    }


def aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot aggregate an empty campaign")
    strata: dict[str, dict[str, object]] = {}
    for field in ("n", "graph_family", "mec_size_bin", "target_regime"):
        grouped: dict[object, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[row[field]].append(row)
        strata[field] = {
            str(key): _group_summary(group)
            for key, group in sorted(grouped.items(), key=lambda item: str(item[0]))
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "overall": _group_summary(rows),
        "stratified": strata,
        "mec_size_distribution": dict(
            sorted(Counter(int(row["mec_size"]) for row in rows).items())
        ),
        "validation": {
            "all_matching_leq_DAG": all(
                int(row["T_match_star"]) <= int(row["T_DAG_star"]) for row in rows
            ),
            "all_bellman_bounds_hold": all(
                0
                <= int(row["T_match_star"])
                <= int(row["T_DAG_star"])
                <= int(row["mec_size"]) - 1
                for row in rows
            ),
            "uncertified_action_instances": sum(
                bool(row["optimizer_degeneracy_count"])
                or bool(row["unresolved_within_mode_tie_count"])
                for row in rows
            ),
        },
    }


def _aggregate_rows(summary: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    groups = [("overall", "all", summary["overall"])]
    for field, values in summary["stratified"].items():
        groups.extend((field, label, group) for label, group in values.items())
    for stratum, label, group in groups:
        row: dict[str, object] = {
            "stratum": stratum,
            "value": label,
            "instances": group["instances"],
            "strict_vs_DAG": group["frequency_strict_savings_vs_DAG"],
            "strict_vs_Alg1_first": group["frequency_strict_savings_vs_Alg1_first"],
            "strict_vs_Alg1_maxdeg": group["frequency_strict_savings_vs_Alg1_maxdeg"],
        }
        for field in COUNT_FIELDS:
            row[f"mean_{field}"] = group["query_counts"][field]["mean"]
            row[f"median_{field}"] = group["query_counts"][field]["median"]
        for field in SAVING_FIELDS:
            for statistic in ("mean", "median", "maximum"):
                row[f"{statistic}_{field}"] = group["savings"][field][statistic]
        output.append(row)
    return output


def _plots(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "savefig.facecolor": "white",
        }
    )
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    bins = [
        label
        for label in ("2-4", "5-8", "9-16", "17-32")
        if any(row["mec_size_bin"] == label for row in rows)
    ]
    measures = (
        ("T_match_star", r"$T^*_{match}$"),
        ("T_DAG_star", r"$T^*_{DAG}$"),
        ("T_Alg1_first", "Algorithm 1"),
    )
    positions = np.arange(len(bins))
    width = 0.24
    figure, axis = plt.subplots(figsize=(6.2, 3.4))
    for index, (field, label) in enumerate(measures):
        means = [
            statistics.fmean(
                float(row[field]) for row in rows if row["mec_size_bin"] == bin_label
            )
            for bin_label in bins
        ]
        axis.bar(positions + (index - 1) * width, means, width=width, label=label)
    axis.set_xticks(positions, bins)
    axis.set_xlabel("observational MEC size")
    axis.set_ylabel("mean singleton queries")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            figures / f"main_query_comparison.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    savings = [int(row["DAG_absolute_savings"]) for row in rows]
    values = list(range(min(savings), max(savings) + 1))
    axes[0].hist(
        savings,
        bins=np.arange(min(values) - 0.5, max(values) + 1.5),
        color="#4C78A8",
        rwidth=0.85,
    )
    axes[0].set_xticks(values)
    axes[0].set_xlabel(r"$T^*_{DAG}-T^*_{match}$")
    axes[0].set_ylabel("instances")
    regimes = list(dict.fromkeys(str(row["target_regime"]) for row in rows))
    fractions = [
        statistics.fmean(
            bool(row["strict_match_saving_vs_DAG"])
            for row in rows
            if row["target_regime"] == regime
        )
        for regime in regimes
    ]
    axes[1].bar(regimes, fractions, color="#F58518")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("fraction with strict savings")
    axes[1].tick_params(axis="x", rotation=15)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            figures / f"matching_savings.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def _tables(
    rows: Sequence[Mapping[str, object]], summary: Mapping[str, object], output: Path
) -> None:
    table_rows = [
        row
        for row in _aggregate_rows(summary)
        if row["stratum"] in {"overall", "n", "target_regime", "mec_size_bin"}
    ]
    _write_csv(output / "tables" / "query_complexity_summary.csv", table_rows)
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Stratum & Value & $N$ & $T^*_{match}$ & $T^*_{DAG}$ & Alg. 1 \\",
        r"\midrule",
    ]
    for row in table_rows:
        lines.append(
            f"{row['stratum']} & {str(row['value']).replace('_', r'\_')} & {row['instances']} & "
            f"{float(row['mean_T_match_star']):.2f} & {float(row['mean_T_DAG_star']):.2f} & "
            f"{float(row['mean_T_Alg1_first']):.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path = output / "tables" / "query_complexity_summary.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _handoff(
    summary: Mapping[str, object], rows: Sequence[Mapping[str, object]], output: Path
) -> str:
    overall = summary["overall"]
    query = overall["query_counts"]
    savings = overall["savings"]
    return f"""# Small-MEC exact matching-query experiment handoff

## Status

The exact campaign completed with {len(rows)} valid source/target instances. All
instance-wise checks gave $0 \\le T^*_{{\\rm match}} \\le T^*_{{\\rm DAG}} \\le |\\mathcal M|-1$.
No uncertified continuous optimizer degeneracy entered the results.

## Proposed Methods paragraph

We enumerated the complete observational Markov-equivalence class for small
linear-Gaussian SCMs with $n\\in\\{{5,6,7\\}}$ from path, random rooted-tree,
and sparse Erdős--Rényi families. Each candidate DAG was equipped with the
unique conditional Gaussian mechanisms induced by the same observational
mean and covariance. Under budget $B=2$, we reused the exact bounded sparse
shift--scale dynamic program and retained every tied finite-mode global action.
Matching complexity used the terminal condition that candidate optimal-action
sets have nonempty intersection; DAG complexity used the singleton terminal
condition under the identical exact singleton-shift response partitions. We
also ran population Algorithm 1 with first-unresolved and maximum-undirected-
degree policies. Instances with an unresolved continuous degeneracy were
excluded and logged before aggregation.

## Proposed Results paragraph

Across {len(rows)} instances, matching-specific stopping used fewer queries than
exact DAG identification in {100 * overall["frequency_strict_savings_vs_DAG"]:.1f}%
of cases and fewer than Algorithm 1 (first-unresolved) in
{100 * overall["frequency_strict_savings_vs_Alg1_first"]:.1f}%. Mean query counts
were {query["T_match_star"]["mean"]:.2f} for $T^*_{{\\rm match}}$,
{query["T_DAG_star"]["mean"]:.2f} for $T^*_{{\\rm DAG}}$, and
{query["T_Alg1_first"]["mean"]:.2f} for Algorithm 1. The DAG-identification
saving had mean {savings["DAG_absolute_savings"]["mean"]:.2f}, median
{savings["DAG_absolute_savings"]["median"]:.2f}, and maximum
{savings["DAG_absolute_savings"]["maximum"]:.0f} singleton queries. These data
show whether the matching terminal condition yields savings on ordinary small
random problems; target-regime and MEC-size strata are reported in the table.

## Suggested caption

Exact singleton-shift query complexity on fully enumerated small observational
MECs. Bars show mean matching-specific minimax complexity, DAG-identification
minimax complexity under the same response partitions, and the physical query
count of the existing graph-aware Algorithm 1. The companion panel reports the
distribution and frequency of strict matching-specific savings.

## Suggested insertion

Insert the Methods paragraph at the end of the experimental-protocol subsection,
the Results paragraph after the theoretical general-MEC comparison, and place
`figures/main_query_comparison.pdf` with the suggested caption nearby. Use
`tables/query_complexity_summary.tex` as the compact supplementary table.

## Action equality and diagnostics

The action intersection is not inferred from equal KL values. For every candidate
DAG, the production optimizer's finite local modes are enumerated under the
sparse coordinate budget and all globally tied full actions are retained. Full
shift/scale vectors are compared at recorded strict tolerances. Distinct within-
mode numerical ties or continuous degeneracies that this representation cannot
certify cause fail-loud instance rejection. Root scale no-ops are canonicalized
to one. See `machine_readable/failures.json` and the raw tie columns.
"""


def run_campaign(
    config: CampaignConfig, output: Path, *, allow_existing: bool = False
) -> dict[str, object]:
    config.validate()
    raw_path = output / "machine_readable" / "instances.csv"
    if raw_path.exists() and not allow_existing:
        raise FileExistsError(f"refusing to overwrite existing campaign: {raw_path}")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", asdict(config))
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    campaign_started = perf_counter()
    for n in config.node_counts:
        for family in config.graph_families:
            for replicate in range(config.replicates_per_cell):
                source, rejected = _source_problem(config, family, n, replicate)
                failures.extend(
                    {"n": n, "graph_family": family, "replicate": replicate, **item}
                    for item in rejected
                )
                if source is None:
                    failures.append(
                        {
                            "stage": "source",
                            "reason": "attempt_limit",
                            "n": n,
                            "graph_family": family,
                            "replicate": replicate,
                        }
                    )
                    continue
                for regime in config.target_regimes:
                    try:
                        rows.append(solve_instance(source, regime, config))
                    except ValueError as error:
                        failures.append(
                            {
                                "stage": "target_instance",
                                "reason": type(error).__name__,
                                "message": str(error),
                                "seed": source.source_seed,
                                "n": n,
                                "graph_family": family,
                                "replicate": replicate,
                                "target_regime": regime,
                            }
                        )
    if not rows:
        raise RuntimeError("campaign produced no valid instances")
    summary = aggregate(rows)
    summary["campaign_runtime_seconds"] = perf_counter() - campaign_started
    summary["requested_source_problems"] = (
        len(config.node_counts)
        * len(config.graph_families)
        * config.replicates_per_cell
    )
    summary["valid_source_target_instances"] = len(rows)
    summary["failure_records"] = len(failures)
    _write_csv(raw_path, rows)
    _write_json(output / "machine_readable" / "instances.json", rows)
    _write_json(output / "machine_readable" / "failures.json", failures)
    _write_json(output / "summary.json", summary)
    _write_csv(output / "aggregate_summary.csv", _aggregate_rows(summary))
    _tables(rows, summary, output)
    _plots(rows, output)
    handoff = _handoff(summary, rows, output)
    handoff_path = output / "small_mec_matching_query_handoff.md"
    handoff_path.write_text(handoff, encoding="utf-8")
    docs_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "small_mec_matching_query_handoff.md"
    )
    if output.resolve() == DEFAULT_OUTPUT.resolve():
        docs_path.write_text(handoff, encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args(argv)
    config = CampaignConfig.from_json(args.config)
    summary = run_campaign(config, args.output, allow_existing=args.allow_existing)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
