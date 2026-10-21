"""Separate stopping and query-selection effects on the frozen small-MEC bank.

This diagnostic is deliberately artifact-only.  It reads the corrected Table 2
audit, replays its three target-independent full-DAG query schedules for every
candidate DAG, and truncates each schedule at its first already-certified
matching terminal.  It never regenerates a source, target, candidate DAG,
partition, optimizer result, or Algorithm 1 trajectory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts._output_safety import prepare_output_directory


SCHEMA_VERSION = "small_mec_stopping_decomposition_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT / "outputs" / "small_mec_command_space_paired_v2" / "corrected_audit"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "small_mec_stopping_decomposition"
FIXED_POLICIES = (
    "algorithm1_first_unresolved",
    "algorithm1_max_undirected_degree",
)
DAG_POLICY = "dag_optimal_min_node"
POLICIES = (*FIXED_POLICIES, DAG_POLICY)
REFERENCE_MEANS = {
    "T_match_star_worst_case": 1.3388888888888888,
    "T_DAG_star_worst_case": 1.8666666666666667,
    "Alg1_first_worst_case": 2.9444444444444446,
    "Alg1_maxdeg_worst_case": 2.077777777777778,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_id(indices: Iterable[int]) -> str:
    return ",".join(str(index) for index in sorted(indices))


def _json_compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _certificate_valid(
    certificate: Mapping[str, Any], surviving_dags: Sequence[str], budget: int
) -> bool:
    losses = certificate.get("candidate_losses", [])
    loss_ids = sorted(str(item["candidate_dag_identifier"]) for item in losses)
    if loss_ids != sorted(surviving_dags):
        return False
    if int(certificate["commanded_support"]) > budget:
        return False
    for item in losses:
        direct = float(item["direct_kl"])
        optimal = float(item["optimal_kl"])
        residual = float(item["absolute_residual"])
        tolerance = float(item["tolerance"])
        if abs(abs(direct - optimal) - residual) > 1e-14:
            return False
        if residual > tolerance:
            return False
    maximum = max((float(item["absolute_residual"]) for item in losses), default=0.0)
    return abs(maximum - float(certificate["maximum_loss_residual"])) <= 1e-14


def _first_matching_terminal(
    state_ids: Sequence[str],
    matching_certificates: Mapping[str, Mapping[str, Any]],
) -> tuple[int, str, Mapping[str, Any]]:
    """Return path depth, state ID, and certificate for the first terminal."""
    for depth, state in enumerate(state_ids):
        certificate = matching_certificates.get(state)
        if certificate is not None:
            return depth, state, certificate
    raise AssertionError("full-DAG trajectory never reached a certified matching terminal")


def _fixed_full_path(
    path: Mapping[str, Any], index_by_dag: Mapping[str, int]
) -> tuple[list[int], list[str], list[list[str]]]:
    full_nodes = [int(node) for node in path["queried_nodes"]]
    states = [_state_id(index_by_dag.values())]
    candidates = [sorted(index_by_dag)]
    for step in path["steps"]:
        cell = sorted(str(item) for item in step["observed_cell_dag_identifiers"])
        states.append(_state_id(index_by_dag[item] for item in cell))
        candidates.append(cell)
    if len(full_nodes) + 1 != len(states):
        raise AssertionError("fixed-policy state and query lengths disagree")
    if int(path["realized_count"]) != len(full_nodes):
        raise AssertionError("archived fixed-policy realized count disagrees with path")
    return full_nodes, states, candidates


def _dag_optimal_full_path(
    *,
    root_state: str,
    true_dag: str,
    dag_states: Mapping[str, Mapping[str, Any]],
) -> tuple[list[int], list[str], list[list[str]]]:
    nodes: list[int] = []
    states = [root_state]
    candidates: list[list[str]] = []
    state = root_state
    while True:
        row = dag_states[state]
        current = sorted(str(item) for item in row["candidate_dag_identifiers"])
        candidates.append(current)
        if len(current) == 1:
            break
        selected = int(row["selected_query"])
        if selected != min(int(item) for item in row["optimal_queries"]):
            raise AssertionError("DAG Bellman tie-break is not minimum optimal node")
        partition = next(
            item
            for item in row["query_partitions"]
            if int(item["query_node"]) == selected
        )
        cell = next(
            item
            for item in partition["response_cells"]
            if true_dag in item["candidate_dag_identifiers"]
        )
        nodes.append(selected)
        state = str(cell["child_state_id"])
        states.append(state)
    return nodes, states, candidates


def _path_row(
    *,
    instance: Mapping[str, str],
    policy: str,
    candidate_index: int,
    true_dag: str,
    sampled_true: bool,
    full_nodes: Sequence[int],
    state_ids: Sequence[str],
    state_candidates: Sequence[Sequence[str]],
    matching_certificates: Mapping[str, Mapping[str, Any]],
    full_path_reference: str,
) -> dict[str, object]:
    q_match, terminal_state, certificate = _first_matching_terminal(
        state_ids, matching_certificates
    )
    q_full = len(full_nodes)
    matching_nodes = list(full_nodes[:q_match])
    terminal_candidates = list(state_candidates[q_match])
    budget = int(instance["budget"])
    valid = _certificate_valid(certificate, terminal_candidates, budget)
    if not valid:
        raise AssertionError("serialized direct-KL terminal certificate failed validation")
    prefix = matching_nodes == list(full_nodes[: len(matching_nodes)])
    if not prefix or q_match > q_full:
        raise AssertionError("matching path is not a prefix of the full-DAG path")
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance["instance_id"],
        "source_id": f"n{instance['n']}:{instance['graph_family']}:r{instance['replicate']}",
        "source_seed": instance["source_seed"],
        "target_seed": instance["target_seed"],
        "n": int(instance["n"]),
        "graph_family": instance["graph_family"],
        "replicate": int(instance["replicate"]),
        "target_regime": instance["target_regime"],
        "policy": policy,
        "candidate_index": candidate_index,
        "true_candidate_dag_identifier": true_dag,
        "is_sampled_true_dag": sampled_true,
        "q_full": q_full,
        "q_match": q_match,
        "pathwise_reduction": q_full - q_match,
        "full_query_nodes": _json_compact(list(full_nodes)),
        "matching_query_nodes": _json_compact(matching_nodes),
        "matching_is_exact_prefix": prefix,
        "matching_terminal_state_id": terminal_state,
        "matching_terminal_candidate_count": len(terminal_candidates),
        "matching_terminal_candidate_dag_identifiers": _json_compact(terminal_candidates),
        "matching_terminal_commanded_support": int(certificate["commanded_support"]),
        "matching_terminal_maximum_loss_residual": float(
            certificate["maximum_loss_residual"]
        ),
        "matching_terminal_certificate_valid": valid,
        "matching_terminal_certificate_reference": (
            "certificates.jsonl#instance_id="
            f"{instance['instance_id']}&state_id={terminal_state}"
        ),
        "full_terminal_state_id": state_ids[-1],
        "full_terminal_candidate_count": len(state_candidates[-1]),
        "full_terminal_is_singleton": len(state_candidates[-1]) == 1,
        "full_path_reference": full_path_reference,
    }


def _distribution(values: Iterable[int]) -> str:
    counts = Counter(int(value) for value in values)
    return _json_compact({str(key): counts[key] for key in sorted(counts)})


def _summary_row(
    scope: str,
    stratum: str,
    policy: str,
    rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    def mean(field: str) -> float:
        return statistics.fmean(float(row[field]) for row in rows)

    candidate_iids = {str(row["instance_id"]) for row in rows}
    paths = [
        row
        for row in candidate_rows
        if row["policy"] == policy and row["instance_id"] in candidate_iids
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "stratum": stratum,
        "policy": policy,
        "instance_count": len(rows),
        "source_unit_count": len({str(row["source_seed"]) for row in rows}),
        "candidate_path_count": len(paths),
        "mean_W_full": mean("W_full"),
        "mean_W_match": mean("W_match"),
        "mean_T_match_star": mean("T_match_star"),
        "mean_T_DAG_star": mean("T_DAG_star"),
        "mean_total_gap": mean("total_gap"),
        "mean_stopping_component": mean("stopping_component"),
        "mean_selection_component": mean("selection_component"),
        "strictly_positive_stopping_count": sum(
            int(row["stopping_component"]) > 0 for row in rows
        ),
        "strictly_positive_selection_count": sum(
            int(row["selection_component"]) > 0 for row in rows
        ),
        "zero_stopping_count": sum(int(row["stopping_component"]) == 0 for row in rows),
        "zero_selection_count": sum(int(row["selection_component"]) == 0 for row in rows),
        "W_full_distribution": _distribution(int(row["W_full"]) for row in rows),
        "W_match_distribution": _distribution(int(row["W_match"]) for row in rows),
        "total_gap_distribution": _distribution(int(row["total_gap"]) for row in rows),
        "stopping_component_distribution": _distribution(
            int(row["stopping_component"]) for row in rows
        ),
        "selection_component_distribution": _distribution(
            int(row["selection_component"]) for row in rows
        ),
        "candidate_pathwise_reduction_distribution": _distribution(
            int(row["pathwise_reduction"]) for row in paths
        ),
    }


def _report(
    *,
    input_dir: Path,
    output_dir: Path,
    summary_rows: Sequence[Mapping[str, object]],
    validation: Mapping[str, Any],
    fingerprints: Mapping[str, str],
) -> str:
    overall = {
        str(row["policy"]): row
        for row in summary_rows
        if row["scope"] == "overall"
    }
    labels = {
        "algorithm1_first_unresolved": "Algorithm 1: first unresolved",
        "algorithm1_max_undirected_degree": "Algorithm 1: max undirected degree",
        "dag_optimal_min_node": "DAG-optimal Bellman (minimum-node tie-break)",
    }
    lines = [
        "# Matching-aware stopping decomposition on the frozen small-MEC bank",
        "",
        "## Scope and implementation",
        "",
        "This is an artifact-only population diagnostic. It reads the corrected paired "
        "audit and does not regenerate instances, rerun optimizers, alter production "
        "algorithms, or modify archived catalogs. The implementation is "
        "`research/small_mec_matching_query/stopping_decomposition.py`.",
        "",
        f"Input audit: `{input_dir}`. Output directory: `{output_dir}`.",
        "",
        "The two Algorithm 1 schedules are the archived population first-unresolved and "
        "maximum-undirected-degree trajectories, including their existing public-graph, "
        "mechanism-update, and Meek-orientation semantics. The third schedule follows the "
        "archived DAG-identification Bellman policy, choosing the minimum node index among "
        "all Bellman-optimal queries. That tie-break is fixed across targets and both "
        "stopping criteria.",
        "",
        "Matching stopping truncates a full-DAG schedule immediately upon reaching an "
        "archived matching-terminal state. These terminals use simultaneous intersections "
        "of complete physical-command families, one global commanded-support budget, "
        "bounded free root scales, and independent Gaussian KL certificates under every "
        "surviving DAG. No canonical-vector or pairwise-intersection test is used here.",
        "",
        "Implementation and evidence references:",
        "",
        "- `research/matching_query_complexity.py`: corrected physical-command family "
        "intersection and independent Gaussian KL certification.",
        "- `research/small_mec_matching_query/table2_audit.py`: exhaustive corrected "
        "Table 2 audit and population-policy replay.",
        "- `../frozen_inputs/instances.jsonl`: frozen source/target moments and candidate "
        "DAG bank (path is relative to the corrected-audit input directory).",
        "- `candidate_dags.jsonl`, `certificates.jsonl`, and "
        "`traces/bellman_state_catalog.jsonl`: optimal actions, terminal witnesses, "
        "response partitions, and matching/DAG Bellman states.",
        "- `traces/fixed_policy_candidate_paths.jsonl`: exhaustive archived Algorithm 1 "
        "paths under both graph-only selection rules.",
        "",
        "## Frozen-bank identity",
        "",
        f"The retained bank contains {validation['instance_count']} distinct instances "
        f"from {validation['source_count']} source units, with exactly two targets per "
        "source and all candidate-DAG paths retained. Reference means were reproduced: "
        f"matching minimax {validation['reference_means']['T_match_star_worst_case']:.10f}, "
        f"DAG minimax {validation['reference_means']['T_DAG_star_worst_case']:.10f}, "
        f"first-unresolved worst case {validation['reference_means']['Alg1_first_worst_case']:.10f}, "
        f"and max-degree worst case {validation['reference_means']['Alg1_maxdeg_worst_case']:.10f}.",
        "",
        "Input SHA-256 fingerprints:",
        "",
    ]
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in fingerprints.items())
    lines.extend(
        [
            "",
            "## Aggregate findings",
            "",
            "| Fixed query schedule | Mean full | Mean matching-stop | Mean total gap | Mean stopping | Mean selection | Positive stopping | Positive selection |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy in POLICIES:
        row = overall[policy]
        lines.append(
            f"| {labels[policy]} | {float(row['mean_W_full']):.4f} | "
            f"{float(row['mean_W_match']):.4f} | {float(row['mean_total_gap']):.4f} | "
            f"{float(row['mean_stopping_component']):.4f} | "
            f"{float(row['mean_selection_component']):.4f} | "
            f"{row['strictly_positive_stopping_count']}/{row['instance_count']} | "
            f"{row['strictly_positive_selection_count']}/{row['instance_count']} |"
        )
    lines.extend(
        [
            "",
            "For each instance and fixed schedule, `total gap = W_full - T_match*`, "
            "`stopping = W_full - W_match`, and `selection = W_match - T_match*`. "
            "All quantities are integer worst-case query counts. The exact identity held "
            "for every one of the 540 policy-instance rows, and both components were "
            "nonnegative.",
            "",
            "Descriptive graph-family, dimension, and target-regime breakdowns are stored "
            "as additional rows in `summary.csv`; they use the same frozen bank and are "
            "not new experiments.",
            "",
            "## Validation",
            "",
        ]
    )
    check_labels = {
        "all_candidate_paths_covered": "All candidate-DAG paths covered for all three schedules",
        "frozen_moment_bank_identity_matched": "Frozen moments, identifiers, and candidate DAGs match the corrected audit",
        "full_costs_reproduced": "Archived fixed-policy full costs and DAG Bellman values reproduced",
        "all_matching_paths_are_prefixes": "Every matching-stopped trajectory is an exact full-path prefix",
        "all_matching_terminals_certified": "Every used matching terminal has a feasible common command and valid direct-KL certificate",
        "all_order_inequalities_hold": "W_match <= W_full and T_match* <= W_match everywhere",
        "all_decompositions_exact": "Every integer decomposition identity holds exactly",
        "dag_tie_break_valid": "DAG policy always uses the minimum Bellman-optimal node",
        "query_schedules_target_independent": "Full query schedules agree across paired targets",
        "upstream_corrected_audit_passed": "All top-level corrected-audit checks pass",
    }
    lines.extend(
        f"- {label}: **{'PASS' if validation[key] else 'FAIL'}**"
        for key, label in check_labels.items()
    )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The stopping component changes only the terminal rule along a fixed query "
            "schedule. The selection component is the remaining excess worst-case cost "
            "above optimal matching acquisition. These are not pathwise causal effects: "
            "the DAG attaining `W_full` can differ from the DAG attaining `W_match`. "
            "Candidate-level pathwise reductions are therefore exported separately in "
            "`candidate_path_results.csv`, including all zero-saving cases.",
            "",
            "No uncertainty intervals or significance tests are reported; the requested "
            "descriptive frozen-bank results are complete. Any later bootstrap must "
            "resample the 90 source units with their two targets kept together.",
            "",
            "## Manuscript-ready factual summary",
            "",
            _manuscript_paragraph(overall),
            "",
        ]
    )
    return "\n".join(lines)


def _manuscript_paragraph(overall: Mapping[str, Mapping[str, object]]) -> str:
    first = overall["algorithm1_first_unresolved"]
    maxdeg = overall["algorithm1_max_undirected_degree"]
    dag = overall["dag_optimal_min_node"]
    return (
        "On the frozen corrected bank of 180 source-target instances, applying "
        "matching-aware stopping to the same full-DAG query schedules reduced mean "
        f"worst-case queries from {float(first['mean_W_full']):.3f} to "
        f"{float(first['mean_W_match']):.3f} for first-unresolved Algorithm 1, from "
        f"{float(maxdeg['mean_W_full']):.3f} to {float(maxdeg['mean_W_match']):.3f} "
        "for maximum-undirected-degree Algorithm 1, and from "
        f"{float(dag['mean_W_full']):.3f} to {float(dag['mean_W_match']):.3f} along "
        "the DAG-optimal Bellman schedule. Relative to the matching-optimal mean of "
        f"{float(first['mean_T_match_star']):.3f}, the corresponding mean stopping/"
        f"selection components were {float(first['mean_stopping_component']):.3f}/"
        f"{float(first['mean_selection_component']):.3f}, "
        f"{float(maxdeg['mean_stopping_component']):.3f}/"
        f"{float(maxdeg['mean_selection_component']):.3f}, and "
        f"{float(dag['mean_stopping_component']):.3f}/"
        f"{float(dag['mean_selection_component']):.3f}, respectively. These are "
        "worst-case decompositions, not pathwise causal effects, because the maxima "
        "under the two stopping rules need not be attained by the same candidate DAG."
    )


def run_diagnostic(
    input_dir: Path, output_dir: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    output_dir = prepare_output_directory(
        output_dir, overwrite=overwrite, protected=(input_dir,)
    )
    input_files = {
        "per_instance.csv": input_dir / "per_instance.csv",
        "candidate_dags.jsonl": input_dir / "candidate_dags.jsonl",
        "frozen_inputs/instances.jsonl": input_dir.parent
        / "frozen_inputs"
        / "instances.jsonl",
        "certificates.jsonl": input_dir / "certificates.jsonl",
        "fixed_policy_candidate_paths.jsonl": input_dir
        / "traces"
        / "fixed_policy_candidate_paths.jsonl",
        "bellman_state_catalog.jsonl": input_dir
        / "traces"
        / "bellman_state_catalog.jsonl",
        "checks.json": input_dir / "checks.json",
    }
    fingerprints_before = {name: _sha256(path) for name, path in input_files.items()}
    instances = _read_csv(input_files["per_instance.csv"])
    instance_by_id = {row["instance_id"]: row for row in instances}
    if len(instances) != len(instance_by_id):
        raise AssertionError("duplicate instance IDs in corrected audit")

    candidate_catalog: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(input_files["candidate_dags.jsonl"]):
        candidate_catalog[str(row["instance_id"])].append(row)
    for rows in candidate_catalog.values():
        rows.sort(key=lambda row: int(row["candidate_index"]))

    frozen_rows = _read_jsonl(input_files["frozen_inputs/instances.jsonl"])
    frozen_by_id = {str(row["instance_id"]): row for row in frozen_rows}
    frozen_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frozen_rows:
        frozen_sources[str(row["source_seed"])].append(row)
    frozen_identity_matched = (
        len(frozen_rows) == len(frozen_by_id) == 180
        and set(frozen_by_id) == set(instance_by_id)
        and all(
            str(frozen_by_id[iid]["source_seed"]) == instance["source_seed"]
            and str(frozen_by_id[iid]["target_seed"]) == instance["target_seed"]
            and int(frozen_by_id[iid]["mec_size"]) == int(instance["mec_size"])
            and [
                (
                    int(item["candidate_index"]),
                    str(item["candidate_dag_identifier"]),
                    item["parent_sets_by_node"],
                )
                for item in frozen_by_id[iid]["candidate_dags"]
            ]
            == [
                (
                    int(item["candidate_index"]),
                    str(item["candidate_dag_identifier"]),
                    item["parent_sets_by_node"],
                )
                for item in candidate_catalog[iid]
            ]
            for iid, instance in instance_by_id.items()
        )
        and all(
            len(rows) == 2
            and rows[0]["source_mean"] == rows[1]["source_mean"]
            and rows[0]["source_covariance"] == rows[1]["source_covariance"]
            for rows in frozen_sources.values()
        )
    )

    certificates: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in _read_jsonl(input_files["certificates.jsonl"]):
        certificates[str(row["instance_id"])][str(row["state_id"])] = row["certificate"]

    dag_states: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in _read_jsonl(input_files["bellman_state_catalog.jsonl"]):
        if row["objective"] == "dag":
            dag_states[str(row["instance_id"])][str(row["state_id"])] = row

    fixed_paths: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(input_files["fixed_policy_candidate_paths.jsonl"]):
        fixed_paths[(str(row["instance_id"]), str(row["policy"]))].append(row)
    for rows in fixed_paths.values():
        rows.sort(key=lambda row: int(row["candidate_index"]))

    candidate_results: list[dict[str, object]] = []
    dag_tie_break_valid = True
    for iid, instance in sorted(instance_by_id.items()):
        catalog = candidate_catalog[iid]
        index_by_dag = {
            str(row["candidate_dag_identifier"]): int(row["candidate_index"])
            for row in catalog
        }
        sampled = str(instance["sampled_true_dag_identifier"])
        matching = certificates[iid]
        root_state = _state_id(index_by_dag.values())

        for policy in FIXED_POLICIES:
            for archived in fixed_paths[(iid, policy)]:
                true_dag = str(archived["candidate_dag_identifier"])
                nodes, states, state_candidates = _fixed_full_path(archived, index_by_dag)
                candidate_results.append(
                    _path_row(
                        instance=instance,
                        policy=policy,
                        candidate_index=int(archived["candidate_index"]),
                        true_dag=true_dag,
                        sampled_true=bool(archived["is_sampled_true_dag"]),
                        full_nodes=nodes,
                        state_ids=states,
                        state_candidates=state_candidates,
                        matching_certificates=matching,
                        full_path_reference=(
                            "traces/fixed_policy_candidate_paths.jsonl#instance_id="
                            f"{iid}&policy={policy}&candidate_index={archived['candidate_index']}"
                        ),
                    )
                )

        for candidate in catalog:
            true_dag = str(candidate["candidate_dag_identifier"])
            try:
                nodes, states, state_candidates = _dag_optimal_full_path(
                    root_state=root_state,
                    true_dag=true_dag,
                    dag_states=dag_states[iid],
                )
            except AssertionError:
                dag_tie_break_valid = False
                raise
            candidate_results.append(
                _path_row(
                    instance=instance,
                    policy=DAG_POLICY,
                    candidate_index=int(candidate["candidate_index"]),
                    true_dag=true_dag,
                    sampled_true=true_dag == sampled,
                    full_nodes=nodes,
                    state_ids=states,
                    state_candidates=state_candidates,
                    matching_certificates=matching,
                    full_path_reference=(
                        "traces/bellman_state_catalog.jsonl#instance_id="
                        f"{iid}&objective=dag&root_state_id={root_state}"
                    ),
                )
            )

    paths_by_instance_policy: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in candidate_results:
        paths_by_instance_policy[(str(row["instance_id"]), str(row["policy"]))].append(row)

    per_instance_results: list[dict[str, object]] = []
    full_field = {
        "algorithm1_first_unresolved": "Alg1_first_worst_case",
        "algorithm1_max_undirected_degree": "Alg1_maxdeg_worst_case",
        DAG_POLICY: "T_DAG_star_worst_case",
    }
    for iid, instance in sorted(instance_by_id.items()):
        for policy in POLICIES:
            paths = paths_by_instance_policy[(iid, policy)]
            w_full = max(int(row["q_full"]) for row in paths)
            w_match = max(int(row["q_match"]) for row in paths)
            t_match = int(instance["T_match_star_worst_case"])
            t_dag = int(instance["T_DAG_star_worst_case"])
            total = w_full - t_match
            stopping = w_full - w_match
            selection = w_match - t_match
            per_instance_results.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "instance_id": iid,
                    "source_id": f"n{instance['n']}:{instance['graph_family']}:r{instance['replicate']}",
                    "source_seed": instance["source_seed"],
                    "target_seed": instance["target_seed"],
                    "n": int(instance["n"]),
                    "graph_family": instance["graph_family"],
                    "replicate": int(instance["replicate"]),
                    "target_regime": instance["target_regime"],
                    "mec_size": int(instance["mec_size"]),
                    "policy": policy,
                    "W_full": w_full,
                    "W_match": w_match,
                    "T_match_star": t_match,
                    "T_DAG_star": t_dag,
                    "total_gap": total,
                    "stopping_component": stopping,
                    "selection_component": selection,
                    "decomposition_exact": total == stopping + selection,
                    "W_full_attaining_candidate_dag_identifiers": _json_compact(
                        sorted(
                            str(row["true_candidate_dag_identifier"])
                            for row in paths
                            if int(row["q_full"]) == w_full
                        )
                    ),
                    "W_match_attaining_candidate_dag_identifiers": _json_compact(
                        sorted(
                            str(row["true_candidate_dag_identifier"])
                            for row in paths
                            if int(row["q_match"]) == w_match
                        )
                    ),
                    "same_candidate_attains_both_maxima": bool(
                        {
                            str(row["true_candidate_dag_identifier"])
                            for row in paths
                            if int(row["q_full"]) == w_full
                        }
                        & {
                            str(row["true_candidate_dag_identifier"])
                            for row in paths
                            if int(row["q_match"]) == w_match
                        }
                    ),
                    "archived_full_cost": int(instance[full_field[policy]]),
                    "full_cost_reproduced": w_full == int(instance[full_field[policy]]),
                }
            )

    summary_rows: list[dict[str, object]] = []
    groups: list[tuple[str, str, list[dict[str, object]]]] = [
        ("overall", "all", per_instance_results)
    ]
    for field, scope in (
        ("graph_family", "graph_family"),
        ("n", "dimension"),
        ("target_regime", "target_type"),
    ):
        values = sorted({str(row[field]) for row in per_instance_results})
        groups.extend(
            (
                scope,
                value,
                [row for row in per_instance_results if str(row[field]) == value],
            )
            for value in values
        )
    for scope, stratum, group_rows in groups:
        for policy in POLICIES:
            policy_rows = [row for row in group_rows if row["policy"] == policy]
            summary_rows.append(
                _summary_row(scope, stratum, policy, policy_rows, candidate_results)
            )

    means = {
        key: statistics.fmean(float(row[key]) for row in instances)
        for key in REFERENCE_MEANS
    }
    source_target_counts = Counter(str(row["source_seed"]) for row in instances)
    expected_paths = sum(int(row["mec_size"]) for row in instances)

    schedule_pairs: dict[tuple[str, str, int, str], dict[str, tuple[int, ...]]] = defaultdict(dict)
    for row in candidate_results:
        key = (
            str(row["source_seed"]),
            str(row["policy"]),
            int(row["candidate_index"]),
            str(row["true_candidate_dag_identifier"]),
        )
        schedule_pairs[key][str(row["target_regime"])] = tuple(
            json.loads(str(row["full_query_nodes"]))
        )
    target_independent = all(
        len(pair) == 2 and len(set(pair.values())) == 1 for pair in schedule_pairs.values()
    )

    upstream_checks = json.loads(input_files["checks.json"].read_text(encoding="utf-8"))
    upstream_top_level = {
        key: value
        for key, value in upstream_checks.items()
        if key.startswith("all_") and isinstance(value, bool)
    }
    validation: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "instance_count": len(instances),
        "source_count": len(source_target_counts),
        "exactly_two_targets_per_source": set(source_target_counts.values()) == {2},
        "candidate_paths_per_policy": {
            policy: sum(row["policy"] == policy for row in candidate_results)
            for policy in POLICIES
        },
        "expected_candidate_paths_per_policy": expected_paths,
        "reference_means": means,
        "reference_means_reproduced": all(
            abs(means[key] - expected) <= 1e-12
            for key, expected in REFERENCE_MEANS.items()
        ),
        "frozen_moment_bank_identity_matched": frozen_identity_matched,
        "all_candidate_paths_covered": all(
            sum(row["policy"] == policy for row in candidate_results) == expected_paths
            for policy in POLICIES
        ),
        "full_costs_reproduced": all(
            bool(row["full_cost_reproduced"]) for row in per_instance_results
        ),
        "all_matching_paths_are_prefixes": all(
            bool(row["matching_is_exact_prefix"]) for row in candidate_results
        ),
        "all_matching_terminals_certified": all(
            bool(row["matching_terminal_certificate_valid"])
            for row in candidate_results
        ),
        "all_full_terminals_are_singletons": all(
            bool(row["full_terminal_is_singleton"]) for row in candidate_results
        ),
        "all_order_inequalities_hold": all(
            int(row["W_match"]) <= int(row["W_full"])
            and int(row["T_match_star"]) <= int(row["W_match"])
            for row in per_instance_results
        ),
        "all_decompositions_exact": all(
            bool(row["decomposition_exact"]) for row in per_instance_results
        ),
        "all_components_nonnegative": all(
            int(row["stopping_component"]) >= 0
            and int(row["selection_component"]) >= 0
            for row in per_instance_results
        ),
        "dag_tie_break_valid": dag_tie_break_valid,
        "query_schedules_target_independent": target_independent,
        "upstream_corrected_audit_checks": upstream_top_level,
        "upstream_corrected_audit_passed": bool(upstream_top_level)
        and all(upstream_top_level.values()),
        "no_omissions_replacements_or_filtering": (
            len(instances) == 180
            and len(source_target_counts) == 90
            and set(source_target_counts.values()) == {2}
            and all(
                sum(row["policy"] == policy for row in candidate_results)
                == expected_paths
                for policy in POLICIES
            )
        ),
    }
    required_checks = [
        "exactly_two_targets_per_source",
        "reference_means_reproduced",
        "frozen_moment_bank_identity_matched",
        "all_candidate_paths_covered",
        "full_costs_reproduced",
        "all_matching_paths_are_prefixes",
        "all_matching_terminals_certified",
        "all_full_terminals_are_singletons",
        "all_order_inequalities_hold",
        "all_decompositions_exact",
        "all_components_nonnegative",
        "dag_tie_break_valid",
        "query_schedules_target_independent",
        "upstream_corrected_audit_passed",
        "no_omissions_replacements_or_filtering",
    ]
    if len(instances) != 180 or len(source_target_counts) != 90:
        raise AssertionError("frozen-bank size does not match the required 180/90 identity")
    if not all(bool(validation[key]) for key in required_checks):
        failures = [key for key in required_checks if not validation[key]]
        raise AssertionError(f"diagnostic validation failed: {failures}")

    _write_csv(output_dir / "per_instance_results.csv", per_instance_results)
    _write_csv(output_dir / "candidate_path_results.csv", candidate_results)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_json(output_dir / "validation.json", validation)
    report = _report(
        input_dir=input_dir,
        output_dir=output_dir,
        summary_rows=summary_rows,
        validation=validation,
        fingerprints=fingerprints_before,
    )
    (output_dir / "diagnostic_report.md").write_text(report, encoding="utf-8")
    overall = {
        str(row["policy"]): row
        for row in summary_rows
        if row["scope"] == "overall"
    }
    (output_dir / "manuscript_ready_summary.md").write_text(
        _manuscript_paragraph(overall) + "\n", encoding="utf-8"
    )

    fingerprints_after = {name: _sha256(path) for name, path in input_files.items()}
    if fingerprints_after != fingerprints_before:
        raise AssertionError("an archived input changed while the diagnostic was running")
    return {
        "validation": validation,
        "summary": overall,
        "output_dir": str(output_dir),
        "input_fingerprints": fingerprints_before,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_diagnostic(
        args.input.resolve(), args.output.resolve(), overwrite=args.overwrite
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
