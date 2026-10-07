"""Restart-safe paired ablation of the finite-sample local-evidence gate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from environments.blackbox import EnvironmentQueryLedger
from experiments.executors import build_production_runtime
from experiments.finite_sample_campaign import (
    BASE_SEEDS,
    COVARIANCE,
    _accepted_diagnostics,
    _algorithm_config,
    _coordinate_labels,
    _enrich_query_history,
    _exact_by_id,
    _graph_diagnostics,
    _kl,
    _sample_fingerprint,
    _stable_hash,
)
from experiments.finite_sample_campaign import campaign_specs as frozen_specs
from experiments.finite_sample_probe_ablation import (
    FROZEN_ROOT,
    KeyedFiniteSampleEnvironment,
    _checksum,
    _frozen_seed_overrides,
)
from experiments.nonoracle import FiniteStageAudit, run_finite_sample_nonoracle
from graphs.interventions import ShiftScaleIntervention
from graphs.pdag import PartiallyDirectedGraph
from graphs.problem_gen import PublicProblemView
from matching.confidence_calibrated_source import (
    ConfidenceCalibratedSourceSettings,
)
from matching.coordinate_search import _FixedMomentsProvider
from matching.empirical_moments import EmpiricalMomentsProvider
from matching.local_evidence import (
    ConfidenceCertificateLocalEvidenceProvider,
    PointEstimateLocalEvidenceProvider,
)
from matching.local_losses import CompositeGaussianLocalLoss
from matching.source_providers import ConfidenceCalibratedSourceInferenceProvider
from matching.structure_learning import SuppliedExactCPDAGProvider
from oracle.diagnostic_local_evidence_provider import (
    OracleLocalActionabilityProvider,
)
from oracle.diagnostic_source_provider import OracleLabelOverrideSourceInferenceProvider
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation

PROTOCOL_ID = "finite_sample_local_evidence_ablation_v1"
CAMPAIGN_ID = "finite_sample_local_evidence_p08_v1"
SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/finite_sample_local_evidence_ablation_v1"
PRIOR_PROBE_ROOT = REPOSITORY_ROOT / "outputs/finite_sample_probe_ablation_v1"
N_INTS = (100, 500, 2000)
POLICIES = (
    "confidence_certificate",
    "point_estimate_diagnostic",
    "oracle_local_actionability",
)
LocalPolicy = Literal[
    "confidence_certificate",
    "point_estimate_diagnostic",
    "oracle_local_actionability",
]
CALIBRATED_SETTINGS = ConfidenceCalibratedSourceSettings(0.1, 0.05, 0.95, 30)


@dataclass(frozen=True)
class LocalEvidenceAblationSpec:
    run_id: str
    policy_identifier: LocalPolicy
    base_seed: int
    paired_problem_id: str
    exact_run_id: str
    exact_result_path: str
    N_obs: int
    N_target: int
    N_int: int
    sample_seeds: dict[str, int]
    frozen_conservative_path: str
    frozen_conservative_checksum: str
    fingerprint: str


def _frozen_reference(old) -> Path:
    matches = []
    for path in (FROZEN_ROOT / "results").glob("*.json"):
        row = json.loads(path.read_text())
        if (
            row["base_seed"] == old.base_seed
            and row["metrics"]["N_obs"] == 2000
            and row["metrics"]["N_target"] == 2000
            and row["metrics"]["N_int"] == old.N_int
        ):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one frozen reference for {old.base_seed}/{old.N_int}"
        )
    return matches[0]


@lru_cache(maxsize=1)
def campaign_specs() -> tuple[LocalEvidenceAblationSpec, ...]:
    specs = []
    for old in frozen_specs():
        if old.N_obs != 2000 or old.N_target != 2000 or old.N_int not in N_INTS:
            continue
        frozen_path = _frozen_reference(old)
        for policy in POLICIES:
            scientific = {
                "campaign": CAMPAIGN_ID,
                "protocol": PROTOCOL_ID,
                "schema": SCHEMA_VERSION,
                "policy": policy,
                "source_policy": "oracle_probe_outcome",
                "base_seed": old.base_seed,
                "paired_problem_id": old.paired_problem_id,
                "exact_run_id": old.exact_run_id,
                "N_obs": 2000,
                "N_target": 2000,
                "N_int": old.N_int,
                "frozen_conservative_checksum": _checksum(frozen_path),
            }
            fingerprint = _stable_hash(scientific)
            specs.append(
                LocalEvidenceAblationSpec(
                    run_id=f"local-{policy[:8]}-{fingerprint[:18]}",
                    policy_identifier=policy,
                    base_seed=old.base_seed,
                    paired_problem_id=old.paired_problem_id,
                    exact_run_id=old.exact_run_id,
                    exact_result_path=old.exact_result_path,
                    N_obs=2000,
                    N_target=2000,
                    N_int=old.N_int,
                    sample_seeds=dict(old.sample_seeds),
                    frozen_conservative_path=str(frozen_path),
                    frozen_conservative_checksum=_checksum(frozen_path),
                    fingerprint=fingerprint,
                )
            )
    if len(specs) != 90 or len({item.run_id for item in specs}) != 90:
        raise RuntimeError("local-evidence ablation must contain 90 unique jobs")
    return tuple(sorted(specs, key=lambda item: item.run_id))


def _tree_checksums(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(REPOSITORY_ROOT)): _checksum(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def manifest_payload() -> dict[str, object]:
    frozen_manifest = FROZEN_ROOT / "manifest.json"
    prior_probe_manifest = PRIOR_PROBE_ROOT / "manifest.json"
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_identifier": CAMPAIGN_ID,
        "protocol_identifier": PROTOCOL_ID,
        "output_root": str(OUTPUT_ROOT),
        "policies": list(POLICIES),
        "source_policy": "oracle_probe_outcome_for_all_arms",
        "policy_scope": "binary_local_actionability_only",
        "N_obs": 2000,
        "N_target": 2000,
        "N_int": list(N_INTS),
        "base_seeds": list(BASE_SEEDS),
        "local_loss_threshold_rule": "actionable iff selected evidence exceeds tau",
        "bootstrap_evidence": "computed_and_recorded_identically_for_all_policies",
        "noninferential_diagnostic": True,
        "frozen_manifest": str(frozen_manifest),
        "frozen_manifest_checksum": _checksum(frozen_manifest),
        "prior_probe_manifest": str(prior_probe_manifest),
        "prior_probe_manifest_checksum": _checksum(prior_probe_manifest),
        "frozen_tree_checksums": _tree_checksums(FROZEN_ROOT),
        "prior_probe_tree_checksums": _tree_checksums(PRIOR_PROBE_ROOT),
        "runs": [asdict(item) for item in campaign_specs()],
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@lru_cache(maxsize=1)
def write_manifest() -> Path:
    path = OUTPUT_ROOT / "manifest.json"
    payload = json.loads(json.dumps(manifest_payload()))
    if path.exists() and json.loads(path.read_text()) != payload:
        artifacts = tuple((OUTPUT_ROOT / "results").glob("*.json"))
        if artifacts:
            raise RuntimeError("existing local-evidence manifest is incompatible")
    _atomic_json(path, payload)
    return path


def _provider(policy: LocalPolicy, problem):
    if policy == "confidence_certificate":
        return ConfidenceCertificateLocalEvidenceProvider()
    if policy == "point_estimate_diagnostic":
        return PointEstimateLocalEvidenceProvider()
    target_mean, target_covariance = problem.ground_truth.scm.interventional_moments(
        problem.ground_truth.generating_intervention
    )
    return OracleLocalActionabilityProvider(
        problem.ground_truth.scm.interventional_moments,
        target_mean,
        target_covariance,
    )


def _enrich_local_evidence(
    problem, public_problem, config, audit: FiniteStageAudit
) -> None:
    target_mean, target_covariance = problem.ground_truth.scm.interventional_moments(
        problem.ground_truth.generating_intervention
    )
    true_problem = replace(
        public_problem,
        target_mean=target_mean,
        target_covariance=target_covariance,
    )
    for event in audit.local_evidence_events:
        intervention = ShiftScaleIntervention(
            np.asarray(event["intervention_shift"]),
            np.asarray(event["intervention_scale"]),
        )
        graph = PartiallyDirectedGraph(
            frozenset(range(public_problem.n_nodes)),
            frozenset(tuple(edge) for edge in event["graph_directed_edges"]),
            frozenset(tuple(edge) for edge in event["graph_undirected_edges"]),
        )
        mean, covariance = problem.ground_truth.scm.interventional_moments(intervention)
        losses = CompositeGaussianLocalLoss(
            _FixedMomentsProvider(mean, covariance),
            config.local_weights,
            config.pairwise_neighborhood,
        ).node_losses(true_problem, intervention, graph)
        for decision, loss in zip(event["node_decisions"], losses, strict=True):
            prior = decision["true_local_loss"]
            if prior is not None and not np.isclose(prior, loss.composite):
                raise RuntimeError("oracle local decision and post-run truth disagree")
            decision["true_local_loss"] = loss.composite
            decision["true_loss_logging_scope"] = "post_run_evaluation"
            decision["point_actionable"] = (
                decision["point_local_loss"] > decision["configured_threshold"]
            )
            decision["confidence_actionable"] = (
                decision["confidence_interval"][0] > decision["configured_threshold"]
            )
            decision["oracle_actionable"] = (
                loss.composite > decision["configured_threshold"]
            )


def _actionability_metrics(audit: FiniteStageAudit) -> dict[str, object]:
    decisions = [
        decision
        for event in audit.local_evidence_events
        for decision in event["node_decisions"]
    ]
    confusion: dict[str, int] = {}
    agreements = 0
    for item in decisions:
        key = f"{bool(item['actionable'])}->{bool(item['oracle_actionable'])}"
        confusion[key] = confusion.get(key, 0) + 1
        agreements += int(item["actionable"] == item["oracle_actionable"])
    return {
        "local_evidence_evaluations": len(decisions),
        "point_actionable_count": sum(bool(x["point_actionable"]) for x in decisions),
        "confidence_actionable_count": sum(
            bool(x["confidence_actionable"]) for x in decisions
        ),
        "oracle_actionable_count": sum(bool(x["oracle_actionable"]) for x in decisions),
        "empirical_oracle_actionability_agreement": (
            agreements / len(decisions) if decisions else None
        ),
        "local_actionability_confusion_counts": dict(sorted(confusion.items())),
        "structural_certificates_obtained": sum(
            bool(x.get("structurally_actionable_source")) for x in decisions
        ),
    }


def _proposal_diagnostics(problem, result) -> list[dict[str, object]]:
    current = problem.public.current_intervention
    target_mean = problem.public.target_mean
    target_covariance = problem.public.target_covariance
    mean, covariance = problem.ground_truth.scm.interventional_moments(current)
    current_true = _kl(target_mean, target_covariance, mean, covariance)
    rows = []
    for record in result.history:
        if record.action != "confidence_aware_candidate_evaluation":
            continue
        candidate = ShiftScaleIntervention(
            np.asarray(record.candidate_shift), np.asarray(record.candidate_scale)
        )
        mean, covariance = problem.ground_truth.scm.interventional_moments(candidate)
        candidate_true = _kl(target_mean, target_covariance, mean, covariance)
        rows.append(
            {
                "step": record.step,
                "coordinate": f"{record.coordinate.kind}:{record.coordinate.node}",
                "decision": record.decision,
                "estimated_improvement": record.empirical_global_improvement,
                "true_improvement": current_true - candidate_true,
                "estimated_candidate_kl": record.global_loss_after,
                "true_candidate_kl": candidate_true,
                "decision_metadata": list(record.decision_metadata),
            }
        )
        if record.decision == "accepted":
            current = candidate
            current_true = candidate_true
    return rows


def execute(spec: LocalEvidenceAblationSpec) -> dict[str, object]:
    if spec.policy_identifier not in POLICIES:
        raise ValueError("execution requires an explicit local-evidence policy")
    exact = _exact_by_id(spec.exact_run_id)
    runtime = build_production_runtime(exact)
    problem = runtime.problem
    seeds = spec.sample_seeds
    source_samples = problem.ground_truth.scm.sample_interventional(
        problem.public.current_intervention,
        spec.N_obs,
        np.random.default_rng(seeds["source_observational"]),
    )
    target_samples = problem.ground_truth.scm.sample_interventional(
        problem.ground_truth.generating_intervention,
        spec.N_target,
        np.random.default_rng(seeds["target"]),
    )
    frozen = json.loads(Path(spec.frozen_conservative_path).read_text())
    if (
        _sample_fingerprint(source_samples)
        != frozen["metrics"]["source_sample_fingerprint"]
        or _sample_fingerprint(target_samples)
        != frozen["metrics"]["target_sample_fingerprint"]
    ):
        raise RuntimeError("source/target samples differ from frozen pairing")
    moments = EmpiricalMomentsProvider(COVARIANCE)
    source_estimate = moments.estimate(source_samples)
    target_estimate = moments.estimate(target_samples)
    public_problem = PublicProblemView(
        observational_samples=source_samples,
        target_mean=target_estimate.mean,
        target_covariance=target_estimate.regularized_covariance,
        current_intervention=problem.public.current_intervention,
        algorithm_seeds=problem.public.algorithm_seeds,
    )
    environment = KeyedFiniteSampleEnvironment(
        problem.ground_truth.scm,
        instance_id=spec.paired_problem_id,
        ledger=EnvironmentQueryLedger(None, None),
        seed_namespace=f"{CAMPAIGN_ID}:{spec.paired_problem_id}:nint-{spec.N_int}",
        overrides=_frozen_seed_overrides(spec),
    )
    structure = SuppliedExactCPDAGProvider(
        cpdag_from_true_dag_for_simulation(problem.ground_truth.scm.dag)
    )
    empirical_probe = ConfidenceCalibratedSourceInferenceProvider(CALIBRATED_SETTINGS)
    oracle_probe = OracleLabelOverrideSourceInferenceProvider(
        problem.ground_truth.scm.dag, empirical_probe
    )
    local_provider = _provider(spec.policy_identifier, problem)
    diagnostic_mode = (
        "exact_cpdag_exact_source_oracle_local_actionability_empirical_matching"
        if spec.policy_identifier == "oracle_local_actionability"
        else "exact_cpdag_exact_source_empirical_matching"
    )
    audit = FiniteStageAudit()
    config = _algorithm_config(
        exact, type("C", (), {"N_obs": 2000, "N_int": spec.N_int})()
    )
    started = perf_counter()
    result = run_finite_sample_nonoracle(
        public_problem,
        environment,
        structure,
        config,
        environment_seed=seeds["environment_root"],
        bootstrap_seed=seeds["bootstrap"],
        source_inference_provider=oracle_probe,
        local_evidence_provider=local_provider,
        diagnostic_mode=diagnostic_mode,
        result_mode="finite_sample_non_oracle_supplied_cpdag",
        stage_audit=audit,
    )
    runtime_seconds = perf_counter() - started
    _enrich_local_evidence(problem, public_problem, config, audit)
    proposal_diagnostics = _proposal_diagnostics(problem, result)
    initial_mean, initial_covariance = problem.current_moments()
    final_mean, final_covariance = problem.ground_truth.scm.interventional_moments(
        result.final_intervention
    )
    true_initial = _kl(
        problem.public.target_mean,
        problem.public.target_covariance,
        initial_mean,
        initial_covariance,
    )
    true_final = _kl(
        problem.public.target_mean,
        problem.public.target_covariance,
        final_mean,
        final_covariance,
    )
    labels = _coordinate_labels(result.final_intervention)
    query_history = _enrich_query_history(
        [asdict(x) for x in environment.query_history]
    )
    metrics = {
        "local_evidence_policy_identifier": spec.policy_identifier,
        "probe_policy_identifier": "oracle_probe_outcome",
        "N_obs": spec.N_obs,
        "N_target": spec.N_target,
        "N_int": spec.N_int,
        "local_loss_tolerance": config.local_loss_tolerance,
        "estimated_initial_reverse_kl": _kl(
            public_problem.target_mean,
            public_problem.target_covariance,
            source_estimate.mean,
            source_estimate.regularized_covariance,
        ),
        "true_initial_reverse_kl": true_initial,
        "estimated_final_reverse_kl": result.final_global_loss,
        "true_final_reverse_kl": true_final,
        "estimated_minus_true_final_kl": result.final_global_loss - true_final,
        "true_relative_kl_reduction": (
            (true_initial - true_final) / true_initial if true_initial > 0 else 0.0
        ),
        "actionable_orientation_count": len(result.direct_orientation_evidence),
        "meek_orientation_count": len(result.meek_orientation_evidence),
        "actionable_source_events": audit.actionable_source_events,
        "actionable_coordinates_constructed": audit.coordinate_searches_started,
        "candidate_search_reached": audit.coordinate_searches_started > 0,
        "candidate_proposals_constructed": audit.candidate_proposals_constructed,
        "coordinate_optimizations_completed": audit.candidate_proposals_constructed,
        "global_acceptance_reached": audit.acceptance_evaluations_started > 0,
        "acceptance_evaluations": audit.acceptance_evaluations_started,
        "accepted_candidates": result.accepted_proposals,
        "at_least_one_accepted_update": result.accepted_proposals > 0,
        "rejected_candidates": result.rejected_proposals,
        "inconclusive_candidates": result.inconclusive_proposals,
        "candidate_evaluations": result.optimizer_search_evaluations,
        "accepted_shift_coordinates": [x for x in labels if x.startswith("shift:")],
        "accepted_scale_coordinates": [x for x in labels if x.startswith("log_scale:")],
        "accepted_final_support": list(labels),
        "final_support_size": len(labels),
        "probe_queries": int(result.query_accounting["probe_queries"] or 0),
        "intervention_queries": int(result.query_accounting["candidate_queries"] or 0),
        "total_samples_drawn": spec.N_target
        + int(result.query_accounting["total_observations"] or 0),
        "query_accounting": result.query_accounting,
        "query_history": query_history,
        "query_sample_audit": environment.sample_audit,
        "termination_reason": result.termination_reason,
        "oriented_edges": [
            list(edge) for edge in sorted(result.final_graph.directed_edges)
        ],
        "proposal_diagnostics": proposal_diagnostics,
        "runtime_seconds": runtime_seconds,
        "problem_fingerprint": runtime.problem_fingerprint,
        "source_sample_fingerprint": _sample_fingerprint(source_samples),
        "target_sample_fingerprint": _sample_fingerprint(target_samples),
        "local_state_audit": audit.local_state_events,
        "local_evidence_audit": audit.local_evidence_events,
        "structural_state_audit": audit.structural_state_events,
        **_actionability_metrics(audit),
        **_accepted_diagnostics(problem, result),
        **_graph_diagnostics(problem.ground_truth.scm.dag, result),
    }
    metrics["environment_fingerprint"] = _stable_hash(environment.sample_audit)
    metrics["result_fingerprint"] = _stable_hash(
        {
            "run_fingerprint": spec.fingerprint,
            "termination": result.termination_reason,
            "final_support": list(labels),
            "query_sample_audit": environment.sample_audit,
            "local_evidence_audit": audit.local_evidence_events,
            "proposal_diagnostics": proposal_diagnostics,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "campaign_identifier": CAMPAIGN_ID,
        "protocol_identifier": PROTOCOL_ID,
        "run_id": spec.run_id,
        "run_fingerprint": spec.fingerprint,
        "policy_identifier": spec.policy_identifier,
        "paired_problem_id": spec.paired_problem_id,
        "base_seed": spec.base_seed,
        "exact_reference": {
            "run_id": spec.exact_run_id,
            "path": spec.exact_result_path,
            "checksum": _checksum(Path(spec.exact_result_path)),
        },
        "frozen_conservative_reference": {
            "path": spec.frozen_conservative_path,
            "checksum": spec.frozen_conservative_checksum,
        },
        "spec": asdict(spec),
        "metrics": metrics,
    }


def result_path(run_id: str) -> Path:
    return OUTPUT_ROOT / "results" / f"{run_id}.json"


def run_one(run_id: str) -> dict[str, object]:
    write_manifest()
    spec = next((item for item in campaign_specs() if item.run_id == run_id), None)
    if spec is None:
        raise ValueError(f"unknown local-evidence run ID {run_id}")
    path = result_path(run_id)
    failure = OUTPUT_ROOT / "failures" / f"{run_id}.json"
    if path.exists():
        row = json.loads(path.read_text())
        if row.get("run_fingerprint") != spec.fingerprint or failure.exists():
            raise RuntimeError(f"incompatible or duplicate artifact for {run_id}")
        return {"status": "skipped_completed", "run_id": run_id}
    try:
        row = execute(spec)
        _atomic_json(path, row)
        if failure.exists():
            failure.unlink()
        return row
    except Exception as error:  # noqa: BLE001
        row = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "campaign_identifier": CAMPAIGN_ID,
            "protocol_identifier": PROTOCOL_ID,
            "run_id": run_id,
            "run_fingerprint": spec.fingerprint,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(failure, row)
        return row


def smoke_specs() -> tuple[LocalEvidenceAblationSpec, ...]:
    return tuple(
        next(
            item
            for item in campaign_specs()
            if item.base_seed == BASE_SEEDS[0]
            and item.policy_identifier == policy
            and item.N_int == n_int
        )
        for policy in POLICIES
        for n_int in N_INTS
    )


def smoke() -> dict[str, object]:
    rows = []
    for spec in smoke_specs():
        row = execute(spec)
        metrics = row["metrics"]
        rows.append(
            {
                "run_id": spec.run_id,
                "policy": spec.policy_identifier,
                "N_int": spec.N_int,
                "candidate_search_reached": metrics["candidate_search_reached"],
                "candidate_proposals_constructed": metrics[
                    "candidate_proposals_constructed"
                ],
                "acceptance_evaluations": metrics["acceptance_evaluations"],
                "accepted_candidates": metrics["accepted_candidates"],
                "global_acceptance_reached": metrics["global_acceptance_reached"],
                "accepted_coordinates": metrics["accepted_final_support"],
                "oriented_edges": metrics["oriented_edges"],
                "estimated_initial_kl": metrics["estimated_initial_reverse_kl"],
                "estimated_final_kl": metrics["estimated_final_reverse_kl"],
                "true_initial_kl": metrics["true_initial_reverse_kl"],
                "true_final_kl": metrics["true_final_reverse_kl"],
                "true_improvement": metrics["true_initial_reverse_kl"]
                - metrics["true_final_reverse_kl"],
                "proposal_diagnostics": metrics["proposal_diagnostics"],
                "termination_reason": metrics["termination_reason"],
                "local_evidence_audit": metrics["local_evidence_audit"],
                "structural_state_audit": metrics["structural_state_audit"],
                "query_sample_audit": metrics["query_sample_audit"],
            }
        )
    oracle_rows = [row for row in rows if row["policy"] == "oracle_local_actionability"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
        "prespecified_gate": {
            "rule": "continue iff at least one oracle-local smoke reaches candidate search",
            "oracle_local_search_count": sum(
                bool(row["candidate_search_reached"]) for row in oracle_rows
            ),
            "continue_campaign": any(
                bool(row["candidate_search_reached"]) for row in oracle_rows
            ),
        },
    }
    _atomic_json(OUTPUT_ROOT / "smoke_summary.json", payload)
    return payload




def repair_audit_metadata() -> dict[str, object]:
    """Add deterministic structural-gate metadata without rerunning simulations."""
    expected = {item.run_id: item for item in campaign_specs()}
    repaired_files = 0
    repaired_decisions = 0
    for path in sorted((OUTPUT_ROOT / "results").glob("*.json")):
        row = json.loads(path.read_text())
        spec = expected.get(path.stem)
        if (
            spec is None
            or row.get("status") != "completed"
            or row.get("run_fingerprint") != spec.fingerprint
        ):
            raise RuntimeError(f"refusing to repair incompatible artifact {path}")
        metrics = row["metrics"]
        structural = {
            int(event["step"]): event
            for event in metrics.get("structural_state_audit", [])
        }
        changed = False
        for event in metrics.get("local_evidence_audit", []):
            snapshot = structural.get(int(event["step"]))
            decisions = event["node_decisions"]
            if snapshot is not None:
                active = {int(node) for node in snapshot["active_nodes"]}
                sources = {
                    int(node) for node in snapshot["certified_actionable_sources"]
                }
                components = snapshot["top_unresolved_chain_components"]
                for decision in decisions:
                    node = int(decision["node"])
                    if node in sources:
                        state = "certified_actionable_source"
                        reason = "structural_certificate_obtained"
                    elif node in active:
                        state = "structural_source_unavailable"
                        reason = "structural_source_unavailable"
                    else:
                        state = "not_applicable_local_mismatch_absent"
                        reason = "no_localizable_mismatch"
                    component = next(
                        (list(item) for item in components if node in item), None
                    )
                    values = {
                        "structural_certificate_state": state,
                        "structural_decision_reason": reason,
                        "structurally_actionable_source": node in sources,
                        "top_unresolved_chain_component": component,
                    }
                    record_changed = any(
                        decision.get(key) != value for key, value in values.items()
                    )
                    repaired_decisions += int(record_changed)
                    changed |= record_changed
                    decision.update(values)
            else:
                local_block = any(
                    decision["applied_state"] == "uncertain"
                    for decision in decisions
                )
                if local_block:
                    state = "not_evaluated_local_evidence_block"
                    reason = "ci_crosses_threshold"
                elif metrics["termination_reason"] == "global_mismatch_not_localized":
                    state = "not_evaluated_no_localizable_mismatch"
                    reason = "no_localizable_mismatch"
                elif metrics["termination_reason"] == (
                    "confidence_aware_loss_tolerance_reached"
                ):
                    state = "not_evaluated_global_tolerance_reached"
                    reason = "no_localizable_mismatch"
                else:
                    state = "not_evaluated_structural_source_unavailable"
                    reason = "structural_source_unavailable"
                values = {
                    "structural_certificate_state": state,
                    "structural_decision_reason": reason,
                    "structurally_actionable_source": None,
                    "top_unresolved_chain_component": None,
                }
                for decision in decisions:
                    record_changed = any(
                        decision.get(key) != value for key, value in values.items()
                    )
                    repaired_decisions += int(record_changed)
                    changed |= record_changed
                    decision.update(values)
        metrics["structural_certificates_obtained"] = sum(
            decision.get("structurally_actionable_source") is True
            for event in metrics.get("local_evidence_audit", [])
            for decision in event["node_decisions"]
        )
        metrics["result_fingerprint"] = _stable_hash(
            {
                "run_fingerprint": spec.fingerprint,
                "termination": metrics["termination_reason"],
                "final_support": metrics["accepted_final_support"],
                "query_sample_audit": metrics["query_sample_audit"],
                "local_evidence_audit": metrics["local_evidence_audit"],
                "proposal_diagnostics": metrics["proposal_diagnostics"],
            }
        )
        if row.get("audit_metadata_revision") != 2:
            row["audit_metadata_revision"] = 2
            changed = True
        if changed:
            _atomic_json(path, row)
            repaired_files += 1
    smoke_path = OUTPUT_ROOT / "smoke_summary.json"
    repaired_smoke_decisions = 0
    repaired_smoke_files = 0
    if smoke_path.exists():
        smoke_payload = json.loads(smoke_path.read_text())
        smoke_changed = False
        for smoke_row in smoke_payload["rows"]:
            structural = {
                int(event["step"]): event
                for event in smoke_row.get("structural_state_audit", [])
            }
            for event in smoke_row.get("local_evidence_audit", []):
                snapshot = structural.get(int(event["step"]))
                decisions = event["node_decisions"]
                if snapshot is None:
                    local_block = any(
                        decision["applied_state"] == "uncertain"
                        for decision in decisions
                    )
                    state = (
                        "not_evaluated_local_evidence_block"
                        if local_block
                        else "not_evaluated_structural_source_unavailable"
                    )
                    reason = (
                        "ci_crosses_threshold"
                        if local_block
                        else "structural_source_unavailable"
                    )
                    for decision in decisions:
                        values = {
                            "structural_certificate_state": state,
                            "structural_decision_reason": reason,
                            "structurally_actionable_source": None,
                            "top_unresolved_chain_component": None,
                        }
                        record_changed = any(
                            decision.get(key) != value
                            for key, value in values.items()
                        )
                        repaired_smoke_decisions += int(record_changed)
                        smoke_changed |= record_changed
                        decision.update(values)
                    continue
                active = {int(node) for node in snapshot["active_nodes"]}
                sources = {
                    int(node) for node in snapshot["certified_actionable_sources"]
                }
                components = snapshot["top_unresolved_chain_components"]
                for decision in decisions:
                    node = int(decision["node"])
                    if node in sources:
                        state = "certified_actionable_source"
                        reason = "structural_certificate_obtained"
                    elif node in active:
                        state = "structural_source_unavailable"
                        reason = "structural_source_unavailable"
                    else:
                        state = "not_applicable_local_mismatch_absent"
                        reason = "no_localizable_mismatch"
                    values = {
                        "structural_certificate_state": state,
                        "structural_decision_reason": reason,
                        "structurally_actionable_source": node in sources,
                        "top_unresolved_chain_component": next(
                            (list(item) for item in components if node in item), None
                        ),
                    }
                    record_changed = any(
                        decision.get(key) != value for key, value in values.items()
                    )
                    repaired_smoke_decisions += int(record_changed)
                    smoke_changed |= record_changed
                    decision.update(values)
        if smoke_payload.get("audit_metadata_revision") != 2:
            smoke_payload["audit_metadata_revision"] = 2
            smoke_changed = True
        if smoke_changed:
            _atomic_json(smoke_path, smoke_payload)
            repaired_smoke_files = 1
    return {
        "completed_files_inspected": len(expected),
        "repaired_result_files": repaired_files,
        "repaired_decision_records": repaired_decisions,
        "simulations_rerun": 0,
        "repaired_smoke_files": repaired_smoke_files,
        "repaired_smoke_decision_records": repaired_smoke_decisions,
        "audit_metadata_revision": 2,
    }


def validate_outputs() -> dict[str, object]:
    expected = {item.run_id: item for item in campaign_specs()}
    results = sorted((OUTPUT_ROOT / "results").glob("*.json"))
    failures = sorted((OUTPUT_ROOT / "failures").glob("*.json"))
    corrupt: list[str] = []
    provenance_errors: list[str] = []
    decision_schema_errors: list[str] = []
    rows = []
    required_decision_fields = {
        "node", "point_local_loss", "confidence_interval", "true_local_loss",
        "configured_threshold", "policy_identifier", "actionable",
        "decision_reason", "hidden_information_used", "bootstrap_seed",
        "directed_parent_set", "used_terms", "omitted_terms",
        "structural_certificate_state", "structural_decision_reason",
        "structurally_actionable_source",
    }
    for path in results:
        try:
            row = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            corrupt.append(str(path))
            continue
        rows.append(row)
        spec = expected.get(path.stem)
        metrics = row.get("metrics", {})
        if (
            spec is None or row.get("status") != "completed"
            or row.get("schema_version") != SCHEMA_VERSION
            or row.get("campaign_identifier") != CAMPAIGN_ID
            or row.get("protocol_identifier") != PROTOCOL_ID
            or row.get("run_id") != path.stem
            or row.get("run_fingerprint") != spec.fingerprint
            or row.get("policy_identifier") != spec.policy_identifier
            or row.get("base_seed") != spec.base_seed
            or row.get("paired_problem_id") != spec.paired_problem_id
            or metrics.get("N_int") != spec.N_int
            or metrics.get("N_obs") != spec.N_obs
            or metrics.get("N_target") != spec.N_target
        ):
            provenance_errors.append(str(path))
            continue
        for event in metrics.get("local_evidence_audit", []):
            for decision in event["node_decisions"]:
                if required_decision_fields - set(decision):
                    decision_schema_errors.append(
                        f"{path}:step={event['step']}:node={decision.get('node')}"
                    )
    run_ids = [row.get("run_id") for row in rows]
    fingerprints = [row.get("run_fingerprint") for row in rows]
    duplicate_identifiers = sorted({
        str(value)
        for values in (run_ids, fingerprints)
        for value in values
        if values.count(value) > 1
    })
    paired: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        metrics = row.get("metrics", {})
        if "N_int" in metrics and "base_seed" in row:
            paired.setdefault((row["base_seed"], metrics["N_int"]), []).append(row)
    paired_problem_conflicts = 0
    shared_environment_comparisons = 0
    shared_environment_fingerprint_conflicts = 0
    shared_local_statistic_comparisons = 0
    shared_bootstrap_conflicts = 0
    for group in paired.values():
        for field in (
            "paired_problem_id", "problem_fingerprint",
            "source_sample_fingerprint", "target_sample_fingerprint",
        ):
            values = {
                row.get(field) if field == "paired_problem_id"
                else row.get("metrics", {}).get(field)
                for row in group
            }
            paired_problem_conflicts += int(len(values) != 1)
        environments: dict[tuple[object, ...], list[str]] = {}
        for row in group:
            for item in row.get("metrics", {}).get("query_sample_audit", []):
                key = (
                    item.get("budget_category"), item.get("intervention_id"),
                    item.get("replication_index"), item.get("sample_size"),
                )
                environments.setdefault(key, []).append(item["sample_fingerprint"])
        for values in environments.values():
            if len(values) > 1:
                shared_environment_comparisons += 1
                shared_environment_fingerprint_conflicts += int(len(set(values)) != 1)
        local_statistics: dict[
            tuple[object, ...], dict[str, list[tuple[object, ...]]]
        ] = {}
        for row in group:
            policy = str(row["policy_identifier"])
            for event in row.get("metrics", {}).get("local_evidence_audit", []):
                event_key = (
                    event["baseline_sample_fingerprint"],
                    tuple(event["intervention_shift"]),
                    tuple(event["intervention_scale"]),
                    tuple(tuple(edge) for edge in event["graph_directed_edges"]),
                    tuple(tuple(edge) for edge in event["graph_undirected_edges"]),
                )
                for decision in event["node_decisions"]:
                    key = (*event_key, decision["node"])
                    evidence = (
                        decision["point_local_loss"],
                        tuple(decision["confidence_interval"]),
                        decision["bootstrap_seed"],
                        decision["bootstrap_replicates"],
                    )
                    local_statistics.setdefault(key, {}).setdefault(
                        policy, []
                    ).append(evidence)
        for by_policy in local_statistics.values():
            if len(by_policy) > 1:
                shared_local_statistic_comparisons += 1
                shared_bootstrap_conflicts += int(
                    len({values[0] for values in by_policy.values()}) != 1
                )
    missing = sorted(set(expected) - {path.stem for path in results})
    valid = len(results) - len(set(corrupt + provenance_errors))
    return {
        "expected_jobs": 90,
        "completed_files": len(results),
        "valid_completed_jobs": valid,
        "failure_files": len(failures),
        "corrupt_results": corrupt,
        "provenance_errors": provenance_errors,
        "decision_schema_errors": decision_schema_errors,
        "missing_jobs": missing,
        "duplicate_identifiers": duplicate_identifiers,
        "paired_cells": len(paired),
        "paired_problem_conflicts": paired_problem_conflicts,
        "shared_environment_comparisons": shared_environment_comparisons,
        "shared_environment_fingerprint_conflicts": shared_environment_fingerprint_conflicts,
        "shared_local_statistic_comparisons": shared_local_statistic_comparisons,
        "shared_bootstrap_conflicts": shared_bootstrap_conflicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "manifest", "smoke", "run-one", "run-all", "repair-audit", "validate"
        ),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.command == "manifest":
        output = {"manifest": str(write_manifest()), "jobs": len(campaign_specs())}
    elif args.command == "smoke":
        write_manifest()
        output = smoke()
    elif args.command == "run-one":
        if args.run_id is None:
            parser.error("run-one requires --run-id")
        output = run_one(args.run_id)
    elif args.command == "run-all":
        output = {"rows": [run_one(item.run_id) for item in campaign_specs()]}
    elif args.command == "repair-audit":
        output = repair_audit_metadata()
    else:
        output = validate_outputs()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
