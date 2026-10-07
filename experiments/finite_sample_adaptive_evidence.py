"""Restart-safe paired adaptive local-evidence sampling experiment."""

from __future__ import annotations

import argparse
import hashlib
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
from experiments.finite_sample_local_evidence_ablation import (
    CALIBRATED_SETTINGS,
    _actionability_metrics,
    _enrich_local_evidence,
    _proposal_diagnostics,
)
from experiments.finite_sample_local_evidence_ablation import (
    campaign_specs as local_campaign_specs,
)
from experiments.finite_sample_probe_ablation import (
    KeyedFiniteSampleEnvironment,
    _checksum,
    _frozen_seed_overrides,
)
from experiments.nonoracle import FiniteStageAudit, run_finite_sample_nonoracle
from graphs.problem_gen import PublicProblemView
from matching.capabilities import ProviderCapability
from matching.empirical_moments import EmpiricalMomentsProvider
from matching.local_evidence import (
    AdaptiveConfidenceCertificateLocalEvidenceProvider,
    AdaptiveEvidenceSampleBatch,
)
from matching.source_providers import ConfidenceCalibratedSourceInferenceProvider
from matching.structure_learning import SuppliedExactCPDAGProvider
from oracle.diagnostic_source_provider import OracleLabelOverrideSourceInferenceProvider
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation

PROTOCOL_ID = "finite_sample_adaptive_evidence_v1"
CAMPAIGN_ID = "finite_sample_adaptive_evidence_p08_v1"
SCHEMA_VERSION = 1
POLICY = "adaptive_confidence_certificate"
SAMPLE_CAPS = (4000, 8000)
INITIAL_SAMPLE_SIZE = 2000
BATCH_SIZE = 2000
N_INT = 2000
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/finite_sample_adaptive_evidence_v1"
LOCAL_ROOT = REPOSITORY_ROOT / "outputs/finite_sample_local_evidence_ablation_v1"
FOCUSED_ROOT = REPOSITORY_ROOT / "outputs/finite_sample_focused_v1"
PROBE_ROOT = REPOSITORY_ROOT / "outputs/finite_sample_probe_ablation_v1"
AdaptivePolicy = Literal["adaptive_confidence_certificate"]


@dataclass(frozen=True)
class AdaptiveEvidenceSpec:
    run_id: str
    policy_identifier: AdaptivePolicy
    sample_cap: int
    base_seed: int
    paired_problem_id: str
    exact_run_id: str
    exact_result_path: str
    N_obs: int
    N_target: int
    N_int: int
    sample_seeds: dict[str, int]
    frozen_confidence_path: str
    frozen_confidence_checksum: str
    fingerprint: str


def _local_rows(policy: str) -> dict[int, tuple[Path, dict[str, object]]]:
    rows: dict[int, tuple[Path, dict[str, object]]] = {}
    for path in sorted((LOCAL_ROOT / "results").glob("*.json")):
        row = json.loads(path.read_text())
        metrics = row.get("metrics", {})
        if row.get("policy_identifier") == policy and metrics.get("N_int") == N_INT:
            seed = int(row["base_seed"])
            if seed in rows:
                raise RuntimeError(f"duplicate local reference for {policy}/{seed}")
            rows[seed] = (path, row)
    if set(rows) != set(BASE_SEEDS):
        raise RuntimeError(f"missing local references for {policy}")
    return rows


@lru_cache(maxsize=1)
def campaign_specs() -> tuple[AdaptiveEvidenceSpec, ...]:
    confidence = _local_rows("confidence_certificate")
    old_specs = {
        item.base_seed: item
        for item in local_campaign_specs()
        if item.policy_identifier == "confidence_certificate" and item.N_int == N_INT
    }
    specs = []
    for seed in BASE_SEEDS:
        old = old_specs[seed]
        frozen_path, _ = confidence[seed]
        for cap in SAMPLE_CAPS:
            scientific = {
                "campaign": CAMPAIGN_ID,
                "protocol": PROTOCOL_ID,
                "schema": SCHEMA_VERSION,
                "policy": POLICY,
                "sample_cap": cap,
                "initial_sample_size": INITIAL_SAMPLE_SIZE,
                "batch_size": BATCH_SIZE,
                "source_policy": "oracle_probe_outcome",
                "base_seed": seed,
                "paired_problem_id": old.paired_problem_id,
                "exact_run_id": old.exact_run_id,
                "N_obs": INITIAL_SAMPLE_SIZE,
                "N_target": INITIAL_SAMPLE_SIZE,
                "N_int": N_INT,
                "frozen_confidence_checksum": _checksum(frozen_path),
            }
            fingerprint = _stable_hash(scientific)
            specs.append(
                AdaptiveEvidenceSpec(
                    run_id=f"adaptive-cap{cap}-{fingerprint[:18]}",
                    policy_identifier=POLICY,
                    sample_cap=cap,
                    base_seed=seed,
                    paired_problem_id=old.paired_problem_id,
                    exact_run_id=old.exact_run_id,
                    exact_result_path=old.exact_result_path,
                    N_obs=INITIAL_SAMPLE_SIZE,
                    N_target=INITIAL_SAMPLE_SIZE,
                    N_int=N_INT,
                    sample_seeds=dict(old.sample_seeds),
                    frozen_confidence_path=str(frozen_path),
                    frozen_confidence_checksum=_checksum(frozen_path),
                    fingerprint=fingerprint,
                )
            )
    if len(specs) != 20 or len({item.run_id for item in specs}) != 20:
        raise RuntimeError("adaptive campaign must contain exactly 20 unique jobs")
    return tuple(sorted(specs, key=lambda item: item.run_id))


def _reference_payload() -> list[dict[str, object]]:
    references = []
    for policy in (
        "confidence_certificate",
        "point_estimate_diagnostic",
        "oracle_local_actionability",
    ):
        for seed, (path, row) in sorted(_local_rows(policy).items()):
            references.append(
                {
                    "policy_identifier": policy,
                    "sample_cap": INITIAL_SAMPLE_SIZE
                    if policy == "confidence_certificate"
                    else None,
                    "diagnostic_only": policy != "confidence_certificate",
                    "base_seed": seed,
                    "paired_problem_id": row["paired_problem_id"],
                    "run_id": row["run_id"],
                    "path": str(path),
                    "checksum": _checksum(path),
                    "run_fingerprint": row["run_fingerprint"],
                    "source_sample_fingerprint": row["metrics"][
                        "source_sample_fingerprint"
                    ],
                    "target_sample_fingerprint": row["metrics"][
                        "target_sample_fingerprint"
                    ],
                }
            )
    return references


def _tree_checksums(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(REPOSITORY_ROOT)): _checksum(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def manifest_payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_identifier": CAMPAIGN_ID,
        "protocol_identifier": PROTOCOL_ID,
        "output_root": str(OUTPUT_ROOT),
        "policy_identifier": POLICY,
        "sample_caps": list(SAMPLE_CAPS),
        "initial_sample_size": INITIAL_SAMPLE_SIZE,
        "batch_size": BATCH_SIZE,
        "N_int": N_INT,
        "base_seeds": list(BASE_SEEDS),
        "source_policy": "oracle_probe_outcome",
        "hidden_population_policy": "post_run_evaluation_only",
        "nested_sampling_rule": (
            "cap-independent deterministic batches keyed by paired problem, "
            "environment, role, adaptive stage, and replication index"
        ),
        "cache_policy": (
            "accepted/current environment pools and target pools are reused; "
            "identical completed jobs are skipped by scientific fingerprint"
        ),
        "prior_manifests": {
            "focused": str(FOCUSED_ROOT / "manifest.json"),
            "focused_checksum": _checksum(FOCUSED_ROOT / "manifest.json"),
            "probe": str(PROBE_ROOT / "manifest.json"),
            "probe_checksum": _checksum(PROBE_ROOT / "manifest.json"),
            "local": str(LOCAL_ROOT / "manifest.json"),
            "local_checksum": _checksum(LOCAL_ROOT / "manifest.json"),
        },
        "prior_tree_checksums": {
            "focused": _tree_checksums(FOCUSED_ROOT),
            "probe": _tree_checksums(PROBE_ROOT),
            "local": _tree_checksums(LOCAL_ROOT),
        },
        "reused_references": _reference_payload(),
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
        artifacts = tuple((OUTPUT_ROOT / "results").glob("*.json")) + tuple(
            (OUTPUT_ROOT / "failures").glob("*.json")
        )
        if artifacts:
            raise RuntimeError("existing adaptive manifest is incompatible")
    _atomic_json(path, payload)
    return path


class KeyedAdaptiveTargetSampler:
    """Deterministic sample-only target boundary with restart-safe caching."""

    capabilities = frozenset({ProviderCapability.EMPIRICAL_SAMPLES})

    def __init__(self, sampler, intervention, *, namespace: str) -> None:
        self.__sampler = sampler
        self.__intervention = intervention
        self.__namespace = namespace
        self.__cache: dict[tuple[int, int, int], AdaptiveEvidenceSampleBatch] = {}
        self.sample_audit: list[dict[str, object]] = []

    def sample_batch(
        self,
        *,
        sample_size: int,
        adaptive_stage: int,
        replication_index: int,
    ) -> AdaptiveEvidenceSampleBatch:
        key = (sample_size, adaptive_stage, replication_index)
        if key in self.__cache:
            return self.__cache[key]
        digest = hashlib.sha256(
            (
                f"{self.__namespace}:target:{adaptive_stage}:"
                f"{replication_index}:{sample_size}"
            ).encode()
        ).digest()
        seed = int.from_bytes(digest[:8], "big") % np.iinfo(np.int64).max
        samples = self.__sampler(
            self.__intervention, sample_size, np.random.default_rng(seed)
        )
        batch = AdaptiveEvidenceSampleBatch(
            samples=samples,
            distribution_role="target",
            adaptive_stage=adaptive_stage,
            replication_index=replication_index,
            batch_seed=seed,
            sample_fingerprint=_sample_fingerprint(samples),
        )
        self.__cache[key] = batch
        self.sample_audit.append(
            {
                "distribution_role": "target",
                "adaptive_stage": adaptive_stage,
                "replication_index": replication_index,
                "sample_size": sample_size,
                "batch_seed": seed,
                "sample_fingerprint": batch.sample_fingerprint,
            }
        )
        return batch


def _adaptive_summary(audit: FiniteStageAudit) -> dict[str, object]:
    events = [
        event["node_decisions"][0]
        for event in audit.local_evidence_events
        if event.get("node_decisions")
    ]
    decisions = [
        decision
        for event in audit.local_evidence_events
        for decision in event["node_decisions"]
    ]
    return {
        "adaptive_local_evidence_decisions": len(decisions),
        "adaptive_decision_events": len(events),
        "adaptive_rounds": sum(
            int(item["number_of_adaptive_rounds"]) for item in events
        ),
        "resolved_crossing_intervals": sum(
            bool(item["resolved_because_of_additional_sampling"]) for item in decisions
        ),
        "unresolved_at_cap_count": sum(
            bool(item["unresolved_at_cap"]) for item in decisions
        ),
        "unresolved_at_cap_event_count": sum(
            any(bool(item["unresolved_at_cap"]) for item in event["node_decisions"])
            for event in audit.local_evidence_events
        ),
        "additional_source_samples": sum(
            int(item["additional_source_samples"]) for item in events
        ),
        "additional_target_samples": sum(
            int(item["additional_target_samples"]) for item in events
        ),
        "additional_candidate_environment_samples": sum(
            int(item["additional_candidate_environment_samples"]) for item in events
        ),
        "total_extra_samples": sum(int(item["total_extra_samples"]) for item in events),
        "adaptive_sampling_runtime_seconds": sum(
            float(item["adaptive_sampling_runtime_seconds"]) for item in events
        ),
    }


def execute(spec: AdaptiveEvidenceSpec) -> dict[str, object]:
    if spec.policy_identifier != POLICY or spec.sample_cap not in SAMPLE_CAPS:
        raise ValueError("execution requires an explicit adaptive policy and cap")
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
    frozen = json.loads(Path(spec.frozen_confidence_path).read_text())
    if (
        _sample_fingerprint(source_samples)
        != frozen["metrics"]["source_sample_fingerprint"]
        or _sample_fingerprint(target_samples)
        != frozen["metrics"]["target_sample_fingerprint"]
    ):
        raise RuntimeError("initial samples differ from the frozen paired baseline")
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
        seed_namespace=(
            f"finite_sample_local_evidence_p08_v1:{spec.paired_problem_id}:nint-{N_INT}"
        ),
        overrides=_frozen_seed_overrides(
            next(
                item
                for item in local_campaign_specs()
                if item.base_seed == spec.base_seed
                and item.N_int == N_INT
                and item.policy_identifier == "confidence_certificate"
            )
        ),
    )
    target_provider = KeyedAdaptiveTargetSampler(
        problem.ground_truth.scm.sample_interventional,
        problem.ground_truth.generating_intervention,
        namespace=f"{CAMPAIGN_ID}:{spec.paired_problem_id}",
    )
    structure = SuppliedExactCPDAGProvider(
        cpdag_from_true_dag_for_simulation(problem.ground_truth.scm.dag)
    )
    empirical_probe = ConfidenceCalibratedSourceInferenceProvider(CALIBRATED_SETTINGS)
    oracle_probe = OracleLabelOverrideSourceInferenceProvider(
        problem.ground_truth.scm.dag, empirical_probe
    )
    config = replace(
        _algorithm_config(
            exact,
            type("C", (), {"N_obs": INITIAL_SAMPLE_SIZE, "N_int": N_INT})(),
        ),
        adaptive_local_evidence_initial_sample_size=INITIAL_SAMPLE_SIZE,
        adaptive_local_evidence_batch_size=BATCH_SIZE,
        adaptive_local_evidence_max_sample_size=spec.sample_cap,
    )
    audit = FiniteStageAudit()
    started = perf_counter()
    result = run_finite_sample_nonoracle(
        public_problem,
        environment,
        structure,
        config,
        environment_seed=seeds["environment_root"],
        bootstrap_seed=seeds["bootstrap"],
        source_inference_provider=oracle_probe,
        local_evidence_provider=(AdaptiveConfidenceCertificateLocalEvidenceProvider()),
        adaptive_target_samples=target_samples,
        adaptive_target_sample_provider=target_provider,
        diagnostic_mode="exact_cpdag_exact_source_empirical_matching",
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
        [asdict(item) for item in environment.query_history]
    )
    metrics = {
        "local_evidence_policy_identifier": POLICY,
        "probe_policy_identifier": "oracle_probe_outcome",
        "sample_cap": spec.sample_cap,
        "initial_sample_size": INITIAL_SAMPLE_SIZE,
        "adaptive_batch_size": BATCH_SIZE,
        "N_obs": spec.N_obs,
        "N_target": spec.N_target,
        "N_int": spec.N_int,
        "local_loss_tolerance": config.local_loss_tolerance,
        "confidence_level": config.confidence_level,
        "bootstrap_replicates": config.bootstrap_replicates,
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
        "global_acceptance_attempts": audit.acceptance_evaluations_started,
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
        + int(result.query_accounting["total_observations"] or 0)
        + sum(item["sample_size"] for item in target_provider.sample_audit),
        "query_accounting": result.query_accounting,
        "query_history": query_history,
        "query_sample_audit": environment.sample_audit,
        "adaptive_target_sample_audit": target_provider.sample_audit,
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
        "full_problem_and_sample_provenance": {
            "paired_problem_id": spec.paired_problem_id,
            "exact_run_id": spec.exact_run_id,
            "sample_seeds": dict(spec.sample_seeds),
            "source_sample_fingerprint": _sample_fingerprint(source_samples),
            "target_sample_fingerprint": _sample_fingerprint(target_samples),
            "environment_seed_namespace": (
                f"finite_sample_local_evidence_p08_v1:"
                f"{spec.paired_problem_id}:nint-{N_INT}"
            ),
            "target_seed_namespace": f"{CAMPAIGN_ID}:{spec.paired_problem_id}",
        },
        **_adaptive_summary(audit),
        **_actionability_metrics(audit),
        **_accepted_diagnostics(problem, result),
        **_graph_diagnostics(problem.ground_truth.scm.dag, result),
    }
    metrics["sample_provenance_fingerprint"] = _stable_hash(
        {
            "source": metrics["source_sample_fingerprint"],
            "target": metrics["target_sample_fingerprint"],
            "environment": metrics["query_sample_audit"],
            "adaptive_target": metrics["adaptive_target_sample_audit"],
        }
    )
    metrics["result_fingerprint"] = _stable_hash(
        {
            "run_fingerprint": spec.fingerprint,
            "termination": result.termination_reason,
            "final_support": list(labels),
            "query_sample_audit": environment.sample_audit,
            "adaptive_target_sample_audit": target_provider.sample_audit,
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
        "sample_cap": spec.sample_cap,
        "paired_problem_id": spec.paired_problem_id,
        "base_seed": spec.base_seed,
        "exact_reference": {
            "run_id": spec.exact_run_id,
            "path": spec.exact_result_path,
            "checksum": _checksum(Path(spec.exact_result_path)),
        },
        "frozen_confidence_reference": {
            "path": spec.frozen_confidence_path,
            "checksum": spec.frozen_confidence_checksum,
        },
        "spec": asdict(spec),
        "adaptive_configuration": {
            "initial_sample_size": INITIAL_SAMPLE_SIZE,
            "batch_size": BATCH_SIZE,
            "sample_cap": spec.sample_cap,
        },
        "metrics": metrics,
    }


def result_path(run_id: str) -> Path:
    return OUTPUT_ROOT / "results" / f"{run_id}.json"


def run_one(run_id: str) -> dict[str, object]:
    write_manifest()
    spec = next((item for item in campaign_specs() if item.run_id == run_id), None)
    if spec is None:
        raise ValueError(f"unknown adaptive run ID {run_id}")
    path = result_path(run_id)
    failure = OUTPUT_ROOT / "failures" / f"{run_id}.json"
    if path.exists():
        row = json.loads(path.read_text())
        if (
            row.get("run_fingerprint") != spec.fingerprint
            or row.get("status") != "completed"
            or failure.exists()
        ):
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


def smoke_specs() -> tuple[AdaptiveEvidenceSpec, ...]:
    return tuple(
        next(
            item
            for item in campaign_specs()
            if item.base_seed == BASE_SEEDS[0] and item.sample_cap == cap
        )
        for cap in SAMPLE_CAPS
    )


def _smoke_row(row: dict[str, object]) -> dict[str, object]:
    metrics = row["metrics"]
    crossing = []
    for event in metrics["local_evidence_audit"]:
        for decision in event["node_decisions"]:
            if any(
                item["confidence_state"] == "uncertain"
                for item in decision["adaptive_round_history"]
            ):
                crossing.append(
                    {
                        "step": event["step"],
                        "node": decision["node"],
                        "initial_interval": (
                            decision["confidence_interval_after_each_round"][0]
                        ),
                        "final_interval": (
                            decision["confidence_interval_after_each_round"][-1]
                        ),
                        "initial_sample_count": decision[
                            "initial_current_sample_count"
                        ],
                        "final_sample_count": decision["sample_count_per_distribution"],
                        "adaptive_rounds": decision["number_of_adaptive_rounds"],
                        "resolved": decision["resolved_because_of_additional_sampling"],
                        "final_decision": decision["final_decision"],
                        "reason_code": decision["reason_code"],
                        "nested_sample_fingerprints": decision[
                            "nested_sample_fingerprints"
                        ],
                    }
                )
    return {
        "run_id": row["run_id"],
        "sample_cap": row["sample_cap"],
        "crossing_intervals": crossing,
        "structural_certificates_obtained": metrics["structural_certificates_obtained"],
        "candidate_search_reached": metrics["candidate_search_reached"],
        "candidate_evaluations": metrics["candidate_evaluations"],
        "accepted_updates": metrics["accepted_candidates"],
        "estimated_final_kl": metrics["estimated_final_reverse_kl"],
        "true_final_kl": metrics["true_final_reverse_kl"],
        "termination_reason": metrics["termination_reason"],
        "total_extra_samples": metrics["total_extra_samples"],
    }


def smoke() -> dict[str, object]:
    rows = [_smoke_row(execute(spec)) for spec in smoke_specs()]
    crossing = [item for row in rows for item in row["crossing_intervals"]]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
        "prespecified_gate": {
            "crossing_intervals": len(crossing),
            "extra_sampling_observed": any(
                int(row["total_extra_samples"]) > 0 for row in rows
            ),
            "continue_campaign": bool(crossing)
            and any(int(row["total_extra_samples"]) > 0 for row in rows),
        },
    }
    _validate_nested_smoke(rows)
    payload["nested_sample_fingerprint_validation"] = "passed"
    _atomic_json(OUTPUT_ROOT / "smoke_summary.json", payload)
    return payload


def _validate_nested_smoke(rows: list[dict[str, object]]) -> None:
    by_cap = {int(row["sample_cap"]): row for row in rows}
    low = by_cap[4000]
    high = by_cap[8000]
    low_crossing = {
        (item["step"], item["node"]): item for item in low["crossing_intervals"]
    }
    high_crossing = {
        (item["step"], item["node"]): item for item in high["crossing_intervals"]
    }
    compared = 0
    for key in sorted(set(low_crossing) & set(high_crossing)):
        low_fingerprints = low_crossing[key]["nested_sample_fingerprints"]
        high_fingerprints = high_crossing[key]["nested_sample_fingerprints"]
        for low_item, high_item in zip(
            low_fingerprints, high_fingerprints, strict=False
        ):
            if int(low_item["adaptive_stage"]) > 1:
                continue
            compared += 1
            if low_item != high_item:
                raise RuntimeError("nested smoke sample fingerprints conflict")
    if compared == 0:
        raise RuntimeError("nested smoke validation found no common sample prefix")


def validate_outputs() -> dict[str, object]:
    expected = {item.run_id: item for item in campaign_specs()}
    results = sorted((OUTPUT_ROOT / "results").glob("*.json"))
    failures = sorted((OUTPUT_ROOT / "failures").glob("*.json"))
    corrupt = []
    provenance_errors = []
    schema_errors = []
    rows = []
    required_decision_fields = {
        "policy_identifier",
        "initial_sample_count",
        "final_pooled_sample_count",
        "configured_sample_cap",
        "number_of_adaptive_rounds",
        "sample_count_per_distribution",
        "batch_sizes",
        "batch_seeds",
        "nested_sample_fingerprints",
        "empirical_point_loss_after_each_round",
        "confidence_interval_after_each_round",
        "true_local_loss",
        "configured_threshold",
        "final_decision",
        "reason_code",
        "resolved_because_of_additional_sampling",
        "unresolved_at_cap",
        "additional_source_samples",
        "additional_target_samples",
        "additional_candidate_environment_samples",
        "total_extra_samples",
        "adaptive_sampling_runtime_seconds",
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
            spec is None
            or row.get("status") != "completed"
            or row.get("schema_version") != SCHEMA_VERSION
            or row.get("campaign_identifier") != CAMPAIGN_ID
            or row.get("protocol_identifier") != PROTOCOL_ID
            or row.get("run_fingerprint") != spec.fingerprint
            or row.get("policy_identifier") != POLICY
            or row.get("sample_cap") != spec.sample_cap
            or row.get("base_seed") != spec.base_seed
            or row.get("paired_problem_id") != spec.paired_problem_id
            or metrics.get("N_int") != N_INT
        ):
            provenance_errors.append(str(path))
            continue
        for event in metrics.get("local_evidence_audit", []):
            for decision in event["node_decisions"]:
                if required_decision_fields - set(decision):
                    schema_errors.append(
                        f"{path}:step={event['step']}:node={decision.get('node')}"
                    )
    identifiers = [row.get("run_id") for row in rows]
    fingerprints = [row.get("run_fingerprint") for row in rows]
    duplicates = sorted(
        {
            str(value)
            for values in (identifiers, fingerprints)
            for value in values
            if values.count(value) > 1
        }
    )
    reference_errors = []
    manifest = manifest_payload()
    for reference in manifest["reused_references"]:
        path = Path(reference["path"])
        if not path.exists() or _checksum(path) != reference["checksum"]:
            reference_errors.append(str(path))
    paired_conflicts = 0
    nested_prefix_comparisons = 0
    nested_prefix_conflicts = 0
    by_seed: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["base_seed"]), []).append(row)
    for group in by_seed.values():
        for field in ("paired_problem_id",):
            paired_conflicts += int(len({row[field] for row in group}) != 1)
        for field in (
            "problem_fingerprint",
            "source_sample_fingerprint",
            "target_sample_fingerprint",
        ):
            paired_conflicts += int(len({row["metrics"][field] for row in group}) != 1)
        if {row["sample_cap"] for row in group} != set(SAMPLE_CAPS):
            continue
        low = next(row for row in group if row["sample_cap"] == 4000)
        high = next(row for row in group if row["sample_cap"] == 8000)
        low_events = {
            (
                tuple(event["intervention_shift"]),
                tuple(event["intervention_scale"]),
                tuple(tuple(edge) for edge in event["graph_directed_edges"]),
                tuple(tuple(edge) for edge in event["graph_undirected_edges"]),
            ): event
            for event in low["metrics"]["local_evidence_audit"]
        }
        high_events = {
            (
                tuple(event["intervention_shift"]),
                tuple(event["intervention_scale"]),
                tuple(tuple(edge) for edge in event["graph_directed_edges"]),
                tuple(tuple(edge) for edge in event["graph_undirected_edges"]),
            ): event
            for event in high["metrics"]["local_evidence_audit"]
        }
        for key in set(low_events) & set(high_events):
            low_rounds = low_events[key]["node_decisions"][0]["adaptive_round_history"]
            high_rounds = high_events[key]["node_decisions"][0][
                "adaptive_round_history"
            ]
            high_by_count = {
                (
                    item["current_sample_count"],
                    item["target_sample_count"],
                ): item
                for item in high_rounds
            }
            for low_item in low_rounds:
                count_key = (
                    low_item["current_sample_count"],
                    low_item["target_sample_count"],
                )
                if count_key not in high_by_count:
                    continue
                nested_prefix_comparisons += 1
                high_item = high_by_count[count_key]
                nested_prefix_conflicts += int(
                    low_item["current_pool_fingerprint"]
                    != high_item["current_pool_fingerprint"]
                    or low_item["target_pool_fingerprint"]
                    != high_item["target_pool_fingerprint"]
                )
    missing = sorted(set(expected) - {path.stem for path in results})
    valid = len(results) - len(set(corrupt + provenance_errors))
    return {
        "expected_jobs": 20,
        "completed_files": len(results),
        "valid_completed_jobs": valid,
        "failure_files": len(failures),
        "missing_jobs": missing,
        "corrupt_results": corrupt,
        "provenance_errors": provenance_errors,
        "decision_schema_errors": schema_errors,
        "duplicate_identifiers": duplicates,
        "reused_reference_errors": reference_errors,
        "paired_seed_groups": len(by_seed),
        "paired_fingerprint_conflicts": paired_conflicts,
        "nested_prefix_comparisons": nested_prefix_comparisons,
        "nested_prefix_conflicts": nested_prefix_conflicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("manifest", "smoke", "run-one", "run-all", "validate"),
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
    else:
        output = validate_outputs()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
