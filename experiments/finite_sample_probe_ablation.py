"""Restart-safe paired finite-sample probe-gate ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import traceback
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import networkx as nx
import numpy as np

from environments.blackbox import (
    EnvironmentQueryLedger,
    FiniteSampleBlackBoxEnvironment,
    QueryBudgetCategory,
    QueryKind,
    SampleQueryResult,
)
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
from experiments.finite_sample_campaign import (
    PROTOCOL_ID as FROZEN_PROTOCOL_ID,
)
from experiments.finite_sample_campaign import (
    SCHEMA_VERSION as FROZEN_SCHEMA_VERSION,
)
from experiments.finite_sample_campaign import (
    campaign_specs as frozen_specs,
)
from experiments.nonoracle import FiniteStageAudit, run_finite_sample_nonoracle
from graphs.problem_gen import PublicProblemView
from matching.confidence_calibrated_source import (
    ConfidenceCalibratedSourceSettings,
)
from matching.empirical_moments import EmpiricalMomentsProvider
from matching.source_providers import ConfidenceCalibratedSourceInferenceProvider
from matching.structure_learning import SuppliedExactCPDAGProvider
from matching.types import InterventionBackground
from oracle.diagnostic_source_provider import (
    OracleLabelOverrideSourceInferenceProvider,
)
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation

PROTOCOL_ID = "finite_sample_probe_ablation_v1"
CAMPAIGN_ID = "finite_sample_probe_gate_p08_v1"
SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/finite_sample_probe_ablation_v1"
FROZEN_ROOT = REPOSITORY_ROOT / "outputs/finite_sample_focused_v1"
N_INTS = (100, 500, 2000)
NEW_POLICIES = ("confidence_calibrated", "oracle_probe_outcome")
ALL_POLICIES = ("conservative_empirical", *NEW_POLICIES)
ProbePolicy = Literal[
    "conservative_empirical", "confidence_calibrated", "oracle_probe_outcome"
]
CALIBRATED_SETTINGS = ConfidenceCalibratedSourceSettings(0.1, 0.05, 0.95, 30)


@dataclass(frozen=True)
class ProbeAblationSpec:
    run_id: str
    policy_identifier: str
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


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_rows() -> dict[tuple[int, int], tuple[Path, dict[str, object]]]:
    rows = {}
    for path in sorted((FROZEN_ROOT / "results").glob("*.json")):
        payload = json.loads(path.read_text())
        metrics = payload["metrics"]
        key = (int(payload["base_seed"]), int(metrics["N_int"]))
        if (
            metrics["N_obs"] == 2000
            and metrics["N_target"] == 2000
            and key[1] in N_INTS
        ):
            if key in rows:
                raise RuntimeError(f"duplicate frozen conservative cell {key}")
            rows[key] = (path, payload)
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 frozen conservative rows, found {len(rows)}")
    return rows


def _validate_frozen_row(
    path: Path,
    payload: dict[str, object],
    *,
    base_seed: int,
    n_int: int,
    paired_problem_id: str,
    exact_run_id: str,
) -> None:
    metrics = payload["metrics"]
    expected_spec = next(
        item
        for item in frozen_specs()
        if item.base_seed == base_seed and item.N_obs == 2000 and item.N_int == n_int
    )
    if (
        payload["schema_version"] != FROZEN_SCHEMA_VERSION
        or payload["campaign_identifier"] != "finite_sample_graph_aware_p08_v1"
        or payload["protocol_identifier"] != FROZEN_PROTOCOL_ID
        or payload["status"] != "completed"
        or payload["run_fingerprint"] != expected_spec.fingerprint
        or payload["paired_problem_id"] != paired_problem_id
        or payload["exact_reference"]["run_id"] != exact_run_id
        or metrics["N_obs"] != 2000
        or metrics["N_target"] != 2000
        or metrics["N_int"] != n_int
        or metrics["termination_reason"] != "insufficient_evidence"
    ):
        raise RuntimeError(f"incompatible frozen conservative row: {path}")


def campaign_specs() -> tuple[ProbeAblationSpec, ...]:
    frozen = _frozen_rows()
    specs = []
    for old in frozen_specs():
        if old.N_obs != 2000 or old.N_target != 2000 or old.N_int not in N_INTS:
            continue
        path, row = frozen[(old.base_seed, old.N_int)]
        _validate_frozen_row(
            path,
            row,
            base_seed=old.base_seed,
            n_int=old.N_int,
            paired_problem_id=old.paired_problem_id,
            exact_run_id=old.exact_run_id,
        )
        for policy in NEW_POLICIES:
            scientific = {
                "campaign": CAMPAIGN_ID,
                "protocol": PROTOCOL_ID,
                "schema": SCHEMA_VERSION,
                "policy": policy,
                "base_seed": old.base_seed,
                "paired_problem_id": old.paired_problem_id,
                "exact_run_id": old.exact_run_id,
                "N_obs": 2000,
                "N_target": 2000,
                "N_int": old.N_int,
                "calibrated_settings": asdict(CALIBRATED_SETTINGS),
                "frozen_conservative_checksum": _checksum(path),
            }
            fingerprint = _stable_hash(scientific)
            specs.append(
                ProbeAblationSpec(
                    run_id=f"probe-{policy[:6]}-{fingerprint[:18]}",
                    policy_identifier=policy,
                    base_seed=old.base_seed,
                    paired_problem_id=old.paired_problem_id,
                    exact_run_id=old.exact_run_id,
                    exact_result_path=old.exact_result_path,
                    N_obs=2000,
                    N_target=2000,
                    N_int=old.N_int,
                    sample_seeds=dict(old.sample_seeds),
                    frozen_conservative_path=str(path),
                    frozen_conservative_checksum=_checksum(path),
                    fingerprint=fingerprint,
                )
            )
    if len(specs) != 60 or len({item.run_id for item in specs}) != 60:
        raise RuntimeError("probe ablation must contain exactly 60 unique new jobs")
    return tuple(sorted(specs, key=lambda item: item.run_id))


def conservative_references() -> tuple[dict[str, object], ...]:
    references = []
    for (seed, n_int), (path, row) in sorted(_frozen_rows().items()):
        references.append(
            {
                "policy_identifier": "conservative_empirical",
                "base_seed": seed,
                "N_int": n_int,
                "run_id": row["run_id"],
                "path": str(path),
                "checksum": _checksum(path),
                "run_fingerprint": row["run_fingerprint"],
                "paired_problem_id": row["paired_problem_id"],
                "source_sample_fingerprint": row["metrics"][
                    "source_sample_fingerprint"
                ],
                "target_sample_fingerprint": row["metrics"][
                    "target_sample_fingerprint"
                ],
            }
        )
    return tuple(references)


def manifest_payload() -> dict[str, object]:
    frozen_manifest = FROZEN_ROOT / "manifest.json"
    exact_manifest = json.loads(frozen_manifest.read_text())["exact_manifest"]
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_identifier": CAMPAIGN_ID,
        "protocol_identifier": PROTOCOL_ID,
        "output_root": str(OUTPUT_ROOT),
        "policies": list(ALL_POLICIES),
        "new_policies": list(NEW_POLICIES),
        "N_obs": 2000,
        "N_target": 2000,
        "N_int": list(N_INTS),
        "base_seeds": list(BASE_SEEDS),
        "calibrated_settings": asdict(CALIBRATED_SETTINGS),
        "minimum_evidence_rule": "at_least_30_observations_per_arm",
        "equivalence_margin_rule": "0.1 + 0.05 * abs(probe_increment)",
        "frozen_manifest": str(frozen_manifest),
        "frozen_manifest_checksum": _checksum(frozen_manifest),
        "exact_manifest": exact_manifest,
        "exact_manifest_checksum": _checksum(Path(exact_manifest)),
        "conservative_references": list(conservative_references()),
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


def write_manifest() -> Path:
    path = OUTPUT_ROOT / "manifest.json"
    payload = json.loads(json.dumps(manifest_payload()))
    if path.exists():
        prior = json.loads(path.read_text())
        if prior != payload:
            artifacts = tuple((OUTPUT_ROOT / "results").glob("*.json")) + tuple(
                (OUTPUT_ROOT / "failures").glob("*.json")
            )
            if artifacts:
                raise RuntimeError("existing probe-ablation manifest is incompatible")
            _atomic_json(path, payload)
        return path
    _atomic_json(path, payload)
    return path


def _frozen_seed_overrides(spec: ProbeAblationSpec) -> dict[tuple[str, str, int], int]:
    frozen = json.loads(Path(spec.frozen_conservative_path).read_text())
    overrides = {}
    for query in frozen["metrics"]["query_history"]:
        key = (
            str(query["intervention_id"]),
            str(query["budget_category"]),
            int(query["replication_index"]),
        )
        seed = query["query_seed"]
        if seed is None or key in overrides:
            raise RuntimeError("invalid frozen query seed map")
        overrides[key] = int(seed)
    return overrides


class KeyedFiniteSampleEnvironment(FiniteSampleBlackBoxEnvironment):
    """Policy-independent query-key seeds with frozen conservative overrides."""

    def __init__(
        self,
        hidden_scm,
        *,
        instance_id: str,
        ledger: EnvironmentQueryLedger,
        seed_namespace: str,
        overrides: dict[tuple[str, str, int], int],
    ) -> None:
        super().__init__(hidden_scm, instance_id=instance_id, ledger=ledger)
        self._seed_namespace = seed_namespace
        self._overrides = dict(overrides)
        self._replications: Counter[tuple[str, str]] = Counter()
        self.sample_audit: list[dict[str, object]] = []

    def query(
        self,
        intervention,
        *,
        sample_size: int,
        query_seed: int,
        kind: QueryKind,
        budget_category: QueryBudgetCategory | None = None,
        background_id: str,
    ) -> SampleQueryResult:
        del query_seed
        category = budget_category
        if category is None:
            if kind == "baseline":
                category = "baseline"
            elif kind == "probe":
                category = "probe"
            elif kind == "candidate":
                category = "candidate"
            else:
                raise ValueError("repeat queries require an explicit budget category")
        intervention_id = InterventionBackground.from_intervention(
            intervention
        ).identifier
        counter_key = (intervention_id, category)
        replication = self._replications[counter_key]
        self._replications[counter_key] += 1
        key = (intervention_id, category, replication)
        resolved = self._overrides.get(key)
        seed_source = "frozen_conservative_override"
        if resolved is None:
            digest = hashlib.sha256(
                f"{self._seed_namespace}:{intervention_id}:{category}:{replication}".encode()
            ).digest()
            resolved = int.from_bytes(digest[:8], "big") % np.iinfo(np.int64).max
            seed_source = "policy_independent_keyed_seed"
        result = super().query(
            intervention,
            sample_size=sample_size,
            query_seed=resolved,
            kind=kind,
            budget_category=category,
            background_id=background_id,
        )
        self.sample_audit.append(
            {
                "intervention_id": intervention_id,
                "budget_category": category,
                "replication_index": replication,
                "sample_size": sample_size,
                "query_seed": resolved,
                "seed_source": seed_source,
                "sample_fingerprint": _sample_fingerprint(result.samples),
            }
        )
        return result


def _probe_metrics(result, true_dag: nx.DiGraph) -> dict[str, object]:
    rows = []
    applied_counts: Counter[str] = Counter()
    empirical_counts: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    empirical_correct = 0
    applied_correct = 0
    total = 0
    for record in result.history:
        if not record.classifications:
            continue
        targets = {node for node, _ in record.classifications}
        details = {
            int(item["node"]): item
            for item in record.decision_metadata
            if "node" in item
        }
        for node, applied in record.classifications:
            detail = details[node]
            empirical = str(detail.get("empirical_classification", applied))
            oracle = (
                "source"
                if not (set(nx.ancestors(true_dag, node)) & targets)
                else "non_source"
            )
            applied_counts[applied] += 1
            empirical_counts[empirical] += 1
            confusion[f"{empirical}->{oracle}"] += 1
            empirical_correct += int(empirical == oracle)
            applied_correct += int(applied == oracle)
            total += 1
            rows.append(
                {
                    "step": record.step,
                    "node": node,
                    "targets": sorted(targets),
                    "empirical_classification": empirical,
                    "applied_classification": applied,
                    "oracle_classification": oracle,
                    "empirical_correct": empirical == oracle,
                    "applied_correct": applied == oracle,
                    "evidence": detail,
                }
            )
    return {
        "probe_decisions": rows,
        "probe_classification_count": total,
        "source_classification_count": applied_counts["source"],
        "non_source_classification_count": applied_counts["non_source"],
        "unresolved_classification_count": applied_counts["inconclusive"],
        "empirical_source_count": empirical_counts["source"],
        "empirical_non_source_count": empirical_counts["non_source"],
        "empirical_unresolved_count": empirical_counts["inconclusive"],
        "empirical_probe_accuracy": empirical_correct / total if total else None,
        "applied_probe_accuracy": applied_correct / total if total else None,
        "empirical_to_oracle_confusion": dict(sorted(confusion.items())),
        "unresolved_classification_rate": (
            empirical_counts["inconclusive"] / total if total else None
        ),
    }


def execute(spec: ProbeAblationSpec) -> dict[str, object]:
    if spec.policy_identifier not in NEW_POLICIES:
        raise ValueError("new execution requires an explicit new policy")
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
        raise RuntimeError("source/target samples do not match frozen conservative row")
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
    calibrated = ConfidenceCalibratedSourceInferenceProvider(CALIBRATED_SETTINGS)
    if spec.policy_identifier == "confidence_calibrated":
        provider = calibrated
        diagnostic_mode = None
    else:
        provider = OracleLabelOverrideSourceInferenceProvider(
            problem.ground_truth.scm.dag, calibrated
        )
        diagnostic_mode = "exact_cpdag_exact_source_empirical_matching"
    audit = FiniteStageAudit()
    started = perf_counter()
    result = run_finite_sample_nonoracle(
        public_problem,
        environment,
        structure,
        _algorithm_config(exact, type("C", (), {"N_obs": 2000, "N_int": spec.N_int})()),
        environment_seed=seeds["environment_root"],
        bootstrap_seed=seeds["bootstrap"],
        source_inference_provider=provider,
        diagnostic_mode=diagnostic_mode,
        result_mode="finite_sample_non_oracle_supplied_cpdag",
        stage_audit=audit,
    )
    runtime_seconds = perf_counter() - started
    initial_mean, initial_cov = problem.current_moments()
    final_mean, final_cov = problem.ground_truth.scm.interventional_moments(
        result.final_intervention
    )
    true_initial = _kl(
        problem.public.target_mean,
        problem.public.target_covariance,
        initial_mean,
        initial_cov,
    )
    true_final = _kl(
        problem.public.target_mean,
        problem.public.target_covariance,
        final_mean,
        final_cov,
    )
    labels = _coordinate_labels(result.final_intervention)
    query_history = _enrich_query_history(
        [asdict(x) for x in environment.query_history]
    )
    sample_audit = environment.sample_audit
    metrics = {
        "probe_policy_identifier": spec.policy_identifier,
        "calibrated_confidence_level": CALIBRATED_SETTINGS.confidence_level,
        "calibrated_equivalence_absolute": CALIBRATED_SETTINGS.equivalence_absolute,
        "calibrated_equivalence_relative": CALIBRATED_SETTINGS.equivalence_relative,
        "calibrated_equivalence_margin_at_probe": (
            CALIBRATED_SETTINGS.equivalence_absolute
            + CALIBRATED_SETTINGS.equivalence_relative * 0.5
        ),
        "minimum_evidence_rule": (
            f"at_least_{CALIBRATED_SETTINGS.minimum_sample_size}_observations_per_arm"
        ),
        "N_obs": spec.N_obs,
        "N_target": spec.N_target,
        "N_int": spec.N_int,
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
        "coordinate_optimization_reached": audit.coordinate_searches_started > 0,
        "candidate_proposals_constructed": audit.candidate_proposals_constructed,
        "global_acceptance_reached": audit.acceptance_evaluations_started > 0,
        "acceptance_evaluations": audit.acceptance_evaluations_started,
        "at_least_one_accepted_update": result.accepted_proposals > 0,
        "accepted_candidates": result.accepted_proposals,
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
        "query_sample_audit": sample_audit,
        "termination_reason": result.termination_reason,
        "runtime_seconds": runtime_seconds,
        "problem_fingerprint": runtime.problem_fingerprint,
        "source_sample_fingerprint": _sample_fingerprint(source_samples),
        "target_sample_fingerprint": _sample_fingerprint(target_samples),
        "sample_provenance_fingerprint": _stable_hash(
            {
                "source": _sample_fingerprint(source_samples),
                "target": _sample_fingerprint(target_samples),
                "queries": sample_audit,
            }
        ),
        "coordinates_audited": audit.coordinates,
        "local_state_audit": audit.local_state_events,
        "structural_state_audit": audit.structural_state_events,
        **_probe_metrics(result, problem.ground_truth.scm.dag),
        **_accepted_diagnostics(problem, result),
        **_graph_diagnostics(problem.ground_truth.scm.dag, result),
    }
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
            "run_id": frozen["run_id"],
        },
        "spec": asdict(spec),
        "metrics": metrics,
    }


def result_path(run_id: str) -> Path:
    return OUTPUT_ROOT / "results" / f"{run_id}.json"


def run_one(run_id: str) -> dict[str, object]:
    write_manifest()
    spec = next((x for x in campaign_specs() if x.run_id == run_id), None)
    if spec is None:
        raise ValueError(f"unknown probe-ablation run ID {run_id}")
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
    except Exception as error:  # noqa: BLE001 - durable failure boundary
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


def validate_outputs() -> dict[str, object]:
    expected = {x.run_id: x for x in campaign_specs()}
    results = sorted((OUTPUT_ROOT / "results").glob("*.json"))
    failures = sorted((OUTPUT_ROOT / "failures").glob("*.json"))
    duplicates = {x.stem for x in results} & {x.stem for x in failures}
    foreign = {x.stem for x in (*results, *failures) if x.stem not in expected}
    corrupt = []
    valid = 0
    for path in results:
        try:
            row = json.loads(path.read_text())
            spec = expected[path.stem]
            metrics = row["metrics"]
            if (
                row["schema_version"] != SCHEMA_VERSION
                or row["status"] != "completed"
                or row["campaign_identifier"] != CAMPAIGN_ID
                or row["protocol_identifier"] != PROTOCOL_ID
                or row["run_fingerprint"] != spec.fingerprint
                or row["policy_identifier"] != spec.policy_identifier
                or row["paired_problem_id"] != spec.paired_problem_id
                or metrics["N_int"] != spec.N_int
                or row["frozen_conservative_reference"]["checksum"]
                != _checksum(Path(spec.frozen_conservative_path))
            ):
                raise ValueError("provenance mismatch")
            valid += 1
        except Exception:  # noqa: BLE001 - corrupt artifact boundary
            corrupt.append(str(path))
    return {
        "expected_new_jobs": 60,
        "completed_files": len(results),
        "valid_completed_jobs": valid,
        "failure_files": len(failures),
        "duplicates": sorted(duplicates),
        "foreign_artifacts": sorted(foreign),
        "corrupt_results": corrupt,
        "missing_jobs": sorted(set(expected) - {x.stem for x in results}),
        "validated_conservative_references": len(conservative_references()),
        "comparison_rows_available": valid + len(conservative_references()),
    }


def smoke_specs() -> tuple[ProbeAblationSpec, ...]:
    return tuple(
        next(
            x
            for x in campaign_specs()
            if x.base_seed == BASE_SEEDS[0]
            and x.policy_identifier == policy
            and x.N_int == n_int
        )
        for policy in NEW_POLICIES
        for n_int in N_INTS
    )


def smoke() -> dict[str, object]:
    rows = []
    any_oracle_search = False
    for spec in smoke_specs():
        row = execute(spec)
        metrics = row["metrics"]
        if spec.policy_identifier == "oracle_probe_outcome":
            any_oracle_search |= bool(metrics["candidate_search_reached"])
        rows.append(
            {
                "run_id": spec.run_id,
                "policy": spec.policy_identifier,
                "N_int": spec.N_int,
                "probe_decisions": metrics["probe_decisions"],
                "actionable_orientations": metrics["actionable_orientation_count"],
                "meek_orientations": metrics["meek_orientation_count"],
                "actionable_coordinates": metrics["actionable_coordinates_constructed"],
                "candidate_evaluations": metrics["candidate_evaluations"],
                "accepted_updates": metrics["accepted_candidates"],
                "estimated_final_kl": metrics["estimated_final_reverse_kl"],
                "true_final_kl": metrics["true_final_reverse_kl"],
                "termination_reason": metrics["termination_reason"],
                "local_state_audit": metrics["local_state_audit"],
                "structural_state_audit": metrics["structural_state_audit"],
                "problem_fingerprint": metrics["problem_fingerprint"],
                "source_sample_fingerprint": metrics["source_sample_fingerprint"],
                "target_sample_fingerprint": metrics["target_sample_fingerprint"],
                "query_sample_audit": metrics["query_sample_audit"],
            }
        )
    shared_matches = 0
    for n_int in N_INTS:
        calibrated_row = next(
            row
            for row in rows
            if row["policy"] == "confidence_calibrated" and row["N_int"] == n_int
        )
        oracle_row = next(
            row
            for row in rows
            if row["policy"] == "oracle_probe_outcome" and row["N_int"] == n_int
        )
        for field in (
            "problem_fingerprint",
            "source_sample_fingerprint",
            "target_sample_fingerprint",
        ):
            if calibrated_row[field] != oracle_row[field]:
                raise AssertionError(
                    f"smoke pairing mismatch for {field}, N_int={n_int}"
                )
        calibrated_queries = {
            (
                item["intervention_id"],
                item["budget_category"],
                item["replication_index"],
            ): item
            for item in calibrated_row["query_sample_audit"]
        }
        oracle_queries = {
            (
                item["intervention_id"],
                item["budget_category"],
                item["replication_index"],
            ): item
            for item in oracle_row["query_sample_audit"]
        }
        for key in calibrated_queries.keys() & oracle_queries.keys():
            shared_matches += 1
            left = calibrated_queries[key]
            right = oracle_queries[key]
            if (
                left["query_seed"],
                left["sample_size"],
                left["sample_fingerprint"],
            ) != (
                right["query_seed"],
                right["sample_size"],
                right["sample_fingerprint"],
            ):
                raise AssertionError(
                    f"shared smoke query mismatch for N_int={n_int}, key={key}"
                )
    payload = {
        "smokes": rows,
        "oracle_candidate_search_reached": any_oracle_search,
        "shared_sample_validation": {
            "paired_cells": len(N_INTS),
            "problem_source_target_mismatches": 0,
            "shared_query_matches": shared_matches,
            "shared_query_mismatches": 0,
        },
    }
    _atomic_json(OUTPUT_ROOT / "preflight" / "smoke_summary.json", payload)
    if not any_oracle_search:
        raise RuntimeError(
            "no oracle smoke reached candidate search; stop before full campaign"
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.write_manifest:
        print(write_manifest())
    if args.run_id:
        print(json.dumps(run_one(args.run_id), sort_keys=True))
    if args.smoke:
        print(json.dumps(smoke(), indent=2, sort_keys=True))
    if args.all:
        write_manifest()
        for index, spec in enumerate(campaign_specs(), 1):
            outcome = run_one(spec.run_id)
            print(
                json.dumps(
                    {
                        "index": index,
                        "total": 60,
                        "run_id": spec.run_id,
                        "status": outcome["status"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if args.validate:
        print(json.dumps(validate_outputs(), indent=2, sort_keys=True))
    if not any((args.write_manifest, args.run_id, args.all, args.smoke, args.validate)):
        parser.error("select at least one action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
