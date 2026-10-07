"""Restart-safe driver for the opt-in held-out top-three finite protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np

from environments.blackbox import (
    EnvironmentQueryLedger,
    FiniteSampleBlackBoxEnvironment,
)
from experiments.executors import build_production_runtime
from experiments.finite_sample_adaptive_evidence import (
    campaign_specs as adaptive_specs,
)
from experiments.finite_sample_campaign import (
    BASE_SEEDS,
    COVARIANCE,
    IMPROVEMENT_TOLERANCE,
    SampleCondition,
    _algorithm_config,
    _exact_by_id,
    _kl,
    _sample_fingerprint,
    _stable_hash,
)
from experiments.finite_sample_campaign import campaign_specs as focused_specs
from experiments.finite_sample_probe_ablation import KeyedFiniteSampleEnvironment
from experiments.heldout_shortlist import (
    HeldoutShortlistConfig,
    run_heldout_shortlist,
)
from experiments.provenance import problem_provenance
from graphs.noise import noise_variance_summary
from graphs.problem_gen import PublicProblemView
from matching.coordinate_search import FiniteCoordinateSearchSettings
from matching.empirical_moments import EmpiricalMomentsProvider
from matching.heldout_improvement import (
    MULTIPLICITY_METHOD,
    OVERALL_ALPHA,
    SHORTLIST_SIZE,
    VALIDATION_LOOKS,
    HeldoutImprovementSettings,
)
from matching.sample_roles import (
    KeyedTargetValidationSampler,
    RoleSeparatedValidationEnvironment,
)
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation

PROTOCOL_ID = "finite_sample_shortlist_heldout_v1"
CAMPAIGN_ID = "finite_sample_shortlist_heldout_p08_v1"
SCHEMA_VERSION = 1
BOOTSTRAP_REPLICATES = 999
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/finite_sample_shortlist_heldout_v1"
FOCUSED_ROOT = REPOSITORY_ROOT / "outputs/finite_sample_focused_v1"
LOCAL_ROOT = REPOSITORY_ROOT / "outputs/finite_sample_local_evidence_ablation_v1"
ADAPTIVE_ROOT = REPOSITORY_ROOT / "outputs/finite_sample_adaptive_evidence_v1"


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_focused_reference(base_seed: int) -> Path:
    matches = []
    for path in sorted((FOCUSED_ROOT / "results").glob("*.json")):
        row = json.loads(path.read_text())
        metrics = row["metrics"]
        if (
            row["base_seed"] == base_seed
            and metrics["N_obs"] == 2000
            and metrics["N_target"] == 2000
            and metrics["N_int"] == 2000
        ):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"expected one focused reference for seed {base_seed}.")
    return matches[0]


def _find_adaptive_reference(base_seed: int) -> Path:
    spec = next(
        item
        for item in adaptive_specs()
        if item.base_seed == base_seed and item.sample_cap == 8000
    )
    path = ADAPTIVE_ROOT / "results" / f"{spec.run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing adaptive reference {path}.")
    return path


def _exact_for_seed(base_seed: int):
    focused = next(
        item
        for item in focused_specs()
        if item.base_seed == base_seed
        and item.N_obs == 2000
        and item.N_target == 2000
        and item.N_int == 2000
    )
    return _exact_by_id(focused.exact_run_id), focused


@dataclass(frozen=True)
class HeldoutRunSpec:
    run_id: str
    base_seed: int
    paired_problem_id: str
    exact_run_id: str
    exact_result_path: str
    max_validation_cap: int
    sample_seeds: dict[str, int]
    focused_reference_path: str
    focused_reference_checksum: str
    adaptive_8000_reference_path: str
    adaptive_8000_reference_checksum: str
    fingerprint: str


@lru_cache(maxsize=1)
def campaign_specs() -> tuple[HeldoutRunSpec, ...]:
    specs = []
    for base_seed in BASE_SEEDS:
        exact, focused = _exact_for_seed(base_seed)
        focused_path = _find_focused_reference(base_seed)
        adaptive_path = _find_adaptive_reference(base_seed)
        for cap in VALIDATION_LOOKS:
            scientific = {
                "campaign": CAMPAIGN_ID,
                "protocol": PROTOCOL_ID,
                "schema": SCHEMA_VERSION,
                "base_seed": base_seed,
                "paired_problem_id": focused.paired_problem_id,
                "exact_run_id": exact.run_id,
                "training_sizes": {"target": 2000, "current": 2000, "candidate": 2000},
                "shortlist_size": SHORTLIST_SIZE,
                "shortlist_rule": "top3_distinct_nodes_then_second_kind_v1",
                "kind_order": ["shift", "log_scale"],
                "validation_looks": list(VALIDATION_LOOKS),
                "max_validation_cap": cap,
                "overall_alpha": OVERALL_ALPHA,
                "alpha_test": OVERALL_ALPHA / 9.0,
                "multiplicity_method": MULTIPLICITY_METHOD,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "acceptance_margin": IMPROVEMENT_TOLERANCE,
                "focused_reference_checksum": _checksum(focused_path),
                "adaptive_reference_checksum": _checksum(adaptive_path),
            }
            fingerprint = _stable_hash(scientific)
            specs.append(
                HeldoutRunSpec(
                    run_id=f"heldout-cap{cap}-{fingerprint[:18]}",
                    base_seed=base_seed,
                    paired_problem_id=focused.paired_problem_id,
                    exact_run_id=exact.run_id,
                    exact_result_path=focused.exact_result_path,
                    max_validation_cap=cap,
                    sample_seeds=dict(focused.sample_seeds),
                    focused_reference_path=str(focused_path),
                    focused_reference_checksum=_checksum(focused_path),
                    adaptive_8000_reference_path=str(adaptive_path),
                    adaptive_8000_reference_checksum=_checksum(adaptive_path),
                    fingerprint=fingerprint,
                )
            )
    if len(specs) != 30 or len({item.run_id for item in specs}) != 30:
        raise RuntimeError("held-out campaign must contain exactly 30 unique specs.")
    return tuple(sorted(specs, key=lambda item: (item.max_validation_cap, item.base_seed)))


def _reference_checksums() -> dict[str, str]:
    paths = {
        "focused_manifest": FOCUSED_ROOT / "manifest.json",
        "local_manifest": LOCAL_ROOT / "manifest.json",
        "adaptive_manifest": ADAPTIVE_ROOT / "manifest.json",
    }
    return {name: _checksum(path) for name, path in paths.items()}


def manifest_payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_identifier": CAMPAIGN_ID,
        "protocol_identifier": PROTOCOL_ID,
        "output_root": str(OUTPUT_ROOT),
        "protocol_isolation": "new_opt_in_runner; all old protocol defaults unchanged",
        "shortlist": {
            "q": SHORTLIST_SIZE,
            "rule_identifier": "top3_distinct_nodes_then_second_kind_v1",
            "kind_order": ["shift", "log_scale"],
            "tuning_permitted": False,
        },
        "data_role_policy": {
            "training_namespace": f"{PROTOCOL_ID}:training",
            "validation_namespace": f"{PROTOCOL_ID}:validation",
            "training_validation_collision_permitted": False,
        },
        "validation": {
            "looks": list(VALIDATION_LOOKS),
            "overall_alpha": OVERALL_ALPHA,
            "alpha_test": OVERALL_ALPHA / 9.0,
            "one_sided_confidence_level": 1.0 - OVERALL_ALPHA / 9.0,
            "multiplicity_method": MULTIPLICITY_METHOD,
            "bootstrap": "shared_target_nonparametric_refit_oas",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "acceptance_margin": IMPROVEMENT_TOLERANCE,
            "anytime_valid_claim": False,
        },
        "base_seeds": list(BASE_SEEDS),
        "authoritative_reference_checksums": _reference_checksums(),
        "runs": [asdict(item) for item in campaign_specs()],
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
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
            artifacts = tuple((OUTPUT_ROOT / "results").glob("*.json"))
            if artifacts:
                raise RuntimeError("existing held-out manifest is incompatible.")
            _atomic_json(path, payload)
        return path
    _atomic_json(path, payload)
    return path


def _frozen_seed_overrides(path: Path) -> dict[tuple[str, str, int], int]:
    row = json.loads(path.read_text())
    overrides: dict[tuple[str, str, int], int] = {}
    for query in row["metrics"]["query_history"]:
        key = (
            str(query["intervention_id"]),
            str(query["budget_category"]),
            int(query["replication_index"]),
        )
        if query["query_seed"] is None or key in overrides:
            raise RuntimeError("invalid authoritative query seed history.")
        overrides[key] = int(query["query_seed"])
    return overrides


def _config(exact) -> HeldoutShortlistConfig:
    old = _algorithm_config(
        exact,
        SampleCondition(2000, 2000, 2000, ("heldout_training",)),
    )
    search = replace(
        old.finite_search,
        proposal_mode="training_point",
    )
    if not isinstance(search, FiniteCoordinateSearchSettings):
        raise TypeError("training search configuration has wrong type.")
    return HeldoutShortlistConfig(
        final_support_budget=old.final_support_budget,
        finite_search=search,
        heldout=HeldoutImprovementSettings(
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            acceptance_margin=IMPROVEMENT_TOLERANCE,
        ),
        covariance=COVARIANCE,
        local_weights=old.local_weights,
        pairwise_neighborhood=old.pairwise_neighborhood,
        training_current_sample_size=2000,
        max_steps=old.max_steps or 20,
        allowed_coordinate_kinds=old.allowed_coordinate_kinds,
    )


def _validation_rows_by_candidate(events: Sequence[dict[str, object]]):
    for event in events:
        for comparison in event["comparisons"]:
            yield event, comparison


def execute(spec: HeldoutRunSpec) -> dict[str, object]:
    exact = _exact_by_id(spec.exact_run_id)
    runtime = build_production_runtime(exact)
    problem = runtime.problem
    seeds = spec.sample_seeds
    source_samples = problem.ground_truth.scm.sample_interventional(
        problem.public.current_intervention,
        2000,
        np.random.default_rng(seeds["source_observational"]),
    )
    training_target_samples = problem.ground_truth.scm.sample_interventional(
        problem.ground_truth.generating_intervention,
        2000,
        np.random.default_rng(seeds["target"]),
    )
    focused = json.loads(Path(spec.focused_reference_path).read_text())
    if (
        _sample_fingerprint(source_samples)
        != focused["metrics"]["source_sample_fingerprint"]
        or _sample_fingerprint(training_target_samples)
        != focused["metrics"]["target_sample_fingerprint"]
    ):
        raise RuntimeError("training samples differ from authoritative paired samples.")
    moments = EmpiricalMomentsProvider(COVARIANCE)
    target_estimate = moments.estimate(training_target_samples)
    public_problem = PublicProblemView(
        observational_samples=source_samples,
        target_mean=target_estimate.mean,
        target_covariance=target_estimate.regularized_covariance,
        current_intervention=problem.public.current_intervention,
        algorithm_seeds=problem.public.algorithm_seeds,
    )

    training_ledger = EnvironmentQueryLedger(None, None)
    training_environment = KeyedFiniteSampleEnvironment(
        problem.ground_truth.scm,
        instance_id=f"{spec.paired_problem_id}:training",
        ledger=training_ledger,
        seed_namespace=f"{PROTOCOL_ID}:training:{spec.paired_problem_id}",
        overrides=_frozen_seed_overrides(Path(spec.focused_reference_path)),
    )
    validation_ledger = EnvironmentQueryLedger(None, None)
    validation_base = FiniteSampleBlackBoxEnvironment(
        problem.ground_truth.scm,
        instance_id=f"{spec.paired_problem_id}:validation",
        ledger=validation_ledger,
    )
    validation_environment = RoleSeparatedValidationEnvironment(
        validation_base,
        namespace=f"{PROTOCOL_ID}:validation_environment",
        paired_problem_id=spec.paired_problem_id,
    )
    target_sampler = KeyedTargetValidationSampler(
        lambda size, rng: problem.ground_truth.scm.sample_interventional(
            problem.ground_truth.generating_intervention,
            size,
            rng,
        ),
        namespace=f"{PROTOCOL_ID}:validation_target",
        paired_problem_id=spec.paired_problem_id,
    )
    graph = cpdag_from_true_dag_for_simulation(problem.ground_truth.scm.dag)
    config = _config(exact)
    started = perf_counter()
    operational = run_heldout_shortlist(
        public_problem,
        graph,
        training_environment,
        validation_environment,
        target_sampler,
        config,
        paired_problem_id=spec.paired_problem_id,
        max_validation_cap=spec.max_validation_cap,
        training_seed=seeds["environment_root"],
        bootstrap_seed=seeds["bootstrap"],
    )
    runtime_seconds = perf_counter() - started

    initial_mean, initial_covariance = problem.current_moments()
    true_initial = _kl(
        problem.public.target_mean,
        problem.public.target_covariance,
        initial_mean,
        initial_covariance,
    )
    final_mean, final_covariance = problem.ground_truth.scm.interventional_moments(
        operational.final_intervention
    )
    true_final = _kl(
        problem.public.target_mean,
        problem.public.target_covariance,
        final_mean,
        final_covariance,
    )
    update_evaluation = []
    prior = problem.public.current_intervention
    prior_mean, prior_cov = problem.ground_truth.scm.interventional_moments(prior)
    prior_kl = gaussian_kl_target_to_candidate(
        problem.public.target_mean,
        problem.public.target_covariance,
        prior_mean,
        prior_cov,
    )
    for step, intervention in enumerate(operational.accepted_interventions):
        mean, covariance = problem.ground_truth.scm.interventional_moments(intervention)
        candidate_kl = gaussian_kl_target_to_candidate(
            problem.public.target_mean,
            problem.public.target_covariance,
            mean,
            covariance,
        )
        update_evaluation.append(
            {
                "step": step,
                "scope": "hidden_population_evaluation_only",
                "true_kl_before": prior_kl,
                "true_kl_after": candidate_kl,
                "true_improvement": prior_kl - candidate_kl,
                "increased_true_kl": candidate_kl > prior_kl,
            }
        )
        prior_kl = candidate_kl

    training_audit = []
    for item in training_environment.sample_audit:
        training_audit.append(
            {
                **item,
                "role": "training_current"
                if item["budget_category"] == "baseline"
                else "training_candidate",
                "namespace": f"{PROTOCOL_ID}:training:{spec.paired_problem_id}",
            }
        )
    observations_by_role = {
        "training_target": 2000,
        "training_current": sum(
            int(item["sample_size"])
            for item in training_audit
            if item["role"] == "training_current"
        ),
        "training_candidate": sum(
            int(item["sample_size"])
            for item in training_audit
            if item["role"] == "training_candidate"
        ),
        "validation_target": sum(
            int(item["sample_size"]) for item in target_sampler.sample_audit
        ),
        "validation_current": sum(
            int(item["sample_size"])
            for item in validation_environment.sample_audit
            if item["role"] == "validation_current"
        ),
        "validation_candidate": sum(
            int(item["sample_size"])
            for item in validation_environment.sample_audit
            if item["role"] == "validation_candidate"
        ),
    }
    validation_comparisons = list(
        _validation_rows_by_candidate(operational.validation_events)
    )
    provenance = problem_provenance(problem).as_dict()
    noise = {
        "noise_variance_regime": getattr(exact, "noise_variance_regime", "homoskedastic"),
        "noise_variances": problem.ground_truth.scm.noise_variances.tolist(),
        **noise_variance_summary(problem.ground_truth.scm.noise_variances),
    }
    metrics: dict[str, object] = {
        "shortlist_configuration": {
            "q": SHORTLIST_SIZE,
            "rule_identifier": "top3_distinct_nodes_then_second_kind_v1",
            "kind_order": ["shift", "log_scale"],
            "distinct_node_first_pass": True,
        },
        "training_validation_namespace_policy": {
            "training": f"{PROTOCOL_ID}:training:{spec.paired_problem_id}",
            "validation_environment": f"{PROTOCOL_ID}:validation_environment",
            "validation_target": f"{PROTOCOL_ID}:validation_target",
            "collision_permitted": False,
        },
        "training_target": {
            "role": "training_target",
            "sample_size": 2000,
            "sample_seed": seeds["target"],
            "sample_fingerprint": _sample_fingerprint(training_target_samples),
            "covariance_metadata": target_estimate.metadata(),
        },
        "training_query_sample_audit": training_audit,
        "validation_environment_sample_audit": validation_environment.sample_audit,
        "validation_target_sample_audit": target_sampler.sample_audit,
        "shortlist_events": list(operational.shortlist_events),
        "training_coordinate_search_traces": list(
            operational.training_search_events
        ),
        "validation_events": list(operational.validation_events),
        "max_validation_cap": spec.max_validation_cap,
        "validation_looks": list(VALIDATION_LOOKS),
        "overall_alpha": OVERALL_ALPHA,
        "alpha_test": OVERALL_ALPHA / 9.0,
        "one_sided_confidence_level": 1.0 - OVERALL_ALPHA / 9.0,
        "multiplicity_method": MULTIPLICITY_METHOD,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "acceptance_margin": IMPROVEMENT_TOLERANCE,
        "inferential_look_policy": (
            "prespecified repeated held-out bootstrap comparisons; not anytime-valid"
        ),
        "accepted_background_identifiers": list(
            operational.training_background_ids
        ),
        "accepted_updates": operational.accepted_updates,
        "accepted_at_least_one_coordinate": operational.accepted_updates > 0,
        "rejected_rounds": operational.rejected_rounds,
        "unresolved_rounds": operational.unresolved_rounds,
        "termination_reason": operational.termination_reason,
        "final_intervention": {
            "shift": operational.final_intervention.shift.tolist(),
            "scale": operational.final_intervention.scale.tolist(),
        },
        "observations_by_role": observations_by_role,
        "total_observations": sum(observations_by_role.values()),
        "training_query_accounting": training_ledger.snapshot(),
        "validation_query_accounting": validation_ledger.snapshot(),
        "training_environment_batches": len(training_audit),
        "validation_environment_batches": len(
            validation_environment.sample_audit
        ),
        "candidate_environment_batches": sum(
            item["role"] in {"training_candidate", "validation_candidate"}
            for item in (*training_audit, *validation_environment.sample_audit)
        ),
        "candidate_comparisons": len(validation_comparisons),
        "runtime_seconds": runtime_seconds,
        "hidden_population_evaluation": {
            "scope": "evaluation_only_not_available_to_operational_runner",
            "true_initial_reverse_kl": true_initial,
            "true_final_reverse_kl": true_final,
            "true_relative_kl_reduction": (
                (true_initial - true_final) / true_initial if true_initial > 0.0 else 0.0
            ),
            "accepted_update_evaluations": update_evaluation,
            "accepted_update_increased_true_kl": any(
                item["increased_true_kl"] for item in update_evaluation
            ),
        },
        "problem_provenance": provenance,
        "problem_fingerprint": runtime.problem_fingerprint,
        "paired_problem_id": spec.paired_problem_id,
        "sample_seeds": dict(seeds),
        "sample_provenance_fingerprint": _stable_hash(
            {
                "training_target": _sample_fingerprint(training_target_samples),
                "training_queries": training_audit,
                "validation_environment": validation_environment.sample_audit,
                "validation_target": target_sampler.sample_audit,
            }
        ),
        **noise,
    }
    metrics["result_fingerprint"] = _stable_hash(
        {
            "run_fingerprint": spec.fingerprint,
            "termination": operational.termination_reason,
            "final_intervention": metrics["final_intervention"],
            "shortlists": metrics["shortlist_events"],
            "searches": metrics["training_coordinate_search_traces"],
            "validation": metrics["validation_events"],
            "sample_provenance": metrics["sample_provenance_fingerprint"],
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "campaign_identifier": CAMPAIGN_ID,
        "protocol_identifier": PROTOCOL_ID,
        "run_id": spec.run_id,
        "run_fingerprint": spec.fingerprint,
        "paired_problem_id": spec.paired_problem_id,
        "base_seed": spec.base_seed,
        "max_validation_cap": spec.max_validation_cap,
        "exact_reference": {
            "run_id": spec.exact_run_id,
            "path": spec.exact_result_path,
            "checksum": _checksum(Path(spec.exact_result_path)),
        },
        "archived_finite_references": {
            "fixed_confidence": {
                "path": spec.focused_reference_path,
                "checksum": spec.focused_reference_checksum,
            },
            "adaptive_8000": {
                "path": spec.adaptive_8000_reference_path,
                "checksum": spec.adaptive_8000_reference_checksum,
            },
            "manifest_checksums": _reference_checksums(),
        },
        "spec": asdict(spec),
        "metrics": metrics,
    }


def result_path(run_id: str) -> Path:
    return OUTPUT_ROOT / "results" / f"{run_id}.json"


def failure_path(run_id: str) -> Path:
    return OUTPUT_ROOT / "failures" / f"{run_id}.json"


def run_one(run_id: str) -> dict[str, object]:
    write_manifest()
    spec = next((item for item in campaign_specs() if item.run_id == run_id), None)
    if spec is None:
        raise ValueError(f"unknown held-out run ID {run_id}.")
    completed = result_path(run_id)
    failure = failure_path(run_id)
    if completed.exists():
        row = json.loads(completed.read_text())
        if (
            row.get("run_fingerprint") != spec.fingerprint
            or row.get("status") != "completed"
            or failure.exists()
        ):
            raise RuntimeError(f"incompatible or duplicate artifact for {run_id}.")
        return {"status": "skipped_completed", "run_id": run_id}
    try:
        row = execute(spec)
        _atomic_json(completed, row)
        return row
    except Exception as error:
        _atomic_json(
            failure,
            {
                "status": "failed",
                "run_id": run_id,
                "run_fingerprint": spec.fingerprint,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def validate_outputs(*, cap: int | None = None) -> dict[str, object]:
    selected = [
        item for item in campaign_specs() if cap is None or item.max_validation_cap == cap
    ]
    missing = []
    corrupt = []
    valid = 0
    reference_checksum_errors = []
    for spec in selected:
        path = result_path(spec.run_id)
        if not path.exists():
            missing.append(spec.run_id)
            continue
        try:
            row = json.loads(path.read_text())
            metrics = row["metrics"]
            if (
                row["status"] != "completed"
                or row["run_fingerprint"] != spec.fingerprint
                or row["protocol_identifier"] != PROTOCOL_ID
                or metrics["shortlist_configuration"]["q"] != 3
                or not np.isclose(metrics["alpha_test"], 0.05 / 9.0)
                or metrics["multiplicity_method"] != MULTIPLICITY_METHOD
            ):
                raise ValueError("schema or protocol invariant failed")
            for reference in row["archived_finite_references"].values():
                if not isinstance(reference, dict) or "path" not in reference:
                    continue
                if _checksum(Path(reference["path"])) != reference["checksum"]:
                    reference_checksum_errors.append(spec.run_id)
            valid += 1
        except Exception as error:  # noqa: BLE001
            corrupt.append({"run_id": spec.run_id, "error": str(error)})
    failures = [
        path.name for path in (OUTPUT_ROOT / "failures").glob("*.json")
    ] if (OUTPUT_ROOT / "failures").exists() else []
    return {
        "expected_jobs": len(selected),
        "valid_completed_jobs": valid,
        "missing_jobs": missing,
        "corrupt_results": corrupt,
        "failure_files": failures,
        "reference_checksum_errors": reference_checksum_errors,
        "cap_filter": cap,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--one-row-smoke", action="store_true")
    parser.add_argument("--cap-2000-smoke", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--approve-full-campaign", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-cap", type=int, choices=VALIDATION_LOOKS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.run_id:
        print(json.dumps(run_one(args.run_id), indent=2, sort_keys=True))
    elif args.one_row_smoke:
        spec = next(
            item
            for item in campaign_specs()
            if item.base_seed == BASE_SEEDS[0] and item.max_validation_cap == 2000
        )
        print(json.dumps(run_one(spec.run_id), indent=2, sort_keys=True))
    elif args.cap_2000_smoke:
        for spec in campaign_specs():
            if spec.max_validation_cap == 2000:
                row = run_one(spec.run_id)
                print(json.dumps({"run_id": spec.run_id, "status": row["status"]}))
    elif args.all:
        if not args.approve_full_campaign:
            raise RuntimeError(
                "the 30-row campaign requires explicit --approve-full-campaign"
            )
        for spec in campaign_specs():
            row = run_one(spec.run_id)
            print(json.dumps({"run_id": spec.run_id, "status": row["status"]}))
    if args.validate or args.validate_cap is not None:
        print(json.dumps(validate_outputs(cap=args.validate_cap), indent=2, sort_keys=True))
    if not any(
        (
            args.run_id,
            args.one_row_smoke,
            args.cap_2000_smoke,
            args.all,
            args.validate,
            args.validate_cap is not None,
        )
    ):
        _parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
