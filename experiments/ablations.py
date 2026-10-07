"""Declarative ablation specifications and deterministic RunnerConfig generation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from itertools import product
from math import ceil
from pathlib import Path
from typing import Any

from graphs.interventions import INTERVENTION_SEMANTICS

from .executors import _validate_run_spec
from .runner import (
    RunnerConfig,
    RunnerDimensions,
    aggregate_result_payloads,
    config_from_json,
    expand_run_specs,
    runner_config_fingerprint,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT = REPOSITORY_ROOT / "ablation_configs" / "specs"
GENERATED_ROOT = REPOSITORY_ROOT / "launch_configs" / "ablations"
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "ablations"
PAPER_FAMILIES = ("scaling", "intervention", "algorithm", "information", "robustness")
ADDITIONAL_FAMILIES = (
    "focused_density_sweep",
    "graph_algorithm",
    "graph_intervention",
    "graph_action_class",
    "support_budget_curve",
)
FAMILIES = (*PAPER_FAMILIES, *ADDITIONAL_FAMILIES)
CAMPAIGN_DESIGN_IDENTIFIER = "dense_er_campaign_v1"
TIERS = ("smoke", "development", "paper")
SCHEMA_VERSION = 1

_INFORMATION_REGIMES = {
    "known_support_oracle": "known_target_support",
    "known_dag_oracle": "known_dag_and_coefficients",
    "idealized_unknown_graph_oracle": "unknown_graph_oracle_active_probing",
    "non_oracle_noiseless_supplied_cpdag": "supplied_exact_cpdag_active_probing",
    "non_oracle_noiseless_estimated_cpdag": "estimated_cpdag_active_probing",
    "exact_nonrealisable": "exact_nonrealisable_active_probing",
}
_V2_MODES = frozenset(
    {
        "non_oracle_noiseless_supplied_cpdag",
        "non_oracle_noiseless_estimated_cpdag",
        "exact_nonrealisable",
    }
)


def load_ablation_spec(family: str) -> dict[str, Any]:
    """Load and structurally validate one checked-in declarative specification."""
    if family not in FAMILIES:
        raise ValueError(f"unsupported ablation family {family!r}.")
    path = SPEC_ROOT / f"{family}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "campaign_family",
        "reference_setting",
        "varied_dimensions",
        "fixed_dimensions",
        "validity_constraints",
        "seeds",
        "output_root",
        "protocol",
        "expected_run_counts",
        "tiers",
        "future_only_dimensions",
    }
    if not isinstance(raw, dict) or not required <= set(raw):
        raise ValueError(f"{path} is missing required ablation-spec fields.")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{path} has unsupported schema_version.")
    if raw["campaign_family"] != family:
        raise ValueError(f"{path} campaign_family does not match its filename.")
    if raw["protocol"] != "exact_noiseless_protocol_v2":
        raise ValueError(f"{path} must preserve exact_noiseless_protocol_v2.")
    if set(raw["seeds"]) != set(TIERS) or set(raw["tiers"]) != set(TIERS):
        raise ValueError(f"{path} must define smoke, development, and paper tiers.")
    expected_seed_counts = raw.get(
        "expected_seed_counts",
        {"smoke": 1, "development": 3, "paper": 10},
    )
    for tier in TIERS:
        expected_seed_count = int(expected_seed_counts[tier])
        seeds = raw["seeds"][tier]
        if not isinstance(seeds, list) or len(seeds) != expected_seed_count:
            raise ValueError(
                f"{path} {tier} tier must define {expected_seed_count} seeds."
            )
        if len(set(seeds)) != len(seeds) or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        ):
            raise ValueError(
                f"{path} {tier} seeds must be unique nonnegative integers."
            )
    return raw


def expand_ablation_cells(
    spec: dict[str, Any], tier: str
) -> tuple[dict[str, Any], ...]:
    """Expand one focused family without mixing incompatible mode combinations."""
    if tier not in TIERS:
        raise ValueError(f"unsupported ablation tier {tier!r}.")
    family = spec["campaign_family"]
    reference = dict(spec["reference_setting"])
    tier_spec = spec["tiers"][tier]
    cells: list[dict[str, Any]] = []
    if family == "scaling":
        for graph, n_nodes, rho in product(
            tier_spec["graph_conditions"],
            tier_spec["node_counts"],
            tier_spec["support_ratios"],
        ):
            cells.append(
                {
                    **reference,
                    "graph_family": graph["family"],
                    "edge_probability": graph["edge_probability"],
                    "n_nodes": n_nodes,
                    "rho": rho,
                }
            )
    elif family == "intervention":
        if reference["intervention_composition"] != "mixed":
            raise ValueError("intervention requires realizable mixed targets.")
        for magnitude, multiplier in product(
            tier_spec["magnitude_regimes"],
            tier_spec["final_coordinate_budget_multipliers"],
        ):
            cells.append(
                {
                    **reference,
                    "intervention_magnitude_regime": magnitude,
                    "final_coordinate_budget_multiplier": multiplier,
                }
            )
    elif family == "focused_density_sweep":
        if reference["intervention_composition"] != "mixed":
            raise ValueError("focused_density_sweep requires a mixed target.")
        if (
            reference.get("target_coordinate_scheme")
            != "nested_er_shared_mixed_support"
        ):
            raise ValueError(
                "focused_density_sweep requires nested ER shared mixed support."
            )
        for edge_probability in tier_spec["edge_probabilities"]:
            cells.append({**reference, "edge_probability": edge_probability})
    elif family in {"algorithm", "information"}:
        for variant in tier_spec["variants"]:
            cells.append({**reference, **variant})
    elif family == "graph_algorithm":
        for graph, variant in product(
            tier_spec["graph_conditions"],
            tier_spec["variants"],
        ):
            graph_family = graph["family"]
            algorithm_variant = variant["algorithm_variant"]
            cells.append(
                {
                    **reference,
                    **variant,
                    "graph_family": graph_family,
                    "edge_probability": graph["edge_probability"],
                    "cell_slug": f"{graph_family}-{algorithm_variant}",
                }
            )
    elif family == "graph_intervention":
        if reference["intervention_composition"] != "mixed":
            raise ValueError("graph_intervention requires realizable mixed targets.")
        for graph, magnitude, variant in product(
            tier_spec["graph_conditions"],
            tier_spec["magnitude_regimes"],
            tier_spec["variants"],
        ):
            graph_family = graph["family"]
            algorithm_variant = variant["algorithm_variant"]
            cells.append(
                {
                    **reference,
                    **variant,
                    "graph_family": graph_family,
                    "edge_probability": graph["edge_probability"],
                    "intervention_magnitude_regime": magnitude,
                    "cell_slug": (f"{graph_family}-{magnitude}-{algorithm_variant}"),
                }
            )
    elif family == "graph_action_class":
        for graph, action, variant in product(
            tier_spec["graph_conditions"],
            tier_spec["action_conditions"],
            tier_spec["variants"],
        ):
            graph_family = graph["family"]
            algorithm_variant = variant["algorithm_variant"]
            action_variant = action["action_variant"]
            cells.append(
                {
                    **reference,
                    **action,
                    **variant,
                    "graph_family": graph_family,
                    "edge_probability": graph["edge_probability"],
                    "cell_slug": (
                        f"{graph_family}-{action_variant}-{algorithm_variant}"
                    ),
                }
            )
    elif family == "support_budget_curve":
        n_nodes = int(reference["n_nodes"])
        target_k = ceil(float(reference["rho"]) * n_nodes)
        if target_k != n_nodes:
            raise ValueError(
                "support_budget_curve requires all-node target support k=n."
            )
        if reference["intervention_composition"] != "mixed":
            raise ValueError(
                "support_budget_curve requires a source-safe mixed target."
            )
        if (
            reference.get("target_coordinate_scheme")
            != "source_shift_nonsource_scale_full_support"
        ):
            raise ValueError(
                "support_budget_curve requires source shifts and non-source scales."
            )
        if reference.get("intervention_action_class") != "shift_scale":
            raise ValueError(
                "support_budget_curve must retain source-safe shift/scale search."
            )
        edge_probabilities = tuple(tier_spec["edge_probabilities"])
        if edge_probabilities != (0.2, 0.4, 0.6, 0.8):
            raise ValueError(
                "support_budget_curve requires ER densities 0.2, 0.4, 0.6, and 0.8."
            )
        for edge_probability, budget in product(
            edge_probabilities, tier_spec["final_coordinate_budgets"]
        ):
            if isinstance(budget, bool) or not isinstance(budget, int):
                raise TypeError("support-budget values must be integers.")
            if not 1 <= budget <= 2 * n_nodes:
                raise ValueError(
                    "support-budget values must lie between one and 2*n_nodes."
                )
            cells.append(
                {
                    **reference,
                    "edge_probability": edge_probability,
                    "final_coordinate_budget": budget,
                    "final_coordinate_budget_multiplier": budget / target_k,
                    "cell_slug": (
                        f"density-p{edge_probability:.1f}-budget-b{budget:02d}"
                    ),
                }
            )
    elif family == "robustness":
        for coefficient, magnitude, target in product(
            tier_spec["coefficient_regimes"],
            tier_spec["magnitude_regimes"],
            tier_spec["target_variants"],
        ):
            cells.append(
                {
                    **reference,
                    **target,
                    "coefficient_regime": coefficient,
                    "intervention_magnitude_regime": magnitude,
                }
            )
    else:  # pragma: no cover - load_ablation_spec constrains families
        raise ValueError(f"no builder for ablation family {family!r}.")
    canonical = sorted(cells, key=_canonical_json)
    if len({_canonical_json(cell) for cell in canonical}) != len(canonical):
        raise ValueError(f"{family}/{tier} contains duplicate scientific cells.")
    return tuple(canonical)


def build_ablation_configs(
    spec: dict[str, Any],
    tier: str,
) -> tuple[tuple[str, RunnerConfig], ...]:
    """Build and executor-validate deterministic one-cell RunnerConfigs."""
    family = spec["campaign_family"]
    seeds = tuple(spec["seeds"][tier])
    cells = expand_ablation_cells(spec, tier)
    configs: list[tuple[str, RunnerConfig]] = []
    for index, cell in enumerate(cells, start=1):
        cell_id = f"{index:03d}-{_cell_slug(cell)}"
        config = _runner_config_for_cell(family, tier, cell_id, cell, seeds)
        fingerprint = runner_config_fingerprint(config)
        metadata = dict(config.ablation_metadata or {})
        metadata["config_fingerprint"] = fingerprint
        config = RunnerConfig(
            **{
                **asdict(config),
                "dimensions": config.dimensions,
                "ablation_metadata": metadata,
            }
        )
        if runner_config_fingerprint(config) != fingerprint:
            raise RuntimeError("ablation config fingerprint was not stable.")
        for run_spec in expand_run_specs(config):
            _validate_run_spec(run_spec)
        configs.append((cell_id, config))
    expected = int(spec["expected_run_counts"][tier])
    actual = sum(len(expand_run_specs(config)) for _, config in configs)
    if actual != expected:
        raise ValueError(
            f"{family}/{tier} expected {expected} runs but generated {actual}."
        )
    return tuple(configs)


def generation_preview(family: str, tier: str) -> dict[str, Any]:
    """Return deterministic counts and fingerprints without writing files."""
    spec = load_ablation_spec(family)
    configs = build_ablation_configs(spec, tier)
    return {
        "campaign_family": family,
        "tier": tier,
        "config_count": len(configs),
        "run_count": sum(len(expand_run_specs(config)) for _, config in configs),
        "paired_problem_count": len(
            {
                run.paired_problem_id
                for _, config in configs
                for run in expand_run_specs(config)
            }
        ),
        "config_fingerprints": {
            cell_id: runner_config_fingerprint(config) for cell_id, config in configs
        },
    }


def write_ablation_configs(family: str, tier: str) -> Path:
    """Atomically materialize generated configs plus a completeness manifest."""
    spec = load_ablation_spec(family)
    configs = build_ablation_configs(spec, tier)
    destination = GENERATED_ROOT / family / tier
    destination.mkdir(parents=True, exist_ok=True)
    expected_names = {f"config-{cell_id}.json" for cell_id, _ in configs}
    existing_names = {path.name for path in destination.glob("config-*.json")}
    stale = sorted(existing_names - expected_names)
    if stale:
        raise RuntimeError(
            f"{destination} contains stale generated configs {stale}; "
            "move them aside instead of reusing or deleting automatically."
        )
    entries = []
    for cell_id, config in configs:
        path = destination / f"config-{cell_id}.json"
        _atomic_json(path, asdict(config))
        entries.append(
            {
                "cell_id": cell_id,
                "path": _portable_path(path),
                "config_fingerprint": runner_config_fingerprint(config),
                "output_directory": config.output_directory,
                "run_count": len(expand_run_specs(config)),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "intervention_semantics": INTERVENTION_SEMANTICS,
        "campaign_design_identifier": CAMPAIGN_DESIGN_IDENTIFIER,
        "campaign_family": family,
        "tier": tier,
        "spec_fingerprint": _fingerprint(spec),
        "config_count": len(configs),
        "expected_run_count": sum(entry["run_count"] for entry in entries),
        "configs": entries,
    }
    manifest_path = destination / "generation_manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


def validate_generation_manifest(family: str, tier: str) -> dict[str, Any]:
    """Reject incomplete, edited, missing, or extra generated config sets."""
    directory = GENERATED_ROOT / family / tier
    manifest_path = directory / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("intervention_semantics") != INTERVENTION_SEMANTICS
        or manifest.get("campaign_design_identifier") != CAMPAIGN_DESIGN_IDENTIFIER
    ):
        raise RuntimeError(f"stale intervention semantics: {manifest_path}")
    entries = manifest.get("configs")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{manifest_path} contains no generated configs.")
    declared_paths = {REPOSITORY_ROOT / entry["path"] for entry in entries}
    actual_paths = set(directory.glob("config-*.json"))
    if declared_paths != actual_paths:
        raise RuntimeError("generated ablation config set is incomplete or has extras.")
    total = 0
    for entry in entries:
        path = REPOSITORY_ROOT / entry["path"]
        config = config_from_json(path)
        fingerprint = runner_config_fingerprint(config)
        if fingerprint != entry.get("config_fingerprint"):
            raise RuntimeError(f"generated config fingerprint mismatch: {path}")
        runs = len(expand_run_specs(config))
        if runs != entry.get("run_count"):
            raise RuntimeError(f"generated config run-count mismatch: {path}")
        total += runs
    if total != manifest.get("expected_run_count"):
        raise RuntimeError("generation manifest is not run-count complete.")
    return manifest


def generated_config_paths(family: str, tier: str) -> tuple[Path, ...]:
    """Return the validated deterministic generated-config sequence."""
    manifest = validate_generation_manifest(family, tier)
    return tuple(REPOSITORY_ROOT / entry["path"] for entry in manifest["configs"])


def validate_output_completeness(family: str, tier: str) -> dict[str, int]:
    """Validate manifests and require one durable artifact for every run ID."""
    completed = 0
    failed = 0
    expected_total = 0
    for config_path in generated_config_paths(family, tier):
        config = config_from_json(config_path)
        fingerprint = runner_config_fingerprint(config)
        output = REPOSITORY_ROOT / config.output_directory
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("intervention_semantics") != INTERVENTION_SEMANTICS
            or manifest.get("campaign_design_identifier") != CAMPAIGN_DESIGN_IDENTIFIER
        ):
            raise RuntimeError(f"stale output semantics: {manifest_path}")
        if manifest.get("config_fingerprint") != fingerprint:
            raise RuntimeError(f"stale output manifest fingerprint: {manifest_path}")
        runs = manifest.get("runs")
        if not isinstance(runs, list):
            raise TypeError(f"{manifest_path} contains no run list.")
        run_ids = {run.get("run_id") for run in runs if isinstance(run, dict)}
        if len(run_ids) != len(runs) or not all(
            isinstance(value, str) for value in run_ids
        ):
            raise ValueError(f"{manifest_path} has invalid or duplicate run IDs.")
        expected_total += len(run_ids)
        results = {path.stem: path for path in (output / "results").glob("*.json")}
        failures = {path.stem: path for path in (output / "failures").glob("*.json")}
        if set(results) | set(failures) != run_ids:
            raise RuntimeError(f"{output} is not manifest-complete.")
        if set(results) & set(failures):
            raise RuntimeError(f"{output} has duplicate result/failure artifacts.")
        for path in (*results.values(), *failures.values()):
            artifact = json.loads(path.read_text(encoding="utf-8"))
            if (
                artifact.get("intervention_semantics") != INTERVENTION_SEMANTICS
                or artifact.get("campaign_design_identifier")
                != CAMPAIGN_DESIGN_IDENTIFIER
            ):
                raise RuntimeError(f"stale output artifact semantics: {path}")
            if artifact.get("config_fingerprint") != fingerprint:
                raise RuntimeError(f"stale output artifact fingerprint: {path}")
        completed += len(results)
        failed += len(failures)
    return {
        "expected_runs": expected_total,
        "completed_runs": completed,
        "failed_runs": failed,
    }


def aggregate_ablation_family(
    family: str, tier: str
) -> tuple[Path, Path, dict[str, object]]:
    """Write family-level grouped JSON/CSV after completeness validation."""
    completeness = validate_output_completeness(family, tier)
    payloads = []
    failure_names = []
    failure_payloads = []
    for config_path in generated_config_paths(family, tier):
        config = config_from_json(config_path)
        output = REPOSITORY_ROOT / config.output_directory
        payloads.extend(
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((output / "results").glob("*.json"))
        )
        failure_names.extend(
            str(path.relative_to(REPOSITORY_ROOT))
            for path in sorted((output / "failures").glob("*.json"))
        )
        failure_payloads.extend(
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((output / "failures").glob("*.json"))
        )
    aggregate = aggregate_result_payloads(
        payloads,
        failure_files=failure_names,
        failure_payloads=failure_payloads,
        bootstrap_replicates=1_000,
        bootstrap_seed=0,
    )
    aggregate.update(
        {
            "campaign_family": family,
            "tier": tier,
            "manifest_completeness": completeness,
        }
    )
    output_root = OUTPUT_ROOT / family / tier
    json_path = output_root / "grouped_aggregate.json"
    csv_path = output_root / "grouped_aggregate.csv"
    _atomic_json(json_path, aggregate)
    rows = []
    for group in aggregate["grouped_ablation_summaries"]:
        row = {
            **group["dimensions"],
            "group_id": group["group_id"],
            "run_count": group["run_count"],
            "expected_run_count": group["expected_run_count"],
            "completed_result_count": group["completed_result_count"],
            "unexpected_execution_failures": group["unexpected_execution_failures"],
            "scientific_matching_successes": group["scientific_matching_successes"],
            "scientific_matching_failures": group["scientific_matching_failures"],
            "orientation_conflict_frequency": group["orientation_conflict_frequency"],
            "execution_success_rate": group["execution_success_rate"],
            "scientific_success_rate": group["scientific_success_rate"],
            "termination_reason_frequencies": json.dumps(
                group["termination_reason_frequencies"], sort_keys=True
            ),
        }
        for metric, summary in group["metric_summaries"].items():
            row[f"{metric}_mean"] = summary["mean"]
        rows.append(row)
    _atomic_csv(csv_path, rows)
    return json_path, csv_path, aggregate


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        fieldnames = sorted({key for row in rows for key in row})
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _runner_config_for_cell(
    family: str,
    tier: str,
    cell_id: str,
    cell: dict[str, Any],
    seeds: tuple[int, ...],
) -> RunnerConfig:
    mode = cell["experiment_mode"]
    protocol = (
        "exact_noiseless_protocol_v2" if mode in _V2_MODES else "exact_protocol_v1"
    )
    provider = cell["initial_cpdag_provider"]
    output = f"outputs/ablations/{family}/{tier}/cells/{cell_id}"
    group_values = {
        "graph_family": cell["graph_family"],
        "edge_probability": cell.get("edge_probability"),
        "node_count": cell["n_nodes"],
        "support_ratio": cell["rho"],
        "intervention_composition": cell["intervention_composition"],
        "intervention_action_class": cell.get(
            "intervention_action_class", "shift_scale"
        ),
        "action_variant": cell.get("action_variant", "shift_scale_2k"),
        "magnitude_regime": cell["intervention_magnitude_regime"],
        "policy": cell["policy"],
        "algorithm_variant": cell.get("algorithm_variant", cell["policy"]),
        "information_regime": _INFORMATION_REGIMES[mode],
        "coefficient_regime": cell["coefficient_regime"],
        "noise_variance_regime": cell.get(
            "noise_variance_regime", "homoskedastic"
        ),
        "final_coordinate_budget_multiplier": cell[
            "final_coordinate_budget_multiplier"
        ],
        "local_loss_weights": cell.get("local_loss_weights", [1.0, 1.0, 1.0]),
    }
    if "target_coordinate_scheme" in cell:
        group_values["target_coordinate_scheme"] = cell["target_coordinate_scheme"]
    if "final_coordinate_budget" in cell:
        group_values["final_coordinate_budget"] = cell["final_coordinate_budget"]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "campaign_family": family,
        "tier": tier,
        "cell_id": cell_id,
        "protocol": protocol,
        "campaign_design_identifier": CAMPAIGN_DESIGN_IDENTIFIER,
        "group_values": group_values,
    }
    return RunnerConfig(
        dimensions=RunnerDimensions(
            experiment_modes=(mode,),
            graph_families=(cell["graph_family"],),
            node_counts=(cell["n_nodes"],),
            coefficient_regimes=(cell["coefficient_regime"],),
            noise_variance_regimes=(
                cell.get("noise_variance_regime", "homoskedastic"),
            ),
            target_families=(cell["target_family"],),
            target_realisability=(cell["target_is_realisable"],),
            support_ratios=(cell["rho"],),
            policies=(cell["policy"],),
            initial_cpdag_providers=(provider,),
            sample_sizes=(500 if provider == "estimated_pc_cpdag" else 0,),
            edge_probabilities=(cell.get("edge_probability"),),
            intervention_compositions=(cell["intervention_composition"],),
            intervention_magnitude_regimes=(cell["intervention_magnitude_regime"],),
            final_coordinate_budget_multipliers=(
                cell["final_coordinate_budget_multiplier"],
            ),
            coordinate_selection_rules=("best_improvement",),
        ),
        base_seeds=seeds,
        query_budget=None if protocol == "exact_noiseless_protocol_v2" else 200,
        observation_budget=0,
        probe_target_limit="n_nodes",
        probe_amplitudes=(0.5,),
        local_loss_weights=tuple(cell.get("local_loss_weights", [1.0, 1.0, 1.0])),
        source_equivalence_absolute=0.1,
        source_equivalence_relative=0.05,
        source_confidence_level=0.95,
        candidate_acceptance_confidence=0.95,
        covariance_estimator="oas",
        backend=str(cell.get("backend", "numpy")),
        device=str(cell.get("device", "cpu")),
        output_directory=output,
        campaign_design_identifier=CAMPAIGN_DESIGN_IDENTIFIER,
        couple_edge_probability=family
        in {
            "focused_density_sweep",
            "support_budget_curve",
        },
        couple_intervention_composition=False,
        couple_intervention_magnitude=family in {"intervention", "graph_intervention"},
        bootstrap_aggregate_replicates=1_000,
        coordinate_coarse_grid_size=9,
        coordinate_max_values_per_coordinate=30,
        improvement_tolerance=float(cell.get("improvement_tolerance", 1e-10)),
        ablation_metadata=metadata,
    )


def _cell_slug(cell: dict[str, Any]) -> str:
    label = str(
        cell.get("cell_slug")
        or cell.get("algorithm_variant")
        or cell.get("information_variant")
        or "-".join(
            (
                str(cell["graph_family"]),
                f"n{cell['n_nodes']}",
                f"rho{cell['rho']}",
                str(cell["intervention_composition"]),
                str(cell["intervention_magnitude_regime"]),
                f"b{cell['final_coordinate_budget_multiplier']}",
                str(cell["coefficient_regime"]),
                str(cell["target_family"]),
            )
        )
    )
    return "".join(
        character if character.isalnum() else "-" for character in label
    ).strip("-")[:100]


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
