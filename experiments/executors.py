"""Production executors bridging restartable runner specs to exact assemblies.

The bridge is intentionally strict.  It supports only the six engineering-ready
exact/noiseless modes and rejects every unmapped runner value before an
environment can execute a query.  Finite-sample modes remain gated by the
calibration/readiness workflow and are deliberately absent from the executor map.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from math import ceil, exp
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx
import numpy as np

from environments.blackbox import EnvironmentQueryLedger, ExactMomentBlackBoxEnvironment
from graphs.DAG_gen import DEFAULT_ERDOS_RENYI_EDGE_PROBABILITY
from graphs.interventions import INTERVENTION_SEMANTICS, ShiftScaleIntervention
from graphs.noise import (
    generate_noise_variances,
    noise_variance_summary,
    validate_noise_variance_regime,
)
from graphs.nonrealisable_targets import (
    correlated_noise_target,
    external_gaussian_target,
    graph_perturbed_target,
    nonproportional_coefficient_target,
)
from graphs.problem_gen import (
    GeneratedProblem,
    PublicProblemView,
    SimulationGroundTruth,
)
from graphs.registry import GRAPH_GENERATORS, generate_graph
from graphs.scm import LinearGaussianSCM, sample_coefficients
from graphs.seeds import AlgorithmSeeds, SimulationSeeds
from graphs.support import coordinate_support_masks
from graphs.target_realisability import gaussian_dag_parameters
from matching.coordinate_search import (
    CoordinateSearchBounds,
    ExactCoordinateSearchSettings,
    FiniteCoordinateSearchSettings,
)
from matching.empirical_decisions import EmpiricalAcceptanceSettings
from matching.empirical_moments import (
    CovarianceEstimatorSettings,
    EmpiricalMomentsProvider,
)
from matching.local_losses import LocalLossWeights, PublicPairwiseNeighborhood
from matching.scheduler import SchedulerResult, SchedulerSettings
from matching.structure_learning import (
    EstimatedPCCPDAGProvider,
    PCStructureSettings,
    SuppliedExactCPDAGProvider,
)
from matching.types import FinalSupportBudget, ParameterCoordinate
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation

from .assemblies import RunnableExperimentAssembly
from .modes import ExperimentMode
from .noiseless_unknown_graph_pilot import (
    NoiselessPilotConfig,
    NoiselessPilotResult,
    PilotRunSpec,
)
from .nonoracle import (
    NonOracleExperimentConfig,
    NonOracleExperimentResult,
    ProbeRecoverySettings,
)
from .nonrealisable import NonRealisableExperimentResult
from .protocols import (
    ExperimentProtocol,
    exact_noiseless_protocol_v2,
    exact_protocol_v1,
    protocol_by_version,
)
from .provenance import ProblemProvenance, problem_provenance
from .registry import (
    PRODUCTION_REGISTRY,
    ProductionRunConfiguration,
    ProductionRuntimeComponents,
    assemble_production_run,
)
from .runner import ExperimentReliability, RunOutcome, RunSpec

SUPPORTED_PRODUCTION_MODES: tuple[ExperimentMode, ...] = (
    "known_support_oracle",
    "known_dag_oracle",
    "idealized_unknown_graph_oracle",
    "non_oracle_noiseless_supplied_cpdag",
    "non_oracle_noiseless_estimated_cpdag",
    "exact_nonrealisable",
)

_PILOT_GRAPH_FAMILIES = frozenset(GRAPH_GENERATORS)
_PILOT_POLICIES = frozenset({"random", "exact_minmaxc", "clique_tree"})
_FIXED_POLICIES: dict[str, str] = {
    "known_support_oracle": "known_support_oracle",
    "known_dag_oracle": "known_dag_oracle",
    "non_oracle_noiseless_supplied_cpdag": "non_oracle_noiseless",
    "non_oracle_noiseless_estimated_cpdag": "non_oracle_noiseless",
    "exact_nonrealisable": "non_oracle_noiseless",
}
_STRUCTURE_BY_MODE: dict[str, tuple[str, ...]] = {
    "known_support_oracle": ("true_dag_and_coefficients",),
    "known_dag_oracle": ("true_dag_and_coefficients",),
    "idealized_unknown_graph_oracle": ("oracle_exact_cpdag",),
    "non_oracle_noiseless_supplied_cpdag": ("supplied_exact_public_cpdag",),
    "non_oracle_noiseless_estimated_cpdag": ("estimated_pc_cpdag",),
    "exact_nonrealisable": (
        "supplied_exact_public_cpdag",
        "estimated_pc_cpdag",
    ),
}
_COEFFICIENT_REGIMES = {
    "weak": (0.05, 0.25),
    "moderate": (0.25, 0.75),
    "strong": (0.75, 1.25),
}
_TARGET_MAGNITUDE_REGIMES = {
    "weak": (0.25, np.log(1.1)),
    "moderate": (0.5, np.log(1.25)),
    "strong": (1.0, np.log(1.75)),
}
_INTERVENTION_COMPOSITIONS = frozenset({"shift_only", "scale_only", "mixed"})
_TARGET_COORDINATE_SCHEMES = frozenset(
    {
        "composition_default",
        "nested_er_shared_mixed_support",
        "source_shift_nonsource_scale_full_support",
    }
)
_NESTED_ER_TARGET_BASE_PROBABILITY = 0.2
_INTERVENTION_ACTION_CLASSES = {
    "shift_only": ("shift",),
    "scale_only": ("log_scale",),
    "shift_scale": ("shift", "log_scale"),
}
_DENSITY_GRAPH_FAMILIES = frozenset({"erdos_renyi", "moralized_erdos_renyi"})
_EXACT_SOURCE_EQUIVALENCE = (0.1, 0.05)
# Exact moment responses require only a floating-point zero test. Reusing the
# finite-sample equivalence margin can erase real weak ancestral responses and
# turn a non-source into a source.
_EXACT_SOURCE_NUMERICAL_TOLERANCE = (1e-8, 1e-8)
_COEFFICIENT_TARGET_PROVENANCE_FIELDS = (
    "selected_node",
    "selected_parent",
    "selected_node_indegree",
    "incoming_parents",
    "source_incoming_coefficients",
    "allowed_mixed_incoming_coefficients",
    "final_target_incoming_coefficients",
    "multiplier",
    "source_residual_variances",
    "target_residual_variances",
    "markov_with_respect_to_source_dag",
    "residual_variances_preserved",
    "selected_node_positive_proportionality",
    "node_wise_shift_scale_realisable",
    "intercept_coefficient_realisable",
    "intervention_family_realisability_classification",
    "minimum_required_coordinate_support",
    "minimum_required_intercept_coefficient_support",
    "configured_support_budget",
    "exactly_one_extra_coefficient_modified",
    "other_incoming_coefficient_unchanged",
    "no_scale_coordinate_on_root",
)
_UNBUDGETED_EXACT_MODES = frozenset(
    {
        "non_oracle_noiseless_supplied_cpdag",
        "non_oracle_noiseless_estimated_cpdag",
        "exact_nonrealisable",
    }
)


@dataclass(frozen=True)
class ProductionExecutionRuntime:
    """One fully validated, still-unexecuted production run."""

    spec: RunSpec
    configuration: ProductionRunConfiguration
    problem: GeneratedProblem
    runtime_components: ProductionRuntimeComponents
    assembly: RunnableExperimentAssembly[object]
    problem_fingerprint: str
    problem_provenance: ProblemProvenance
    target_public_metadata: dict[str, object]


def build_production_runtime(spec: RunSpec) -> ProductionExecutionRuntime:
    """Build and validate one exact production assembly before its first query."""
    _validate_run_spec(spec)
    protocol = _protocol_for_spec(spec)
    configuration = _production_configuration(spec, protocol)
    problem, target_metadata = _generated_problem(spec, configuration, protocol)
    provenance = problem_provenance(problem)
    values: dict[str, object] = {
        "effective_runtime_settings": configuration.effective_runtime_settings(),
        "problem": problem,
        "problem_fingerprint": provenance.problem_fingerprint,
        "problem_provenance": provenance,
        "paired_problem_id": spec.paired_problem_id,
        "run_id": spec.run_id,
    }

    if spec.experiment_mode in {"known_support_oracle", "known_dag_oracle"}:
        values.update(_oracle_runtime_values(spec, configuration, problem))
    elif spec.experiment_mode == "idealized_unknown_graph_oracle":
        values.update(_idealized_runtime_values(spec, configuration))
    else:
        values.update(_nonoracle_runtime_values(spec, configuration, problem, protocol))
        if spec.experiment_mode == "exact_nonrealisable":
            values["evaluation_problem"] = problem

    runtime_components = ProductionRuntimeComponents(values)
    assembly = assemble_production_run(configuration, runtime_components)
    if assembly.mode != spec.experiment_mode:
        raise ValueError(
            "runtime assembly mode mismatch: "
            f"expected {spec.experiment_mode!r}, got {assembly.mode!r}."
        )
    return ProductionExecutionRuntime(
        spec=spec,
        configuration=configuration,
        problem=problem,
        runtime_components=runtime_components,
        assembly=assembly,
        problem_fingerprint=provenance.problem_fingerprint,
        problem_provenance=provenance,
        target_public_metadata=target_metadata,
    )


def execute_production_run(spec: RunSpec) -> RunOutcome:
    """Build and execute exactly one runner spec through the production registry."""
    started = perf_counter()
    runtime = build_production_runtime(spec)
    raw_result = runtime.assembly.run()
    elapsed = perf_counter() - started
    metrics, termination_reason, reliability = _result_payload(
        runtime,
        raw_result,
        elapsed,
    )
    return RunOutcome(
        metrics=metrics,
        termination_reason=termination_reason,
        success=reliability.numerical_failures == 0,
        reliability=reliability,
        protocol_version=runtime.configuration.protocol_version,
        resolved_protocol=protocol_by_version(
            runtime.configuration.protocol_version
        ).resolved_values(),
    )


EXPERIMENT_EXECUTORS = {
    mode: execute_production_run for mode in SUPPORTED_PRODUCTION_MODES
}


def _protocol_for_spec(spec: RunSpec) -> ExperimentProtocol:
    """Select unbudgeted query semantics only for exact non-oracle modes."""
    if spec.experiment_mode in _UNBUDGETED_EXACT_MODES:
        return exact_noiseless_protocol_v2
    return exact_protocol_v1


def _validate_run_spec(spec: RunSpec) -> None:
    if not isinstance(spec, RunSpec):
        raise TypeError("production executors require a runner RunSpec.")
    if spec.experiment_mode not in SUPPORTED_PRODUCTION_MODES:
        raise ValueError(
            f"production executor does not support mode {spec.experiment_mode!r}; "
            "finite-sample modes require scoped calibration readiness."
        )
    validate_noise_variance_regime(spec.noise_variance_regime)
    if (
        spec.noise_variance_regime != "homoskedastic"
        and spec.noise_variance_seed is None
    ):
        raise ValueError("heterogeneous production specs require a noise_variance_seed.")
    if (
        spec.noise_variance_regime != "homoskedastic"
        and not spec.cross_noise_regime_pair_id
    ):
        raise ValueError(
            "heterogeneous production specs require a cross_noise_regime_pair_id."
        )
    if not spec.run_id or not spec.paired_problem_id:
        raise ValueError("run_id and paired_problem_id must be nonempty.")
    if spec.graph_family not in GRAPH_GENERATORS:
        raise ValueError(f"unsupported production graph family {spec.graph_family!r}.")
    if spec.graph_family in _DENSITY_GRAPH_FAMILIES:
        if spec.edge_probability is not None and not 0.0 < spec.edge_probability <= 1.0:
            raise ValueError("edge_probability must lie in (0, 1].")
    elif spec.edge_probability is not None:
        raise ValueError("edge_probability is valid only for density graph families.")
    if spec.coefficient_regime not in _COEFFICIENT_REGIMES:
        raise ValueError(
            f"unmapped coefficient regime {spec.coefficient_regime!r}; "
            f"supported regimes: {tuple(_COEFFICIENT_REGIMES)}."
        )
    if spec.intervention_composition not in _INTERVENTION_COMPOSITIONS:
        raise ValueError("unsupported intervention_composition.")
    if spec.target_coordinate_scheme not in _TARGET_COORDINATE_SCHEMES:
        raise ValueError("unsupported target_coordinate_scheme.")
    if (
        spec.target_coordinate_scheme == "source_shift_nonsource_scale_full_support"
        and not (
            spec.intervention_composition == "mixed"
            and spec.intervention_action_class == "shift_scale"
            and spec.k == spec.n_nodes
        )
    ):
        raise ValueError(
            "source-safe full-support mixed targets require mixed composition, "
            "shift-scale actions, and k=n."
        )
    if spec.target_coordinate_scheme == "nested_er_shared_mixed_support" and not (
        spec.graph_family == "erdos_renyi"
        and spec.intervention_composition == "mixed"
        and spec.intervention_action_class == "shift_scale"
        and spec.edge_probability is not None
        and spec.edge_probability >= _NESTED_ER_TARGET_BASE_PROBABILITY
    ):
        raise ValueError(
            "nested ER shared mixed targets require ER graphs at p>=0.2, "
            "mixed composition, and shift-scale actions."
        )
    if spec.intervention_action_class not in _INTERVENTION_ACTION_CLASSES:
        raise ValueError("unsupported intervention_action_class.")
    if spec.intervention_action_class != "shift_scale" and spec.experiment_mode not in {
        "idealized_unknown_graph_oracle",
        "non_oracle_noiseless_supplied_cpdag",
        "non_oracle_noiseless_estimated_cpdag",
    }:
        raise ValueError(
            "restricted intervention action classes are supported only for "
            "public or idealized unknown-graph methods."
        )
    if spec.intervention_magnitude_regime not in _TARGET_MAGNITUDE_REGIMES:
        raise ValueError("unsupported intervention_magnitude_regime.")
    if isinstance(spec.n_nodes, bool) or not isinstance(spec.n_nodes, int):
        raise TypeError("n_nodes must be an integer.")
    if spec.n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")
    if not 0.0 < float(spec.rho) <= 1.0:
        raise ValueError("rho must lie in (0, 1].")
    expected_k = ceil(float(spec.rho) * spec.n_nodes)
    if spec.k != expected_k:
        raise ValueError(f"k must equal ceil(rho*n_nodes)={expected_k}.")
    multiplier = spec.final_coordinate_budget_multiplier
    if isinstance(multiplier, bool) or not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("final_coordinate_budget_multiplier must be positive.")
    expected_budget = min(ceil(multiplier * spec.k), 2 * spec.n_nodes)
    if spec.final_parameter_coordinate_budget != expected_budget:
        raise ValueError(
            "final_parameter_coordinate_budget must equal "
            "min(ceil(multiplier*k), 2*n_nodes)."
        )
    if spec.intervention_semantics != INTERVENTION_SEMANTICS:
        raise ValueError("run spec has stale intervention semantics.")
    protocol = _protocol_for_spec(spec)
    if spec.query_budget != protocol.budgets.total_executed_query_budget:
        raise ValueError(
            "exact production query_budget must match the selected protocol: "
            f"{protocol.budgets.total_executed_query_budget}."
        )
    if spec.coordinate_selection_rule != "best_improvement":
        raise ValueError(
            "production currently supports coordinate_selection_rule="
            "'best_improvement' only."
        )
    if spec.coordinate_coarse_grid_size != protocol.coordinate_search.coarse_grid_size:
        raise ValueError("non-canonical coordinate_coarse_grid_size is unsupported.")
    if (
        spec.coordinate_max_values_per_coordinate
        != protocol.coordinate_search.maximum_candidate_values_per_coordinate
    ):
        raise ValueError(
            "non-canonical coordinate_max_values_per_coordinate is unsupported."
        )
    if not np.isfinite(spec.improvement_tolerance) or spec.improvement_tolerance < 0:
        raise ValueError("improvement_tolerance must be finite and nonnegative.")
    if spec.observation_budget != protocol.budgets.total_observation_budget:
        raise ValueError("exact production observation_budget must be zero.")
    if not 0 < spec.probe_target_limit <= spec.n_nodes:
        raise ValueError("probe_target_limit must lie in [1, n_nodes].")
    if (
        protocol.protocol_version == "exact_noiseless_protocol_v2"
        and spec.probe_target_limit != spec.n_nodes
    ):
        raise ValueError("exact/noiseless probe_target_limit must resolve to n_nodes.")
    amplitudes = tuple(spec.probe_amplitudes)
    if len(amplitudes) != 1 or not np.isfinite(amplitudes[0]) or amplitudes[0] == 0:
        raise ValueError(
            "exact production requires one finite nonzero probe amplitude."
        )
    weights = tuple(spec.local_loss_weights)
    if len(weights) != 3 or any(
        not np.isfinite(value) or value < 0 for value in weights
    ):
        raise ValueError("local_loss_weights must be three finite nonnegative values.")
    if not any(value > 0 for value in weights):
        raise ValueError("at least one local loss weight must be positive.")
    if (
        spec.source_equivalence_absolute,
        spec.source_equivalence_relative,
    ) != _EXACT_SOURCE_EQUIVALENCE:
        raise ValueError(
            "exact runner source-equivalence metadata must be (0.1, 0.05)."
        )
    if spec.source_confidence_level != 0.95:
        raise ValueError("exact runner source_confidence_level must be 0.95.")
    if spec.candidate_acceptance_confidence != 0.95:
        raise ValueError("exact runner candidate_acceptance_confidence must be 0.95.")
    if spec.covariance_estimator != "oas":
        raise ValueError("exact production covariance_estimator must be 'oas'.")
    numpy_cpu = spec.backend == "numpy" and spec.device == "cpu"
    torch_exact = (
        spec.backend == "torch"
        and spec.device in {"cpu", "cuda"}
        and spec.experiment_mode in _UNBUDGETED_EXACT_MODES
    )
    if not (numpy_cpu or torch_exact):
        raise ValueError(
            "exact production supports numpy/cpu generally and torch/cpu or "
            "torch/cuda for exact black-box modes."
        )
    permitted_structures = _STRUCTURE_BY_MODE[str(spec.experiment_mode)]
    if spec.initial_cpdag_provider not in permitted_structures:
        raise ValueError(
            f"mode {spec.experiment_mode!r} rejects initial_cpdag_provider="
            f"{spec.initial_cpdag_provider!r}; expected one of {permitted_structures}."
        )
    if spec.experiment_mode == "idealized_unknown_graph_oracle":
        if spec.graph_family not in _PILOT_GRAPH_FAMILIES:
            raise ValueError("idealized unknown-graph mode rejects this graph family.")
        if spec.policy not in _PILOT_POLICIES:
            raise ValueError(
                "idealized unknown-graph policy must be random, exact_minmaxc, "
                "or clique_tree."
            )
    else:
        expected_policy = _FIXED_POLICIES[str(spec.experiment_mode)]
        if spec.policy != expected_policy:
            raise ValueError(
                f"mode {spec.experiment_mode!r} requires policy {expected_policy!r}."
            )
    estimated = spec.initial_cpdag_provider == "estimated_pc_cpdag"
    expected_samples = (
        protocol.observational_structure.observational_sample_size if estimated else 0
    )
    if spec.sample_size != expected_samples:
        raise ValueError(
            f"sample_size must be {expected_samples} for provider "
            f"{spec.initial_cpdag_provider!r}."
        )
    nonrealisable = spec.experiment_mode == "exact_nonrealisable"
    if spec.target_is_realisable == nonrealisable:
        raise ValueError("target realisability is incompatible with the selected mode.")
    descriptor = PRODUCTION_REGISTRY[spec.experiment_mode]
    if spec.target_family not in descriptor.target_families:
        raise ValueError("target_family is incompatible with the selected mode.")
    for name, value in asdict(spec.seeds).items():
        if name == "noise_variances" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"seed {name!r} must be a nonnegative integer.")
    if isinstance(spec.base_seed, bool) or not isinstance(spec.base_seed, int):
        raise TypeError("base_seed must be an integer.")
    if spec.base_seed < 0:
        raise ValueError("base_seed must be nonnegative.")


def _production_configuration(
    spec: RunSpec,
    protocol: ExperimentProtocol,
) -> ProductionRunConfiguration:
    descriptor = PRODUCTION_REGISTRY[spec.experiment_mode]
    estimated = spec.initial_cpdag_provider == "estimated_pc_cpdag"
    raw: dict[str, object] = {
        "mode": spec.experiment_mode,
        "intervention_semantics": spec.intervention_semantics,
        "protocol_version": protocol.protocol_version,
        "structure_provider": spec.initial_cpdag_provider,
        "environment_provider": descriptor.permitted_environment_providers[0],
        "local_loss_provider": descriptor.permitted_local_loss_providers[0],
        "source_inference_provider": descriptor.permitted_source_inference_providers[0],
        "orientation_provider": descriptor.permitted_orientation_providers[0],
        "coordinate_search_provider": descriptor.permitted_coordinate_search_providers[
            0
        ],
        "evaluator": descriptor.permitted_evaluators[0],
        "unresolved_state_provider": descriptor.permitted_unresolved_state_providers[0],
        "target_family": spec.target_family,
        "target_is_realisable": spec.target_is_realisable,
        "recover_target_support": spec.experiment_mode == "known_support_oracle",
        "n_nodes": spec.n_nodes,
        "rho": spec.rho,
        "k": spec.k,
        "final_parameter_coordinate_budget": spec.final_parameter_coordinate_budget,
        "query_budget": spec.query_budget,
        "observation_budget": spec.observation_budget,
        "baseline_query_budget": protocol.budgets.baseline_query_budget,
        "probe_query_budget": protocol.budgets.probe_query_budget,
        "candidate_query_budget": protocol.budgets.candidate_query_budget,
        "sample_size": None,
        "observational_sample_size": (
            protocol.observational_structure.observational_sample_size
            if estimated
            else None
        ),
        "probe_amplitude_schedule": tuple(spec.probe_amplitudes),
        "max_probe_targets": spec.probe_target_limit,
        "local_loss_weights": tuple(spec.local_loss_weights),
        "local_loss_tolerance": protocol.unresolved_state.local_loss_tolerance,
        "global_loss_tolerance": protocol.unresolved_state.global_kl_success_tolerance,
        "improvement_tolerance": spec.improvement_tolerance,
        "exploration_seed": spec.seeds.policy,
        "candidate_evaluation_seed": spec.seeds.coordinate_search,
        "environment_seed": spec.seeds.environment_samples,
        "bootstrap_seed": spec.seeds.bootstrap,
        "problem_seed": spec.base_seed,
        "noise_variance_regime": spec.noise_variance_regime,
        "noise_variance_seed": spec.noise_variance_seed,
        "cross_noise_regime_pair_id": spec.cross_noise_regime_pair_id,
        "backend": spec.backend,
        "shift_domain": protocol.coordinate_search.shift_domain,
        "log_scale_domain": protocol.coordinate_search.log_scale_domain,
        "confidence_level": protocol.unresolved_state.global_confidence_level,
        "error_probability": 1.0 - protocol.unresolved_state.global_confidence_level,
        "output_directory": f"runner-managed/{spec.run_id}",
        "device": spec.device,
        "result_schema_version": descriptor.result_schema_version,
    }
    configuration = ProductionRunConfiguration.from_mapping(raw)
    configuration.validate()
    return configuration


def _generated_problem(
    spec: RunSpec,
    configuration: ProductionRunConfiguration,
    protocol: ExperimentProtocol,
) -> tuple[GeneratedProblem, dict[str, object]]:
    graph_rng = np.random.default_rng(spec.seeds.graph)
    shared_support_rng = (
        np.random.default_rng(spec.seeds.graph)
        if spec.target_coordinate_scheme == "nested_er_shared_mixed_support"
        else None
    )
    shared_scale_eligible_nodes: np.ndarray | None = None
    graph_attempt = 0
    while True:
        graph_attempt += 1
        dag = generate_graph(
            spec.graph_family,
            spec.n_nodes,
            graph_rng,
            **_graph_parameters(spec.graph_family, spec.n_nodes, spec.edge_probability),
        )
        if shared_support_rng is not None:
            support_dag = generate_graph(
                "erdos_renyi",
                spec.n_nodes,
                shared_support_rng,
                edge_probability=_NESTED_ER_TARGET_BASE_PROBABILITY,
            )
            shared_scale_eligible_nodes = np.array(
                [node for node, degree in support_dag.in_degree if degree > 0],
                dtype=int,
            )
            eligible_count = int(shared_scale_eligible_nodes.size)
        else:
            eligible_count = sum(degree > 0 for _, degree in dag.in_degree)
        needs_scale = (
            spec.intervention_composition in {"scale_only", "mixed"}
            or spec.couple_intervention_composition
        )
        required_scale_eligible = (
            1
            if spec.target_coordinate_scheme
            == "source_shift_nonsource_scale_full_support"
            else spec.k
        )
        requires_two_parent_node = (
            spec.target_family == "nonproportional_structural_coefficients"
        )
        has_two_parent_node = any(degree >= 2 for _, degree in dag.in_degree)
        if (not needs_scale or eligible_count >= required_scale_eligible) and (
            not requires_two_parent_node or has_two_parent_node
        ):
            break
        if graph_attempt >= 1_000:
            raise ValueError(
                "bounded deterministic graph regeneration could not satisfy "
                "scale eligibility and corrected-target indegree requirements."
            )
    coefficient_min, coefficient_max = _COEFFICIENT_REGIMES[spec.coefficient_regime]
    coefficients = sample_coefficients(
        dag,
        np.random.default_rng(spec.seeds.coefficients),
        coefficient_min,
        coefficient_max,
    )
    noise_variances = generate_noise_variances(
        spec.n_nodes,
        spec.noise_variance_regime,
        spec.noise_variance_seed,
    )
    scm = LinearGaussianSCM(
        dag,
        coefficients,
        noise_variances=noise_variances,
    )
    target_intervention = _target_intervention(
        spec,
        dag,
        eligible_nodes=shared_scale_eligible_nodes,
    )
    if spec.target_is_realisable:
        target_mean, target_covariance = scm.interventional_moments(target_intervention)
        target_metadata: dict[str, object] = {
            "target_family": "realisable",
            "graph_generation_attempt": graph_attempt,
        }
    else:
        target = _nonrealisable_target(spec, scm, target_intervention)
        target_mean, target_covariance = target.mean, target.covariance
        target_metadata = dict(target.public_metadata)
        target_metadata["graph_generation_attempt"] = graph_attempt
    observational_size = protocol.observational_structure.observational_sample_size
    observational_samples = scm.sample_observational(
        observational_size,
        np.random.default_rng(spec.seeds.observational_samples),
    )
    public = PublicProblemView(
        observational_samples=observational_samples,
        target_mean=np.asarray(target_mean),
        target_covariance=np.asarray(target_covariance),
        current_intervention=ShiftScaleIntervention.identity(spec.n_nodes),
        algorithm_seeds=AlgorithmSeeds(
            exploration_seed=configuration.exploration_seed,
            candidate_evaluation_seed=configuration.candidate_evaluation_seed,
        ),
    )
    ground_truth = SimulationGroundTruth(
        scm=scm,
        generating_intervention=target_intervention,
        seeds=SimulationSeeds(
            root_seed=spec.base_seed,
            graph_generation_seed=spec.seeds.graph,
            coefficient_generation_seed=spec.seeds.coefficients,
            observational_sampling_seed=spec.seeds.observational_samples,
            target_intervention_seed=spec.seeds.target,
            target_sampling_seed=spec.seeds.environment_samples,
            noise_variance_seed=(
                spec.noise_variance_seed
                if spec.noise_variance_regime != "homoskedastic"
                else None
            ),
        ),
    )
    return GeneratedProblem(public, ground_truth), target_metadata


def _graph_parameters(
    graph_family: str, n_nodes: int, edge_probability: float | None
) -> dict[str, object]:
    if graph_family == "path":
        return {"permute": False}
    if graph_family == "rooted_tree":
        return {"root": None}
    if graph_family in {"erdos_renyi", "moralized_erdos_renyi"}:
        return {
            "edge_probability": DEFAULT_ERDOS_RENYI_EDGE_PROBABILITY
            if edge_probability is None
            else edge_probability
        }
    if graph_family == "barabasi_albert":
        if n_nodes < 2:
            raise ValueError("barabasi_albert production runs require n_nodes >= 2.")
        return {"attachment_edges": min(2, n_nodes - 1)}
    if graph_family == "complete":
        return {"node_order": None}
    if graph_family == "tree_of_cliques":
        if n_nodes < 2:
            raise ValueError("tree_of_cliques production runs require n_nodes >= 2.")
        return {
            "branching_degree": 2,
            "min_clique_size": 2,
            "max_clique_size": min(4, n_nodes),
        }
    return {}


def _target_intervention(
    spec: RunSpec,
    dag: nx.DiGraph,
    *,
    eligible_nodes: np.ndarray | None = None,
) -> ShiftScaleIntervention:
    """Sample a hidden endogenous intervention under a declared composition."""
    rng = np.random.default_rng(spec.seeds.target)
    target_coordinate_scheme = getattr(
        spec, "target_coordinate_scheme", "composition_default"
    )
    if target_coordinate_scheme == "source_shift_nonsource_scale_full_support":
        if spec.intervention_composition != "mixed" or spec.k != spec.n_nodes:
            raise ValueError(
                "source-safe full-support mixed targets require mixed composition "
                "and k=n."
            )
        roots = np.array(
            [node for node, degree in dag.in_degree if degree == 0], dtype=int
        )
        nonroots = np.array(
            [node for node, degree in dag.in_degree if degree > 0], dtype=int
        )
        if roots.size == 0 or nonroots.size == 0:
            raise ValueError(
                "source-safe mixed targets require at least one source and one "
                "non-source node."
            )
        shift_signs = rng.choice(np.array([-1.0, 1.0]), size=spec.n_nodes)
        scale_signs = rng.choice(np.array([-1.0, 1.0]), size=spec.n_nodes)
        shift_magnitude, log_scale_magnitude = _TARGET_MAGNITUDE_REGIMES[
            spec.intervention_magnitude_regime
        ]
        shift = np.zeros(spec.n_nodes)
        scale = np.ones(spec.n_nodes)
        shift[roots] = shift_magnitude * shift_signs[roots]
        scale[nonroots] = np.exp(log_scale_magnitude * scale_signs[nonroots])
        return ShiftScaleIntervention(shift, scale)
    if target_coordinate_scheme == "nested_er_shared_mixed_support":
        if eligible_nodes is None:
            raise ValueError("nested ER mixed targets require shared eligible nodes.")
        eligible = np.asarray(eligible_nodes, dtype=int)
        current_nonroots = {node for node, degree in dag.in_degree if degree > 0}
        if not set(eligible).issubset(current_nonroots):
            raise ValueError(
                "shared mixed-target support is not scale-eligible in this graph."
            )
    elif (
        spec.intervention_composition == "shift_only"
        and not spec.couple_intervention_composition
    ):
        eligible = np.arange(spec.n_nodes)
    else:
        eligible = np.array(
            [node for node, degree in dag.in_degree if degree > 0],
            dtype=int,
        )
        if eligible.size < spec.k:
            raise ValueError(
                f"graph has {eligible.size} scale-eligible nodes but target requires "
                f"k={spec.k}; regenerate with a deterministic eligible graph."
            )
    nodes = np.sort(rng.choice(eligible, size=spec.k, replace=False))
    shift_signs = rng.choice(np.array([-1.0, 1.0]), size=spec.k)
    scale_signs = rng.choice(np.array([-1.0, 1.0]), size=spec.k)
    shift_magnitude, log_scale_magnitude = _TARGET_MAGNITUDE_REGIMES[
        spec.intervention_magnitude_regime
    ]
    shift = np.zeros(spec.n_nodes)
    scale = np.ones(spec.n_nodes)
    if spec.intervention_composition in {"shift_only", "mixed"}:
        shift[nodes] = shift_magnitude * shift_signs
    if spec.intervention_composition in {"scale_only", "mixed"}:
        scale[nodes] = np.exp(log_scale_magnitude * scale_signs)
    return ShiftScaleIntervention(shift, scale)


def _nonrealisable_target(
    spec: RunSpec,
    scm: LinearGaussianSCM,
    intervention: ShiftScaleIntervention,
) -> Any:
    if spec.n_nodes < 2:
        raise ValueError("non-realisable production targets require n_nodes >= 2.")
    if spec.target_family == "nonproportional_structural_coefficients":
        eligible_nodes = np.asarray(
            sorted(node for node, degree in scm.dag.in_degree if degree >= 2),
            dtype=int,
        )
        if eligible_nodes.size == 0:
            raise ValueError(
                "nonproportional_structural_coefficients requires a node with "
                "at least two parents."
            )
        rng = np.random.default_rng(
            np.random.SeedSequence([spec.seeds.target, spec.seeds.graph, 135])
        )
        selected_node = int(rng.choice(eligible_nodes))
        parents = np.asarray(sorted(scm.dag.predecessors(selected_node)), dtype=int)
        selected_parent = int(rng.choice(parents))
        return nonproportional_coefficient_target(
            scm,
            intervention,
            selected_node=selected_node,
            selected_parent=selected_parent,
            multiplier=1.35,
            configured_support_budget=spec.final_parameter_coordinate_budget,
        )
    if spec.target_family == "graph_perturbed_linear_gaussian":
        alternative_dag, alternative_coefficients = _eligible_graph_perturbation(
            scm,
            intervention,
        )
        alternative = LinearGaussianSCM(
            alternative_dag,
            alternative_coefficients,
            noise_variances=scm.noise_variances,
        )
        return graph_perturbed_target(alternative, intervention)
    correlation = 0.25
    if spec.target_family == "correlated_noise_gaussian":
        standard_deviations = scm.noise_standard_deviations
        correlation_matrix = np.full((spec.n_nodes, spec.n_nodes), correlation)
        np.fill_diagonal(correlation_matrix, 1.0)
        noise = correlation_matrix * np.outer(
            standard_deviations,
            standard_deviations,
        )
        return correlated_noise_target(scm, noise, intervention)
    if spec.target_family == "external_spd_gaussian":
        variances = 1.0 + 0.2 * np.arange(spec.n_nodes)
        standard_deviations = np.sqrt(variances)
        correlation_matrix = np.full((spec.n_nodes, spec.n_nodes), correlation)
        np.fill_diagonal(correlation_matrix, 1.0)
        covariance = correlation_matrix * np.outer(
            standard_deviations,
            standard_deviations,
        )
        return external_gaussian_target(
            np.asarray(intervention.shift) + 0.1,
            covariance,
        )
    raise ValueError(f"unsupported non-realisable target {spec.target_family!r}.")


def _eligible_graph_perturbation(
    scm: LinearGaussianSCM,
    intervention: ShiftScaleIntervention,
) -> tuple[nx.DiGraph, np.ndarray]:
    """Perturb one edge while preserving eligibility of scaled target nodes."""
    scaled_nodes = frozenset(np.flatnonzero(intervention.scale != 1.0).tolist())
    edges = tuple(sorted(scm.dag.edges))
    for removed in (*edges, None):
        for parent in range(scm.dag.number_of_nodes()):
            for child in range(scm.dag.number_of_nodes()):
                if parent == child or (parent, child) in scm.dag.edges:
                    continue
                candidate = nx.DiGraph(scm.dag)
                if removed is not None:
                    candidate.remove_edge(*removed)
                candidate.add_edge(parent, child)
                if not nx.is_directed_acyclic_graph(candidate):
                    continue
                if any(candidate.in_degree(node) == 0 for node in scaled_nodes):
                    continue
                coefficients = np.zeros_like(scm.coefficients)
                for edge_parent, edge_child in candidate.edges:
                    if (edge_parent, edge_child) in scm.dag.edges:
                        coefficients[edge_parent, edge_child] = scm.coefficients[
                            edge_parent,
                            edge_child,
                        ]
                    else:
                        coefficients[edge_parent, edge_child] = 0.5
                return candidate, coefficients
    raise ValueError(
        "could not construct a deterministic graph perturbation that preserves "
        "scale-target eligibility."
    )


def _scheduler_settings(configuration: ProductionRunConfiguration) -> SchedulerSettings:
    return SchedulerSettings(
        loss_tolerance=configuration.local_loss_tolerance,
        global_loss_tolerance=configuration.global_loss_tolerance,
        improvement_tolerance=configuration.improvement_tolerance,
        patience=3,
        max_consecutive_exploration_trials=max(1, configuration.probe_query_budget),
        max_probe_targets=configuration.max_probe_targets,
        scale_min=exp(configuration.log_scale_domain[0]),
        scale_max=exp(configuration.log_scale_domain[1]),
        max_steps=configuration.query_budget,
    )


def _oracle_runtime_values(
    spec: RunSpec,
    configuration: ProductionRunConfiguration,
    problem: GeneratedProblem,
) -> dict[str, object]:
    values: dict[str, object] = {
        "scheduler_settings": _scheduler_settings(configuration),
        "final_support_budget": FinalSupportBudget(
            configuration.final_parameter_coordinate_budget
        ),
        "max_experiment_trials": configuration.query_budget,
        "runner_kwargs": {
            "probe_shift_magnitude": configuration.probe_amplitude_schedule[0],
            "optimizer_max_iterations": 1_000,
        },
    }
    if spec.experiment_mode == "known_support_oracle":
        values["active_coordinates"] = _intervention_coordinates(
            problem.ground_truth.generating_intervention
        )
    return values


def _idealized_runtime_values(
    spec: RunSpec,
    configuration: ProductionRunConfiguration,
) -> dict[str, object]:
    coefficient_min, coefficient_max = _COEFFICIENT_REGIMES[spec.coefficient_regime]
    shift_magnitude, log_scale_magnitude = _TARGET_MAGNITUDE_REGIMES[
        spec.intervention_magnitude_regime
    ]
    config = NoiselessPilotConfig(
        node_counts=(spec.n_nodes,),
        graph_families=(spec.graph_family,),  # type: ignore[arg-type]
        policies=(spec.policy,),  # type: ignore[arg-type]
        seeds=(spec.base_seed,),
        target_support_size=spec.k,
        support_fraction=spec.rho,
        final_parameter_coordinate_budget=(
            configuration.final_parameter_coordinate_budget
        ),
        intervention_composition=spec.intervention_composition,  # type: ignore[arg-type]
        allowed_coordinate_kinds=_INTERVENTION_ACTION_CLASSES[
            spec.intervention_action_class
        ],  # type: ignore[arg-type]
        local_unresolved_threshold=configuration.local_loss_tolerance,
        local_marginal_weight=configuration.local_loss_weights[0],
        local_pairwise_weight=configuration.local_loss_weights[1],
        local_conditional_weight=configuration.local_loss_weights[2],
        target_shift_magnitude=shift_magnitude,
        target_log_scale_magnitude=log_scale_magnitude,
        n_observational_samples=(
            exact_protocol_v1.observational_structure.observational_sample_size
        ),
        edge_probability=(
            spec.edge_probability or DEFAULT_ERDOS_RENYI_EDGE_PROBABILITY
        ),
        coefficient_min_abs=coefficient_min,
        coefficient_max_abs=coefficient_max,
        max_probe_targets=configuration.max_probe_targets,
        max_experiment_trials=configuration.query_budget,
        max_consecutive_exploration_trials=max(1, configuration.probe_query_budget),
        exact_minmaxc_max_nodes=max(1, spec.n_nodes),
        loss_tolerance=configuration.global_loss_tolerance,
        improvement_tolerance=configuration.improvement_tolerance,
        scale_min=exp(configuration.log_scale_domain[0]),
        scale_max=exp(configuration.log_scale_domain[1]),
        optimizer_max_iterations=1_000,
        max_steps=configuration.query_budget,
    )
    pilot_spec = PilotRunSpec(
        n_nodes=spec.n_nodes,
        graph_family=spec.graph_family,  # type: ignore[arg-type]
        policy=spec.policy,  # type: ignore[arg-type]
        seed=spec.base_seed,
    )
    return {"pilot_config": config, "pilot_spec": pilot_spec}


def _nonoracle_runtime_values(
    spec: RunSpec,
    configuration: ProductionRunConfiguration,
    problem: GeneratedProblem,
    protocol: ExperimentProtocol,
) -> dict[str, object]:
    ledger = EnvironmentQueryLedger(
        max_query_batches=configuration.query_budget,
        max_observations=None,
        max_baseline_queries=configuration.baseline_query_budget,
        max_probe_queries=configuration.probe_query_budget,
        max_candidate_queries=configuration.candidate_query_budget,
    )
    environment = ExactMomentBlackBoxEnvironment(
        problem.ground_truth.scm,
        instance_id=spec.paired_problem_id,
        ledger=ledger,
        numerical_configuration={"backend": spec.backend, "device": spec.device},
    )
    structure = _structure_provider(spec, problem, protocol)
    algorithm_config = _nonoracle_algorithm_configuration(spec, configuration, protocol)
    return {
        "problem": problem.public,
        "environment": environment,
        "structure_provider": structure,
        "algorithm_config": algorithm_config,
    }


def _structure_provider(
    spec: RunSpec,
    problem: GeneratedProblem,
    protocol: ExperimentProtocol,
) -> SuppliedExactCPDAGProvider | EstimatedPCCPDAGProvider:
    if spec.initial_cpdag_provider == "supplied_exact_public_cpdag":
        return SuppliedExactCPDAGProvider(
            cpdag_from_true_dag_for_simulation(problem.ground_truth.scm.dag)
        )
    observational = protocol.observational_structure
    return EstimatedPCCPDAGProvider(
        PCStructureSettings(
            significance_level=observational.pc_significance_level,
            max_conditioning_set_size=observational.maximum_conditioning_set_size,
            minimum_effective_sample_size=20,
        ),
        EmpiricalMomentsProvider(
            CovarianceEstimatorSettings(
                observational.covariance.estimator,
                observational.covariance.diagonal_loading,
                observational.covariance.minimum_eigenvalue,
            )
        ),
    )


def _nonoracle_algorithm_configuration(
    spec: RunSpec,
    configuration: ProductionRunConfiguration,
    protocol: ExperimentProtocol,
) -> NonOracleExperimentConfig:
    bounds = CoordinateSearchBounds(
        *configuration.shift_domain,
        *configuration.log_scale_domain,
    )
    observational = protocol.observational_structure
    maximum_values = protocol.coordinate_search.maximum_candidate_values_per_coordinate
    exact_search = ExactCoordinateSearchSettings(
        bounds=bounds,
        coarse_grid_size=protocol.coordinate_search.coarse_grid_size,
        refine=True,
        refinement_max_evaluations=max(
            1,
            maximum_values - protocol.coordinate_search.coarse_grid_size,
        ),
    )
    finite_search = FiniteCoordinateSearchSettings(
        bounds=bounds,
        coarse_grid_size=protocol.coordinate_search.coarse_grid_size,
        initial_sample_size=2,
        max_sample_size_per_candidate=2,
        max_query_batches=maximum_values,
        sample_growth_factor=2.0,
        bootstrap_replicates=2,
        confidence_level=configuration.confidence_level,
        improvement_tolerance=configuration.improvement_tolerance,
    )
    acceptance = EmpiricalAcceptanceSettings(
        initial_candidate_sample_size=2,
        max_sample_size_per_arm=2,
        sample_growth_factor=2.0,
        max_repeat_queries=1,
        bootstrap_replicates=2,
        confidence_level=configuration.confidence_level,
        improvement_tolerance=configuration.improvement_tolerance,
    )
    covariance = CovarianceEstimatorSettings(
        observational.covariance.estimator,
        observational.covariance.diagonal_loading,
        observational.covariance.minimum_eigenvalue,
    )
    return NonOracleExperimentConfig(
        final_support_budget=FinalSupportBudget(
            configuration.final_parameter_coordinate_budget
        ),
        exact_search=exact_search,
        finite_search=finite_search,
        acceptance=acceptance,
        covariance=covariance,
        probe=ProbeRecoverySettings(
            initial_probe_sample_size=2,
            max_probe_sample_size=2,
            sample_growth_factor=2.0,
            max_probe_repeats=1,
            probe_amplitude_schedule=configuration.probe_amplitude_schedule,
            max_probe_targets=configuration.max_probe_targets,
            source_tolerance_absolute=_EXACT_SOURCE_NUMERICAL_TOLERANCE[0],
            source_tolerance_relative=_EXACT_SOURCE_NUMERICAL_TOLERANCE[1],
            source_confidence_level=configuration.confidence_level,
        ),
        local_weights=LocalLossWeights(*configuration.local_loss_weights),
        pairwise_neighborhood=PublicPairwiseNeighborhood(),
        local_loss_tolerance=configuration.local_loss_tolerance,
        global_loss_tolerance=configuration.global_loss_tolerance,
        improvement_tolerance=configuration.improvement_tolerance,
        bootstrap_replicates=2,
        confidence_level=configuration.confidence_level,
        baseline_sample_size=2,
        max_steps=None,
        allowed_coordinate_kinds=_INTERVENTION_ACTION_CLASSES[
            spec.intervention_action_class
        ],  # type: ignore[arg-type]
        patience=2 * configuration.n_nodes + 1,
    )


def _result_payload(
    runtime: ProductionExecutionRuntime,
    result: object,
    elapsed: float,
) -> tuple[dict[str, object], str, ExperimentReliability]:
    if isinstance(result, NonRealisableExperimentResult):
        algorithm_result = result.underlying_result
        metrics = _nonoracle_metrics(runtime, algorithm_result)
        metrics.update(
            {
                "initial_global_reverse_kl": result.initial_reverse_kl,
                "final_global_reverse_kl": result.final_reverse_kl,
                "positive_residual_kl_permitted": result.positive_residual_kl_permitted,
                "best_found_comparator_loss": result.best_found_comparator_loss,
                "algorithm_to_best_found_gap": result.algorithm_to_best_found_gap,
                "fraction_observed_gap_closed": result.fraction_observed_gap_closed,
                "comparator_label": result.comparator_label,
                "comparator_proven_lower_bound": result.comparator_proven_lower_bound,
            }
        )
        termination = result.termination_reason
    elif isinstance(result, NonOracleExperimentResult):
        metrics = _nonoracle_metrics(runtime, result)
        termination = result.termination_reason
    elif isinstance(result, NoiselessPilotResult):
        metrics = _pilot_metrics(runtime, result)
        termination = result.termination_reason
    elif isinstance(result, SchedulerResult):
        metrics = _scheduler_metrics(runtime, result)
        termination = result.termination_reason
    else:
        raise TypeError(
            f"production assembly returned unsupported result {type(result)!r}."
        )

    metrics.update(_common_metrics(runtime, elapsed))
    structure_failures = int(metrics.get("structure_learning_failure_count", 0))
    reliability = ExperimentReliability(
        numerical_failures=int(termination == "numerical_failure"),
        structure_learning_failures=structure_failures,
        query_budget_exhaustions=int(
            termination in {"experiment_budget_exhausted", "query_budget_exhausted"}
        ),
        observation_budget_exhaustions=int(
            termination == "observation_budget_exhausted"
        ),
        support_budget_exhaustions=int(
            termination in {"final_support_budget_exhausted", "final_support_exhausted"}
        ),
        no_action_terminations=int(termination == "no_legal_action"),
        global_mismatch_not_localized=int(
            termination == "global_mismatch_not_localized"
        ),
        inconclusive_terminations=int(termination == "insufficient_evidence"),
        orientation_conflicts=int(metrics.get("orientation_conflict_count", 0) or 0),
    )
    return metrics, termination, reliability


def _common_metrics(
    runtime: ProductionExecutionRuntime,
    elapsed: float,
) -> dict[str, object]:
    spec = runtime.spec
    target_support = (
        sorted(runtime.problem.ground_truth.generating_intervention.targeted_nodes())
        if spec.target_is_realisable
        else None
    )
    target_coordinates = (
        [
            _coordinate_label(coordinate)
            for coordinate in _intervention_coordinates(
                runtime.problem.ground_truth.generating_intervention
            )
        ]
        if spec.target_is_realisable
        else None
    )
    target_shift_count = (
        sum(label.startswith("shift:") for label in target_coordinates)
        if target_coordinates is not None
        else None
    )
    target_scale_count = (
        sum(label.startswith("log_scale:") for label in target_coordinates)
        if target_coordinates is not None
        else None
    )
    scm = runtime.problem.ground_truth.scm
    dag = scm.dag
    edge_count = dag.number_of_edges()
    possible_edge_count = spec.n_nodes * (spec.n_nodes - 1) // 2
    realized_edge_density = (
        edge_count / possible_edge_count if possible_edge_count else 0.0
    )
    in_degrees = np.asarray([degree for _, degree in dag.in_degree], dtype=float)
    coefficients = np.asarray(scm.coefficients, dtype=float)
    observational_covariance = scm.observational_moments()[1]
    target_covariance = np.asarray(
        runtime.problem.public.target_covariance, dtype=float
    )
    target_parameters = gaussian_dag_parameters(
        dag,
        np.asarray(runtime.problem.public.target_mean, dtype=float),
        target_covariance,
    )
    target_residual_covariance = target_parameters.residual_covariance
    target_residual_variances = target_parameters.residual_variances
    residual_off_diagonal = target_residual_covariance - np.diag(
        target_residual_variances
    )
    coefficient_target_provenance = {
        field: runtime.target_public_metadata.get(field)
        for field in _COEFFICIENT_TARGET_PROVENANCE_FIELDS
    }
    coefficient_target_provenance["source_residual_variances"] = (
        scm.noise_variances.tolist()
    )
    coefficient_target_provenance["target_residual_variances"] = (
        target_residual_variances.tolist()
    )
    coefficient_target_provenance["markov_with_respect_to_source_dag"] = bool(
        np.allclose(residual_off_diagonal, 0.0, rtol=1e-8, atol=1e-8)
    )
    coefficient_target_provenance["residual_variances_preserved"] = bool(
        np.allclose(
            target_residual_variances,
            scm.noise_variances,
            rtol=1e-8,
            atol=1e-8,
        )
    )
    return {
        "run_id": spec.run_id,
        "experiment_mode": spec.experiment_mode,
        "graph_information_regime": _graph_information_regime(spec),
        "protocol_version": runtime.configuration.protocol_version,
        "result_schema_version": runtime.configuration.result_schema_version,
        "intervention_semantics": runtime.configuration.intervention_semantics,
        "campaign_design_identifier": spec.campaign_design_identifier,
        "exact_source_classification_tolerance": {
            "absolute": _EXACT_SOURCE_NUMERICAL_TOLERANCE[0],
            "relative": _EXACT_SOURCE_NUMERICAL_TOLERANCE[1],
        },
        "actionable_parameter_coordinate_count": (
            runtime.problem.ground_truth.scm.actionable_parameter_coordinate_count
        ),
        "scale_eligible_node_count": len(
            runtime.problem.ground_truth.scm.scale_eligible_nodes
        ),
        "scale_eligible_nodes": sorted(
            runtime.problem.ground_truth.scm.scale_eligible_nodes
        ),
        "mixed_budget_structurally_insufficient": (
            spec.intervention_composition == "mixed"
            and target_coordinates is not None
            and spec.final_parameter_coordinate_budget < len(target_coordinates)
        ),
        "paired_problem_id": spec.paired_problem_id,
        "cross_noise_regime_pair_id": spec.cross_noise_regime_pair_id,
        "noise_variance_regime": spec.noise_variance_regime,
        "noise_variance_seed": spec.noise_variance_seed,
        "noise_variances": runtime.problem.ground_truth.scm.noise_variances.tolist(),
        **noise_variance_summary(
            runtime.problem.ground_truth.scm.noise_variances
        ),
        "problem_fingerprint": runtime.problem_fingerprint,
        **runtime.problem_provenance.as_dict(),
        "problem_provenance": runtime.problem_provenance.as_dict(),
        # The assembly received the exact object fingerprinted above.  Equality
        # across distinct records is validated before paired analysis.
        "pairing_validated": True,
        "git_commit": _git_commit(),
        "seeds": asdict(spec.seeds),
        "base_seed": spec.base_seed,
        "graph_family": spec.graph_family,
        "edge_probability": spec.edge_probability,
        "edge_count": edge_count,
        "realized_edge_density": realized_edge_density,
        "maximum_in_degree": int(in_degrees.max(initial=0.0)),
        "average_in_degree": float(in_degrees.mean()) if in_degrees.size else 0.0,
        "target_covariance_condition_number": float(np.linalg.cond(target_covariance)),
        "observational_covariance_condition_number": float(
            np.linalg.cond(observational_covariance)
        ),
        "coefficient_norm_summaries": {
            "frobenius": float(np.linalg.norm(coefficients)),
            "spectral": float(np.linalg.norm(coefficients, ord=2)),
            "maximum_absolute": float(np.max(np.abs(coefficients), initial=0.0)),
            "mean_absolute_nonzero": float(
                np.mean(np.abs(coefficients[coefficients != 0.0]))
            )
            if edge_count
            else 0.0,
        },
        "n_nodes": spec.n_nodes,
        "coefficient_regime": spec.coefficient_regime,
        "target_family": spec.target_family,
        "target_is_realisable": spec.target_is_realisable,
        "intervention_composition": spec.intervention_composition,
        "intervention_action_class": spec.intervention_action_class,
        "target_coordinate_scheme": spec.target_coordinate_scheme,
        "intervention_magnitude_regime": spec.intervention_magnitude_regime,
        "target_node_support": target_support,
        "target_parameter_coordinate_support": target_coordinates,
        "target_shift_coordinate_count": target_shift_count,
        "target_scale_coordinate_count": target_scale_count,
        "target_public_metadata": runtime.target_public_metadata,
        **coefficient_target_provenance,
        "rho": spec.rho,
        "k": spec.k,
        "final_parameter_coordinate_budget": spec.final_parameter_coordinate_budget,
        "final_coordinate_budget_multiplier": (spec.final_coordinate_budget_multiplier),
        "coordinate_selection_rule": spec.coordinate_selection_rule,
        "coordinate_coarse_grid_size": spec.coordinate_coarse_grid_size,
        "coordinate_max_values_per_coordinate": (
            spec.coordinate_max_values_per_coordinate
        ),
        "improvement_tolerance": spec.improvement_tolerance,
        "configured_query_budget": spec.query_budget,
        "configured_baseline_query_budget": (
            runtime.configuration.baseline_query_budget
        ),
        "configured_probe_query_budget": runtime.configuration.probe_query_budget,
        "configured_candidate_query_budget": (
            runtime.configuration.candidate_query_budget
        ),
        "configured_observation_budget": spec.observation_budget,
        "backend": spec.backend,
        "device": spec.device,
        "wall_seconds": elapsed,
        "mode_idealizations": _mode_idealizations(spec.experiment_mode),
        "production_configuration": asdict(runtime.configuration),
        "effective_runtime_settings": runtime.configuration.effective_runtime_settings(),
    }


def _scheduler_metrics(
    runtime: ProductionExecutionRuntime,
    result: SchedulerResult,
) -> dict[str, object]:
    problem = runtime.problem
    final_mean, final_covariance = problem.ground_truth.scm.interventional_moments(
        result.final_candidate
    )
    initial_mean, initial_covariance = problem.current_moments()
    ledger = result.budget_ledger
    accepted_support = _intervention_coordinate_labels(result.final_candidate)
    local_losses = tuple(
        record.local_loss_after
        for record in result.history
        if record.local_loss_after is not None
    )
    total_edges = problem.ground_truth.scm.dag.number_of_edges()
    return {
        "initial_global_reverse_kl": gaussian_kl_target_to_candidate(
            problem.public.target_mean,
            problem.public.target_covariance,
            initial_mean,
            initial_covariance,
        ),
        "final_global_reverse_kl": result.final_loss,
        "local_composite_losses": local_losses,
        "mean_error": float(np.linalg.norm(problem.public.target_mean - final_mean)),
        "covariance_error": float(
            np.linalg.norm(problem.public.target_covariance - final_covariance)
        ),
        "accepted_final_support": accepted_support,
        "accepted_final_coordinate_count": len(accepted_support),
        "accepted_shift_coordinate_count": sum(
            label.startswith("shift:") for label in accepted_support
        ),
        "accepted_scale_coordinate_count": sum(
            label.startswith("log_scale:") for label in accepted_support
        ),
        "accepted_candidates": ledger.accepted_matching_trials,
        "rejected_candidates": ledger.rejected_matching_trials,
        "inconclusive_candidates": 0,
        "initial_oriented_edges": total_edges,
        "final_oriented_edges": total_edges,
        "probe_count": ledger.exploration_trials,
        "source_outcomes": [],
        "algorithm_environment_calls": ledger.experiment_trials_used,
        "executed_environment_queries": ledger.experiment_trials_used,
        "total_observations": 0,
        "optimizer_search_evaluations": sum(
            record.phase == "OPTIMIZE" for record in result.history
        ),
        "shift_coordinate_uses": ledger.shift_coordinate_uses,
        "scale_coordinate_uses": ledger.scale_coordinate_uses,
        "structure_learning_failure_count": 0,
        "local_global_surrogate_diagnostics": [],
        "termination_detail": result.termination_reason,
    }


def _pilot_metrics(
    runtime: ProductionExecutionRuntime,
    result: NoiselessPilotResult,
) -> dict[str, object]:
    del runtime
    return {
        "initial_global_reverse_kl": result.initial_gaussian_kl,
        "final_global_reverse_kl": result.final_gaussian_kl,
        "local_composite_losses": (),
        "mean_error": None,
        "covariance_error": None,
        "accepted_final_support": result.accepted_final_parameter_support,
        "accepted_final_coordinate_count": len(result.accepted_final_parameter_support),
        "accepted_shift_coordinate_count": sum(
            label.startswith("shift:")
            for label in result.accepted_final_parameter_support
        ),
        "accepted_scale_coordinate_count": sum(
            label.startswith("log_scale:")
            for label in result.accepted_final_parameter_support
        ),
        "accepted_candidates": len(result.accepted_matching_kl),
        "rejected_candidates": max(
            0,
            result.candidate_evaluation_trials - len(result.accepted_matching_kl),
        ),
        "inconclusive_candidates": 0,
        "initial_oriented_edges": result.initial_oriented_edges,
        "final_oriented_edges": result.final_oriented_edges,
        "probe_count": result.exploration_trials,
        "source_outcomes": [],
        "algorithm_environment_calls": result.total_experiment_trials,
        "executed_environment_queries": result.total_experiment_trials,
        "total_observations": 0,
        "optimizer_search_evaluations": result.candidate_evaluation_trials,
        "shift_coordinate_uses": result.shift_coordinate_uses,
        "scale_coordinate_uses": result.scale_coordinate_uses,
        "directed_shd": None,
        "skeleton_accuracy": None,
        "orientation_accuracy": result.final_oriented_fraction,
        "structure_learning_failure_count": 0,
        "local_global_surrogate_diagnostics": [],
        "metric_availability": {
            "mean_error": "standalone idealized result does not retain final moments",
            "covariance_error": "standalone idealized result does not retain final moments",
        },
        "termination_detail": result.termination_reason,
    }


def _nonoracle_metrics(
    runtime: ProductionExecutionRuntime,
    result: NonOracleExperimentResult,
) -> dict[str, object]:
    problem = runtime.problem
    environment = runtime.runtime_components.values.get("environment")
    if not isinstance(environment, ExactMomentBlackBoxEnvironment):
        raise TypeError("non-oracle runtime is missing its exact environment.")
    backend_diagnostics = dict(environment.backend_diagnostics)
    final_mean, final_covariance = problem.ground_truth.scm.interventional_moments(
        result.final_intervention
    )
    initial_mean, initial_covariance = problem.current_moments()
    source_outcomes = [
        {
            "step": record.step,
            "background_id": record.background_id,
            "classifications": [list(item) for item in record.classifications],
        }
        for record in result.history
        if record.classifications
    ]
    graph_metrics = _graph_metrics(problem.ground_truth.scm.dag, result)
    accounting = result.query_accounting
    metadata_conflicts = len(result.structure_estimate.metadata.orientation_conflicts)
    structure_failures = (
        len(result.structure_estimate.metadata.failed_tests)
        + len(result.orientation_conflicts)
        - metadata_conflicts
    )
    serialized_conflicts = [asdict(item) for item in result.orientation_conflicts]
    return {
        "initial_global_reverse_kl": gaussian_kl_target_to_candidate(
            problem.public.target_mean,
            problem.public.target_covariance,
            initial_mean,
            initial_covariance,
        ),
        "final_global_reverse_kl": result.final_global_loss,
        "backend_diagnostics": backend_diagnostics,
        "resolved_backend": backend_diagnostics["backend"],
        "resolved_device": backend_diagnostics["device"],
        "local_composite_losses": result.final_local_losses,
        "mean_error": float(np.linalg.norm(problem.public.target_mean - final_mean)),
        "covariance_error": float(
            np.linalg.norm(problem.public.target_covariance - final_covariance)
        ),
        "accepted_final_support": _intervention_coordinate_labels(
            result.final_intervention
        ),
        "accepted_final_coordinate_count": result.final_intervention.support_cost(),
        "accepted_shift_coordinate_count": sum(
            label.startswith("shift:")
            for label in _intervention_coordinate_labels(result.final_intervention)
        ),
        "accepted_scale_coordinate_count": sum(
            label.startswith("log_scale:")
            for label in _intervention_coordinate_labels(result.final_intervention)
        ),
        "accepted_candidates": result.accepted_proposals,
        "rejected_candidates": result.rejected_proposals,
        "inconclusive_candidates": result.inconclusive_proposals,
        "probe_count": int(accounting.get("probe_queries", 0) or 0),
        "source_outcomes": source_outcomes,
        "algorithm_environment_calls": accounting["algorithm_environment_calls"],
        "executed_environment_queries": accounting["executed_environment_queries"],
        "cache_hits": accounting["cache_hits"],
        "failed_environment_queries": accounting["failed_environment_queries"],
        "total_observations": accounting["total_observations"],
        "shift_coordinate_uses": accounting["shift_coordinate_uses"],
        "scale_coordinate_uses": accounting["scale_coordinate_uses"],
        "baseline_queries": accounting.get("baseline_queries", 0),
        "initialization_evaluations": accounting.get("baseline_queries", 0),
        "probe_queries": accounting.get("probe_queries", 0),
        "diagnostic_query_count": accounting.get("probe_queries", 0),
        "candidate_queries": accounting.get("candidate_queries", 0),
        "candidate_search_evaluations": result.optimizer_search_evaluations,
        "optimizer_search_evaluations": result.optimizer_search_evaluations,
        "structure_estimation_metadata": asdict(result.structure_estimate.metadata),
        "structure_learning_failure_count": structure_failures,
        "orientation_conflict_count": len(result.orientation_conflicts),
        "orientation_conflicts": serialized_conflicts,
        "local_global_surrogate_diagnostics": [
            asdict(item) for item in result.surrogate_proposal_diagnostics
        ],
        "local_global_surrogate_summary": asdict(result.surrogate_diagnostic_summary),
        "public_action_signature": result.public_action_signature,
        "termination_detail": (
            {
                "reason": result.termination_reason,
                "orientation_conflicts": serialized_conflicts,
            }
            if result.orientation_conflicts
            else result.termination_reason
        ),
        **graph_metrics,
    }


def _graph_information_regime(spec: RunSpec) -> str:
    values = {
        "known_support_oracle": "known_target_support",
        "known_dag_oracle": "known_dag_and_coefficients",
        "idealized_unknown_graph_oracle": "unknown_graph_oracle_active_probing",
        "non_oracle_noiseless_supplied_cpdag": "supplied_exact_cpdag_active_probing",
        "non_oracle_noiseless_estimated_cpdag": "estimated_cpdag_active_probing",
    }
    if spec.experiment_mode == "exact_nonrealisable":
        return (
            "estimated_cpdag_active_probing"
            if spec.initial_cpdag_provider == "estimated_pc_cpdag"
            else "supplied_exact_cpdag_active_probing"
        )
    return values[str(spec.experiment_mode)]


def _graph_metrics(
    truth: nx.DiGraph,
    result: NonOracleExperimentResult,
) -> dict[str, object]:
    initial = result.structure_estimate.graph
    final = result.final_graph
    truth_directed = {(int(a), int(b)) for a, b in truth.edges}
    truth_skeleton = {frozenset(edge) for edge in truth_directed}
    final_directed = set(final.directed_edges)
    final_skeleton = {
        frozenset(edge) for edge in final.directed_edges | final.undirected_edges
    }
    possible = max(1, truth.number_of_nodes() * (truth.number_of_nodes() - 1) // 2)
    skeleton_accuracy = 1.0 - len(truth_skeleton ^ final_skeleton) / possible
    orientation_accuracy = (
        len(final_directed & truth_directed) / len(final_directed)
        if final_directed
        else None
    )
    return {
        "initial_oriented_edges": len(initial.directed_edges),
        "final_oriented_edges": len(final.directed_edges),
        "directed_shd": len(final_directed ^ truth_directed),
        "skeleton_accuracy": float(skeleton_accuracy),
        "orientation_accuracy": orientation_accuracy,
        "direct_orientation_count": len(result.direct_orientation_evidence),
        "meek_orientation_count": len(result.meek_orientation_evidence),
    }


def _mode_idealizations(mode: ExperimentMode) -> tuple[str, ...]:
    values = {
        "known_support_oracle": (
            "true_dag",
            "true_coefficients",
            "known_target_coordinate_support",
            "analytic_reverse_kl",
        ),
        "known_dag_oracle": (
            "true_dag",
            "true_coefficients",
            "all_legal_coordinates_exposed",
            "analytic_reverse_kl",
        ),
        "idealized_unknown_graph_oracle": (
            "oracle_exact_cpdag",
            "oracle_noiseless_probe_outcomes",
            "oracle_exact_moments_and_gradients",
            "analytic_reverse_kl",
        ),
        "non_oracle_noiseless_supplied_cpdag": (
            "supplied_exact_public_cpdag",
            "exact_public_moment_blackbox",
        ),
        "non_oracle_noiseless_estimated_cpdag": ("exact_public_moment_blackbox",),
        "exact_nonrealisable": (
            "exact_public_moment_blackbox",
            "evaluation_only_hidden_target_generator",
        ),
    }
    return values[str(mode)]


def _intervention_coordinates(
    intervention: ShiftScaleIntervention,
) -> tuple[ParameterCoordinate, ...]:
    shifted, scaled = coordinate_support_masks(intervention.shift, intervention.scale)
    return tuple(
        [ParameterCoordinate(int(node), "shift") for node in np.flatnonzero(shifted)]
        + [
            ParameterCoordinate(int(node), "log_scale")
            for node in np.flatnonzero(scaled)
        ]
    )


def _coordinate_label(coordinate: ParameterCoordinate) -> str:
    return f"{coordinate.kind}:{coordinate.node}"


def _intervention_coordinate_labels(
    intervention: ShiftScaleIntervention,
) -> tuple[str, ...]:
    return tuple(
        _coordinate_label(coordinate)
        for coordinate in _intervention_coordinates(intervention)
    )



def _git_commit() -> str | None:
    repository = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None
