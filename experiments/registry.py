"""Validated production experiment registry and provider-graph resolution."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from numbers import Real
from pathlib import Path
from typing import Literal

from graphs.interventions import INTERVENTION_SEMANTICS
from graphs.noise import validate_noise_variance_regime
from matching.capabilities import AssemblyComponent, validate_experiment_components

from .assemblies import (
    RunnableExperimentAssembly,
    assemble_exact_nonrealisable,
    assemble_finite_sample_nonoracle,
    assemble_finite_sample_nonrealisable,
    assemble_idealized_unknown_graph_oracle,
    assemble_known_dag_oracle,
    assemble_known_support_oracle,
    assemble_nonoracle_noiseless,
)
from .protocols import ExperimentProtocol, protocol_by_version

ProductionMode = Literal[
    "known_support_oracle",
    "known_dag_oracle",
    "idealized_unknown_graph_oracle",
    "non_oracle_noiseless_supplied_cpdag",
    "non_oracle_noiseless_estimated_cpdag",
    "finite_sample_non_oracle_supplied_cpdag",
    "finite_sample_non_oracle_estimated_cpdag",
    "exact_nonrealisable",
    "finite_sample_nonrealisable",
]

PRODUCTION_RESULT_SCHEMA_VERSION = 2

_EXACT_NOISELESS_MODES = frozenset(
    {
        "non_oracle_noiseless_supplied_cpdag",
        "non_oracle_noiseless_estimated_cpdag",
    }
)


@dataclass(frozen=True)
class ProductionModeDescriptor:
    mode: ProductionMode
    assembly_factory: str
    permitted_structure_providers: tuple[str, ...]
    permitted_environment_providers: tuple[str, ...]
    permitted_local_loss_providers: tuple[str, ...]
    permitted_source_inference_providers: tuple[str, ...]
    permitted_orientation_providers: tuple[str, ...]
    permitted_coordinate_search_providers: tuple[str, ...]
    permitted_evaluators: tuple[str, ...]
    permitted_unresolved_state_providers: tuple[str, ...]
    target_families: tuple[str, ...]
    oracle_capabilities_allowed: bool
    required_configuration_fields: tuple[str, ...]
    result_schema_version: int = PRODUCTION_RESULT_SCHEMA_VERSION

    def provider_graph(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "assembly_factory": self.assembly_factory,
            "structure_provider": self.permitted_structure_providers,
            "environment_provider": self.permitted_environment_providers,
            "local_loss_provider": self.permitted_local_loss_providers,
            "source_inference_provider": self.permitted_source_inference_providers,
            "orientation_provider": self.permitted_orientation_providers,
            "coordinate_search_provider": self.permitted_coordinate_search_providers,
            "evaluator": self.permitted_evaluators,
            "unresolved_state_provider": self.permitted_unresolved_state_providers,
            "target_families": self.target_families,
            "oracle_capabilities_allowed": self.oracle_capabilities_allowed,
            "result_schema_version": self.result_schema_version,
        }


_COMMON_REQUIRED = (
    "mode",
    "protocol_version",
    "structure_provider",
    "environment_provider",
    "local_loss_provider",
    "source_inference_provider",
    "orientation_provider",
    "coordinate_search_provider",
    "evaluator",
    "unresolved_state_provider",
    "target_family",
    "target_is_realisable",
    "recover_target_support",
    "query_budget",
    "observation_budget",
    "shift_domain",
    "log_scale_domain",
    "confidence_level",
    "error_probability",
    "n_nodes",
    "rho",
    "k",
    "final_parameter_coordinate_budget",
    "baseline_query_budget",
    "probe_query_budget",
    "candidate_query_budget",
    "probe_amplitude_schedule",
    "max_probe_targets",
    "local_loss_weights",
    "local_loss_tolerance",
    "global_loss_tolerance",
    "improvement_tolerance",
    "exploration_seed",
    "candidate_evaluation_seed",
    "environment_seed",
    "bootstrap_seed",
    "problem_seed",
    "backend",
    "output_directory",
)
_PREFLIGHT_EXTENSION_FIELDS = frozenset({"regime_id", "calibration_artifacts"})


def _descriptor(
    mode: ProductionMode,
    factory: str,
    *,
    structure: tuple[str, ...],
    environment: tuple[str, ...],
    local: tuple[str, ...],
    source: tuple[str, ...],
    orientation: tuple[str, ...],
    search: tuple[str, ...],
    evaluator: tuple[str, ...],
    unresolved: tuple[str, ...],
    targets: tuple[str, ...],
    oracle: bool,
) -> ProductionModeDescriptor:
    return ProductionModeDescriptor(
        mode,
        factory,
        structure,
        environment,
        local,
        source,
        orientation,
        search,
        evaluator,
        unresolved,
        targets,
        oracle,
        _COMMON_REQUIRED,
    )


PRODUCTION_REGISTRY: dict[ProductionMode, ProductionModeDescriptor] = {
    descriptor.mode: descriptor
    for descriptor in (
        _descriptor(
            "known_support_oracle",
            "assemble_known_support_oracle",
            structure=("true_dag_and_coefficients",),
            environment=("oracle_analytic_environment",),
            local=("oracle_composite_local_loss",),
            source=("known_target_coordinate_support",),
            orientation=("true_dag_orientation",),
            search=("oracle_gradient_coordinate_search",),
            evaluator=("oracle_analytic_reverse_kl",),
            unresolved=("oracle_exact_unresolved",),
            targets=("realisable",),
            oracle=True,
        ),
        _descriptor(
            "known_dag_oracle",
            "assemble_known_dag_oracle",
            structure=("true_dag_and_coefficients",),
            environment=("oracle_analytic_environment",),
            local=("oracle_composite_local_loss",),
            source=("oracle_known_dag_source",),
            orientation=("true_dag_orientation",),
            search=("oracle_gradient_coordinate_search",),
            evaluator=("oracle_analytic_reverse_kl",),
            unresolved=("oracle_exact_unresolved",),
            targets=("realisable",),
            oracle=True,
        ),
        _descriptor(
            "idealized_unknown_graph_oracle",
            "assemble_idealized_unknown_graph_oracle",
            structure=("oracle_exact_cpdag",),
            environment=("oracle_analytic_environment",),
            local=("oracle_composite_local_loss",),
            source=("oracle_probe_outcomes",),
            orientation=("oracle_orientation_outcomes",),
            search=("oracle_gradient_coordinate_search",),
            evaluator=("oracle_analytic_reverse_kl",),
            unresolved=("oracle_exact_unresolved",),
            targets=("realisable",),
            oracle=True,
        ),
        _descriptor(
            "non_oracle_noiseless_supplied_cpdag",
            "assemble_nonoracle_noiseless",
            structure=("supplied_exact_public_cpdag",),
            environment=("exact_public_moment_blackbox",),
            local=("public_composite_local_loss",),
            source=("exact_public_probe_response",),
            orientation=("public_source_to_nonsource_meek",),
            search=("bounded_exact_derivative_free",),
            evaluator=("exact_public_reverse_kl",),
            unresolved=("public_exact_local_unresolved",),
            targets=("realisable",),
            oracle=False,
        ),
        _descriptor(
            "non_oracle_noiseless_estimated_cpdag",
            "assemble_nonoracle_noiseless",
            structure=("estimated_pc_cpdag",),
            environment=("exact_public_moment_blackbox",),
            local=("public_composite_local_loss",),
            source=("exact_public_probe_response",),
            orientation=("public_source_to_nonsource_meek",),
            search=("bounded_exact_derivative_free",),
            evaluator=("exact_public_reverse_kl",),
            unresolved=("public_exact_local_unresolved",),
            targets=("realisable",),
            oracle=False,
        ),
        _descriptor(
            "finite_sample_non_oracle_supplied_cpdag",
            "assemble_finite_sample_nonoracle",
            structure=("supplied_exact_public_cpdag",),
            environment=("finite_sample_blackbox",),
            local=("empirical_composite_local_loss",),
            source=("welch_equivalence_holm",),
            orientation=("public_source_to_nonsource_meek",),
            search=("finite_sequential_coordinate_search",),
            evaluator=("bootstrap_reverse_kl_acceptance",),
            unresolved=("bootstrap_three_way_unresolved",),
            targets=("realisable",),
            oracle=False,
        ),
        _descriptor(
            "finite_sample_non_oracle_estimated_cpdag",
            "assemble_finite_sample_nonoracle",
            structure=("estimated_pc_cpdag",),
            environment=("finite_sample_blackbox",),
            local=("empirical_composite_local_loss",),
            source=("welch_equivalence_holm",),
            orientation=("public_source_to_nonsource_meek",),
            search=("finite_sequential_coordinate_search",),
            evaluator=("bootstrap_reverse_kl_acceptance",),
            unresolved=("bootstrap_three_way_unresolved",),
            targets=("realisable",),
            oracle=False,
        ),
        _descriptor(
            "exact_nonrealisable",
            "assemble_exact_nonrealisable",
            structure=("supplied_exact_public_cpdag", "estimated_pc_cpdag"),
            environment=("exact_public_moment_blackbox",),
            local=("public_composite_local_loss",),
            source=("exact_public_probe_response",),
            orientation=("public_source_to_nonsource_meek",),
            search=("bounded_exact_derivative_free",),
            evaluator=("exact_public_reverse_kl",),
            unresolved=("public_exact_local_unresolved",),
            targets=(
                "nonproportional_structural_coefficients",
                "graph_perturbed_linear_gaussian",
                "correlated_noise_gaussian",
                "external_spd_gaussian",
            ),
            oracle=False,
        ),
        _descriptor(
            "finite_sample_nonrealisable",
            "assemble_finite_sample_nonrealisable",
            structure=("supplied_exact_public_cpdag", "estimated_pc_cpdag"),
            environment=("finite_sample_blackbox",),
            local=("empirical_composite_local_loss",),
            source=("welch_equivalence_holm",),
            orientation=("public_source_to_nonsource_meek",),
            search=("finite_sequential_coordinate_search",),
            evaluator=("bootstrap_reverse_kl_acceptance",),
            unresolved=("bootstrap_three_way_unresolved",),
            targets=(
                "nonproportional_structural_coefficients",
                "graph_perturbed_linear_gaussian",
                "correlated_noise_gaussian",
                "external_spd_gaussian",
            ),
            oracle=False,
        ),
    )
}


def _values_match(actual: object, expected: object) -> bool:
    if (
        isinstance(actual, Real)
        and not isinstance(actual, bool)
        and isinstance(expected, Real)
        and not isinstance(expected, bool)
    ):
        return isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _values_match(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        return len(actual) == len(expected) and all(
            _values_match(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _require_match(field: str, actual: object, expected: object) -> None:
    if not _values_match(actual, expected):
        raise ValueError(
            f"runtime/configuration mismatch for {field}: "
            f"expected {expected!r}, got {actual!r}."
        )


@dataclass(frozen=True)
class ProductionRunConfiguration:
    mode: ProductionMode
    protocol_version: str
    intervention_semantics: str
    structure_provider: str
    environment_provider: str
    n_nodes: int
    rho: float
    k: int
    final_parameter_coordinate_budget: int
    baseline_query_budget: int | None
    probe_query_budget: int | None
    candidate_query_budget: int | None
    probe_amplitude_schedule: tuple[float, ...]
    max_probe_targets: int
    local_loss_weights: tuple[float, float, float]
    local_loss_tolerance: float
    global_loss_tolerance: float
    improvement_tolerance: float
    exploration_seed: int
    candidate_evaluation_seed: int
    environment_seed: int
    bootstrap_seed: int
    problem_seed: int
    backend: str
    local_loss_provider: str
    source_inference_provider: str
    orientation_provider: str
    coordinate_search_provider: str
    evaluator: str
    unresolved_state_provider: str
    target_family: str
    target_is_realisable: bool
    recover_target_support: bool
    query_budget: int | None
    observation_budget: int
    sample_size: int | None
    observational_sample_size: int | None
    shift_domain: tuple[float, float]
    log_scale_domain: tuple[float, float]
    confidence_level: float
    error_probability: float
    output_directory: str
    noise_variance_regime: str = "homoskedastic"
    noise_variance_seed: int | None = None
    cross_noise_regime_pair_id: str = ""
    device: str = "cpu"
    result_schema_version: int = PRODUCTION_RESULT_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ProductionRunConfiguration:
        missing = sorted(set(_COMMON_REQUIRED) - set(raw))
        if missing:
            raise ValueError(f"missing required configuration fields: {missing}.")
        field_names = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - field_names - _PREFLIGHT_EXTENSION_FIELDS)
        if unknown:
            raise ValueError(f"unknown production configuration fields: {unknown}.")
        values = {name: value for name, value in raw.items() if name in field_names}
        values.setdefault("intervention_semantics", INTERVENTION_SEMANTICS)
        values.setdefault("sample_size", None)
        values.setdefault("observational_sample_size", None)
        values.setdefault("device", "cpu")
        values.setdefault("result_schema_version", PRODUCTION_RESULT_SCHEMA_VERSION)
        for name in (
            "shift_domain",
            "log_scale_domain",
            "local_loss_weights",
            "probe_amplitude_schedule",
        ):
            values[name] = tuple(values[name])  # type: ignore[arg-type]
        return cls(**values)  # type: ignore[arg-type]

    def validate(self) -> ExperimentProtocol:
        try:
            descriptor = PRODUCTION_REGISTRY[self.mode]
        except KeyError as error:
            raise ValueError(f"unregistered production mode {self.mode!r}.") from error
        if self.intervention_semantics != INTERVENTION_SEMANTICS:
            raise ValueError(
                "production configuration has stale intervention semantics."
            )
        validate_noise_variance_regime(self.noise_variance_regime)
        if (
            self.noise_variance_regime != "homoskedastic"
            and self.noise_variance_seed is None
        ):
            raise ValueError("heterogeneous noise requires noise_variance_seed.")
        protocol = protocol_by_version(self.protocol_version)
        if (
            self.mode in _EXACT_NOISELESS_MODES
            and protocol.protocol_version != "exact_noiseless_protocol_v2"
        ):
            raise ValueError(
                "exact non-oracle modes require exact_noiseless_protocol_v2."
            )
        provider_values = {
            "structure_provider": descriptor.permitted_structure_providers,
            "environment_provider": descriptor.permitted_environment_providers,
            "local_loss_provider": descriptor.permitted_local_loss_providers,
            "source_inference_provider": descriptor.permitted_source_inference_providers,
            "orientation_provider": descriptor.permitted_orientation_providers,
            "coordinate_search_provider": descriptor.permitted_coordinate_search_providers,
            "evaluator": descriptor.permitted_evaluators,
            "unresolved_state_provider": descriptor.permitted_unresolved_state_providers,
        }
        for name, permitted in provider_values.items():
            if getattr(self, name) not in permitted:
                raise ValueError(
                    f"mode {self.mode!r} rejects {name}={getattr(self, name)!r}."
                )
        finite = self.environment_provider == "finite_sample_blackbox"
        estimated = self.structure_provider == "estimated_pc_cpdag"
        if finite and (self.sample_size is None or self.sample_size <= 0):
            raise ValueError("empirical modes require a positive finite sample size.")
        if estimated and (
            self.observational_sample_size is None
            or self.observational_sample_size <= 0
        ):
            raise ValueError("estimated-CPDAG mode requires observational samples.")
        if not self.target_is_realisable and self.recover_target_support:
            raise ValueError(
                "non-realisable modes cannot recover hidden target support."
            )
        if self.target_family not in descriptor.target_families:
            raise ValueError("target family is incompatible with the selected mode.")
        if self.target_is_realisable != (self.target_family == "realisable"):
            raise ValueError("target realisability flag and family disagree.")
        if (
            isinstance(self.n_nodes, bool)
            or not isinstance(self.n_nodes, int)
            or self.n_nodes <= 0
        ):
            raise ValueError("n_nodes must be a positive integer.")
        if not isfinite(self.rho) or not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [0, 1].")
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k <= 0:
            raise ValueError("k must be a positive integer.")
        if self.k > self.n_nodes:
            raise ValueError("k cannot exceed n_nodes.")
        if (
            isinstance(self.final_parameter_coordinate_budget, bool)
            or not isinstance(self.final_parameter_coordinate_budget, int)
            or not 0 <= self.final_parameter_coordinate_budget <= 2 * self.n_nodes
        ):
            raise ValueError(
                "final_parameter_coordinate_budget must lie in [0, 2*n_nodes]."
            )
        for name in (
            "baseline_query_budget",
            "probe_query_budget",
            "candidate_query_budget",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None.")
        categorized_query_budget = sum(
            value
            for value in (
                self.baseline_query_budget,
                self.probe_query_budget,
                self.candidate_query_budget,
            )
            if value is not None
        )
        if (
            self.query_budget is not None
            and categorized_query_budget > self.query_budget
        ):
            raise ValueError("per-category query caps exceed query_budget.")
        if not self.probe_amplitude_schedule or any(
            not isfinite(value) or value == 0.0
            for value in self.probe_amplitude_schedule
        ):
            raise ValueError(
                "probe_amplitude_schedule must contain finite nonzero values."
            )
        if (
            isinstance(self.max_probe_targets, bool)
            or not isinstance(self.max_probe_targets, int)
            or self.max_probe_targets <= 0
        ):
            raise ValueError("max_probe_targets must be a positive integer.")
        if len(self.local_loss_weights) != 3 or any(
            not isfinite(value) or value < 0.0 for value in self.local_loss_weights
        ):
            raise ValueError(
                "local_loss_weights must contain three finite nonnegative values."
            )
        if not any(value > 0.0 for value in self.local_loss_weights):
            raise ValueError("at least one local loss weight must be positive.")
        for name in (
            "local_loss_tolerance",
            "global_loss_tolerance",
            "improvement_tolerance",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in (
            "exploration_seed",
            "candidate_evaluation_seed",
            "environment_seed",
            "bootstrap_seed",
            "problem_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must be nonempty.")
        if self.query_budget is not None and (
            isinstance(self.query_budget, bool)
            or not isinstance(self.query_budget, int)
            or self.query_budget <= 0
        ):
            raise ValueError("query_budget must be a positive integer or None.")
        if self.observation_budget < 0 or (finite and self.observation_budget == 0):
            raise ValueError("observation budget is inconsistent with the mode.")
        if (
            finite
            and self.sample_size is not None
            and self.sample_size > self.observation_budget
        ):
            raise ValueError("one finite sample batch exceeds the observation budget.")
        if (
            self.query_budget is not None
            and protocol.budgets.total_executed_query_budget is not None
            and self.query_budget > protocol.budgets.total_executed_query_budget
        ):
            raise ValueError("query budget exceeds the protocol hard cap.")
        if self.observation_budget > protocol.budgets.total_observation_budget:
            raise ValueError("observation budget exceeds the protocol hard cap.")
        if (
            len(self.shift_domain) != 2
            or not all(isfinite(value) for value in self.shift_domain)
            or self.shift_domain[0] >= self.shift_domain[1]
        ):
            raise ValueError("shift domain must be an increasing pair.")
        if (
            len(self.log_scale_domain) != 2
            or not all(isfinite(value) for value in self.log_scale_domain)
            or self.log_scale_domain[0] >= self.log_scale_domain[1]
        ):
            raise ValueError("log-scale domain must be an increasing pair.")
        for name in ("confidence_level", "error_probability"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must lie in (0, 1).")
        if self.result_schema_version != descriptor.result_schema_version:
            raise ValueError("result schema version does not match the registry.")
        if not self.output_directory:
            raise ValueError("output_directory must be nonempty.")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'.")
        if finite != (protocol.purpose != "exact"):
            raise ValueError("protocol purpose is incompatible with the mode.")
        _require_match(
            "query_budget",
            self.query_budget,
            protocol.budgets.total_executed_query_budget,
        )
        _require_match(
            "observation_budget",
            self.observation_budget,
            protocol.budgets.total_observation_budget,
        )
        _require_match(
            "baseline_query_budget",
            self.baseline_query_budget,
            protocol.budgets.baseline_query_budget,
        )
        _require_match(
            "probe_query_budget",
            self.probe_query_budget,
            protocol.budgets.probe_query_budget,
        )
        _require_match(
            "candidate_query_budget",
            self.candidate_query_budget,
            protocol.budgets.candidate_query_budget,
        )
        if protocol.protocol_version == "exact_noiseless_protocol_v2":
            _require_match("max_probe_targets", self.max_probe_targets, self.n_nodes)
        _require_match(
            "shift_domain",
            self.shift_domain,
            protocol.coordinate_search.shift_domain,
        )
        _require_match(
            "log_scale_domain",
            self.log_scale_domain,
            protocol.coordinate_search.log_scale_domain,
        )
        _require_match(
            "local_loss_tolerance",
            self.local_loss_tolerance,
            protocol.unresolved_state.local_loss_tolerance,
        )
        _require_match(
            "global_loss_tolerance",
            self.global_loss_tolerance,
            protocol.unresolved_state.global_kl_success_tolerance,
        )
        _require_match(
            "confidence_level",
            self.confidence_level,
            protocol.unresolved_state.global_confidence_level,
        )
        _require_match(
            "error_probability",
            self.error_probability,
            1.0 - self.confidence_level,
        )
        if protocol.candidate_acceptance is not None:
            _require_match(
                "improvement_tolerance",
                self.improvement_tolerance,
                protocol.candidate_acceptance.minimum_improvement_tolerance,
            )
        if estimated and protocol.observational_structure is not None:
            _require_match(
                "observational_sample_size",
                self.observational_sample_size,
                protocol.observational_structure.observational_sample_size,
            )
        return protocol

    def effective_runtime_settings(self) -> dict[str, object]:
        """Return the canonical contract that concrete runtime objects must match."""
        protocol = self.validate()
        return {
            "mode": self.mode,
            "noise_variance_regime": self.noise_variance_regime,
            "noise_variance_seed": self.noise_variance_seed,
            "cross_noise_regime_pair_id": self.cross_noise_regime_pair_id,
            "protocol_version": self.protocol_version,
            "n_nodes": self.n_nodes,
            "rho": self.rho,
            "k": self.k,
            "final_parameter_coordinate_budget": self.final_parameter_coordinate_budget,
            "query_caps": {
                "total": self.query_budget,
                "observations": self.observation_budget,
                "baseline": self.baseline_query_budget,
                "probe": self.probe_query_budget,
                "candidate": self.candidate_query_budget,
            },
            "shift_domain": self.shift_domain,
            "log_scale_domain": self.log_scale_domain,
            "probe_amplitude_schedule": self.probe_amplitude_schedule,
            "max_probe_targets": self.max_probe_targets,
            "local_loss_weights": self.local_loss_weights,
            "local_loss_tolerance": self.local_loss_tolerance,
            "global_loss_tolerance": self.global_loss_tolerance,
            "improvement_tolerance": self.improvement_tolerance,
            "sample_size": self.sample_size,
            "observational_sample_size": self.observational_sample_size,
            "confidence_level": self.confidence_level,
            "error_probability": self.error_probability,
            "rng_seeds": {
                "exploration": self.exploration_seed,
                "candidate_evaluation": self.candidate_evaluation_seed,
                "environment": self.environment_seed,
                "bootstrap": self.bootstrap_seed,
                "problem": self.problem_seed,
            },
            "backend": self.backend,
            "device": self.device,
            "protocol": protocol.resolved_values(),
        }

    def resolved(self) -> dict[str, object]:
        protocol = self.validate()
        descriptor = PRODUCTION_REGISTRY[self.mode]
        return {
            "effective_runtime_settings": self.effective_runtime_settings(),
            "configuration": asdict(self),
            "protocol": protocol.resolved_values(),
            "provider_graph": descriptor.provider_graph(),
        }


@dataclass(frozen=True)
class ProductionRuntimeComponents:
    values: Mapping[str, object]

    def require(self, *names: str) -> tuple[object, ...]:
        missing = [name for name in names if name not in self.values]
        if missing:
            raise ValueError(f"missing runtime components: {missing}.")
        return tuple(self.values[name] for name in names)


def _validate_runtime_binding(
    configuration: ProductionRunConfiguration,
    runtime: ProductionRuntimeComponents,
) -> None:
    """Reject any injected runtime that differs from the resolved contract."""
    from graphs.problem_gen import GeneratedProblem, PublicProblemView
    from matching.blackbox import (
        ExactMomentBlackBoxEnvironment,
        FiniteSampleBlackBoxEnvironment,
    )
    from matching.scheduler import SchedulerSettings
    from matching.structure_learning import (
        EstimatedPCCPDAGProvider,
        SuppliedExactCPDAGProvider,
    )
    from matching.types import FinalSupportBudget

    from .noiseless_unknown_graph_pilot import NoiselessPilotConfig, PilotRunSpec
    from .nonoracle import NonOracleExperimentConfig

    (declared_settings,) = runtime.require("effective_runtime_settings")
    _require_match(
        "effective_runtime_settings",
        declared_settings,
        configuration.effective_runtime_settings(),
    )
    protocol = configuration.validate()
    mode = configuration.mode

    if mode == "idealized_unknown_graph_oracle":
        pilot_config, pilot_spec = runtime.require("pilot_config", "pilot_spec")
        if not isinstance(pilot_config, NoiselessPilotConfig):
            raise ValueError("runtime/configuration mismatch for pilot_config type.")
        if not isinstance(pilot_spec, PilotRunSpec):
            raise ValueError("runtime/configuration mismatch for pilot_spec type.")
        _require_match("n_nodes", pilot_spec.n_nodes, configuration.n_nodes)
        _require_match("rho", pilot_config.support_fraction, configuration.rho)
        _require_match("k", pilot_config.target_support_size, configuration.k)
        _require_match(
            "final_parameter_coordinate_budget",
            pilot_config.parameter_budget_for(pilot_spec.n_nodes),
            configuration.final_parameter_coordinate_budget,
        )
        _require_match(
            "query_budget",
            pilot_config.max_experiment_trials,
            configuration.query_budget,
        )
        _require_match("problem_seed", pilot_spec.seed, configuration.problem_seed)
        _require_match(
            "local_loss_weights",
            (
                pilot_config.local_marginal_weight,
                pilot_config.local_pairwise_weight,
                pilot_config.local_conditional_weight,
            ),
            configuration.local_loss_weights,
        )
        _require_match(
            "local_loss_tolerance",
            pilot_config.local_unresolved_threshold,
            configuration.local_loss_tolerance,
        )
        _require_match(
            "improvement_tolerance",
            pilot_config.improvement_tolerance,
            configuration.improvement_tolerance,
        )
        _require_match(
            "max_probe_targets",
            pilot_config.max_probe_targets,
            configuration.max_probe_targets,
        )
        _require_match(
            "log_scale_domain",
            (pilot_config.scale_min, pilot_config.scale_max),
            tuple(
                __import__("math").exp(value)
                for value in configuration.log_scale_domain
            ),
        )
        (problem,) = runtime.require("problem")
        if not isinstance(problem, GeneratedProblem):
            raise TypeError("runtime/configuration mismatch for problem type.")
        _require_match("n_nodes", problem.public.n_nodes, configuration.n_nodes)
        _require_match(
            "exploration_seed",
            problem.public.algorithm_seeds.exploration_seed,
            configuration.exploration_seed,
        )
        _require_match(
            "candidate_evaluation_seed",
            problem.public.algorithm_seeds.candidate_evaluation_seed,
            configuration.candidate_evaluation_seed,
        )
        return

    (problem,) = runtime.require("problem")
    public_problem = (
        problem.public if isinstance(problem, GeneratedProblem) else problem
    )
    if not isinstance(public_problem, PublicProblemView):
        raise TypeError("runtime/configuration mismatch for problem type.")
    _require_match("n_nodes", public_problem.n_nodes, configuration.n_nodes)
    _require_match(
        "exploration_seed",
        public_problem.algorithm_seeds.exploration_seed,
        configuration.exploration_seed,
    )
    _require_match(
        "candidate_evaluation_seed",
        public_problem.algorithm_seeds.candidate_evaluation_seed,
        configuration.candidate_evaluation_seed,
    )
    if configuration.structure_provider == "estimated_pc_cpdag":
        _require_match(
            "observational_sample_size",
            public_problem.observational_samples.shape[0],
            configuration.observational_sample_size,
        )

    if mode in {"known_support_oracle", "known_dag_oracle"}:
        scheduler, support, max_trials = runtime.require(
            "scheduler_settings", "final_support_budget", "max_experiment_trials"
        )
        if not isinstance(scheduler, SchedulerSettings):
            raise ValueError(
                "runtime/configuration mismatch for scheduler_settings type."
            )
        if not isinstance(support, FinalSupportBudget):
            raise ValueError(
                "runtime/configuration mismatch for final_support_budget type."
            )
        _require_match(
            "final_parameter_coordinate_budget",
            support.max_parameter_coordinates,
            configuration.final_parameter_coordinate_budget,
        )
        _require_match("query_budget", max_trials, configuration.query_budget)
        _require_match(
            "local_loss_tolerance",
            scheduler.loss_tolerance,
            configuration.local_loss_tolerance,
        )
        _require_match(
            "global_loss_tolerance",
            scheduler.global_loss_tolerance,
            configuration.global_loss_tolerance,
        )
        _require_match(
            "improvement_tolerance",
            scheduler.improvement_tolerance,
            configuration.improvement_tolerance,
        )
        _require_match(
            "max_probe_targets",
            scheduler.max_probe_targets,
            configuration.max_probe_targets,
        )
        _require_match(
            "log_scale_domain",
            (scheduler.scale_min, scheduler.scale_max),
            tuple(
                __import__("math").exp(value)
                for value in configuration.log_scale_domain
            ),
        )
        runner_kwargs = runtime.values.get("runner_kwargs", {})
        if not isinstance(runner_kwargs, Mapping):
            raise ValueError("runtime/configuration mismatch for runner_kwargs type.")
        _require_match(
            "probe_amplitude_schedule",
            (runner_kwargs.get("probe_shift_magnitude", 0.5),),
            configuration.probe_amplitude_schedule,
        )
        return

    environment, structure, algorithm_config = runtime.require(
        "environment", "structure_provider", "algorithm_config"
    )
    if not isinstance(algorithm_config, NonOracleExperimentConfig):
        raise TypeError("runtime/configuration mismatch for algorithm_config type.")
    finite = configuration.environment_provider == "finite_sample_blackbox"
    expected_environment = (
        FiniteSampleBlackBoxEnvironment if finite else ExactMomentBlackBoxEnvironment
    )
    if not isinstance(environment, expected_environment):
        raise TypeError(
            "runtime/configuration mismatch for evaluator/environment type."
        )
    expected_structure = (
        EstimatedPCCPDAGProvider
        if configuration.structure_provider == "estimated_pc_cpdag"
        else SuppliedExactCPDAGProvider
    )
    if not isinstance(structure, expected_structure):
        raise TypeError("runtime/configuration mismatch for structure_provider.")
    _validate_nonoracle_runtime(
        configuration,
        protocol,
        public_problem,
        environment,
        structure,
        algorithm_config,
        runtime,
    )


def _validate_nonoracle_runtime(
    configuration: ProductionRunConfiguration,
    protocol: ExperimentProtocol,
    problem: object,
    environment: object,
    structure: object,
    algorithm_config: object,
    runtime: ProductionRuntimeComponents,
) -> None:
    from matching.structure_learning import EstimatedPCCPDAGProvider

    config = algorithm_config
    ledger = environment.ledger  # type: ignore[attr-defined]
    for field, actual, expected in (
        (
            "final_parameter_coordinate_budget",
            config.final_support_budget.max_parameter_coordinates,
            configuration.final_parameter_coordinate_budget,
        ),
        ("query_budget", ledger.max_query_batches, configuration.query_budget),
        (
            "observation_budget",
            ledger.max_observations or 0,
            configuration.observation_budget,
        ),
        (
            "baseline_query_budget",
            ledger.max_baseline_queries,
            configuration.baseline_query_budget,
        ),
        (
            "probe_query_budget",
            ledger.max_probe_queries,
            configuration.probe_query_budget,
        ),
        (
            "candidate_query_budget",
            ledger.max_candidate_queries,
            configuration.candidate_query_budget,
        ),
        (
            "local_loss_weights",
            (
                config.local_weights.marginal,
                config.local_weights.pairwise,
                config.local_weights.conditional,
            ),
            configuration.local_loss_weights,
        ),
        (
            "local_loss_tolerance",
            config.local_loss_tolerance,
            configuration.local_loss_tolerance,
        ),
        (
            "global_loss_tolerance",
            config.global_loss_tolerance,
            configuration.global_loss_tolerance,
        ),
        (
            "improvement_tolerance",
            config.improvement_tolerance,
            configuration.improvement_tolerance,
        ),
        (
            "probe_amplitude_schedule",
            config.probe.probe_amplitude_schedule,
            configuration.probe_amplitude_schedule,
        ),
        (
            "max_probe_targets",
            config.probe.max_probe_targets,
            configuration.max_probe_targets,
        ),
    ):
        _require_match(field, actual, expected)
    search = (
        config.finite_search
        if configuration.sample_size is not None
        else config.exact_search
    )
    _require_match(
        "shift_domain",
        (search.bounds.shift_min, search.bounds.shift_max),
        configuration.shift_domain,
    )
    _require_match(
        "log_scale_domain",
        (search.bounds.log_scale_min, search.bounds.log_scale_max),
        configuration.log_scale_domain,
    )
    _require_match(
        "coarse_grid_size",
        search.coarse_grid_size,
        protocol.coordinate_search.coarse_grid_size,
    )
    if isinstance(structure, EstimatedPCCPDAGProvider):
        observational = protocol.observational_structure
        if observational is None:
            raise ValueError(
                "runtime/configuration mismatch for observational protocol."
            )
        _require_match(
            "pc_significance_level",
            structure.settings.significance_level,
            observational.pc_significance_level,
        )
        _require_match(
            "maximum_conditioning_set_size",
            structure.settings.max_conditioning_set_size,
            observational.maximum_conditioning_set_size,
        )
        covariance = structure.moments_provider.settings
        for field in ("estimator", "diagonal_loading", "minimum_eigenvalue"):
            _require_match(
                f"covariance_{field}",
                getattr(covariance, field),
                getattr(observational.covariance, field),
            )
    if configuration.sample_size is not None:
        probe = protocol.probe_inference
        acceptance = protocol.candidate_acceptance
        if probe is None or acceptance is None:
            raise ValueError(
                "runtime/configuration mismatch for finite protocol components."
            )
        _require_match(
            "sample_size", config.baseline_sample_size, configuration.sample_size
        )
        _require_match(
            "initial_probe_sample_size",
            config.probe.initial_probe_sample_size,
            probe.initial_probe_sample_size,
        )
        _require_match(
            "maximum_probe_sample_size",
            config.probe.max_probe_sample_size,
            probe.maximum_probe_sample_size,
        )
        _require_match(
            "probe_sample_growth_factor",
            config.probe.sample_growth_factor,
            probe.sample_growth_factor,
        )
        _require_match(
            "maximum_probe_repeats",
            config.probe.max_probe_repeats,
            probe.maximum_repeats,
        )
        _require_match(
            "source_confidence_level",
            config.probe.source_confidence_level,
            probe.confidence_level,
        )
        _require_match(
            "source_equivalence_absolute",
            config.probe.source_tolerance_absolute,
            probe.source_equivalence_absolute,
        )
        _require_match(
            "source_equivalence_relative",
            config.probe.source_tolerance_relative,
            probe.source_equivalence_relative,
        )
        _require_match(
            "initial_candidate_sample_size",
            config.finite_search.initial_sample_size,
            protocol.coordinate_search.initial_candidate_sample_size,
        )
        _require_match(
            "maximum_candidate_sample_size",
            config.finite_search.max_sample_size_per_candidate,
            protocol.coordinate_search.maximum_candidate_sample_size,
        )
        _require_match(
            "maximum_candidate_values_per_coordinate",
            config.finite_search.max_query_batches,
            protocol.coordinate_search.maximum_candidate_values_per_coordinate,
        )
        _require_match(
            "candidate_confidence_level",
            config.acceptance.confidence_level,
            acceptance.confidence_level,
        )
        _require_match(
            "bootstrap_replicates",
            config.bootstrap_replicates,
            acceptance.bootstrap_replicates,
        )
        _require_match(
            "maximum_repeated_candidate_evaluations",
            config.acceptance.max_repeat_queries,
            acceptance.maximum_repeated_candidate_evaluations,
        )
        _require_match(
            "acceptance_improvement_tolerance",
            config.acceptance.improvement_tolerance,
            acceptance.minimum_improvement_tolerance,
        )
        environment_seed, bootstrap_seed = runtime.require(
            "environment_seed", "bootstrap_seed"
        )
        _require_match(
            "environment_seed", environment_seed, configuration.environment_seed
        )
        _require_match("bootstrap_seed", bootstrap_seed, configuration.bootstrap_seed)
        covariance = config.covariance
        expected_covariance = protocol.observational_structure.covariance
        for field in ("estimator", "diagonal_loading", "minimum_eigenvalue"):
            _require_match(
                f"covariance_{field}",
                getattr(covariance, field),
                getattr(expected_covariance, field),
            )
    del problem


def assemble_production_run(
    configuration: ProductionRunConfiguration,
    runtime: ProductionRuntimeComponents,
) -> RunnableExperimentAssembly[object]:
    """Resolve one validated production mode into its complete assembly."""
    configuration.validate()
    _validate_runtime_binding(configuration, runtime)
    mode = configuration.mode
    if mode == "known_support_oracle":
        values = runtime.require(
            "problem",
            "active_coordinates",
            "scheduler_settings",
            "final_support_budget",
            "max_experiment_trials",
        )
        return assemble_known_support_oracle(  # type: ignore[arg-type]
            *values,
            runner_kwargs=runtime.values.get("runner_kwargs"),
        )
    if mode == "known_dag_oracle":
        values = runtime.require(
            "problem",
            "scheduler_settings",
            "final_support_budget",
            "max_experiment_trials",
        )
        return assemble_known_dag_oracle(  # type: ignore[arg-type]
            *values,
            runner_kwargs=runtime.values.get("runner_kwargs"),
        )
    if mode == "idealized_unknown_graph_oracle":
        pilot_config, pilot_spec, problem = runtime.require(
            "pilot_config", "pilot_spec", "problem"
        )
        return assemble_idealized_unknown_graph_oracle(
            pilot_config, pilot_spec, problem=problem  # type: ignore[arg-type]
        )
    problem, environment, structure, algorithm_config = runtime.require(
        "problem", "environment", "structure_provider", "algorithm_config"
    )
    if not PRODUCTION_REGISTRY[mode].oracle_capabilities_allowed:
        validate_experiment_components(
            (
                "finite_sample_non_oracle"
                if configuration.environment_provider == "finite_sample_blackbox"
                else "non_oracle_noiseless"
            ),
            (
                AssemblyComponent("environment", environment),
                AssemblyComponent("initial_structure", structure),
            ),
        )
    if configuration.environment_provider == "finite_sample_blackbox":
        environment_seed, bootstrap_seed = runtime.require(
            "environment_seed", "bootstrap_seed"
        )
        if mode == "finite_sample_nonrealisable":
            return assemble_finite_sample_nonrealisable(
                problem,  # type: ignore[arg-type]
                environment,  # type: ignore[arg-type]
                structure,  # type: ignore[arg-type]
                algorithm_config,  # type: ignore[arg-type]
                environment_seed=environment_seed,  # type: ignore[arg-type]
                bootstrap_seed=bootstrap_seed,  # type: ignore[arg-type]
                evaluation_problem=runtime.values.get("evaluation_problem"),  # type: ignore[arg-type]
                comparator=runtime.values.get("comparator"),  # type: ignore[arg-type]
            )
        return assemble_finite_sample_nonoracle(
            problem,  # type: ignore[arg-type]
            environment,  # type: ignore[arg-type]
            structure,  # type: ignore[arg-type]
            algorithm_config,  # type: ignore[arg-type]
            environment_seed=environment_seed,  # type: ignore[arg-type]
            bootstrap_seed=bootstrap_seed,  # type: ignore[arg-type]
        )
    if mode == "exact_nonrealisable":
        return assemble_exact_nonrealisable(
            problem,  # type: ignore[arg-type]
            environment,  # type: ignore[arg-type]
            structure,  # type: ignore[arg-type]
            algorithm_config,  # type: ignore[arg-type]
            evaluation_problem=runtime.values.get("evaluation_problem"),  # type: ignore[arg-type]
            comparator=runtime.values.get("comparator"),  # type: ignore[arg-type]
        )
    return assemble_nonoracle_noiseless(
        problem,  # type: ignore[arg-type]
        environment,  # type: ignore[arg-type]
        structure,  # type: ignore[arg-type]
        algorithm_config,  # type: ignore[arg-type]
    )


def configuration_from_json(path: Path) -> ProductionRunConfiguration:
    raw = json.loads(path.read_text())
    if not isinstance(raw, Mapping):
        raise TypeError("production configuration must be a JSON object.")
    if raw.get("intervention_semantics") != INTERVENTION_SEMANTICS:
        raise ValueError("production JSON has missing or stale intervention_semantics.")
    return ProductionRunConfiguration.from_mapping(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--print-resolved", action="store_true", required=True)
    arguments = parser.parse_args(argv)
    configuration = configuration_from_json(arguments.config)
    print(json.dumps(configuration.resolved(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
