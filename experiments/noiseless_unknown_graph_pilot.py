"""Small idealized unknown-graph pilot configuration and smoke runner.

This module intentionally has no full-grid execution option. It records the
approved grid and can execute exactly one five-node smoke case before a later
approval authorizes the complete benchmark.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from experiments.modes import ExperimentMode, validate_experiment_assembly
from graphs.DAG_gen import DEFAULT_ERDOS_RENYI_EDGE_PROBABILITY
from graphs.interventions import INTERVENTION_SEMANTICS, ShiftScaleIntervention
from graphs.noise import (
    derive_noise_variance_seed,
    generate_noise_variances,
    validate_noise_variance_regime,
)
from graphs.problem_gen import (
    GeneratedProblem,
    PublicProblemView,
    SimulationGroundTruth,
)
from graphs.registry import generate_graph
from graphs.scm import LinearGaussianSCM, sample_coefficients
from graphs.seeds import AlgorithmSeeds, SimulationSeeds
from matching.local_losses import (
    CompositeGaussianLocalLoss,
    CompositeGaussianMismatchProvider,
    LocalLossWeights,
)
from matching.scheduler import (
    AlternatingMatchingScheduler,
    SchedulerResult,
    SchedulerSettings,
)
from matching.types import (
    ExperimentBudgetLedger,
    FinalSupportBudget,
    ParameterCoordinate,
)
from matching.unknown_graph_scheduler import (
    ProbeOutcomeMatchingEnvironment,
    UnknownGraphMatchingKnowledge,
)
from numerical.device import ComputeConfig
from oracle.analytic_gaussian_kl import OracleAnalyticGaussianKLLoss
from oracle.batching import (
    OracleBatchProblem,
    OracleInterventionBatchEvaluator,
    OracleInterventionBatchResult,
)
from oracle.known_dag_baseline import (
    OracleAnalyticSCMEnvironment,
    run_known_support_oracle_baseline,
)
from oracle.local_gaussian import OracleCandidateMomentsProvider
from oracle.noiseless_probe_environment import NoiselessOracleProbeEnvironment
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation
from oracle.oracle_optimizer import OracleActiveCoordinateOptimizer
from policies.clique_tree import CliqueTreeProbePolicy
from policies.exact_minmaxc import ExactMinMaxCExplorationPolicy
from policies.random_exploration import RandomExplorationPolicy

EXPERIMENT_LABEL = "noiseless graph-refinement experiments"
KL_DIRECTION = "target_to_candidate"
RESULT_SCHEMA_VERSION = 2
PilotGraphFamily = Literal[
    "path",
    "rooted_tree",
    "randomly_oriented_tree",
    "erdos_renyi",
    "barabasi_albert",
    "complete",
    "moralized_erdos_renyi",
    "tree_of_cliques",
]
PilotPolicy = Literal[
    "known_dag_oracle",
    "random",
    "exact_minmaxc",
    "clique_tree",
]

_GRAPH_FAMILIES = (
    "path",
    "rooted_tree",
    "randomly_oriented_tree",
    "erdos_renyi",
    "barabasi_albert",
    "complete",
    "moralized_erdos_renyi",
    "tree_of_cliques",
)
_DEFAULT_GRAPH_FAMILIES: tuple[PilotGraphFamily, ...] = (
    "path",
    "rooted_tree",
    "erdos_renyi",
    "moralized_erdos_renyi",
)
_POLICIES = (
    "known_dag_oracle",
    "random",
    "exact_minmaxc",
    "clique_tree",
)
_SUCCESS_REASONS = frozenset(
    {
        "loss_tolerance_reached",
    }
)


def _positive_integer(name: str, value: object) -> int:
    """Validate one positive integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _nonnegative_integer(name: str, value: object) -> int:
    """Validate one nonnegative integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _finite_real(name: str, value: object) -> float:
    """Validate and normalize one finite real configuration value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


@dataclass(frozen=True)
class NoiselessPilotConfig:
    """Configuration recorded before any full pilot-grid execution."""

    node_counts: tuple[int, ...] = (5, 10, 20)
    graph_families: tuple[PilotGraphFamily, ...] = _DEFAULT_GRAPH_FAMILIES
    policies: tuple[PilotPolicy, ...] = _POLICIES
    seeds: tuple[int, ...] = (20260726,)
    target_support_size: int = 2
    support_fraction: float = 0.4
    final_parameter_coordinate_budget: int | None = None
    allowed_coordinate_kinds: tuple[str, ...] = ("shift", "log_scale")
    intervention_composition: Literal["shift_only", "scale_only", "mixed"] = (
        "shift_only"
    )
    local_unresolved_threshold: float = 1e-8
    local_marginal_weight: float = 1.0
    local_pairwise_weight: float = 1.0
    local_conditional_weight: float = 1.0
    target_shift_magnitude: float = 0.5
    target_log_scale_magnitude: float = 0.4
    n_observational_samples: int = 1_000
    edge_probability: float = DEFAULT_ERDOS_RENYI_EDGE_PROBABILITY
    coefficient_min_abs: float = 0.25
    coefficient_max_abs: float = 0.75
    max_probe_targets: int = 2
    max_experiment_trials: int = 50
    max_consecutive_exploration_trials: int = 20
    exact_minmaxc_max_nodes: int = 10
    loss_tolerance: float = 1e-8
    improvement_tolerance: float = 1e-10
    mean_mismatch_atol: float = 1e-8
    parameter_recovery_atol: float = 1e-6
    scale_min: float = 0.25
    scale_max: float = 3.0
    optimizer_max_iterations: int = 1_000
    max_steps: int = 1_000
    noise_variance_regime: str = "homoskedastic"
    noise_variance_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the complete idealized-pilot configuration."""
        validate_noise_variance_regime(self.noise_variance_regime)
        if self.noise_variance_seed is not None:
            _nonnegative_integer("noise_variance_seed", self.noise_variance_seed)

        if not isinstance(self.node_counts, tuple) or not self.node_counts:
            raise TypeError("node_counts must be a nonempty tuple.")
        if len(set(self.node_counts)) != len(self.node_counts):
            raise ValueError("node_counts must not contain duplicates.")
        for value in self.node_counts:
            _positive_integer("node count", value)

        if (
            not isinstance(self.graph_families, tuple)
            or not self.graph_families
            or any(family not in _GRAPH_FAMILIES for family in self.graph_families)
        ):
            raise ValueError("graph_families must contain supported pilot families.")
        if len(set(self.graph_families)) != len(self.graph_families):
            raise ValueError("graph_families must not contain duplicates.")
        if (
            not isinstance(self.policies, tuple)
            or not self.policies
            or any(policy not in _POLICIES for policy in self.policies)
        ):
            raise ValueError("policies must contain supported pilot policies.")
        if len(set(self.policies)) != len(self.policies):
            raise ValueError("policies must not contain duplicates.")
        if not isinstance(self.seeds, tuple) or not self.seeds:
            raise TypeError("seeds must be a nonempty tuple.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates.")
        for seed in self.seeds:
            _nonnegative_integer("seed", seed)

        if (
            not isinstance(self.allowed_coordinate_kinds, tuple)
            or not self.allowed_coordinate_kinds
            or len(set(self.allowed_coordinate_kinds))
            != len(self.allowed_coordinate_kinds)
            or any(
                kind not in {"shift", "log_scale"}
                for kind in self.allowed_coordinate_kinds
            )
        ):
            raise ValueError(
                "allowed_coordinate_kinds must contain shift and/or log_scale."
            )
        if self.intervention_composition not in {"shift_only", "scale_only", "mixed"}:
            raise ValueError("intervention_composition is unsupported.")
        if self.final_parameter_coordinate_budget is not None:
            budget = _nonnegative_integer(
                "final_parameter_coordinate_budget",
                self.final_parameter_coordinate_budget,
            )
            if budget > 2 * min(self.node_counts):
                raise ValueError(
                    "final_parameter_coordinate_budget exceeds the configured 2n cap."
                )

        positive_integers = (
            "target_support_size",
            "n_observational_samples",
            "max_probe_targets",
            "max_experiment_trials",
            "max_consecutive_exploration_trials",
            "exact_minmaxc_max_nodes",
            "optimizer_max_iterations",
            "max_steps",
        )
        for name in positive_integers:
            _positive_integer(name, getattr(self, name))
        if self.target_support_size > min(self.node_counts):
            raise ValueError(
                "target_support_size must not exceed the smallest node count."
            )
        if (
            self.intervention_composition != "shift_only"
            and self.target_support_size > min(self.node_counts) - 1
        ):
            raise ValueError("scale-bearing target support cannot exceed n_nodes - 1.")
        if self.max_probe_targets > min(self.node_counts):
            raise ValueError(
                "max_probe_targets must not exceed the smallest node count."
            )

        target_shift_magnitude = _finite_real(
            "target_shift_magnitude",
            self.target_shift_magnitude,
        )
        if target_shift_magnitude == 0.0:
            raise ValueError("target_shift_magnitude must be nonzero.")
        target_log_scale_magnitude = _finite_real(
            "target_log_scale_magnitude",
            self.target_log_scale_magnitude,
        )
        if target_log_scale_magnitude == 0.0:
            raise ValueError("target_log_scale_magnitude must be nonzero.")
        object.__setattr__(
            self,
            "target_log_scale_magnitude",
            target_log_scale_magnitude,
        )
        edge_probability = _finite_real("edge_probability", self.edge_probability)
        if not 0.0 <= edge_probability <= 1.0:
            raise ValueError("edge_probability must lie in [0, 1].")
        support_fraction = _finite_real("support_fraction", self.support_fraction)
        if not 0.0 < support_fraction <= 1.0:
            raise ValueError("support_fraction must lie in (0, 1].")
        object.__setattr__(self, "support_fraction", support_fraction)
        coefficient_min_abs = _finite_real(
            "coefficient_min_abs",
            self.coefficient_min_abs,
        )
        coefficient_max_abs = _finite_real(
            "coefficient_max_abs",
            self.coefficient_max_abs,
        )
        if not 0.0 < coefficient_min_abs <= coefficient_max_abs:
            raise ValueError(
                "coefficient bounds must satisfy "
                "0 < coefficient_min_abs <= coefficient_max_abs."
            )
        nonnegative_reals = (
            "loss_tolerance",
            "improvement_tolerance",
            "mean_mismatch_atol",
            "parameter_recovery_atol",
            "local_unresolved_threshold",
            "local_marginal_weight",
            "local_pairwise_weight",
            "local_conditional_weight",
        )
        for name in nonnegative_reals:
            value = _finite_real(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
            object.__setattr__(self, name, value)
        scale_min = _finite_real("scale_min", self.scale_min)
        scale_max = _finite_real("scale_max", self.scale_max)
        if not 0.0 < scale_min <= scale_max:
            raise ValueError("scale bounds must satisfy 0 < scale_min <= scale_max.")

        object.__setattr__(
            self,
            "target_shift_magnitude",
            target_shift_magnitude,
        )
        object.__setattr__(self, "edge_probability", edge_probability)
        object.__setattr__(self, "coefficient_min_abs", coefficient_min_abs)
        object.__setattr__(self, "coefficient_max_abs", coefficient_max_abs)
        object.__setattr__(self, "scale_min", scale_min)
        object.__setattr__(self, "scale_max", scale_max)

    def k_for(self, n_nodes: int) -> int:
        """Return the independent experiment hyperparameter ``ceil(rho * n)``."""
        _positive_integer("n_nodes", n_nodes)
        return ceil(self.support_fraction * n_nodes)

    def parameter_budget_for(self, n_nodes: int) -> int:
        """Return the configured budget or the graph-independent DAG ceiling."""
        _positive_integer("n_nodes", n_nodes)
        if self.final_parameter_coordinate_budget is not None:
            return self.final_parameter_coordinate_budget
        return min(2 * self.k_for(n_nodes), 2 * n_nodes - 1)

    @property
    def max_parameter_coordinates(self) -> int:
        """Compatibility view for the smallest configured node count."""
        return self.parameter_budget_for(min(self.node_counts))


@dataclass(frozen=True)
class PilotRunSpec:
    """One deterministic cell in the recorded pilot configuration."""

    n_nodes: int
    graph_family: PilotGraphFamily
    policy: PilotPolicy
    seed: int

    def __post_init__(self) -> None:
        """Validate one run without accessing any hidden simulation state."""
        _positive_integer("n_nodes", self.n_nodes)
        if self.graph_family not in _GRAPH_FAMILIES:
            raise ValueError("graph_family must be a supported pilot family.")
        if self.policy not in _POLICIES:
            raise ValueError("policy must be a supported pilot policy.")
        _nonnegative_integer("seed", self.seed)


@dataclass(frozen=True)
class NoiselessPilotResult:
    """JSON-compatible metrics for one idealized graph-refinement run."""

    experiment_label: str
    spec: PilotRunSpec
    experiment_mode: ExperimentMode
    n: int
    rho: float
    k: int
    final_parameter_coordinate_budget: int
    target_node_support: tuple[int, ...]
    target_parameter_coordinate_support: tuple[str, ...]
    accepted_final_parameter_support: tuple[str, ...]
    initial_gaussian_kl: float
    final_gaussian_kl: float
    accepted_matching_kl: tuple[float, ...]
    exploration_trials: int
    candidate_evaluation_trials: int
    total_experiment_trials: int
    shift_coordinate_uses: int
    scale_coordinate_uses: int
    total_atomic_coordinate_uses: int
    final_unique_support_cost: int
    total_edges: int
    initial_oriented_edges: int
    final_oriented_edges: int
    initial_oriented_fraction: float
    final_oriented_fraction: float
    alternating_exploration_blocks: int
    runtime_seconds: float
    termination_reason: str
    shift_parameter_linf_error: float
    scale_parameter_linf_error: float
    parameter_recovered: bool
    policy_failed_or_no_action: bool
    result_schema_version: int = RESULT_SCHEMA_VERSION
    kl_direction: Literal["target_to_candidate"] = KL_DIRECTION
    intervention_semantics: str = INTERVENTION_SEMANTICS

    def __post_init__(self) -> None:
        """Validate key cross-metric accounting relationships."""
        if self.experiment_label != EXPERIMENT_LABEL:
            raise ValueError("experiment_label must identify the noiseless pilot.")
        if self.result_schema_version != RESULT_SCHEMA_VERSION:
            raise ValueError("result_schema_version must identify the current schema.")
        if self.intervention_semantics != INTERVENTION_SEMANTICS:
            raise ValueError("result has stale intervention semantics.")
        if self.kl_direction != KL_DIRECTION:
            raise ValueError("kl_direction must be 'target_to_candidate'.")
        if not isinstance(self.spec, PilotRunSpec):
            raise TypeError("spec must be a PilotRunSpec.")
        expected_mode: ExperimentMode = (
            "known_support_oracle"
            if self.spec.policy == "known_dag_oracle"
            else "idealized_unknown_graph_oracle"
        )
        if self.experiment_mode != expected_mode:
            raise ValueError("experiment_mode does not match the pilot policy.")
        _positive_integer("n", self.n)
        _positive_integer("k", self.k)
        if self.n != self.spec.n_nodes:
            raise ValueError("n must equal spec.n_nodes.")
        rho = _finite_real("rho", self.rho)
        if not 0.0 < rho <= 1.0:
            raise ValueError("rho must lie in (0, 1].")
        if not 0 <= self.final_parameter_coordinate_budget <= 2 * self.n:
            raise ValueError(
                "final parameter-coordinate budget exceeds the configured 2n cap."
            )
        for name in (
            "target_node_support",
            "target_parameter_coordinate_support",
            "accepted_final_parameter_support",
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(set(value)) != len(value):
                raise TypeError(f"{name} must be a duplicate-free tuple.")
        if len(self.accepted_final_parameter_support) != self.final_unique_support_cost:
            raise ValueError("accepted support labels must match final support cost.")
        nonnegative_counts = (
            "exploration_trials",
            "candidate_evaluation_trials",
            "total_experiment_trials",
            "shift_coordinate_uses",
            "scale_coordinate_uses",
            "total_atomic_coordinate_uses",
            "final_unique_support_cost",
            "final_parameter_coordinate_budget",
            "total_edges",
            "initial_oriented_edges",
            "final_oriented_edges",
            "alternating_exploration_blocks",
        )
        for name in nonnegative_counts:
            _nonnegative_integer(name, getattr(self, name))
        if (
            self.exploration_trials + self.candidate_evaluation_trials
            != self.total_experiment_trials
        ):
            raise ValueError("trial metrics must satisfy the budget partition.")
        if (
            self.shift_coordinate_uses + self.scale_coordinate_uses
            != self.total_atomic_coordinate_uses
        ):
            raise ValueError("atomic coordinate metrics must sum.")
        if not (
            0
            <= self.initial_oriented_edges
            <= self.final_oriented_edges
            <= self.total_edges
        ):
            raise ValueError("oriented-edge counts must be monotone and bounded.")


@dataclass(frozen=True)
class NoiselessPilotNumericalBatchResult:
    """An explicit pilot preflight batch without any scheduler execution."""

    specs: tuple[PilotRunSpec, ...]
    evaluation: OracleInterventionBatchResult

    def __post_init__(self) -> None:
        """Require a one-to-one mapping between specs and numerical results."""
        if not isinstance(self.specs, tuple) or not self.specs:
            raise TypeError("specs must be a nonempty tuple.")
        if any(not isinstance(spec, PilotRunSpec) for spec in self.specs):
            raise TypeError("specs must contain PilotRunSpec values.")
        if not isinstance(self.evaluation, OracleInterventionBatchResult):
            raise TypeError("evaluation must be an OracleInterventionBatchResult.")
        if len(self.specs) != len(self.evaluation.request_ids):
            raise ValueError("specs and evaluation must have equal lengths.")


def pilot_run_specs(config: NoiselessPilotConfig) -> tuple[PilotRunSpec, ...]:
    """Return the recorded grid without launching any experiments."""
    if not isinstance(config, NoiselessPilotConfig):
        raise TypeError("config must be a NoiselessPilotConfig.")
    return tuple(
        PilotRunSpec(n_nodes, graph_family, policy, seed)
        for seed in config.seeds
        for n_nodes in config.node_counts
        for graph_family in config.graph_families
        for policy in config.policies
        if not (policy == "exact_minmaxc" and n_nodes > config.exact_minmaxc_max_nodes)
    )


def pilot_configuration_payload(config: NoiselessPilotConfig) -> dict[str, object]:
    """Return a deterministic JSON-ready description of the unlaunched grid."""
    specs = pilot_run_specs(config)
    return {
        "experiment_label": EXPERIMENT_LABEL,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "intervention_semantics": INTERVENTION_SEMANTICS,
        "kl_direction": KL_DIRECTION,
        "full_grid_launched": False,
        "assumptions": [
            f"realisable {config.intervention_composition.replace('_', '-')} targets",
            "exact observational CPDAG",
            "noiseless public probe outcomes",
            "exact graph updates",
            "oracle analytic matching optimizer",
            "exploration policies receive no true DAG or coefficient matrix",
        ],
        "configuration": {
            **asdict(config),
            "max_parameter_coordinates": config.max_parameter_coordinates,
            "parameter_budgets_by_n": {
                str(n_nodes): config.parameter_budget_for(n_nodes)
                for n_nodes in config.node_counts
            },
        },
        "policy_applicability": {
            "exact_minmaxc": {
                "maximum_n_nodes": config.exact_minmaxc_max_nodes,
            }
        },
        "configured_run_count": len(specs),
        "runs": [asdict(spec) for spec in specs],
    }


def _problem_for_spec(
    config: NoiselessPilotConfig,
    spec: PilotRunSpec,
) -> GeneratedProblem:
    """Generate one deterministic realizable problem of the configured type."""
    child_sequences = np.random.SeedSequence(spec.seed).spawn(7)
    seeds = tuple(
        int(sequence.generate_state(1, dtype=np.uint64)[0])
        for sequence in child_sequences
    )
    graph_parameters: dict[str, object]
    if spec.graph_family == "path":
        graph_parameters = {"permute": False}
    elif spec.graph_family == "rooted_tree":
        graph_parameters = {"root": None}
    elif spec.graph_family in {"erdos_renyi", "moralized_erdos_renyi"}:
        graph_parameters = {"edge_probability": config.edge_probability}
    elif spec.graph_family == "barabasi_albert":
        graph_parameters = {"attachment_edges": min(2, spec.n_nodes - 1)}
    elif spec.graph_family == "complete":
        graph_parameters = {"node_order": None}
    elif spec.graph_family == "tree_of_cliques":
        graph_parameters = {
            "branching_degree": 2,
            "min_clique_size": 2,
            "max_clique_size": min(4, spec.n_nodes),
        }
    else:
        graph_parameters = {}
    graph_rng = np.random.default_rng(seeds[0])
    for _ in range(1_000):
        dag = generate_graph(
            spec.graph_family,
            spec.n_nodes,
            graph_rng,
            **graph_parameters,
        )
        eligible = np.array(
            [node for node, degree in dag.in_degree if degree > 0],
            dtype=int,
        )
        if (
            config.intervention_composition == "shift_only"
            or eligible.size >= config.target_support_size
        ):
            break
    else:
        raise ValueError(
            "deterministic pilot graph regeneration found too few scale-eligible nodes."
        )
    coefficients = sample_coefficients(
        dag,
        np.random.default_rng(seeds[1]),
        config.coefficient_min_abs,
        config.coefficient_max_abs,
    )
    noise_variance_seed = (
        config.noise_variance_seed
        if config.noise_variance_seed is not None
        else derive_noise_variance_seed(spec.seed)
    )
    scm = LinearGaussianSCM(
        dag,
        coefficients,
        noise_variances=generate_noise_variances(
            spec.n_nodes,
            config.noise_variance_regime,
            noise_variance_seed,
        ),
    )

    target_rng = np.random.default_rng(seeds[3])
    target_pool = (
        np.arange(spec.n_nodes)
        if config.intervention_composition == "shift_only"
        else eligible
    )
    target_nodes = target_rng.choice(
        target_pool,
        size=config.target_support_size,
        replace=False,
    )
    target_shift = np.zeros(spec.n_nodes)
    target_scale = np.ones(spec.n_nodes)
    if config.intervention_composition in {"shift_only", "mixed"}:
        target_shift[target_nodes] = config.target_shift_magnitude * target_rng.choice(
            np.array([-1.0, 1.0]),
            size=config.target_support_size,
        )
    if config.intervention_composition in {"scale_only", "mixed"}:
        target_scale[target_nodes] = np.exp(
            config.target_log_scale_magnitude
            * target_rng.choice(
                np.array([-1.0, 1.0]),
                size=config.target_support_size,
            )
        )
    target_intervention = ShiftScaleIntervention(target_shift, target_scale)
    target_mean, target_covariance = scm.interventional_moments(target_intervention)
    public = PublicProblemView(
        observational_samples=scm.sample_observational(
            config.n_observational_samples,
            np.random.default_rng(seeds[2]),
        ),
        target_mean=target_mean,
        target_covariance=target_covariance,
        current_intervention=ShiftScaleIntervention.identity(spec.n_nodes),
        algorithm_seeds=AlgorithmSeeds(seeds[5], seeds[6]),
    )
    return GeneratedProblem(
        public=public,
        ground_truth=SimulationGroundTruth(
            scm=scm,
            generating_intervention=target_intervention,
            seeds=SimulationSeeds(
                root_seed=spec.seed,
                graph_generation_seed=seeds[0],
                coefficient_generation_seed=seeds[1],
                observational_sampling_seed=seeds[2],
                target_intervention_seed=seeds[3],
                target_sampling_seed=seeds[4],
                noise_variance_seed=(
                    noise_variance_seed
                    if config.noise_variance_regime != "homoskedastic"
                    else None
                ),
            ),
        ),
    )


def _validate_pilot_spec(
    config: NoiselessPilotConfig,
    spec: PilotRunSpec,
) -> None:
    """Validate that one spec belongs to the recorded pilot grid."""
    if not isinstance(config, NoiselessPilotConfig):
        raise TypeError("config must be a NoiselessPilotConfig.")
    if not isinstance(spec, PilotRunSpec):
        raise TypeError("spec must be a PilotRunSpec.")
    if spec.n_nodes not in config.node_counts:
        raise ValueError("spec n_nodes must belong to the configured grid.")
    if spec.graph_family not in config.graph_families:
        raise ValueError("spec graph_family must belong to the configured grid.")
    if spec.policy not in config.policies:
        raise ValueError("spec policy must belong to the configured grid.")
    if spec.seed not in config.seeds:
        raise ValueError("spec seed must belong to the configured grid.")


def _pilot_request_id(spec: PilotRunSpec) -> str:
    """Return a stable identifier containing every pilot cell coordinate."""
    return (
        f"seed={spec.seed}/nodes={spec.n_nodes}/graph={spec.graph_family}/"
        f"policy={spec.policy}"
    )


def evaluate_noiseless_pilot_initial_batch(
    config: NoiselessPilotConfig,
    specs: tuple[PilotRunSpec, ...],
    compute_config: ComputeConfig | None = None,
    *,
    gradients: bool = True,
) -> NoiselessPilotNumericalBatchResult:
    """Batch initial numerical evaluations without running any pilot scheduler.

    Problem generation and all graph state stay on CPU.  This explicit
    preflight prepares only coefficients and target moments for numerical KL
    evaluation; it neither launches the configured grid nor changes the
    existing single-case smoke runner.
    """
    if not isinstance(config, NoiselessPilotConfig):
        raise TypeError("config must be a NoiselessPilotConfig.")
    if not isinstance(specs, tuple) or not specs:
        raise TypeError("specs must be a nonempty tuple.")
    for spec in specs:
        _validate_pilot_spec(config, spec)
    if len(set(specs)) != len(specs):
        raise ValueError("specs must not contain duplicate pilot cells.")
    if compute_config is not None and not isinstance(compute_config, ComputeConfig):
        raise TypeError("compute_config must be a ComputeConfig or None.")
    if not isinstance(gradients, bool):
        raise TypeError("gradients must be a boolean.")

    problems = tuple(_problem_for_spec(config, spec) for spec in specs)
    evaluator = OracleInterventionBatchEvaluator(
        tuple(
            OracleBatchProblem(_pilot_request_id(spec), problem)
            for spec, problem in zip(specs, problems, strict=True)
        ),
        compute_config,
        auto_operation=("value_and_gradient" if gradients else "value"),
    )
    evaluation = evaluator.evaluate(
        tuple(problem.public.current_intervention for problem in problems),
        gradients=gradients,
    )
    return NoiselessPilotNumericalBatchResult(specs, evaluation)


def _scheduler_settings(config: NoiselessPilotConfig) -> SchedulerSettings:
    """Build the shared event-driven scheduler controls."""
    return SchedulerSettings(
        loss_tolerance=config.loss_tolerance,
        global_loss_tolerance=config.loss_tolerance,
        improvement_tolerance=config.improvement_tolerance,
        patience=3,
        max_consecutive_exploration_trials=(config.max_consecutive_exploration_trials),
        max_probe_targets=config.max_probe_targets,
        scale_min=config.scale_min,
        scale_max=config.scale_max,
        allowed_coordinate_kinds=config.allowed_coordinate_kinds,  # type: ignore[arg-type]
        max_steps=config.max_steps,
    )


def _exploration_policy(
    config: NoiselessPilotConfig,
    spec: PilotRunSpec,
) -> object:
    """Construct one public-only exploration policy."""
    if spec.policy == "random":
        return RandomExplorationPolicy(
            max_probe_targets=config.max_probe_targets,
            shift_magnitude=config.target_shift_magnitude,
        )
    if spec.policy == "exact_minmaxc":
        if spec.n_nodes > config.exact_minmaxc_max_nodes:
            raise ValueError(
                "exact_minmaxc is configured only through "
                f"{config.exact_minmaxc_max_nodes} nodes."
            )
        return ExactMinMaxCExplorationPolicy(
            max_probe_targets=config.max_probe_targets,
            shift_magnitude=config.target_shift_magnitude,
            max_component_size=config.exact_minmaxc_max_nodes,
        )
    if spec.policy == "clique_tree":
        return CliqueTreeProbePolicy(
            max_probe_targets=config.max_probe_targets,
            shift_magnitude=config.target_shift_magnitude,
        )
    raise ValueError("known_dag_oracle does not construct an exploration policy.")


def _run_unknown_graph(
    config: NoiselessPilotConfig,
    spec: PilotRunSpec,
    problem: GeneratedProblem,
    loss: OracleAnalyticGaussianKLLoss,
) -> tuple[SchedulerResult, int, int, int]:
    """Run one policy through the public unknown-graph scheduler boundary."""
    validate_experiment_assembly(
        "idealized_unknown_graph_oracle",
        uses_true_dag=True,
        uses_true_coefficients=True,
        uses_oracle_loss=True,
    )
    initial_graph = cpdag_from_true_dag_for_simulation(problem.ground_truth.scm.dag)
    local_loss = CompositeGaussianLocalLoss(
        OracleCandidateMomentsProvider(problem.ground_truth.scm),
        LocalLossWeights(
            marginal=config.local_marginal_weight,
            pairwise=config.local_pairwise_weight,
            conditional=config.local_conditional_weight,
        ),
    )
    knowledge = UnknownGraphMatchingKnowledge(
        initial_graph=initial_graph,
        mismatch_provider=CompositeGaussianMismatchProvider(
            local_loss,
            unresolved_threshold=config.local_unresolved_threshold,
        ),
    )
    environment = ProbeOutcomeMatchingEnvironment(
        probe_environment=NoiselessOracleProbeEnvironment(problem.ground_truth.scm.dag),
        candidate_environment=OracleAnalyticSCMEnvironment(
            problem.ground_truth.scm,
            n_probe_samples=1,
        ),
    )
    scheduler = AlternatingMatchingScheduler(
        knowledge=knowledge,
        exploration_policy=_exploration_policy(config, spec),
        loss=loss,
        optimizer=OracleActiveCoordinateOptimizer(
            loss=loss,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
            max_iterations=config.optimizer_max_iterations,
        ),
        environment=environment,
        final_support_budget=FinalSupportBudget(
            config.parameter_budget_for(problem.public.n_nodes)
        ),
        settings=_scheduler_settings(config),
    )
    result = scheduler.run(
        problem.public,
        ExperimentBudgetLedger(config.max_experiment_trials),
    )
    return (
        result,
        len(initial_graph.directed_edges),
        len(knowledge.graph.directed_edges),
        problem.ground_truth.scm.dag.number_of_edges(),
    )


def _run_known_support(
    config: NoiselessPilotConfig,
    problem: GeneratedProblem,
) -> tuple[SchedulerResult, int, int, int]:
    """Run the known-DAG, known-B, known-coordinate-support oracle."""
    validate_experiment_assembly(
        "known_support_oracle",
        uses_true_dag=True,
        uses_true_coefficients=True,
        uses_oracle_loss=True,
    )
    target = problem.ground_truth.generating_intervention
    active_coordinates = tuple(
        ParameterCoordinate(int(node), "shift")
        for node in np.flatnonzero(target.shift != 0.0)
    ) + tuple(
        ParameterCoordinate(int(node), "log_scale")
        for node in np.flatnonzero(target.scale != 1.0)
    )
    result = run_known_support_oracle_baseline(
        problem,
        active_coordinates,
        _scheduler_settings(config),
        FinalSupportBudget(config.parameter_budget_for(problem.public.n_nodes)),
        config.max_experiment_trials,
        optimizer_max_iterations=config.optimizer_max_iterations,
    )
    total_edges = problem.ground_truth.scm.dag.number_of_edges()
    return result, total_edges, total_edges, total_edges


def _oriented_fraction(oriented_edges: int, total_edges: int) -> float:
    """Return one bounded orientation fraction, including the empty skeleton."""
    return 1.0 if total_edges == 0 else oriented_edges / total_edges


def _exploration_blocks(result: SchedulerResult) -> int:
    """Count maximal contiguous EXPLORE segments in scheduler history."""
    blocks = 0
    previous_was_exploration = False
    for record in result.history:
        is_exploration = record.phase == "EXPLORE"
        if is_exploration and not previous_was_exploration:
            blocks += 1
        previous_was_exploration = is_exploration
    return blocks


def run_noiseless_pilot_case(
    config: NoiselessPilotConfig,
    spec: PilotRunSpec,
    *,
    problem: GeneratedProblem | None = None,
) -> NoiselessPilotResult:
    """Execute one labeled idealized run; never expand it into the full grid."""
    _validate_pilot_spec(config, spec)

    if problem is None:
        problem = _problem_for_spec(config, spec)
    elif not isinstance(problem, GeneratedProblem):
        raise TypeError("problem must be GeneratedProblem or None.")
    elif problem.public.n_nodes != spec.n_nodes:
        raise ValueError("problem dimension must match the pilot run specification.")
    elif (
        len(problem.ground_truth.generating_intervention.targeted_nodes())
        != config.target_support_size
    ):
        raise ValueError("problem target support must match the pilot configuration.")
    loss = OracleAnalyticGaussianKLLoss(problem.ground_truth.scm)
    initial_kl = loss.value(problem.public.current_intervention, problem.public)
    start = perf_counter()
    if spec.policy == "known_dag_oracle":
        result, initial_oriented, final_oriented, total_edges = _run_known_support(
            config,
            problem,
        )
    else:
        result, initial_oriented, final_oriented, total_edges = _run_unknown_graph(
            config,
            spec,
            problem,
            loss,
        )
    runtime_seconds = perf_counter() - start

    target = problem.ground_truth.generating_intervention
    shift_error = float(np.max(np.abs(result.final_candidate.shift - target.shift)))
    scale_error = float(np.max(np.abs(result.final_candidate.scale - target.scale)))
    accepted_kl = tuple(
        float(record.loss_after)
        for record in result.history
        if record.phase == "EVALUATE"
        and record.accepted is True
        and record.loss_after is not None
    )
    ledger = result.budget_ledger
    target_shift_nodes = tuple(
        int(node) for node in np.flatnonzero(target.shift != 0.0)
    )
    target_scale_nodes = tuple(
        int(node) for node in np.flatnonzero(target.scale != 1.0)
    )
    target_node_support = tuple(
        sorted(set(target_shift_nodes) | set(target_scale_nodes))
    )
    target_parameter_support = tuple(
        [f"shift:{node}" for node in target_shift_nodes]
        + [f"log_scale:{node}" for node in target_scale_nodes]
    )
    final_shift_nodes = tuple(
        int(node) for node in np.flatnonzero(result.final_candidate.shift != 0.0)
    )
    final_scale_nodes = tuple(
        int(node) for node in np.flatnonzero(result.final_candidate.scale != 1.0)
    )
    accepted_parameter_support = tuple(
        [f"shift:{node}" for node in final_shift_nodes]
        + [f"log_scale:{node}" for node in final_scale_nodes]
    )
    experiment_mode: ExperimentMode = (
        "known_support_oracle"
        if spec.policy == "known_dag_oracle"
        else "idealized_unknown_graph_oracle"
    )
    return NoiselessPilotResult(
        experiment_label=EXPERIMENT_LABEL,
        spec=spec,
        experiment_mode=experiment_mode,
        n=spec.n_nodes,
        rho=config.support_fraction,
        k=config.k_for(spec.n_nodes),
        final_parameter_coordinate_budget=config.parameter_budget_for(spec.n_nodes),
        target_node_support=target_node_support,
        target_parameter_coordinate_support=target_parameter_support,
        accepted_final_parameter_support=accepted_parameter_support,
        initial_gaussian_kl=initial_kl,
        final_gaussian_kl=result.final_loss,
        accepted_matching_kl=accepted_kl,
        exploration_trials=ledger.exploration_trials,
        candidate_evaluation_trials=ledger.candidate_evaluation_trials,
        total_experiment_trials=ledger.experiment_trials_used,
        shift_coordinate_uses=ledger.shift_coordinate_uses,
        scale_coordinate_uses=ledger.scale_coordinate_uses,
        total_atomic_coordinate_uses=ledger.total_atomic_coordinate_uses,
        final_unique_support_cost=result.final_candidate.support_cost(),
        total_edges=total_edges,
        initial_oriented_edges=initial_oriented,
        final_oriented_edges=final_oriented,
        initial_oriented_fraction=_oriented_fraction(
            initial_oriented,
            total_edges,
        ),
        final_oriented_fraction=_oriented_fraction(
            final_oriented,
            total_edges,
        ),
        alternating_exploration_blocks=_exploration_blocks(result),
        runtime_seconds=runtime_seconds,
        termination_reason=result.termination_reason,
        shift_parameter_linf_error=shift_error,
        scale_parameter_linf_error=scale_error,
        parameter_recovered=(
            max(shift_error, scale_error) <= config.parameter_recovery_atol
        ),
        policy_failed_or_no_action=result.termination_reason not in _SUCCESS_REASONS,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one deterministic, human-readable JSON artifact."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_pilot_configuration(
    config: NoiselessPilotConfig,
    path: Path,
) -> None:
    """Persist the grid configuration without running it."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    _write_json(path, pilot_configuration_payload(config))


def write_pilot_result(result: NoiselessPilotResult, path: Path) -> None:
    """Persist one labeled smoke result."""
    if not isinstance(result, NoiselessPilotResult):
        raise TypeError("result must be a NoiselessPilotResult.")
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    _write_json(path, asdict(result))


def pilot_numerical_batch_payload(
    result: NoiselessPilotNumericalBatchResult,
) -> dict[str, object]:
    """Return a compact JSON-ready numerical-preflight record."""
    if not isinstance(result, NoiselessPilotNumericalBatchResult):
        raise TypeError("result must be a NoiselessPilotNumericalBatchResult.")
    evaluation = result.evaluation
    return {
        "experiment_label": EXPERIMENT_LABEL,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "kl_direction": KL_DIRECTION,
        "full_grid_launched": False,
        "scheduler_executed": False,
        "specs": [asdict(spec) for spec in result.specs],
        "operation": evaluation.operation,
        "request_ids": list(evaluation.request_ids),
        "values": evaluation.values.tolist(),
        "shift_gradients": (
            [gradient.tolist() for gradient in evaluation.shift_gradients]
            if evaluation.shift_gradients is not None
            else None
        ),
        "log_scale_gradients": (
            [gradient.tolist() for gradient in evaluation.log_scale_gradients]
            if evaluation.log_scale_gradients is not None
            else None
        ),
        "backend_diagnostics": asdict(evaluation.backend_diagnostics),
        "batches": [
            {
                **asdict(batch),
                "request_ids": list(batch.request_ids),
            }
            for batch in evaluation.batches
        ],
        "host_to_device_bytes": evaluation.host_to_device_bytes,
        "device_to_host_bytes": evaluation.device_to_host_bytes,
    }


def write_pilot_numerical_batch(
    result: NoiselessPilotNumericalBatchResult,
    path: Path,
) -> None:
    """Persist one explicit numerical preflight without scheduler state."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    _write_json(path, pilot_numerical_batch_payload(result))


def build_parser() -> argparse.ArgumentParser:
    """Build a CLI that can record the grid and run at most one smoke case."""
    parser = argparse.ArgumentParser(
        description=(
            "Record the noiseless graph-refinement pilot and optionally run "
            "one five-node CliqueTree smoke case."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--smoke",
        action="store_true",
        help="Run exactly one path/5-node/CliqueTree case after writing the grid.",
    )
    execution.add_argument(
        "--numerical-preflight",
        action="store_true",
        help=(
            "Evaluate one path/5-node initial KL and gradient without running "
            "a scheduler."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Numerical-preflight device; ignored modes reject non-auto values.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write pilot artifacts without exposing a full-grid launch option."""
    namespace = build_parser().parse_args(argv)
    if namespace.device != "auto" and not namespace.numerical_preflight:
        raise ValueError(
            "--device applies only to --numerical-preflight; the scheduler smoke "
            "remains on CPU."
        )
    output_dir: Path = namespace.output_dir
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"{output_dir} is not a directory.")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"{output_dir} must be empty.")
    else:
        output_dir.mkdir(parents=True)

    config = NoiselessPilotConfig()
    write_pilot_configuration(config, output_dir / "pilot_configuration.json")
    if namespace.numerical_preflight:
        preflight_spec = PilotRunSpec(
            n_nodes=5,
            graph_family="path",
            policy="known_dag_oracle",
            seed=config.seeds[0],
        )
        preflight = evaluate_noiseless_pilot_initial_batch(
            config,
            (preflight_spec,),
            ComputeConfig(device=namespace.device),
            gradients=True,
        )
        write_pilot_numerical_batch(
            preflight,
            output_dir / "numerical_preflight.json",
        )
        diagnostics = preflight.evaluation.backend_diagnostics
        print(
            f"{EXPERIMENT_LABEL}: numerical preflight used "
            f"{diagnostics.backend}/{diagnostics.device}; no scheduler run."
        )
    elif namespace.smoke:
        smoke_spec = PilotRunSpec(
            n_nodes=5,
            graph_family="path",
            policy="clique_tree",
            seed=config.seeds[0],
        )
        result = run_noiseless_pilot_case(config, smoke_spec)
        write_pilot_result(result, output_dir / "smoke_result.json")
        print(
            f"{EXPERIMENT_LABEL}: smoke terminated with "
            f"{result.termination_reason}; final KL(target || candidate)="
            f"{result.final_gaussian_kl:.6g}."
        )
    else:
        print(f"{EXPERIMENT_LABEL}: configuration recorded; no run launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
