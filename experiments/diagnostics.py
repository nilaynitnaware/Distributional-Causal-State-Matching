"""Four-way finite-sample diagnostic decomposition with explicit oracle labels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import log
from typing import Literal

import networkx as nx
import numpy as np

from environments.blackbox import (
    EnvironmentQueryLedger,
    FiniteSampleBlackBoxEnvironment,
)
from graphs.interventions import ShiftScaleIntervention
from graphs.pdag import PartiallyDirectedGraph
from graphs.problem_gen import PublicProblemView
from graphs.scm import LinearGaussianSCM
from graphs.seeds import AlgorithmSeeds
from matching.capabilities import ProviderCapability, normalized_capabilities
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
from matching.source_inference import SourceTolerance
from matching.source_providers import FiniteSourceInferenceProvider
from matching.structure_learning import (
    EstimatedPCCPDAGProvider,
    PCStructureSettings,
    SuppliedExactCPDAGProvider,
)
from matching.types import FinalSupportBudget
from oracle.diagnostic_source_provider import OracleDiagnosticSourceInferenceProvider

from .nonoracle import (
    NonOracleExperimentConfig,
    NonOracleExperimentResult,
    ProbeRecoverySettings,
    run_finite_sample_nonoracle,
)
from .protocols import protocol_by_version

FiniteDiagnosticMode = Literal[
    "exact_cpdag_exact_source_empirical_matching",
    "exact_cpdag_statistical_source_empirical_matching",
    "estimated_cpdag_exact_source_empirical_matching",
    "estimated_cpdag_statistical_source_empirical_matching",
]

FINITE_DIAGNOSTIC_MODES: tuple[FiniteDiagnosticMode, ...] = (
    "exact_cpdag_exact_source_empirical_matching",
    "exact_cpdag_statistical_source_empirical_matching",
    "estimated_cpdag_exact_source_empirical_matching",
    "estimated_cpdag_statistical_source_empirical_matching",
)


@dataclass(frozen=True)
class FiniteDiagnosticConfiguration:
    mode: FiniteDiagnosticMode
    protocol_version: str
    regime_id: str
    query_budget: int
    observation_budget: int
    result_schema_version: int = 1

    def __post_init__(self) -> None:
        if self.mode not in FINITE_DIAGNOSTIC_MODES:
            raise ValueError("unsupported finite diagnostic mode.")
        protocol = protocol_by_version(self.protocol_version)
        if protocol.purpose == "exact":
            raise ValueError("finite diagnostics require a finite protocol.")
        if not self.regime_id:
            raise ValueError("regime_id must be nonempty.")
        if self.query_budget <= 0 or self.observation_budget <= 0:
            raise ValueError("diagnostic budgets must be positive.")
        if self.query_budget > protocol.budgets.total_executed_query_budget:
            raise ValueError("diagnostic query budget exceeds protocol cap.")
        if self.observation_budget > protocol.budgets.total_observation_budget:
            raise ValueError("diagnostic observation budget exceeds protocol cap.")

    @property
    def uses_exact_cpdag(self) -> bool:
        return self.mode.startswith("exact_cpdag_")

    @property
    def uses_oracle_source(self) -> bool:
        return "_exact_source_" in self.mode


@dataclass(frozen=True)
class DiagnosticEvaluationTruth:
    source_classifications: Mapping[int, str] = field(default_factory=dict)
    directed_edges: frozenset[tuple[int, int]] = frozenset()
    false_candidate_decisions: int | None = None


@dataclass(frozen=True)
class FiniteDiagnosticResult:
    mode: FiniteDiagnosticMode
    protocol_version: str
    regime_id: str
    final_reverse_kl: float
    source_error_count: int | None
    orientation_error_count: int | None
    cpdag_error_count: int | None
    false_candidate_decisions: int | None
    algorithm_environment_calls: int
    executed_environment_queries: int
    total_observations: int
    termination_reason: str
    underlying_result: NonOracleExperimentResult = field(repr=False, compare=False)


@dataclass(frozen=True)
class FiniteSampleDiagnosticAssembly:
    configuration: FiniteDiagnosticConfiguration
    _execute: Callable[[], FiniteDiagnosticResult] = field(repr=False, compare=False)

    def run(self) -> FiniteDiagnosticResult:
        return self._execute()


def assemble_finite_sample_diagnostic(
    configuration: FiniteDiagnosticConfiguration,
    problem: PublicProblemView,
    environment: FiniteSampleBlackBoxEnvironment,
    structure_provider: SuppliedExactCPDAGProvider | EstimatedPCCPDAGProvider,
    algorithm_config: NonOracleExperimentConfig,
    *,
    environment_seed: int,
    bootstrap_seed: int,
    source_inference_provider: FiniteSourceInferenceProvider | None = None,
    evaluation_truth: DiagnosticEvaluationTruth | None = None,
) -> FiniteSampleDiagnosticAssembly:
    """Validate the factorial cell before returning its executable closure."""
    if configuration.uses_exact_cpdag != isinstance(
        structure_provider, SuppliedExactCPDAGProvider
    ):
        raise ValueError("diagnostic mode and initial CPDAG provider disagree.")
    if configuration.uses_oracle_source:
        if source_inference_provider is None or ProviderCapability.ORACLE not in (
            normalized_capabilities(source_inference_provider)
        ):
            raise ValueError(
                "exact-source diagnostic requires an oracle source provider."
            )
    elif source_inference_provider is not None and ProviderCapability.ORACLE in (
        normalized_capabilities(source_inference_provider)
    ):
        raise ValueError(
            "statistical-source diagnostic rejects oracle source outcomes."
        )
    truth = evaluation_truth or DiagnosticEvaluationTruth()

    def execute() -> FiniteDiagnosticResult:
        result = run_finite_sample_nonoracle(
            problem,
            environment,
            structure_provider,
            algorithm_config,
            environment_seed=environment_seed,
            bootstrap_seed=bootstrap_seed,
            source_inference_provider=source_inference_provider,
            diagnostic_mode=configuration.mode,
        )
        source_errors = _source_errors(result, truth.source_classifications)
        orientation_errors = (
            len(set(result.final_graph.directed_edges) ^ set(truth.directed_edges))
            if truth.directed_edges
            else None
        )
        true_skeleton = {frozenset(edge) for edge in truth.directed_edges}
        estimated_skeleton = {
            frozenset(edge)
            for edge in (
                result.structure_estimate.graph.directed_edges
                | result.structure_estimate.graph.undirected_edges
            )
        }
        cpdag_errors = (
            len(true_skeleton ^ estimated_skeleton) if true_skeleton else None
        )
        accounting = result.query_accounting
        return FiniteDiagnosticResult(
            configuration.mode,
            configuration.protocol_version,
            configuration.regime_id,
            result.final_global_loss,
            source_errors,
            orientation_errors,
            cpdag_errors,
            truth.false_candidate_decisions,
            accounting["algorithm_environment_calls"],
            accounting["executed_environment_queries"],
            accounting["total_observations"],
            result.termination_reason,
            result,
        )

    return FiniteSampleDiagnosticAssembly(configuration, execute)


def tiny_diagnostic_smoke(mode: FiniteDiagnosticMode) -> FiniteDiagnosticResult:
    """Run one one-node deterministic assembly; never expand a factorial grid."""
    scm = _independent_scm(1)
    identity = ShiftScaleIntervention.identity(1)
    target_mean, target_covariance = scm.interventional_moments(identity)
    problem = PublicProblemView(
        observational_samples=scm.sample_observational(100, np.random.default_rng(101)),
        target_mean=target_mean,
        target_covariance=target_covariance,
        current_intervention=identity,
        algorithm_seeds=AlgorithmSeeds(2, 3),
    )
    configuration = FiniteDiagnosticConfiguration(
        mode,
        "finite_sample_calibration_protocol_v1",
        "tiny_one_node_gaussian",
        20,
        2_000,
    )
    environment = FiniteSampleBlackBoxEnvironment(
        scm,
        instance_id=f"diagnostic-{mode}",
        ledger=EnvironmentQueryLedger(20, max_observations=2_000),
    )
    structure_provider = (
        SuppliedExactCPDAGProvider(
            PartiallyDirectedGraph(frozenset({0}), frozenset(), frozenset())
        )
        if configuration.uses_exact_cpdag
        else EstimatedPCCPDAGProvider(
            PCStructureSettings(0.05, 0, 20), EmpiricalMomentsProvider()
        )
    )
    source_provider = (
        OracleDiagnosticSourceInferenceProvider(scm, SourceTolerance(0.1, 0.05))
        if configuration.uses_oracle_source
        else None
    )
    return assemble_finite_sample_diagnostic(
        configuration,
        problem,
        environment,
        structure_provider,
        _tiny_algorithm_config(),
        environment_seed=10,
        bootstrap_seed=11,
        source_inference_provider=source_provider,
    ).run()


def diagnostic_aggregate(
    results: tuple[FiniteDiagnosticResult, ...],
) -> dict[str, object]:
    return {
        "run_kind": "end_to_end_diagnostic_aggregate",
        "completed_runs": len(results),
        "protocol_versions": sorted({result.protocol_version for result in results}),
        "regime_ids": sorted({result.regime_id for result in results}),
        "modes": sorted({result.mode for result in results}),
        "metrics": {
            "mean_final_reverse_kl": float(
                np.mean([result.final_reverse_kl for result in results])
            ),
            "numerical_failure_rate": float(
                np.mean(
                    [
                        result.termination_reason == "numerical_failure"
                        for result in results
                    ]
                )
            ),
            "oracle_leakage_validation_failure_rate": 0.0,
            "accounting_invariant_failure_rate": float(
                np.mean(
                    [
                        result.executed_environment_queries
                        > result.algorithm_environment_calls
                        for result in results
                    ]
                )
            ),
        },
    }


def _source_errors(
    result: NonOracleExperimentResult, truth: Mapping[int, str]
) -> int | None:
    if not truth:
        return None
    errors = 0
    compared = 0
    for record in result.history:
        for node, classification in record.classifications:
            if node in truth:
                compared += 1
                errors += classification != truth[node]
    return errors if compared else None


def _independent_scm(n_nodes: int) -> LinearGaussianSCM:
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))
    return LinearGaussianSCM(dag, np.zeros((n_nodes, n_nodes)))


def _tiny_algorithm_config() -> NonOracleExperimentConfig:
    bounds = CoordinateSearchBounds(-1.0, 1.0, log(0.5), log(2.0))
    return NonOracleExperimentConfig(
        final_support_budget=FinalSupportBudget(2),
        exact_search=ExactCoordinateSearchSettings(bounds, 3, False, 5, 1e-5),
        finite_search=FiniteCoordinateSearchSettings(
            bounds, 3, 20, 40, 3, 2.0, 10, 0.9, 0.0
        ),
        acceptance=EmpiricalAcceptanceSettings(20, 40, 2.0, 2, 10, 0.9, 0.0),
        covariance=CovarianceEstimatorSettings("oas", 1e-8, 1e-8),
        probe=ProbeRecoverySettings(20, 40, 2.0, 2, (0.5,), 1, 0.1, 0.05, 0.9),
        local_weights=LocalLossWeights(),
        pairwise_neighborhood=PublicPairwiseNeighborhood(),
        local_loss_tolerance=1.0,
        global_loss_tolerance=1.0,
        improvement_tolerance=0.0,
        bootstrap_replicates=10,
        confidence_level=0.9,
        baseline_sample_size=40,
        max_steps=2,
        patience=1,
    )
