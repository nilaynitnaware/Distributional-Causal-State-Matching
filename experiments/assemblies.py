"""Explicit runnable assemblies for all five scientific matching modes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from graphs.problem_gen import GeneratedProblem, PublicProblemView
from matching.blackbox import (
    ExactMomentBlackBoxEnvironment,
    FiniteSampleBlackBoxEnvironment,
)
from matching.capabilities import (
    AssemblyComponent,
    validate_experiment_components,
)
from matching.structure_learning import (
    EstimatedPCCPDAGProvider,
    SuppliedExactCPDAGProvider,
)
from matching.types import FinalSupportBudget, ParameterCoordinate
from oracle.comparator import OracleComparatorResult
from oracle.known_dag_baseline import (
    run_known_dag_unknown_support_oracle_baseline,
    run_known_support_oracle_baseline,
)

from .modes import ExperimentMode, validate_experiment_assembly
from .noiseless_unknown_graph_pilot import (
    NoiselessPilotConfig,
    PilotRunSpec,
    run_noiseless_pilot_case,
)
from .nonoracle import (
    NonOracleExperimentConfig,
    run_finite_sample_nonoracle,
    run_nonoracle_noiseless,
)
from .nonrealisable import (
    NonRealisableExperimentResult,
    audit_nonrealisable_result,
)

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class ModeAssumptions:
    """Machine-checkable oracle and support assumptions for one assembly."""

    uses_true_dag: bool
    uses_true_coefficients: bool
    uses_oracle_loss: bool
    uses_oracle_source_outcomes: bool
    knows_target_coordinate_support: bool


@dataclass(frozen=True)
class RunnableExperimentAssembly(Generic[ResultT]):
    """A validated zero-argument experiment execution closure."""

    mode: ExperimentMode
    assumptions: ModeAssumptions
    _execute: Callable[[], ResultT] = field(repr=False, compare=False)

    def run(self) -> ResultT:
        """Execute this one assembly; this method never expands a grid."""
        return self._execute()


def assemble_known_support_oracle(
    problem: GeneratedProblem,
    active_coordinates: tuple[ParameterCoordinate, ...],
    scheduler_settings: object,
    final_support_budget: FinalSupportBudget,
    max_experiment_trials: int,
    *,
    runner_kwargs: Mapping[str, object] | None = None,
) -> RunnableExperimentAssembly[object]:
    """Assemble N1 with known coordinate support but no copied values."""
    assumptions = ModeAssumptions(True, True, True, False, True)
    _validate_assumptions("known_support_oracle", assumptions)
    kwargs = dict(runner_kwargs or {})
    return RunnableExperimentAssembly(
        "known_support_oracle",
        assumptions,
        lambda: run_known_support_oracle_baseline(
            problem,
            active_coordinates,
            scheduler_settings,  # type: ignore[arg-type]
            final_support_budget,
            max_experiment_trials,
            **kwargs,
        ),
    )


def assemble_known_dag_oracle(
    problem: GeneratedProblem,
    scheduler_settings: object,
    final_support_budget: FinalSupportBudget,
    max_experiment_trials: int,
    *,
    runner_kwargs: Mapping[str, object] | None = None,
) -> RunnableExperimentAssembly[object]:
    """Assemble N2 with all coordinates exposed to greedy selection."""
    assumptions = ModeAssumptions(True, True, True, False, False)
    _validate_assumptions("known_dag_oracle", assumptions)
    kwargs = dict(runner_kwargs or {})
    return RunnableExperimentAssembly(
        "known_dag_oracle",
        assumptions,
        lambda: run_known_dag_unknown_support_oracle_baseline(
            problem,
            scheduler_settings,  # type: ignore[arg-type]
            final_support_budget,
            max_experiment_trials,
            **kwargs,
        ),
    )


def assemble_idealized_unknown_graph_oracle(
    config: NoiselessPilotConfig,
    spec: PilotRunSpec,
    *,
    problem: GeneratedProblem | None = None,
) -> RunnableExperimentAssembly[object]:
    """Assemble N3 from the already verified single-case pilot path."""
    if spec.policy == "known_dag_oracle":
        raise ValueError("idealized unknown-graph assembly requires a graph policy.")
    assumptions = ModeAssumptions(True, True, True, True, False)
    _validate_assumptions("idealized_unknown_graph_oracle", assumptions)
    return RunnableExperimentAssembly(
        "idealized_unknown_graph_oracle",
        assumptions,
        lambda: run_noiseless_pilot_case(config, spec, problem=problem),
    )


def assemble_nonoracle_noiseless(
    problem: PublicProblemView,
    environment: ExactMomentBlackBoxEnvironment,
    structure_provider: (SuppliedExactCPDAGProvider | EstimatedPCCPDAGProvider),
    config: NonOracleExperimentConfig,
) -> RunnableExperimentAssembly[object]:
    """Assemble N4 using exact public moments and derivative-free search."""
    assumptions = ModeAssumptions(False, False, False, False, False)
    mode: ExperimentMode = (
        "non_oracle_noiseless_estimated_cpdag"
        if isinstance(structure_provider, EstimatedPCCPDAGProvider)
        else "non_oracle_noiseless_supplied_cpdag"
    )
    _validate_assumptions(mode, assumptions)
    validate_experiment_components(
        "non_oracle_noiseless",
        (
            AssemblyComponent("environment", environment),
            AssemblyComponent("initial_structure", structure_provider),
        ),
    )
    return RunnableExperimentAssembly(
        mode,
        assumptions,
        lambda: run_nonoracle_noiseless(
            problem,
            environment,
            structure_provider,
            config,
            result_mode=mode,  # type: ignore[arg-type]
        ),
    )


def assemble_finite_sample_nonoracle(
    problem: PublicProblemView,
    environment: FiniteSampleBlackBoxEnvironment,
    structure_provider: (SuppliedExactCPDAGProvider | EstimatedPCCPDAGProvider),
    config: NonOracleExperimentConfig,
    *,
    environment_seed: int,
    bootstrap_seed: int,
) -> RunnableExperimentAssembly[object]:
    """Assemble N5 using only public samples and statistical decisions."""
    assumptions = ModeAssumptions(False, False, False, False, False)
    mode: ExperimentMode = (
        "finite_sample_non_oracle_estimated_cpdag"
        if isinstance(structure_provider, EstimatedPCCPDAGProvider)
        else "finite_sample_non_oracle_supplied_cpdag"
    )
    _validate_assumptions(mode, assumptions)
    validate_experiment_components(
        "finite_sample_non_oracle",
        (
            AssemblyComponent("environment", environment),
            AssemblyComponent("initial_structure", structure_provider),
        ),
    )
    return RunnableExperimentAssembly(
        mode,
        assumptions,
        lambda: run_finite_sample_nonoracle(
            problem,
            environment,
            structure_provider,
            config,
            environment_seed=environment_seed,
            bootstrap_seed=bootstrap_seed,
            result_mode=mode,  # type: ignore[arg-type]
        ),
    )


def assemble_exact_nonrealisable(
    problem: PublicProblemView,
    environment: ExactMomentBlackBoxEnvironment,
    structure_provider: (SuppliedExactCPDAGProvider | EstimatedPCCPDAGProvider),
    config: NonOracleExperimentConfig,
    *,
    evaluation_problem: GeneratedProblem | None = None,
    comparator: OracleComparatorResult | None = None,
) -> RunnableExperimentAssembly[NonRealisableExperimentResult]:
    """Assemble exact matching with non-realisable stopping and gap reporting."""
    base = assemble_nonoracle_noiseless(
        problem,
        environment,
        structure_provider,
        config,
    )
    return RunnableExperimentAssembly(
        "exact_nonrealisable",
        base.assumptions,
        lambda: audit_nonrealisable_result(
            base.run(),
            evaluation_problem=evaluation_problem,
            comparator=comparator,
        ),
    )


def assemble_finite_sample_nonrealisable(
    problem: PublicProblemView,
    environment: FiniteSampleBlackBoxEnvironment,
    structure_provider: (SuppliedExactCPDAGProvider | EstimatedPCCPDAGProvider),
    config: NonOracleExperimentConfig,
    *,
    environment_seed: int,
    bootstrap_seed: int,
    evaluation_problem: GeneratedProblem | None = None,
    comparator: OracleComparatorResult | None = None,
) -> RunnableExperimentAssembly[NonRealisableExperimentResult]:
    """Assemble empirical matching with non-realisable stopping and gaps."""
    base = assemble_finite_sample_nonoracle(
        problem,
        environment,
        structure_provider,
        config,
        environment_seed=environment_seed,
        bootstrap_seed=bootstrap_seed,
    )
    return RunnableExperimentAssembly(
        "finite_sample_nonrealisable",
        base.assumptions,
        lambda: audit_nonrealisable_result(
            base.run(),
            evaluation_problem=evaluation_problem,
            comparator=comparator,
        ),
    )


def _validate_assumptions(
    mode: ExperimentMode,
    assumptions: ModeAssumptions,
) -> None:
    validate_experiment_assembly(
        mode,
        uses_true_dag=assumptions.uses_true_dag,
        uses_true_coefficients=assumptions.uses_true_coefficients,
        uses_oracle_loss=assumptions.uses_oracle_loss,
    )
    if mode == "known_support_oracle":
        expected = (False, True)
    elif mode == "idealized_unknown_graph_oracle":
        expected = (True, False)
    else:
        expected = (False, False)
    if assumptions.uses_oracle_source_outcomes != expected[0]:
        raise ValueError(f"mode {mode!r} has inconsistent source assumptions.")
    if assumptions.knows_target_coordinate_support != expected[1]:
        raise ValueError(f"mode {mode!r} has inconsistent support assumptions.")
