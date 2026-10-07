"""Known-DAG oracle components and end-to-end baseline assembly."""

from dataclasses import dataclass, field
from numbers import Real
from typing import Literal

import networkx as nx
import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import GeneratedProblem, PublicProblemView
from graphs.scm import LinearGaussianSCM
from matching.losses import DifferentiableInterventionLoss, InterventionLoss
from matching.scheduler import (
    AlgorithmStateView,
    AlternatingMatchingScheduler,
    BudgetView,
    CandidateEvaluation,
    ExplorationObservation,
    MatchingKnowledge,
    SchedulerResult,
    SchedulerSettings,
)
from matching.types import (
    ExperimentBudgetLedger,
    FinalSupportBudget,
    ParameterCoordinate,
    ProbeProposal,
)
from metrics.gaussian_kl import gaussian_kl_target_to_candidate

from .analytic_gaussian_kl import OracleAnalyticGaussianKLLoss
from .oracle_optimizer import OracleActiveCoordinateOptimizer


def _frozen_indexed_dag(dag: object) -> nx.DiGraph:
    """Validate, defensively copy, and freeze an indexed simple DAG."""
    if not isinstance(dag, nx.DiGraph) or dag.is_multigraph():
        raise TypeError("dag must be a simple networkx.DiGraph.")
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("dag must be acyclic.")
    n_nodes = dag.number_of_nodes()
    if any(
        isinstance(node, bool) or not isinstance(node, (int, np.integer))
        for node in dag.nodes
    ) or {int(node) for node in dag.nodes} != set(range(n_nodes)):
        raise ValueError("dag nodes must be labeled 0, ..., n_nodes - 1.")
    return nx.freeze(nx.DiGraph(dag))


def _nonnegative_integer(name: str, value: object) -> int:
    """Validate one nonnegative integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _positive_integer(name: str, value: object) -> int:
    """Validate one positive integer."""
    normalized = _nonnegative_integer(name, value)
    if normalized == 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _finite_nonzero_real(name: str, value: object) -> float:
    """Validate one finite, nonzero real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    if normalized == 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return normalized


@dataclass(frozen=True)
class OracleKnownDAGKnowledge:
    """Known-DAG, known-B, known-coordinate-support oracle knowledge.

    Coordinate support remains an explicit input and hidden target values are
    never copied. ``required_probe_count``
    only gates when that set becomes visible so tests can exercise scheduler
    alternation; it is not an unknown-graph inference claim.
    """

    dag: nx.DiGraph
    controlled_coordinates: tuple[ParameterCoordinate, ...]
    required_probe_count: int = 0
    observed_probe_count: int = field(
        default=0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Freeze the true DAG and validate the controlled coordinate set."""
        object.__setattr__(self, "dag", _frozen_indexed_dag(self.dag))
        if not isinstance(self.controlled_coordinates, tuple):
            raise TypeError("controlled_coordinates must be a tuple.")
        if any(
            not isinstance(coordinate, ParameterCoordinate)
            for coordinate in self.controlled_coordinates
        ):
            raise TypeError(
                "controlled_coordinates must contain ParameterCoordinate objects."
            )
        if any(
            coordinate.kind == "log_scale" and self.dag.in_degree(coordinate.node) == 0
            for coordinate in self.controlled_coordinates
        ):
            raise ValueError("root-node scale coordinates are not actionable.")
        if len(set(self.controlled_coordinates)) != len(self.controlled_coordinates):
            raise ValueError("controlled_coordinates must not contain duplicates.")
        n_nodes = self.dag.number_of_nodes()
        if any(
            coordinate.node >= n_nodes for coordinate in self.controlled_coordinates
        ):
            raise ValueError("controlled coordinate nodes must fit the true DAG.")
        object.__setattr__(
            self,
            "required_probe_count",
            _nonnegative_integer(
                "required_probe_count",
                self.required_probe_count,
            ),
        )

    def unresolved_variables_remain(self, state: AlgorithmStateView) -> bool:
        """Treat positive public loss as unresolved for this oracle baseline."""
        self._validate_state_dimension(state)
        return state.current_loss > 0.0

    def actionable_coordinates(
        self,
        state: AlgorithmStateView,
    ) -> tuple[ParameterCoordinate, ...]:
        """Expose the controlled set after the optional probe gate."""
        self._validate_state_dimension(state)
        if self.observed_probe_count < self.required_probe_count:
            return ()
        return self.controlled_coordinates

    def update_from_probe(
        self,
        observation: ExplorationObservation,
        state: AlgorithmStateView,
    ) -> None:
        """Record a public probe without claiming graph inference."""
        self._validate_state_dimension(state)
        if not isinstance(observation, ExplorationObservation):
            raise TypeError("observation must be an ExplorationObservation.")
        if observation.intervention.shift.shape != (self.dag.number_of_nodes(),):
            raise ValueError("probe dimension must agree with the true DAG.")
        object.__setattr__(
            self,
            "observed_probe_count",
            self.observed_probe_count + 1,
        )

    def _validate_state_dimension(self, state: AlgorithmStateView) -> None:
        """Validate public scheduler state against the known DAG."""
        if not isinstance(state, AlgorithmStateView):
            raise TypeError("state must be an AlgorithmStateView.")
        if state.problem_view.n_nodes != self.dag.number_of_nodes():
            raise ValueError("public state dimension must agree with the true DAG.")


@dataclass(frozen=True)
class OracleSourceProbePolicy:
    """Probe a known source of the true DAG with one additive shift."""

    dag: nx.DiGraph
    shift_magnitude: float

    def __post_init__(self) -> None:
        """Freeze the true DAG and validate the probe magnitude."""
        object.__setattr__(self, "dag", _frozen_indexed_dag(self.dag))
        object.__setattr__(
            self,
            "shift_magnitude",
            _finite_nonzero_real("shift_magnitude", self.shift_magnitude),
        )

    def propose_probe(
        self,
        public_state: AlgorithmStateView,
        knowledge: MatchingKnowledge,
        budget: BudgetView,
        rng: np.random.Generator,
    ) -> ProbeProposal | None:
        """Return a deterministic one-source shift when budget remains."""
        del knowledge, rng
        if not isinstance(public_state, AlgorithmStateView):
            raise TypeError("public_state must be an AlgorithmStateView.")
        if not isinstance(budget, BudgetView):
            raise TypeError("budget must be a BudgetView.")
        if public_state.problem_view.n_nodes != self.dag.number_of_nodes():
            raise ValueError("public state dimension must agree with the true DAG.")
        if budget.remaining_experiment_trials == 0:
            return None
        sources = sorted(
            int(node) for node, in_degree in self.dag.in_degree if in_degree == 0
        )
        if not sources:
            return None
        source = sources[public_state.step % len(sources)]
        shift_increment = np.zeros(self.dag.number_of_nodes())
        shift_increment[source] = self.shift_magnitude
        return ProbeProposal(
            shift_increment=shift_increment,
            reason=f"oracle-known-source-{source}",
        )


@dataclass(frozen=True)
class OracleSCMEnvironment:
    """Compatibility environment with an explicit evaluation mode.

    New code sets ``candidate_evaluation_mode`` instead of inferring scientific
    semantics from whether sampling configuration happens to be present.

    """

    scm: LinearGaussianSCM
    n_probe_samples: int
    candidate_evaluation_requires_experiment: bool = False
    n_candidate_evaluation_samples: int = 0
    candidate_evaluation_mode: Literal["oracle_analytic", "empirical"] | None = None
    covariance_regularization: float = 1e-8

    def __post_init__(self) -> None:
        """Validate hidden environment and explicit evaluation mode."""
        if not isinstance(self.scm, LinearGaussianSCM):
            raise TypeError("scm must be a LinearGaussianSCM.")
        object.__setattr__(
            self,
            "n_probe_samples",
            _positive_integer("n_probe_samples", self.n_probe_samples),
        )
        if not isinstance(self.candidate_evaluation_requires_experiment, bool):
            raise TypeError(
                "candidate_evaluation_requires_experiment must be a boolean."
            )
        mode = self.candidate_evaluation_mode
        if mode is None:
            mode = (
                "empirical"
                if self.candidate_evaluation_requires_experiment
                else "oracle_analytic"
            )
        if mode not in ("oracle_analytic", "empirical"):
            raise ValueError(
                "candidate_evaluation_mode must be oracle_analytic or empirical."
            )
        object.__setattr__(self, "candidate_evaluation_mode", mode)
        object.__setattr__(
            self,
            "candidate_evaluation_requires_experiment",
            mode == "empirical",
        )
        evaluation_samples = _nonnegative_integer(
            "n_candidate_evaluation_samples",
            self.n_candidate_evaluation_samples,
        )
        if mode == "empirical" and evaluation_samples < 2:
            raise ValueError(
                "n_candidate_evaluation_samples must be at least two in empirical mode."
            )
        object.__setattr__(
            self,
            "n_candidate_evaluation_samples",
            evaluation_samples,
        )
        regularization = _finite_nonzero_real(
            "covariance_regularization",
            self.covariance_regularization,
        )
        if regularization < 0.0:
            raise ValueError("covariance_regularization must be positive.")
        object.__setattr__(self, "covariance_regularization", regularization)

    def execute_probe(
        self,
        current_candidate: ShiftScaleIntervention,
        proposal: ProbeProposal,
        problem_view: PublicProblemView,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample the hidden SCM under a temporary probe."""
        intervention = proposal.apply_to(
            current_candidate,
            max_probe_targets=len(proposal.targets),
        )
        self._validate_public_request(intervention, problem_view, rng)
        return self.scm.sample_interventional(
            intervention,
            self.n_probe_samples,
            rng,
        )

    def evaluate_candidate(
        self,
        candidate: ShiftScaleIntervention,
        problem_view: PublicProblemView,
        loss: InterventionLoss,
        rng: np.random.Generator,
    ) -> CandidateEvaluation | float:
        """Evaluate analytically or from queried samples, never mixing the two."""
        self._validate_public_request(candidate, problem_view, rng)
        if not isinstance(loss, InterventionLoss):
            raise TypeError("loss must implement InterventionLoss.")
        if self.candidate_evaluation_mode == "oracle_analytic":
            return loss.value(candidate, problem_view)

        samples = self.scm.sample_interventional(
            candidate,
            self.n_candidate_evaluation_samples,
            rng,
        )
        empirical_mean = np.mean(samples, axis=0)
        empirical_covariance = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))
        empirical_covariance = empirical_covariance + (
            self.covariance_regularization * np.eye(problem_view.n_nodes)
        )
        empirical_loss = gaussian_kl_target_to_candidate(
            problem_view.target_mean,
            problem_view.target_covariance,
            empirical_mean,
            empirical_covariance,
        )
        return CandidateEvaluation(
            loss=empirical_loss,
            mean=empirical_mean,
            covariance=empirical_covariance,
            samples=samples,
            sample_count=samples.shape[0],
            covariance_regularization=self.covariance_regularization,
            loss_standard_error=None,
        )

    def _validate_public_request(
        self,
        intervention: ShiftScaleIntervention,
        problem_view: PublicProblemView,
        rng: np.random.Generator,
    ) -> None:
        """Validate public request dimensions against the hidden SCM."""
        if not isinstance(intervention, ShiftScaleIntervention):
            raise TypeError("intervention must be a ShiftScaleIntervention.")
        if not isinstance(problem_view, PublicProblemView):
            raise TypeError("problem_view must be a PublicProblemView.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        n_nodes = self.scm.dag.number_of_nodes()
        if intervention.shift.shape != (n_nodes,):
            raise ValueError("intervention dimension must agree with the hidden SCM.")
        if problem_view.n_nodes != n_nodes:
            raise ValueError("problem view dimension must agree with the hidden SCM.")


class OracleAnalyticSCMEnvironment(OracleSCMEnvironment):
    """Unambiguously exact evaluator that never charges candidate trials."""

    def __init__(self, scm: LinearGaussianSCM, n_probe_samples: int) -> None:
        super().__init__(
            scm=scm,
            n_probe_samples=n_probe_samples,
            candidate_evaluation_mode="oracle_analytic",
        )


class EmpiricalSCMEnvironment(OracleSCMEnvironment):
    """Sample-derived evaluator whose candidate executions require trials."""

    def __init__(
        self,
        scm: LinearGaussianSCM,
        n_probe_samples: int,
        n_candidate_evaluation_samples: int,
        covariance_regularization: float = 1e-8,
    ) -> None:
        super().__init__(
            scm=scm,
            n_probe_samples=n_probe_samples,
            candidate_evaluation_mode="empirical",
            n_candidate_evaluation_samples=n_candidate_evaluation_samples,
            covariance_regularization=covariance_regularization,
        )


def run_known_support_oracle_baseline(
    problem: GeneratedProblem,
    active_coordinates: tuple[ParameterCoordinate, ...],
    scheduler_settings: SchedulerSettings,
    final_support_budget: FinalSupportBudget,
    max_experiment_trials: int,
    *,
    required_probe_count: int = 0,
    probe_shift_magnitude: float = 0.5,
    n_probe_samples: int = 100,
    candidate_evaluation_requires_experiment: bool = False,
    n_candidate_evaluation_samples: int = 0,
    optimizer_max_iterations: int = 1_000,
    loss: DifferentiableInterventionLoss | None = None,
) -> SchedulerResult:
    """Run the known-DAG, known-B, known-coordinate-support oracle baseline.

    Hidden target values are recovered by optimization and are never copied.
    The canonical analytic CPU loss remains the default. An explicit
    differentiable loss may be injected to exercise the same optimizer,
    scheduler, environment, budgets, and acceptance rules on another
    numerical backend.
    """
    if not isinstance(problem, GeneratedProblem):
        raise TypeError("problem must be a GeneratedProblem.")
    if not isinstance(scheduler_settings, SchedulerSettings):
        raise TypeError("scheduler_settings must be SchedulerSettings.")
    if not isinstance(final_support_budget, FinalSupportBudget):
        raise TypeError("final_support_budget must be a FinalSupportBudget.")

    scm = problem.ground_truth.scm
    active_loss = OracleAnalyticGaussianKLLoss(scm) if loss is None else loss
    if not isinstance(active_loss, DifferentiableInterventionLoss):
        raise TypeError("loss must implement DifferentiableInterventionLoss.")
    environment = (
        EmpiricalSCMEnvironment(
            scm,
            n_probe_samples,
            n_candidate_evaluation_samples,
        )
        if candidate_evaluation_requires_experiment
        else OracleAnalyticSCMEnvironment(scm, n_probe_samples)
    )
    scheduler = AlternatingMatchingScheduler(
        knowledge=OracleKnownDAGKnowledge(
            dag=scm.dag,
            controlled_coordinates=active_coordinates,
            required_probe_count=required_probe_count,
        ),
        exploration_policy=OracleSourceProbePolicy(
            dag=scm.dag,
            shift_magnitude=probe_shift_magnitude,
        ),
        loss=active_loss,
        optimizer=OracleActiveCoordinateOptimizer(
            loss=active_loss,
            scale_min=scheduler_settings.scale_min,
            scale_max=scheduler_settings.scale_max,
            support_atol=scheduler_settings.support_atol,
            max_iterations=optimizer_max_iterations,
        ),
        environment=environment,
        final_support_budget=final_support_budget,
        settings=scheduler_settings,
    )
    return scheduler.run(
        problem.public,
        ExperimentBudgetLedger(max_experiment_trials),
    )


def run_known_dag_unknown_support_oracle_baseline(
    problem: GeneratedProblem,
    scheduler_settings: SchedulerSettings,
    final_support_budget: FinalSupportBudget,
    max_experiment_trials: int,
    **kwargs: object,
) -> SchedulerResult:
    """Run known DAG/B matching without receiving hidden target support.

    Both shift and log-scale coordinates at every node enter the scheduler's
    ordinary one-coordinate best-improvement selection. Hidden target values
    and hidden support are never copied into the knowledge provider.
    """
    if not isinstance(problem, GeneratedProblem):
        raise TypeError("problem must be a GeneratedProblem.")
    coordinates = tuple(
        ParameterCoordinate(node, kind)
        for node in range(problem.public.n_nodes)
        for kind in (
            ("shift", "log_scale")
            if problem.ground_truth.scm.dag.in_degree(node) > 0
            else ("shift",)
        )
    )
    return run_known_support_oracle_baseline(
        problem,
        coordinates,
        scheduler_settings,
        final_support_budget,
        max_experiment_trials,
        **kwargs,
    )


# Historical compatibility: old callers supplied support to this misleading name.
run_known_dag_oracle_baseline = run_known_support_oracle_baseline
