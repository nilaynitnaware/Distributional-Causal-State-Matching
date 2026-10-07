"""Profile the existing NumPy/SciPy CPU implementation.

This harness deliberately exercises the current public implementation without
adding a numerical backend or changing scientific behavior.  It stores compact
timing summaries and selected cProfile statistics, never raw samples or profiler
traces.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import os
import platform
import pstats
import resource
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import networkx as nx
import numpy as np
import scipy

from generate import save_problem
from graphs.DAG_gen import generate_erdos_renyi_dag, generate_path_dag
from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import (
    GeneratedProblem,
    PublicProblemView,
    SimulationGroundTruth,
)
from graphs.scm import LinearGaussianSCM, sample_coefficients
from graphs.seeds import AlgorithmSeeds, SimulationSeeds
from matching.graph_updates import update_graph_from_probe
from matching.probes import ShiftProbe
from matching.scheduler import AlternatingMatchingScheduler, SchedulerSettings
from matching.types import (
    CandidateProposal,
    ExperimentBudgetLedger,
    FinalSupportBudget,
    ParameterCoordinate,
)
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from oracle.analytic_gaussian_kl import OracleAnalyticGaussianKLLoss
from oracle.known_dag_baseline import (
    OracleSCMEnvironment,
    run_known_dag_oracle_baseline,
)
from oracle.noiseless_probe_environment import NoiselessOracleProbeEnvironment
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation
from oracle.oracle_optimizer import OracleActiveCoordinateOptimizer

FULL_NODE_COUNTS = (5, 10, 20, 50, 100, 200, 500)
QUICK_NODE_COUNTS = (5, 20, 100)
GRADIENT_NODE_COUNTS = frozenset((5, 10, 20, 50))
OPTIMIZER_NODE_COUNTS = frozenset((5, 10, 20))
SERIALIZATION_NODE_COUNTS = frozenset((5, 20, 100, 500))
BASE_SEED = 20_260_726
DTYPE = "float64"
BACKEND = "numpy"
DEVICE = "cpu"


@dataclass(frozen=True)
class _Fixture:
    """Prepared immutable inputs excluded from operation timings."""

    scm: LinearGaussianSCM
    problem: GeneratedProblem
    target: ShiftScaleIntervention
    candidate: ShiftScaleIntervention
    candidate_mean: np.ndarray
    candidate_covariance: np.ndarray
    active_coordinates: tuple[ParameterCoordinate, ...]


def _edge_probability(n_nodes: int) -> float:
    """Use a sparse graph with expected total degree near four."""
    return min(1.0, 4.0 / max(1, n_nodes - 1))


def _sample_count(n_nodes: int) -> int:
    """Bound memory and runtime while retaining large sample workloads."""
    if n_nodes >= 500:
        return 2_000
    if n_nodes >= 200:
        return 5_000
    return 10_000


def _simulation_seeds(seed: int) -> SimulationSeeds:
    return SimulationSeeds(
        root_seed=seed,
        graph_generation_seed=seed + 1,
        coefficient_generation_seed=seed + 2,
        observational_sampling_seed=seed + 3,
        target_intervention_seed=seed + 4,
        target_sampling_seed=seed + 5,
    )


def _make_fixture(n_nodes: int, seed: int) -> _Fixture:
    """Build one stable, moderately conditioned numerical problem."""
    dag = generate_erdos_renyi_dag(
        n_nodes,
        _edge_probability(n_nodes),
        np.random.default_rng(seed),
    )
    coefficients = sample_coefficients(
        dag,
        np.random.default_rng(seed + 1),
        0.05,
        0.20,
    )
    scm = LinearGaussianSCM(dag, coefficients)

    target_shift = np.zeros(n_nodes, dtype=float)
    target_shift[0] = 0.35
    if n_nodes > 1:
        target_shift[-1] = -0.25
    target_scale = np.ones(n_nodes, dtype=float)
    if n_nodes > 1:
        target_scale[1] = 1.15
    target = ShiftScaleIntervention(target_shift, target_scale)

    candidate_shift = 0.4 * target_shift
    candidate_scale = np.sqrt(target_scale)
    candidate = ShiftScaleIntervention(candidate_shift, candidate_scale)
    target_mean, target_covariance = scm.interventional_moments(target)
    candidate_mean, candidate_covariance = scm.interventional_moments(candidate)
    public = PublicProblemView(
        observational_samples=scm.sample_observational(
            8,
            np.random.default_rng(seed + 2),
        ),
        target_mean=target_mean,
        target_covariance=target_covariance,
        current_intervention=ShiftScaleIntervention.identity(n_nodes),
        algorithm_seeds=AlgorithmSeeds(seed + 6, seed + 7),
    )
    problem = GeneratedProblem(
        public=public,
        ground_truth=SimulationGroundTruth(
            scm=scm,
            generating_intervention=target,
            seeds=_simulation_seeds(seed),
        ),
    )
    active_coordinates = [ParameterCoordinate(0, "shift")]
    if n_nodes > 1:
        active_coordinates.extend(
            (
                ParameterCoordinate(1, "log_scale"),
                ParameterCoordinate(n_nodes - 1, "shift"),
            )
        )
    return _Fixture(
        scm=scm,
        problem=problem,
        target=target,
        candidate=candidate,
        candidate_mean=candidate_mean,
        candidate_covariance=candidate_covariance,
        active_coordinates=tuple(active_coordinates),
    )


def _time_callable(
    function: Callable[[], object],
    *,
    warmup_runs: int = 1,
    requested_repetitions: int = 5,
    minimum_batch_seconds: float = 0.03,
    maximum_inner_iterations: int = 100,
) -> dict[str, object]:
    """Return stable per-call timings without retaining function results."""
    for _ in range(warmup_runs):
        function()

    start = perf_counter()
    function()
    calibration_seconds = perf_counter() - start
    if calibration_seconds <= 0.0:
        inner_iterations = maximum_inner_iterations
    else:
        inner_iterations = min(
            maximum_inner_iterations,
            max(1, math.ceil(minimum_batch_seconds / calibration_seconds)),
        )
    repetitions = 3 if calibration_seconds >= 0.25 else requested_repetitions

    timings = []
    for _ in range(repetitions):
        start = perf_counter()
        for _ in range(inner_iterations):
            function()
        elapsed = perf_counter() - start
        timings.append(elapsed / inner_iterations)

    return {
        "warmup_runs": warmup_runs,
        "repetitions": repetitions,
        "inner_iterations": inner_iterations,
        "timings_seconds": timings,
        "median_seconds": statistics.median(timings),
        "minimum_seconds": min(timings),
        "maximum_seconds": max(timings),
    }


def _benchmark_record(
    operation: str,
    category: str,
    n_nodes: int,
    function: Callable[[], object],
    *,
    sample_count: int = 0,
    candidate_count: int = 1,
) -> dict[str, object]:
    timing = _time_callable(function)
    median_seconds = float(timing["median_seconds"])
    return {
        "operation": operation,
        "category": category,
        "n_nodes": n_nodes,
        "batch_size": 1,
        "candidate_count": candidate_count,
        "sample_count": sample_count,
        "dtype": DTYPE,
        "backend": BACKEND,
        "device": DEVICE,
        "preparation_time": 0.0,
        "transfer_time": 0.0,
        "compute_time": median_seconds,
        "total_time": median_seconds,
        "peak_memory": None,
        "maximum_absolute_error": None,
        "maximum_relative_error": None,
        "speedup": None,
        **timing,
    }


class _OneStepKnowledge:
    def unresolved_variables_remain(self, state: object) -> bool:
        return True

    def actionable_coordinates(
        self,
        state: object,
    ) -> tuple[ParameterCoordinate, ...]:
        return (ParameterCoordinate(0, "shift"),)

    def update_from_probe(self, observation: object, state: object) -> None:
        raise AssertionError("the orchestration benchmark must not explore")


class _NoProbePolicy:
    def propose_probe(
        self,
        public_state: object,
        knowledge: object,
        budget: object,
        rng: np.random.Generator,
    ) -> None:
        return None


class _CheapLoss:
    def value(
        self,
        candidate: ShiftScaleIntervention,
        problem_view: PublicProblemView,
    ) -> float:
        return 1.0 if candidate.shift[0] == 0.0 else 0.0


class _CheapOptimizer:
    def optimize(
        self,
        current_candidate: ShiftScaleIntervention,
        active_coordinates: tuple[ParameterCoordinate, ...],
        problem_view: PublicProblemView,
        final_support_budget: FinalSupportBudget,
    ) -> CandidateProposal:
        shift = current_candidate.shift.copy()
        shift[0] = 1.0
        intervention = ShiftScaleIntervention(shift, current_candidate.scale)
        return CandidateProposal(intervention, active_coordinates, 0.0)


class _CheapEnvironment:
    candidate_evaluation_requires_experiment = False

    def execute_probe(
        self,
        current_candidate: ShiftScaleIntervention,
        proposal: ShiftProbe,
        problem_view: PublicProblemView,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise AssertionError("the orchestration benchmark must not explore")

    def evaluate_candidate(
        self,
        candidate: ShiftScaleIntervention,
        problem_view: PublicProblemView,
        loss: _CheapLoss,
        rng: np.random.Generator,
    ) -> float:
        return loss.value(candidate, problem_view)


def _scheduler_settings() -> SchedulerSettings:
    return SchedulerSettings(
        loss_tolerance=1e-10,
        improvement_tolerance=0.0,
        patience=3,
        max_consecutive_exploration_trials=3,
        max_probe_targets=1,
        scale_min=0.25,
        scale_max=3.0,
        max_steps=8,
    )


def _scheduler_orchestration_function(n_nodes: int) -> Callable[[], object]:
    problem_view = PublicProblemView(
        observational_samples=np.zeros((1, n_nodes), dtype=float),
        target_mean=np.zeros(n_nodes, dtype=float),
        target_covariance=np.eye(n_nodes, dtype=float),
        current_intervention=ShiftScaleIntervention.identity(n_nodes),
        algorithm_seeds=AlgorithmSeeds(BASE_SEED, BASE_SEED + 1),
    )
    scheduler = AlternatingMatchingScheduler(
        knowledge=_OneStepKnowledge(),
        exploration_policy=_NoProbePolicy(),
        loss=_CheapLoss(),
        optimizer=_CheapOptimizer(),
        environment=_CheapEnvironment(),
        final_support_budget=FinalSupportBudget(1),
        settings=_scheduler_settings(),
    )

    def run() -> object:
        return scheduler.run(problem_view, ExperimentBudgetLedger(1))

    return run


def _graph_update_function(n_nodes: int, seed: int) -> Callable[[], object]:
    dag = generate_path_dag(n_nodes, np.random.default_rng(seed), permute=False)
    cpdag = cpdag_from_true_dag_for_simulation(dag)
    increment = np.zeros(n_nodes, dtype=float)
    increment[0] = 0.5
    probe = ShiftProbe(increment, frozenset((0,)), "cpu-baseline")
    outcome = NoiselessOracleProbeEnvironment(dag).execute_probe(
        ShiftScaleIntervention.identity(n_nodes),
        probe,
    )

    def update() -> object:
        return update_graph_from_probe(cpdag, outcome)

    return update


def _serialization_function(
    fixture: _Fixture,
    n_nodes: int,
    sample_count: int,
    seed: int,
) -> Callable[[], object]:
    samples = fixture.scm.sample_observational(
        sample_count,
        np.random.default_rng(seed),
    )
    public = PublicProblemView(
        observational_samples=samples,
        target_mean=fixture.problem.public.target_mean,
        target_covariance=fixture.problem.public.target_covariance,
        current_intervention=fixture.problem.public.current_intervention,
        algorithm_seeds=fixture.problem.public.algorithm_seeds,
    )
    problem = GeneratedProblem(public=public, ground_truth=fixture.problem.ground_truth)

    def serialize() -> object:
        with TemporaryDirectory(prefix="shift_scale_cpu_baseline_") as directory:
            return save_problem(problem, Path(directory), 0, {"n_nodes": n_nodes})

    return serialize


def _optimizer_function(fixture: _Fixture) -> Callable[[], object]:
    loss = OracleAnalyticGaussianKLLoss(fixture.scm)
    optimizer = OracleActiveCoordinateOptimizer(
        loss=loss,
        scale_min=0.25,
        scale_max=3.0,
        max_iterations=200,
    )

    def optimize() -> object:
        return optimizer.optimize(
            fixture.problem.public.current_intervention,
            fixture.active_coordinates,
            fixture.problem.public,
            FinalSupportBudget(len(fixture.active_coordinates)),
        )

    return optimize


def _scheduler_iteration_function(fixture: _Fixture) -> Callable[[], object]:
    def run() -> object:
        return run_known_dag_oracle_baseline(
            fixture.problem,
            fixture.active_coordinates,
            _scheduler_settings(),
            FinalSupportBudget(len(fixture.active_coordinates)),
            max_experiment_trials=1,
            optimizer_max_iterations=200,
        )

    return run


def _profile_function(function: Callable[[], object]) -> dict[str, object]:
    """Return selected cProfile rows without writing a raw trace."""
    profiler = cProfile.Profile()
    profiler.runcall(function)
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, function_name), values in sorted(
        stats.stats.items(),
        key=lambda item: item[1][3],
        reverse=True,
    )[:25]:
        primitive_calls, total_calls, self_time, cumulative_time, _ = values
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}({function_name})",
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_time": self_time,
                "cumulative_time": cumulative_time,
            }
        )
    return {
        "total_time": stats.total_tt,
        "top_functions_by_cumulative_time": rows,
    }


def _category_breakdown(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize an explicitly documented equal-call representative mix."""
    selectors = (
        ("graph_generation", 100),
        ("coefficient_generation", 100),
        ("graph_policy_update", 100),
        ("interventional_moments", 100),
        ("gaussian_kl_target_to_candidate", 100),
        ("analytic_gradient", 20),
        ("full_candidate_evaluation", 100),
        ("observational_sampling", 100),
        ("interventional_sampling", 100),
        ("optimizer_step", 10),
        ("serialization", 100),
        ("scheduler_orchestration", 100),
    )
    selected = []
    for operation, preferred_n_nodes in selectors:
        candidates = [record for record in records if record["operation"] == operation]
        if not candidates:
            raise ValueError(f"missing benchmark operation {operation!r}")
        selected.append(
            min(
                candidates,
                key=lambda record: abs(int(record["n_nodes"]) - preferred_n_nodes),
            )
        )

    category_seconds: dict[str, float] = {}
    for record in selected:
        category = str(record["category"])
        category_seconds[category] = category_seconds.get(category, 0.0) + float(
            record["median_seconds"]
        )
    total_seconds = sum(category_seconds.values())
    return [
        {
            "category": category,
            "seconds": seconds,
            "percentage": 100.0 * seconds / total_seconds,
        }
        for category, seconds in sorted(
            category_seconds.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "networkx": nx.__version__,
        "backend": BACKEND,
        "device": DEVICE,
        "dtype": DTYPE,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def run_cpu_baseline(node_counts: tuple[int, ...]) -> dict[str, object]:
    """Run controlled CPU timings and return a JSON-serializable summary."""
    fixtures = {
        n_nodes: _make_fixture(n_nodes, BASE_SEED + n_nodes) for n_nodes in node_counts
    }
    records: list[dict[str, object]] = []
    scheduler_profile_function: Callable[[], object] | None = None

    for n_nodes in node_counts:
        seed = BASE_SEED + 10_000 + n_nodes
        fixture = fixtures[n_nodes]
        sample_count = _sample_count(n_nodes)
        graph_function = lambda n=n_nodes, s=seed: generate_erdos_renyi_dag(
            n,
            _edge_probability(n),
            np.random.default_rng(s),
        )
        records.append(
            _benchmark_record(
                "graph_generation",
                "graph_operations",
                n_nodes,
                graph_function,
            )
        )
        records.append(
            _benchmark_record(
                "coefficient_generation",
                "graph_operations",
                n_nodes,
                lambda f=fixture, s=seed: sample_coefficients(
                    f.scm.dag,
                    np.random.default_rng(s),
                    0.05,
                    0.20,
                ),
            )
        )
        records.append(
            _benchmark_record(
                "observational_sampling",
                "sampling",
                n_nodes,
                lambda f=fixture, count=sample_count, s=seed: (
                    f.scm.sample_observational(count, np.random.default_rng(s))
                ),
                sample_count=sample_count,
            )
        )
        records.append(
            _benchmark_record(
                "interventional_sampling",
                "sampling",
                n_nodes,
                lambda f=fixture, count=sample_count, s=seed: (
                    f.scm.sample_interventional(
                        f.candidate,
                        count,
                        np.random.default_rng(s),
                    )
                ),
                sample_count=sample_count,
            )
        )
        records.append(
            _benchmark_record(
                "interventional_moments",
                "numerical_linear_algebra",
                n_nodes,
                lambda f=fixture: f.scm.interventional_moments(f.candidate),
            )
        )
        records.append(
            _benchmark_record(
                "gaussian_kl_target_to_candidate",
                "numerical_linear_algebra",
                n_nodes,
                lambda f=fixture: gaussian_kl_target_to_candidate(
                    target_mean=f.problem.public.target_mean,
                    target_covariance=f.problem.public.target_covariance,
                    candidate_mean=f.candidate_mean,
                    candidate_covariance=f.candidate_covariance,
                ),
            )
        )
        loss = OracleAnalyticGaussianKLLoss(fixture.scm)
        if n_nodes in GRADIENT_NODE_COUNTS:
            records.append(
                _benchmark_record(
                    "analytic_gradient",
                    "numerical_linear_algebra",
                    n_nodes,
                    lambda f=fixture, current_loss=loss: (
                        current_loss.value_and_gradient(
                            f.candidate,
                            f.problem.public,
                        )
                    ),
                )
            )
        if n_nodes in OPTIMIZER_NODE_COUNTS:
            optimizer_function = _optimizer_function(fixture)
            records.append(
                _benchmark_record(
                    "optimizer_step",
                    "optimization",
                    n_nodes,
                    optimizer_function,
                )
            )
            scheduler_iteration = _scheduler_iteration_function(fixture)
            records.append(
                _benchmark_record(
                    "scheduler_iteration_end_to_end",
                    "composite",
                    n_nodes,
                    scheduler_iteration,
                )
            )
            if n_nodes == 10:
                scheduler_profile_function = scheduler_iteration
        environment = OracleSCMEnvironment(fixture.scm, n_probe_samples=1)
        records.append(
            _benchmark_record(
                "full_candidate_evaluation",
                "numerical_linear_algebra",
                n_nodes,
                lambda f=fixture, current_loss=loss, current_environment=environment, s=seed: (
                    current_environment.evaluate_candidate(
                        f.candidate,
                        f.problem.public,
                        current_loss,
                        np.random.default_rng(s),
                    )
                ),
            )
        )
        records.append(
            _benchmark_record(
                "observational_cpdag",
                "graph_operations",
                n_nodes,
                lambda f=fixture: cpdag_from_true_dag_for_simulation(f.scm.dag),
            )
        )
        records.append(
            _benchmark_record(
                "graph_policy_update",
                "graph_operations",
                n_nodes,
                _graph_update_function(n_nodes, seed),
            )
        )
        if n_nodes in SERIALIZATION_NODE_COUNTS:
            records.append(
                _benchmark_record(
                    "serialization",
                    "serialization",
                    n_nodes,
                    _serialization_function(
                        fixture,
                        n_nodes,
                        sample_count,
                        seed,
                    ),
                    sample_count=sample_count,
                )
            )
        records.append(
            _benchmark_record(
                "scheduler_orchestration",
                "python_orchestration",
                n_nodes,
                _scheduler_orchestration_function(n_nodes),
            )
        )

    if scheduler_profile_function is None:
        profile_fixture = _make_fixture(10, BASE_SEED + 10)
        scheduler_profile_function = _scheduler_iteration_function(profile_fixture)

    result = {
        "schema_version": 2,
        "kl_direction": "target_to_candidate",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": _environment(),
        "methodology": {
            "timer": "time.perf_counter",
            "statistic": "median per-call wall time",
            "warmup_runs": 1,
            "requested_repetitions": 5,
            "adaptive_minimum_batch_seconds": 0.03,
            "maximum_inner_iterations": 100,
            "large_operation_repetitions": 3,
            "node_counts": list(node_counts),
            "sample_count_policy": {
                "n_nodes_below_200": 10_000,
                "n_nodes_200_to_499": 5_000,
                "n_nodes_500_and_above": 2_000,
            },
            "numerical_fixture": (
                "sparse Erdos-Renyi DAG with expected total degree near four "
                "and coefficient magnitudes in [0.05, 0.20]"
            ),
            "graph_policy_update_fixture": (
                "unpermuted path CPDAG, a noiseless probe at node zero, "
                "and public Meek closure"
            ),
            "optimizer_measurement": (
                "one complete bounded SciPy L-BFGS-B optimize call over the "
                "known nonidentity target coordinates"
            ),
            "category_percentage_basis": (
                "sum of one representative median call per selected operation; "
                "this is a controlled operation mix, not an end-to-end experiment grid"
            ),
            "gradient_limit": (
                "analytic gradients stop at 50 nodes because static inspection "
                "shows a per-node loop of dense matrix solves"
            ),
            "optimizer_limit": (
                "optimizer and end-to-end scheduler timings stop at 20 nodes "
                "to avoid an uncontrolled large run before hotspot confirmation"
            ),
        },
        "operations": records,
        "category_breakdown": _category_breakdown(records),
        "cprofile": {
            "workload": "one 10-node known-DAG scheduler iteration",
            **_profile_function(scheduler_profile_function),
        },
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    return result


def _write_result(result: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile the existing NumPy/SciPy CPU implementation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("cpu_baseline.json"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 5, 20, and 100 nodes instead of the representative full sizes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    node_counts = QUICK_NODE_COUNTS if namespace.quick else FULL_NODE_COUNTS
    result = run_cpu_baseline(node_counts)
    _write_result(result, namespace.output)
    print(
        f"wrote {len(result['operations'])} CPU benchmark records to {namespace.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
