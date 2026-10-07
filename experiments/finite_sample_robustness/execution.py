"""One-task execution for the isolated robustness campaign."""

from __future__ import annotations

import hashlib
import resource
import traceback
from dataclasses import asdict
from time import perf_counter

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.nonrealisable_targets import (
    external_gaussian_target,
    nonproportional_coefficient_target,
)
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from oracle.observational_cpdag import cpdag_from_true_dag_for_simulation
from oracle.sparse_shift_scale_kl import solve_sparse_shift_scale_kl
from query_complexity import FirstUnresolvedPolicy
from query_complexity.finite_sample import (
    FiniteLocalMechanism,
    FiniteSampleAlgorithmResult,
    FiniteSampleSettings,
    run_finite_sample_algorithm1,
)
from scripts.run_query_complexity_local_mechanism_campaign import _source

from .design import CampaignTask, derive_seed
from .graph_conditions import (
    CorruptionResult,
    corrupt_graph,
    estimate_pc_cpdag,
    graph_metrics,
    graph_payload,
    metadata_payload,
)

SUCCESS = "SUCCESS"


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


class DeterministicProbeProvider:
    """Lazy simulation environment with Stage-B-compatible per-node seeds."""

    def __init__(self, scm, task: CampaignTask, shift: float) -> None:
        self._scm = scm
        self._task = task
        self._shift = float(shift)
        self.n_int = int(task.N_int)
        self._queried: set[int] = set()
        self.sample_hashes: dict[int, str] = {}
        self.sampling_seconds = 0.0

    def query(self, node: int) -> np.ndarray:
        self._queried.add(int(node))
        action_shift = np.zeros(self._task.n)
        action_shift[node] = self._shift
        rng = np.random.default_rng(
            derive_seed(
                self._task.seed,
                self._task.n,
                self._task.graph_index,
                self._task.replicate,
                "probe",
                node,
            )
        )
        started = perf_counter()
        samples = self._scm.sample_interventional(
            ShiftScaleIntervention(action_shift, np.ones(self._task.n)),
            self.n_int,
            rng,
        )
        self.sampling_seconds += perf_counter() - started
        self.sample_hashes[node] = _array_hash(samples)
        return samples

    @property
    def physical_query_count(self) -> int:
        return len(self._queried)

    @property
    def queried_nodes(self) -> tuple[int, ...]:
        return tuple(sorted(self._queried))


def _sample_source(task: CampaignTask, scm) -> np.ndarray:
    rng = np.random.default_rng(
        derive_seed(
            task.seed,
            task.n,
            task.graph_index,
            task.replicate,
            "source_samples",
        )
    )
    # Always draw the Stage-B maximum pool, then take a prefix. This guarantees
    # exact pairing across the N_obs/N_graph axes.
    return scm.sample_observational(10_000, rng)[: task.N_obs]


def _target_for_family(config: dict[str, object], task: CampaignTask, scm, canonical):
    if task.target_family == "realisable":
        return canonical
    intervention = canonical.generating_intervention
    if intervention is None:
        raise RuntimeError("canonical Stage-B target lost its generating intervention")
    if task.target_family == "nonproportional_structural_coefficients":
        eligible = sorted(node for node, degree in scm.dag.in_degree if degree >= 2)
        if not eligible:
            raise ValueError(
                "no node with at least two parents for nonproportional target"
            )
        rng = np.random.default_rng(
            derive_seed(
                task.seed,
                task.source_problem_id,
                "nonproportional_target",
            )
        )
        node = int(rng.choice(np.asarray(eligible, dtype=int)))
        parents = np.asarray(sorted(scm.dag.predecessors(node)), dtype=int)
        parent = int(rng.choice(parents))
        return nonproportional_coefficient_target(
            scm,
            intervention,
            selected_node=node,
            selected_parent=parent,
            multiplier=1.35,
            configured_support_budget=min(6, task.n),
        )
    if task.target_family == "external_spd_gaussian":
        variances = 1.0 + 0.2 * np.arange(task.n)
        standard_deviations = np.sqrt(variances)
        correlation = np.full((task.n, task.n), 0.25)
        np.fill_diagonal(correlation, 1.0)
        covariance = correlation * np.outer(standard_deviations, standard_deviations)
        return external_gaussian_target(
            np.asarray(intervention.shift) + 0.1,
            covariance,
            public_metadata={
                "construction": "existing_external_spd_gaussian_pattern",
            },
        )
    raise ValueError(f"unsupported target family: {task.target_family}")


def _sample_target(task: CampaignTask, target) -> np.ndarray:
    rng = np.random.default_rng(
        derive_seed(
            task.seed,
            task.n,
            task.graph_index,
            task.replicate,
            "target_samples",
            task.target_family,
        )
    )
    # Direct Gaussian sampling is valid for every existing Gaussian target and
    # avoids accessing absent generating interventions for non-realisable cases.
    pool = rng.multivariate_normal(target.mean, target.covariance, size=10_000)
    return pool[: task.N_target]


def _parent_metrics(
    mechanisms: tuple[FiniteLocalMechanism, ...], scm
) -> dict[str, object]:
    tp = fp = fn = 0
    errors: list[float] = []
    node_errors: list[dict[str, object]] = []
    for item in mechanisms:
        truth = frozenset(int(parent) for parent in scm.dag.predecessors(item.node))
        estimated = frozenset(item.parents)
        tp += len(truth & estimated)
        fp += len(estimated - truth)
        fn += len(truth - estimated)
        true_row = (
            np.eye(item.mechanism_row.size)[item.node] - scm.coefficients[:, item.node]
        )
        error = float(np.linalg.norm(item.mechanism_row - true_row))
        errors.append(error)
        node_errors.append(
            {
                "node": item.node,
                "true_parents": sorted(truth),
                "estimated_parents": sorted(estimated),
                "mechanism_l2_error": error,
                "valid": item.valid,
                "invalid_reason": item.invalid_reason,
            }
        )
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "queried_parent_precision": precision,
        "queried_parent_recall": recall,
        "queried_parent_f1": f1,
        "queried_mechanism_l2_error": float(np.mean(errors)) if errors else None,
        "queried_mechanism_details": node_errors,
    }


def _graph_resolved_metrics(
    result: FiniteSampleAlgorithmResult, scm
) -> dict[str, object]:
    if result.final_graph is None:
        return {
            "graph_resolved_mechanism_l2_error": None,
            "number_analytically_recovered_mechanisms": 0,
            "graph_resolved_mechanism_details": [],
        }
    queried = set(result.queried_nodes)
    covariance = result.source_estimate.covariance
    details = []
    errors = []
    for node in range(result.source_estimate.mean.size):
        if node in queried or result.final_graph.undirected_neighbors_of(node):
            continue
        parents = tuple(sorted(result.final_graph.parents_of(node)))
        beta = np.zeros(result.source_estimate.mean.size)
        if parents:
            index = np.asarray(parents, dtype=int)
            beta[index] = np.linalg.solve(
                covariance[np.ix_(index, index)], covariance[index, node]
            )
        estimated_row = np.eye(beta.size)[node] - beta
        true_row = np.eye(beta.size)[node] - scm.coefficients[:, node]
        error = float(np.linalg.norm(estimated_row - true_row))
        errors.append(error)
        details.append(
            {
                "node": node,
                "estimated_parents": list(parents),
                "true_parents": sorted(scm.dag.predecessors(node)),
                "mechanism_l2_error": error,
            }
        )
    return {
        "graph_resolved_mechanism_l2_error": (
            float(np.mean(errors)) if errors else None
        ),
        "number_analytically_recovered_mechanisms": len(details),
        "graph_resolved_mechanism_details": details,
    }


def _support_jaccard(left: set[str], right: set[str]) -> float:
    return 1.0 if not left | right else len(left & right) / len(left | right)


def _orientation_metrics(result: FiniteSampleAlgorithmResult, scm) -> dict[str, object]:
    direct = [item for item in result.orientation_events if item.route == "direct"]
    meek = [item for item in result.orientation_events if item.route == "meek"]
    wrong_direct = sum(
        not scm.dag.has_edge(item.source, item.target) for item in direct
    )
    wrong_meek = sum(not scm.dag.has_edge(item.source, item.target) for item in meek)
    final = result.final_graph
    wrong_final = (
        sum(not scm.dag.has_edge(*edge) for edge in final.directed_edges)
        if final is not None
        else 0
    )
    unresolved = len(final.undirected_edges) if final is not None else 0
    final_skeleton = (
        {frozenset(edge) for edge in final.directed_edges | final.undirected_edges}
        if final is not None
        else set()
    )
    true_skeleton = {frozenset(edge) for edge in scm.dag.edges}
    skeleton_error = len(final_skeleton ^ true_skeleton)
    return {
        "number_direct_queries": result.physical_query_count,
        "direct_orientations": len(direct),
        "direct_orientation_errors": wrong_direct,
        "meek_propagated_orientations": len(meek),
        "meek_propagated_orientation_errors": wrong_meek,
        "final_wrong_directions": wrong_final,
        "final_unresolved_edges": unresolved,
        "final_skeleton_error": skeleton_error,
        "final_directed_graph_error": wrong_final + unresolved + skeleton_error,
    }


def _corruption_relevance(
    corruption: CorruptionResult | None,
    task: CampaignTask,
    scm,
    optimal_support: tuple[str, ...],
) -> dict[str, object]:
    if corruption is None or task.corruption_type != "edge_deletion":
        return {
            "deleted_optimal_action_parent_adjacencies": None,
            "deletion_relevance_label": None,
        }
    active_nodes = {
        int(label.split(":", 1)[1]) for label in optimal_support if ":" in label
    }
    relevant = 0
    for node_a, node_b in corruption.affected_edges:
        parent, child = (
            (node_a, node_b) if scm.dag.has_edge(node_a, node_b) else (node_b, node_a)
        )
        del parent
        relevant += child in active_nodes
    return {
        "deleted_optimal_action_parent_adjacencies": relevant,
        "deletion_relevance_label": (
            "optimal_action_parent_adjacency" if relevant else "unrelated_adjacency"
        ),
    }


def _failure_row(
    task: CampaignTask,
    *,
    status: str,
    stage: str,
    error: BaseException | str,
    stage_times: dict[str, float],
) -> dict[str, object]:
    message = str(error)
    return {
        **asdict(task),
        "schema_version": "finite_sample_robustness_row_v1",
        "status": status,
        "stage": stage,
        "error_type": type(error).__name__
        if isinstance(error, BaseException)
        else status,
        "error": message,
        "traceback": traceback.format_exc()
        if isinstance(error, BaseException)
        else None,
        "stage_runtime_seconds": stage_times,
        "runtime_seconds": float(sum(stage_times.values())),
        "catastrophic_failure": True,
    }


def run_task(
    config: dict[str, object], task: CampaignTask, *, bootstrap_replicates: int
) -> dict[str, object]:
    """Execute one task and always return a serializable success/failure row."""
    stage_times: dict[str, float] = {}
    started_total = perf_counter()
    try:
        started = perf_counter()
        regime = config["graph_regimes"][task.graph_index]
        source_config = dict(config)
        source_config["base_seed"] = task.seed
        scm, canonical_target = _source(source_config, regime, task.n, task.replicate)
        stage_times["source_problem_construction"] = perf_counter() - started

        try:
            started = perf_counter()
            target = _target_for_family(config, task, scm, canonical_target)
            stage_times["target_construction"] = perf_counter() - started
        except Exception as error:  # noqa: BLE001
            stage_times["target_construction"] = perf_counter() - started
            return _failure_row(
                task,
                status="TARGET_CONSTRUCTION_FAILED",
                stage="target_construction",
                error=error,
                stage_times=stage_times,
            )

        started = perf_counter()
        source_samples = _sample_source(task, scm)
        stage_times["observational_sampling"] = perf_counter() - started
        started = perf_counter()
        target_samples = _sample_target(task, target)
        stage_times["target_sampling"] = perf_counter() - started

        started = perf_counter()
        exact_cpdag = cpdag_from_true_dag_for_simulation(scm.dag)
        stage_times["exact_cpdag_construction"] = perf_counter() - started
        structure_metadata = None
        corruption = None
        try:
            started = perf_counter()
            if task.graph_condition == "estimated_pc_cpdag":
                initial_graph, structure_metadata = estimate_pc_cpdag(
                    source_samples, config["pc_estimator"]
                )
            elif task.corruption_type is not None:
                corruption = corrupt_graph(
                    exact_cpdag,
                    scm.dag,
                    corruption_type=task.corruption_type,
                    severity=str(task.corruption_severity),
                    seed=int(task.corruption_seed),
                )
                if corruption.status != SUCCESS or corruption.graph is None:
                    stage_times["graph_preparation"] = perf_counter() - started
                    return _failure_row(
                        task,
                        status=corruption.status,
                        stage="controlled_corruption",
                        error=corruption.note,
                        stage_times=stage_times,
                    )
                initial_graph = corruption.graph
            else:
                initial_graph = exact_cpdag.copy()
            stage_times["graph_preparation"] = perf_counter() - started
        except Exception as error:  # noqa: BLE001
            stage_times["graph_preparation"] = perf_counter() - started
            status = (
                "GRAPH_ESTIMATION_FAILED"
                if task.graph_condition == "estimated_pc_cpdag"
                else "INVALID_ESTIMATED_GRAPH"
            )
            return _failure_row(
                task,
                status=status,
                stage="graph_preparation",
                error=error,
                stage_times=stage_times,
            )

        before_graph_metrics = graph_metrics(initial_graph, exact_cpdag, scm.dag)

        started = perf_counter()
        try:
            oracle = solve_sparse_shift_scale_kl(
                scm=scm,
                target_mean=target.mean,
                target_covariance=target.covariance,
                budget=min(6, task.n),
                shift_bounds=tuple(float(value) for value in config["shift_bounds"]),
                log_scale_bounds=tuple(
                    float(np.log(value)) for value in config["scale_bounds"]
                ),
            )
        except Exception as error:  # noqa: BLE001
            stage_times["exact_sparse_oracle"] = perf_counter() - started
            return _failure_row(
                task,
                status="NUMERICAL_FAILURE",
                stage="exact_sparse_oracle",
                error=error,
                stage_times=stage_times,
            )
        stage_times["exact_sparse_oracle"] = perf_counter() - started

        provider = DeterministicProbeProvider(
            scm, task, float(config["probe_shift_magnitude"])
        )
        settings = FiniteSampleSettings(
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=float(config["confidence_level"]),
            orientation_confidence=float(config["orientation_confidence"]),
            null_quantile=float(config["null_quantile"]),
            support_probability_threshold=float(
                config["support_probability_threshold"]
            ),
            backend=str(config["backend"]),
            device=str(config["device"]),
        )
        started = perf_counter()
        try:
            result = run_finite_sample_algorithm1(
                source_samples=source_samples,
                target_samples=target_samples,
                probe_provider=provider,
                initial_graph=initial_graph,
                shift_magnitude=float(config["probe_shift_magnitude"]),
                budget=min(6, task.n),
                shift_bounds=tuple(float(value) for value in config["shift_bounds"]),
                scale_bounds=tuple(float(value) for value in config["scale_bounds"]),
                settings=settings,
                bootstrap_seed=derive_seed(
                    task.seed,
                    task.source_problem_id,
                    task.N_obs,
                    task.N_int,
                    task.graph_condition,
                    task.corruption_seed,
                    "bootstrap",
                ),
                singleton_shift_policy=FirstUnresolvedPolicy(),
            )
        except Exception as error:  # noqa: BLE001
            stage_times["algorithm_total"] = perf_counter() - started
            lowered = str(error).lower()
            status = (
                "NUMERICAL_FAILURE"
                if any(
                    token in lowered
                    for token in (
                        "singular",
                        "positive definite",
                        "inversion",
                        "nan",
                        "finite",
                    )
                )
                else "ALGORITHM_FAILED"
            )
            return _failure_row(
                task,
                status=status,
                stage="algorithm",
                error=error,
                stage_times=stage_times,
            )
        stage_times["algorithm_total"] = perf_counter() - started
        stage_times["singleton_intervention_sampling"] = provider.sampling_seconds

        started = perf_counter()
        initial_mean, initial_covariance = scm.observational_moments()
        final_mean, final_covariance = scm.interventional_moments(result.intervention)
        initial_true = gaussian_kl_target_to_candidate(
            target.mean, target.covariance, initial_mean, initial_covariance
        )
        final_true = gaussian_kl_target_to_candidate(
            target.mean, target.covariance, final_mean, final_covariance
        )
        returned_support = set(result.dp_result.budgeted_support)
        optimal_support = set(oracle.budgeted_support)
        local_mode_disagreement = int(
            sum(
                left != right
                for left, right in zip(
                    result.dp_result.selected_budgeted_state_labels,
                    oracle.selected_budgeted_state_labels,
                    strict=True,
                )
            )
        )
        stage_times["evaluation"] = perf_counter() - started
        stage_times["task_total"] = perf_counter() - started_total

        result_graph = result.final_graph
        final_graph_payload = (
            graph_payload(result_graph) if result_graph is not None else None
        )
        row = {
            **asdict(task),
            "schema_version": "finite_sample_robustness_row_v1",
            "status": SUCCESS,
            "stage": "complete",
            "error_type": None,
            "error": None,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": derive_seed(
                task.seed,
                task.source_problem_id,
                task.N_obs,
                task.N_int,
                task.graph_condition,
                task.corruption_seed,
                "bootstrap",
            ),
            "backend": config["backend"],
            "device": config["device"],
            "initial_true_KL": initial_true,
            "final_true_KL": final_true,
            "K_B_star": oracle.raw_K_B_star,
            "excess_KL": final_true - oracle.raw_K_B_star,
            "K_B_star_strictly_positive": oracle.raw_K_B_star > 1e-12,
            "returned_shift": result.intervention.shift.tolist(),
            "returned_scale": result.intervention.scale.tolist(),
            "returned_support": sorted(returned_support),
            "population_optimal_shift": oracle.budgeted_shift.tolist(),
            "population_optimal_scale": oracle.budgeted_scale.tolist(),
            "population_optimal_support": sorted(optimal_support),
            "support_jaccard": _support_jaccard(returned_support, optimal_support),
            "support_disagreement": returned_support != optimal_support,
            "selected_local_modes": list(
                result.dp_result.selected_budgeted_state_labels
            ),
            "population_optimal_local_modes": list(
                oracle.selected_budgeted_state_labels
            ),
            "local_mode_disagreement_count": local_mode_disagreement,
            "local_mode_disagreement_fraction": local_mode_disagreement / task.n,
            "T_q": result.physical_query_count,
            "T_q_over_n": result.physical_query_count / task.n,
            "queried_nodes": list(result.queried_nodes),
            "total_intervention_observations": result.total_intervention_observations,
            "total_observations": result.total_observations,
            "invalid_mechanism_estimates": result.invalid_estimates,
            "termination_reason": result.termination_reason,
            "oas_shrinkage": result.source_estimate.shrinkage,
            "target_oas_shrinkage": result.target_estimate.shrinkage,
            "covariance_minimum_eigenvalue": result.source_estimate.minimum_eigenvalue,
            "covariance_condition_number": result.source_estimate.covariance_condition_number,
            "precision_condition_number": result.source_estimate.precision_condition_number,
            "numerical_inversion_failure": result.source_estimate.inversion_failed,
            "graph_metrics_before_algorithm": before_graph_metrics,
            "structure_estimation_metadata": metadata_payload(structure_metadata),
            "initial_public_graph": graph_payload(initial_graph),
            "exact_evaluation_cpdag": graph_payload(exact_cpdag),
            "final_public_graph": final_graph_payload,
            "corruption": asdict(corruption) if corruption is not None else None,
            "observational_sample_sha256": _array_hash(source_samples),
            "target_sample_sha256": _array_hash(target_samples),
            "probe_sample_sha256_by_node": provider.sample_hashes,
            "stage_runtime_seconds": stage_times,
            "runtime_seconds": stage_times["task_total"],
            "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
            "catastrophic_failure": final_true >= initial_true,
            **_parent_metrics(result.mechanisms, scm),
            **_graph_resolved_metrics(result, scm),
            **_orientation_metrics(result, scm),
            **_corruption_relevance(corruption, task, scm, oracle.budgeted_support),
        }
        return row
    except Exception as error:  # noqa: BLE001
        stage_times.setdefault("task_total", perf_counter() - started_total)
        return _failure_row(
            task,
            status="ALGORITHM_FAILED",
            stage="unhandled_task_exception",
            error=error,
            stage_times=stage_times,
        )
