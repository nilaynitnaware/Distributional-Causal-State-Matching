#!/usr/bin/env python3
"""Small public-query audit for additive quadratic reconstruction and sparse DP.

This script is deliberately separate from production matching.  Hidden SCM
parameters are used only to construct simulation environments and the analytic
evaluation oracle.  Reconstruction sees only public target moments and exact
moment environment query results.
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from unittest.mock import patch

import networkx as nx
import numpy as np

import oracle.sparse_shift_scale_kl as sparse_oracle
from environments.blackbox import (
    EnvironmentQueryLedger,
    ExactMomentBlackBoxEnvironment,
)
from graphs.interventions import ShiftScaleIntervention
from graphs.nonrealisable_targets import (
    correlated_noise_target,
    external_gaussian_target,
    graph_perturbed_target,
    nonproportional_coefficient_target,
)
from graphs.scm import LinearGaussianSCM
from graphs.support import DEFAULT_SUPPORT_ATOL
from graphs.target_gen import GaussianTarget, generate_realisable_target
from matching.types import InterventionBackground
from metrics.gaussian_kl import gaussian_kl_target_to_candidate

SHIFT_BOUNDS = (-2.0, 2.0)
LOG_SCALE_BOUNDS = (float(np.log(0.25)), float(np.log(3.0)))
SCALE_BOUNDS = tuple(float(np.exp(value)) for value in LOG_SCALE_BOUNDS)
BUDGET = 6
LOSS_ATOL = 1e-12
LOSS_RTOL = 1e-11
DEGENERACY_ATOL = 1e-14
JOINT_DRAWS = 8


@dataclass(frozen=True)
class Quadratic:
    A: float
    B: float
    C: float
    D: float
    E: float

    def value(self, shift: float, dscale: float) -> float:
        return float(
            self.A * shift**2
            + self.B * shift
            + self.C * dscale**2
            + self.D * dscale
            + self.E * shift * dscale
        )

    def array(self) -> np.ndarray:
        return np.array([self.A, self.B, self.C, self.D, self.E])


@dataclass(frozen=True)
class AuditProblem:
    label: str
    scm: LinearGaussianSCM
    target: GaussianTarget


def _base_scm(noise_variances: np.ndarray | None = None) -> LinearGaussianSCM:
    edges = ((0, 2), (1, 2), (0, 3), (2, 3), (1, 4), (3, 4))
    dag = nx.DiGraph()
    dag.add_nodes_from(range(5))
    dag.add_edges_from(edges)
    coefficients = np.zeros((5, 5))
    for parent, child, value in (
        (0, 2, 0.60),
        (1, 2, -0.45),
        (0, 3, 0.35),
        (2, 3, 0.70),
        (1, 4, 0.50),
        (3, 4, -0.40),
    ):
        coefficients[parent, child] = value
    return LinearGaussianSCM(
        dag,
        coefficients,
        noise_variances=noise_variances,
    )


def _hidden_intervention() -> ShiftScaleIntervention:
    return ShiftScaleIntervention(
        np.array([0.45, -0.30, 0.70, -0.55, 0.25]),
        np.array([1.0, 1.0, 1.55, 0.68, 1.30]),
    )


def make_problems() -> tuple[AuditProblem, ...]:
    scm = _base_scm()
    intervention = _hidden_intervention()
    realisable = generate_realisable_target(scm, intervention)
    nonproportional = nonproportional_coefficient_target(
        scm,
        intervention,
        selected_node=2,
        selected_parent=0,
        multiplier=1.35,
        configured_support_budget=BUDGET,
    )

    alternative_coefficients = np.array(scm.coefficients, copy=True)
    alternative_coefficients[1, 3] = 0.30
    alternative_dag = nx.DiGraph(scm.dag)
    alternative_dag.add_edge(1, 3)
    alternative = LinearGaussianSCM(
        alternative_dag,
        alternative_coefficients,
        noise_variances=scm.noise_variances,
    )
    graph_perturbed = graph_perturbed_target(alternative, intervention)

    noise_covariance = np.eye(5)
    noise_covariance[0, 2] = noise_covariance[2, 0] = 0.22
    noise_covariance[1, 4] = noise_covariance[4, 1] = -0.18
    correlated = correlated_noise_target(scm, noise_covariance, intervention)

    external_mean = np.array([0.8, -0.5, 1.1, -0.9, 0.35])
    factor = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.9, 0.0, 0.0, 0.0],
            [-0.3, 0.4, 1.1, 0.0, 0.0],
            [0.1, -0.2, 0.3, 0.8, 0.0],
            [0.25, 0.1, -0.15, 0.2, 1.2],
        ]
    )
    external = external_gaussian_target(
        external_mean,
        factor @ factor.T + 0.35 * np.eye(5),
        public_metadata={"audit_fixture": "deterministic_spd"},
    )

    heterogeneous_scm = _base_scm(np.array([0.55, 1.70, 0.85, 2.20, 1.15]))
    heterogeneous = generate_realisable_target(heterogeneous_scm, intervention)
    return (
        AuditProblem("realisable", scm, realisable),
        AuditProblem("nonproportional_coefficients", scm, nonproportional),
        AuditProblem("graph_perturbed", scm, graph_perturbed),
        AuditProblem("correlated_noise", scm, correlated),
        AuditProblem("external_spd", scm, external),
        AuditProblem("heterogeneous_invariant_noise", heterogeneous_scm, heterogeneous),
    )


class PublicObjective:
    def __init__(self, problem: AuditProblem) -> None:
        self.problem = problem
        self.ledger = EnvironmentQueryLedger(max_query_batches=None)
        self.environment = ExactMomentBlackBoxEnvironment(
            problem.scm,
            instance_id=f"blackbox-quadratic-audit:{problem.label}",
            ledger=self.ledger,
        )
        self.objective_evaluations = 0
        self.background_id = InterventionBackground.from_intervention(
            ShiftScaleIntervention.identity(problem.target.mean.size)
        ).identifier

    def value(self, intervention: ShiftScaleIntervention, kind: str = "candidate") -> float:
        result = self.environment.query(
            intervention,
            kind=kind,  # type: ignore[arg-type]
            background_id=self.background_id,
        )
        self.objective_evaluations += 1
        return gaussian_kl_target_to_candidate(
            self.problem.target.mean,
            self.problem.target.covariance,
            result.mean,
            result.covariance,
        )


def _steps() -> tuple[float, float]:
    shift_radius = min(-SHIFT_BOUNDS[0], SHIFT_BOUNDS[1])
    dscale_bounds = (SCALE_BOUNDS[0] - 1.0, SCALE_BOUNDS[1] - 1.0)
    dscale_radius = min(-dscale_bounds[0], dscale_bounds[1])
    if shift_radius <= 0.0 or dscale_radius <= 0.0:
        raise RuntimeError("configured domains do not contain identity in their interior")
    return 0.5 * shift_radius, 0.5 * dscale_radius


def _isolated(n: int, node: int, shift: float, dscale: float) -> ShiftScaleIntervention:
    shifts = np.zeros(n)
    scales = np.ones(n)
    shifts[node] = shift
    scales[node] = 1.0 + dscale
    return ShiftScaleIntervention(shifts, scales)


def reconstruct_minimal(
    objective: PublicObjective,
    baseline: float,
    node: int,
    h_shift: float,
    h_dscale: float,
) -> tuple[Quadratic, dict[tuple[float, float], float]]:
    n = objective.problem.target.mean.size
    points = (
        (h_shift, 0.0),
        (-h_shift, 0.0),
        (0.0, h_dscale),
        (0.0, -h_dscale),
        (h_shift, h_dscale),
    )
    values = {
        point: objective.value(_isolated(n, node, *point)) - baseline
        for point in points
    }
    fp, fm = values[(h_shift, 0.0)], values[(-h_shift, 0.0)]
    gp, gm = values[(0.0, h_dscale)], values[(0.0, -h_dscale)]
    A = (fp + fm) / (2.0 * h_shift**2)
    B = (fp - fm) / (2.0 * h_shift)
    C = (gp + gm) / (2.0 * h_dscale**2)
    D = (gp - gm) / (2.0 * h_dscale)
    cross = values[(h_shift, h_dscale)]
    E = (
        cross
        - A * h_shift**2
        - B * h_shift
        - C * h_dscale**2
        - D * h_dscale
    ) / (h_shift * h_dscale)
    return Quadratic(A, B, C, D, E), values


def reconstruct_overdetermined(
    objective: PublicObjective,
    baseline: float,
    node: int,
    h_shift: float,
    h_dscale: float,
    existing: dict[tuple[float, float], float],
) -> Quadratic:
    n = objective.problem.target.mean.size
    points = (
        (h_shift, 0.0),
        (-h_shift, 0.0),
        (0.0, h_dscale),
        (0.0, -h_dscale),
        (h_shift, h_dscale),
        (h_shift, -h_dscale),
        (-h_shift, h_dscale),
        (-h_shift, -h_dscale),
    )
    values = dict(existing)
    for point in points:
        if point not in values:
            values[point] = objective.value(_isolated(n, node, *point)) - baseline
    design = np.array(
        [[a**2, a, d**2, d, a * d] for a, d in points], dtype=float
    )
    response = np.array([values[point] for point in points])
    fitted, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank != 5:
        raise RuntimeError("overdetermined reconstruction design lost rank")
    return Quadratic(*(float(value) for value in fitted))


def _moment_from_quadratic(node: int, quadratic: Quadratic) -> sparse_oracle.NodeMoments:
    if quadratic.A <= 0.0:
        raise RuntimeError(f"node {node} reconstructed a nonpositive shift curvature")
    noise_variance = 1.0 / (2.0 * quadratic.A)
    mean_z = quadratic.E * noise_variance
    mean_y = mean_z - quadratic.B * noise_variance
    second_z = max(0.0, 2.0 * quadratic.C * noise_variance)
    second_yz = second_z - quadratic.D * noise_variance
    variance_z = max(0.0, second_z - mean_z**2)
    covariance_yz = second_yz - mean_y * mean_z
    max_shift = max(abs(value) for value in SHIFT_BOUNDS)
    max_scale = max(abs(value) for value in SCALE_BOUNDS)
    second_y = 1.0 + (
        max_shift**2
        + max_scale**2 * second_z
        + 2.0 * max_shift * abs(mean_y)
        + 2.0 * max_scale * abs(second_yz)
        + 2.0 * max_shift * max_scale * abs(mean_z)
    )
    scale_signal = max(abs(quadratic.C), abs(quadratic.D), abs(quadratic.E))
    scale_eligible = scale_signal > 256.0 * np.finfo(float).eps * max(
        1.0, *(abs(value) for value in quadratic.array())
    )
    matrix = np.array([[1.0, mean_z], [mean_z, second_z]])
    eigenvalues = np.linalg.eigvalsh(matrix)
    positive = eigenvalues[eigenvalues > DEGENERACY_ATOL * max(1.0, eigenvalues[-1])]
    condition = float(eigenvalues[-1] / positive[0]) if positive.size == 2 else None
    return sparse_oracle.NodeMoments(
        node=node,
        parents=(() if not scale_eligible else (max(0, node - 1),)),
        source_intercept=0.0,
        noise_variance=noise_variance,
        mean_y=mean_y,
        mean_z=mean_z,
        second_y=second_y,
        second_z=second_z,
        second_yz=second_yz,
        variance_y=second_y - mean_y**2,
        variance_z=variance_z,
        covariance_yz=covariance_yz,
        coefficient_norm=1.0 if scale_eligible else 0.0,
        root=not scale_eligible,
        scale_eligible=scale_eligible,
        hessian_eigenvalues=(float(eigenvalues[0]), float(eigenvalues[1])),
        hessian_condition_number=condition,
        numerically_degenerate=not scale_eligible,
        degeneracy_reason=None if scale_eligible else "publicly_identified_scale_noop",
    )


def reconstructed_dp(
    quadratics: tuple[Quadratic, ...], baseline: float
) -> sparse_oracle.SparseShiftScaleOracleResult:
    """Route reconstructed states through the repository's validated DP code.

    The oracle entry point has no state-input adapter, so this audit substitutes
    only its moment-extraction and additive-constant stages.  Local solving,
    snapping, pruning, tie-breaking, DP, and traceback remain the production
    implementation.
    """
    moments = tuple(
        _moment_from_quadratic(node, quadratic)
        for node, quadratic in enumerate(quadratics)
    )
    synthetic_constant = baseline - sum(
        sparse_oracle.local_kl_loss(moment, 0.0, 1.0) for moment in moments
    )
    n = len(moments)
    dummy_dag = nx.DiGraph()
    dummy_dag.add_nodes_from(range(n))
    dummy_coefficients = np.zeros((n, n))
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(sparse_oracle, "extract_node_moments", return_value=moments)
        )
        stack.enter_context(
            patch.object(
                sparse_oracle,
                "kl_additive_constant",
                return_value=synthetic_constant,
            )
        )
        return sparse_oracle.solve_sparse_shift_scale_kl_from_parameters(
            dag=dummy_dag,
            source_intercepts=np.zeros(n),
            coefficients=dummy_coefficients,
            noise_variances=np.ones(n),
            target_mean=np.zeros(n),
            target_covariance=np.eye(n),
            budget=BUDGET,
            shift_bounds=SHIFT_BOUNDS,
            log_scale_bounds=LOG_SCALE_BOUNDS,
            support_atol=DEFAULT_SUPPORT_ATOL,
            loss_atol=LOSS_ATOL,
            loss_rtol=LOSS_RTOL,
            degeneracy_atol=DEGENERACY_ATOL,
            provenance={"audit_adapter": "public_reconstructed_quadratics"},
        )


def _joint_interventions(n: int) -> tuple[ShiftScaleIntervention, ...]:
    rng = np.random.default_rng(20260810)
    output = []
    for _ in range(JOINT_DRAWS):
        shifts = rng.uniform(SHIFT_BOUNDS[0] * 0.8, SHIFT_BOUNDS[1] * 0.8, n)
        log_scales = rng.uniform(
            LOG_SCALE_BOUNDS[0] * 0.8,
            LOG_SCALE_BOUNDS[1] * 0.8,
            n,
        )
        shifts[rng.random(n) < 0.35] = 0.0
        log_scales[rng.random(n) < 0.35] = 0.0
        if np.count_nonzero(shifts) < 2:
            shifts[:2] = (0.4, -0.6)
        if np.count_nonzero(log_scales) < 2:
            log_scales[-2:] = (0.2, -0.3)
        output.append(ShiftScaleIntervention(shifts, np.exp(log_scales)))
    return tuple(output)


def _relative_error(predicted: float, actual: float) -> float | None:
    scale = max(abs(predicted), abs(actual))
    return None if scale <= np.finfo(float).eps else abs(predicted - actual) / scale


def audit_problem(problem: AuditProblem) -> dict[str, object]:
    n = problem.target.mean.size
    objective = PublicObjective(problem)
    identity = ShiftScaleIntervention.identity(n)
    baseline = objective.value(identity, kind="baseline")
    h_shift, h_dscale = _steps()
    minimal = []
    overdetermined = []
    coefficient_differences = []
    for node in range(n):
        quadratic, observations = reconstruct_minimal(
            objective, baseline, node, h_shift, h_dscale
        )
        diagnostic = reconstruct_overdetermined(
            objective,
            baseline,
            node,
            h_shift,
            h_dscale,
            observations,
        )
        minimal.append(quadratic)
        overdetermined.append(diagnostic)
        coefficient_differences.append(
            float(np.max(np.abs(quadratic.array() - diagnostic.array())))
        )
    quadratics = tuple(minimal)

    identity_errors = []
    relative_errors = []
    joint_records = []
    for intervention in _joint_interventions(n):
        predicted = baseline + sum(
            quadratics[node].value(
                float(intervention.shift[node]),
                float(intervention.scale[node] - 1.0),
            )
            for node in range(n)
        )
        actual = objective.value(intervention)
        absolute_error = abs(predicted - actual)
        relative_error = _relative_error(predicted, actual)
        identity_errors.append(absolute_error)
        if relative_error is not None:
            relative_errors.append(relative_error)
        joint_records.append(
            {
                "predicted_kl": predicted,
                "actual_kl": actual,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
            }
        )

    oracle = sparse_oracle.solve_sparse_shift_scale_kl(
        scm=problem.scm,
        target_mean=problem.target.mean,
        target_covariance=problem.target.covariance,
        budget=BUDGET,
        shift_bounds=SHIFT_BOUNDS,
        log_scale_bounds=LOG_SCALE_BOUNDS,
        support_atol=DEFAULT_SUPPORT_ATOL,
        loss_atol=LOSS_ATOL,
        loss_rtol=LOSS_RTOL,
        degeneracy_atol=DEGENERACY_ATOL,
    )
    reconstructed = reconstructed_dp(quadratics, baseline)
    dp_intervention = ShiftScaleIntervention(
        reconstructed.budgeted_shift,
        reconstructed.budgeted_scale,
    )
    actual_dp_kl = objective.value(dp_intervention)
    predicted_dp_kl = baseline + sum(
        quadratics[node].value(
            float(dp_intervention.shift[node]),
            float(dp_intervention.scale[node] - 1.0),
        )
        for node in range(n)
    )

    local_delta_errors = []
    for node, states in enumerate(oracle.node_local_candidate_states):
        none_loss = sparse_oracle.local_kl_loss(oracle.node_moments[node], 0.0, 1.0)
        for state in states:
            oracle_delta = state.raw_local_loss - none_loss
            reconstructed_delta = quadratics[node].value(
                state.shift, state.scale - 1.0
            )
            local_delta_errors.append(abs(oracle_delta - reconstructed_delta))

    ledger = objective.ledger.snapshot()
    return {
        "problem": problem.label,
        "target_is_realisable": problem.target.is_realisable,
        "n_nodes": n,
        "baseline_kl": baseline,
        "steps": {"shift": h_shift, "dscale": h_dscale},
        "minimal_coefficients": [asdict(item) for item in minimal],
        "overdetermined_coefficients": [asdict(item) for item in overdetermined],
        "minimal_vs_overdetermined_max_coefficient_error": max(
            coefficient_differences
        ),
        "joint_identity_gate": {
            "draws": joint_records,
            "maximum_absolute_error": max(identity_errors),
            "median_absolute_error": float(np.median(identity_errors)),
            "maximum_relative_error": max(relative_errors),
            "median_relative_error": float(np.median(relative_errors)),
        },
        "dp_gate": {
            "reconstructed_predicted_kl": predicted_dp_kl,
            "reconstructed_actual_query_kl": actual_dp_kl,
            "analytic_K_B_star": oracle.raw_K_B_star,
            "predicted_minus_oracle": predicted_dp_kl - oracle.raw_K_B_star,
            "actual_query_minus_oracle": actual_dp_kl - oracle.raw_K_B_star,
            "reconstructed_support_cost": reconstructed.budget_used,
            "oracle_support_cost": oracle.budget_used,
            "reconstructed_support": reconstructed.budgeted_support,
            "oracle_support": oracle.budgeted_support,
            "reconstructed_state_labels": reconstructed.selected_budgeted_state_labels,
            "oracle_state_labels": oracle.selected_budgeted_state_labels,
            "maximum_shift_parameter_error": float(
                np.max(np.abs(reconstructed.budgeted_shift - oracle.budgeted_shift))
            ),
            "maximum_scale_parameter_error": float(
                np.max(np.abs(reconstructed.budgeted_scale - oracle.budgeted_scale))
            ),
            "maximum_local_state_delta_error": max(local_delta_errors),
            "oracle_direct_decomposition_error": oracle.budgeted_direct_absolute_error,
        },
        "query_accounting": {
            "minimal_reconstruction_objective_evaluations": 1 + 5 * n,
            "overdetermined_diagnostic_extra_evaluations": 3 * n,
            "joint_gate_evaluations": JOINT_DRAWS,
            "dp_validation_evaluations": 1,
            "observed_objective_evaluations": objective.objective_evaluations,
            "environment_ledger": ledger,
            "analytic_oracle_evaluations": 1,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    results = [audit_problem(problem) for problem in make_problems()]
    summary = {
        "configuration": {
            "shift_bounds": SHIFT_BOUNDS,
            "log_scale_bounds": LOG_SCALE_BOUNDS,
            "natural_scale_bounds": SCALE_BOUNDS,
            "budget": BUDGET,
            "support_atol": DEFAULT_SUPPORT_ATOL,
            "loss_atol": LOSS_ATOL,
            "loss_rtol": LOSS_RTOL,
            "degeneracy_atol": DEGENERACY_ATOL,
            "joint_draws_per_problem": JOINT_DRAWS,
        },
        "problems": results,
        "aggregate": {
            "maximum_joint_absolute_error": max(
                item["joint_identity_gate"]["maximum_absolute_error"]  # type: ignore[index]
                for item in results
            ),
            "median_joint_absolute_error": float(
                np.median(
                    [
                        draw["absolute_error"]
                        for item in results
                        for draw in item["joint_identity_gate"]["draws"]  # type: ignore[index]
                    ]
                )
            ),
            "maximum_dp_actual_query_minus_oracle_absolute": max(
                abs(item["dp_gate"]["actual_query_minus_oracle"])  # type: ignore[index]
                for item in results
            ),
            "maximum_minimal_vs_overdetermined_coefficient_error": max(
                item["minimal_vs_overdetermined_max_coefficient_error"]  # type: ignore[index]
                for item in results
            ),
        },
    }
    print(json.dumps(summary, indent=None if args.compact else 2, sort_keys=True))


if __name__ == "__main__":
    main()
