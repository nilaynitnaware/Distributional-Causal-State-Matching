"""Evaluation-only best-found comparators for non-realisable targets."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import GeneratedProblem
from matching.coordinate_search import CoordinateSearchBounds
from matching.types import FinalSupportBudget, ParameterCoordinate
from metrics.gaussian_kl import gaussian_kl_target_to_candidate


@dataclass(frozen=True)
class OracleComparatorSettings:
    """Small exhaustive-support or larger multistart comparator settings."""

    exhaustive_max_nodes: int = 4
    exhaustive_max_supports: int = 1_000
    multistart_count: int = 20
    optimizer_max_iterations: int = 500
    seed: int = 0


@dataclass(frozen=True)
class OracleComparatorResult:
    """A numerical best-found result that is never called a lower bound."""

    label: str
    best_loss: float
    best_intervention: ShiftScaleIntervention
    supports_evaluated: int
    starts_evaluated: int
    all_configured_supports_enumerated: bool
    proven_lower_bound: bool = False


@dataclass(frozen=True)
class NonRealisableGapMetrics:
    """Identity/final/comparator losses and observed-gap closure."""

    identity_loss: float
    final_algorithm_loss: float
    oracle_comparator_loss: float
    fraction_observed_gap_closed: float | None
    comparator_label: str


def best_found_oracle_comparator(
    problem: GeneratedProblem,
    final_support_budget: FinalSupportBudget,
    bounds: CoordinateSearchBounds,
    settings: OracleComparatorSettings,
) -> OracleComparatorResult:
    """Search all small supports or use labeled multistart on larger cases."""
    if not isinstance(problem, GeneratedProblem):
        raise TypeError("problem must be GeneratedProblem.")
    if not isinstance(final_support_budget, FinalSupportBudget):
        raise TypeError("final_support_budget must be FinalSupportBudget.")
    if not isinstance(bounds, CoordinateSearchBounds):
        raise TypeError("bounds must be CoordinateSearchBounds.")
    n_nodes = problem.public.n_nodes
    coordinates = tuple(
        ParameterCoordinate(node, kind)
        for node in range(n_nodes)
        for kind in (
            ("shift", "log_scale")
            if problem.ground_truth.scm.dag.in_degree(node) > 0
            else ("shift",)
        )
    )
    max_support = min(
        final_support_budget.max_parameter_coordinates,
        len(coordinates),
    )
    support_count = sum(
        1 for size in range(max_support + 1) for _ in combinations(coordinates, size)
    )
    exhaustive = (
        n_nodes <= settings.exhaustive_max_nodes
        and support_count <= settings.exhaustive_max_supports
    )
    if exhaustive:
        supports = tuple(
            support
            for size in range(max_support + 1)
            for support in combinations(coordinates, size)
        )
        label = "exhaustive_small_support_best_found_oracle_comparator"
    else:
        rng = np.random.default_rng(settings.seed)
        sampled: list[tuple[ParameterCoordinate, ...]] = [()]
        for _ in range(settings.multistart_count):
            size = int(rng.integers(0, max_support + 1))
            indices = sorted(
                rng.choice(len(coordinates), size=size, replace=False).tolist()
            )
            sampled.append(tuple(coordinates[index] for index in indices))
        supports = tuple(sampled)
        label = "best_found_oracle_comparator"

    best_loss = float("inf")
    best_intervention = ShiftScaleIntervention.identity(n_nodes)
    starts = 0
    for support_index, support in enumerate(supports):
        rng = np.random.default_rng(settings.seed + support_index)
        starts_for_support = 1 if exhaustive else max(1, settings.multistart_count)
        if not support:
            starts += 1
            loss = _oracle_loss(ShiftScaleIntervention.identity(n_nodes), problem)
            if loss < best_loss:
                best_loss = loss
                best_intervention = ShiftScaleIntervention.identity(n_nodes)
            continue
        for start in range(starts_for_support):
            starts += 1
            initial = np.array(
                [
                    (
                        rng.uniform(bounds.shift_min, bounds.shift_max)
                        if coordinate.kind == "shift"
                        else rng.uniform(
                            bounds.log_scale_min,
                            bounds.log_scale_max,
                        )
                    )
                    if start
                    else 0.0
                    for coordinate in support
                ],
                dtype=float,
            )
            result = minimize(
                lambda values, support=support: _oracle_support_loss(
                    values,
                    support,
                    problem,
                ),
                initial,
                method="L-BFGS-B",
                bounds=[bounds.domain(coordinate) for coordinate in support],
                options={"maxiter": settings.optimizer_max_iterations},
            )
            loss = float(result.fun)
            if loss < best_loss:
                best_loss = loss
                best_intervention = _support_intervention(
                    np.asarray(result.x),
                    support,
                    n_nodes,
                )
    return OracleComparatorResult(
        label=label,
        best_loss=best_loss,
        best_intervention=best_intervention,
        supports_evaluated=len(supports),
        starts_evaluated=starts,
        all_configured_supports_enumerated=exhaustive,
    )


def nonrealisable_gap_metrics(
    problem: GeneratedProblem,
    final_intervention: ShiftScaleIntervention,
    comparator: OracleComparatorResult,
) -> NonRealisableGapMetrics:
    """Summarize the observed loss gap without claiming attainability."""
    identity = ShiftScaleIntervention.identity(problem.public.n_nodes)
    identity_loss = _oracle_loss(identity, problem)
    final_loss = _oracle_loss(final_intervention, problem)
    denominator = identity_loss - comparator.best_loss
    fraction = (identity_loss - final_loss) / denominator if denominator > 0.0 else None
    return NonRealisableGapMetrics(
        identity_loss=identity_loss,
        final_algorithm_loss=final_loss,
        oracle_comparator_loss=comparator.best_loss,
        fraction_observed_gap_closed=fraction,
        comparator_label=comparator.label,
    )


def _oracle_support_loss(
    values: np.ndarray,
    support: tuple[ParameterCoordinate, ...],
    problem: GeneratedProblem,
) -> float:
    return _oracle_loss(
        _support_intervention(values, support, problem.public.n_nodes),
        problem,
    )


def _oracle_loss(
    intervention: ShiftScaleIntervention,
    problem: GeneratedProblem,
) -> float:
    mean, covariance = problem.ground_truth.scm.interventional_moments(intervention)
    return gaussian_kl_target_to_candidate(
        problem.public.target_mean,
        problem.public.target_covariance,
        mean,
        covariance,
    )


def _support_intervention(
    values: np.ndarray,
    support: tuple[ParameterCoordinate, ...],
    n_nodes: int,
) -> ShiftScaleIntervention:
    shift = np.zeros(n_nodes)
    scale = np.ones(n_nodes)
    for value, coordinate in zip(values, support, strict=True):
        if coordinate.kind == "shift":
            shift[coordinate.node] = value
        else:
            scale[coordinate.node] = np.exp(value)
    return ShiftScaleIntervention(shift, scale)
