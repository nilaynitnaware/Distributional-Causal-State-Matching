"""Independent tiny B=1 reference for the frozen S01-B3 cases.

The reference enumerates identity plus every one-coordinate shift/scale mode.
Its continuous minimizers and Gaussian KL implementation are local to this
file; it never calls the sparse production optimizer or its local-state solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path
import json

import networkx as nx
import numpy as np

from graphs.support import parameter_support_cost
from research.matching_query_complexity import (
    CanonicalAction,
    CandidateSCM,
    compute_all_optimal_action_sets,
    enumerate_consistent_dags,
    find_common_optimal_command,
    infer_outward_tree_root,
    pdag_from_undirected_graph,
    reconstruct_all_candidate_scms,
)

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ReferenceCandidateResult:
    root: int
    optimal_kl: float
    actions: tuple[CanonicalAction, ...]


def load_frozen() -> dict[str, object]:
    return json.loads((HERE / "frozen_cases.json").read_text())


def _independent_moments(
    scm: CandidateSCM, shift: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.eye(shift.size) - scale[:, None] * scm.coefficients.T
    mean = np.linalg.solve(matrix, scm.intercepts + shift)
    propagation = np.linalg.solve(matrix, np.diag(np.sqrt(scm.noise_variances)))
    return mean, propagation @ propagation.T


def _independent_gaussian_kl(
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_covariance: np.ndarray,
) -> float:
    target_factor = np.linalg.cholesky(target_covariance)
    candidate_factor = np.linalg.cholesky(candidate_covariance)
    whitened_target = np.linalg.solve(candidate_factor, target_factor)
    whitened_mean = np.linalg.solve(candidate_factor, candidate_mean - target_mean)
    value = 0.5 * (
        float(np.sum(whitened_target**2))
        + float(whitened_mean @ whitened_mean)
        - target_mean.size
        + 2.0 * float(np.log(np.diag(candidate_factor)).sum())
        - 2.0 * float(np.log(np.diag(target_factor)).sum())
    )
    return 0.0 if abs(value) < 1e-14 else float(value)


def independent_direct_kl(
    scm: CandidateSCM,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    action: CanonicalAction,
) -> float:
    mean, covariance = _independent_moments(
        scm, np.asarray(action.shift), np.asarray(action.scale)
    )
    return _independent_gaussian_kl(
        target_mean, target_covariance, mean, covariance
    )


def independent_budget_one_reference(
    scm: CandidateSCM,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    shift_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
    *,
    tie_atol: float,
) -> ReferenceCandidateResult:
    """Exhaust all legal B=1 modes and solve each scalar quadratic exactly."""
    n = target_mean.size
    second = target_covariance + np.outer(target_mean, target_mean)
    candidates = [CanonicalAction((0.0,) * n, (1.0,) * n)]
    for node in range(n):
        beta = scm.coefficients[:, node]
        mean_y = float(target_mean[node] - scm.intercepts[node])
        mean_z = float(beta @ target_mean)
        shift_value = float(np.clip(mean_y - mean_z, *shift_bounds))
        shift = np.zeros(n)
        shift[node] = shift_value
        candidates.append(CanonicalAction.from_arrays(shift, np.ones(n)))

        parents = scm.candidate.parent_sets[node]
        second_z = float(beta @ second @ beta)
        if parents and second_z > 0.0:
            second_yz = float(
                beta @ (target_covariance[:, node] + target_mean * mean_y)
            )
            scale_value = float(np.clip(second_yz / second_z, *scale_bounds))
            scale = np.ones(n)
            scale[node] = scale_value
            candidates.append(CanonicalAction.from_arrays(np.zeros(n), scale))

    unique: list[CanonicalAction] = []
    for action in candidates:
        if parameter_support_cost(
            np.asarray(action.shift), np.asarray(action.scale)
        ) <= 1 and action not in unique:
            unique.append(action)
    losses = [
        independent_direct_kl(scm, target_mean, target_covariance, action)
        for action in unique
    ]
    optimum = min(losses)
    tied = tuple(
        sorted(
            action
            for action, loss in zip(unique, losses, strict=True)
            if abs(loss - optimum) <= tie_atol
        )
    )
    return ReferenceCandidateResult(
        root=infer_outward_tree_root(scm.candidate),
        optimal_kl=float(optimum),
        actions=tied,
    )


def build_case(case: dict[str, object]):
    n = 2 if case["skeleton"] == "path_2" else 3
    candidates = enumerate_consistent_dags(
        pdag_from_undirected_graph(nx.path_graph(n))
    )
    source_mean = np.asarray(case["source_mean"], dtype=float)
    source_covariance = np.asarray(case["source_covariance"], dtype=float)
    target_mean = np.asarray(case["target_mean"], dtype=float)
    target_covariance = np.asarray(case["target_covariance"], dtype=float)
    scms = reconstruct_all_candidate_scms(
        candidates, source_mean, source_covariance
    )
    return candidates, scms, target_mean, target_covariance


def evaluate_case(case: dict[str, object], conventions: dict[str, object]):
    candidates, scms, target_mean, target_covariance = build_case(case)
    shift_bounds = tuple(float(x) for x in conventions["shift_bounds"])
    log_bounds = tuple(float(x) for x in conventions["log_scale_bounds"])
    scale_bounds = (exp(log_bounds[0]), exp(log_bounds[1]))
    tie_atol = float(conventions["reference_tie_atol"])
    reference = {
        item.root: item
        for item in (
            independent_budget_one_reference(
                scm,
                target_mean,
                target_covariance,
                shift_bounds,
                scale_bounds,
                tie_atol=tie_atol,
            )
            for scm in scms.values()
        )
    }
    production = compute_all_optimal_action_sets(
        scms,
        target_mean=target_mean,
        target_covariance=target_covariance,
        budget=int(conventions["budget"]),
        shift_bounds=shift_bounds,
        log_scale_bounds=log_bounds,
    )
    certificate = find_common_optimal_command(
        {candidate.id for candidate in candidates}, production
    )
    return candidates, scms, reference, production, certificate


def action_record(action: CanonicalAction) -> dict[str, list[float]]:
    return {"shift": list(action.shift), "scale": list(action.scale)}

