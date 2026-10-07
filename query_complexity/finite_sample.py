"""Finite-sample Algorithm 1 built around the validated exact sparse DP.

The algorithm-facing functions in this module accept samples, a public PDAG,
and bounded singleton-shift environments.  Hidden SCM objects are deliberately
absent from their signatures.  Ground truth is used only by experiment-level
evaluation helpers outside this module.

This validated finite-sample path assumes a known nonzero probe magnitude.
Calibration-free population identification is implemented separately; a
finite-sample normalized estimator additionally needs denominator stability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import networkx as nx
import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.meek import apply_meek_closure
from graphs.pdag import PartiallyDirectedGraph
from metrics.gaussian_kl import gaussian_kl_target_to_candidate
from oracle.sparse_shift_scale_kl import (
    LocalObjective,
    NodeMoments,
    SparseShiftScaleOracleResult,
    local_kl_loss,
    solve_sparse_shift_scale_kl_from_parameters,
)
from query_complexity.core import FirstUnresolvedPolicy, SingletonShiftPolicy

FINITE_SAMPLE_PROTOCOL_VERSION = "finite_sample_algorithm1_v1"
FiniteSampleMethod = Literal[
    "finite_sample_algorithm1",
    "graph_free_finite_sample",
    "known_dag_finite_sample",
]


def _immutable(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class OASDiagnostics:
    """OAS covariance/precision estimate and numerical diagnostics."""

    mean: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    shrinkage: float
    covariance_condition_number: float
    precision_condition_number: float
    minimum_eigenvalue: float
    inversion_failed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _immutable(self.mean))
        object.__setattr__(self, "covariance", _immutable(self.covariance))
        object.__setattr__(self, "precision", _immutable(self.precision))



def _oas_from_empirical(
    mean: np.ndarray,
    empirical_covariance: np.ndarray,
    n_samples: int,
) -> OASDiagnostics:
    """Apply OAS shrinkage to a backend-computed empirical summary."""
    covariance = np.array(empirical_covariance, dtype=float, copy=True)
    n_features = covariance.shape[0]
    mu = float(np.trace(covariance) / n_features)
    alpha = float(np.mean(covariance**2))
    denominator = (n_samples + 1.0) * (alpha - (mu * mu) / n_features)
    shrinkage = 1.0 if denominator <= 0.0 else min(
        (alpha + mu * mu) / denominator, 1.0
    )
    covariance *= 1.0 - shrinkage
    covariance.flat[:: n_features + 1] += shrinkage * mu
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    minimum = float(eigenvalues[0])
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("OAS covariance is not positive definite")
    try:
        precision = np.linalg.solve(covariance, np.eye(n_features))
    except np.linalg.LinAlgError as error:
        raise ValueError("OAS covariance inversion failed") from error
    precision = 0.5 * (precision + precision.T)
    return OASDiagnostics(
        mean=mean,
        covariance=covariance,
        precision=precision,
        shrinkage=float(shrinkage),
        covariance_condition_number=float(np.linalg.cond(covariance)),
        precision_condition_number=float(np.linalg.cond(precision)),
        minimum_eigenvalue=minimum,
    )
def estimate_oas(samples: np.ndarray) -> OASDiagnostics:
    """Estimate OAS covariance exactly as ``sklearn.covariance.OAS`` does.

    The maximum-likelihood empirical covariance (division by ``N``) and the
    Chen et al. shrinkage formula match scikit-learn's implementation.  The
    precision is obtained by a symmetric linear solve so inversion failures
    are explicit.
    """
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise TypeError("samples must be a two-dimensional numpy array")
    if samples.shape[0] < 2 or samples.shape[1] < 1:
        raise ValueError("OAS requires at least two samples and one feature")
    values = np.asarray(samples, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / values.shape[0]
    n_features = values.shape[1]
    mu = float(np.trace(covariance) / n_features)
    alpha = float(np.mean(covariance**2))
    denominator = (values.shape[0] + 1.0) * (
        alpha - (mu * mu) / n_features
    )
    shrinkage = 1.0 if denominator <= 0.0 else min(
        (alpha + mu * mu) / denominator, 1.0
    )
    covariance = (1.0 - shrinkage) * covariance
    covariance.flat[:: n_features + 1] += shrinkage * mu
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    minimum = float(eigenvalues[0])
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("OAS covariance is not positive definite")
    try:
        precision = np.linalg.solve(covariance, np.eye(n_features))
    except np.linalg.LinAlgError as error:
        raise ValueError("OAS covariance inversion failed") from error
    precision = 0.5 * (precision + precision.T)
    return OASDiagnostics(
        mean=mean,
        covariance=covariance,
        precision=precision,
        shrinkage=float(shrinkage),
        covariance_condition_number=float(np.linalg.cond(covariance)),
        precision_condition_number=float(np.linalg.cond(precision)),
        minimum_eigenvalue=minimum,
    )


@dataclass(frozen=True)
class FiniteSampleSettings:
    """Predeclared estimation and confidence-gating settings."""

    bootstrap_replicates: int = 64
    confidence_level: float = 0.95
    orientation_confidence: float = 0.95
    null_quantile: float = 0.95
    minimum_self_precision_effect: float = 1e-10
    support_probability_threshold: float = 0.5
    backend: Literal["numpy", "torch"] = "numpy"
    device: Literal["cpu", "cuda"] = "cpu"

    def __post_init__(self) -> None:
        if isinstance(self.bootstrap_replicates, bool) or self.bootstrap_replicates < 2:
            raise ValueError("bootstrap_replicates must be at least two")
        for name in ("confidence_level", "orientation_confidence", "null_quantile"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
        if not 0.0 <= self.support_probability_threshold <= 1.0:
            raise ValueError("support_probability_threshold must lie in [0, 1]")
        if self.minimum_self_precision_effect <= 0.0:
            raise ValueError("minimum_self_precision_effect must be positive")
        if self.backend not in {"numpy", "torch"}:
            raise ValueError("backend must be numpy or torch")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.backend == "numpy" and self.device != "cpu":
            raise ValueError("the numpy backend supports only the cpu device")
        if self.backend == "torch":
            try:
                import torch
            except ImportError as error:
                raise ValueError("the torch backend requires PyTorch") from error
            if self.device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable")


@dataclass(frozen=True)
class ParentCoefficientEvidence:
    """Uncertainty-aware support evidence for one possible incoming edge."""

    node: int
    coefficient: float
    standard_error: float
    confidence_interval: tuple[float, float]
    parent_probability: float
    selected_parent: bool
    orientation_decision: Literal["parent", "nonparent", "unresolved"]
    structurally_possible_parent: bool
    known_parent: bool


@dataclass(frozen=True)
class FiniteLocalMechanism:
    """One finite-sample estimate of the precision-times-effect identity."""

    node: int
    parents: tuple[int, ...]
    incoming_coefficients: np.ndarray
    mechanism_row: np.ndarray
    total_effect_column: np.ndarray
    noise_variance: float
    source_intercept: float
    valid: bool
    invalid_reason: str | None
    null_threshold: float
    bootstrap_valid_replicates: int
    evidence: tuple[ParentCoefficientEvidence, ...]
    provenance: tuple[str, ...] = (
        "sample_source_mean",
        "sample_singleton_shift_mean",
        "oas_source_precision",
        "bootstrap_graph_restricted_support",
        "no_hidden_structural_inputs",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "incoming_coefficients", _immutable(self.incoming_coefficients)
        )
        object.__setattr__(self, "mechanism_row", _immutable(self.mechanism_row))
        object.__setattr__(
            self, "total_effect_column", _immutable(self.total_effect_column)
        )


@dataclass(frozen=True)
class BootstrapSourceCache:
    """Reusable source bootstrap estimates for every probe in one dataset."""

    means: np.ndarray
    precisions: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "means", _immutable(self.means))
        precisions = np.array(self.precisions, dtype=float, copy=True)
        valid = np.array(self.valid, dtype=bool, copy=True)
        precisions.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "precisions", precisions)
        object.__setattr__(self, "valid", valid)


def bootstrap_source_cache(
    source_samples: np.ndarray,
    *,
    replicates: int,
    seed: int,
    backend: Literal["numpy", "torch"] = "numpy",
    device: Literal["cpu", "cuda"] = "cpu",
) -> BootstrapSourceCache:
    """Resample source data and recompute OAS using the selected backend."""
    if replicates < 2:
        raise ValueError("replicates must be at least two")
    if backend not in {"numpy", "torch"} or device not in {"cpu", "cuda"}:
        raise ValueError("unsupported bootstrap backend or device")
    if backend == "numpy" and device != "cpu":
        raise ValueError("the numpy backend supports only cpu")
    rng = np.random.default_rng(seed)
    count, dimension = source_samples.shape
    means = np.full((replicates, dimension), np.nan)
    precisions = np.full((replicates, dimension, dimension), np.nan)
    valid = np.zeros(replicates, dtype=bool)
    tensor = None
    torch = None
    if backend == "torch":
        import torch as torch_module

        torch = torch_module
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        tensor = torch.as_tensor(source_samples, dtype=torch.float64, device=device)
    for replicate in range(replicates):
        indices = rng.integers(0, count, size=count)
        try:
            if tensor is None:
                estimate = estimate_oas(source_samples[indices])
            else:
                selected = tensor[torch.as_tensor(indices, dtype=torch.long, device=device)]
                mean_tensor = selected.mean(dim=0)
                centered = selected - mean_tensor
                empirical = centered.T @ centered / count
                estimate = _oas_from_empirical(
                    mean_tensor.cpu().numpy(), empirical.cpu().numpy(), count
                )
        except (ValueError, np.linalg.LinAlgError):
            continue
        means[replicate] = estimate.mean
        precisions[replicate] = estimate.precision
        valid[replicate] = True
    return BootstrapSourceCache(means, precisions, valid)


def _mechanism_from_moments(
    *,
    node: int,
    source_mean: np.ndarray,
    precision: np.ndarray,
    intervention_mean: np.ndarray,
    shift_magnitude: float,
    minimum_self_effect: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    total_effect = (intervention_mean - source_mean) / shift_magnitude
    precision_effect = precision @ total_effect
    self_effect = float(precision_effect[node])
    if not np.isfinite(self_effect) or self_effect <= minimum_self_effect:
        return None
    omega = 1.0 / self_effect
    mechanism = omega * precision_effect
    if not np.all(np.isfinite(mechanism)):
        return None
    mechanism = np.array(mechanism, copy=True)
    mechanism[node] = 1.0
    return total_effect, mechanism, float(omega)


def estimate_local_mechanism(
    *,
    node: int,
    source_samples: np.ndarray,
    intervention_samples: np.ndarray,
    shift_magnitude: float,
    source_estimate: OASDiagnostics,
    settings: FiniteSampleSettings,
    bootstrap_seed: int,
    possible_parents: frozenset[int] | None,
    known_parents: frozenset[int] = frozenset(),
    known_nonparents: frozenset[int] = frozenset(),
    source_bootstrap: BootstrapSourceCache | None = None,
) -> FiniteLocalMechanism:
    """Estimate one local mechanism and graph-restricted parent support.

    ``possible_parents=None`` is the graph-free regime.  In the graph-aware
    regime, known non-neighbors form an empirical null; hidden DAG support is
    not accepted by this function.
    """
    if source_samples.ndim != 2 or intervention_samples.ndim != 2:
        raise TypeError("source and intervention samples must be matrices")
    if source_samples.shape[1] != intervention_samples.shape[1]:
        raise ValueError("source and intervention dimensions must agree")
    n = source_samples.shape[1]
    if node < 0 or node >= n:
        raise ValueError("node is outside the sample dimension")
    shift = float(shift_magnitude)
    if not np.isfinite(shift) or shift == 0.0:
        raise ValueError("shift_magnitude must be finite and nonzero")
    if possible_parents is None:
        candidates = frozenset(range(n)) - {node} - known_nonparents
    else:
        candidates = frozenset(possible_parents)
    if node in candidates or not known_parents <= candidates:
        raise ValueError("possible parent set is inconsistent")
    structural_zeros = (
        frozenset(range(n)) - {node} - candidates - known_nonparents
    ) | known_nonparents

    point = _mechanism_from_moments(
        node=node,
        source_mean=source_estimate.mean,
        precision=source_estimate.precision,
        intervention_mean=intervention_samples.mean(axis=0),
        shift_magnitude=shift,
        minimum_self_effect=settings.minimum_self_precision_effect,
    )
    if point is None:
        total_effect = (
            intervention_samples.mean(axis=0) - source_estimate.mean
        ) / shift
        mechanism = np.eye(n)[node]
        omega = float("nan")
        point_valid = False
        invalid_reason = "nonpositive_or_nonfinite_self_precision_effect"
    else:
        total_effect, mechanism, omega = point
        point_valid = True
        invalid_reason = None
    point_coefficients = -mechanism
    point_coefficients[node] = 0.0

    if source_bootstrap is None:
        source_bootstrap = bootstrap_source_cache(
            source_samples,
            replicates=settings.bootstrap_replicates,
            seed=bootstrap_seed,
            backend=settings.backend,
            device=settings.device,
        )
    elif source_bootstrap.means.shape[0] != settings.bootstrap_replicates:
        raise ValueError("source bootstrap replicate count disagrees with settings")
    rng = np.random.default_rng(bootstrap_seed ^ 0x5A17C9E3)
    boot = np.full((settings.bootstrap_replicates, n), np.nan)
    int_count = intervention_samples.shape[0]
    intervention_tensor = None
    torch = None
    if settings.backend == "torch":
        import torch as torch_module

        torch = torch_module
        intervention_tensor = torch.as_tensor(
            intervention_samples, dtype=torch.float64, device=settings.device
        )
    for replicate in range(settings.bootstrap_replicates):
        if not source_bootstrap.valid[replicate]:
            continue
        indices = rng.integers(0, int_count, size=int_count)
        if intervention_tensor is None:
            intervention_mean = intervention_samples[indices].mean(axis=0)
        else:
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=settings.device)
            intervention_mean = intervention_tensor[index_tensor].mean(dim=0).cpu().numpy()
        recovered = _mechanism_from_moments(
            node=node,
            source_mean=source_bootstrap.means[replicate],
            precision=source_bootstrap.precisions[replicate],
            intervention_mean=intervention_mean,
            shift_magnitude=shift,
            minimum_self_effect=settings.minimum_self_precision_effect,
        )
        if recovered is not None:
            boot[replicate] = -recovered[1]
            boot[replicate, node] = 0.0
    valid_rows = np.all(np.isfinite(boot), axis=1)
    valid_count = int(valid_rows.sum())
    if valid_count < 2:
        point_valid = False
        invalid_reason = "fewer_than_two_valid_bootstrap_replicates"
    valid_boot = boot[valid_rows]
    alpha = 1.0 - settings.confidence_level
    if structural_zeros and valid_count:
        zero_index = np.array(sorted(structural_zeros), dtype=int)
        null_values = np.abs(valid_boot[:, zero_index]).ravel()
        null_threshold = float(np.quantile(null_values, settings.null_quantile))
        replicate_thresholds = np.quantile(
            np.abs(valid_boot[:, zero_index]), settings.null_quantile, axis=1
        )
    else:
        null_threshold = 0.0
        replicate_thresholds = np.zeros(valid_count)

    evidence: list[ParentCoefficientEvidence] = []
    selected: set[int] = set(known_parents)
    for candidate in sorted(candidates):
        coefficient = float(point_coefficients[candidate])
        if valid_count:
            distribution = valid_boot[:, candidate]
            standard_error = float(np.std(distribution, ddof=1))
            interval = tuple(
                float(value)
                for value in np.quantile(distribution, (alpha / 2.0, 1.0 - alpha / 2.0))
            )
            if structural_zeros:
                selection = np.abs(distribution) > replicate_thresholds
                parent_probability = float(np.mean(selection))
            else:
                sign_stability = max(
                    float(np.mean(distribution > 0.0)),
                    float(np.mean(distribution < 0.0)),
                )
                parent_probability = max(0.0, min(1.0, 2.0 * sign_stability - 1.0))
        else:
            standard_error = float("nan")
            interval = (float("nan"), float("nan"))
            parent_probability = 0.0
        excludes_zero = interval[0] > 0.0 or interval[1] < 0.0
        contains_zero = interval[0] <= 0.0 <= interval[1]
        if candidate in known_parents:
            decision: Literal["parent", "nonparent", "unresolved"] = "parent"
            is_selected = True
            parent_probability = 1.0
        elif (
            point_valid
            and excludes_zero
            and parent_probability >= settings.orientation_confidence
        ):
            decision = "parent"
            is_selected = True
        elif (
            point_valid
            and contains_zero
            and parent_probability <= 1.0 - settings.orientation_confidence
        ):
            decision = "nonparent"
            is_selected = False
        else:
            decision = "unresolved"
            is_selected = bool(
                excludes_zero
                and parent_probability >= settings.support_probability_threshold
            )
        if is_selected:
            selected.add(candidate)
        evidence.append(
            ParentCoefficientEvidence(
                node=candidate,
                coefficient=coefficient,
                standard_error=standard_error,
                confidence_interval=interval,
                parent_probability=parent_probability,
                selected_parent=is_selected,
                orientation_decision=decision,
                structurally_possible_parent=True,
                known_parent=candidate in known_parents,
            )
        )

    incoming = np.zeros(n)
    if point_valid:
        chosen = np.array(sorted(selected), dtype=int)
        if chosen.size:
            incoming[chosen] = point_coefficients[chosen]
    row = np.eye(n)[node] - incoming
    source_intercept = float(source_estimate.mean[node] - incoming @ source_estimate.mean)
    return FiniteLocalMechanism(
        node=node,
        parents=tuple(sorted(selected)),
        incoming_coefficients=incoming,
        mechanism_row=row,
        total_effect_column=total_effect,
        noise_variance=omega,
        source_intercept=source_intercept,
        valid=point_valid,
        invalid_reason=invalid_reason,
        null_threshold=null_threshold,
        bootstrap_valid_replicates=valid_count,
        evidence=tuple(evidence),
    )


def estimated_node_moment(
    *,
    node: int,
    incoming_coefficients: np.ndarray,
    source_intercept: float,
    noise_variance: float,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    parents: tuple[int, ...],
    degeneracy_atol: float = 1e-14,
) -> NodeMoments:
    """Construct the validated local KL objective from estimated quantities."""
    beta = np.asarray(incoming_coefficients, dtype=float)
    if beta.shape != target_mean.shape:
        raise ValueError("incoming coefficients and target mean must agree")
    if not np.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("estimated noise variance must be positive")
    second = target_covariance + np.outer(target_mean, target_mean)
    mean_y = float(target_mean[node] - source_intercept)
    mean_z = float(beta @ target_mean)
    second_y = float(target_covariance[node, node] + mean_y**2)
    second_z = float(beta @ second @ beta)
    second_yz = float(beta @ (target_covariance[:, node] + target_mean * mean_y))
    variance_z = max(0.0, second_z - mean_z**2)
    eigenvalues = np.linalg.eigvalsh(np.array([[1.0, mean_z], [mean_z, second_z]]))
    positive = eigenvalues[
        eigenvalues > degeneracy_atol * max(1.0, float(eigenvalues[-1]))
    ]
    root = not parents
    return NodeMoments(
        node=node,
        parents=parents,
        source_intercept=float(source_intercept),
        noise_variance=float(noise_variance),
        mean_y=mean_y,
        mean_z=mean_z,
        second_y=second_y,
        second_z=second_z,
        second_yz=second_yz,
        variance_y=float(target_covariance[node, node]),
        variance_z=variance_z,
        covariance_yz=second_yz - mean_y * mean_z,
        coefficient_norm=float(np.linalg.norm(beta)),
        root=root,
        scale_eligible=not root and second_z != 0.0,
        hessian_eigenvalues=(float(eigenvalues[0]), float(eigenvalues[1])),
        hessian_condition_number=(
            float(eigenvalues[-1] / positive[0]) if positive.size == 2 else None
        ),
        numerically_degenerate=root
        or variance_z <= degeneracy_atol * max(1.0, second_z),
        degeneracy_reason="zero_parental_predictor" if root else None,
    )


def conditional_node_moment(
    *,
    node: int,
    parents: frozenset[int],
    source_estimate: OASDiagnostics,
    target_estimate: OASDiagnostics,
) -> NodeMoments:
    """Estimate a graph-resolved mechanism by OAS conditional regression."""
    n = source_estimate.mean.size
    beta = np.zeros(n)
    ordered = tuple(sorted(parents))
    if ordered:
        index = np.array(ordered, dtype=int)
        beta[index] = np.linalg.solve(
            source_estimate.covariance[np.ix_(index, index)],
            source_estimate.covariance[index, node],
        )
    intercept = float(source_estimate.mean[node] - beta @ source_estimate.mean)
    noise = float(
        source_estimate.covariance[node, node]
        - source_estimate.covariance[node] @ beta
    )
    return estimated_node_moment(
        node=node,
        incoming_coefficients=beta,
        source_intercept=intercept,
        noise_variance=noise,
        target_mean=target_estimate.mean,
        target_covariance=target_estimate.covariance,
        parents=ordered,
    )


class FiniteSampleProbeProvider(Protocol):
    """Algorithm-facing provider for physical singleton-shift environments."""

    n_int: int

    def query(self, node: int) -> np.ndarray: ...

    @property
    def physical_query_count(self) -> int: ...


class CachedProbeSamples:
    """Deterministic in-memory provider with physical-query/cache accounting."""

    def __init__(self, samples: Mapping[int, np.ndarray], n_int: int) -> None:
        self._samples = dict(samples)
        self.n_int = int(n_int)
        self._queried: set[int] = set()

    def query(self, node: int) -> np.ndarray:
        if node not in self._samples:
            raise KeyError(f"no singleton-shift samples for node {node}")
        self._queried.add(int(node))
        value = self._samples[node]
        if value.shape[0] < self.n_int:
            raise ValueError("probe sample pool is smaller than n_int")
        return value[: self.n_int]

    @property
    def physical_query_count(self) -> int:
        return len(self._queried)

    @property
    def queried_nodes(self) -> tuple[int, ...]:
        return tuple(sorted(self._queried))


@dataclass(frozen=True)
class OrientationEvent:
    query_index: int
    probed_node: int
    source: int
    target: int
    route: Literal["direct", "meek"]


@dataclass(frozen=True)
class FiniteSampleAlgorithmResult:
    method: FiniteSampleMethod
    intervention: ShiftScaleIntervention
    dp_result: SparseShiftScaleOracleResult
    source_estimate: OASDiagnostics
    target_estimate: OASDiagnostics
    mechanisms: tuple[FiniteLocalMechanism, ...]
    final_graph: PartiallyDirectedGraph | None
    orientation_events: tuple[OrientationEvent, ...]
    queried_nodes: tuple[int, ...]
    physical_query_count: int
    total_intervention_observations: int
    total_observations: int
    unresolved_edges: int
    invalid_estimates: int
    termination_reason: str
    estimated_initial_kl: float


def _solve_estimated_dp(
    objectives: tuple[LocalObjective, ...],
    *,
    source_estimate: OASDiagnostics,
    target_estimate: OASDiagnostics,
    budget: int,
    shift_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
    method: FiniteSampleMethod,
) -> SparseShiftScaleOracleResult:
    baseline = gaussian_kl_target_to_candidate(
        target_estimate.mean,
        target_estimate.covariance,
        source_estimate.mean,
        source_estimate.covariance,
    )
    constant = baseline - sum(
        local_kl_loss(objective, 0.0, 1.0) for objective in objectives
    )
    n = len(objectives)
    neutral_graph = nx.DiGraph()
    neutral_graph.add_nodes_from(range(n))
    return solve_sparse_shift_scale_kl_from_parameters(
        dag=neutral_graph,
        source_intercepts=np.zeros(n),
        coefficients=np.zeros((n, n)),
        noise_variances=np.ones(n),
        target_mean=np.zeros(n),
        target_covariance=np.eye(n),
        budget=budget,
        shift_bounds=shift_bounds,
        log_scale_bounds=(float(np.log(scale_bounds[0])), float(np.log(scale_bounds[1]))),
        precomputed_node_moments=objectives,
        precomputed_kl_constant=constant,
        provenance={
            "protocol": FINITE_SAMPLE_PROTOCOL_VERSION,
            "method": method,
            "estimator": "OAS plus bootstrap precision-times-effect identity",
            "hidden_information_access": "none",
            "optimizer": "validated exact four-state sparse dynamic program",
        },
    )


def _gated_graph_update(
    graph: PartiallyDirectedGraph,
    mechanism: FiniteLocalMechanism,
    query_index: int,
) -> tuple[PartiallyDirectedGraph, tuple[OrientationEvent, ...]]:
    decisions = {item.node: item.orientation_decision for item in mechanism.evidence}
    direct = []
    for neighbor in sorted(graph.undirected_neighbors_of(mechanism.node)):
        decision = decisions.get(neighbor, "unresolved")
        if decision == "parent":
            direct.append((neighbor, mechanism.node))
        elif decision == "nonparent":
            direct.append((mechanism.node, neighbor))
    if not direct:
        return graph.copy(), ()
    candidate = graph.copy()
    try:
        for source, target in direct:
            candidate = candidate.orient(source, target)
        before_meek = candidate.directed_edges
        closed = apply_meek_closure(candidate)
    except ValueError:
        # A conflicting finite-sample batch is rejected atomically.
        return graph.copy(), ()
    events = [
        OrientationEvent(query_index, mechanism.node, source, target, "direct")
        for source, target in direct
    ]
    events.extend(
        OrientationEvent(query_index, mechanism.node, source, target, "meek")
        for source, target in sorted(closed.directed_edges - before_meek)
    )
    return closed, tuple(events)


def run_finite_sample_algorithm1(
    *,
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    probe_provider: FiniteSampleProbeProvider,
    initial_graph: PartiallyDirectedGraph,
    shift_magnitude: float,
    budget: int,
    shift_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
    settings: FiniteSampleSettings,
    bootstrap_seed: int,
    singleton_shift_policy: SingletonShiftPolicy | None = None,
) -> FiniteSampleAlgorithmResult:
    """Run confidence-gated finite-sample Algorithm 1 without hidden inputs."""
    source = estimate_oas(source_samples)
    target = estimate_oas(target_samples)
    n = source.mean.size
    if initial_graph.nodes != frozenset(range(n)):
        raise ValueError("initial graph dimension does not match samples")
    graph = initial_graph.copy()
    policy = singleton_shift_policy or FirstUnresolvedPolicy()
    source_boot = bootstrap_source_cache(
        source_samples,
        replicates=settings.bootstrap_replicates,
        seed=bootstrap_seed,
        backend=settings.backend,
        device=settings.device,
    )
    objectives: dict[int, LocalObjective] = {}
    mechanisms: dict[int, FiniteLocalMechanism] = {}
    events: list[OrientationEvent] = []
    queried: set[int] = set()

    for node in range(n):
        if not graph.undirected_neighbors_of(node):
            objectives[node] = conditional_node_moment(
                node=node,
                parents=graph.parents_of(node),
                source_estimate=source,
                target_estimate=target,
            )

    while len(objectives) < n:
        resolved = frozenset(
            node for node in range(n) if not graph.undirected_neighbors_of(node)
        )
        try:
            node = policy.select_target(
                graph=graph,
                queried_nodes=frozenset(queried),
                resolved_nodes=resolved,
            )
        except RuntimeError:
            remaining = sorted(set(range(n)) - set(objectives) - queried)
            if not remaining:
                break
            node = remaining[0]
        queried.add(node)
        probe_samples = probe_provider.query(node)
        possible = graph.parents_of(node) | graph.undirected_neighbors_of(node)
        nonparents = frozenset(range(n)) - {node} - graph.adjacent_nodes(node)
        mechanism = estimate_local_mechanism(
            node=node,
            source_samples=source_samples,
            intervention_samples=probe_samples,
            shift_magnitude=shift_magnitude,
            source_estimate=source,
            settings=settings,
            bootstrap_seed=bootstrap_seed + 104729 * (node + 1),
            possible_parents=possible,
            known_parents=graph.parents_of(node),
            known_nonparents=nonparents,
            source_bootstrap=source_boot,
        )
        mechanisms[node] = mechanism
        if mechanism.valid:
            objectives[node] = estimated_node_moment(
                node=node,
                incoming_coefficients=mechanism.incoming_coefficients,
                source_intercept=mechanism.source_intercept,
                noise_variance=mechanism.noise_variance,
                target_mean=target.mean,
                target_covariance=target.covariance,
                parents=mechanism.parents,
            )
            graph, new_events = _gated_graph_update(
                graph, mechanism, len(queried)
            )
            events.extend(new_events)
        else:
            # Invalid probe estimates never orient.  A conservative conditional
            # objective on currently known parents permits explicit termination.
            objectives[node] = conditional_node_moment(
                node=node,
                parents=graph.parents_of(node),
                source_estimate=source,
                target_estimate=target,
            )
        for candidate in range(n):
            if candidate not in objectives and not graph.undirected_neighbors_of(candidate):
                objectives[candidate] = conditional_node_moment(
                    node=candidate,
                    parents=graph.parents_of(candidate),
                    source_estimate=source,
                    target_estimate=target,
                )

    if len(objectives) != n:
        raise RuntimeError("finite-sample Algorithm 1 did not resolve all objectives")
    ordered = tuple(objectives[node] for node in range(n))
    dp = _solve_estimated_dp(
        ordered,
        source_estimate=source,
        target_estimate=target,
        budget=budget,
        shift_bounds=shift_bounds,
        scale_bounds=scale_bounds,
        method="finite_sample_algorithm1",
    )
    intervention = ShiftScaleIntervention(dp.budgeted_shift, dp.budgeted_scale)
    query_count = probe_provider.physical_query_count
    return FiniteSampleAlgorithmResult(
        method="finite_sample_algorithm1",
        intervention=intervention,
        dp_result=dp,
        source_estimate=source,
        target_estimate=target,
        mechanisms=tuple(mechanisms[node] for node in sorted(mechanisms)),
        final_graph=graph,
        orientation_events=tuple(events),
        queried_nodes=tuple(sorted(queried)),
        physical_query_count=query_count,
        total_intervention_observations=query_count * probe_provider.n_int,
        total_observations=(
            source_samples.shape[0]
            + target_samples.shape[0]
            + query_count * probe_provider.n_int
        ),
        unresolved_edges=len(graph.undirected_edges),
        invalid_estimates=sum(not item.valid for item in mechanisms.values()),
        termination_reason="all_local_objectives_estimated",
        estimated_initial_kl=gaussian_kl_target_to_candidate(
            target.mean, target.covariance, source.mean, source.covariance
        ),
    )


def run_graph_free_finite_sample(
    *,
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    probe_provider: FiniteSampleProbeProvider,
    shift_magnitude: float,
    budget: int,
    shift_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
    settings: FiniteSampleSettings,
    bootstrap_seed: int,
) -> FiniteSampleAlgorithmResult:
    """Probe every node and estimate every mechanism without graph support."""
    source = estimate_oas(source_samples)
    target = estimate_oas(target_samples)
    n = source.mean.size
    source_boot = bootstrap_source_cache(
        source_samples,
        replicates=settings.bootstrap_replicates,
        seed=bootstrap_seed,
        backend=settings.backend,
        device=settings.device,
    )
    mechanisms = []
    objectives = []
    for node in range(n):
        mechanism = estimate_local_mechanism(
            node=node,
            source_samples=source_samples,
            intervention_samples=probe_provider.query(node),
            shift_magnitude=shift_magnitude,
            source_estimate=source,
            settings=settings,
            bootstrap_seed=bootstrap_seed + 104729 * (node + 1),
            possible_parents=None,
            source_bootstrap=source_boot,
        )
        mechanisms.append(mechanism)
        if mechanism.valid:
            objective = estimated_node_moment(
                node=node,
                incoming_coefficients=mechanism.incoming_coefficients,
                source_intercept=mechanism.source_intercept,
                noise_variance=mechanism.noise_variance,
                target_mean=target.mean,
                target_covariance=target.covariance,
                parents=mechanism.parents,
            )
        else:
            objective = conditional_node_moment(
                node=node,
                parents=frozenset(),
                source_estimate=source,
                target_estimate=target,
            )
        objectives.append(objective)
    dp = _solve_estimated_dp(
        tuple(objectives),
        source_estimate=source,
        target_estimate=target,
        budget=budget,
        shift_bounds=shift_bounds,
        scale_bounds=scale_bounds,
        method="graph_free_finite_sample",
    )
    count = probe_provider.physical_query_count
    return FiniteSampleAlgorithmResult(
        method="graph_free_finite_sample",
        intervention=ShiftScaleIntervention(dp.budgeted_shift, dp.budgeted_scale),
        dp_result=dp,
        source_estimate=source,
        target_estimate=target,
        mechanisms=tuple(mechanisms),
        final_graph=None,
        orientation_events=(),
        queried_nodes=tuple(range(n)),
        physical_query_count=count,
        total_intervention_observations=count * probe_provider.n_int,
        total_observations=source_samples.shape[0] + target_samples.shape[0] + count * probe_provider.n_int,
        unresolved_edges=0,
        invalid_estimates=sum(not item.valid for item in mechanisms),
        termination_reason="all_nodes_probed",
        estimated_initial_kl=gaussian_kl_target_to_candidate(
            target.mean, target.covariance, source.mean, source.covariance
        ),
    )


def run_known_dag_finite_sample(
    *,
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    parent_sets: Mapping[int, frozenset[int]],
    budget: int,
    shift_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
) -> FiniteSampleAlgorithmResult:
    """Known-DAG reference using only sample moments after parent disclosure."""
    source = estimate_oas(source_samples)
    target = estimate_oas(target_samples)
    n = source.mean.size
    if set(parent_sets) != set(range(n)):
        raise ValueError("known-DAG reference requires one parent set per node")
    objectives = tuple(
        conditional_node_moment(
            node=node,
            parents=frozenset(parent_sets[node]),
            source_estimate=source,
            target_estimate=target,
        )
        for node in range(n)
    )
    dp = _solve_estimated_dp(
        objectives,
        source_estimate=source,
        target_estimate=target,
        budget=budget,
        shift_bounds=shift_bounds,
        scale_bounds=scale_bounds,
        method="known_dag_finite_sample",
    )
    return FiniteSampleAlgorithmResult(
        method="known_dag_finite_sample",
        intervention=ShiftScaleIntervention(dp.budgeted_shift, dp.budgeted_scale),
        dp_result=dp,
        source_estimate=source,
        target_estimate=target,
        mechanisms=(),
        final_graph=None,
        orientation_events=(),
        queried_nodes=(),
        physical_query_count=0,
        total_intervention_observations=0,
        total_observations=source_samples.shape[0] + target_samples.shape[0],
        unresolved_edges=0,
        invalid_estimates=0,
        termination_reason="known_parent_sets",
        estimated_initial_kl=gaussian_kl_target_to_candidate(
            target.mean, target.covariance, source.mean, source.covariance
        ),
    )
