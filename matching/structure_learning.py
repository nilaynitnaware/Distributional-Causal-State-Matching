"""Replaceable exact and PC-style observational CPDAG providers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Real
from statistics import NormalDist

import numpy as np

from graphs.meek import (
    MeekOrientationConflict,
    MeekOrientationConflictDetail,
    apply_meek_closure,
)
from graphs.pdag import PartiallyDirectedGraph

from .capabilities import ProviderCapability
from .empirical_moments import EmpiricalMomentsProvider


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class PCStructureSettings:
    """Partial-correlation/Fisher-z structure-estimation settings."""

    significance_level: float
    max_conditioning_set_size: int
    minimum_effective_sample_size: int
    population_partial_correlation_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        alpha = _finite_real("significance_level", self.significance_level)
        if not 0.0 < alpha < 1.0:
            raise ValueError("significance_level must lie in (0, 1).")
        object.__setattr__(self, "significance_level", alpha)
        for name in (
            "max_conditioning_set_size",
            "minimum_effective_sample_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if self.minimum_effective_sample_size < 4:
            raise ValueError("minimum_effective_sample_size must be at least four.")
        tolerance = _finite_real(
            "population_partial_correlation_tolerance",
            self.population_partial_correlation_tolerance,
        )
        if tolerance < 0.0:
            raise ValueError(
                "population_partial_correlation_tolerance must be nonnegative."
            )
        object.__setattr__(
            self,
            "population_partial_correlation_tolerance",
            tolerance,
        )


@dataclass(frozen=True)
class StructureEstimationMetadata:
    """Auditable CI tests, separating sets, and numerical failures."""

    provider: str
    tests_performed: int
    separating_sets: tuple[tuple[tuple[int, int], tuple[int, ...]], ...]
    unresolved_tests: int
    failed_tests: tuple[str, ...]
    regularization: dict[str, object] | None
    sample_size: int
    deterministic_variable_order: tuple[int, ...]
    orientation_conflicts: tuple[MeekOrientationConflictDetail, ...] = ()


@dataclass(frozen=True)
class StructureEstimate:
    """One standalone public graph snapshot and estimation metadata."""

    graph: PartiallyDirectedGraph
    metadata: StructureEstimationMetadata


@dataclass(frozen=True)
class SuppliedExactCPDAGProvider:
    """Return an already-separated public CPDAG without retaining its DAG."""

    public_cpdag: PartiallyDirectedGraph
    capabilities = frozenset({ProviderCapability.EXACT_PUBLIC_STRUCTURE})

    def estimate(self) -> StructureEstimate:
        """Return an independent immutable public snapshot."""
        graph = self.public_cpdag.copy()
        return StructureEstimate(
            graph,
            StructureEstimationMetadata(
                provider="supplied_exact_public_cpdag",
                tests_performed=0,
                separating_sets=(),
                unresolved_tests=0,
                failed_tests=(),
                regularization=None,
                sample_size=0,
                deterministic_variable_order=tuple(sorted(graph.nodes)),
            ),
        )


@dataclass(frozen=True)
class EstimatedPCCPDAGProvider:
    """Learn a CPDAG using only public observational samples/covariance."""

    settings: PCStructureSettings
    moments_provider: EmpiricalMomentsProvider
    capabilities = frozenset({ProviderCapability.ESTIMATED_STRUCTURE})

    def __post_init__(self) -> None:
        if not isinstance(self.settings, PCStructureSettings):
            raise TypeError("settings must be PCStructureSettings.")
        if not isinstance(self.moments_provider, EmpiricalMomentsProvider):
            raise TypeError("moments_provider must be EmpiricalMomentsProvider.")

    def estimate(self, observational_samples: np.ndarray) -> StructureEstimate:
        """Estimate moments, skeleton, colliders, and Meek closure."""
        if observational_samples.shape[0] < (
            self.settings.minimum_effective_sample_size
        ):
            raise ValueError(
                "observational samples are below minimum_effective_sample_size."
            )
        moments = self.moments_provider.estimate(observational_samples)
        return self._estimate_from_covariance(
            moments.regularized_covariance,
            sample_size=observational_samples.shape[0],
            population=False,
            regularization=moments.metadata(),
        )

    def estimate_from_population_covariance(
        self,
        covariance: np.ndarray,
    ) -> StructureEstimate:
        """Run deterministic population-CI PC using a configured zero tolerance."""
        if not isinstance(covariance, np.ndarray):
            raise TypeError("covariance must be a numpy.ndarray.")
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be square.")
        return self._estimate_from_covariance(
            np.asarray(covariance, dtype=float),
            sample_size=0,
            population=True,
            regularization=None,
        )

    def _estimate_from_covariance(
        self,
        covariance: np.ndarray,
        *,
        sample_size: int,
        population: bool,
        regularization: dict[str, object] | None,
    ) -> StructureEstimate:
        dimension = covariance.shape[0]
        nodes = tuple(range(dimension))
        adjacency = {node: set(nodes) - {node} for node in nodes}
        separating_sets: dict[tuple[int, int], tuple[int, ...]] = {}
        tests_performed = 0
        unresolved_tests = 0
        failed_tests: list[str] = []
        orientation_conflicts: list[MeekOrientationConflictDetail] = []

        for conditioning_size in range(self.settings.max_conditioning_set_size + 1):
            edges = [
                (node_a, node_b)
                for node_a in nodes
                for node_b in sorted(adjacency[node_a])
                if node_a < node_b
            ]
            any_testable = False
            for node_a, node_b in edges:
                if node_b not in adjacency[node_a]:
                    continue
                pools = (
                    sorted(adjacency[node_a] - {node_b}),
                    sorted(adjacency[node_b] - {node_a}),
                )
                conditioning_sets = sorted(
                    {
                        subset
                        for pool in pools
                        if len(pool) >= conditioning_size
                        for subset in combinations(pool, conditioning_size)
                    }
                )
                if not conditioning_sets:
                    continue
                any_testable = True
                for conditioning_set in conditioning_sets:
                    tests_performed += 1
                    try:
                        partial = _partial_correlation(
                            covariance,
                            node_a,
                            node_b,
                            conditioning_set,
                        )
                    except np.linalg.LinAlgError:
                        failed_tests.append(
                            f"singular:{node_a}:{node_b}:{conditioning_set}"
                        )
                        continue
                    if population:
                        independent = (
                            abs(partial)
                            <= self.settings.population_partial_correlation_tolerance
                        )
                    elif sample_size <= len(conditioning_set) + 3:
                        unresolved_tests += 1
                        continue
                    else:
                        statistic = np.sqrt(
                            sample_size - len(conditioning_set) - 3
                        ) * abs(np.arctanh(np.clip(partial, -0.999999, 0.999999)))
                        critical = NormalDist().inv_cdf(
                            1.0 - self.settings.significance_level / 2.0
                        )
                        independent = statistic <= critical
                    if independent:
                        adjacency[node_a].remove(node_b)
                        adjacency[node_b].remove(node_a)
                        separating_sets[(node_a, node_b)] = tuple(conditioning_set)
                        break
            if not any_testable:
                break

        undirected_edges = {
            (node_a, node_b)
            for node_a in nodes
            for node_b in adjacency[node_a]
            if node_a < node_b
        }
        collider_directions: dict[frozenset[int], set[tuple[int, int]]] = {}
        for middle in nodes:
            for endpoint_a, endpoint_b in combinations(
                sorted(adjacency[middle]),
                2,
            ):
                if endpoint_b in adjacency[endpoint_a]:
                    continue
                key = tuple(sorted((endpoint_a, endpoint_b)))
                if middle in separating_sets.get(key, ()):
                    continue
                for endpoint in (endpoint_a, endpoint_b):
                    pair = frozenset((endpoint, middle))
                    collider_directions.setdefault(pair, set()).add((endpoint, middle))

        graph = PartiallyDirectedGraph(
            nodes=frozenset(nodes),
            directed_edges=frozenset(),
            undirected_edges=frozenset(undirected_edges),
        )
        for pair in sorted(
            collider_directions,
            key=lambda item: tuple(sorted(item)),
        ):
            directions = collider_directions[pair]
            if len(directions) != 1:
                failed_tests.append(f"collider_conflict:{tuple(sorted(pair))}")
                continue
            source, target = next(iter(directions))
            try:
                graph = graph.orient(source, target)
            except ValueError as error:
                failed_tests.append(f"collider_orientation:{source}:{target}:{error}")
        try:
            graph = apply_meek_closure(graph)
        except MeekOrientationConflict as error:
            orientation_conflicts.append(error.detail)
            failed_tests.append(f"meek_closure:{error}")
        except ValueError as error:
            failed_tests.append(f"meek_closure:{error}")

        separating_metadata = tuple(
            (edge, separating_sets[edge]) for edge in sorted(separating_sets)
        )
        return StructureEstimate(
            graph=graph,
            metadata=StructureEstimationMetadata(
                provider=(
                    "estimated_pc_population" if population else "estimated_pc_fisher_z"
                ),
                tests_performed=tests_performed,
                separating_sets=separating_metadata,
                unresolved_tests=unresolved_tests,
                failed_tests=tuple(failed_tests),
                regularization=regularization,
                sample_size=sample_size,
                deterministic_variable_order=nodes,
                orientation_conflicts=tuple(orientation_conflicts),
            ),
        )


def _partial_correlation(
    covariance: np.ndarray,
    node_a: int,
    node_b: int,
    conditioning_set: tuple[int, ...],
) -> float:
    indices = (node_a, node_b, *conditioning_set)
    subcovariance = covariance[np.ix_(indices, indices)]
    precision = np.linalg.inv(subcovariance)
    denominator = np.sqrt(precision[0, 0] * precision[1, 1])
    if denominator <= 0.0:
        raise np.linalg.LinAlgError("nonpositive partial-correlation denominator")
    return float(np.clip(-precision[0, 1] / denominator, -1.0, 1.0))
