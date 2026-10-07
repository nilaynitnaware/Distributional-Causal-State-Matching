"""Public exact-moment and finite-sample black-box environments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Literal

import numpy as np

from graphs._immutability import immutable_array
from graphs.interventions import INTERVENTION_SEMANTICS, ShiftScaleIntervention
from graphs.scm import LinearGaussianSCM
from matching.capabilities import ProviderCapability
from matching.types import AtomicCoordinateCost, InterventionBackground
from numerical import ComputeConfig, NumericalWorkload, select_backend

QueryKind = Literal["baseline", "probe", "candidate", "repeat"]
QueryBudgetCategory = Literal["baseline", "probe", "candidate"]
_QUERY_BUDGET_CATEGORIES = frozenset({"baseline", "probe", "candidate"})
_QUERY_KINDS = frozenset({"baseline", "probe", "candidate", "repeat"})


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _intervention_key(
    instance_id: str,
    semantics_fingerprint: str,
    intervention: ShiftScaleIntervention,
) -> tuple[str, str, bytes, bytes]:
    return (
        instance_id,
        semantics_fingerprint,
        intervention.shift.tobytes(),
        intervention.scale.tobytes(),
    )


def _intervention_id(intervention: ShiftScaleIntervention) -> str:
    return InterventionBackground.from_intervention(intervention).identifier


def _validate_endogenous_intervention(
    intervention: ShiftScaleIntervention,
    n_endogenous_nodes: int,
) -> None:
    if not isinstance(intervention, ShiftScaleIntervention):
        raise TypeError("intervention must be a ShiftScaleIntervention.")
    if intervention.shift.shape != (n_endogenous_nodes,):
        raise ValueError(
            "intervention coordinates must match the endogenous graph nodes; "
            "exogenous disturbances are not actionable."
        )


@dataclass(frozen=True)
class QueryReservation:
    """One query charged before environment execution."""

    query_id: str
    kind: QueryKind
    budget_category: QueryBudgetCategory
    intervention_id: str
    background_id: str
    requested_sample_size: int
    query_seed: int | None
    action_cost: AtomicCoordinateCost
    executed: bool
    cache_hit: bool


@dataclass
class EnvironmentQueryLedger:
    """Durable in-memory accounting for every black-box execution attempt."""

    max_query_batches: int | None
    max_observations: int | None = None
    max_baseline_queries: int | None = None
    max_probe_queries: int | None = None
    max_candidate_queries: int | None = None
    algorithm_environment_calls: int = 0
    executed_environment_queries: int = 0
    cache_hits: int = 0
    failed_environment_queries: int = 0
    queried_intervention_batches: int = 0
    total_observations: int = 0
    total_observations_sampled: int = 0
    baseline_queries: int = 0
    probe_queries: int = 0
    candidate_queries: int = 0
    repeat_queries: int = 0
    baseline_budget_queries: int = 0
    probe_budget_queries: int = 0
    candidate_budget_queries: int = 0
    failed_queries: int = 0
    shift_coordinate_uses: int = 0
    scale_coordinate_uses: int = 0
    _next_query_index: int = field(default=0, init=False, repr=False)
    _pending: set[str] = field(default_factory=set, init=False, repr=False)
    _observations_recorded: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.max_query_batches is not None:
            _positive_integer("max_query_batches", self.max_query_batches)
        for name in (
            "max_baseline_queries",
            "max_probe_queries",
            "max_candidate_queries",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None.")
        if self.max_observations is not None:
            _positive_integer("max_observations", self.max_observations)

    @property
    def remaining_query_batches(self) -> int | None:
        """Return remaining physical-query capacity, or ``None`` if unbounded."""
        if self.max_query_batches is None:
            return None
        return self.max_query_batches - self.executed_environment_queries

    @property
    def total_atomic_coordinate_uses(self) -> int:
        """Return cumulative shift-plus-scale physical action cost."""
        return self.shift_coordinate_uses + self.scale_coordinate_uses

    def reserve(
        self,
        *,
        kind: QueryKind,
        budget_category: QueryBudgetCategory | None = None,
        intervention: ShiftScaleIntervention,
        background_id: str,
        requested_sample_size: int,
        query_seed: int | None,
        executed: bool = True,
        cache_hit: bool = False,
    ) -> QueryReservation:
        """Record one API call and pessimistically charge physical execution."""
        if kind not in _QUERY_KINDS:
            raise ValueError("unsupported query kind.")
        resolved_category = (
            kind if kind != "repeat" else (budget_category or "candidate")
        )
        if resolved_category not in _QUERY_BUDGET_CATEGORIES:
            raise ValueError("unsupported query budget category.")
        if kind != "repeat" and budget_category not in (None, kind):
            raise ValueError("non-repeat query kind and budget category disagree.")
        if not isinstance(intervention, ShiftScaleIntervention):
            raise TypeError("intervention must be a ShiftScaleIntervention.")
        if not isinstance(background_id, str) or not background_id:
            raise ValueError("background_id must be a nonempty string.")
        if isinstance(requested_sample_size, bool) or not isinstance(
            requested_sample_size, int
        ):
            raise TypeError("requested_sample_size must be an integer.")
        if requested_sample_size < 0:
            raise ValueError("requested_sample_size must be nonnegative.")
        if query_seed is not None and (
            isinstance(query_seed, bool)
            or not isinstance(query_seed, int)
            or query_seed < 0
        ):
            raise ValueError("query_seed must be a nonnegative integer or None.")

        self.algorithm_environment_calls += 1
        query_id = f"q{self._next_query_index:08d}"
        self._next_query_index += 1
        cost = AtomicCoordinateCost.from_intervention(intervention)
        if not executed:
            if not cache_hit:
                raise ValueError("a nonexecuted environment call must be a cache hit.")
            self.cache_hits += 1
            return QueryReservation(
                query_id=query_id,
                kind=kind,
                budget_category=resolved_category,
                intervention_id=_intervention_id(intervention),
                background_id=background_id,
                requested_sample_size=requested_sample_size,
                query_seed=query_seed,
                action_cost=cost,
                executed=False,
                cache_hit=True,
            )
        if cache_hit:
            raise ValueError("an executed query cannot be marked as a cache hit.")
        category_count = getattr(self, f"{resolved_category}_budget_queries")
        category_cap = getattr(self, f"max_{resolved_category}_queries")
        if category_cap is not None and category_count >= category_cap:
            raise RuntimeError(
                f"environment {resolved_category} query budget is exhausted."
            )
        remaining_queries = self.remaining_query_batches
        if remaining_queries is not None and remaining_queries <= 0:
            raise RuntimeError("environment query budget is exhausted.")
        if (
            self.max_observations is not None
            and self.total_observations + requested_sample_size > self.max_observations
        ):
            raise RuntimeError("environment observation budget is exhausted.")

        setattr(
            self,
            f"{resolved_category}_budget_queries",
            category_count + 1,
        )
        self.executed_environment_queries += 1
        self.queried_intervention_batches += 1
        self.failed_environment_queries += 1
        self.failed_queries += 1
        counter_name = {
            "baseline": "baseline_queries",
            "probe": "probe_queries",
            "candidate": "candidate_queries",
            "repeat": "repeat_queries",
        }[kind]
        setattr(self, counter_name, getattr(self, counter_name) + 1)
        self.shift_coordinate_uses += cost.shift_coordinate_uses
        self.scale_coordinate_uses += cost.scale_coordinate_uses
        self._pending.add(query_id)
        return QueryReservation(
            query_id=query_id,
            kind=kind,
            budget_category=resolved_category,
            intervention_id=_intervention_id(intervention),
            background_id=background_id,
            requested_sample_size=requested_sample_size,
            query_seed=query_seed,
            action_cost=cost,
            executed=True,
            cache_hit=False,
        )

    def record_observations(self, reservation: QueryReservation) -> None:
        """Record rows immediately after the hidden sampler returns."""
        if not isinstance(reservation, QueryReservation):
            raise TypeError("reservation must be a QueryReservation.")
        if reservation.query_id not in self._pending:
            raise RuntimeError("query reservation is not pending.")
        if reservation.query_id in self._observations_recorded:
            raise RuntimeError("query observations were already recorded.")
        self.total_observations += reservation.requested_sample_size
        self.total_observations_sampled += reservation.requested_sample_size
        self._observations_recorded.add(reservation.query_id)

    def complete(self, reservation: QueryReservation) -> None:
        """Replace one pessimistic failure after successful post-processing."""
        if not isinstance(reservation, QueryReservation):
            raise TypeError("reservation must be a QueryReservation.")
        if reservation.query_id not in self._pending:
            raise RuntimeError("query reservation is not pending.")
        if (
            reservation.requested_sample_size > 0
            and reservation.query_id not in self._observations_recorded
        ):
            raise RuntimeError("sample observations were not recorded.")
        self._pending.remove(reservation.query_id)
        self._observations_recorded.discard(reservation.query_id)
        self.failed_environment_queries -= 1
        self.failed_queries -= 1

    def snapshot(self) -> dict[str, int | None]:
        """Return JSON-compatible diagnostic counts and configured caps."""
        names = (
            "max_query_batches",
            "algorithm_environment_calls",
            "executed_environment_queries",
            "cache_hits",
            "failed_environment_queries",
            "queried_intervention_batches",
            "total_observations",
            "total_observations_sampled",
            "baseline_queries",
            "probe_queries",
            "candidate_queries",
            "repeat_queries",
            "baseline_budget_queries",
            "probe_budget_queries",
            "candidate_budget_queries",
            "failed_queries",
            "shift_coordinate_uses",
            "scale_coordinate_uses",
        )
        result = {
            name: (None if getattr(self, name) is None else int(getattr(self, name)))
            for name in names
        }
        for category in _QUERY_BUDGET_CATEGORIES:
            cap = getattr(self, f"max_{category}_queries")
            if cap is not None:
                result[f"max_{category}_queries"] = cap
        if self.max_observations is not None:
            result["max_observations"] = self.max_observations
        return result


@dataclass(frozen=True)
class QueryMetadata:
    """Public metadata common to exact and sampled query results."""

    query_id: str
    kind: QueryKind
    budget_category: QueryBudgetCategory
    instance_id: str
    intervention_id: str
    background_id: str
    requested_sample_size: int
    query_seed: int | None
    cache_hit: bool
    executed_environment_query: bool = True
    trial_cost: int = 1


@dataclass(frozen=True)
class ExactMomentQueryResult:
    """Exact public moments with no structural or derivative information."""

    mean: np.ndarray
    covariance: np.ndarray
    metadata: QueryMetadata

    def __post_init__(self) -> None:
        mean = immutable_array(self.mean, dtype=float)
        covariance = immutable_array(self.covariance, dtype=float)
        if mean.ndim != 1 or covariance.shape != (mean.size, mean.size):
            raise ValueError("exact moment dimensions must agree.")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("exact moments must be finite.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class SampleQueryResult:
    """Public samples and query metadata, with no exact hidden moments."""

    samples: np.ndarray
    metadata: QueryMetadata

    def __post_init__(self) -> None:
        samples = immutable_array(self.samples, dtype=float)
        if samples.ndim != 2 or not np.all(np.isfinite(samples)):
            raise ValueError("samples must be a finite two-dimensional array.")
        if samples.shape[0] != self.metadata.requested_sample_size:
            raise ValueError("sample row count must match requested_sample_size.")
        object.__setattr__(self, "samples", samples)

    @property
    def sample_size(self) -> int:
        """Return the public number of observations."""
        return int(self.samples.shape[0])


class ExactMomentBlackBoxEnvironment:
    """Hidden-SCM execution boundary exposing only exact public moments."""

    capabilities = frozenset({ProviderCapability.EXACT_PUBLIC_MOMENTS})

    def __init__(
        self,
        hidden_scm: LinearGaussianSCM,
        *,
        instance_id: str,
        ledger: EnvironmentQueryLedger,
        environment_semantics: str = INTERVENTION_SEMANTICS,
        numerical_configuration: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(hidden_scm, LinearGaussianSCM):
            raise TypeError("hidden_scm must be a LinearGaussianSCM.")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("instance_id must be a nonempty string.")
        if not isinstance(ledger, EnvironmentQueryLedger):
            raise TypeError("ledger must be an EnvironmentQueryLedger.")
        if not isinstance(environment_semantics, str) or not environment_semantics:
            raise ValueError("environment_semantics must be a nonempty string.")
        configuration = dict(numerical_configuration or {})
        self._n_endogenous_nodes = hidden_scm.dag.number_of_nodes()
        selection = select_backend(
            ComputeConfig(
                backend=configuration.get("backend", "numpy"),
                device=configuration.get("device", "cpu"),
            ),
            NumericalWorkload(
                operation="interventional_moments",
                n_nodes=self._n_endogenous_nodes,
                batch_size=1,
            ),
        )
        self.backend_diagnostics = MappingProxyType(asdict(selection.diagnostics))
        if selection.diagnostics.backend == "torch":
            prepare = selection.backend.prepare_scm_batch
            prepared_scm = prepare(
                hidden_scm.coefficients,
                hidden_scm.noise_variances,
            )

            def hidden_moments(
                intervention: ShiftScaleIntervention,
            ) -> tuple[np.ndarray, np.ndarray]:
                mean, covariance = prepared_scm.interventional_moments(
                    intervention.shift,
                    intervention.scale,
                )
                return (
                    mean[0].detach().cpu().numpy().copy(),
                    covariance[0].detach().cpu().numpy().copy(),
                )

            self.__hidden_moments = hidden_moments
        else:
            self.__hidden_moments = hidden_scm.interventional_moments
        self.instance_id = instance_id
        self.ledger = ledger
        self.environment_semantics = environment_semantics
        self.numerical_configuration = MappingProxyType(configuration)
        self._semantics_fingerprint = sha256(
            json.dumps(
                {
                    "environment_semantics": environment_semantics,
                    "numerical_configuration": configuration,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        self._cache: dict[
            tuple[str, str, bytes, bytes], tuple[np.ndarray, np.ndarray]
        ] = {}

    def query(
        self,
        intervention: ShiftScaleIntervention,
        *,
        kind: QueryKind,
        background_id: str,
    ) -> ExactMomentQueryResult:
        """Return moments while charging only an uncached physical execution."""
        _validate_endogenous_intervention(
            intervention,
            self._n_endogenous_nodes,
        )
        key = _intervention_key(
            self.instance_id,
            self._semantics_fingerprint,
            intervention,
        )
        cache_hit = key in self._cache
        reservation = self.ledger.reserve(
            kind=kind,
            intervention=intervention,
            background_id=background_id,
            requested_sample_size=0,
            query_seed=None,
            executed=not cache_hit,
            cache_hit=cache_hit,
        )
        if cache_hit:
            mean, covariance = self._cache[key]
        else:
            mean, covariance = self.__hidden_moments(intervention)
        metadata = QueryMetadata(
            query_id=reservation.query_id,
            kind=kind,
            budget_category=reservation.budget_category,
            instance_id=self.instance_id,
            intervention_id=reservation.intervention_id,
            background_id=reservation.background_id,
            requested_sample_size=0,
            query_seed=None,
            cache_hit=cache_hit,
            executed_environment_query=reservation.executed,
            trial_cost=1 if reservation.executed else 0,
        )
        result = ExactMomentQueryResult(mean, covariance, metadata)
        if reservation.executed:
            self._cache[key] = (result.mean, result.covariance)
            self.ledger.complete(reservation)
        return result


class FiniteSampleBlackBoxEnvironment:
    """Hidden-SCM boundary exposing only independently sampled observations."""

    capabilities = frozenset({ProviderCapability.EMPIRICAL_SAMPLES})
    paired_samples = False

    def __init__(
        self,
        hidden_scm: LinearGaussianSCM,
        *,
        instance_id: str,
        ledger: EnvironmentQueryLedger,
    ) -> None:
        if not isinstance(hidden_scm, LinearGaussianSCM):
            raise TypeError("hidden_scm must be a LinearGaussianSCM.")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("instance_id must be a nonempty string.")
        if not isinstance(ledger, EnvironmentQueryLedger):
            raise TypeError("ledger must be an EnvironmentQueryLedger.")
        self.__hidden_sampler = hidden_scm.sample_interventional
        self._n_endogenous_nodes = hidden_scm.dag.number_of_nodes()
        self.instance_id = instance_id
        self.ledger = ledger
        self._query_history: list[QueryMetadata] = []

    @property
    def query_history(self) -> tuple[QueryMetadata, ...]:
        return tuple(self._query_history)

    def query(
        self,
        intervention: ShiftScaleIntervention,
        *,
        sample_size: int,
        query_seed: int,
        kind: QueryKind,
        budget_category: QueryBudgetCategory | None = None,
        background_id: str,
    ) -> SampleQueryResult:
        """Execute one charged, reproducible finite-sample query."""
        _validate_endogenous_intervention(
            intervention,
            self._n_endogenous_nodes,
        )
        _positive_integer("sample_size", sample_size)
        reservation = self.ledger.reserve(
            kind=kind,
            intervention=intervention,
            budget_category=budget_category,
            background_id=background_id,
            requested_sample_size=sample_size,
            query_seed=query_seed,
        )
        samples = self.__hidden_sampler(
            intervention,
            sample_size,
            np.random.default_rng(query_seed),
        )
        self.ledger.record_observations(reservation)
        metadata = QueryMetadata(
            query_id=reservation.query_id,
            budget_category=reservation.budget_category,
            kind=kind,
            instance_id=self.instance_id,
            intervention_id=reservation.intervention_id,
            background_id=reservation.background_id,
            requested_sample_size=sample_size,
            query_seed=query_seed,
            cache_hit=False,
        )
        result = SampleQueryResult(samples, metadata)
        self.ledger.complete(reservation)
        self._query_history.append(metadata)
        return result


def deterministic_instance_id(label: str, seed: int) -> str:
    """Derive a stable public simulation-instance identifier."""
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a nonempty string.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    digest = sha256(f"{label}:{seed}".encode()).hexdigest()
    return digest


EMPTY_PUBLIC_METADATA = MappingProxyType({})
