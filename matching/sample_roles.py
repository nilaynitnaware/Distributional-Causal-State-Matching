"""Immutable, deterministic training/validation sample-role primitives."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from environments.blackbox import FiniteSampleBlackBoxEnvironment
from graphs._immutability import immutable_array
from graphs.interventions import ShiftScaleIntervention

from .types import InterventionBackground

SampleRole = Literal[
    "training_target",
    "training_current",
    "training_candidate",
    "validation_target",
    "validation_current",
    "validation_candidate",
]
ValidationRole = Literal[
    "validation_target",
    "validation_current",
    "validation_candidate",
]


def sample_fingerprint(samples: np.ndarray) -> str:
    """Return a stable content hash for a public sample matrix."""
    contiguous = np.ascontiguousarray(np.asarray(samples, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.dtype.str.encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def deterministic_role_seed(
    *,
    namespace: str,
    paired_problem_id: str,
    background_id: str,
    candidate_id: str,
    role: SampleRole,
    validation_look: int,
    validation_cap: int,
    replication_index: int,
    sample_size: int,
) -> int:
    """Derive one seed from every split/comparison provenance dimension."""
    values = (
        namespace,
        paired_problem_id,
        background_id,
        candidate_id,
        role,
    )
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("role seed string fields must be nonempty.")
    integers = (validation_look, validation_cap, replication_index, sample_size)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integers):
        raise ValueError("role seed integer fields must be nonnegative integers.")
    digest = hashlib.sha256(
        ":".join((*values, *(str(value) for value in integers))).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % np.iinfo(np.int64).max


@dataclass(frozen=True)
class RoleSampleBatch:
    """One immutable independently generated role-labelled batch."""

    samples: np.ndarray
    role: ValidationRole
    paired_problem_id: str
    background_id: str
    candidate_id: str
    validation_look: int
    validation_cap: int
    replication_index: int
    batch_seed: int
    namespace: str
    query_id: str | None = None
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        frozen = immutable_array(self.samples, dtype=float)
        if frozen.ndim != 2 or frozen.shape[0] <= 0 or not np.all(np.isfinite(frozen)):
            raise ValueError("role samples must be a nonempty finite matrix.")
        if self.role not in {
            "validation_target",
            "validation_current",
            "validation_candidate",
        }:
            raise ValueError("unsupported validation role.")
        for name in ("paired_problem_id", "background_id", "candidate_id", "namespace"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string.")
        for name in (
            "validation_look",
            "validation_cap",
            "replication_index",
            "batch_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        object.__setattr__(self, "samples", frozen)
        object.__setattr__(self, "fingerprint", sample_fingerprint(frozen))

    @property
    def sample_size(self) -> int:
        return int(self.samples.shape[0])

    def metadata(self) -> dict[str, object]:
        return {
            "role": self.role,
            "paired_problem_id": self.paired_problem_id,
            "background_id": self.background_id,
            "candidate_id": self.candidate_id,
            "validation_look": self.validation_look,
            "validation_cap": self.validation_cap,
            "replication_index": self.replication_index,
            "sample_size": self.sample_size,
            "batch_seed": self.batch_seed,
            "namespace": self.namespace,
            "query_id": self.query_id,
            "sample_fingerprint": self.fingerprint,
        }


@dataclass
class NestedRoleSamplePool:
    """Append-only pooled history for exactly one validation distribution."""

    role: ValidationRole
    paired_problem_id: str
    background_id: str
    candidate_id: str
    _batches: list[RoleSampleBatch] = field(default_factory=list, init=False, repr=False)

    @property
    def batches(self) -> tuple[RoleSampleBatch, ...]:
        return tuple(self._batches)

    @property
    def sample_size(self) -> int:
        return sum(batch.sample_size for batch in self._batches)

    @property
    def samples(self) -> np.ndarray:
        if not self._batches:
            raise RuntimeError("sample pool is empty.")
        return immutable_array(
            np.concatenate([batch.samples for batch in self._batches], axis=0),
            dtype=float,
        )

    @property
    def pooled_fingerprint(self) -> str | None:
        return None if not self._batches else sample_fingerprint(self.samples)

    def append(self, batch: RoleSampleBatch) -> None:
        if not isinstance(batch, RoleSampleBatch):
            raise TypeError("batch must be a RoleSampleBatch.")
        identity = (
            batch.role,
            batch.paired_problem_id,
            batch.background_id,
            batch.candidate_id,
        )
        expected = (self.role, self.paired_problem_id, self.background_id, self.candidate_id)
        if identity != expected:
            raise ValueError("batch identity does not match its validation pool.")
        if batch.replication_index != len(self._batches):
            raise ValueError("validation batches must append in replication order.")
        if self._batches and batch.validation_look <= self._batches[-1].validation_look:
            raise ValueError("validation looks must increase strictly.")
        if batch.validation_cap != self.sample_size + batch.sample_size:
            raise ValueError("batch cap must equal the new pooled sample size.")
        self._batches.append(batch)

    def history(self) -> dict[str, object]:
        return {
            "role": self.role,
            "paired_problem_id": self.paired_problem_id,
            "background_id": self.background_id,
            "candidate_id": self.candidate_id,
            "sample_size": self.sample_size,
            "pooled_sample_fingerprint": self.pooled_fingerprint,
            "batches": [batch.metadata() for batch in self._batches],
        }


class KeyedTargetValidationSampler:
    """Sample-only deterministic target boundary, bound outside matching logic."""

    def __init__(
        self,
        sampler: Callable[[int, np.random.Generator], np.ndarray],
        *,
        namespace: str,
        paired_problem_id: str,
    ) -> None:
        if not callable(sampler):
            raise TypeError("sampler must be callable.")
        self.__sampler = sampler
        self.namespace = namespace
        self.paired_problem_id = paired_problem_id
        self._cache: dict[tuple[object, ...], RoleSampleBatch] = {}
        self.sample_audit: list[dict[str, object]] = []

    def sample_batch(
        self,
        *,
        background_id: str,
        validation_look: int,
        validation_cap: int,
        replication_index: int,
        sample_size: int,
    ) -> RoleSampleBatch:
        candidate_id = "shared_target"
        key = (
            background_id,
            validation_look,
            validation_cap,
            replication_index,
            sample_size,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        seed = deterministic_role_seed(
            namespace=self.namespace,
            paired_problem_id=self.paired_problem_id,
            background_id=background_id,
            candidate_id=candidate_id,
            role="validation_target",
            validation_look=validation_look,
            validation_cap=validation_cap,
            replication_index=replication_index,
            sample_size=sample_size,
        )
        batch = RoleSampleBatch(
            samples=self.__sampler(sample_size, np.random.default_rng(seed)),
            role="validation_target",
            paired_problem_id=self.paired_problem_id,
            background_id=background_id,
            candidate_id=candidate_id,
            validation_look=validation_look,
            validation_cap=validation_cap,
            replication_index=replication_index,
            batch_seed=seed,
            namespace=self.namespace,
        )
        self._cache[key] = batch
        self.sample_audit.append(batch.metadata())
        return batch


class RoleSeparatedValidationEnvironment:
    """Role-keyed wrapper around the existing finite black-box environment."""

    def __init__(
        self,
        environment: FiniteSampleBlackBoxEnvironment,
        *,
        namespace: str,
        paired_problem_id: str,
    ) -> None:
        if not isinstance(environment, FiniteSampleBlackBoxEnvironment):
            raise TypeError("environment must be FiniteSampleBlackBoxEnvironment.")
        self.environment = environment
        self.namespace = namespace
        self.paired_problem_id = paired_problem_id
        self._cache: dict[tuple[object, ...], RoleSampleBatch] = {}
        self.sample_audit: list[dict[str, object]] = []

    def query_batch(
        self,
        intervention: ShiftScaleIntervention,
        *,
        role: Literal["validation_current", "validation_candidate"],
        background_id: str,
        candidate_id: str,
        validation_look: int,
        validation_cap: int,
        replication_index: int,
        sample_size: int,
    ) -> RoleSampleBatch:
        intervention_id = InterventionBackground.from_intervention(intervention).identifier
        if role == "validation_current" and intervention_id != background_id:
            raise ValueError("validation current intervention must match background.")
        if role == "validation_candidate" and intervention_id != candidate_id:
            raise ValueError("validation candidate ID must match its intervention.")
        key = (
            intervention_id,
            role,
            background_id,
            candidate_id,
            validation_look,
            validation_cap,
            replication_index,
            sample_size,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        seed = deterministic_role_seed(
            namespace=self.namespace,
            paired_problem_id=self.paired_problem_id,
            background_id=background_id,
            candidate_id=candidate_id,
            role=role,
            validation_look=validation_look,
            validation_cap=validation_cap,
            replication_index=replication_index,
            sample_size=sample_size,
        )
        result = self.environment.query(
            intervention,
            sample_size=sample_size,
            query_seed=seed,
            kind="baseline" if role == "validation_current" else "candidate",
            background_id=background_id,
        )
        batch = RoleSampleBatch(
            samples=result.samples,
            role=role,
            paired_problem_id=self.paired_problem_id,
            background_id=background_id,
            candidate_id=candidate_id,
            validation_look=validation_look,
            validation_cap=validation_cap,
            replication_index=replication_index,
            batch_seed=seed,
            namespace=self.namespace,
            query_id=result.metadata.query_id,
        )
        self._cache[key] = batch
        self.sample_audit.append(batch.metadata())
        return batch
