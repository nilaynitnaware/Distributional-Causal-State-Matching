"""Canonical, component-level provenance for generated simulation problems.

The pairing identifier describes the intended experimental unit.  These hashes
describe the actual numerical problem consumed by an executor.  They are kept
separate deliberately: equality of an identifier is never treated as evidence
that two serialized runs evaluated the same problem.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from graphs.interventions import INTERVENTION_SEMANTICS
from graphs.problem_gen import GeneratedProblem

PROVENANCE_VERSION = "generated_problem_sha256_v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _array_record(array: np.ndarray) -> dict[str, object]:
    """Return a platform-independent binary record for one numeric array."""
    normalized = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return {
        "dtype": "<f8",
        "shape": list(normalized.shape),
        "data_sha256": hashlib.sha256(normalized.tobytes(order="C")).hexdigest(),
    }


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class ProblemProvenance:
    """Stable hashes of one fully materialized source-target problem."""

    provenance_version: str
    problem_fingerprint: str
    graph_fingerprint: str
    coefficient_fingerprint: str
    target_intervention_fingerprint: str
    source_moment_fingerprint: str
    target_moment_fingerprint: str
    observational_sample_fingerprint: str
    intervention_semantics: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def problem_provenance(problem: GeneratedProblem) -> ProblemProvenance:
    """Fingerprint a problem without unstable repr or native-endian encoding."""
    if not isinstance(problem, GeneratedProblem):
        raise TypeError("problem provenance requires a GeneratedProblem.")

    scm = problem.ground_truth.scm
    n_nodes = problem.public.n_nodes
    edges = sorted((int(parent), int(child)) for parent, child in scm.dag.edges)
    coefficients = _array_record(scm.coefficients)
    source_mean, source_covariance = scm.observational_moments()
    target = problem.ground_truth.generating_intervention
    target_support = {
        "nodes": sorted(int(node) for node in target.targeted_nodes()),
        "shift_nodes": [int(node) for node in np.flatnonzero(target.shift != 0.0)],
        "scale_nodes": [int(node) for node in np.flatnonzero(target.scale != 1.0)],
    }

    graph_record = {"n_nodes": n_nodes, "directed_edges": edges}
    coefficient_record = {"n_nodes": n_nodes, "coefficients": coefficients}
    intervention_record = {
        "intervention_semantics": INTERVENTION_SEMANTICS,
        "shift": _array_record(target.shift),
        "scale": _array_record(target.scale),
        "support": target_support,
    }
    # Record actual independent exogenous noise. The identity-noise default
    # retains the exact legacy record and therefore its fingerprints.
    source_record = {
        "source_intercept": _array_record(np.zeros(n_nodes, dtype=float)),
        "exogenous_noise_covariance": _array_record(scm.noise_covariance),
        "source_mean": _array_record(source_mean),
        "source_covariance": _array_record(source_covariance),
    }
    target_moment_record = {
        "target_mean": _array_record(problem.public.target_mean),
        "target_covariance": _array_record(problem.public.target_covariance),
    }
    sample_record = {
        "observational_samples": _array_record(problem.public.observational_samples)
    }
    generation_record = {
        key: int(value)
        for key, value in asdict(problem.ground_truth.seeds).items()
        if value is not None
    }
    complete_record = {
        "provenance_version": PROVENANCE_VERSION,
        "graph": graph_record,
        "coefficients": coefficient_record,
        "target_intervention": intervention_record,
        "source_moments": source_record,
        "target_moments": target_moment_record,
        "observational_samples": sample_record,
        "generation_seeds": generation_record,
    }
    return ProblemProvenance(
        provenance_version=PROVENANCE_VERSION,
        problem_fingerprint=_digest(complete_record),
        graph_fingerprint=_digest(graph_record),
        coefficient_fingerprint=_digest(coefficient_record),
        target_intervention_fingerprint=_digest(intervention_record),
        source_moment_fingerprint=_digest(source_record),
        target_moment_fingerprint=_digest(target_moment_record),
        observational_sample_fingerprint=_digest(sample_record),
        intervention_semantics=INTERVENTION_SEMANTICS,
    )


PAIRING_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "problem_fingerprint",
    "graph_fingerprint",
    "coefficient_fingerprint",
    "target_intervention_fingerprint",
    "source_moment_fingerprint",
    "target_moment_fingerprint",
    "observational_sample_fingerprint",
)


def provenance_from_result(payload: Mapping[str, Any]) -> dict[str, object]:
    """Extract normalized provenance from one serialized runner result."""
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("result has no metrics mapping.")
    nested = metrics.get("problem_provenance")
    source: Mapping[str, Any] = nested if isinstance(nested, Mapping) else metrics
    required = ("provenance_version", *PAIRING_FINGERPRINT_FIELDS)
    missing = [name for name in required if not isinstance(source.get(name), str)]
    if missing:
        raise ValueError(f"result is missing canonical provenance fields: {missing}.")
    return {name: source[name] for name in required}


def validate_paired_results(
    payloads: Sequence[Mapping[str, Any]],
    *,
    require_multiple: bool = True,
) -> str:
    """Hard-fail unless all records share one ID and one actual problem hash."""
    if not payloads:
        raise ValueError("paired validation requires at least one result.")
    if require_multiple and len(payloads) < 2:
        raise ValueError("paired validation requires at least two results.")
    paired_ids = {payload.get("paired_problem_id") for payload in payloads}
    if len(paired_ids) != 1 or None in paired_ids or "" in paired_ids:
        raise ValueError("paired results must share one nonempty paired_problem_id.")
    provenances = [provenance_from_result(payload) for payload in payloads]
    for field in PAIRING_FINGERPRINT_FIELDS:
        values = {provenance[field] for provenance in provenances}
        if len(values) != 1:
            raise ValueError(
                f"paired problem {next(iter(paired_ids))!r} has unequal {field}."
            )
    return str(provenances[0]["problem_fingerprint"])

CROSS_NOISE_PAIRING_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "graph_fingerprint",
    "coefficient_fingerprint",
    "target_intervention_fingerprint",
)


def validate_cross_noise_regime_results(
    payloads: Sequence[Mapping[str, Any]],
) -> str:
    """Validate structural pairing while permitting noise-dependent quantities."""
    if len(payloads) < 2:
        raise ValueError("cross-noise validation requires at least two results.")
    cross_ids = {payload.get("cross_noise_regime_pair_id") for payload in payloads}
    if len(cross_ids) != 1 or None in cross_ids or "" in cross_ids:
        raise ValueError(
            "cross-noise results must share one nonempty cross_noise_regime_pair_id."
        )
    regimes = {payload.get("noise_variance_regime") for payload in payloads}
    if None in regimes or "" in regimes or len(regimes) < 2:
        raise ValueError("cross-noise validation requires at least two noise regimes.")
    provenances = [provenance_from_result(payload) for payload in payloads]
    for field in CROSS_NOISE_PAIRING_FINGERPRINT_FIELDS:
        values = {provenance[field] for provenance in provenances}
        if len(values) != 1:
            raise ValueError(
                f"cross-noise pair {next(iter(cross_ids))!r} has unequal {field}."
            )
    return str(next(iter(cross_ids)))
