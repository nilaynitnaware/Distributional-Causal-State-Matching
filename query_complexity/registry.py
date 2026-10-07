"""Versioned condition registry for query campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .core import PROTOCOL_VERSION

ConditionName = Literal[
    "legacy_public_greedy",
    "known_dag_exact_dp",
    "graph_free_blackbox_dp",
    "graph_free_total_effect_dp",
    "passive_cpdag_hybrid_dp",
    "active_graph_hybrid_dp",
    "adaptive_singleton_shift_dp",
    "public_adaptive_singleton_shift_dp",
    "public_adaptive_local_mechanism_dp",
]


@dataclass(frozen=True)
class ConditionDescriptor:
    method: ConditionName
    implementation: str
    information_regime: str
    experimental_query_semantics: str


CONDITION_REGISTRY = {
    "legacy_public_greedy": ConditionDescriptor(
        "legacy_public_greedy",
        "historical existing public matcher; unchanged",
        "public graph and public outcomes",
        "historical accounting",
    ),
    "known_dag_exact_dp": ConditionDescriptor(
        "known_dag_exact_dp",
        "oracle.sparse_shift_scale_kl.solve_sparse_shift_scale_kl",
        "hidden DAG and structural parameters; evaluation only",
        "zero identification queries",
    ),
    "graph_free_blackbox_dp": ConditionDescriptor(
        "graph_free_blackbox_dp",
        "query_complexity.run_query_method",
        "public observational/target moments and exact responses",
        "five singleton queries per node; no baseline query",
    ),
    "graph_free_total_effect_dp": ConditionDescriptor(
        "graph_free_total_effect_dp",
        "query_complexity.run_query_method",
        "public observational moments and singleton shift responses",
        "one singleton shift per node; no baseline query",
    ),
    "passive_cpdag_hybrid_dp": ConditionDescriptor(
        "passive_cpdag_hybrid_dp",
        "query_complexity.run_query_method",
        "public source moments and supplied/current PDAG",
        "analytic resolved nodes plus five-query fallback",
    ),
    "active_graph_hybrid_dp": ConditionDescriptor(
        "active_graph_hybrid_dp",
        "query_complexity.run_query_method",
        "public source moments, PDAG, and public probe outcomes",
        "pure-shift probes plus analytic resolution and exact fallback",
    ),
    "public_adaptive_singleton_shift_dp": ConditionDescriptor(
        "public_adaptive_singleton_shift_dp",
        "query_complexity.run_query_method",
        "public source moments, PDAG, and exact singleton-shift moments only",
        "conservative nonzero-response orientations plus at most n total-effect columns",
    ),
    "public_adaptive_local_mechanism_dp": ConditionDescriptor(
        "public_adaptive_local_mechanism_dp",
        "query_complexity.run_query_method",
        "public source/target moments, PDAG, and exact singleton-shift means",
        "one physical query recovers one local mechanism and orients all incident edges",
    ),
    "adaptive_singleton_shift_dp": ConditionDescriptor(
        "adaptive_singleton_shift_dp",
        "query_complexity.run_query_method",
        "public moments plus idealized hidden-DAG structural probe payload",
        "at most one distinct singleton shift per node; information roles are non-additive",
    ),
}


def validate_condition_registry() -> None:
    expected = {
        "legacy_public_greedy",
        "known_dag_exact_dp",
        "graph_free_blackbox_dp",
        "graph_free_total_effect_dp",
        "passive_cpdag_hybrid_dp",
        "active_graph_hybrid_dp",
        "adaptive_singleton_shift_dp",
        "public_adaptive_singleton_shift_dp",
        "public_adaptive_local_mechanism_dp",
    }
    if set(CONDITION_REGISTRY) != expected:
        raise RuntimeError("query condition registry is incomplete")
    if PROTOCOL_VERSION != "kl_query_complexity_v1":
        raise RuntimeError("unexpected protocol version")
