"""Deterministic public-data shortlist for held-out finite-sample matching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.pdag import PartiallyDirectedGraph

from .types import FinalSupportBudget, ParameterCoordinate, ParameterKind

SHORTLIST_SIZE = 3
KIND_ORDER: tuple[ParameterKind, ...] = ("shift", "log_scale")


@dataclass(frozen=True)
class ShortlistEntry:
    """One frozen coordinate selected using public training evidence only."""

    shortlist_position: int
    coordinate: ParameterCoordinate
    node_local_loss: float
    node_score_rank: int
    selection_pass: int
    kind_rank: int
    exact_score_tie_with_previous: bool
    tie_break_rule: str

    def as_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["coordinate"] = {
            "node": self.coordinate.node,
            "kind": self.coordinate.kind,
        }
        return row


@dataclass(frozen=True)
class DeterministicShortlist:
    """Top-three result plus a complete legality and ordering trace."""

    entries: tuple[ShortlistEntry, ...]
    node_order: tuple[int, ...]
    trace: tuple[dict[str, object], ...]
    shortlist_size: int = SHORTLIST_SIZE
    kind_order: tuple[ParameterKind, ...] = KIND_ORDER
    rule_identifier: str = "top3_distinct_nodes_then_second_kind_v1"

    def __post_init__(self) -> None:
        if self.shortlist_size != SHORTLIST_SIZE:
            raise ValueError("the held-out protocol fixes shortlist_size at three.")
        if self.kind_order != KIND_ORDER:
            raise ValueError("the held-out protocol fixes shift before log_scale.")
        if len(self.entries) > SHORTLIST_SIZE:
            raise ValueError("shortlist contains more than three coordinates.")


def deterministic_top3_shortlist(
    node_local_losses: Sequence[float],
    graph: PartiallyDirectedGraph,
    current: ShiftScaleIntervention,
    final_support_budget: FinalSupportBudget,
    *,
    allowed_coordinate_kinds: tuple[ParameterKind, ...] = KIND_ORDER,
) -> DeterministicShortlist:
    """Rank legal coordinates without thresholds, intervals, or hidden truth.

    Shift is preferred. Log-scale is public-legal only when the supplied CPDAG
    has a directed parent. Distinct nodes fill the first pass before a node's
    second legal kind can enter the shortlist.
    """
    if not isinstance(graph, PartiallyDirectedGraph):
        raise TypeError("graph must be a PartiallyDirectedGraph.")
    if not isinstance(current, ShiftScaleIntervention):
        raise TypeError("current must be a ShiftScaleIntervention.")
    if not isinstance(final_support_budget, FinalSupportBudget):
        raise TypeError("final_support_budget must be a FinalSupportBudget.")
    if current.shift.shape != (len(node_local_losses),):
        raise ValueError("node losses must match the intervention dimension.")
    scores = tuple(float(value) for value in node_local_losses)
    if not all(np.isfinite(value) for value in scores):
        raise ValueError("node local losses must be finite.")
    if (
        not isinstance(allowed_coordinate_kinds, tuple)
        or not allowed_coordinate_kinds
        or any(kind not in KIND_ORDER for kind in allowed_coordinate_kinds)
    ):
        raise ValueError("allowed_coordinate_kinds must contain public kinds.")

    node_order = tuple(sorted(range(len(scores)), key=lambda node: (-scores[node], node)))
    remaining = final_support_budget.remaining_after(current)
    legal_by_node: dict[int, tuple[ParameterKind, ...]] = {}
    trace: list[dict[str, object]] = []
    for rank, node in enumerate(node_order):
        legal: list[ParameterKind] = []
        for kind in KIND_ORDER:
            if kind not in allowed_coordinate_kinds:
                continue
            if kind == "log_scale" and not graph.parents_of(node):
                continue
            already_active = (
                not np.isclose(current.shift[node], 0.0)
                if kind == "shift"
                else not np.isclose(current.scale[node], 1.0)
            )
            if already_active or remaining > 0:
                legal.append(kind)
        legal_by_node[node] = tuple(legal)
        tied = rank > 0 and scores[node] == scores[node_order[rank - 1]]
        trace.append(
            {
                "node": node,
                "node_local_loss": scores[node],
                "node_score_rank": rank,
                "directed_parents": sorted(graph.parents_of(node)),
                "undirected_neighbors": sorted(graph.undirected_neighbors_of(node)),
                "legal_coordinate_kinds": list(legal),
                "preferred_kind": legal[0] if legal else None,
                "exact_score_tie_with_previous": tied,
                "node_tie_break": "ascending_node_index",
                "root_scale_excluded": (
                    "log_scale" in allowed_coordinate_kinds
                    and not graph.parents_of(node)
                ),
            }
        )

    selected: list[tuple[int, ParameterKind, int, int]] = []
    for rank, node in enumerate(node_order):
        kinds = legal_by_node[node]
        if kinds:
            selected.append((node, kinds[0], rank, 1))
        if len(selected) == SHORTLIST_SIZE:
            break
    if len(selected) < SHORTLIST_SIZE:
        for rank, node in enumerate(node_order):
            for kind in legal_by_node[node][1:]:
                selected.append((node, kind, rank, 2))
                if len(selected) == SHORTLIST_SIZE:
                    break
            if len(selected) == SHORTLIST_SIZE:
                break

    entries = []
    for position, (node, kind, rank, selection_pass) in enumerate(selected):
        tied = rank > 0 and scores[node] == scores[node_order[rank - 1]]
        entries.append(
            ShortlistEntry(
                shortlist_position=position,
                coordinate=ParameterCoordinate(node, kind),
                node_local_loss=scores[node],
                node_score_rank=rank,
                selection_pass=selection_pass,
                kind_rank=KIND_ORDER.index(kind),
                exact_score_tie_with_previous=tied,
                tie_break_rule=(
                    "descending_node_local_loss_then_ascending_node_then_"
                    "shift_before_log_scale_then_distinct_node_first_pass"
                ),
            )
        )
    return DeterministicShortlist(tuple(entries), node_order, tuple(trace))
