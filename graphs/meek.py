"""Deterministic closure of partially directed graphs under Meek rules."""

from dataclasses import dataclass
from itertools import combinations

from .pdag import DirectedEdge, PartiallyDirectedGraph, UndirectedEdge


@dataclass(frozen=True)
class MeekRuleWitness:
    """One concrete graph pattern supporting a Meek-rule conclusion."""

    rule: str
    nodes: tuple[int, ...]


@dataclass(frozen=True)
class MeekOrientationProposal:
    """One proposed direction with every rule witness supporting it."""

    direction: DirectedEdge
    rules: tuple[str, ...]
    witnesses: tuple[MeekRuleWitness, ...]


@dataclass(frozen=True)
class MeekOrientationConflictDetail:
    """Serializable evidence for opposing Meek conclusions on one edge."""

    edge: UndirectedEdge
    proposals: tuple[MeekOrientationProposal, ...]


class MeekOrientationConflict(ValueError):
    """Raised when sound Meek patterns imply both directions of one edge."""

    def __init__(self, detail: MeekOrientationConflictDetail) -> None:
        self.detail = detail
        rendered = ", ".join(
            f"{source}->{target} by {','.join(proposal.rules)}"
            for proposal in detail.proposals
            for source, target in (proposal.direction,)
        )
        super().__init__(
            f"Meek rules imply conflicting orientations for {detail.edge}: {rendered}."
        )


def _are_adjacent(
    graph: PartiallyDirectedGraph,
    node_a: int,
    node_b: int,
) -> bool:
    """Return whether two nodes share an edge of either kind."""
    return node_b in graph.adjacent_nodes(node_a)


def _rule_one_witnesses(
    graph: PartiallyDirectedGraph,
    source: int,
    target: int,
) -> tuple[MeekRuleWitness, ...]:
    """Return whether R1 compels ``source -> target``.

    R1 pattern: ``parent -> source - target``, where ``parent`` and
    ``target`` are nonadjacent. Orienting ``target -> source`` would create a
    new unshielded collider, so ``source - target`` becomes
    ``source -> target``.
    """
    return tuple(
        MeekRuleWitness("R1", (parent, source, target))
        for parent in sorted(graph.parents_of(source))
        if not _are_adjacent(graph, parent, target)
    )


def _rule_two_witnesses(
    graph: PartiallyDirectedGraph,
    source: int,
    target: int,
) -> tuple[MeekRuleWitness, ...]:
    """Return whether R2 compels ``source -> target``.

    R2 pattern: ``source -> middle -> target`` and ``source - target``.
    The reverse direction would close a directed cycle.
    """
    return tuple(
        MeekRuleWitness("R2", (source, middle, target))
        for middle in sorted(graph.children_of(source) & graph.parents_of(target))
    )


def _rule_three_witnesses(
    graph: PartiallyDirectedGraph,
    source: int,
    target: int,
) -> tuple[MeekRuleWitness, ...]:
    """Return whether R3 compels ``source -> target``.

    R3 pattern: ``source - first -> target`` and
    ``source - second -> target``, where ``first`` and ``second`` are
    nonadjacent. Both edges incident to ``source`` are undirected.
    """
    candidate_parents = graph.undirected_neighbors_of(source) & graph.parents_of(target)
    return tuple(
        MeekRuleWitness("R3", (source, first, second, target))
        for first, second in combinations(sorted(candidate_parents), 2)
        if not _are_adjacent(graph, first, second)
    )


def _rule_four_witnesses(
    graph: PartiallyDirectedGraph,
    source: int,
    target: int,
) -> tuple[MeekRuleWitness, ...]:
    """Return whether R4 compels ``source -> target``.

    R4 pattern: ``source - first -> second -> target``,
    ``source`` is adjacent to ``second``, and ``first`` is nonadjacent to
    ``target``. The edge ``source - first`` is undirected; the adjacency
    between ``source`` and ``second`` may already be directed.
    """
    witnesses = []
    for first in sorted(graph.undirected_neighbors_of(source)):
        if _are_adjacent(graph, first, target):
            continue
        path_middles = graph.children_of(first) & graph.parents_of(target)
        witnesses.extend(
            MeekRuleWitness("R4", (source, first, second, target))
            for second in sorted(path_middles)
            if _are_adjacent(graph, source, second)
        )
    return tuple(witnesses)


def _implied_witnesses(
    graph: PartiallyDirectedGraph,
    source: int,
    target: int,
) -> tuple[MeekRuleWitness, ...]:
    """Return concrete Meek witnesses that compel one candidate orientation."""
    witness_functions = (
        _rule_one_witnesses,
        _rule_two_witnesses,
        _rule_three_witnesses,
        _rule_four_witnesses,
    )
    return tuple(
        witness
        for find_witnesses in witness_functions
        for witness in find_witnesses(graph, source, target)
    )


def _round_orientations(
    graph: PartiallyDirectedGraph,
) -> dict[UndirectedEdge, dict[DirectedEdge, tuple[MeekRuleWitness, ...]]]:
    """Collect every orientation implied by the current snapshot."""
    conclusions: dict[
        UndirectedEdge,
        dict[DirectedEdge, tuple[MeekRuleWitness, ...]],
    ] = {}
    for node_a, node_b in sorted(graph.undirected_edges):
        for source, target in ((node_a, node_b), (node_b, node_a)):
            witnesses = _implied_witnesses(graph, source, target)
            if witnesses:
                conclusions.setdefault((node_a, node_b), {})[(source, target)] = (
                    witnesses
                )
    return conclusions


def apply_meek_closure(
    graph: PartiallyDirectedGraph,
) -> PartiallyDirectedGraph:
    """Return a new graph closed to a fixed point under Meek R1--R4.

    Rules are evaluated against one immutable snapshot per round. Opposing
    conclusions for the same edge are rejected explicitly. Each accepted
    conclusion only orients an existing undirected adjacency; the
    :class:`PartiallyDirectedGraph` constructor prevents directed cycles.
    """
    if not isinstance(graph, PartiallyDirectedGraph):
        raise TypeError("graph must be a PartiallyDirectedGraph.")

    closed = graph.copy()
    while True:
        conclusions = _round_orientations(closed)
        if not conclusions:
            return closed

        orientations: list[DirectedEdge] = []
        for edge, directions in sorted(conclusions.items()):
            if len(directions) > 1:
                proposals = tuple(
                    MeekOrientationProposal(
                        direction=direction,
                        rules=tuple(
                            dict.fromkeys(witness.rule for witness in witnesses)
                        ),
                        witnesses=witnesses,
                    )
                    for direction, witnesses in sorted(directions.items())
                )
                raise MeekOrientationConflict(
                    MeekOrientationConflictDetail(edge=edge, proposals=proposals)
                )
            orientations.extend(directions)

        for source, target in sorted(orientations):
            try:
                closed = closed.orient(source, target)
            except ValueError as error:
                raise ValueError(
                    f"Meek orientation {source}->{target} is inconsistent: {error}"
                ) from error
