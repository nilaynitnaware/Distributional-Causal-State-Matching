"""Immutable partially directed graphs for public graph knowledge."""

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

import networkx as nx

DirectedEdge = tuple[int, int]
UndirectedEdge = tuple[int, int]


def _normalize_node(node: object, *, name: str) -> int:
    """Validate and normalize one integer node label."""
    if isinstance(node, bool) or not isinstance(node, Integral):
        raise TypeError(f"{name} must be an integer.")
    return int(node)


def _normalize_nodes(nodes: object, *, name: str) -> frozenset[int]:
    """Defensively freeze an iterable of integer node labels."""
    if isinstance(nodes, (str, bytes)) or not isinstance(nodes, Iterable):
        raise TypeError(f"{name} must be an iterable of integers.")
    return frozenset(_normalize_node(node, name=f"{name} entry") for node in nodes)


def _normalize_edge(
    edge: object,
    *,
    name: str,
) -> tuple[int, int]:
    """Validate and normalize one ordered pair of node labels."""
    if not isinstance(edge, tuple) or len(edge) != 2:
        raise TypeError(f"{name} must be a two-node tuple.")
    return (
        _normalize_node(edge[0], name=f"{name} source"),
        _normalize_node(edge[1], name=f"{name} target"),
    )


def _normalize_edges(
    edges: object,
    *,
    name: str,
    undirected: bool,
) -> frozenset[tuple[int, int]]:
    """Defensively freeze and validate an iterable of graph edges."""
    if isinstance(edges, (str, bytes)) or not isinstance(edges, Iterable):
        raise TypeError(f"{name} must be an iterable of two-node tuples.")

    normalized: set[tuple[int, int]] = set()
    for position, raw_edge in enumerate(edges):
        edge = _normalize_edge(raw_edge, name=f"{name} entry {position}")
        if undirected and edge[0] > edge[1]:
            raise ValueError(
                "undirected edge endpoints must use canonical ascending order."
            )
        if edge in normalized:
            raise ValueError(f"{name} must not contain duplicate edges.")
        normalized.add(edge)
    return frozenset(normalized)


@dataclass(frozen=True)
class PartiallyDirectedGraph:
    """A defensively frozen graph with directed and undirected adjacencies.

    Undirected edges are stored as ascending endpoint tuples. The object is an
    immutable graph snapshot: methods that refine or restrict it return a new
    validated object and never mutate the receiver.
    """

    nodes: frozenset[int]
    directed_edges: frozenset[DirectedEdge]
    undirected_edges: frozenset[UndirectedEdge]

    def __post_init__(self) -> None:
        """Validate graph invariants and defensively freeze all collections."""
        nodes = _normalize_nodes(self.nodes, name="nodes")
        directed_edges = _normalize_edges(
            self.directed_edges,
            name="directed_edges",
            undirected=False,
        )
        undirected_edges = _normalize_edges(
            self.undirected_edges,
            name="undirected_edges",
            undirected=True,
        )

        all_edges = directed_edges | undirected_edges
        if any(source == target for source, target in all_edges):
            raise ValueError("self-loops are not permitted.")
        if any(
            source not in nodes or target not in nodes for source, target in all_edges
        ):
            raise ValueError("every edge endpoint must belong to nodes.")

        directed_pairs: set[frozenset[int]] = set()
        for source, target in directed_edges:
            pair = frozenset((source, target))
            if pair in directed_pairs:
                raise ValueError("both directions of the same edge cannot be present.")
            directed_pairs.add(pair)

        undirected_pairs = {
            frozenset((node_a, node_b)) for node_a, node_b in undirected_edges
        }
        if directed_pairs & undirected_pairs:
            raise ValueError("a node pair cannot be both directed and undirected.")

        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(nodes)
        directed_graph.add_edges_from(directed_edges)
        if not nx.is_directed_acyclic_graph(directed_graph):
            raise ValueError("the directed part must be acyclic.")

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "directed_edges", directed_edges)
        object.__setattr__(self, "undirected_edges", undirected_edges)

    def has_directed_edge(self, source: int, target: int) -> bool:
        """Return whether ``source -> target`` is present."""
        normalized_source = self._require_node(source)
        normalized_target = self._require_node(target)
        return (normalized_source, normalized_target) in self.directed_edges

    def has_undirected_edge(self, node_a: int, node_b: int) -> bool:
        """Return whether the unordered adjacency ``node_a - node_b`` exists."""
        normalized_a = self._require_node(node_a)
        normalized_b = self._require_node(node_b)
        if normalized_a == normalized_b:
            return False
        edge = tuple(sorted((normalized_a, normalized_b)))
        return edge in self.undirected_edges

    def parents_of(self, node: int) -> frozenset[int]:
        """Return nodes with directed edges entering ``node``."""
        normalized = self._require_node(node)
        return frozenset(
            source for source, target in self.directed_edges if target == normalized
        )

    def children_of(self, node: int) -> frozenset[int]:
        """Return nodes reached by directed edges leaving ``node``."""
        normalized = self._require_node(node)
        return frozenset(
            target for source, target in self.directed_edges if source == normalized
        )

    def undirected_neighbors_of(self, node: int) -> frozenset[int]:
        """Return neighbors joined to ``node`` by undirected edges."""
        normalized = self._require_node(node)
        neighbors: set[int] = set()
        for node_a, node_b in self.undirected_edges:
            if node_a == normalized:
                neighbors.add(node_b)
            elif node_b == normalized:
                neighbors.add(node_a)
        return frozenset(neighbors)

    def adjacent_nodes(self, node: int) -> frozenset[int]:
        """Return all directed and undirected neighbors of ``node``."""
        normalized = self._require_node(node)
        return frozenset(
            self.parents_of(normalized)
            | self.children_of(normalized)
            | self.undirected_neighbors_of(normalized)
        )

    def orient(self, node_a: int, node_b: int) -> "PartiallyDirectedGraph":
        """Return a snapshot where ``node_a - node_b`` becomes ``node_a -> node_b``.

        Reapplying an already present direction returns an equal copy. The
        reverse direction, a missing adjacency, or an orientation that creates
        a directed cycle is rejected.
        """
        source = self._require_node(node_a)
        target = self._require_node(node_b)
        if source == target:
            raise ValueError("cannot orient a self-loop.")
        if (source, target) in self.directed_edges:
            return self.copy()
        if (target, source) in self.directed_edges:
            raise ValueError("the requested orientation conflicts with an edge.")

        undirected_edge = tuple(sorted((source, target)))
        if undirected_edge not in self.undirected_edges:
            raise ValueError("only an existing undirected edge can be oriented.")

        return PartiallyDirectedGraph(
            nodes=self.nodes,
            directed_edges=self.directed_edges | {(source, target)},
            undirected_edges=self.undirected_edges - {undirected_edge},
        )

    def copy(self) -> "PartiallyDirectedGraph":
        """Return an equal, independently constructed immutable snapshot."""
        return PartiallyDirectedGraph(
            nodes=set(self.nodes),
            directed_edges=set(self.directed_edges),
            undirected_edges=set(self.undirected_edges),
        )

    def induced_subgraph(
        self,
        nodes: Iterable[int],
    ) -> "PartiallyDirectedGraph":
        """Return the graph induced by a subset of this graph's nodes."""
        selected = _normalize_nodes(nodes, name="induced nodes")
        if not selected <= self.nodes:
            raise ValueError("induced nodes must be a subset of graph nodes.")
        return PartiallyDirectedGraph(
            nodes=selected,
            directed_edges={
                (source, target)
                for source, target in self.directed_edges
                if source in selected and target in selected
            },
            undirected_edges={
                (node_a, node_b)
                for node_a, node_b in self.undirected_edges
                if node_a in selected and node_b in selected
            },
        )

    def directed_ancestors_of(self, node: int) -> frozenset[int]:
        """Return ancestors reachable through directed edges only."""
        normalized = self._require_node(node)
        ancestors: set[int] = set()
        frontier = list(self.parents_of(normalized))
        while frontier:
            parent = frontier.pop()
            if parent in ancestors:
                continue
            ancestors.add(parent)
            frontier.extend(self.parents_of(parent) - ancestors)
        return frozenset(ancestors)

    def chain_components(self) -> tuple[frozenset[int], ...]:
        """Return undirected connected components, including isolated nodes."""
        remaining = set(self.nodes)
        components: list[frozenset[int]] = []
        while remaining:
            root = min(remaining)
            component = {root}
            frontier = [root]
            while frontier:
                node = frontier.pop()
                new_neighbors = (
                    self.undirected_neighbors_of(node) & remaining - component
                )
                component.update(new_neighbors)
                frontier.extend(new_neighbors)
            remaining.difference_update(component)
            components.append(frozenset(component))
        return tuple(sorted(components, key=lambda item: tuple(sorted(item))))

    def top_chain_components(self) -> tuple[frozenset[int], ...]:
        """Return chain components with no directed edge entering from outside."""
        components = self.chain_components()
        component_by_node = {
            node: component for component in components for node in component
        }
        non_top_components = {
            component_by_node[target]
            for source, target in self.directed_edges
            if component_by_node[source] != component_by_node[target]
        }
        return tuple(
            component for component in components if component not in non_top_components
        )

    def _require_node(self, node: object) -> int:
        """Normalize a query node and require membership in this graph."""
        normalized = _normalize_node(node, name="node")
        if normalized not in self.nodes:
            raise ValueError("node must belong to the graph.")
        return normalized
