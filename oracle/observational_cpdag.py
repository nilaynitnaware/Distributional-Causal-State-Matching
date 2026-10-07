"""Simulation-only construction of an exact observational CPDAG."""

from itertools import combinations
from numbers import Integral

import networkx as nx

from graphs.meek import apply_meek_closure
from graphs.pdag import PartiallyDirectedGraph


def _validated_indexed_dag(true_dag: object) -> nx.DiGraph:
    """Return a frozen defensive copy of a simple, indexed DAG."""
    if not isinstance(true_dag, nx.DiGraph) or true_dag.is_multigraph():
        raise TypeError("true_dag must be a simple networkx.DiGraph.")
    if not nx.is_directed_acyclic_graph(true_dag):
        raise ValueError("true_dag must be acyclic.")

    n_nodes = true_dag.number_of_nodes()
    if any(
        isinstance(node, bool) or not isinstance(node, Integral)
        for node in true_dag.nodes
    ) or {int(node) for node in true_dag.nodes} != set(range(n_nodes)):
        raise ValueError("true_dag nodes must be labeled 0, ..., n_nodes - 1.")

    copied = nx.DiGraph()
    copied.add_nodes_from(range(n_nodes))
    copied.add_edges_from(
        (int(source), int(target)) for source, target in true_dag.edges
    )
    return nx.freeze(copied)


def _canonical_edge(node_a: int, node_b: int) -> tuple[int, int]:
    """Return one undirected edge in ascending endpoint order."""
    return (node_a, node_b) if node_a < node_b else (node_b, node_a)


def cpdag_from_true_dag_for_simulation(
    true_dag: nx.DiGraph,
) -> PartiallyDirectedGraph:
    """Construct the exact observational CPDAG of a hidden simulation DAG.

    This is an explicitly oracle-side adapter. It may inspect the true DAG to
    construct the observational Markov-equivalence class, but it returns only
    a standalone public graph snapshot:

    1. copy the true skeleton as undirected;
    2. orient every unshielded collider from the true DAG;
    3. close the resulting PDAG under Meek R1--R4.

    The hidden DAG is retained only as a local validation snapshot and is not
    stored in or attached to the returned graph.
    """
    dag = _validated_indexed_dag(true_dag)
    nodes = frozenset(int(node) for node in dag.nodes)
    skeleton = frozenset(
        _canonical_edge(int(source), int(target)) for source, target in dag.edges
    )
    cpdag = PartiallyDirectedGraph(
        nodes=nodes,
        directed_edges=frozenset(),
        undirected_edges=skeleton,
    )

    skeleton_pairs = {frozenset(edge) for edge in skeleton}
    for child in sorted(nodes):
        for parent_a, parent_b in combinations(
            sorted(int(parent) for parent in dag.predecessors(child)),
            2,
        ):
            if frozenset((parent_a, parent_b)) in skeleton_pairs:
                continue
            cpdag = cpdag.orient(parent_a, child)
            cpdag = cpdag.orient(parent_b, child)

    cpdag = apply_meek_closure(cpdag)

    result_skeleton = {
        _canonical_edge(source, target)
        for source, target in cpdag.directed_edges | cpdag.undirected_edges
    }
    if result_skeleton != set(skeleton):
        raise RuntimeError("CPDAG construction changed the true DAG skeleton.")
    if any(not dag.has_edge(source, target) for source, target in cpdag.directed_edges):
        raise RuntimeError("CPDAG construction contradicted the true DAG.")

    return cpdag
