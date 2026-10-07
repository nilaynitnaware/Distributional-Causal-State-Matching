# ruff: noqa: N999
"""Reproducible directed-acyclic-graph generators."""

from collections.abc import Iterable
from itertools import combinations, pairwise

import networkx as nx
import numpy as np

DEFAULT_ERDOS_RENYI_EDGE_PROBABILITY = 0.8


def generate_path_dag(
    n_nodes: int,
    rng: np.random.Generator,
    *,
    permute: bool = False,
) -> nx.DiGraph:
    """Generate a directed path on nodes ``0, ..., n_nodes - 1``.

    By default, the path follows the deterministic order
    ``0 -> 1 -> ... -> n_nodes - 1``. When ``permute`` is true, the supplied
    random-number generator samples a node order and the path follows that
    order. In either case, every requested node is present in the returned
    graph.

    Args:
        n_nodes: Positive number of nodes in the graph.
        rng: Explicit NumPy random-number generator.
        permute: Whether to sample the path order with ``rng``.

    Returns:
        A NetworkX directed acyclic graph containing one directed path.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If ``n_nodes`` is not positive.
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int):
        raise TypeError("n_nodes must be an integer.")
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    if not isinstance(permute, bool):
        raise TypeError("permute must be a boolean.")

    node_order = rng.permutation(n_nodes).tolist() if permute else list(range(n_nodes))

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))
    dag.add_edges_from(pairwise(node_order))
    return dag


def generate_rooted_tree_dag(
    n_nodes: int,
    rng: np.random.Generator,
    *,
    root: int | None = None,
) -> nx.DiGraph:
    """Generate a random labeled tree with every edge directed away from a root.

    An undirected tree is sampled uniformly with NetworkX's public
    ``random_labeled_tree`` API. If ``root`` is omitted, the supplied random
    number generator also samples the root. A breadth-first traversal then
    orients each edge from parent to child, so every non-root node has exactly
    one parent and the resulting DAG has no unshielded colliders.

    Args:
        n_nodes: Positive number of nodes in the graph.
        rng: Explicit NumPy random-number generator.
        root: Optional root node. When omitted, a root is sampled with ``rng``.

    Returns:
        A rooted NetworkX directed tree on nodes ``0, ..., n_nodes - 1``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If ``n_nodes`` is not positive or ``root`` is out of range.
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int):
        raise TypeError("n_nodes must be an integer.")
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    if root is not None:
        if isinstance(root, bool) or not isinstance(root, int):
            raise TypeError("root must be an integer or None.")
        if not 0 <= root < n_nodes:
            raise ValueError(
                f"root must be between 0 and {n_nodes - 1}, received {root}."
            )

    undirected_tree = nx.random_labeled_tree(n_nodes, seed=rng)
    root_node = int(rng.integers(n_nodes)) if root is None else root

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))
    dag.add_edges_from(nx.bfs_edges(undirected_tree, source=root_node))
    return dag


def generate_randomly_oriented_tree_dag(
    n_nodes: int,
    rng: np.random.Generator,
) -> nx.DiGraph:
    """Generate a random labeled tree with independently oriented edges.

    NetworkX first samples a uniform undirected labeled tree. Each skeleton
    edge is then oriented by an independent fair Bernoulli draw from ``rng``.
    Because the undirected skeleton is a tree, no orientation can contain a
    directed cycle, although nodes may have multiple parents and form
    unshielded colliders.

    Args:
        n_nodes: Positive number of nodes in the graph.
        rng: Explicit NumPy random-number generator.

    Returns:
        A NetworkX DAG whose undirected skeleton is a labeled tree.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If ``n_nodes`` is not positive.
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int):
        raise TypeError("n_nodes must be an integer.")
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    undirected_tree = nx.random_labeled_tree(n_nodes, seed=rng)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))

    for first_node, second_node in undirected_tree.edges:
        if rng.integers(2) == 0:
            dag.add_edge(first_node, second_node)
        else:
            dag.add_edge(second_node, first_node)

    return dag


def generate_erdos_renyi_dag(
    n_nodes: int,
    edge_probability: float,
    rng: np.random.Generator,
) -> nx.DiGraph:
    """Generate an Erdős–Rényi DAG using a sampled total node order.

    A random permutation defines the topological order. Each of the
    ``n_nodes * (n_nodes - 1) / 2`` possible order-consistent edges is then
    independently included with probability ``edge_probability`` and directed
    from the earlier node to the later node. Thus, ``edge_probability`` is the
    expected edge density relative to a complete DAG.

    Args:
        n_nodes: Positive number of nodes in the graph.
        edge_probability: Edge-inclusion probability between zero and one,
            inclusive.
        rng: Explicit NumPy random-number generator.

    Returns:
        A NetworkX DAG on nodes ``0, ..., n_nodes - 1``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If ``n_nodes`` or ``edge_probability`` is out of range.
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int):
        raise TypeError("n_nodes must be an integer.")
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    if isinstance(edge_probability, bool) or not isinstance(
        edge_probability, (int, float, np.integer, np.floating)
    ):
        raise TypeError("edge_probability must be a real number.")

    probability = float(edge_probability)
    if not np.isfinite(probability):
        raise ValueError("edge_probability must be finite.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("edge_probability must be between 0 and 1.")

    node_order = rng.permutation(n_nodes)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))

    for earlier_position in range(n_nodes):
        for later_position in range(earlier_position + 1, n_nodes):
            if rng.random() < probability:
                parent = int(node_order[earlier_position])
                child = int(node_order[later_position])
                dag.add_edge(parent, child)

    return dag


def generate_barabasi_albert_dag(
    n_nodes: int,
    attachment_edges: int,
    rng: np.random.Generator,
) -> nx.DiGraph:
    """Generate a Barabási–Albert skeleton oriented by a total node order.

    NetworkX samples the undirected preferential-attachment skeleton using
    its default initial star. This yields
    ``attachment_edges * (n_nodes - attachment_edges)`` skeleton edges. A
    random permutation then defines a total order, and every skeleton edge is
    directed from its earlier endpoint to its later endpoint in that order.

    Args:
        n_nodes: Positive number of nodes in the graph.
        attachment_edges: Number of existing nodes each new node attaches to.
        rng: Explicit NumPy random-number generator.

    Returns:
        A NetworkX DAG with a Barabási–Albert undirected skeleton.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If ``n_nodes`` or ``attachment_edges`` is out of range.
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int):
        raise TypeError("n_nodes must be an integer.")
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    if isinstance(attachment_edges, bool) or not isinstance(attachment_edges, int):
        raise TypeError("attachment_edges must be an integer.")
    if not 1 <= attachment_edges < n_nodes:
        raise ValueError(
            "attachment_edges must satisfy 1 <= attachment_edges < n_nodes."
        )

    undirected_graph = nx.barabasi_albert_graph(n_nodes, attachment_edges, seed=rng)
    node_order = rng.permutation(n_nodes).tolist()
    positions = {node: position for position, node in enumerate(node_order)}

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))
    for first_node, second_node in undirected_graph.edges:
        if positions[first_node] < positions[second_node]:
            dag.add_edge(first_node, second_node)
        else:
            dag.add_edge(second_node, first_node)

    return dag


def generate_complete_dag(
    n_nodes: int,
    rng: np.random.Generator,
    *,
    node_order: Iterable[int] | None = None,
) -> nx.DiGraph:
    """Generate a complete DAG oriented according to a total node order.

    If ``node_order`` is omitted, the supplied random-number generator samples
    a permutation of ``0, ..., n_nodes - 1``. If an order is supplied, it must
    be a permutation of those labels and ``rng`` is not consumed. Every node
    pair is directed from its earlier endpoint to its later endpoint.

    Args:
        n_nodes: Positive number of nodes in the graph.
        rng: Explicit NumPy random-number generator.
        node_order: Optional explicit permutation of all node labels.

    Returns:
        A NetworkX complete DAG with ``n_nodes * (n_nodes - 1) / 2`` edges.

    Raises:
        TypeError: If an argument or order element has an invalid type.
        ValueError: If ``n_nodes`` is invalid or the order is not a permutation.
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int):
        raise TypeError("n_nodes must be an integer.")
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    if node_order is None:
        order = rng.permutation(n_nodes).tolist()
    else:
        try:
            order_values = list(node_order)
        except TypeError as error:
            raise TypeError("node_order must be an iterable of integers.") from error

        if any(
            isinstance(node, bool) or not isinstance(node, (int, np.integer))
            for node in order_values
        ):
            raise TypeError("every node_order element must be an integer.")
        order = [int(node) for node in order_values]
        if len(order) != n_nodes or set(order) != set(range(n_nodes)):
            raise ValueError("node_order must be a permutation of 0, ..., n_nodes - 1.")

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))
    dag.add_edges_from(combinations(order, 2))
    return dag


def generate_moralized_erdos_renyi_dag(
    n_nodes: int,
    edge_probability: float,
    rng: np.random.Generator,
) -> nx.DiGraph:
    """Generate a connected chordal DAG with no unshielded colliders.

    Nodes are processed in construction positions ``0, ..., n_nodes - 1``.
    At position ``i > 0``, the parent count is sampled as
    ``max(1, Binomial(i, edge_probability))``. That many parents are selected
    uniformly without replacement from earlier positions. The minimum of one
    parent connects every nonfirst node to the existing graph.

    Final parent sets are made cliques by processing children in reverse and
    adding earlier-to-later edges between every pair of their parents. Reverse
    processing ensures that newly introduced parents belong only to nodes that
    have not yet undergone clique closure.

    ``edge_probability`` controls the initial parent-selection probability,
    not final edge density. Moralization may add edges and densify the graph.
    Finally, an RNG-sampled permutation relabels the construction positions.

    Args:
        n_nodes: Positive number of nodes in the graph.
        edge_probability: Initial parent-selection probability in ``[0, 1]``.
        rng: Explicit NumPy random-number generator.

    Returns:
        A connected NetworkX chordal DAG on nodes ``0, ..., n_nodes - 1``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If ``n_nodes`` or ``edge_probability`` is out of range.
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int):
        raise TypeError("n_nodes must be an integer.")
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    if isinstance(edge_probability, bool) or not isinstance(
        edge_probability, (int, float, np.integer, np.floating)
    ):
        raise TypeError("edge_probability must be a real number.")

    probability = float(edge_probability)
    if not np.isfinite(probability):
        raise ValueError("edge_probability must be finite.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("edge_probability must be between 0 and 1.")

    construction_dag = nx.DiGraph()
    construction_dag.add_nodes_from(range(n_nodes))

    for child_position in range(1, n_nodes):
        n_parents = max(1, int(rng.binomial(child_position, probability)))
        parents = rng.choice(child_position, size=n_parents, replace=False)
        construction_dag.add_edges_from(
            (int(parent), child_position) for parent in parents
        )

    for child_position in reversed(range(1, n_nodes)):
        parents = sorted(construction_dag.predecessors(child_position))
        for earlier_parent, later_parent in combinations(parents, 2):
            construction_dag.add_edge(earlier_parent, later_parent)

    permutation = rng.permutation(n_nodes)
    mapping = {position: int(label) for position, label in enumerate(permutation)}
    return nx.relabel_nodes(construction_dag, mapping, copy=True)


def generate_tree_of_cliques_dag(
    n_nodes: int,
    branching_degree: int,
    min_clique_size: int,
    max_clique_size: int,
    rng: np.random.Generator,
) -> nx.DiGraph:
    """Generate a chordal DAG from a bounded-degree tree of overlapping cliques.

    A root clique is sampled first. Each child clique intersects its parent in
    a nonempty proper clique separator and introduces at least one new node.
    Child counts are bounded by ``branching_degree``, clique sizes lie in the
    inclusive interval ``[min_clique_size, max_clique_size]``, and the last
    child is sized so that the graph contains exactly ``n_nodes`` nodes.

    Construction labels record when nodes are introduced. Reverse construction
    order is a perfect-elimination order: after later-created descendants have
    been eliminated, every introduced node's remaining neighbors lie in its
    construction clique. Therefore, directing each edge from its
    earlier-created endpoint to its later-created endpoint produces a DAG whose
    parent sets are cliques and hence has no unshielded colliders. An
    RNG-sampled permutation finally removes the construction labels.

    Args:
        n_nodes: Number of nodes in the graph.
        branching_degree: Maximum number of child cliques per clique.
        min_clique_size: Inclusive minimum maximal-clique size.
        max_clique_size: Inclusive maximum maximal-clique size.
        rng: Explicit NumPy random-number generator.

    Returns:
        A connected NetworkX chordal DAG on nodes ``0, ..., n_nodes - 1``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If a count or clique-size bound is out of range.
    """
    integer_arguments = {
        "n_nodes": n_nodes,
        "branching_degree": branching_degree,
        "min_clique_size": min_clique_size,
        "max_clique_size": max_clique_size,
    }
    for name, value in integer_arguments.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")

    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")
    if branching_degree < 1:
        raise ValueError("branching_degree must be positive.")
    if min_clique_size < 2:
        raise ValueError("min_clique_size must be at least 2.")
    if min_clique_size > max_clique_size:
        raise ValueError("min_clique_size must not exceed max_clique_size.")
    if max_clique_size > n_nodes:
        raise ValueError("max_clique_size must not exceed n_nodes.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    root_size = int(rng.integers(min_clique_size, max_clique_size + 1))
    root_clique = tuple(range(root_size))

    skeleton = nx.Graph()
    skeleton.add_nodes_from(root_clique)
    skeleton.add_edges_from(combinations(root_clique, 2))

    cliques = [root_clique]
    parent_index = 0
    next_node = root_size

    while next_node < n_nodes:
        parent_clique = cliques[parent_index]
        parent_index += 1

        for _ in range(branching_degree):
            remaining_nodes = n_nodes - next_node
            if remaining_nodes == 0:
                break

            max_new_nodes = min(remaining_nodes, max_clique_size - 1)
            n_new_nodes = int(rng.integers(1, max_new_nodes + 1))

            min_overlap = max(1, min_clique_size - n_new_nodes)
            max_overlap = min(
                len(parent_clique) - 1,
                max_clique_size - n_new_nodes,
            )
            overlap_size = int(rng.integers(min_overlap, max_overlap + 1))
            overlap = sorted(
                int(node)
                for node in rng.choice(
                    parent_clique,
                    size=overlap_size,
                    replace=False,
                )
            )

            new_nodes = tuple(range(next_node, next_node + n_new_nodes))
            child_clique = (*overlap, *new_nodes)
            skeleton.add_nodes_from(new_nodes)
            skeleton.add_edges_from(combinations(child_clique, 2))
            cliques.append(child_clique)
            next_node += n_new_nodes

    construction_dag = nx.DiGraph()
    construction_dag.add_nodes_from(range(n_nodes))
    construction_dag.add_edges_from(
        (min(first_node, second_node), max(first_node, second_node))
        for first_node, second_node in skeleton.edges
    )

    permutation = rng.permutation(n_nodes)
    mapping = {position: int(label) for position, label in enumerate(permutation)}
    return nx.relabel_nodes(construction_dag, mapping, copy=True)
