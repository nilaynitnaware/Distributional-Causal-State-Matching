"""Public graph-generator registry and dispatch."""

from collections.abc import Callable
from inspect import signature
from typing import Final, Literal

import networkx as nx
import numpy as np

from .DAG_gen import (
    generate_barabasi_albert_dag,
    generate_complete_dag,
    generate_erdos_renyi_dag,
    generate_moralized_erdos_renyi_dag,
    generate_path_dag,
    generate_randomly_oriented_tree_dag,
    generate_rooted_tree_dag,
    generate_tree_of_cliques_dag,
)

GraphType = Literal[
    "path",
    "rooted_tree",
    "randomly_oriented_tree",
    "erdos_renyi",
    "barabasi_albert",
    "complete",
    "moralized_erdos_renyi",
    "tree_of_cliques",
]
GraphGenerator = Callable[..., nx.DiGraph]

GRAPH_GENERATORS: Final[dict[GraphType, GraphGenerator]] = {
    "path": generate_path_dag,
    "rooted_tree": generate_rooted_tree_dag,
    "randomly_oriented_tree": generate_randomly_oriented_tree_dag,
    "erdos_renyi": generate_erdos_renyi_dag,
    "barabasi_albert": generate_barabasi_albert_dag,
    "complete": generate_complete_dag,
    "moralized_erdos_renyi": generate_moralized_erdos_renyi_dag,
    "tree_of_cliques": generate_tree_of_cliques_dag,
}


def generate_graph(
    graph_type: str,
    n_nodes: int,
    rng: np.random.Generator,
    **parameters: object,
) -> nx.DiGraph:
    """Dispatch to a registered DAG generator.

    ``parameters`` must contain exactly the family-specific configuration
    accepted by the selected generator. Signature binding catches missing or
    irrelevant parameters before dispatch, so unrelated configuration is never
    forwarded to a graph generator.

    Args:
        graph_type: Descriptive registered graph-family name.
        n_nodes: Number of graph nodes.
        rng: Explicit NumPy random-number generator.
        **parameters: Configuration specific to ``graph_type``.

    Returns:
        The DAG returned by the selected registered generator.

    Raises:
        TypeError: If ``graph_type`` is not a string or its parameters do not
            match the selected generator's interface.
        ValueError: If ``graph_type`` is unsupported or generator validation
            rejects a supplied value.
    """
    if not isinstance(graph_type, str):
        raise TypeError("graph_type must be a string.")

    try:
        generator = GRAPH_GENERATORS[graph_type]
    except KeyError as error:
        supported_names = ", ".join(GRAPH_GENERATORS)
        raise ValueError(
            f"Unsupported graph type {graph_type!r}. "
            f"Supported graph types: {supported_names}."
        ) from error

    try:
        signature(generator).bind(n_nodes=n_nodes, rng=rng, **parameters)
    except TypeError as error:
        raise TypeError(
            f"Invalid parameters for graph type {graph_type!r}: {error}"
        ) from error

    return generator(n_nodes=n_nodes, rng=rng, **parameters)
