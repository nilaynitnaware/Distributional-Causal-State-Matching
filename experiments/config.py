from dataclasses import dataclass
from typing import Literal

from graphs.DAG_gen import DEFAULT_ERDOS_RENYI_EDGE_PROBABILITY
from graphs.registry import GraphType
from graphs.support import DEFAULT_SUPPORT_ATOL


@dataclass(frozen=True)
class ExperimentConfig:
    # Graph generation
    graph_type: GraphType = "erdos_renyi"
    n_nodes: int = 10

    # Generic graph parameters
    edge_probability: float = DEFAULT_ERDOS_RENYI_EDGE_PROBABILITY
    expected_degree: float | None = None

    # Barabási–Albert parameter
    attachment_edges: int = 2

    # Tree-of-cliques parameters
    clique_tree_degree: int = 3
    min_clique_size: int = 3
    max_clique_size: int = 5

    # SCM coefficients
    coefficient_min_abs: float = 0.25
    coefficient_max_abs: float = 1.0

    # Target state
    target_type: Literal["realisable", "non_realisable"] = "realisable"
    target_support_size: int = 3

    # Intervention restrictions
    max_probe_targets: int = 1
    max_parameter_coordinates: int = 6
    scale_min: float = 0.25
    scale_max: float = 3.0
    support_atol: float = DEFAULT_SUPPORT_ATOL

    # Sampling
    n_observational_samples: int = 10_000
    n_interventional_samples: int = 10_000

    # Algorithm
    max_experiment_trials: int = 100
    kl_tolerance: float = 1e-4

    # Reproducibility
    random_seed: int = 42
