"""Simulation-only exact observational CPDAG provider."""

from dataclasses import dataclass

import networkx as nx

from matching.capabilities import ProviderCapability
from matching.structure_learning import (
    StructureEstimate,
    StructureEstimationMetadata,
)

from .observational_cpdag import cpdag_from_true_dag_for_simulation


@dataclass(frozen=True)
class OracleExactCPDAGProvider:
    """Retain the true DAG only inside an explicitly oracle-side provider."""

    true_dag: nx.DiGraph
    capabilities = frozenset({ProviderCapability.ORACLE})

    def estimate(self) -> StructureEstimate:
        """Construct an exact CPDAG and return only its standalone snapshot."""
        graph = cpdag_from_true_dag_for_simulation(self.true_dag)
        return StructureEstimate(
            graph,
            StructureEstimationMetadata(
                provider="exact_oracle_cpdag",
                tests_performed=0,
                separating_sets=(),
                unresolved_tests=0,
                failed_tests=(),
                regularization=None,
                sample_size=0,
                deterministic_variable_order=tuple(sorted(graph.nodes)),
            ),
        )
