"""Oracle source outcomes permitted only in explicitly labelled diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.scm import LinearGaussianSCM
from matching.capabilities import ProviderCapability
from matching.source_inference import (
    ExactSourceClassification,
    SourceTolerance,
    classify_exact_source_targets,
)
from matching.source_providers import FiniteSourceInferenceProvider


@dataclass
class OracleDiagnosticSourceInferenceProvider:
    """Use hidden exact moments without exposing a public SCM attribute."""

    hidden_scm: LinearGaussianSCM
    tolerance: SourceTolerance
    capabilities = frozenset({ProviderCapability.ORACLE})

    def __post_init__(self) -> None:
        if not isinstance(self.hidden_scm, LinearGaussianSCM):
            raise TypeError("hidden_scm must be LinearGaussianSCM.")
        self.__moments = self.hidden_scm.interventional_moments
        del self.hidden_scm

    def classify(
        self,
        baseline_samples: np.ndarray,
        probe_samples: np.ndarray,
        background: ShiftScaleIntervention,
        shift_increment: np.ndarray,
        targets: frozenset[int],
        *,
        background_id: str,
        paired_samples: bool,
    ) -> tuple[ExactSourceClassification, ...]:
        del baseline_samples, probe_samples, background_id, paired_samples
        probe = ShiftScaleIntervention(
            background.shift + shift_increment,
            background.scale,
        )
        baseline_mean, _ = self.__moments(background)
        probe_mean, _ = self.__moments(probe)
        return classify_exact_source_targets(
            baseline_mean,
            probe_mean,
            shift_increment,
            targets,
            self.tolerance,
        )


@dataclass(frozen=True)
class OracleLabelOverrideClassification:
    """An empirical decision record with only its final label overridden."""

    node: int
    classification: str
    empirical_classification: str
    oracle_classification: str
    empirical_oracle_agree: bool
    empirical_evidence: dict[str, object]
    hidden_fact_used: str
    hidden_targeted_ancestors: tuple[int, ...]
    reason: str


@dataclass
class OracleLabelOverrideSourceInferenceProvider:
    """Draw empirical evidence, then replace only the applied source label."""

    hidden_dag: nx.DiGraph
    empirical_provider: FiniteSourceInferenceProvider
    capabilities = frozenset({ProviderCapability.ORACLE})

    def __post_init__(self) -> None:
        if not isinstance(self.hidden_dag, nx.DiGraph):
            raise TypeError("hidden_dag must be a networkx.DiGraph.")
        if not nx.is_directed_acyclic_graph(self.hidden_dag):
            raise ValueError("hidden_dag must be acyclic.")
        self.__dag = nx.freeze(nx.DiGraph(self.hidden_dag))
        del self.hidden_dag

    def classify(
        self,
        baseline_samples: np.ndarray,
        probe_samples: np.ndarray,
        background: ShiftScaleIntervention,
        shift_increment: np.ndarray,
        targets: frozenset[int],
        *,
        background_id: str,
        paired_samples: bool,
    ) -> tuple[OracleLabelOverrideClassification, ...]:
        empirical = self.empirical_provider.classify(
            baseline_samples,
            probe_samples,
            background,
            shift_increment,
            targets,
            background_id=background_id,
            paired_samples=paired_samples,
        )
        rows = []
        for item in empirical:
            targeted_ancestors = tuple(
                sorted(set(nx.ancestors(self.__dag, item.node)) & set(targets))
            )
            oracle = "source" if not targeted_ancestors else "non_source"
            rows.append(
                OracleLabelOverrideClassification(
                    node=item.node,
                    classification=oracle,
                    empirical_classification=item.classification,
                    oracle_classification=oracle,
                    empirical_oracle_agree=item.classification == oracle,
                    empirical_evidence=asdict(item),
                    hidden_fact_used=(
                        "no_targeted_ancestor"
                        if not targeted_ancestors
                        else "targeted_ancestor_present"
                    ),
                    hidden_targeted_ancestors=targeted_ancestors,
                    reason="diagnostic_hidden_dag_label_override_only",
                )
            )
        return tuple(rows)
