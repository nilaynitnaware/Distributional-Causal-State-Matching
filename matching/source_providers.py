"""Injectable finite-source providers for production and labelled diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from graphs.interventions import ShiftScaleIntervention

from .capabilities import ProviderCapability
from .confidence_calibrated_source import (
    ConfidenceCalibratedSourceSettings,
    classify_confidence_calibrated_source_targets,
)
from .source_inference import (
    ExactSourceClassification,
    FiniteSourceClassification,
    FiniteSourceInferenceSettings,
    classify_finite_source_targets,
)

SourceProviderResult = FiniteSourceClassification | ExactSourceClassification


@runtime_checkable
class FiniteSourceInferenceProvider(Protocol):
    capabilities: frozenset[ProviderCapability]

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
    ) -> tuple[SourceProviderResult, ...]: ...


@dataclass(frozen=True)
class StatisticalFiniteSourceInferenceProvider:
    settings: FiniteSourceInferenceSettings
    capabilities = frozenset({ProviderCapability.EMPIRICAL_SAMPLES})

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
    ) -> tuple[SourceProviderResult, ...]:
        del background
        return classify_finite_source_targets(
            baseline_samples,
            probe_samples,
            shift_increment,
            targets,
            background_id=background_id,
            settings=self.settings,
            paired_samples=paired_samples,
        )


@dataclass(frozen=True)
class ConfidenceCalibratedSourceInferenceProvider:
    """Empirical CI-equivalence provider with no hidden-label capability."""

    settings: ConfidenceCalibratedSourceSettings
    capabilities = frozenset({ProviderCapability.EMPIRICAL_SAMPLES})

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
    ) -> tuple[object, ...]:
        del background
        return classify_confidence_calibrated_source_targets(
            baseline_samples,
            probe_samples,
            shift_increment,
            targets,
            background_id=background_id,
            settings=self.settings,
            paired_samples=paired_samples,
        )
