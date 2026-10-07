"""Runtime capability declarations and non-oracle assembly validation.

This module is an API boundary, not a security sandbox.  It prevents accidental
scientific leakage through normal provider attributes and declared
capabilities; it deliberately makes no claim about adversarial Python
reflection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import networkx as nx
import numpy as np


class ProviderCapability(StrEnum):
    """Scientifically relevant capabilities exposed by a provider."""

    ORACLE = "oracle"
    EXACT_PUBLIC_MOMENTS = "exact_public_moments"
    EMPIRICAL_SAMPLES = "empirical_samples"
    ESTIMATED_STRUCTURE = "estimated_structure"
    EXACT_PUBLIC_STRUCTURE = "exact_public_structure"


@runtime_checkable
class CapabilityProvider(Protocol):
    """A component that explicitly declares its scientific capabilities."""

    @property
    def capabilities(self) -> frozenset[ProviderCapability]:
        """Return immutable provider capabilities."""
        ...


_NON_ORACLE_MODES = frozenset({"non_oracle_noiseless", "finite_sample_non_oracle"})
_LEAKING_ATTRIBUTE_NAMES = frozenset(
    {
        "b",
        "beta",
        "coefficients",
        "coefficient_matrix",
        "dag",
        "true_dag",
        "scm",
        "ground_truth",
        "generating_intervention",
        "true_ancestors",
        "true_parents",
        "true_sources",
    }
)


@dataclass(frozen=True)
class AssemblyComponent:
    """One named component in an experiment assembly."""

    role: str
    provider: object

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("component role must be a nonempty string.")


def normalized_capabilities(provider: object) -> frozenset[ProviderCapability]:
    """Validate and return a provider's declared capabilities."""
    raw = getattr(provider, "capabilities", frozenset())
    if not isinstance(raw, frozenset):
        raise TypeError("provider capabilities must be a frozenset.")
    try:
        return frozenset(ProviderCapability(value) for value in raw)
    except (TypeError, ValueError) as error:
        raise ValueError("provider declares an unsupported capability.") from error


def validate_experiment_components(
    mode: str,
    components: Sequence[AssemblyComponent],
) -> None:
    """Reject capability or ordinary-attribute leakage in non-oracle modes."""
    if not isinstance(mode, str):
        raise TypeError("mode must be a string.")
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        raise TypeError("components must be a sequence of AssemblyComponent values.")
    if any(not isinstance(component, AssemblyComponent) for component in components):
        raise TypeError("components must contain only AssemblyComponent values.")
    if mode not in _NON_ORACLE_MODES:
        return

    for component in components:
        capabilities = normalized_capabilities(component.provider)
        if ProviderCapability.ORACLE in capabilities:
            raise ValueError(
                f"non-oracle mode {mode!r} rejects oracle component {component.role!r}."
            )
        leak = _find_public_leak(component.provider)
        if leak is not None:
            raise ValueError(
                f"non-oracle component {component.role!r} exposes forbidden "
                f"scientific state at {leak}."
            )

    environment_capabilities = frozenset().union(
        *(
            normalized_capabilities(component.provider)
            for component in components
            if component.role == "environment"
        )
    )
    required = (
        ProviderCapability.EXACT_PUBLIC_MOMENTS
        if mode == "non_oracle_noiseless"
        else ProviderCapability.EMPIRICAL_SAMPLES
    )
    if required not in environment_capabilities:
        raise ValueError(
            f"mode {mode!r} requires an environment with capability {required.value!r}."
        )


def _find_public_leak(root: object) -> str | None:
    """Return the first normal public path to hidden state, if one exists."""
    seen: set[int] = set()

    def visit(value: object, path: str) -> str | None:
        if value is None or isinstance(
            value,
            (str, bytes, bool, int, float, complex, np.generic),
        ):
            return None
        identifier = id(value)
        if identifier in seen:
            return None
        seen.add(identifier)

        if isinstance(value, nx.DiGraph):
            return path
        if isinstance(value, np.ndarray):
            return None
        if isinstance(value, Mapping):
            for key, item in value.items():
                leak = visit(item, f"{path}[{key!r}]")
                if leak is not None:
                    return leak
            return None
        if isinstance(value, (tuple, list, set, frozenset)):
            for index, item in enumerate(value):
                leak = visit(item, f"{path}[{index}]")
                if leak is not None:
                    return leak
            return None

        attributes = getattr(value, "__dict__", None)
        if not isinstance(attributes, dict):
            return None
        for name, item in attributes.items():
            if name.startswith("_"):
                continue
            child_path = f"{path}.{name}"
            if name.casefold() in _LEAKING_ATTRIBUTE_NAMES:
                return child_path
            leak = visit(item, child_path)
            if leak is not None:
                return leak
        return None

    return visit(root, type(root).__name__)
