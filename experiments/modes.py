"""Explicit experiment-mode labels and oracle-boundary validation."""

from typing import Literal

ExperimentMode = Literal[
    "known_support_oracle",
    "known_dag_oracle",
    "idealized_unknown_graph_oracle",
    "non_oracle_noiseless",
    "finite_sample_non_oracle",
    "non_oracle_noiseless_supplied_cpdag",
    "non_oracle_noiseless_estimated_cpdag",
    "finite_sample_non_oracle_supplied_cpdag",
    "finite_sample_non_oracle_estimated_cpdag",
    "exact_nonrealisable",
    "finite_sample_nonrealisable",
]

_ORACLE_MODES = frozenset(
    {
        "known_support_oracle",
        "known_dag_oracle",
        "idealized_unknown_graph_oracle",
    }
)
_ALL_MODES = _ORACLE_MODES | {
    "non_oracle_noiseless",
    "finite_sample_non_oracle",
    "non_oracle_noiseless_supplied_cpdag",
    "non_oracle_noiseless_estimated_cpdag",
    "finite_sample_non_oracle_supplied_cpdag",
    "finite_sample_non_oracle_estimated_cpdag",
    "exact_nonrealisable",
    "finite_sample_nonrealisable",
}


def validate_experiment_assembly(
    mode: ExperimentMode,
    *,
    uses_true_dag: bool,
    uses_true_coefficients: bool,
    uses_oracle_loss: bool,
) -> None:
    """Reject hidden-truth providers in every nominally non-oracle mode."""
    if mode not in _ALL_MODES:
        raise ValueError("unsupported experiment mode.")
    flags = {
        "uses_true_dag": uses_true_dag,
        "uses_true_coefficients": uses_true_coefficients,
        "uses_oracle_loss": uses_oracle_loss,
    }
    if any(not isinstance(value, bool) for value in flags.values()):
        raise TypeError("experiment assembly flags must be booleans.")
    if mode not in _ORACLE_MODES and any(flags.values()):
        enabled = ", ".join(name for name, value in flags.items() if value)
        raise ValueError(f"non-oracle experiment mode {mode!r} cannot use: {enabled}.")
