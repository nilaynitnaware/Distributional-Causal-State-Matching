"""Validated records for independent simulation and algorithm RNG streams."""

from dataclasses import dataclass


def _validate_seed(name: str, value: object) -> None:
    """Validate one derived NumPy seed represented as a Python integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


@dataclass(frozen=True)
class AlgorithmSeeds:
    """Public independent seeds for exploration and candidate evaluation."""

    exploration_seed: int
    candidate_evaluation_seed: int

    def __post_init__(self) -> None:
        """Validate both public algorithm seeds."""
        _validate_seed(
            "exploration_seed",
            self.exploration_seed,
        )
        _validate_seed(
            "candidate_evaluation_seed",
            self.candidate_evaluation_seed,
        )


@dataclass(frozen=True)
class SimulationSeeds:
    """Oracle-only seeds for independent problem-generation components."""

    root_seed: int
    graph_generation_seed: int
    coefficient_generation_seed: int
    observational_sampling_seed: int
    target_intervention_seed: int
    target_sampling_seed: int
    noise_variance_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate every private simulation seed."""
        for name in (
            "root_seed",
            "graph_generation_seed",
            "coefficient_generation_seed",
            "observational_sampling_seed",
            "target_intervention_seed",
            "target_sampling_seed",
        ):
            _validate_seed(name, getattr(self, name))
        if self.noise_variance_seed is not None:
            _validate_seed("noise_variance_seed", self.noise_variance_seed)
