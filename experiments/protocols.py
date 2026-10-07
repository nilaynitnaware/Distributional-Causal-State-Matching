"""Versioned, fully resolved experiment protocols.

Defaults are reproducible initial engineering/calibration choices, not claims of
scientific optimality.  Any regime change may require a new protocol version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, log
from numbers import Real
from typing import Literal

ProtocolPurpose = Literal["exact", "finite_calibration", "finite_main"]


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _probability(name: str, value: object) -> float:
    normalized = _finite(name, value)
    if not 0.0 < normalized < 1.0:
        raise ValueError(f"{name} must lie in (0, 1).")
    return normalized


@dataclass(frozen=True)
class CovarianceProtocol:
    estimator: str
    diagonal_loading: float
    minimum_eigenvalue: float

    def __post_init__(self) -> None:
        if self.estimator != "oas":
            raise ValueError("only the reproducible NumPy OAS estimator is supported.")
        if _finite("diagonal_loading", self.diagonal_loading) < 0.0:
            raise ValueError("diagonal_loading must be nonnegative.")
        if _finite("minimum_eigenvalue", self.minimum_eigenvalue) <= 0.0:
            raise ValueError("minimum_eigenvalue must be positive.")


@dataclass(frozen=True)
class ObservationalStructureProtocol:
    observational_sample_size: int
    pc_significance_level: float
    maximum_conditioning_set_size: int
    covariance: CovarianceProtocol

    def __post_init__(self) -> None:
        _positive_int("observational_sample_size", self.observational_sample_size)
        _probability("pc_significance_level", self.pc_significance_level)
        _nonnegative_int(
            "maximum_conditioning_set_size", self.maximum_conditioning_set_size
        )
        if not isinstance(self.covariance, CovarianceProtocol):
            raise TypeError("covariance must be CovarianceProtocol.")


@dataclass(frozen=True)
class ProbeInferenceProtocol:
    initial_probe_sample_size: int
    maximum_probe_sample_size: int
    sample_growth_factor: float
    maximum_repeats: int
    probe_amplitude_schedule: tuple[float, ...]
    source_equivalence_absolute: float
    source_equivalence_relative: float
    confidence_level: float
    multiplicity_correction: Literal["holm", "none"]

    def __post_init__(self) -> None:
        _positive_int("initial_probe_sample_size", self.initial_probe_sample_size)
        _positive_int("maximum_probe_sample_size", self.maximum_probe_sample_size)
        if self.initial_probe_sample_size > self.maximum_probe_sample_size:
            raise ValueError("initial probe size exceeds its maximum.")
        if _finite("sample_growth_factor", self.sample_growth_factor) <= 1.0:
            raise ValueError("sample_growth_factor must exceed one.")
        _positive_int("maximum_repeats", self.maximum_repeats)
        if not self.probe_amplitude_schedule or any(
            _finite("probe amplitude", value) == 0.0
            for value in self.probe_amplitude_schedule
        ):
            raise ValueError("probe amplitude schedule must contain nonzero values.")
        if (
            _finite("source_equivalence_absolute", self.source_equivalence_absolute)
            < 0.0
        ):
            raise ValueError("source_equivalence_absolute must be nonnegative.")
        if (
            _finite("source_equivalence_relative", self.source_equivalence_relative)
            < 0.0
        ):
            raise ValueError("source_equivalence_relative must be nonnegative.")
        _probability("probe confidence_level", self.confidence_level)
        if self.multiplicity_correction not in {"holm", "none"}:
            raise ValueError("unsupported multiplicity correction.")


@dataclass(frozen=True)
class CoordinateSearchProtocol:
    shift_domain: tuple[float, float]
    log_scale_domain: tuple[float, float]
    coarse_grid_size: int
    maximum_candidate_values_per_coordinate: int
    initial_candidate_sample_size: int | None
    maximum_candidate_sample_size: int | None
    sequential_allocation_rule: str

    def __post_init__(self) -> None:
        shift = tuple(_finite("shift bound", value) for value in self.shift_domain)
        scale = tuple(
            _finite("log-scale bound", value) for value in self.log_scale_domain
        )
        if len(shift) != 2 or shift[0] >= shift[1]:
            raise ValueError("shift_domain must be an increasing pair.")
        if len(scale) != 2 or scale[0] >= scale[1]:
            raise ValueError("log_scale_domain must be an increasing pair.")
        _positive_int("coarse_grid_size", self.coarse_grid_size)
        _positive_int(
            "maximum_candidate_values_per_coordinate",
            self.maximum_candidate_values_per_coordinate,
        )
        if self.maximum_candidate_values_per_coordinate < self.coarse_grid_size:
            raise ValueError("candidate-value cap must cover the coarse grid.")
        if (self.initial_candidate_sample_size is None) != (
            self.maximum_candidate_sample_size is None
        ):
            raise ValueError("finite candidate sample sizes must be set together.")
        if self.initial_candidate_sample_size is not None:
            _positive_int(
                "initial_candidate_sample_size", self.initial_candidate_sample_size
            )
            _positive_int(
                "maximum_candidate_sample_size", self.maximum_candidate_sample_size
            )
            if self.initial_candidate_sample_size > self.maximum_candidate_sample_size:
                raise ValueError("initial candidate size exceeds its maximum.")
        if not self.sequential_allocation_rule:
            raise ValueError("sequential_allocation_rule must be named.")


@dataclass(frozen=True)
class CandidateAcceptanceProtocol:
    minimum_improvement_tolerance: float
    confidence_level: float
    bootstrap_replicates: int
    maximum_repeated_candidate_evaluations: int

    def __post_init__(self) -> None:
        if (
            _finite("minimum_improvement_tolerance", self.minimum_improvement_tolerance)
            < 0.0
        ):
            raise ValueError("minimum improvement must be nonnegative.")
        _probability("candidate confidence_level", self.confidence_level)
        _positive_int("bootstrap_replicates", self.bootstrap_replicates)
        _positive_int(
            "maximum_repeated_candidate_evaluations",
            self.maximum_repeated_candidate_evaluations,
        )


@dataclass(frozen=True)
class UnresolvedStateProtocol:
    local_loss_tolerance: float
    local_confidence_level: float
    uncertain_state_recovery_policy: str
    global_kl_success_tolerance: float
    global_confidence_level: float

    def __post_init__(self) -> None:
        if _finite("local_loss_tolerance", self.local_loss_tolerance) < 0.0:
            raise ValueError("local_loss_tolerance must be nonnegative.")
        _probability("local confidence_level", self.local_confidence_level)
        if not self.uncertain_state_recovery_policy:
            raise ValueError("uncertain-state recovery policy must be named.")
        if (
            _finite("global_kl_success_tolerance", self.global_kl_success_tolerance)
            < 0.0
        ):
            raise ValueError("global KL tolerance must be nonnegative.")
        _probability("global confidence_level", self.global_confidence_level)


@dataclass(frozen=True)
class BudgetProtocol:
    total_executed_query_budget: int | None
    total_observation_budget: int
    baseline_query_budget: int | None
    probe_query_budget: int | None
    candidate_query_budget: int | None

    def __post_init__(self) -> None:
        if self.total_executed_query_budget is not None:
            _positive_int(
                "total_executed_query_budget",
                self.total_executed_query_budget,
            )
        _nonnegative_int("total_observation_budget", self.total_observation_budget)
        for name in (
            "baseline_query_budget",
            "probe_query_budget",
            "candidate_query_budget",
        ):
            value = getattr(self, name)
            if value is not None:
                _nonnegative_int(name, value)
        categorized = [
            value
            for value in (
                self.baseline_query_budget,
                self.probe_query_budget,
                self.candidate_query_budget,
            )
            if value is not None
        ]
        if (
            self.total_executed_query_budget is not None
            and sum(categorized) > self.total_executed_query_budget
        ):
            raise ValueError("categorized query budgets exceed the total budget.")


@dataclass(frozen=True)
class ExperimentProtocol:
    protocol_version: str
    purpose: ProtocolPurpose
    observational_structure: ObservationalStructureProtocol | None
    probe_inference: ProbeInferenceProtocol | None
    coordinate_search: CoordinateSearchProtocol
    candidate_acceptance: CandidateAcceptanceProtocol | None
    unresolved_state: UnresolvedStateProtocol
    budgets: BudgetProtocol

    def __post_init__(self) -> None:
        if not self.protocol_version:
            raise ValueError("protocol_version must be nonempty.")
        if self.purpose not in {"exact", "finite_calibration", "finite_main"}:
            raise ValueError("unsupported protocol purpose.")
        finite = self.purpose != "exact"
        if finite and any(
            component is None
            for component in (
                self.observational_structure,
                self.probe_inference,
                self.candidate_acceptance,
            )
        ):
            raise ValueError("finite protocols require all statistical components.")
        if not finite and (
            self.probe_inference is not None or self.candidate_acceptance is not None
        ):
            raise ValueError(
                "exact protocols cannot contain finite decision components."
            )
        if not isinstance(self.coordinate_search, CoordinateSearchProtocol):
            raise TypeError("coordinate_search has the wrong type.")
        if not isinstance(self.unresolved_state, UnresolvedStateProtocol):
            raise TypeError("unresolved_state has the wrong type.")
        if not isinstance(self.budgets, BudgetProtocol):
            raise TypeError("budgets has the wrong type.")
        if finite and self.budgets.total_observation_budget <= 0:
            raise ValueError("finite protocols require a positive observation budget.")

    def resolved_values(self) -> dict[str, object]:
        """Return a complete JSON-ready configuration tree."""
        return asdict(self)


_COVARIANCE_V1 = CovarianceProtocol("oas", 1e-8, 1e-8)
_FINITE_STRUCTURE_V1 = ObservationalStructureProtocol(500, 0.05, 2, _COVARIANCE_V1)
_FINITE_PROBE_V1 = ProbeInferenceProtocol(
    100,
    800,
    2.0,
    3,
    (0.25, 0.5, 1.0),
    0.1,
    0.05,
    0.95,
    "holm",
)
_FINITE_COORDINATE_V1 = CoordinateSearchProtocol(
    (-2.0, 2.0),
    (log(0.25), log(3.0)),
    5,
    5,
    100,
    800,
    "confidence_successive_elimination",
)
_FINITE_ACCEPTANCE_V1 = CandidateAcceptanceProtocol(1e-3, 0.95, 500, 3)
_FINITE_UNRESOLVED_V1 = UnresolvedStateProtocol(
    1e-3,
    0.95,
    "repeat_baseline_then_public_exploration",
    1e-3,
    0.95,
)

exact_protocol_v1 = ExperimentProtocol(
    "exact_protocol_v1",
    "exact",
    _FINITE_STRUCTURE_V1,
    None,
    CoordinateSearchProtocol(
        (-2.0, 2.0),
        (log(0.25), log(3.0)),
        9,
        30,
        None,
        None,
        "bounded_scalar_refinement",
    ),
    None,
    UnresolvedStateProtocol(1e-8, 0.95, "public_probe_then_terminate", 1e-8, 0.95),
    BudgetProtocol(200, 0, 20, 80, 100),
)

exact_noiseless_protocol_v2 = ExperimentProtocol(
    "exact_noiseless_protocol_v2",
    "exact",
    _FINITE_STRUCTURE_V1,
    None,
    CoordinateSearchProtocol(
        (-2.0, 2.0),
        (log(0.25), log(3.0)),
        9,
        30,
        None,
        None,
        "bounded_scalar_refinement",
    ),
    None,
    UnresolvedStateProtocol(1e-8, 0.95, "public_probe_then_terminate", 1e-8, 0.95),
    BudgetProtocol(None, 0, None, None, None),
)

finite_sample_calibration_protocol_v1 = ExperimentProtocol(
    "finite_sample_calibration_protocol_v1",
    "finite_calibration",
    _FINITE_STRUCTURE_V1,
    _FINITE_PROBE_V1,
    _FINITE_COORDINATE_V1,
    _FINITE_ACCEPTANCE_V1,
    _FINITE_UNRESOLVED_V1,
    BudgetProtocol(100, 100_000, 10, 40, 50),
)

finite_sample_main_protocol_v1 = ExperimentProtocol(
    "finite_sample_main_protocol_v1",
    "finite_main",
    ObservationalStructureProtocol(1_000, 0.05, 3, _COVARIANCE_V1),
    ProbeInferenceProtocol(
        200,
        1_600,
        2.0,
        4,
        (0.25, 0.5, 1.0),
        0.1,
        0.05,
        0.95,
        "holm",
    ),
    CoordinateSearchProtocol(
        (-2.0, 2.0),
        (log(0.25), log(3.0)),
        7,
        7,
        200,
        1_600,
        "confidence_successive_elimination",
    ),
    CandidateAcceptanceProtocol(1e-3, 0.95, 1_000, 4),
    _FINITE_UNRESOLVED_V1,
    BudgetProtocol(200, 400_000, 20, 80, 100),
)

PROTOCOLS = {
    protocol.protocol_version: protocol
    for protocol in (
        exact_protocol_v1,
        exact_noiseless_protocol_v2,
        finite_sample_calibration_protocol_v1,
        finite_sample_main_protocol_v1,
    )
}


def protocol_by_version(version: str) -> ExperimentProtocol:
    """Resolve one immutable protocol by exact version string."""
    try:
        return PROTOCOLS[version]
    except KeyError as error:
        raise ValueError(f"unknown protocol version {version!r}.") from error
