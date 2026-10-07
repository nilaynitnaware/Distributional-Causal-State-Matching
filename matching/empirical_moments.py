"""Documented empirical mean and OAS-style covariance estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from graphs._immutability import immutable_array

from .capabilities import ProviderCapability


def _nonnegative_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


@dataclass(frozen=True)
class CovarianceEstimatorSettings:
    """OAS shrinkage plus a single recorded diagonal-loading fallback."""

    estimator: str = "oas"
    diagonal_loading: float = 1e-10
    minimum_eigenvalue: float = 1e-10

    def __post_init__(self) -> None:
        if self.estimator not in {"oas", "sample"}:
            raise ValueError("estimator must be 'oas' or 'sample'.")
        object.__setattr__(
            self,
            "diagonal_loading",
            _nonnegative_real("diagonal_loading", self.diagonal_loading),
        )
        minimum = _nonnegative_real("minimum_eigenvalue", self.minimum_eigenvalue)
        if minimum == 0.0:
            raise ValueError("minimum_eigenvalue must be positive.")
        object.__setattr__(self, "minimum_eigenvalue", minimum)


@dataclass(frozen=True)
class EmpiricalMomentEstimate:
    """All moment estimates and conditioning diagnostics for one sample set."""

    mean: np.ndarray
    raw_covariance: np.ndarray
    regularized_covariance: np.ndarray
    estimator: str
    shrinkage: float
    diagonal_loading: float
    minimum_eigenvalue_before: float
    minimum_eigenvalue_after: float
    condition_number_before: float
    condition_number_after: float
    pseudoinverse_required: bool
    fallback_required: bool
    cholesky_success: bool
    sample_size: int
    dimension: int

    def __post_init__(self) -> None:
        for name in ("mean", "raw_covariance", "regularized_covariance"):
            array = immutable_array(getattr(self, name), dtype=float)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, array)
        if self.mean.shape != (self.dimension,):
            raise ValueError("mean dimension metadata is inconsistent.")
        expected = (self.dimension, self.dimension)
        if self.raw_covariance.shape != expected:
            raise ValueError("raw covariance dimensions are inconsistent.")
        if self.regularized_covariance.shape != expected:
            raise ValueError("regularized covariance dimensions are inconsistent.")

    def metadata(self, *, include_matrices: bool = False) -> dict[str, object]:
        """Return JSON-compatible estimator diagnostics."""
        metadata: dict[str, object] = {
            "estimator": self.estimator,
            "shrinkage": self.shrinkage,
            "diagonal_loading": self.diagonal_loading,
            "minimum_eigenvalue_before": self.minimum_eigenvalue_before,
            "minimum_eigenvalue_after": self.minimum_eigenvalue_after,
            "condition_number_before": self.condition_number_before,
            "condition_number_after": self.condition_number_after,
            "pseudoinverse_required": self.pseudoinverse_required,
            "fallback_required": self.fallback_required,
            "cholesky_success": self.cholesky_success,
            "sample_size": self.sample_size,
            "dimension": self.dimension,
        }
        if include_matrices:
            metadata.update(
                {
                    "mean": self.mean.tolist(),
                    "raw_covariance": self.raw_covariance.tolist(),
                    "regularized_covariance": self.regularized_covariance.tolist(),
                }
            )
        return metadata


@dataclass(frozen=True)
class EmpiricalMomentsProvider:
    """Estimate public moments without access to simulation structure."""

    settings: CovarianceEstimatorSettings = CovarianceEstimatorSettings()
    record_estimates: bool = False
    _audit_log: list[dict[str, object]] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    capabilities = frozenset({ProviderCapability.EMPIRICAL_SAMPLES})

    def __post_init__(self) -> None:
        if not isinstance(self.settings, CovarianceEstimatorSettings):
            raise TypeError("settings must be CovarianceEstimatorSettings.")
        if not isinstance(self.record_estimates, bool):
            raise TypeError("record_estimates must be a boolean.")

    @property
    def audit_log(self) -> tuple[dict[str, object], ...]:
        """Return full finite-sample moment diagnostics in evaluation order."""
        return tuple(dict(item) for item in self._audit_log)

    def estimate(
        self,
        samples: np.ndarray,
        columns: tuple[int, ...] | None = None,
    ) -> EmpiricalMomentEstimate:
        """Estimate moments for all columns or one explicitly public subset."""
        if not isinstance(samples, np.ndarray):
            raise TypeError("samples must be a numpy.ndarray.")
        if samples.ndim != 2:
            raise ValueError("samples must be two-dimensional.")
        if samples.shape[0] < 2:
            raise ValueError("at least two samples are required.")
        if samples.shape[1] < 1:
            raise ValueError("at least one variable is required.")
        if not np.issubdtype(samples.dtype, np.number) or not np.isrealobj(samples):
            raise TypeError("samples must contain real numeric values.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples must be finite.")

        selected = np.asarray(samples, dtype=float)
        if columns is not None:
            if not isinstance(columns, tuple) or not columns:
                raise TypeError("columns must be a nonempty tuple or None.")
            if len(set(columns)) != len(columns):
                raise ValueError("columns must not contain duplicates.")
            if any(
                isinstance(column, bool)
                or not isinstance(column, int)
                or column < 0
                or column >= samples.shape[1]
                for column in columns
            ):
                raise ValueError("columns must index sample variables.")
            selected = selected[:, columns]

        sample_size, dimension = selected.shape
        mean = np.mean(selected, axis=0)
        centered = selected - mean
        raw = centered.T @ centered / (sample_size - 1)
        maximum_likelihood = centered.T @ centered / sample_size

        shrinkage = 0.0
        if self.settings.estimator == "oas":
            trace_mean = float(np.trace(maximum_likelihood) / dimension)
            alpha = float(np.mean(maximum_likelihood**2))
            denominator = (sample_size + 1.0) * (alpha - trace_mean**2 / dimension)
            numerator = alpha + trace_mean**2
            if denominator <= 0.0:
                shrinkage = 1.0
            else:
                shrinkage = float(np.clip(numerator / denominator, 0.0, 1.0))
            regularized = (1.0 - shrinkage) * maximum_likelihood
            regularized += shrinkage * trace_mean * np.eye(dimension)
        else:
            regularized = raw.copy()

        symmetric = (regularized + regularized.T) / 2.0
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric)))
        raw_symmetric = (raw + raw.T) / 2.0
        raw_minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(raw_symmetric)))
        raw_cholesky_success = True
        try:
            np.linalg.cholesky(raw_symmetric)
        except np.linalg.LinAlgError:
            raw_cholesky_success = False
        loading = max(
            self.settings.diagonal_loading,
            self.settings.minimum_eigenvalue - minimum_eigenvalue,
        )
        regularized = symmetric + loading * np.eye(dimension)
        try:
            np.linalg.cholesky(regularized)
            cholesky_success = True
        except np.linalg.LinAlgError:
            cholesky_success = False

        regularized_minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(regularized)))
        estimate = EmpiricalMomentEstimate(
            mean=mean,
            raw_covariance=raw,
            regularized_covariance=regularized,
            estimator=self.settings.estimator,
            shrinkage=shrinkage,
            diagonal_loading=loading,
            minimum_eigenvalue_before=raw_minimum_eigenvalue,
            minimum_eigenvalue_after=regularized_minimum_eigenvalue,
            condition_number_before=float(np.linalg.cond(raw)),
            condition_number_after=float(np.linalg.cond(regularized)),
            pseudoinverse_required=False,
            fallback_required=not raw_cholesky_success,
            cholesky_success=cholesky_success,
            sample_size=sample_size,
            dimension=dimension,
        )
        if self.record_estimates:
            self._audit_log.append(estimate.metadata(include_matrices=True))
        return estimate
