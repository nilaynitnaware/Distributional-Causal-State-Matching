"""Cached float64 Torch evaluation of Gaussian KL divergence."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

ArrayOrTensor = np.ndarray | torch.Tensor

_SYMMETRY_RTOL = 1e-10
_SYMMETRY_ATOL = 1e-12


@dataclass(frozen=True)
class NumericalFailure:
    """Structured information about a failed batched numerical operation."""

    operation: str
    reason: str
    batch_indices: tuple[int, ...]


class TorchNumericalError(RuntimeError):
    """Raised when a Torch numerical operation cannot return a valid result."""

    def __init__(self, failure: NumericalFailure) -> None:
        self.failure = failure
        indices = ", ".join(str(index) for index in failure.batch_indices)
        super().__init__(
            f"{failure.operation} failed for batch indices [{indices}]: "
            f"{failure.reason}."
        )


def _torch_device(device: Literal["cpu", "cuda"]) -> torch.device:
    """Resolve an explicit device to the exact device used by resident tensors."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _real_tensor(
    value: ArrayOrTensor,
    *,
    name: str,
    device: torch.device,
    copy: bool,
) -> torch.Tensor:
    """Convert real numeric input to a float64 tensor on ``device``."""
    if isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.number) or not np.isrealobj(value):
            raise TypeError(f"{name} must contain real numeric values.")
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
    elif isinstance(value, torch.Tensor):
        if value.dtype == torch.bool or torch.is_complex(value):
            raise TypeError(f"{name} must contain real numeric values.")
        tensor = value.to(device=device, dtype=torch.float64)
        if copy:
            tensor = tensor.clone()
    else:
        raise TypeError(f"{name} must be a numpy.ndarray or torch.Tensor.")
    return tensor.detach()


def _mean_batch(
    tensor: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Normalize a mean to ``(batch, p)``."""
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 2:
        raise ValueError(f"{name} must have shape (p,) or (batch, p).")
    if tensor.shape[0] == 0:
        raise ValueError(f"{name} batch must contain at least one row.")
    return tensor


def _covariance_batch(
    tensor: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Normalize a covariance to ``(batch, p, p)``."""
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 3:
        raise ValueError(f"{name} must have shape (p, p) or (batch, p, p).")
    if tensor.shape[-2] != tensor.shape[-1]:
        raise ValueError(f"{name} matrices must be square.")
    if tensor.shape[0] == 0:
        raise ValueError(f"{name} batch must contain at least one matrix.")
    return tensor


def _broadcast_batch_size(*sizes: int, names: str) -> int:
    """Return the singleton-broadcast batch size or reject incompatible inputs."""
    batch_size = max(sizes)
    if any(size not in (1, batch_size) for size in sizes):
        raise ValueError(f"{names} batch sizes must be one or equal.")
    return batch_size


def _indices(mask: torch.Tensor) -> tuple[int, ...]:
    """Materialize only failing batch indices for an exception."""
    return tuple(
        int(index)
        for index in torch.nonzero(mask, as_tuple=False).flatten().cpu().tolist()
    )


def _raise_failure(
    *,
    operation: str,
    reason: str,
    mask: torch.Tensor,
) -> None:
    """Raise a structured failure if any effective batch row failed."""
    if bool(mask.any().item()):
        raise TorchNumericalError(
            NumericalFailure(
                operation=operation,
                reason=reason,
                batch_indices=_indices(mask),
            )
        )


def _validate_values(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    *,
    operation: str,
) -> None:
    """Validate finite values and covariance symmetry with batch attribution."""
    nonfinite = ~torch.isfinite(mean).all(dim=-1)
    nonfinite |= ~torch.isfinite(covariance).all(dim=(-2, -1))
    _raise_failure(
        operation=operation,
        reason="mean and covariance must contain only finite values",
        mask=nonfinite,
    )
    symmetric = torch.isclose(
        covariance,
        covariance.transpose(-1, -2),
        rtol=_SYMMETRY_RTOL,
        atol=_SYMMETRY_ATOL,
    ).all(dim=(-2, -1))
    _raise_failure(
        operation=operation,
        reason="covariance must be symmetric within numerical tolerance",
        mask=~symmetric,
    )


class PreparedGaussianTarget:
    """A target Gaussian retained for target-to-candidate KL evaluation.

    Target means and covariances may be single values or batches. Singleton
    target and candidate batches broadcast. Candidate tensors returned by
    :meth:`prepare_candidates` can be evaluated without NumPy conversion or
    bulk device-to-host transfer. The cached target Cholesky supplies the
    target covariance factor and log determinant; inverse-dependent solves use
    a freshly factorized candidate covariance.
    """

    __slots__ = (
        "_cholesky",
        "_device",
        "_log_determinant",
        "_mean",
        "_n_nodes",
    )

    def __init__(
        self,
        mean: ArrayOrTensor,
        covariance: ArrayOrTensor,
        *,
        device: Literal["cpu", "cuda"],
    ) -> None:
        """Transfer, validate, and factorize the target exactly once."""
        torch_device = _torch_device(device)
        mean_tensor = _mean_batch(
            _real_tensor(
                mean,
                name="target mean",
                device=torch_device,
                copy=True,
            ),
            name="target mean",
        )
        covariance_tensor = _covariance_batch(
            _real_tensor(
                covariance,
                name="target covariance",
                device=torch_device,
                copy=True,
            ),
            name="target covariance",
        )
        n_nodes = covariance_tensor.shape[-1]
        if mean_tensor.shape[-1] != n_nodes:
            raise ValueError("target mean and covariance dimensions must agree.")
        batch_size = _broadcast_batch_size(
            mean_tensor.shape[0],
            covariance_tensor.shape[0],
            names="target mean and covariance",
        )
        mean_tensor = mean_tensor.expand(batch_size, -1)
        covariance_tensor = covariance_tensor.expand(batch_size, -1, -1)
        _validate_values(
            mean_tensor,
            covariance_tensor,
            operation="target preparation",
        )
        cholesky, info = torch.linalg.cholesky_ex(
            covariance_tensor,
            check_errors=False,
        )
        _raise_failure(
            operation="target Cholesky factorization",
            reason="covariance must be positive definite",
            mask=info != 0,
        )

        self._device = torch_device
        self._n_nodes = n_nodes
        self._mean = mean_tensor
        self._cholesky = cholesky
        self._log_determinant = 2.0 * torch.log(
            torch.diagonal(cholesky, dim1=-2, dim2=-1)
        ).sum(dim=-1)

    @property
    def device(self) -> Literal["cpu", "cuda"]:
        """Return the device that owns all cached target tensors."""
        return self._device.type

    @property
    def dtype(self) -> Literal["float64"]:
        """Return the fixed scientific precision."""
        return "float64"

    @property
    def n_nodes(self) -> int:
        """Return the Gaussian dimension."""
        return self._n_nodes

    @property
    def target_batch_size(self) -> int:
        """Return the number of cached target factors."""
        return self._cholesky.shape[0]

    @property
    def prepared_bytes(self) -> int:
        """Return logical bytes occupied by cached target tensors."""
        element_size = self._cholesky.element_size()
        return (
            self._mean.numel() + self._cholesky.numel() + self._log_determinant.numel()
        ) * element_size

    def prepare_candidates(
        self,
        mean: ArrayOrTensor,
        covariance: ArrayOrTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transfer and validate a candidate batch on the target device."""
        mean_tensor = _mean_batch(
            _real_tensor(
                mean,
                name="candidate mean",
                device=self._device,
                copy=False,
            ),
            name="candidate mean",
        )
        covariance_tensor = _covariance_batch(
            _real_tensor(
                covariance,
                name="candidate covariance",
                device=self._device,
                copy=False,
            ),
            name="candidate covariance",
        )
        if (
            mean_tensor.shape[-1] != self._n_nodes
            or covariance_tensor.shape[-1] != self._n_nodes
        ):
            raise ValueError("candidate and target Gaussian dimensions must agree.")
        batch_size = _broadcast_batch_size(
            mean_tensor.shape[0],
            covariance_tensor.shape[0],
            names="candidate mean and covariance",
        )
        mean_tensor = mean_tensor.expand(batch_size, -1)
        covariance_tensor = covariance_tensor.expand(batch_size, -1, -1)
        _validate_values(
            mean_tensor,
            covariance_tensor,
            operation="candidate preparation",
        )
        return mean_tensor, covariance_tensor

    def gaussian_kl_target_to_candidate(
        self,
        mean: ArrayOrTensor,
        covariance: ArrayOrTensor,
    ) -> torch.Tensor:
        """Return ``KL(target || candidate)`` as a resident batch tensor."""
        mean_tensor, covariance_tensor = self.prepare_candidates(
            mean,
            covariance,
        )
        return self.gaussian_kl_target_to_candidate_from_tensors(
            mean_tensor,
            covariance_tensor,
        )

    def gaussian_kl_target_to_candidate_from_tensors(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target-to-candidate KL from resident candidate tensors."""
        mean_tensor = self._resident_mean(mean)
        covariance_tensor = self._resident_covariance(covariance)
        candidate_batch_size = _broadcast_batch_size(
            mean_tensor.shape[0],
            covariance_tensor.shape[0],
            names="candidate mean and covariance",
        )
        mean_tensor = mean_tensor.expand(candidate_batch_size, -1)
        covariance_tensor = covariance_tensor.expand(
            candidate_batch_size,
            -1,
            -1,
        )

        candidate_cholesky, info = torch.linalg.cholesky_ex(
            covariance_tensor,
            check_errors=False,
        )
        _raise_failure(
            operation="candidate Cholesky factorization",
            reason="covariance must be positive definite",
            mask=info != 0,
        )
        candidate_log_determinant = 2.0 * torch.log(
            torch.diagonal(
                candidate_cholesky,
                dim1=-2,
                dim2=-1,
            )
        ).sum(dim=-1)
        return self._gaussian_kl_target_to_candidate_from_factor(
            mean_tensor,
            candidate_cholesky,
            candidate_log_determinant,
        )

    def gaussian_kl_target_to_candidate_from_factor_tensors(
        self,
        mean: torch.Tensor,
        transformed_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target-to-candidate KL from ``Sigma = T T^T``."""
        mean_tensor = self._resident_mean(mean)
        factor_tensor = self._resident_factor(transformed_noise)
        candidate_batch_size = _broadcast_batch_size(
            mean_tensor.shape[0],
            factor_tensor.shape[0],
            names="candidate mean and factor",
        )
        mean_tensor = mean_tensor.expand(candidate_batch_size, -1)
        factor_tensor = factor_tensor.expand(candidate_batch_size, -1, -1)
        nonfinite = ~torch.isfinite(mean_tensor).all(dim=-1)
        nonfinite |= ~torch.isfinite(factor_tensor).all(dim=(-2, -1))
        _raise_failure(
            operation="candidate factor preparation",
            reason="mean and factor must contain only finite values",
            mask=nonfinite,
        )
        bmm_scope = (
            torch.backends.python_native.operations_disabled("bmm")
            if self.device == "cuda"
            else nullcontext()
        )
        with bmm_scope:
            candidate_covariance = factor_tensor @ factor_tensor.transpose(-1, -2)
        return self.gaussian_kl_target_to_candidate_from_tensors(
            mean_tensor,
            candidate_covariance,
        )

    def _gaussian_kl_target_to_candidate_from_factor(
        self,
        mean: torch.Tensor,
        candidate_cholesky: torch.Tensor,
        candidate_log_determinant: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL using a Cholesky factor of the candidate covariance."""
        batch_size = _broadcast_batch_size(
            self.target_batch_size,
            mean.shape[0],
            candidate_cholesky.shape[0],
            candidate_log_determinant.shape[0],
            names="target and candidate",
        )
        mean = mean.expand(batch_size, -1)
        candidate_cholesky = candidate_cholesky.expand(batch_size, -1, -1)
        candidate_log_determinant = candidate_log_determinant.expand(batch_size)
        target_mean = self._mean.expand(batch_size, -1)
        target_cholesky = self._cholesky.expand(batch_size, -1, -1)
        target_log_determinant = self._log_determinant.expand(batch_size)

        whitened_target_covariance_factor = torch.linalg.solve_triangular(
            candidate_cholesky,
            target_cholesky,
            upper=False,
        )
        trace_term = whitened_target_covariance_factor.square().sum(dim=(-2, -1))
        residual = mean - target_mean
        whitened_residual = torch.linalg.solve_triangular(
            candidate_cholesky,
            residual.unsqueeze(-1),
            upper=False,
        )
        quadratic_term = whitened_residual.square().sum(dim=(-2, -1))
        raw_kl = 0.5 * (
            trace_term
            + quadratic_term
            - self._n_nodes
            + candidate_log_determinant
            - target_log_determinant
        )
        finite = torch.isfinite(raw_kl)
        _raise_failure(
            operation="Gaussian KL evaluation",
            reason="result must be finite",
            mask=~finite,
        )

        scale = torch.ones_like(raw_kl)
        for term in (
            trace_term,
            quadratic_term,
            target_log_determinant,
            candidate_log_determinant,
        ):
            scale = torch.maximum(scale, term.abs())
        scale = torch.maximum(
            scale,
            torch.full_like(scale, float(self._n_nodes)),
        )
        roundoff_tolerance = 64.0 * torch.finfo(torch.float64).eps * scale
        _raise_failure(
            operation="Gaussian KL evaluation",
            reason="computed divergence is negative beyond roundoff tolerance",
            mask=raw_kl < -roundoff_tolerance,
        )
        return torch.where(
            raw_kl.abs() <= roundoff_tolerance,
            torch.zeros_like(raw_kl),
            raw_kl,
        )

    def _resident_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """Validate candidate mean metadata without copying."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("candidate mean must be a torch.Tensor.")
        if tensor.device != self._device:
            raise ValueError(f"candidate mean must be on device {self.device}.")
        if tensor.dtype != torch.float64:
            raise ValueError("candidate mean must use torch.float64.")
        tensor = _mean_batch(tensor, name="candidate mean")
        if tensor.shape[-1] != self._n_nodes:
            raise ValueError("candidate and target Gaussian dimensions must agree.")
        return tensor

    def _resident_covariance(self, tensor: torch.Tensor) -> torch.Tensor:
        """Validate candidate covariance metadata without copying."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("candidate covariance must be a torch.Tensor.")
        if tensor.device != self._device:
            raise ValueError(f"candidate covariance must be on device {self.device}.")
        if tensor.dtype != torch.float64:
            raise ValueError("candidate covariance must use torch.float64.")
        tensor = _covariance_batch(tensor, name="candidate covariance")
        if tensor.shape[-1] != self._n_nodes:
            raise ValueError("candidate and target Gaussian dimensions must agree.")
        return tensor

    def _resident_factor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Validate candidate transformed-noise factor metadata without copying."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("candidate factor must be a torch.Tensor.")
        if tensor.device != self._device:
            raise ValueError(f"candidate factor must be on device {self.device}.")
        if tensor.dtype != torch.float64:
            raise ValueError("candidate factor must use torch.float64.")
        tensor = _covariance_batch(tensor, name="candidate factor")
        if tensor.shape[-1] != self._n_nodes:
            raise ValueError("candidate and target Gaussian dimensions must agree.")
        return tensor
