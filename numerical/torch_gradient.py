"""Resident float64 Torch gradients for shift-scale Gaussian KL."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import numpy as np
import torch

from .torch_backend import PreparedSCMBatch
from .torch_gaussian import PreparedGaussianTarget

ArrayOrTensor = np.ndarray | torch.Tensor


class PreparedInterventionKLLoss:
    """An SCM and Gaussian target retained for batched autograd evaluations.

    Shift parameters and log-scales may be single vectors or batches. Singleton
    inputs are materialized to the effective SCM/target batch before autograd,
    so returned gradients always contain one independent row per loss value.
    """

    __slots__ = ("_device", "_scm", "_target")

    def __init__(
        self,
        scm: PreparedSCMBatch,
        target: PreparedGaussianTarget,
    ) -> None:
        """Require compatible prepared objects on the same Torch device."""
        if not isinstance(scm, PreparedSCMBatch):
            raise TypeError("scm must be a PreparedSCMBatch.")
        if not isinstance(target, PreparedGaussianTarget):
            raise TypeError("target must be a PreparedGaussianTarget.")
        if scm.device != target.device:
            raise ValueError("prepared SCM and target must use the same device.")
        if scm.n_nodes != target.n_nodes:
            raise ValueError("prepared SCM and target dimensions must agree.")
        batch_size = max(
            scm.coefficient_batch_size,
            target.target_batch_size,
        )
        if any(
            size not in (1, batch_size)
            for size in (
                scm.coefficient_batch_size,
                target.target_batch_size,
            )
        ):
            raise ValueError(
                "prepared SCM and target batch sizes must be one or equal."
            )

        self._device = torch.device(scm.device)
        if self._device.type == "cuda" and self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        self._scm = scm
        self._target = target

    @property
    def device(self) -> Literal["cpu", "cuda"]:
        """Return the device owning all prepared state."""
        return self._device.type

    @property
    def dtype(self) -> Literal["float64"]:
        """Return the fixed scientific precision."""
        return "float64"

    @property
    def n_nodes(self) -> int:
        """Return the common SCM and target dimension."""
        return self._scm.n_nodes

    @property
    def prepared_bytes(self) -> int:
        """Return logical bytes occupied by retained SCM and target tensors."""
        return self._scm.prepared_bytes + self._target.prepared_bytes

    def prepare_parameters(
        self,
        shift: ArrayOrTensor,
        log_scale: ArrayOrTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transfer candidate parameters without enabling autograd yet."""
        shift_tensor = self._coerce_parameter(shift, name="shift")
        log_scale_tensor = self._coerce_parameter(log_scale, name="log_scale")
        batch_size = self._effective_batch_size(shift_tensor, log_scale_tensor)
        return (
            shift_tensor.expand(batch_size, -1).clone(),
            log_scale_tensor.expand(batch_size, -1).clone(),
        )

    def value(
        self,
        shift: ArrayOrTensor,
        log_scale: ArrayOrTensor,
    ) -> torch.Tensor:
        """Return resident batched losses from NumPy or Torch parameters."""
        parameters = self.prepare_parameters(shift, log_scale)
        return self.value_from_tensors(*parameters)

    def value_from_tensors(
        self,
        shift: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Return losses from resident float64 parameter tensors."""
        shift_tensor = self._resident_parameter(shift, name="shift")
        log_scale_tensor = self._resident_parameter(
            log_scale,
            name="log_scale",
        )
        batch_size = self._effective_batch_size(shift_tensor, log_scale_tensor)
        with torch.no_grad():
            values = self._evaluate(
                shift_tensor.expand(batch_size, -1),
                log_scale_tensor.expand(batch_size, -1),
            )
        return values.detach()

    def value_and_gradient(
        self,
        shift: ArrayOrTensor,
        log_scale: ArrayOrTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transfer parameters and return resident values and full gradients."""
        parameters = self.prepare_parameters(shift, log_scale)
        return self.value_and_gradient_from_tensors(*parameters)

    def value_and_gradient_from_tensors(
        self,
        shift: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return values and independent per-row gradients on the device."""
        shift_tensor = self._resident_parameter(shift, name="shift")
        log_scale_tensor = self._resident_parameter(
            log_scale,
            name="log_scale",
        )
        batch_size = self._effective_batch_size(shift_tensor, log_scale_tensor)
        # PyTorch 2.13 routes CUDA batched outer products in solve backward to
        # a JIT-compiled Triton override. The documented scoped controller
        # keeps this operation on PyTorch's native CUDA kernel, so gradient
        # evaluation does not require a host C compiler.
        bmm_scope = (
            torch.backends.python_native.operations_disabled("bmm")
            if self.device == "cuda"
            else nullcontext()
        )
        with bmm_scope, torch.enable_grad():
            shift_leaf = (
                shift_tensor.expand(batch_size, -1)
                .clone()
                .detach()
                .requires_grad_(True)
            )
            log_scale_leaf = (
                log_scale_tensor.expand(batch_size, -1)
                .clone()
                .detach()
                .requires_grad_(True)
            )
            values = self._evaluate(shift_leaf, log_scale_leaf)
            shift_gradient, log_scale_gradient = torch.autograd.grad(
                values.sum(),
                (shift_leaf, log_scale_leaf),
            )
        return (
            values.detach(),
            shift_gradient.detach(),
            log_scale_gradient.detach(),
        )

    def _evaluate(
        self,
        shift: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate exact intervention moments and target-to-candidate KL."""
        scale = torch.exp(log_scale)
        if not bool(torch.isfinite(scale).all().item()) or not bool(
            torch.all(scale > 0.0).item()
        ):
            raise ValueError("exp(log_scale) must be finite and strictly positive.")
        mean, transformed_noise = (
            self._scm.interventional_mean_and_noise_factor_from_tensors(
                shift,
                scale,
            )
        )
        return self._target.gaussian_kl_target_to_candidate_from_factor_tensors(
            mean,
            transformed_noise,
        )

    def _coerce_parameter(
        self,
        value: ArrayOrTensor,
        *,
        name: str,
    ) -> torch.Tensor:
        """Convert one real parameter array to resident float64 storage."""
        if isinstance(value, np.ndarray):
            if not np.issubdtype(value.dtype, np.number) or not np.isrealobj(value):
                raise TypeError(f"{name} must contain real numeric values.")
            tensor = torch.tensor(
                value,
                dtype=torch.float64,
                device=self._device,
            )
        elif isinstance(value, torch.Tensor):
            if value.dtype == torch.bool or torch.is_complex(value):
                raise TypeError(f"{name} must contain real numeric values.")
            tensor = value.to(device=self._device, dtype=torch.float64).detach()
        else:
            raise TypeError(f"{name} must be a numpy.ndarray or torch.Tensor.")
        return self._validated_shape_and_values(tensor, name=name)

    def _resident_parameter(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        """Validate resident parameter metadata without a host/device copy."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.device != self._device:
            raise ValueError(f"{name} must be on device {self.device}.")
        if tensor.dtype != torch.float64:
            raise ValueError(f"{name} must use torch.float64.")
        return self._validated_shape_and_values(tensor, name=name)

    def _validated_shape_and_values(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        """Normalize one parameter to ``(batch, p)`` and validate it."""
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError(f"{name} must have shape (p,) or (batch, p).")
        if tensor.shape[-1] != self.n_nodes:
            raise ValueError(f"{name} dimension must agree with the SCM.")
        if tensor.shape[0] == 0:
            raise ValueError(f"{name} batch must contain at least one row.")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} must contain only finite values.")
        return tensor

    def _effective_batch_size(
        self,
        shift: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> int:
        """Return the singleton-broadcast batch size for all prepared state."""
        sizes = (
            self._scm.coefficient_batch_size,
            self._target.target_batch_size,
            shift.shape[0],
            log_scale.shape[0],
        )
        batch_size = max(sizes)
        if any(size not in (1, batch_size) for size in sizes):
            raise ValueError(
                "SCM, target, shift, and log_scale batch sizes must be one or equal."
            )
        return batch_size
