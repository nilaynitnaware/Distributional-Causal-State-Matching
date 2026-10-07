"""Float64 Torch backend for moments, Gaussian KL, and SCM sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from .torch_gaussian import PreparedGaussianTarget
from .torch_sampling import (
    SamplingMethod,
    make_generator,
    prepare_topological_structure,
    sample_from_noise,
    validate_generator,
    validate_sample_count,
    validate_sampling_method,
)

if TYPE_CHECKING:
    from graphs.interventions import ShiftScaleIntervention
    from graphs.scm import LinearGaussianSCM

ArrayOrTensor = np.ndarray | torch.Tensor


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
    return tensor


class PreparedSCMBatch:
    """Coefficient tensors retained on one device for repeated moment solves.

    Coefficients have shape ``(scm_batch, p, p)``. Intervention parameters may
    have shape ``(p,)`` or ``(intervention_batch, p)``. Singleton SCM or
    intervention batches broadcast, which supports both many independent SCMs
    and many candidate interventions for one SCM.
    """

    __slots__ = (
        "_coefficients",
        "_device",
        "_identity",
        "_n_nodes",
        "_noise_standard_deviations",
        "_parent_coefficients",
        "_parent_indices",
        "_topological_orders",
    )

    def __init__(
        self,
        coefficients: ArrayOrTensor,
        noise_variances: ArrayOrTensor | None = None,
        *,
        device: Literal["cpu", "cuda"],
        prepare_sampling: bool = False,
        topological_orders: ArrayOrTensor | None = None,
    ) -> None:
        """Copy and validate coefficients once at the device boundary."""
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and torch_device.index is None:
            torch_device = torch.device("cuda", torch.cuda.current_device())
        coefficient_tensor = _real_tensor(
            coefficients,
            name="coefficients",
            device=torch_device,
            copy=True,
        )
        if coefficient_tensor.ndim == 2:
            coefficient_tensor = coefficient_tensor.unsqueeze(0)
        elif coefficient_tensor.ndim != 3:
            raise ValueError("coefficients must have shape (p, p) or (batch, p, p).")
        if coefficient_tensor.shape[-2] != coefficient_tensor.shape[-1]:
            raise ValueError("coefficient matrices must be square.")
        if coefficient_tensor.shape[0] == 0:
            raise ValueError("coefficient batch must contain at least one SCM.")
        if not bool(torch.isfinite(coefficient_tensor).all().item()):
            raise ValueError("coefficients must contain only finite values.")

        noise_standard_deviations = None
        if noise_variances is not None:
            variance_tensor = _real_tensor(
                noise_variances,
                name="noise_variances",
                device=torch_device,
                copy=True,
            )
            if variance_tensor.ndim == 1:
                variance_tensor = variance_tensor.unsqueeze(0)
            elif variance_tensor.ndim != 2:
                raise ValueError(
                    "noise_variances must have shape (p,) or (batch, p)."
                )
            if variance_tensor.shape[-1] != coefficient_tensor.shape[-1]:
                raise ValueError(
                    "noise_variance dimensions must agree with coefficients."
                )
            if variance_tensor.shape[0] == 0:
                raise ValueError("noise_variance batch must contain at least one row.")
            if not bool(torch.isfinite(variance_tensor).all().item()):
                raise ValueError("noise_variances must contain only finite values.")
            if not bool(torch.all(variance_tensor > 0.0).item()):
                raise ValueError("noise_variances must be strictly positive.")
            if not bool(torch.all(variance_tensor == 1.0).item()):
                noise_standard_deviations = torch.sqrt(variance_tensor).detach()

        self._device = torch_device
        self._n_nodes = coefficient_tensor.shape[-1]
        self._coefficients = coefficient_tensor.detach()
        self._noise_standard_deviations = noise_standard_deviations
        self._identity = torch.eye(
            self._n_nodes,
            dtype=torch.float64,
            device=torch_device,
        ).unsqueeze(0)
        self._topological_orders = None
        self._parent_indices = None
        self._parent_coefficients = None
        if prepare_sampling or topological_orders is not None:
            (
                self._topological_orders,
                self._parent_indices,
                self._parent_coefficients,
            ) = prepare_topological_structure(
                self._coefficients,
                topological_orders,
            )

    @property
    def device(self) -> Literal["cpu", "cuda"]:
        """Return the device that owns the prepared tensors."""
        return self._device.type

    @property
    def dtype(self) -> Literal["float64"]:
        """Return the fixed scientific precision."""
        return "float64"

    @property
    def n_nodes(self) -> int:
        """Return the common SCM dimension."""
        return self._n_nodes

    @property
    def coefficient_batch_size(self) -> int:
        """Return the number of retained coefficient matrices."""
        return self._coefficients.shape[0]

    @property
    def prepared_bytes(self) -> int:
        """Return bytes held by all retained numerical and topology tensors."""
        total = (
            self._coefficients.numel() + self._identity.numel()
        ) * self._coefficients.element_size()
        if self._noise_standard_deviations is not None:
            total += (
                self._noise_standard_deviations.numel()
                * self._noise_standard_deviations.element_size()
            )
        for tensor in (
            self._topological_orders,
            self._parent_indices,
            self._parent_coefficients,
        ):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total

    @property
    def has_sampling_structure(self) -> bool:
        """Return whether sparse topological metadata is retained on-device."""
        return self._topological_orders is not None

    def prepare_interventions(
        self,
        shift: ArrayOrTensor,
        scale: ArrayOrTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and transfer intervention parameters once."""
        shift_tensor = _real_tensor(
            shift,
            name="shift",
            device=self._device,
            copy=False,
        )
        scale_tensor = _real_tensor(
            scale,
            name="scale",
            device=self._device,
            copy=False,
        )
        shift_tensor = self._validated_parameter_tensor(shift_tensor, "shift")
        scale_tensor = self._validated_parameter_tensor(scale_tensor, "scale")
        if not bool(torch.isfinite(shift_tensor).all().item()) or not bool(
            torch.isfinite(scale_tensor).all().item()
        ):
            raise ValueError("shift and scale must contain only finite values.")
        if not bool(torch.all(scale_tensor > 0.0).item()):
            raise ValueError("all scales must be strictly positive.")
        self._broadcast_batch_size(shift_tensor, scale_tensor)
        return shift_tensor, scale_tensor

    def interventional_moments(
        self,
        shift: ArrayOrTensor,
        scale: ArrayOrTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare parameters and return float64 moment tensors on-device."""
        shift_tensor, scale_tensor = self.prepare_interventions(shift, scale)
        return self.interventional_moments_from_tensors(
            shift_tensor,
            scale_tensor,
        )

    def interventional_moments_from_tensors(
        self,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve moments from already validated, resident parameter tensors."""
        mean, transformed_noise = (
            self.interventional_mean_and_noise_factor_from_tensors(
                shift,
                scale,
            )
        )
        covariance = transformed_noise @ transformed_noise.transpose(-1, -2)

        if not torch.allclose(
            covariance,
            covariance.transpose(-1, -2),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(
                "computed covariance is not symmetric within numerical tolerance."
            )
        return mean, covariance

    def interventional_mean_and_noise_factor_from_tensors(
        self,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and ``T = solve(I - S B^T, I)`` on-device."""
        shift_tensor = self._validated_parameter_tensor(shift, "shift")
        scale_tensor = self._validated_parameter_tensor(scale, "scale")
        batch_size = self._broadcast_batch_size(shift_tensor, scale_tensor)

        coefficients = self._coefficients.expand(batch_size, -1, -1)
        identity = self._identity.expand(batch_size, -1, -1)
        shift_batch = shift_tensor.expand(batch_size, -1)
        scale_batch = scale_tensor.expand(batch_size, -1)
        system_matrix = identity - scale_batch.unsqueeze(-1) * coefficients.transpose(
            -1, -2
        )
        mean = torch.linalg.solve(
            system_matrix,
            shift_batch.unsqueeze(-1),
        ).squeeze(-1)
        noise_factor = identity
        if self._noise_standard_deviations is not None:
            noise_factor = identity * self._noise_standard_deviations.expand(
                batch_size, -1
            ).unsqueeze(-2)
        transformed_noise = torch.linalg.solve(system_matrix, noise_factor)
        return mean, transformed_noise

    def sample_observational(
        self,
        n_samples: int,
        generator: torch.Generator,
        *,
        method: SamplingMethod = "topological",
    ) -> torch.Tensor:
        """Return a resident observational sample batch."""
        shift = torch.zeros(
            self._n_nodes,
            dtype=torch.float64,
            device=self._device,
        )
        scale = torch.ones_like(shift)
        return self.sample_interventional_from_tensors(
            shift,
            scale,
            n_samples,
            generator,
            method=method,
        )

    def sample_interventional(
        self,
        shift: ArrayOrTensor,
        scale: ArrayOrTensor,
        n_samples: int,
        generator: torch.Generator,
        *,
        method: SamplingMethod = "topological",
    ) -> torch.Tensor:
        """Prepare parameters and return a resident interventional sample batch."""
        shift_tensor, scale_tensor = self.prepare_interventions(shift, scale)
        return self.sample_interventional_from_tensors(
            shift_tensor,
            scale_tensor,
            n_samples,
            generator,
            method=method,
        )

    def sample_interventional_from_tensors(
        self,
        shift: torch.Tensor,
        scale: torch.Tensor,
        n_samples: int,
        generator: torch.Generator,
        *,
        method: SamplingMethod = "topological",
    ) -> torch.Tensor:
        """Draw and transform noise using validated resident parameters."""
        validate_sample_count(n_samples)
        sampling_method = validate_sampling_method(method)
        validate_generator(generator, self._device)
        shift_tensor = self._validated_parameter_tensor(shift, "shift")
        scale_tensor = self._validated_parameter_tensor(scale, "scale")
        if not bool(torch.isfinite(shift_tensor).all().item()) or not bool(
            torch.isfinite(scale_tensor).all().item()
        ):
            raise ValueError("shift and scale must contain only finite values.")
        if not bool(torch.all(scale_tensor > 0.0).item()):
            raise ValueError("all scales must be strictly positive.")
        batch_size = self._broadcast_batch_size(shift_tensor, scale_tensor)
        self._require_sampling_structure(sampling_method)
        noise = torch.randn(
            (batch_size, n_samples, self._n_nodes),
            dtype=torch.float64,
            device=self._device,
            generator=generator,
        )
        if self._noise_standard_deviations is not None:
            noise = noise * self._noise_standard_deviations.expand(
                batch_size, -1
            ).unsqueeze(1)
        return sample_from_noise(
            self._coefficients,
            self._identity,
            self._topological_orders,
            self._parent_indices,
            self._parent_coefficients,
            shift_tensor,
            scale_tensor,
            noise,
            method=sampling_method,
        )

    def samples_from_noise(
        self,
        shift: torch.Tensor,
        scale: torch.Tensor,
        noise: torch.Tensor,
        *,
        method: SamplingMethod = "topological",
    ) -> torch.Tensor:
        """Transform explicit resident standard-normal noise without RNG use."""
        sampling_method = validate_sampling_method(method)
        shift_tensor = self._validated_parameter_tensor(shift, "shift")
        scale_tensor = self._validated_parameter_tensor(scale, "scale")
        noise_tensor = self._validated_noise_tensor(noise)
        if not bool(torch.isfinite(shift_tensor).all().item()) or not bool(
            torch.isfinite(scale_tensor).all().item()
        ):
            raise ValueError("shift and scale must contain only finite values.")
        if not bool(torch.all(scale_tensor > 0.0).item()):
            raise ValueError("all scales must be strictly positive.")
        batch_size = self._broadcast_batch_size(
            shift_tensor,
            scale_tensor,
            noise_tensor.shape[0],
        )
        self._require_sampling_structure(sampling_method)
        return sample_from_noise(
            self._coefficients,
            self._identity,
            self._topological_orders,
            self._parent_indices,
            self._parent_coefficients,
            shift_tensor.expand(batch_size, -1),
            scale_tensor.expand(batch_size, -1),
            (
                noise_tensor.expand(batch_size, -1, -1)
                if self._noise_standard_deviations is None
                else noise_tensor.expand(batch_size, -1, -1)
                * self._noise_standard_deviations.expand(batch_size, -1).unsqueeze(1)
            ),
            method=sampling_method,
        )

    def _validated_parameter_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """Validate resident tensor metadata without copying it."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.device != self._device:
            raise ValueError(f"{name} must be on device {self.device}.")
        if tensor.dtype != torch.float64:
            raise ValueError(f"{name} must use torch.float64.")
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError(f"{name} must have shape (p,) or (batch, p).")
        if tensor.shape[-1] != self._n_nodes:
            raise ValueError(f"{name} dimension must agree with the SCM.")
        if tensor.shape[0] == 0:
            raise ValueError(f"{name} batch must contain at least one row.")
        return tensor

    def _validated_noise_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Validate explicit resident noise metadata and values."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("noise must be a torch.Tensor.")
        if tensor.device != self._device:
            raise ValueError(f"noise must be on device {self.device}.")
        if tensor.dtype != torch.float64:
            raise ValueError("noise must use torch.float64.")
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 3:
            raise ValueError(
                "noise must have shape (samples, p) or (batch, samples, p)."
            )
        if tensor.shape[-1] != self._n_nodes:
            raise ValueError("noise dimension must agree with the SCM.")
        if tensor.shape[0] == 0:
            raise ValueError("noise batch must contain at least one row.")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("noise must contain only finite values.")
        return tensor

    def _require_sampling_structure(self, method: SamplingMethod) -> None:
        """Reject topological sampling on a moments-only prepared object."""
        if method == "topological" and self._topological_orders is None:
            raise ValueError("topological sampling requires prepare_scm_sampler().")

    def _broadcast_batch_size(
        self,
        shift: torch.Tensor,
        scale: torch.Tensor,
        *additional_batch_sizes: int,
    ) -> int:
        """Return a common batch size or reject incompatible inputs."""
        batch_sizes = (
            1 if self._noise_standard_deviations is None else self._noise_standard_deviations.shape[0],
            self.coefficient_batch_size,
            shift.shape[0],
            scale.shape[0],
            *additional_batch_sizes,
        )
        batch_size = max(batch_sizes)
        if any(size not in (1, batch_size) for size in batch_sizes):
            raise ValueError(
                "coefficient, shift, and scale batch sizes must be one or equal."
            )
        return batch_size


class TorchBackend:
    """Torch float64 numerical backend on an explicitly selected device."""

    __slots__ = ("deterministic", "device")

    name: Literal["torch"] = "torch"
    dtype: Literal["float64"] = "float64"

    def __init__(
        self,
        *,
        device: Literal["cpu", "cuda"],
        deterministic: bool = True,
    ) -> None:
        """Validate the requested Torch execution device."""
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'.")
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be a boolean.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to PyTorch.")
        self.device = device
        self.deterministic = deterministic

    def prepare_scm_batch(
        self,
        coefficients: ArrayOrTensor,
        noise_variances: ArrayOrTensor | None = None,
    ) -> PreparedSCMBatch:
        """Retain one or more coefficient matrices on the selected device."""
        return PreparedSCMBatch(
            coefficients,
            noise_variances,
            device=self.device,
        )

    def prepare_scm_sampler(
        self,
        coefficients: ArrayOrTensor,
        *,
        noise_variances: ArrayOrTensor | None = None,
        topological_orders: ArrayOrTensor | None = None,
    ) -> PreparedSCMBatch:
        """Retain coefficients and sparse topological sampling metadata."""
        return PreparedSCMBatch(
            coefficients,
            noise_variances,
            device=self.device,
            prepare_sampling=True,
            topological_orders=topological_orders,
        )

    def make_generator(
        self,
        seed: int | np.random.SeedSequence,
    ) -> torch.Generator:
        """Create a reproducible generator local to this backend's device."""
        torch_device = torch.device(self.device)
        if torch_device.type == "cuda" and torch_device.index is None:
            torch_device = torch.device("cuda", torch.cuda.current_device())
        return make_generator(torch_device, seed)

    def prepare_gaussian_target(
        self,
        mean: ArrayOrTensor,
        covariance: ArrayOrTensor,
    ) -> PreparedGaussianTarget:
        """Cache target moments for target-to-candidate KL evaluations."""
        return PreparedGaussianTarget(
            mean,
            covariance,
            device=self.device,
        )

    def observational_moments(
        self,
        scm: LinearGaussianSCM,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return single-SCM observational moments at the CPU boundary."""
        prepared = self.prepare_scm_batch(scm.coefficients, scm.noise_variances)
        mean, covariance = prepared.interventional_moments(
            np.zeros(prepared.n_nodes, dtype=float),
            np.ones(prepared.n_nodes, dtype=float),
        )
        return (
            mean[0].detach().cpu().numpy().copy(),
            covariance[0].detach().cpu().numpy().copy(),
        )

    def interventional_moments(
        self,
        scm: LinearGaussianSCM,
        intervention: ShiftScaleIntervention,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return single-SCM moments as NumPy arrays at the CPU boundary."""
        prepared = self.prepare_scm_batch(scm.coefficients, scm.noise_variances)
        mean, covariance = prepared.interventional_moments(
            intervention.shift,
            intervention.scale,
        )
        return (
            mean[0].detach().cpu().numpy().copy(),
            covariance[0].detach().cpu().numpy().copy(),
        )

    def interventional_moments_batch(
        self,
        coefficients: ArrayOrTensor,
        shift: ArrayOrTensor,
        scale: ArrayOrTensor,
        noise_variances: ArrayOrTensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return broadcast batched moments as NumPy arrays."""
        prepared = self.prepare_scm_batch(coefficients, noise_variances)
        mean, covariance = prepared.interventional_moments(shift, scale)
        return (
            mean.detach().cpu().numpy().copy(),
            covariance.detach().cpu().numpy().copy(),
        )

    def sample_observational(
        self,
        scm: LinearGaussianSCM,
        n_samples: int,
        generator: torch.Generator,
        *,
        method: SamplingMethod = "topological",
    ) -> np.ndarray:
        """Return single-SCM observational samples at the CPU boundary."""
        samples = self.prepare_scm_sampler(
            scm.coefficients,
            noise_variances=scm.noise_variances,
        ).sample_observational(
            n_samples,
            generator,
            method=method,
        )
        return samples[0].detach().cpu().numpy().copy()

    def sample_interventional(
        self,
        scm: LinearGaussianSCM,
        intervention: ShiftScaleIntervention,
        n_samples: int,
        generator: torch.Generator,
        *,
        method: SamplingMethod = "topological",
    ) -> np.ndarray:
        """Return single-SCM interventional samples at the CPU boundary."""
        scm._validate_intervention(intervention)
        samples = self.prepare_scm_sampler(
            scm.coefficients,
            noise_variances=scm.noise_variances,
        ).sample_interventional(
            intervention.shift,
            intervention.scale,
            n_samples,
            generator,
            method=method,
        )
        return samples[0].detach().cpu().numpy().copy()

    def sample_interventional_batch(
        self,
        coefficients: ArrayOrTensor,
        shift: ArrayOrTensor,
        scale: ArrayOrTensor,
        n_samples: int,
        generator: torch.Generator,
        *,
        method: SamplingMethod = "topological",
        noise_variances: ArrayOrTensor | None = None,
        topological_orders: ArrayOrTensor | None = None,
    ) -> np.ndarray:
        """Return broadcast batched interventional samples as a NumPy array."""
        samples = self.prepare_scm_sampler(
            coefficients,
            noise_variances=noise_variances,
            topological_orders=topological_orders,
        ).sample_interventional(
            shift,
            scale,
            n_samples,
            generator,
            method=method,
        )
        return samples.detach().cpu().numpy().copy()

    def gaussian_kl_target_to_candidate(
        self,
        target_mean: np.ndarray,
        target_covariance: np.ndarray,
        candidate_mean: np.ndarray,
        candidate_covariance: np.ndarray,
    ) -> float:
        """Compute one ``KL(target || candidate)`` value."""
        result = self.prepare_gaussian_target(
            target_mean,
            target_covariance,
        ).gaussian_kl_target_to_candidate(candidate_mean, candidate_covariance)
        if result.numel() != 1:
            raise ValueError(
                "gaussian_kl_target_to_candidate requires single target and "
                "candidate Gaussians."
            )
        return float(result.item())

    def gaussian_kl_target_to_candidate_batch(
        self,
        target_mean: ArrayOrTensor,
        target_covariance: ArrayOrTensor,
        candidate_mean: ArrayOrTensor,
        candidate_covariance: ArrayOrTensor,
    ) -> np.ndarray:
        """Compute broadcast batched ``KL(target || candidate)`` values."""
        result = self.prepare_gaussian_target(
            target_mean,
            target_covariance,
        ).gaussian_kl_target_to_candidate(candidate_mean, candidate_covariance)
        return result.detach().cpu().numpy().copy()
