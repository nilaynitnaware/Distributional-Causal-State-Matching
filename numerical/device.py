"""Explicit backend configuration, runtime detection, and selection."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal

from .numpy_backend import NUMPY_BACKEND
from .protocol import NumericalBackend

BackendPreference = Literal["auto", "numpy", "torch"]
DevicePreference = Literal["auto", "cpu", "cuda"]
DTypeName = Literal["float32", "float64"]
NumericalOperation = Literal[
    "intervention_kl_value",
    "intervention_kl_value_and_gradient",
    "interventional_moments",
    "interventional_sampling",
]

_NUMERICAL_OPERATIONS = frozenset(
    {
        "intervention_kl_value",
        "intervention_kl_value_and_gradient",
        "interventional_moments",
        "interventional_sampling",
    }
)


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested numerical backend cannot be used."""


@dataclass(frozen=True)
class NumericalWorkload:
    """Shape and operation metadata used only by automatic device selection."""

    operation: NumericalOperation
    n_nodes: int
    batch_size: int

    def __post_init__(self) -> None:
        """Validate an explicit compatible numerical workload."""
        if self.operation not in _NUMERICAL_OPERATIONS:
            raise ValueError("operation must be a supported numerical operation.")
        for name in ("n_nodes", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

    @property
    def dense_work(self) -> int:
        """Return the profiled dense-work metric ``p² × batch``."""
        return self.n_nodes * self.n_nodes * self.batch_size


@dataclass(frozen=True)
class AutoDeviceThresholds:
    """Conservative CUDA frontier derived from the saved GPU-9 benchmark."""

    operation: NumericalOperation
    minimum_dense_work: int
    shape_frontier: tuple[tuple[int, int], ...]
    minimum_observed_speedup: float
    profiled_gpu_name: str
    source: str

    def __post_init__(self) -> None:
        """Validate the persisted threshold policy at import time."""
        if self.operation not in _NUMERICAL_OPERATIONS:
            raise ValueError("operation must be a supported numerical operation.")
        if (
            isinstance(self.minimum_dense_work, bool)
            or not isinstance(self.minimum_dense_work, int)
            or self.minimum_dense_work <= 0
        ):
            raise ValueError("minimum_dense_work must be a positive integer.")
        if not isinstance(self.shape_frontier, tuple) or not self.shape_frontier:
            raise TypeError("shape_frontier must be a nonempty tuple.")
        for item in self.shape_frontier:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("shape_frontier entries must be two-item tuples.")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in item
            ):
                raise ValueError("shape_frontier values must be positive integers.")
        if len(set(self.shape_frontier)) != len(self.shape_frontier):
            raise ValueError("shape_frontier entries must be unique.")
        if (
            isinstance(self.minimum_observed_speedup, bool)
            or not isinstance(self.minimum_observed_speedup, Real)
            or not isfinite(float(self.minimum_observed_speedup))
            or self.minimum_observed_speedup <= 1.0
        ):
            raise ValueError("minimum_observed_speedup must be finite and above one.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a nonempty string.")
        if (
            not isinstance(self.profiled_gpu_name, str)
            or not self.profiled_gpu_name.strip()
        ):
            raise ValueError("profiled_gpu_name must be a nonempty string.")

    def allows(self, workload: NumericalWorkload) -> bool:
        """Return whether a workload lies inside the measured safe region."""
        if not isinstance(workload, NumericalWorkload):
            raise TypeError("workload must be a NumericalWorkload.")
        return (
            workload.operation == self.operation
            and workload.dense_work >= self.minimum_dense_work
            and any(
                workload.n_nodes >= minimum_nodes
                and workload.batch_size >= minimum_batch_size
                for minimum_nodes, minimum_batch_size in self.shape_frontier
            )
        )


AUTO_DEVICE_THRESHOLDS = AutoDeviceThresholds(
    operation="intervention_kl_value_and_gradient",
    minimum_dense_work=5_000,
    shape_frontier=((100, 1), (50, 2), (20, 32), (10, 64)),
    minimum_observed_speedup=4.73574093172006,
    profiled_gpu_name="NVIDIA RTX PRO 5000 Blackwell",
    source="benchmarks/device_thresholds.json (schema_version=1)",
)


@dataclass(frozen=True)
class ComputeConfig:
    """User-visible numerical execution preferences.

    ``auto`` uses CUDA only for a supplied workload inside the conservative
    GPU-9 benchmark frontier. Missing or unprofiled workload metadata remains
    on NumPy CPU. Only float64 execution is enabled during backend rollout.
    """

    backend: BackendPreference = "auto"
    device: DevicePreference = "auto"
    dtype: DTypeName = "float64"
    deterministic: bool = True
    batch_size: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid configuration before backend selection."""
        if self.backend not in ("auto", "numpy", "torch"):
            raise ValueError("backend must be 'auto', 'numpy', or 'torch'.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'.")
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be a boolean.")
        if self.batch_size is not None:
            if isinstance(self.batch_size, bool) or not isinstance(
                self.batch_size,
                int,
            ):
                raise TypeError("batch_size must be an integer or None.")
            if self.batch_size <= 0:
                raise ValueError("batch_size must be positive when provided.")


@dataclass(frozen=True)
class TorchRuntimeInfo:
    """Observed optional PyTorch and CUDA runtime state."""

    torch_available: bool
    cuda_available: bool
    torch_version: str | None = None
    cuda_runtime_version: str | None = None
    gpu_name: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class BackendDiagnostics:
    """Record requested settings and the backend that was actually selected."""

    requested_backend: BackendPreference
    requested_device: DevicePreference
    backend: Literal["numpy", "torch"]
    device: Literal["cpu", "cuda"]
    dtype: DTypeName
    deterministic: bool
    batch_size: int | None
    workload_operation: NumericalOperation | None
    workload_n_nodes: int | None
    workload_batch_size: int | None
    workload_dense_work: int | None
    auto_threshold_applied: bool
    auto_threshold_source: str | None
    auto_threshold_minimum_dense_work: int | None
    auto_threshold_profiled_gpu_name: str | None
    torch_available: bool
    cuda_available: bool
    torch_version: str | None
    cuda_runtime_version: str | None
    gpu_name: str | None
    fallback_occurred: bool
    selection_reason: str


@dataclass(frozen=True)
class BackendSelection:
    """A resolved backend paired with auditable diagnostics."""

    backend: NumericalBackend
    diagnostics: BackendDiagnostics


def detect_torch_runtime() -> TorchRuntimeInfo:
    """Probe PyTorch lazily so the NumPy backend works without that package."""
    try:
        import torch
    except (ImportError, OSError) as error:
        return TorchRuntimeInfo(
            torch_available=False,
            cuda_available=False,
            detail=f"{type(error).__name__}: {error}",
        )

    torch_version = str(torch.__version__)
    cuda_runtime_version = torch.version.cuda
    try:
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except (AssertionError, OSError, RuntimeError) as error:
        return TorchRuntimeInfo(
            torch_available=True,
            cuda_available=False,
            torch_version=torch_version,
            cuda_runtime_version=cuda_runtime_version,
            detail=f"{type(error).__name__}: {error}",
        )

    return TorchRuntimeInfo(
        torch_available=True,
        cuda_available=cuda_available,
        torch_version=torch_version,
        cuda_runtime_version=cuda_runtime_version,
        gpu_name=gpu_name,
    )


def _numpy_selection(
    config: ComputeConfig,
    runtime: TorchRuntimeInfo,
    workload: NumericalWorkload | None,
    *,
    fallback_occurred: bool,
    selection_reason: str,
    auto_threshold_applied: bool = False,
) -> BackendSelection:
    """Construct a NumPy selection with complete runtime diagnostics."""
    return BackendSelection(
        backend=NUMPY_BACKEND,
        diagnostics=BackendDiagnostics(
            requested_backend=config.backend,
            requested_device=config.device,
            backend="numpy",
            device="cpu",
            dtype=config.dtype,
            deterministic=config.deterministic,
            batch_size=config.batch_size,
            workload_operation=(workload.operation if workload is not None else None),
            workload_n_nodes=(workload.n_nodes if workload is not None else None),
            workload_batch_size=(workload.batch_size if workload is not None else None),
            workload_dense_work=(workload.dense_work if workload is not None else None),
            auto_threshold_applied=auto_threshold_applied,
            auto_threshold_source=(
                AUTO_DEVICE_THRESHOLDS.source if auto_threshold_applied else None
            ),
            auto_threshold_minimum_dense_work=(
                AUTO_DEVICE_THRESHOLDS.minimum_dense_work
                if auto_threshold_applied
                else None
            ),
            auto_threshold_profiled_gpu_name=(
                AUTO_DEVICE_THRESHOLDS.profiled_gpu_name
                if auto_threshold_applied
                else None
            ),
            torch_available=runtime.torch_available,
            cuda_available=runtime.cuda_available,
            torch_version=runtime.torch_version,
            cuda_runtime_version=runtime.cuda_runtime_version,
            gpu_name=runtime.gpu_name,
            fallback_occurred=fallback_occurred,
            selection_reason=selection_reason,
        ),
    )


def _torch_selection(
    config: ComputeConfig,
    runtime: TorchRuntimeInfo,
    workload: NumericalWorkload | None,
    *,
    device: Literal["cpu", "cuda"],
    selection_reason: str,
    auto_threshold_applied: bool = False,
) -> BackendSelection:
    """Construct an explicit Torch numerical backend selection."""
    from .torch_backend import TorchBackend

    return BackendSelection(
        backend=TorchBackend(
            device=device,
            deterministic=config.deterministic,
        ),
        diagnostics=BackendDiagnostics(
            requested_backend=config.backend,
            requested_device=config.device,
            backend="torch",
            device=device,
            dtype=config.dtype,
            deterministic=config.deterministic,
            batch_size=config.batch_size,
            workload_operation=(workload.operation if workload is not None else None),
            workload_n_nodes=(workload.n_nodes if workload is not None else None),
            workload_batch_size=(workload.batch_size if workload is not None else None),
            workload_dense_work=(workload.dense_work if workload is not None else None),
            auto_threshold_applied=auto_threshold_applied,
            auto_threshold_source=(
                AUTO_DEVICE_THRESHOLDS.source if auto_threshold_applied else None
            ),
            auto_threshold_minimum_dense_work=(
                AUTO_DEVICE_THRESHOLDS.minimum_dense_work
                if auto_threshold_applied
                else None
            ),
            auto_threshold_profiled_gpu_name=(
                AUTO_DEVICE_THRESHOLDS.profiled_gpu_name
                if auto_threshold_applied
                else None
            ),
            torch_available=runtime.torch_available,
            cuda_available=runtime.cuda_available,
            torch_version=runtime.torch_version,
            cuda_runtime_version=runtime.cuda_runtime_version,
            gpu_name=runtime.gpu_name,
            fallback_occurred=False,
            selection_reason=selection_reason,
        ),
    )


def _require_torch_runtime(
    runtime: TorchRuntimeInfo,
    *,
    require_cuda: bool,
) -> None:
    """Raise the most specific error for an unavailable explicit request."""
    if not runtime.torch_available:
        detail = f" ({runtime.detail})" if runtime.detail else ""
        raise BackendUnavailableError(
            f"PyTorch is unavailable{detail}; use backend='numpy' on device='cpu'."
        )
    if require_cuda and not runtime.cuda_available:
        detail = f" ({runtime.detail})" if runtime.detail else ""
        raise BackendUnavailableError(
            f"CUDA is unavailable to PyTorch{detail}; explicit device='cuda' "
            "cannot fall back to CPU."
        )


def _select_backend(
    config: ComputeConfig,
    runtime: TorchRuntimeInfo,
    workload: NumericalWorkload | None,
) -> BackendSelection:
    """Resolve a validated configuration against observed runtime state."""
    if config.dtype != "float64":
        raise ValueError(
            "only float64 is supported until CPU/GPU numerical parity is established."
        )

    if config.backend == "numpy":
        if config.device == "cuda":
            raise BackendUnavailableError(
                "The NumPy backend is CPU-only; device='cuda' requires a CUDA backend."
            )
        return _numpy_selection(
            config,
            runtime,
            workload,
            fallback_occurred=False,
            selection_reason="The NumPy CPU backend was explicitly requested.",
        )

    if config.backend == "torch":
        _require_torch_runtime(
            runtime,
            require_cuda=config.device == "cuda",
        )
        torch_device: Literal["cpu", "cuda"] = (
            "cuda" if config.device == "cuda" else "cpu"
        )
        return _torch_selection(
            config,
            runtime,
            workload,
            device=torch_device,
            selection_reason=(
                f"The Torch {torch_device.upper()} moment backend was "
                "explicitly requested."
            ),
        )

    if config.device == "cuda":
        _require_torch_runtime(runtime, require_cuda=True)
        return _torch_selection(
            config,
            runtime,
            workload,
            device="cuda",
            selection_reason="CUDA execution was explicitly requested.",
        )

    if config.device == "cpu":
        return _numpy_selection(
            config,
            runtime,
            workload,
            fallback_occurred=False,
            selection_reason="CPU execution was explicitly requested.",
        )

    if not runtime.torch_available:
        return _numpy_selection(
            config,
            runtime,
            workload,
            fallback_occurred=True,
            selection_reason=(
                "Auto selection used NumPy on CPU because PyTorch is unavailable."
            ),
        )
    if not runtime.cuda_available:
        return _numpy_selection(
            config,
            runtime,
            workload,
            fallback_occurred=True,
            selection_reason=(
                "Auto selection used NumPy on CPU because CUDA is unavailable."
            ),
        )
    if workload is None:
        return _numpy_selection(
            config,
            runtime,
            workload,
            fallback_occurred=False,
            selection_reason=(
                "Auto selection used NumPy CPU because no profiled workload "
                "shape was provided."
            ),
        )
    if runtime.gpu_name != AUTO_DEVICE_THRESHOLDS.profiled_gpu_name:
        return _numpy_selection(
            config,
            runtime,
            workload,
            fallback_occurred=False,
            auto_threshold_applied=True,
            selection_reason=(
                "Auto selection used NumPy CPU because the available GPU "
                f"'{runtime.gpu_name}' does not match the profiled GPU "
                f"'{AUTO_DEVICE_THRESHOLDS.profiled_gpu_name}'."
            ),
        )
    if AUTO_DEVICE_THRESHOLDS.allows(workload):
        return _torch_selection(
            config,
            runtime,
            workload,
            device="cuda",
            auto_threshold_applied=True,
            selection_reason=(
                "Auto selection used Torch CUDA because the profiled workload "
                f"score {workload.dense_work} and shape "
                f"({workload.n_nodes}, {workload.batch_size}) satisfy the "
                "conservative GPU-9 threshold frontier."
            ),
        )
    return _numpy_selection(
        config,
        runtime,
        workload,
        fallback_occurred=False,
        auto_threshold_applied=True,
        selection_reason=(
            "Auto selection used NumPy CPU because the operation or workload "
            f"shape is outside the conservative GPU-9 threshold frontier; "
            f"observed score={workload.dense_work}, required dense score>="
            f"{AUTO_DEVICE_THRESHOLDS.minimum_dense_work}."
        ),
    )


def select_backend(
    config: ComputeConfig | None = None,
    workload: NumericalWorkload | None = None,
) -> BackendSelection:
    """Select the actual backend without ever silently claiming CUDA use."""
    active_config = ComputeConfig() if config is None else config
    if not isinstance(active_config, ComputeConfig):
        raise TypeError("config must be a ComputeConfig or None.")
    if workload is not None and not isinstance(workload, NumericalWorkload):
        raise TypeError("workload must be a NumericalWorkload or None.")
    return _select_backend(active_config, detect_torch_runtime(), workload)
