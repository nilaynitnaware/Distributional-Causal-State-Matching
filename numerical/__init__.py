"""Public numerical backend and device-selection API."""

from .device import (
    AUTO_DEVICE_THRESHOLDS,
    AutoDeviceThresholds,
    BackendDiagnostics,
    BackendSelection,
    BackendUnavailableError,
    ComputeConfig,
    NumericalWorkload,
    TorchRuntimeInfo,
    detect_torch_runtime,
    select_backend,
)
from .numpy_backend import NUMPY_BACKEND, NumPyBackend
from .protocol import MomentBackend, NumericalBackend, SamplingBackend

__all__ = [
    "AUTO_DEVICE_THRESHOLDS",
    "NUMPY_BACKEND",
    "AutoDeviceThresholds",
    "BackendDiagnostics",
    "BackendSelection",
    "BackendUnavailableError",
    "ComputeConfig",
    "MomentBackend",
    "NumPyBackend",
    "NumericalBackend",
    "NumericalWorkload",
    "SamplingBackend",
    "TorchRuntimeInfo",
    "detect_torch_runtime",
    "select_backend",
]
