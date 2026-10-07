"""Explicit experiment-level batching for oracle numerical evaluations.

The scheduler and every graph-policy decision remain CPU-resident.  This
module groups only independent numerical requests with equal dimensions and
returns durable NumPy results in the caller's original request order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from graphs._immutability import immutable_array
from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import GeneratedProblem
from numerical.device import (
    AUTO_DEVICE_THRESHOLDS,
    BackendDiagnostics,
    ComputeConfig,
    NumericalWorkload,
    select_backend,
)

from .analytic_gaussian_kl import OracleAnalyticGaussianKLLoss


@dataclass(frozen=True)
class OracleBatchProblem:
    """One caller-identified problem participating in a numerical batch."""

    request_id: str
    problem: GeneratedProblem

    def __post_init__(self) -> None:
        """Require a stable nonempty identifier and explicit oracle problem."""
        if not isinstance(self.request_id, str):
            raise TypeError("request_id must be a string.")
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty.")
        if not isinstance(self.problem, GeneratedProblem):
            raise TypeError("problem must be a GeneratedProblem.")


@dataclass(frozen=True)
class NumericalBatchRecord:
    """Auditable metadata for one dimension-compatible numerical batch."""

    group_index: int
    n_nodes: int
    request_ids: tuple[str, ...]
    request_count: int
    numerical_batch_size: int
    backend: Literal["numpy", "torch"]
    device: Literal["cpu", "cuda"]
    dtype: Literal["float64"]
    prepared_bytes: int
    preparation_host_to_device_bytes: int
    preparation_host_to_device_transfers: int
    evaluation_host_to_device_transfers: int
    evaluation_device_to_host_transfers: int

    def __post_init__(self) -> None:
        """Validate counts and actual-execution metadata."""
        integer_fields = (
            "group_index",
            "n_nodes",
            "request_count",
            "numerical_batch_size",
            "prepared_bytes",
            "preparation_host_to_device_bytes",
            "preparation_host_to_device_transfers",
            "evaluation_host_to_device_transfers",
            "evaluation_device_to_host_transfers",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if self.n_nodes == 0:
            raise ValueError("n_nodes must be positive.")
        if not isinstance(self.request_ids, tuple) or not self.request_ids:
            raise TypeError("request_ids must be a nonempty tuple.")
        if any(
            not isinstance(item, str) or not item.strip() for item in self.request_ids
        ):
            raise ValueError("request_ids must contain nonempty strings.")
        if self.request_count != len(self.request_ids):
            raise ValueError("request_count must equal the number of request_ids.")
        if not 1 <= self.numerical_batch_size <= self.request_count:
            raise ValueError(
                "numerical_batch_size must lie between one and request_count."
            )
        if self.backend not in ("numpy", "torch"):
            raise ValueError("backend must be 'numpy' or 'torch'.")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'.")
        if self.dtype != "float64":
            raise ValueError("dtype must be 'float64'.")
        if self.backend == "numpy":
            if self.device != "cpu" or self.numerical_batch_size != 1:
                raise ValueError("NumPy records must describe scalar CPU execution.")
            if any(
                getattr(self, name) != 0
                for name in (
                    "prepared_bytes",
                    "preparation_host_to_device_bytes",
                    "preparation_host_to_device_transfers",
                    "evaluation_host_to_device_transfers",
                    "evaluation_device_to_host_transfers",
                )
            ):
                raise ValueError("NumPy records must not report device transfers.")
        else:
            if self.numerical_batch_size != self.request_count:
                raise ValueError("Torch records must execute the whole recorded batch.")
            if self.prepared_bytes <= 0:
                raise ValueError("Torch records must report retained prepared bytes.")
            expected_transfers = (1, 1, 1) if self.device == "cuda" else (0, 0, 0)
            actual_transfers = (
                self.preparation_host_to_device_transfers,
                self.evaluation_host_to_device_transfers,
                self.evaluation_device_to_host_transfers,
            )
            if actual_transfers != expected_transfers:
                raise ValueError(
                    "Torch records must report device transfers only for CUDA."
                )
            if (self.preparation_host_to_device_bytes > 0) != (self.device == "cuda"):
                raise ValueError(
                    "Torch preparation bytes must be positive only for CUDA."
                )


@dataclass(frozen=True)
class OracleInterventionBatchResult:
    """Durable results and truthful execution metadata for one batch round."""

    operation: Literal["value", "value_and_gradient"]
    request_ids: tuple[str, ...]
    values: np.ndarray
    shift_gradients: tuple[np.ndarray, ...] | None
    log_scale_gradients: tuple[np.ndarray, ...] | None
    backend_diagnostics: BackendDiagnostics
    batches: tuple[NumericalBatchRecord, ...]
    host_to_device_bytes: int
    device_to_host_bytes: int

    def __post_init__(self) -> None:
        """Freeze arrays and enforce original-order/result-shape invariants."""
        if self.operation not in ("value", "value_and_gradient"):
            raise ValueError("operation must be 'value' or 'value_and_gradient'.")
        if not isinstance(self.request_ids, tuple) or not self.request_ids:
            raise TypeError("request_ids must be a nonempty tuple.")
        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a numpy.ndarray.")
        if self.values.shape != (len(self.request_ids),):
            raise ValueError("values must contain one entry per request_id.")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("values must contain only finite values.")
        if not isinstance(self.backend_diagnostics, BackendDiagnostics):
            raise TypeError("backend_diagnostics must be a BackendDiagnostics.")
        if not isinstance(self.batches, tuple) or not self.batches:
            raise TypeError("batches must be a nonempty tuple.")
        flattened_ids = tuple(
            request_id for batch in self.batches for request_id in batch.request_ids
        )
        if sorted(flattened_ids) != sorted(self.request_ids):
            raise ValueError("batch request_ids must partition the result request_ids.")
        if len(flattened_ids) != len(set(flattened_ids)):
            raise ValueError("batch request_ids must be unique.")
        for name in ("host_to_device_bytes", "device_to_host_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")

        gradients = (self.shift_gradients, self.log_scale_gradients)
        if self.operation == "value":
            if gradients != (None, None):
                raise ValueError("value results must not contain gradients.")
        else:
            if any(item is None for item in gradients):
                raise ValueError("value_and_gradient results must contain gradients.")
            assert self.shift_gradients is not None
            assert self.log_scale_gradients is not None
            if (
                not isinstance(self.shift_gradients, tuple)
                or not isinstance(self.log_scale_gradients, tuple)
                or len(self.shift_gradients) != len(self.request_ids)
                or len(self.log_scale_gradients) != len(self.request_ids)
            ):
                raise ValueError("gradients must contain one array per request_id.")
            frozen_shift = self._freeze_gradients(
                self.shift_gradients,
                name="shift_gradients",
            )
            frozen_log_scale = self._freeze_gradients(
                self.log_scale_gradients,
                name="log_scale_gradients",
            )
            object.__setattr__(self, "shift_gradients", frozen_shift)
            object.__setattr__(self, "log_scale_gradients", frozen_log_scale)

        object.__setattr__(self, "values", immutable_array(self.values, dtype=float))

    @staticmethod
    def _freeze_gradients(
        gradients: tuple[np.ndarray, ...],
        *,
        name: str,
    ) -> tuple[np.ndarray, ...]:
        """Validate and defensively freeze one heterogeneous gradient tuple."""
        frozen = []
        for gradient in gradients:
            if not isinstance(gradient, np.ndarray):
                raise TypeError(f"{name} must contain numpy.ndarray values.")
            if gradient.ndim != 1:
                raise ValueError(f"{name} must contain one-dimensional arrays.")
            if not np.all(np.isfinite(gradient)):
                raise ValueError(f"{name} must contain only finite values.")
            frozen.append(immutable_array(gradient, dtype=float))
        return tuple(frozen)


@dataclass(frozen=True)
class _PreparedBatch:
    """Internal mapping from one compatible chunk to prepared device state."""

    indices: tuple[int, ...]
    record: NumericalBatchRecord
    loss: Any | None


class OracleInterventionBatchEvaluator:
    """Prepare compatible problems once and evaluate independent candidates.

    Equal-dimensional requests are chunked by ``ComputeConfig.batch_size``.
    NumPy preserves the canonical one-problem-at-a-time implementation.  An
    explicitly selected Torch backend retains each chunk's coefficients and
    target Cholesky factors, sends all candidate parameters in one transfer,
    and returns each numerical output batch in one transfer.
    """

    __slots__ = (
        "_cpu_losses",
        "_diagnostics",
        "_prepared_batches",
        "_problems",
        "_request_ids",
    )

    def __init__(
        self,
        problems: tuple[OracleBatchProblem, ...],
        config: ComputeConfig | None = None,
        *,
        auto_operation: Literal["value", "value_and_gradient"] | None = None,
    ) -> None:
        """Group compatible requests and prepare stable numerical state."""
        if not isinstance(problems, tuple) or not problems:
            raise TypeError("problems must be a nonempty tuple.")
        if any(not isinstance(problem, OracleBatchProblem) for problem in problems):
            raise TypeError("problems must contain OracleBatchProblem values.")
        request_ids = tuple(problem.request_id for problem in problems)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_id values must be unique.")
        active_config = ComputeConfig() if config is None else config
        if not isinstance(active_config, ComputeConfig):
            raise TypeError("config must be a ComputeConfig or None.")
        if auto_operation not in (None, "value", "value_and_gradient"):
            raise ValueError(
                "auto_operation must be 'value', 'value_and_gradient', or None."
            )

        self._problems = problems
        self._request_ids = request_ids
        self._cpu_losses = tuple(
            OracleAnalyticGaussianKLLoss(problem.problem.ground_truth.scm)
            for problem in problems
        )

        grouped_indices: dict[int, list[int]] = {}
        for index, item in enumerate(problems):
            grouped_indices.setdefault(item.problem.public.n_nodes, []).append(index)
        maximum_batch_size = active_config.batch_size
        chunks = []
        for indices in grouped_indices.values():
            chunk_size = maximum_batch_size or len(indices)
            chunks.extend(
                tuple(indices[start : start + chunk_size])
                for start in range(0, len(indices), chunk_size)
            )

        selection_workload = None
        if auto_operation is not None:
            operation = (
                "intervention_kl_value_and_gradient"
                if auto_operation == "value_and_gradient"
                else "intervention_kl_value"
            )
            workloads = tuple(
                NumericalWorkload(
                    operation=operation,
                    n_nodes=problems[chunk[0]].problem.public.n_nodes,
                    batch_size=len(chunk),
                )
                for chunk in chunks
            )
            ineligible = tuple(
                workload
                for workload in workloads
                if not AUTO_DEVICE_THRESHOLDS.allows(workload)
            )
            selection_workload = (
                ineligible[0]
                if ineligible
                else min(workloads, key=lambda workload: workload.dense_work)
            )
        selection = select_backend(active_config, selection_workload)
        self._diagnostics = selection.diagnostics

        prepared_batches = []
        for group_index, chunk in enumerate(chunks):
            indices = tuple(chunk)
            if self._diagnostics.backend == "torch":
                prepared_loss = self._prepare_torch_loss(selection.backend, indices)
                prepared_bytes = int(prepared_loss.prepared_bytes)
                numerical_batch_size = len(indices)
                uses_cuda = self._diagnostics.device == "cuda"
                n_nodes = problems[indices[0]].problem.public.n_nodes
                preparation_bytes = (
                    len(indices) * (2 * n_nodes * n_nodes + n_nodes) * 8
                    if uses_cuda
                    else 0
                )
                preparation_transfers = 1 if uses_cuda else 0
                input_transfers = 1 if uses_cuda else 0
                output_transfers = 1 if uses_cuda else 0
            else:
                prepared_loss = None
                prepared_bytes = 0
                preparation_bytes = 0
                numerical_batch_size = 1
                preparation_transfers = 0
                input_transfers = 0
                output_transfers = 0
            record = NumericalBatchRecord(
                group_index=group_index,
                n_nodes=problems[indices[0]].problem.public.n_nodes,
                request_ids=tuple(request_ids[index] for index in indices),
                request_count=len(indices),
                numerical_batch_size=numerical_batch_size,
                backend=self._diagnostics.backend,
                device=self._diagnostics.device,
                dtype="float64",
                prepared_bytes=prepared_bytes,
                preparation_host_to_device_bytes=preparation_bytes,
                preparation_host_to_device_transfers=preparation_transfers,
                evaluation_host_to_device_transfers=input_transfers,
                evaluation_device_to_host_transfers=output_transfers,
            )
            prepared_batches.append(_PreparedBatch(indices, record, prepared_loss))
        self._prepared_batches = tuple(prepared_batches)

    @property
    def diagnostics(self) -> BackendDiagnostics:
        """Return the requested and actually selected backend metadata."""
        return self._diagnostics

    @property
    def batches(self) -> tuple[NumericalBatchRecord, ...]:
        """Return stable dimension/chunk metadata in first-seen group order."""
        return tuple(batch.record for batch in self._prepared_batches)

    @property
    def problem_count(self) -> int:
        """Return the number of independent requests."""
        return len(self._problems)

    def evaluate(
        self,
        candidates: tuple[ShiftScaleIntervention, ...],
        *,
        gradients: bool = False,
    ) -> OracleInterventionBatchResult:
        """Evaluate one candidate per problem and restore caller request order."""
        self._validate_candidates(candidates)
        if not isinstance(gradients, bool):
            raise TypeError("gradients must be a boolean.")

        values = np.empty(self.problem_count, dtype=float)
        shift_gradients: list[np.ndarray | None] = [None] * self.problem_count
        log_scale_gradients: list[np.ndarray | None] = [None] * self.problem_count
        host_to_device_bytes = 0
        device_to_host_bytes = 0
        for batch in self._prepared_batches:
            if batch.loss is None:
                self._evaluate_numpy_batch(
                    batch.indices,
                    candidates,
                    values,
                    shift_gradients,
                    log_scale_gradients,
                    gradients=gradients,
                )
                continue
            input_bytes, output_bytes = self._evaluate_torch_batch(
                batch,
                candidates,
                values,
                shift_gradients,
                log_scale_gradients,
                gradients=gradients,
            )
            host_to_device_bytes += input_bytes
            device_to_host_bytes += output_bytes

        operation: Literal["value", "value_and_gradient"] = (
            "value_and_gradient" if gradients else "value"
        )
        return OracleInterventionBatchResult(
            operation=operation,
            request_ids=self._request_ids,
            values=values,
            shift_gradients=(
                tuple(self._complete_gradients(shift_gradients)) if gradients else None
            ),
            log_scale_gradients=(
                tuple(self._complete_gradients(log_scale_gradients))
                if gradients
                else None
            ),
            backend_diagnostics=self._diagnostics,
            batches=self.batches,
            host_to_device_bytes=host_to_device_bytes,
            device_to_host_bytes=device_to_host_bytes,
        )

    def _prepare_torch_loss(self, backend: Any, indices: tuple[int, ...]) -> Any:
        """Transfer one packed stable-input group and cache its factors."""
        import torch

        from numerical.torch_gradient import PreparedInterventionKLLoss

        coefficients = np.stack(
            [
                self._problems[index].problem.ground_truth.scm.coefficients
                for index in indices
            ]
        )
        noise_variances = np.stack(
            [
                self._problems[index].problem.ground_truth.scm.noise_variances
                for index in indices
            ]
        )
        target_means = np.stack(
            [self._problems[index].problem.public.target_mean for index in indices]
        )
        target_covariances = np.stack(
            [
                self._problems[index].problem.public.target_covariance
                for index in indices
            ]
        )
        batch_size, n_nodes = target_means.shape
        packed_numpy = np.concatenate(
            (
                coefficients.reshape(batch_size, -1),
                noise_variances,
                target_means,
                target_covariances.reshape(batch_size, -1),
            ),
            axis=1,
        )
        packed = torch.tensor(
            packed_numpy,
            dtype=torch.float64,
            device=backend.device,
        )
        coefficient_end = n_nodes * n_nodes
        noise_end = coefficient_end + n_nodes
        mean_end = noise_end + n_nodes
        return PreparedInterventionKLLoss(
            backend.prepare_scm_batch(
                packed[:, :coefficient_end].reshape(batch_size, n_nodes, n_nodes),
                packed[:, coefficient_end:noise_end],
            ),
            backend.prepare_gaussian_target(
                packed[:, noise_end:mean_end],
                packed[:, mean_end:].reshape(batch_size, n_nodes, n_nodes),
            ),
        )

    def _validate_candidates(
        self,
        candidates: tuple[ShiftScaleIntervention, ...],
    ) -> None:
        """Require exactly one dimension-compatible candidate per request."""
        if not isinstance(candidates, tuple):
            raise TypeError("candidates must be a tuple.")
        if len(candidates) != self.problem_count:
            raise ValueError("candidates must contain one intervention per problem.")
        for candidate, problem in zip(candidates, self._problems, strict=True):
            if not isinstance(candidate, ShiftScaleIntervention):
                raise TypeError(
                    "candidates must contain ShiftScaleIntervention values."
                )
            if candidate.shift.shape != (problem.problem.public.n_nodes,):
                raise ValueError("candidate dimension must agree with its problem.")

    def _evaluate_numpy_batch(
        self,
        indices: tuple[int, ...],
        candidates: tuple[ShiftScaleIntervention, ...],
        values: np.ndarray,
        shift_gradients: list[np.ndarray | None],
        log_scale_gradients: list[np.ndarray | None],
        *,
        gradients: bool,
    ) -> None:
        """Use the unchanged canonical scalar analytic implementation."""
        for index in indices:
            loss = self._cpu_losses[index]
            if gradients:
                value, shift_gradient, log_scale_gradient = loss.value_and_gradient(
                    candidates[index],
                    self._problems[index].problem.public,
                )
                shift_gradients[index] = shift_gradient
                log_scale_gradients[index] = log_scale_gradient
            else:
                value = loss.value(
                    candidates[index],
                    self._problems[index].problem.public,
                )
            values[index] = value

    def _evaluate_torch_batch(
        self,
        batch: _PreparedBatch,
        candidates: tuple[ShiftScaleIntervention, ...],
        values: np.ndarray,
        shift_gradients: list[np.ndarray | None],
        log_scale_gradients: list[np.ndarray | None],
        *,
        gradients: bool,
    ) -> tuple[int, int]:
        """Use one input and one output transfer for an entire Torch batch."""
        import torch

        parameters = np.stack(
            (
                np.stack([candidates[index].shift for index in batch.indices]),
                np.log(np.stack([candidates[index].scale for index in batch.indices])),
            ),
            axis=1,
        )
        parameter_tensor = torch.tensor(
            parameters,
            dtype=torch.float64,
            device=batch.record.device,
        )
        if gradients:
            resident = batch.loss.value_and_gradient_from_tensors(
                parameter_tensor[:, 0, :],
                parameter_tensor[:, 1, :],
            )
            output_tensor = torch.cat(
                (resident[0].unsqueeze(-1), resident[1], resident[2]),
                dim=-1,
            )
        else:
            output_tensor = batch.loss.value_from_tensors(
                parameter_tensor[:, 0, :],
                parameter_tensor[:, 1, :],
            ).unsqueeze(-1)
        output = output_tensor.detach().cpu().numpy().copy()
        n_nodes = batch.record.n_nodes
        for row, index in enumerate(batch.indices):
            values[index] = output[row, 0]
            if gradients:
                shift_gradients[index] = output[row, 1 : n_nodes + 1]
                log_scale_gradients[index] = output[row, n_nodes + 1 :]
        if batch.record.device == "cuda":
            return int(parameters.nbytes), int(output.nbytes)
        return 0, 0

    @staticmethod
    def _complete_gradients(
        gradients: list[np.ndarray | None],
    ) -> list[np.ndarray]:
        """Narrow a fully populated internal result list."""
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("internal batch evaluation left a gradient unset.")
        return [gradient for gradient in gradients if gradient is not None]


__all__ = [
    "NumericalBatchRecord",
    "OracleBatchProblem",
    "OracleInterventionBatchEvaluator",
    "OracleInterventionBatchResult",
]
