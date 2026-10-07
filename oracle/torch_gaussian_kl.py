"""Optional Torch oracle Gaussian KL loss with float64 autograd gradients."""

from __future__ import annotations

from typing import Literal

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import PublicProblemView
from graphs.scm import LinearGaussianSCM
from numerical.torch_backend import TorchBackend
from numerical.torch_gradient import PreparedInterventionKLLoss


class TorchOracleGaussianKLLoss:
    """Oracle loss adapter backed by a prepared CPU or CUDA Torch calculation.

    The hidden coefficient matrix is transferred once. A public target is
    factorized once per ``PublicProblemView`` object and reused by subsequent
    value and gradient evaluations, including calls from SciPy L-BFGS-B.
    """

    __slots__ = (
        "_cached_loss",
        "_cached_problem_view",
        "_prepared_scm",
        "_target_preparation_count",
        "device",
        "scm",
    )

    def __init__(
        self,
        scm: LinearGaussianSCM,
        *,
        device: Literal["cpu", "cuda"],
    ) -> None:
        """Prepare the hidden SCM on an explicit Torch device."""
        if not isinstance(scm, LinearGaussianSCM):
            raise TypeError("scm must be a LinearGaussianSCM.")
        backend = TorchBackend(device=device)
        self.scm = scm
        self.device = device
        self._prepared_scm = backend.prepare_scm_batch(
            scm.coefficients,
            scm.noise_variances,
        )
        self._cached_problem_view: PublicProblemView | None = None
        self._cached_loss: PreparedInterventionKLLoss | None = None
        self._target_preparation_count = 0

    @property
    def dtype(self) -> Literal["float64"]:
        """Return the fixed scientific precision."""
        return "float64"

    @property
    def target_preparation_count(self) -> int:
        """Return the number of public targets transferred and factorized."""
        return self._target_preparation_count

    @property
    def prepared_bytes(self) -> int:
        """Return logical resident bytes, including a cached target if present."""
        if self._cached_loss is None:
            return self._prepared_scm.prepared_bytes
        return self._cached_loss.prepared_bytes

    def value(
        self,
        candidate: ShiftScaleIntervention,
        problem_view: PublicProblemView,
    ) -> float:
        """Return target-to-candidate Gaussian KL under the true SCM."""
        self._validate_inputs(candidate, problem_view)
        values = self._loss_for(problem_view).value(
            candidate.shift,
            np.log(candidate.scale),
        )
        return float(values.item())

    def value_and_gradient(
        self,
        candidate: ShiftScaleIntervention,
        problem_view: PublicProblemView,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Return loss and autograd derivatives for shift and log-scale."""
        self._validate_inputs(candidate, problem_view)
        value, shift_gradient, log_scale_gradient = self._loss_for(
            problem_view
        ).value_and_gradient(
            candidate.shift,
            np.log(candidate.scale),
        )
        return (
            float(value.item()),
            shift_gradient[0].cpu().numpy().copy(),
            log_scale_gradient[0].cpu().numpy().copy(),
        )

    def _loss_for(
        self,
        problem_view: PublicProblemView,
    ) -> PreparedInterventionKLLoss:
        """Return the cached prepared target loss for one public view."""
        if self._cached_problem_view is not problem_view:
            backend = TorchBackend(device=self.device)
            target = backend.prepare_gaussian_target(
                problem_view.target_mean,
                problem_view.target_covariance,
            )
            self._cached_loss = PreparedInterventionKLLoss(
                self._prepared_scm,
                target,
            )
            self._cached_problem_view = problem_view
            self._target_preparation_count += 1
        if self._cached_loss is None:
            raise RuntimeError("prepared Torch oracle loss was not initialized.")
        return self._cached_loss

    def _validate_inputs(
        self,
        candidate: ShiftScaleIntervention,
        problem_view: PublicProblemView,
    ) -> None:
        """Validate public inputs and dimensions against hidden truth."""
        if not isinstance(candidate, ShiftScaleIntervention):
            raise TypeError("candidate must be a ShiftScaleIntervention.")
        if not isinstance(problem_view, PublicProblemView):
            raise TypeError("problem_view must be a PublicProblemView.")
        n_nodes = self.scm.dag.number_of_nodes()
        if candidate.shift.shape != (n_nodes,):
            raise ValueError("candidate dimension must agree with the SCM.")
        if problem_view.n_nodes != n_nodes:
            raise ValueError("problem view dimension must agree with the SCM.")
