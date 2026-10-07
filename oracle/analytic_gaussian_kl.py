"""Oracle Gaussian KL loss and analytic intervention gradients.

This module intentionally imports the hidden SCM. Generic matching and metric
modules must not import this module or the hidden causal model.
"""

from dataclasses import dataclass

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.problem_gen import PublicProblemView
from graphs.scm import LinearGaussianSCM
from metrics.gaussian_kl import gaussian_kl_target_to_candidate


def _precision_product(
    cholesky: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Return ``covariance^-1 @ matrix`` without forming an inverse."""
    return np.linalg.solve(
        cholesky.T,
        np.linalg.solve(cholesky, matrix),
    )


@dataclass(frozen=True)
class OracleAnalyticGaussianKLLoss:
    """Exact target-to-candidate Gaussian KL using the hidden true SCM.

    Computes ``D_KL(Q_target || P_candidate)``; the expectation is taken under
    the target distribution. The gradient result is ordered as
    ``(shift, log_scale)``. This is an oracle baseline only: evaluating
    candidate moments and their derivatives requires the true coefficient
    matrix.
    """

    scm: LinearGaussianSCM

    def __post_init__(self) -> None:
        """Require an explicit hidden SCM."""
        if not isinstance(self.scm, LinearGaussianSCM):
            raise TypeError("scm must be a LinearGaussianSCM.")

    def value(
        self,
        candidate: ShiftScaleIntervention,
        problem_view: PublicProblemView,
    ) -> float:
        """Return target-to-candidate Gaussian KL under the true SCM."""
        self._validate_inputs(candidate, problem_view)
        candidate_mean, candidate_covariance = self.scm.interventional_moments(
            candidate
        )
        return gaussian_kl_target_to_candidate(
            target_mean=problem_view.target_mean,
            target_covariance=problem_view.target_covariance,
            candidate_mean=candidate_mean,
            candidate_covariance=candidate_covariance,
        )

    def value_and_gradient(
        self,
        candidate: ShiftScaleIntervention,
        problem_view: PublicProblemView,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Return loss and gradients with respect to shift and log-scale."""
        self._validate_inputs(candidate, problem_view)
        n_nodes = problem_view.n_nodes
        coefficients_transpose = self.scm.coefficients.T
        system_matrix = (
            np.eye(n_nodes) - candidate.scale[:, None] * coefficients_transpose
        )
        noise_factor = np.diag(self.scm.noise_standard_deviations)
        transformed_noise = np.linalg.solve(system_matrix, noise_factor)
        candidate_mean = np.linalg.solve(system_matrix, candidate.shift)
        candidate_covariance = transformed_noise @ transformed_noise.T
        value = gaussian_kl_target_to_candidate(
            target_mean=problem_view.target_mean,
            target_covariance=problem_view.target_covariance,
            candidate_mean=candidate_mean,
            candidate_covariance=candidate_covariance,
        )

        candidate_cholesky = np.linalg.cholesky(candidate_covariance)
        mean_residual = candidate_mean - problem_view.target_mean
        mean_gradient = _precision_product(
            candidate_cholesky,
            mean_residual,
        )
        shift_gradient = np.linalg.solve(system_matrix.T, mean_gradient)

        target_second_moment_about_candidate = (
            problem_view.target_covariance + np.outer(mean_residual, mean_residual)
        )
        precision_weighted_second_moment = _precision_product(
            candidate_cholesky,
            target_second_moment_about_candidate,
        )

        scale_gradient = np.empty(n_nodes, dtype=float)
        parental_mean = coefficients_transpose @ candidate_mean
        parental_noise = coefficients_transpose @ transformed_noise
        for node in range(n_nodes):
            mean_rhs = np.zeros(n_nodes, dtype=float)
            mean_rhs[node] = parental_mean[node]
            mean_derivative = np.linalg.solve(system_matrix, mean_rhs)

            noise_rhs = np.zeros((n_nodes, n_nodes), dtype=float)
            noise_rhs[node, :] = parental_noise[node, :]
            noise_derivative = np.linalg.solve(system_matrix, noise_rhs)
            covariance_derivative = (
                noise_derivative @ transformed_noise.T
                + transformed_noise @ noise_derivative.T
            )
            precision_weighted_derivative = _precision_product(
                candidate_cholesky,
                covariance_derivative,
            )
            covariance_gradient = 0.5 * (
                np.trace(precision_weighted_derivative)
                - np.trace(
                    precision_weighted_second_moment @ precision_weighted_derivative
                )
            )
            scale_gradient[node] = mean_gradient @ mean_derivative + covariance_gradient

        log_scale_gradient = candidate.scale * scale_gradient
        return value, shift_gradient, log_scale_gradient

    def _validate_inputs(
        self,
        candidate: ShiftScaleIntervention,
        problem_view: PublicProblemView,
    ) -> None:
        """Validate public inputs and their dimensions against hidden truth."""
        if not isinstance(candidate, ShiftScaleIntervention):
            raise TypeError("candidate must be a ShiftScaleIntervention.")
        if not isinstance(problem_view, PublicProblemView):
            raise TypeError("problem_view must be a PublicProblemView.")
        n_nodes = self.scm.dag.number_of_nodes()
        if candidate.shift.shape != (n_nodes,):
            raise ValueError("candidate dimension must agree with the SCM.")
        if problem_view.n_nodes != n_nodes:
            raise ValueError("problem view dimension must agree with the SCM.")
