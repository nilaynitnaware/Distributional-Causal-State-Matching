"""Numerical components for linear Gaussian structural causal models."""

from dataclasses import dataclass

import networkx as nx
import numpy as np

from ._immutability import immutable_array
from .interventions import ShiftScaleIntervention

DEFAULT_BETA_MIN = 1e-12


def _validate_indexed_dag(dag: object) -> int:
    """Validate the shared simple-DAG and integer-label convention."""
    if not isinstance(dag, nx.DiGraph) or dag.is_multigraph():
        raise TypeError("dag must be a simple networkx.DiGraph.")
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("dag must be acyclic.")

    n_nodes = dag.number_of_nodes()
    if any(
        isinstance(node, bool) or not isinstance(node, (int, np.integer))
        for node in dag.nodes
    ) or {int(node) for node in dag.nodes} != set(range(n_nodes)):
        raise ValueError("dag nodes must be labeled 0, ..., n_nodes - 1.")

    return n_nodes


def _validate_sampling_request(
    n_samples: int,
    rng: np.random.Generator,
) -> None:
    """Validate arguments shared by observational and interventional sampling."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise TypeError("n_samples must be an integer.")
    if n_samples < 0:
        raise ValueError("n_samples must be nonnegative.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")


def sample_coefficients(
    dag: nx.DiGraph,
    rng: np.random.Generator,
    min_abs: float,
    max_abs: float,
) -> np.ndarray:
    """Sample a coefficient matrix supported exactly on a DAG's edges.

    The matrix convention is ``coefficients[parent, child] = beta``. Every
    edge magnitude is sampled uniformly from ``[min_abs, max_abs]`` and given
    an independent uniformly random sign. Non-edges, including the diagonal,
    remain zero. Edges are sorted before sampling so equivalent graphs with
    different edge-insertion orders produce the same matrix for a fixed RNG
    state.

    Args:
        dag: Simple DAG with nodes labeled ``0, ..., n_nodes - 1``.
        rng: Explicit NumPy random-number generator.
        min_abs: Inclusive minimum absolute edge coefficient.
        max_abs: Inclusive maximum absolute edge coefficient.

    Returns:
        A floating-point array with shape ``(n_nodes, n_nodes)``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If the graph or magnitude bounds are invalid.
    """
    n_nodes = _validate_indexed_dag(dag)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    bounds = {"min_abs": min_abs, "max_abs": max_abs}
    for name, value in bounds.items():
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a real number.")

    minimum = float(min_abs)
    maximum = float(max_abs)
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("min_abs and max_abs must be finite.")
    if not 0.0 < minimum <= maximum:
        raise ValueError("coefficient bounds must satisfy 0 < min_abs <= max_abs.")

    coefficients = np.zeros((n_nodes, n_nodes), dtype=float)
    edges = sorted((int(parent), int(child)) for parent, child in dag.edges)
    magnitudes = rng.uniform(minimum, maximum, size=len(edges))
    signs = np.where(rng.integers(0, 2, size=len(edges)) == 0, -1.0, 1.0)

    for (parent, child), coefficient in zip(edges, magnitudes * signs, strict=True):
        coefficients[parent, child] = coefficient

    return coefficients


@dataclass(frozen=True)
class LinearGaussianSCM:
    """A linear Gaussian SCM with independent node-specific Gaussian noise.

    ``coefficients[parent, child]`` is the coefficient on the corresponding
    directed edge. Construction defensively copies and freezes both inputs, so
    the simulation environment cannot be changed through caller-owned objects.
    """

    dag: nx.DiGraph
    coefficients: np.ndarray
    beta_min: float = DEFAULT_BETA_MIN
    noise_variances: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate and defensively freeze the SCM definition."""
        n_nodes = _validate_indexed_dag(self.dag)

        if not isinstance(self.coefficients, np.ndarray):
            raise TypeError("coefficients must be a numpy.ndarray.")
        if self.coefficients.ndim != 2 or (
            self.coefficients.shape[0] != self.coefficients.shape[1]
        ):
            raise ValueError("coefficients must be a square matrix.")
        if self.coefficients.shape != (n_nodes, n_nodes):
            raise ValueError(
                "coefficient dimensions must agree with the number of graph nodes."
            )
        if not np.issubdtype(self.coefficients.dtype, np.number) or not np.isrealobj(
            self.coefficients
        ):
            raise TypeError("coefficients must contain real numeric values.")
        if not np.all(np.isfinite(self.coefficients)):
            raise ValueError("coefficients must contain only finite values.")

        if isinstance(self.beta_min, bool) or not isinstance(
            self.beta_min,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("beta_min must be a real number.")
        beta_min = float(self.beta_min)
        if not np.isfinite(beta_min) or beta_min <= 0.0:
            raise ValueError("beta_min must be finite and positive.")

        edge_support = np.zeros((n_nodes, n_nodes), dtype=bool)
        for parent, child in self.dag.edges:
            edge_support[int(parent), int(child)] = True
        if np.any(self.coefficients[~edge_support] != 0.0):
            raise ValueError("nonzero coefficients are allowed only on graph edges.")
        if np.any(np.abs(self.coefficients[edge_support]) < beta_min):
            raise ValueError(
                "every graph edge coefficient magnitude must be at least beta_min."
            )

        if self.noise_variances is None:
            noise_variances = np.ones(n_nodes, dtype=float)
        else:
            if not isinstance(self.noise_variances, np.ndarray):
                raise TypeError("noise_variances must be a numpy.ndarray.")
            if self.noise_variances.shape != (n_nodes,):
                raise ValueError("noise_variances must have shape (n_nodes,).")
            if not np.issubdtype(
                self.noise_variances.dtype, np.number
            ) or not np.isrealobj(self.noise_variances):
                raise TypeError("noise_variances must contain real numeric values.")
            if not np.all(np.isfinite(self.noise_variances)):
                raise ValueError("noise_variances must contain only finite values.")
            if np.any(self.noise_variances <= 0.0):
                raise ValueError("noise_variances must be strictly positive.")
            noise_variances = self.noise_variances

        frozen_dag = nx.freeze(nx.DiGraph(self.dag))
        frozen_coefficients = immutable_array(self.coefficients, dtype=float)
        frozen_noise_variances = immutable_array(noise_variances, dtype=float)
        object.__setattr__(self, "dag", frozen_dag)
        object.__setattr__(self, "coefficients", frozen_coefficients)
        object.__setattr__(self, "beta_min", beta_min)
        object.__setattr__(self, "noise_variances", frozen_noise_variances)

    @property
    def noise_standard_deviations(self) -> np.ndarray:
        """Return immutable node-wise exogenous standard deviations."""
        return immutable_array(np.sqrt(self.noise_variances), dtype=float)

    @property
    def noise_covariance(self) -> np.ndarray:
        """Return the immutable diagonal exogenous covariance matrix."""
        return immutable_array(np.diag(self.noise_variances), dtype=float)

    def sample_observational(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample observations in deterministic topological order.

        Independent Gaussian noise with the configured node variances is first
        sampled for every row and node. For each node ``i`` in a
        lexicographical topological order, the parental contribution
        ``sum_j B[j, i] X_j`` is then added to its noise.

        Args:
            n_samples: Nonnegative number of independent observations.
            rng: Explicit NumPy random-number generator.

        Returns:
            A floating-point array with shape ``(n_samples, n_nodes)``.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If ``n_samples`` is negative.
        """
        _validate_sampling_request(n_samples, rng)
        n_nodes = self.dag.number_of_nodes()
        return self._sample_shift_scale(
            shift=np.zeros(n_nodes, dtype=float),
            scale=np.ones(n_nodes, dtype=float),
            n_samples=n_samples,
            rng=rng,
        )

    def sample_interventional(
        self,
        intervention: ShiftScaleIntervention,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample under a node-wise shift-scale intervention.

        At each node, scale multiplies only the contribution from already
        intervened parents. Independent node-wise Gaussian noise is added
        afterwards and therefore retains the configured conditional variance.

        Args:
            intervention: Immutable shift-scale intervention for every node.
            n_samples: Nonnegative number of independent observations.
            rng: Explicit NumPy random-number generator.

        Returns:
            A floating-point array with shape ``(n_samples, n_nodes)``.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If the intervention dimension disagrees with the SCM or
                ``n_samples`` is negative.
        """
        self._validate_intervention(intervention)
        _validate_sampling_request(n_samples, rng)

        return self._sample_shift_scale(
            shift=intervention.shift,
            scale=intervention.scale,
            n_samples=n_samples,
            rng=rng,
        )

    def observational_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact observational mean and covariance."""
        return self.interventional_moments(
            ShiftScaleIntervention.identity(self.dag.number_of_nodes())
        )

    def interventional_moments(
        self,
        intervention: ShiftScaleIntervention,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact mean and covariance under a shift-scale intervention.

        With ``S = diag(intervention.scale)``,
        ``D = diag(noise_standard_deviations)``, and
        ``M = I - S @ coefficients.T``, linear solves compute
        ``mean = M^-1 @ shift`` and the noise transform ``M^-1 @ D``. No
        explicit matrix inverse is formed.

        Args:
            intervention: Immutable shift-scale intervention for every node.

        Returns:
            A pair ``(mean, covariance)`` with shapes ``(n_nodes,)`` and
            ``(n_nodes, n_nodes)``.

        Raises:
            TypeError: If ``intervention`` has an invalid type.
            ValueError: If the intervened system is invalid or its moments are
                nonfinite.
            RuntimeError: If numerical output is not symmetric within tolerance.
        """
        self._validate_intervention(intervention)
        n_nodes = self.dag.number_of_nodes()
        system_matrix = (
            np.eye(n_nodes) - intervention.scale[:, None] * self.coefficients.T
        )
        if not np.all(np.isfinite(system_matrix)):
            raise ValueError("intervened linear system must contain finite values.")
        try:
            mean = np.linalg.solve(system_matrix, intervention.shift)
            noise_factor = np.eye(n_nodes)
            if not np.array_equal(
                self.noise_variances,
                np.ones(n_nodes, dtype=float),
            ):
                noise_factor *= self.noise_standard_deviations
            transformed_noise = np.linalg.solve(system_matrix, noise_factor)
        except np.linalg.LinAlgError as error:
            raise ValueError("intervened linear system must be invertible.") from error
        covariance = transformed_noise @ transformed_noise.T

        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("intervened moments must be finite.")
        if not np.allclose(
            covariance,
            covariance.T,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(
                "computed covariance is not symmetric within numerical tolerance."
            )
        return mean, covariance

    def _validate_intervention(
        self,
        intervention: ShiftScaleIntervention,
    ) -> None:
        """Validate intervention type and dimension against this SCM."""
        if not isinstance(intervention, ShiftScaleIntervention):
            raise TypeError("intervention must be a ShiftScaleIntervention.")
        n_nodes = self.dag.number_of_nodes()
        if intervention.shift.shape != (n_nodes,):
            raise ValueError(
                "intervention dimension must agree with the number of SCM nodes."
            )

    def validate_target_intervention(
        self,
        intervention: ShiftScaleIntervention,
    ) -> None:
        """Reject no-op root scales in a target definition."""
        self._validate_intervention(intervention)
        roots = np.array(
            [node for node, degree in self.dag.in_degree if degree == 0],
            dtype=int,
        )
        if roots.size and np.any(intervention.scale[roots] != 1.0):
            raise ValueError("target cannot contain a nondefault root-node scale.")

    @property
    def scale_eligible_nodes(self) -> frozenset[int]:
        """Return nodes whose structural parent mechanism can be scaled."""
        return frozenset(int(node) for node, degree in self.dag.in_degree if degree > 0)

    @property
    def actionable_parameter_coordinate_count(self) -> int:
        """Count all shifts plus scale coordinates at nonroot nodes."""
        return self.dag.number_of_nodes() + len(self.scale_eligible_nodes)

    def _sample_shift_scale(
        self,
        shift: np.ndarray,
        scale: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply validated shift-scale parameters to sampled mechanism outputs."""
        n_nodes = self.dag.number_of_nodes()
        samples = rng.normal(size=(n_samples, n_nodes))
        if not np.array_equal(
            self.noise_variances,
            np.ones(n_nodes, dtype=float),
        ):
            samples *= self.noise_standard_deviations

        for node in nx.lexicographical_topological_sort(self.dag):
            parents = sorted(self.dag.predecessors(node))
            parent_contribution = np.zeros(n_samples, dtype=float)
            if parents:
                parent_contribution = (
                    samples[:, parents] @ self.coefficients[parents, node]
                )
            samples[:, node] = (
                shift[node] + scale[node] * parent_contribution + samples[:, node]
            )

        return samples
