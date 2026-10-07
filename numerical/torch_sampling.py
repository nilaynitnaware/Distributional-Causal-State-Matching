"""Explicit-generator float64 Torch sampling for prepared linear SCMs."""

from __future__ import annotations

import heapq
from typing import Literal

import numpy as np
import torch

ArrayOrTensor = np.ndarray | torch.Tensor
SamplingMethod = Literal["topological", "dense"]

_MAX_TORCH_SEED = (1 << 63) - 1


def derive_torch_seed(seed: int | np.random.SeedSequence) -> int:
    """Derive one portable nonnegative Torch seed without global RNG state."""
    if isinstance(seed, np.random.SeedSequence):
        word = int(seed.generate_state(1, dtype=np.uint64)[0])
        return word % (_MAX_TORCH_SEED + 1)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer or numpy.random.SeedSequence.")
    value = int(seed)
    if not 0 <= value <= _MAX_TORCH_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_TORCH_SEED}, inclusive.")
    return value


def make_generator(
    device: torch.device,
    seed: int | np.random.SeedSequence,
) -> torch.Generator:
    """Create a device-local generator seeded from explicit caller state."""
    generator = torch.Generator(device=device)
    generator.manual_seed(derive_torch_seed(seed))
    return generator


def validate_generator(
    generator: torch.Generator,
    device: torch.device,
) -> None:
    """Require an explicit Torch generator on the computation device."""
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator.")
    generator_device = torch.device(generator.device)
    if generator_device.type != device.type:
        raise ValueError(f"generator must be on device {device.type}.")
    if device.type == "cuda":
        generator_index = (
            torch.cuda.current_device()
            if generator_device.index is None
            else generator_device.index
        )
        if generator_index != device.index:
            raise ValueError(f"generator must be on device {device}.")


def validate_sample_count(n_samples: int) -> None:
    """Validate a requested number of independent samples."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise TypeError("n_samples must be an integer.")
    if n_samples < 0:
        raise ValueError("n_samples must be nonnegative.")


def validate_sampling_method(method: str) -> SamplingMethod:
    """Validate the two explicitly benchmarked sampling algorithms."""
    if method not in ("topological", "dense"):
        raise ValueError("method must be 'topological' or 'dense'.")
    return method


def _derived_order(support: np.ndarray) -> np.ndarray:
    """Return a lexicographical topological order for one coefficient support."""
    n_nodes = support.shape[0]
    indegree = support.sum(axis=0, dtype=np.int64)
    ready = [int(node) for node in np.flatnonzero(indegree == 0)]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for child in np.flatnonzero(support[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, int(child))
    if len(order) != n_nodes:
        raise ValueError(
            "topological sampling requires acyclic nonzero coefficient support."
        )
    return np.asarray(order, dtype=np.int64)


def _orders_array(
    orders: ArrayOrTensor,
    *,
    n_nodes: int,
) -> np.ndarray:
    """Copy caller-provided topological orders to a validated integer array."""
    if isinstance(orders, np.ndarray):
        if not np.issubdtype(orders.dtype, np.integer):
            raise TypeError("topological_orders must contain integers.")
        array = np.array(orders, dtype=np.int64, copy=True)
    elif isinstance(orders, torch.Tensor):
        if (
            orders.dtype == torch.bool
            or orders.is_complex()
            or torch.is_floating_point(orders)
        ):
            raise TypeError("topological_orders must contain integers.")
        array = orders.detach().cpu().numpy().astype(np.int64, copy=True)
    else:
        raise TypeError("topological_orders must be a numpy.ndarray or torch.Tensor.")
    if array.ndim == 1:
        array = array[None, :]
    elif array.ndim != 2:
        raise ValueError("topological_orders must have shape (p,) or (batch, p).")
    if array.shape[-1] != n_nodes:
        raise ValueError("topological order dimension must agree with the SCM.")
    if array.shape[0] == 0:
        raise ValueError("topological order batch must contain at least one row.")
    expected = np.arange(n_nodes, dtype=np.int64)
    if any(not np.array_equal(np.sort(row), expected) for row in array):
        raise ValueError(
            "each topological order must be a permutation of all node indices."
        )
    return array


def prepare_topological_structure(
    coefficients: torch.Tensor,
    topological_orders: ArrayOrTensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded sparse-parent tensors in topological-position order."""
    coefficient_array = coefficients.detach().cpu().numpy()
    coefficient_batch_size, n_nodes, _ = coefficient_array.shape
    if topological_orders is None:
        order_array = np.stack(
            [
                _derived_order(coefficient_array[index] != 0.0)
                for index in range(coefficient_batch_size)
            ]
        )
    else:
        order_array = _orders_array(topological_orders, n_nodes=n_nodes)

    if order_array.shape[0] not in (1, coefficient_batch_size):
        raise ValueError(
            "topological order batch size must be one or equal to the "
            "coefficient batch size."
        )
    structure_batch_size = coefficient_batch_size
    coefficients_expanded = np.broadcast_to(
        coefficient_array,
        (structure_batch_size, n_nodes, n_nodes),
    )
    orders_expanded = np.broadcast_to(
        order_array,
        (structure_batch_size, n_nodes),
    ).copy()

    parent_lists: list[list[np.ndarray]] = []
    maximum_parent_count = 0
    for batch_index in range(structure_batch_size):
        matrix = coefficients_expanded[batch_index]
        order = orders_expanded[batch_index]
        positions = np.empty(n_nodes, dtype=np.int64)
        positions[order] = np.arange(n_nodes, dtype=np.int64)
        batch_parents = []
        for position, node in enumerate(order):
            parents = np.flatnonzero(matrix[:, node] != 0.0).astype(
                np.int64,
                copy=False,
            )
            if np.any(positions[parents] >= position):
                raise ValueError(
                    "topological_orders must place every nonzero parent before "
                    "its child."
                )
            batch_parents.append(parents)
            maximum_parent_count = max(maximum_parent_count, parents.size)
        parent_lists.append(batch_parents)

    parent_indices = np.zeros(
        (structure_batch_size, n_nodes, maximum_parent_count),
        dtype=np.int64,
    )
    parent_coefficients = np.zeros(
        (structure_batch_size, n_nodes, maximum_parent_count),
        dtype=np.float64,
    )
    for batch_index, batch_parents in enumerate(parent_lists):
        matrix = coefficients_expanded[batch_index]
        order = orders_expanded[batch_index]
        for position, parents in enumerate(batch_parents):
            count = parents.size
            if count:
                node = order[position]
                parent_indices[batch_index, position, :count] = parents
                parent_coefficients[batch_index, position, :count] = matrix[
                    parents,
                    node,
                ]

    device = coefficients.device
    return (
        torch.tensor(orders_expanded, dtype=torch.long, device=device),
        torch.tensor(parent_indices, dtype=torch.long, device=device),
        torch.tensor(
            parent_coefficients,
            dtype=torch.float64,
            device=device,
        ),
    )


def sample_from_noise(
    coefficients: torch.Tensor,
    identity: torch.Tensor,
    topological_orders: torch.Tensor | None,
    parent_indices: torch.Tensor | None,
    parent_coefficients: torch.Tensor | None,
    shift: torch.Tensor,
    scale: torch.Tensor,
    noise: torch.Tensor,
    *,
    method: SamplingMethod,
) -> torch.Tensor:
    """Transform resident standard-normal noise using one sampling algorithm."""
    batch_size, n_samples, n_nodes = noise.shape
    coefficient_batch = coefficients.expand(batch_size, -1, -1)
    shift_batch = shift.expand(batch_size, -1)
    scale_batch = scale.expand(batch_size, -1)

    if method == "dense":
        system_matrix = identity.expand(batch_size, -1, -1) - scale_batch.unsqueeze(
            -1
        ) * coefficient_batch.transpose(-1, -2)
        right_hand_side = shift_batch.unsqueeze(1) + noise
        right_hand_side = right_hand_side.transpose(-1, -2)
        if topological_orders is None:
            return torch.linalg.solve(
                system_matrix,
                right_hand_side,
            ).transpose(-1, -2)
        orders = topological_orders.expand(batch_size, -1)
        row_index = orders.unsqueeze(-1).expand(-1, -1, n_nodes)
        ordered_system = system_matrix.gather(1, row_index)
        column_index = orders.unsqueeze(1).expand(-1, n_nodes, -1)
        ordered_system = ordered_system.gather(2, column_index)
        ordered_right_hand_side = right_hand_side.gather(
            1,
            orders.unsqueeze(-1).expand(-1, -1, n_samples),
        )
        ordered_samples = torch.linalg.solve_triangular(
            ordered_system,
            ordered_right_hand_side,
            upper=False,
        )
        samples = torch.empty_like(ordered_samples)
        samples.scatter_(
            1,
            orders.unsqueeze(-1).expand(-1, -1, n_samples),
            ordered_samples,
        )
        return samples.transpose(-1, -2)

    if (
        topological_orders is None
        or parent_indices is None
        or parent_coefficients is None
    ):
        raise ValueError(
            "topological sampling requires a prepared topological structure."
        )
    orders = topological_orders.expand(batch_size, -1)
    parents = parent_indices.expand(batch_size, -1, -1)
    weights = parent_coefficients.expand(batch_size, -1, -1)
    samples = noise.clone()
    for position in range(n_nodes):
        nodes = orders[:, position]
        node_index = nodes[:, None, None].expand(-1, n_samples, 1)
        node_noise = noise.gather(2, node_index).squeeze(-1)
        position_parents = parents[:, position, :]
        parent_index = position_parents[:, None, :].expand(-1, n_samples, -1)
        parent_values = samples.gather(2, parent_index)
        parent_contribution = (
            parent_values * weights[:, position, :].unsqueeze(1)
        ).sum(dim=-1)
        node_shift = shift_batch.gather(1, nodes[:, None])
        node_scale = scale_batch.gather(1, nodes[:, None])
        values = node_shift + node_scale * parent_contribution + node_noise
        samples.scatter_(2, node_index, values.unsqueeze(-1))
    return samples
