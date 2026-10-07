"""Internal helpers for immutable stored NumPy arrays."""

import numpy as np


def immutable_array(
    array: np.ndarray,
    *,
    dtype: object = None,
) -> np.ndarray:
    """Return a defensive C-order copy backed by immutable bytes.

    The returned array cannot have writeability re-enabled with
    ``setflags(write=True)`` because its ultimate buffer is a ``bytes`` object.
    """
    owned = np.array(array, dtype=dtype, copy=True, order="C")
    immutable_buffer = owned.tobytes(order="C")
    frozen = np.frombuffer(immutable_buffer, dtype=owned.dtype)
    return frozen.reshape(owned.shape)
