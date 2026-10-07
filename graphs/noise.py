"""Deterministic independent node-wise exogenous variance regimes."""

from __future__ import annotations

import hashlib

import numpy as np

NOISE_VARIANCE_REGIMES = (
    "homoskedastic",
    "heterogeneous_moderate",
    "heterogeneous_strong",
)


def validate_noise_variance_regime(regime: str) -> str:
    """Return a supported regime or fail loudly."""
    if not isinstance(regime, str):
        raise TypeError("noise variance regime must be a string.")
    if regime not in NOISE_VARIANCE_REGIMES:
        raise ValueError(
            f"unsupported noise variance regime {regime!r}; expected one of "
            f"{NOISE_VARIANCE_REGIMES}."
        )
    return regime


def derive_noise_variance_seed(base_seed: int) -> int:
    """Derive a dedicated variance seed without consuming another RNG stream."""
    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)):
        raise TypeError("base_seed must be an integer.")
    if int(base_seed) < 0:
        raise ValueError("base_seed must be nonnegative.")
    digest = hashlib.sha256(f"noise_variances:{int(base_seed)}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def generate_noise_variances(
    n_nodes: int,
    regime: str = "homoskedastic",
    seed: int | None = None,
) -> np.ndarray:
    """Generate a deterministic positive variance vector for one SCM.

    Heterogeneous regimes share the same centered latent vector for a fixed
    seed. Moderate and strong log variances are normalized respectively to
    maximum absolute values ``log(2)`` and ``log(4)``.
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int):
        raise TypeError("n_nodes must be an integer.")
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")
    regime = validate_noise_variance_regime(regime)
    if regime == "homoskedastic":
        return np.ones(n_nodes, dtype=float)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("heterogeneous noise regimes require an integer seed.")
    if int(seed) < 0:
        raise ValueError("noise variance seed must be nonnegative.")

    latent = np.random.default_rng(int(seed)).uniform(-1.0, 1.0, size=n_nodes)
    centered = latent - np.mean(latent)
    maximum = float(np.max(np.abs(centered)))
    if maximum <= np.finfo(float).eps:
        if n_nodes == 1:
            centered = np.zeros(1, dtype=float)
            maximum = 1.0
        else:
            centered = np.arange(n_nodes, dtype=float) - 0.5 * (n_nodes - 1)
            maximum = float(np.max(np.abs(centered)))
    log_bound = np.log(2.0 if regime == "heterogeneous_moderate" else 4.0)
    variances = np.exp(log_bound * centered / maximum)
    lower = 0.5 if regime == "heterogeneous_moderate" else 0.25
    upper = 2.0 if regime == "heterogeneous_moderate" else 4.0
    return np.clip(variances, lower, upper)


def noise_variance_fingerprint(noise_variances: np.ndarray) -> str:
    """Return a canonical SHA-256 digest of a realized variance vector."""
    normalized = np.ascontiguousarray(np.asarray(noise_variances, dtype="<f8"))
    if normalized.ndim != 1:
        raise ValueError("noise_variances must be one-dimensional.")
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def noise_variance_summary(noise_variances: np.ndarray) -> dict[str, object]:
    """Return JSON-compatible diagnostics for a realized variance vector."""
    values = np.asarray(noise_variances, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("noise_variances must be a nonempty vector.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("noise_variances must be finite and strictly positive.")
    return {
        "noise_variance_minimum": float(np.min(values)),
        "noise_variance_maximum": float(np.max(values)),
        "noise_variance_geometric_mean": float(np.exp(np.mean(np.log(values)))),
        "noise_variance_max_min_ratio": float(np.max(values) / np.min(values)),
        "noise_variance_fingerprint": noise_variance_fingerprint(values),
    }
