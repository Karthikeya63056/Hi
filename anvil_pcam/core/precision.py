"""Adaptive anisotropic precision prediction for Anvil PCAM Lab."""

from __future__ import annotations

import numpy as np

from anvil_pcam.core.memory import DIMENSION, PatternMemoryStore, normalize

DEFAULT_STORE = PatternMemoryStore()


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / (np.sum(exp) + 1e-12)


def _project_mean_one_with_bounds(values: np.ndarray, low: float = 0.1, high: float = 10.0) -> np.ndarray:
    """Scale values so clipped output has mean 1 and stays within bounds."""
    raw = np.maximum(np.asarray(values, dtype=np.float64), 1e-9)
    lo, hi = 1e-6, 1e6
    for _ in range(80):
        mid = (lo + hi) * 0.5
        mean = np.clip(raw * mid, low, high).mean()
        if mean < 1.0:
            lo = mid
        else:
            hi = mid
    projected = np.clip(raw * ((lo + hi) * 0.5), low, high)
    return projected


def predict_precision(corrupted_query):
    """
    corrupted_query : ndarray (64,)
    returns         : ndarray (64,) positive precision values
    """
    q = np.asarray(corrupted_query, dtype=np.float64)
    if q.shape != (DIMENSION,):
        raise ValueError(f"corrupted_query must have shape {(DIMENSION,)}, got {q.shape}")

    query = normalize(q)
    X = DEFAULT_STORE.matrix

    sims = X @ query
    weights = _softmax(8.0 * sims)
    expected = weights @ X

    local_variance = weights @ ((X - expected) ** 2) + 1e-4
    residual = np.abs(query - expected)
    residual_z = residual / np.sqrt(local_variance + 1e-4)

    agreement = np.exp(-0.72 * residual_z)
    basin_stiffness = 1.0 / np.sqrt(local_variance + 0.025)
    basin_stiffness /= basin_stiffness.mean() + 1e-12

    magnitude = np.abs(query)
    magnitude /= np.median(magnitude) + 1e-4
    magnitude = np.clip(magnitude, 0.2, 3.0)

    outlier_gate = 1.0 / (1.0 + np.exp(1.35 * (residual_z - 2.15)))
    raw = 0.30 + 1.05 * agreement + 0.45 * basin_stiffness + 0.25 * magnitude
    raw *= 0.45 + 0.55 * outlier_gate

    precision = _project_mean_one_with_bounds(raw, 0.1, 10.0)
    return precision.astype(np.float64)


def precision_summary(precision: np.ndarray) -> dict:
    p = np.asarray(precision, dtype=np.float64)
    participation = float((p.sum() ** 2) / (np.sum(p ** 2) + 1e-12))
    return {
        "mean": float(p.mean()),
        "min": float(p.min()),
        "max": float(p.max()),
        "std": float(p.std()),
        "anisotropy": float(p.std() / (p.mean() + 1e-12)),
        "participationRatio": participation / float(DIMENSION),
    }
