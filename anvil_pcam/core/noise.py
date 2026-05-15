"""Noise and masking operators for Anvil PCAM retrieval probes."""

from __future__ import annotations

import numpy as np

from anvil_pcam.core.memory import DIMENSION, normalize


def corrupt_pattern(
    pattern: np.ndarray,
    gaussian_sigma: float = 0.58,
    mask_fraction: float = 0.28,
    seed: int | None = None,
    structured_scale: float = 0.0,
    distractor: np.ndarray | None = None,
    distractor_mix: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Apply masking, Gaussian noise, structured noise, and distractor mixing."""
    vector = np.asarray(pattern, dtype=np.float64)
    if vector.shape != (DIMENSION,):
        raise ValueError(f"pattern must have shape {(DIMENSION,)}, got {vector.shape}")

    rng = np.random.default_rng(seed)
    mask_fraction = float(np.clip(mask_fraction, 0.0, 0.9))
    mask_count = int(round(mask_fraction * DIMENSION))
    mask = np.ones(DIMENSION, dtype=np.float64)
    if mask_count:
        masked_idx = rng.choice(DIMENSION, size=mask_count, replace=False)
        mask[masked_idx] = 0.0
    else:
        masked_idx = np.array([], dtype=int)

    # Interpret sigma as approximate vector-level noise energy rather than
    # per-coordinate variance; otherwise a 64D cue is dominated by noise.
    gaussian = rng.normal(0.0, gaussian_sigma / np.sqrt(DIMENSION), DIMENSION)

    dims = np.arange(DIMENSION, dtype=np.float64)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    structured = structured_scale * (
        0.65 * np.sin(dims / 3.7 + phase) +
        0.35 * np.cos(dims / 8.5 - phase)
    ) / np.sqrt(DIMENSION)

    distractor_mix = float(np.clip(distractor_mix, 0.0, 0.92))
    if distractor is None or distractor_mix <= 0.0:
        distractor_vec = np.zeros(DIMENSION, dtype=np.float64)
        distractor_id = None
    else:
        distractor_vec = normalize(np.asarray(distractor, dtype=np.float64))
        if distractor_vec.shape != (DIMENSION,):
            raise ValueError(f"distractor must have shape {(DIMENSION,)}, got {distractor_vec.shape}")
        distractor_id = "provided"

    corrupted = normalize((1.0 - distractor_mix) * vector * mask + distractor_mix * distractor_vec + gaussian + structured)
    metadata = {
        "gaussianSigma": float(gaussian_sigma),
        "maskFraction": mask_fraction,
        "structuredScale": float(structured_scale),
        "distractorMix": distractor_mix,
        "distractorId": distractor_id,
        "maskedDimensions": sorted(int(i) for i in masked_idx.tolist()),
        "noiseNorm": float(np.linalg.norm(gaussian)),
        "structuredNorm": float(np.linalg.norm(structured)),
        "cosineToClean": float(normalize(vector) @ corrupted),
    }
    return corrupted, metadata
