"""64-dimensional memory attractor storage for Anvil PCAM Lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

DIMENSION = 64


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return an L2-normalized vector, preserving zero vectors safely."""
    arr = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        return arr.copy()
    return arr / norm


@dataclass(frozen=True)
class MemoryPattern:
    """A stored Hopfield attractor pattern."""

    id: str
    label: str
    family: str
    vector: np.ndarray
    phase: float

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "family": self.family,
            "phase": self.phase,
            "vector": self.vector.round(5).tolist(),
        }


class PatternMemoryStore:
    """In-memory store of K fixed 64D attractor patterns.

    The default bank is deterministic so the demo is reproducible, while still
    producing nontrivial relationships between attractors.
    """

    def __init__(self, patterns: Iterable[MemoryPattern] | None = None):
        self.patterns = list(patterns) if patterns is not None else make_default_patterns()
        if not self.patterns:
            raise ValueError("PatternMemoryStore requires at least one pattern")
        for pattern in self.patterns:
            if pattern.vector.shape != (DIMENSION,):
                raise ValueError(f"{pattern.id} has shape {pattern.vector.shape}, expected {(DIMENSION,)}")

    @property
    def matrix(self) -> np.ndarray:
        return np.stack([p.vector for p in self.patterns], axis=0)

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.patterns]

    def get(self, pattern_id: str) -> MemoryPattern:
        for pattern in self.patterns:
            if pattern.id == pattern_id:
                return pattern
        raise KeyError(f"Unknown memory attractor: {pattern_id}")

    def nearest(self, vector: np.ndarray) -> tuple[MemoryPattern, float]:
        query = normalize(np.asarray(vector, dtype=np.float64))
        scores = self.matrix @ query
        idx = int(np.argmax(scores))
        return self.patterns[idx], float(scores[idx])

    def graph(self, threshold: float = 0.18) -> dict:
        """Return nodes and similarity edges for the memory graph view."""
        X = self.matrix
        sims = X @ X.T
        nodes = []
        for idx, pattern in enumerate(self.patterns):
            nodes.append({
                "id": pattern.id,
                "label": pattern.label,
                "family": pattern.family,
                "phase": pattern.phase,
                "energyBias": float(-np.log1p(np.exp(10.0 * sims[idx].max())) / 10.0),
            })

        edges = []
        for i in range(len(self.patterns)):
            for j in range(i + 1, len(self.patterns)):
                weight = float(sims[i, j])
                if abs(weight) >= threshold:
                    edges.append({
                        "source": self.patterns[i].id,
                        "target": self.patterns[j].id,
                        "weight": weight,
                    })
        return {"nodes": nodes, "edges": edges}


def make_default_patterns() -> list[MemoryPattern]:
    """Construct a deterministic research-demo attractor bank."""
    rng = np.random.default_rng(2404)
    dims = np.arange(DIMENSION, dtype=np.float64)
    families = [
        ("phase-lattice", "Phase Lattice Attractor"),
        ("spectral-gate", "Spectral Gate Attractor"),
        ("causal-ridge", "Causal Ridge Attractor"),
        ("topology-code", "Topology Code Attractor"),
        ("control-plane", "Control Plane Attractor"),
        ("basin-shear", "Basin Shear Attractor"),
        ("noise-lock", "Noise Lock Attractor"),
        ("curvature-map", "Curvature Map Attractor"),
        ("anisotropy-band", "Anisotropy Band Attractor"),
        ("energy-well", "Energy Well Attractor"),
        ("retrieval-helix", "Retrieval Helix Attractor"),
        ("precision-kernel", "Precision Kernel Attractor"),
    ]

    patterns: list[MemoryPattern] = []
    for idx, (family, label) in enumerate(families):
        phase = (idx + 1) * np.pi / 13.0
        harmonic = np.sin((idx % 5 + 1) * dims / 7.0 + phase)
        carrier = np.cos((idx % 7 + 2) * dims / 11.0 - phase * 0.7)
        block = np.zeros(DIMENSION)
        start = (idx * 5) % DIMENSION
        block[start:start + 10] = 1.15
        if start + 10 > DIMENSION:
            block[:(start + 10) % DIMENSION] = 1.15
        sparse = rng.normal(0.0, 0.33, DIMENSION)
        vector = normalize(0.58 * harmonic + 0.42 * carrier + 0.75 * block + sparse)
        patterns.append(MemoryPattern(
            id=f"A{idx + 1:02d}",
            label=label,
            family=family,
            vector=vector,
            phase=float(phase),
        ))
    return patterns
