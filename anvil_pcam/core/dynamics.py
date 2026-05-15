"""Modern Hopfield-style retrieval dynamics with precision steering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anvil_pcam.core.memory import DIMENSION, PatternMemoryStore, normalize
from anvil_pcam.core.precision import predict_precision, precision_summary


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / (np.sum(exp) + 1e-12)


def logsumexp(values: np.ndarray) -> float:
    m = float(np.max(values))
    return m + float(np.log(np.sum(np.exp(values - m)) + 1e-12))


@dataclass
class RetrievalConfig:
    beta: float = 14.0
    iterations: int = 18
    step_size: float = 0.74
    convergence_tol: float = 1e-4


class PCAMEngine:
    """Precision-controlled associative memory retrieval engine."""

    def __init__(self, store: PatternMemoryStore | None = None, config: RetrievalConfig | None = None):
        self.store = store or PatternMemoryStore()
        self.config = config or RetrievalConfig()

    def energy(self, state: np.ndarray, precision: np.ndarray) -> float:
        """Modern Hopfield free-energy surrogate under diagonal precision Π."""
        xi = normalize(state)
        p = np.asarray(precision, dtype=np.float64)
        X = self.store.matrix
        weighted_scores = self.config.beta * ((X * p) @ xi)
        attraction = -logsumexp(weighted_scores) / self.config.beta
        quadratic = 0.5 * float(np.mean(p * (xi ** 2)))
        return float(attraction + 0.08 * quadratic)

    def retrieve(
        self,
        corrupted_query: np.ndarray,
        precision: np.ndarray | None = None,
        adaptive: bool = False,
        target_id: str | None = None,
    ) -> dict:
        """Run iterative attractor convergence and return a full trace."""
        state = normalize(np.asarray(corrupted_query, dtype=np.float64))
        if state.shape != (DIMENSION,):
            raise ValueError(f"corrupted_query must have shape {(DIMENSION,)}, got {state.shape}")

        X = self.store.matrix
        precision_vec = np.ones(DIMENSION, dtype=np.float64) if precision is None else np.asarray(precision, dtype=np.float64)
        trace = []
        precision_trace = []
        previous_energy: float | None = None

        for iteration in range(self.config.iterations + 1):
            if adaptive:
                current_prediction = predict_precision(state)
                precision_vec = 0.72 * precision_vec + 0.28 * current_prediction
                precision_vec = precision_vec / (precision_vec.mean() + 1e-12)

            scores = (X * precision_vec) @ state
            attention = softmax(self.config.beta * scores)
            candidate = normalize(attention @ X)
            energy = self.energy(state, precision_vec)
            nearest, nearest_score = self.store.nearest(state)
            target_score = None
            if target_id:
                target = self.store.get(target_id)
                target_score = float(target.vector @ state)

            delta = 0.0 if previous_energy is None else previous_energy - energy
            trace.append({
                "iteration": iteration,
                "energy": energy,
                "energyDrop": float(delta),
                "state": state.round(5).tolist(),
                "attention": attention.round(5).tolist(),
                "topId": nearest.id,
                "topLabel": nearest.label,
                "topScore": nearest_score,
                "targetScore": target_score,
                "residualNorm": float(np.linalg.norm(candidate - state)),
            })
            precision_trace.append(precision_vec.round(5).tolist())

            if iteration == self.config.iterations:
                break

            alpha = np.clip(self.config.step_size * np.sqrt(np.clip(precision_vec, 0.1, 10.0)), 0.06, 0.96)
            next_state = normalize(state + alpha * (candidate - state))
            if np.linalg.norm(next_state - state) < self.config.convergence_tol:
                state = next_state
                previous_energy = energy
                continue
            state = next_state
            previous_energy = energy

        final_pattern, final_score = self.store.nearest(state)
        final_target_score = None
        if target_id:
            final_target_score = float(self.store.get(target_id).vector @ state)

        return {
            "mode": "adaptive_precision" if adaptive else "identity_precision",
            "targetId": target_id,
            "finalId": final_pattern.id,
            "finalLabel": final_pattern.label,
            "finalScore": final_score,
            "finalTargetScore": final_target_score,
            "success": bool(target_id is not None and final_pattern.id == target_id),
            "iterations": len(trace) - 1,
            "trace": trace,
            "precisionTrace": precision_trace,
            "precisionSummary": precision_summary(precision_vec),
        }
