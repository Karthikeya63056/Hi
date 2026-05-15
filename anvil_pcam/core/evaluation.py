"""Evaluation helpers for baseline vs precision-controlled retrieval."""

from __future__ import annotations

import numpy as np

from anvil_pcam.core.dynamics import PCAMEngine
from anvil_pcam.core.noise import corrupt_pattern
from anvil_pcam.core.precision import predict_precision, precision_summary


NOISE_PROFILES = {
    "demo": {
        "label": "Demo",
        "description": "Stable default corruption for clear convergence demos.",
        "structured_scale": 0.10,
        "distractor_mix": 0.00,
    },
    "stress": {
        "label": "Stress",
        "description": "Structured noise and mild attractor interference.",
        "structured_scale": 0.34,
        "distractor_mix": 0.18,
    },
    "adversarial": {
        "label": "Adversarial",
        "description": "Hard basin ambiguity with a similar distractor attractor.",
        "structured_scale": 0.52,
        "distractor_mix": 0.34,
    },
}


def convergence_speed(trace: list[dict], threshold: float = 0.92) -> int | None:
    """First iteration where the target cosine clears a stability threshold."""
    for point in trace:
        score = point.get("targetScore")
        if score is not None and score >= threshold:
            return int(point["iteration"])
    return None


def stability_score(trace: list[dict]) -> float:
    """Small residual and low late energy variance imply stable convergence."""
    if not trace:
        return 0.0
    tail = trace[-5:]
    residual = np.mean([p["residualNorm"] for p in tail])
    energies = np.array([p["energy"] for p in tail], dtype=np.float64)
    variance = float(np.std(energies))
    return float(1.0 / (1.0 + residual + 4.0 * variance))


def final_margin(engine: PCAMEngine, final_state: list[float], target_id: str) -> float:
    """Difference between target cosine and strongest non-target cosine."""
    state = np.asarray(final_state, dtype=np.float64)
    scores = engine.store.matrix @ state
    target_index = engine.store.ids.index(target_id)
    target_score = float(scores[target_index])
    non_target = np.delete(scores, target_index)
    return float(target_score - np.max(non_target))


def nearest_distractor(engine: PCAMEngine, pattern_id: str):
    """Choose the most similar non-target attractor as an adversarial distractor."""
    pattern = engine.store.get(pattern_id)
    scores = engine.store.matrix @ pattern.vector
    target_index = engine.store.ids.index(pattern_id)
    scores[target_index] = -np.inf
    idx = int(np.argmax(scores))
    return engine.store.patterns[idx]


def retrieval_verdict(metrics: dict) -> dict:
    adaptive = metrics["adaptive"]
    baseline = metrics["baseline"]
    if adaptive["accuracy"] >= 1.0 and adaptive["margin"] > 0.18:
        status = "locked"
        message = "Adaptive precision stabilized a confident attractor lock."
    elif adaptive["accuracy"] >= baseline["accuracy"] and adaptive["finalTargetScore"] >= baseline["finalTargetScore"]:
        status = "competitive"
        message = "Adaptive precision matched or improved identity precision under this corruption."
    elif adaptive["accuracy"] >= 1.0:
        status = "fragile"
        message = "Correct attractor recovered, but the basin margin is narrow."
    else:
        status = "failed"
        message = "Retrieval fell into the wrong basin; reduce corruption or inspect Π."
    return {"status": status, "message": message}


def evaluate_trial(
    engine: PCAMEngine,
    pattern_id: str,
    gaussian_sigma: float = 0.58,
    mask_fraction: float = 0.28,
    seed: int | None = None,
    profile: str = "demo",
) -> dict:
    """Run one noisy retrieval trial with identity and adaptive precision."""
    pattern = engine.store.get(pattern_id)
    profile_cfg = NOISE_PROFILES.get(profile, NOISE_PROFILES["demo"])
    distractor = nearest_distractor(engine, pattern_id)
    corrupted, noise = corrupt_pattern(
        pattern.vector,
        gaussian_sigma,
        mask_fraction,
        seed,
        structured_scale=profile_cfg["structured_scale"],
        distractor=distractor.vector,
        distractor_mix=profile_cfg["distractor_mix"],
    )
    noise["profile"] = profile
    noise["profileLabel"] = profile_cfg["label"]
    noise["profileDescription"] = profile_cfg["description"]
    noise["distractorId"] = distractor.id if profile_cfg["distractor_mix"] > 0 else None
    noise["distractorLabel"] = distractor.label if profile_cfg["distractor_mix"] > 0 else None
    precision = predict_precision(corrupted)

    baseline = engine.retrieve(corrupted, precision=np.ones_like(precision), adaptive=False, target_id=pattern_id)
    adaptive = engine.retrieve(corrupted, precision=precision, adaptive=True, target_id=pattern_id)

    baseline_speed = convergence_speed(baseline["trace"])
    adaptive_speed = convergence_speed(adaptive["trace"])
    baseline_margin = final_margin(engine, baseline["trace"][-1]["state"], pattern_id)
    adaptive_margin = final_margin(engine, adaptive["trace"][-1]["state"], pattern_id)
    metrics = {
        "baseline": {
            "accuracy": 1.0 if baseline["success"] else 0.0,
            "convergenceSpeed": baseline_speed,
            "finalTargetScore": baseline["finalTargetScore"],
            "stability": stability_score(baseline["trace"]),
            "margin": baseline_margin,
            "energyDrop": float(baseline["trace"][0]["energy"] - baseline["trace"][-1]["energy"]),
        },
        "adaptive": {
            "accuracy": 1.0 if adaptive["success"] else 0.0,
            "convergenceSpeed": adaptive_speed,
            "finalTargetScore": adaptive["finalTargetScore"],
            "stability": stability_score(adaptive["trace"]),
            "anisotropy": precision_summary(precision)["anisotropy"],
            "margin": adaptive_margin,
            "energyDrop": float(adaptive["trace"][0]["energy"] - adaptive["trace"][-1]["energy"]),
        },
    }
    if baseline_speed is not None and adaptive_speed is not None:
        metrics["speedup"] = float((baseline_speed + 1) / (adaptive_speed + 1))
    else:
        metrics["speedup"] = None

    return {
        "target": pattern.as_dict(),
        "corruptedQuery": corrupted.round(5).tolist(),
        "noise": noise,
        "initialPrecision": precision.round(5).tolist(),
        "initialPrecisionSummary": precision_summary(precision),
        "baseline": baseline,
        "adaptive": adaptive,
        "metrics": metrics,
        "verdict": retrieval_verdict(metrics),
        "graph": engine.store.graph(),
    }


def benchmark_bank(engine: PCAMEngine, seed: int = 7, profile: str = "stress") -> dict:
    """Small deterministic bank-level robustness check."""
    rows = []
    for idx, pattern in enumerate(engine.store.patterns):
        trial = evaluate_trial(engine, pattern.id, seed=seed + idx, profile=profile)
        rows.append({
            "id": pattern.id,
            "label": pattern.label,
            "baselineSuccess": trial["baseline"]["success"],
            "adaptiveSuccess": trial["adaptive"]["success"],
            "baselineScore": trial["metrics"]["baseline"]["finalTargetScore"],
            "adaptiveScore": trial["metrics"]["adaptive"]["finalTargetScore"],
            "baselineMargin": trial["metrics"]["baseline"]["margin"],
            "adaptiveMargin": trial["metrics"]["adaptive"]["margin"],
            "anisotropy": trial["metrics"]["adaptive"]["anisotropy"],
        })
    return {
        "profile": profile,
        "profileDescription": NOISE_PROFILES.get(profile, NOISE_PROFILES["demo"])["description"],
        "trials": rows,
        "baselineAccuracy": float(np.mean([r["baselineSuccess"] for r in rows])),
        "adaptiveAccuracy": float(np.mean([r["adaptiveSuccess"] for r in rows])),
        "meanAdaptiveScore": float(np.mean([r["adaptiveScore"] for r in rows])),
        "meanBaselineScore": float(np.mean([r["baselineScore"] for r in rows])),
        "meanAdaptiveMargin": float(np.mean([r["adaptiveMargin"] for r in rows])),
        "meanBaselineMargin": float(np.mean([r["baselineMargin"] for r in rows])),
    }
