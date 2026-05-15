"""Anvil PCAM Lab core: precision-controlled associative memory."""

from anvil_pcam.core.dynamics import PCAMEngine, RetrievalConfig
from anvil_pcam.core.evaluation import NOISE_PROFILES, benchmark_bank, evaluate_trial
from anvil_pcam.core.memory import DIMENSION, MemoryPattern, PatternMemoryStore
from anvil_pcam.core.noise import corrupt_pattern
from anvil_pcam.core.precision import predict_precision, precision_summary

__all__ = [
    "DIMENSION",
    "MemoryPattern",
    "NOISE_PROFILES",
    "PatternMemoryStore",
    "PCAMEngine",
    "RetrievalConfig",
    "benchmark_bank",
    "corrupt_pattern",
    "evaluate_trial",
    "predict_precision",
    "precision_summary",
]
