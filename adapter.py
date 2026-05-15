"""Base adapter interface for P-04 PCAM Precision Agent bench."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class Adapter(ABC):
    """Abstract base for PCAM precision-prediction adapters.

    Subclasses receive the stored patterns and model parameters at
    construction time and must implement ``predict_precision``.
    """

    @abstractmethod
    def __init__(self, stored_patterns: np.ndarray, model_params: dict):
        ...

    @abstractmethod
    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        """Return a positive precision vector of shape (N,) for the query."""
        ...
