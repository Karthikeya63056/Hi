"""Quick smoke test for the PrecisionFlow Engine against the repo pattern bank."""
import sys
sys.path.insert(0, ".")

import numpy as np
from anvil_pcam.core.memory import PatternMemoryStore
from adapters.myteam import Engine

store = PatternMemoryStore()
X = store.matrix
params = {"beta": 1.0}
eng = Engine(X, params)
print(f"Loaded {eng.K} patterns, dim={eng.N}")

rng = np.random.default_rng(2404)
for i in range(eng.K):
    q = X[i] + rng.normal(0, 0.4, eng.N)
    pi = eng.predict_precision(q)
    assert pi.shape == (eng.N,), f"Bad shape: {pi.shape}"
    assert (pi > 0).all(), "Non-positive precision values"
    assert abs(pi.mean() - 1.0) < 0.01, f"Mean not 1: {pi.mean()}"
    print(f"  A{i+1:02d}: min={pi.min():.3f} max={pi.max():.3f} mean={pi.mean():.4f} aniso={pi.std()/pi.mean():.3f}")

print("\nAll 12 patterns passed.")
