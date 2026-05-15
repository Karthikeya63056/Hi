import sys
import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns
from scipy.optimize import minimize

seed = 42
X = make_patterns(K=16, N=64, seed=seed, n_clusters=4, intra_sim=0.5)
R = build_default_R(N=64, seed=seed)
model = PCAMModel(X, R)
N, K = 64, 16

print("=== SciPy L-BFGS-B Multiple Restarts ===")

reductions = []
for j in range(5):
    pattern = X[j]
    a_star = model.find_equilibrium(pattern)
    H = model.hessian(a_star)
    
    pi_base = np.ones(N)
    S_base = (np.sqrt(pi_base)[:, None] * H) * np.sqrt(pi_base)[None, :]
    S_base = 0.5 * (S_base + S_base.T)
    evals = np.linalg.eigvalsh(S_base)
    evals_pos = evals[evals > 1e-9]
    if len(evals_pos) < 2: continue
    base_cond = evals_pos.max() / evals_pos.min()

    def obj(pi_raw):
        pi = model.clip_and_normalise(pi_raw)
        sqrt_pi = np.sqrt(pi)
        M = sqrt_pi[:, None] * H * sqrt_pi[None, :]
        M = 0.5 * (M + M.T)
        e = np.linalg.eigvalsh(M)
        e_pos = e[e > 1e-9]
        if len(e_pos) < 2: return 1e6
        return e_pos.max() / e_pos.min()

    best_cond = base_cond
    best_pi = np.ones(N)
    
    # Restarts
    for i in range(10):
        if i == 0:
            x0 = np.ones(N)
        else:
            x0 = np.exp(np.random.randn(N))
            x0 = model.clip_and_normalise(x0)
            
        res = minimize(obj, x0, method='L-BFGS-B', bounds=[(0.01, 100)]*N, options={'maxiter': 200})
        if res.fun < best_cond:
            best_cond = res.fun
            best_pi = model.clip_and_normalise(res.x)
            
    reductions.append(base_cond / best_cond)
    print(f"Pattern {j:2d}: base={base_cond:7.2f} best={best_cond:7.2f} red={base_cond/best_cond:.2f}x")

print(f"\nMean Reduction: {np.mean(reductions):.2f}x")
