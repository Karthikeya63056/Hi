import sys
import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns

seed = 42
X = make_patterns(K=16, N=64, seed=seed, n_clusters=4, intra_sim=0.5)
R = build_default_R(N=64, seed=seed)
model = PCAMModel(X, R)
N = 64

pattern = X[0]
a_star = model.find_equilibrium(pattern)
H = model.hessian(a_star)

# Test different gradient formulas
def test_opt(grad_func, name):
    log_pi = np.zeros(N)
    best_cond = float('inf')
    best_pi = np.ones(N)
    for _ in range(150):
        pi = np.exp(log_pi)
        pi = np.clip(pi, 0.1, 10.0)
        pi /= pi.mean()
        
        sqrt_pi = np.sqrt(pi)
        M = sqrt_pi[:, None] * H * sqrt_pi[None, :]
        M = 0.5 * (M + M.T)
        evals, evecs = np.linalg.eigh(M)
        evals_pos = evals[evals > 1e-9]
        if len(evals_pos) < 2: break
        
        cond = evals_pos.max() / evals_pos.min()
        if cond < best_cond:
            best_cond = cond
            best_pi = pi.copy()
            
        l_max, l_min = evals_pos.max(), evals_pos.min()
        idx_max = len(evals) - 1
        idx_min = len(evals) - len(evals_pos)
        v_max = evecs[:, idx_max]
        v_min = evecs[:, idx_min]
        
        grad = grad_func(v_max, v_min, l_max, l_min)
        log_pi -= 0.1 * grad
        log_pi -= log_pi.mean()
        
    return best_cond, best_pi

evals = np.linalg.eigvalsh(H)
evals_pos = evals[evals > 1e-9]
base_cond = evals_pos.max() / evals_pos.min()
print(f"Base cond: {base_cond:.4f}")

# User's formula
user_cond, user_pi = test_opt(lambda vmax, vmin, lmax, lmin: (vmax**2)/lmax - (vmin**2)/lmin, "User")
print(f"User formula cond: {user_cond:.4f}")

# Math derivation
math_cond, math_pi = test_opt(lambda vmax, vmin, lmax, lmin: vmax**2 - vmin**2, "Math")
print(f"Math formula cond: {math_cond:.4f}")

# Higher LR
math_cond_lr, _ = test_opt(lambda vmax, vmin, lmax, lmin: 10 * (vmax**2 - vmin**2), "Math 10x LR")
print(f"Math formula (10x LR) cond: {math_cond_lr:.4f}")

# With clipping
print(f"User pi range: {user_pi.min():.3f} to {user_pi.max():.3f}")
print(f"Math pi range: {math_pi.min():.3f} to {math_pi.max():.3f}")
