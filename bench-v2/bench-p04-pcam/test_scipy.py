import sys
import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns
from scipy.optimize import minimize

seed = 42
X = make_patterns(K=16, N=64, seed=seed, n_clusters=4, intra_sim=0.5)
R = build_default_R(N=64, seed=seed)
model = PCAMModel(X, R)
N = 64

pattern = X[2]
a_star = model.find_equilibrium(pattern)
H = model.hessian(a_star)

evals = np.linalg.eigvalsh(H)
evals_pos = evals[evals > 1e-9]
base_cond = evals_pos.max() / evals_pos.min()
print(f"Base cond: {base_cond:.4f}")

def obj(log_pi):
    pi = np.exp(log_pi)
    pi = np.clip(pi, 0.1, 10.0)
    pi /= pi.mean()
    sqrt_pi = np.sqrt(pi)
    M = sqrt_pi[:, None] * H * sqrt_pi[None, :]
    M = 0.5 * (M + M.T)
    evals = np.linalg.eigvalsh(M)
    evals_pos = evals[evals > 1e-9]
    if len(evals_pos) < 2: return 1e6
    return evals_pos.max() / evals_pos.min()

res = minimize(obj, np.zeros(N), method='L-BFGS-B', options={'maxiter': 500})
print(f"SciPy L-BFGS-B cond: {res.fun:.4f} (Reduction: {base_cond/res.fun:.2f}x)")
print(f"Iterations: {res.nit}")
