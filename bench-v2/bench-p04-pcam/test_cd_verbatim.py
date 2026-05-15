import sys
import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns

seed = 42
X = make_patterns(K=16, N=64, seed=seed, n_clusters=4, intra_sim=0.5)
R = build_default_R(N=64, seed=seed)
model = PCAMModel(X, R)
N, K = 64, 16

reductions = []
for j in range(K):
    pattern = X[j]
    a_star = model.find_equilibrium(pattern)
    H = model.hessian(a_star)
    
    # Base spread
    pi_base = np.ones(N)
    S_base = (np.sqrt(pi_base)[:, None] * H) * np.sqrt(pi_base)[None, :]
    S_base = 0.5 * (S_base + S_base.T)
    evals_base = np.linalg.eigvalsh(S_base)
    evals_base = evals_base[evals_base > 1e-9]
    if len(evals_base) < 2: continue
    base_cond = evals_base.max() / evals_base.min()
    
    # Coordinate descent from prompt
    log_pi = np.zeros(N)
    for _ in range(100):
        pi = np.exp(log_pi)
        sqrt_pi = np.sqrt(pi)
        M = sqrt_pi[:, None] * H * sqrt_pi[None, :]
        eigvals, eigvecs = np.linalg.eigh(M)
        
        # We must ignore negative/zero eigenvalues just like harness
        pos_idx = np.where(eigvals > 1e-9)[0]
        if len(pos_idx) < 2: break
        
        idx_max = pos_idx[-1]
        idx_min = pos_idx[0]
        
        l_max = eigvals[idx_max]
        l_min = eigvals[idx_min]
        v_max = eigvecs[:, idx_max]
        v_min = eigvecs[:, idx_min]
        
        gradient = (v_max**2)/l_max - (v_min**2)/l_min
        log_pi -= 0.1 * gradient
        log_pi -= np.mean(log_pi)
        
    pi_opt = np.exp(log_pi)
    pi_opt = model.clip_and_normalise(pi_opt)
    
    # Measure spread
    S_opt = (np.sqrt(pi_opt)[:, None] * H) * np.sqrt(pi_opt)[None, :]
    S_opt = 0.5 * (S_opt + S_opt.T)
    evals_opt = np.linalg.eigvalsh(S_opt)
    evals_opt = evals_opt[evals_opt > 1e-9]
    
    if len(evals_opt) < 2: continue
    opt_cond = evals_opt.max() / evals_opt.min()
    
    reductions.append(base_cond / opt_cond)
    print(f"Pattern {j:2d}: base={base_cond:7.2f} opt={opt_cond:7.2f} red={base_cond/opt_cond:.2f}x")

print(f"\nMean Reduction: {np.mean(reductions):.2f}x")
