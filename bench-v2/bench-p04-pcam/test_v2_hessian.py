"""
Diagnostic script for Bench v2: Analyze Hessian structure at true equilibria.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns

seed = 42
X = make_patterns(K=16, N=64, seed=seed, n_clusters=4, intra_sim=0.5)
R = build_default_R(N=64, seed=seed)
model = PCAMModel(X, R)
N, K = 64, 16

print("=== Bench v2: Hessian at True Equilibria ===\n")

for j in range(min(5, K)):
    pattern = X[j]
    # 1. True equilibrium
    a_star = model.find_equilibrium(pattern)
    
    # 2. Hessian at true equilibrium
    H = model.hessian(a_star)
    
    H_diag = np.diag(H)
    eigs, eigvecs = np.linalg.eigh(H)
    eigs_pos = eigs[eigs > 1e-9]
    
    if len(eigs_pos) > 0:
        cond = eigs_pos.max() / eigs_pos.min()
    else:
        cond = float('inf')
        
    dom_eigvec = eigvecs[:, -1]
    all_ones = np.ones(N) / np.sqrt(N)
    cos_sim = np.abs(np.dot(dom_eigvec, all_ones))
    
    print(f"Pattern {j}:")
    print(f"  H_diag: min={H_diag.min():.6f} max={H_diag.max():.6f} std={H_diag.std():.6f}")
    print(f"  Eigenvalues: min={eigs_pos.min():.6f} max={eigs_pos.max():.6f} cond={cond:.4f}")
    print(f"  Dominant eigvec cosine with all-ones: {cos_sim:.6f}")
    
    # Check softmax distribution
    s = model._softmax(a_star)
    print(f"  Softmax max={s.max():.4f}, >0.1 count={np.sum(s > 0.1)}")
    print()

print("=== Coordinate Descent Optimization ===")

for j in range(min(3, K)):
    pattern = X[j]
    a_star = model.find_equilibrium(pattern)
    H = model.hessian(a_star)
    
    log_pi = np.zeros(N)
    best_pi = np.ones(N)
    best_cond = eigs_pos.max() / eigs_pos.min()
    
    # Optimize log_pi
    for i in range(150):
        pi = np.exp(log_pi)
        pi = np.clip(pi, 0.1, 10.0)
        pi /= pi.mean()
        
        sqrt_pi = np.sqrt(pi)
        M = sqrt_pi[:, None] * H * sqrt_pi[None, :]
        M = 0.5 * (M + M.T)
        evals, evecs = np.linalg.eigh(M)
        evals_pos = evals[evals > 1e-9]
        
        if len(evals_pos) < 2:
            break
            
        l_max, l_min = evals_pos.max(), evals_pos.min()
        v_max, v_min = evecs[:, -1], evecs[:, 0]
        
        current_cond = l_max / l_min
        if current_cond < best_cond:
            best_cond = current_cond
            best_pi = pi.copy()
            
        grad = (v_max**2)/l_max - (v_min**2)/l_min
        log_pi -= 0.1 * grad
        log_pi -= log_pi.mean()
        log_pi = np.clip(log_pi, -np.log(10), np.log(10))

    base_evals = np.linalg.eigvalsh(H)
    base_evals = base_evals[base_evals > 1e-9]
    base_cond = base_evals.max() / base_evals.min()
    
    reduction = base_cond / best_cond if best_cond > 0 else 0
    print(f"Pattern {j}: base={base_cond:.4f} best={best_cond:.4f} reduction={reduction:.4f}x")
    print(f"  Best pi std={best_pi.std():.4f}")
