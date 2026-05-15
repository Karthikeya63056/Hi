import sys
import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns
from metrics import _symmetrised_spread

seed = 42
X = make_patterns(K=16, N=64, seed=seed, n_clusters=4, intra_sim=0.5)
R = build_default_R(N=64, seed=seed)
model = PCAMModel(X, R)
N, K = 64, 16

print("=== Spread test for Class-Conditional variants ===")
for p in [0.5, 1.0, 2.0, 3.0, 5.0]:
    reductions = []
    for j in range(K):
        pattern = X[j]
        a_star = model.find_equilibrium(pattern)
        H = model.hessian(a_star)
        
        pi_base = np.ones(N)
        s_base = _symmetrised_spread(pi_base, H)
        if s_base is None: continue
        
        pi = np.abs(pattern)**p + 0.1
        pi = model.clip_and_normalise(pi)
        
        s_agent = _symmetrised_spread(pi, H)
        if s_agent is None: continue
        
        reductions.append(s_base / s_agent)
    print(f"Power {p:3.1f}: Mean Reduction = {np.mean(reductions):.2f}x")

print("\n=== Spread test for Variance proxy variants ===")
for j in range(K):
    dists = np.linalg.norm(X - X[j], axis=1)
    neighbors = X[np.argsort(dists)[1:6]]
    local_var = np.var(neighbors, axis=0) + 1e-6
    
    #... just test it
for p in [0.5, 1.0, 2.0, -0.5, -1.0, -2.0]:
    reductions = []
    for j in range(K):
        pattern = X[j]
        a_star = model.find_equilibrium(pattern)
        H = model.hessian(a_star)
        
        pi_base = np.ones(N)
        s_base = _symmetrised_spread(pi_base, H)
        if s_base is None: continue
        
        dists = np.linalg.norm(X - X[j], axis=1)
        neighbors = X[np.argsort(dists)[1:6]]
        local_var = np.var(neighbors, axis=0) + 1e-6
        
        pi = local_var**p
        pi = model.clip_and_normalise(pi)
        
        s_agent = _symmetrised_spread(pi, H)
        if s_agent is None: continue
        
        reductions.append(s_base / s_agent)
    print(f"Variance Power {p:4.1f}: Mean Reduction = {np.mean(reductions):.2f}x")

print("\n=== Spread test for 1/diag(H) ===")
reductions = []
for j in range(K):
    pattern = X[j]
    a_star = model.find_equilibrium(pattern)
    H = model.hessian(a_star)
    
    pi_base = np.ones(N)
    s_base = _symmetrised_spread(pi_base, H)
    if s_base is None: continue
    
    H_diag = np.diag(H)
    pi = 1.0 / np.clip(H_diag, 0.05, None)
    pi = model.clip_and_normalise(pi)
    
    s_agent = _symmetrised_spread(pi, H)
    if s_agent is None: continue
    
    reductions.append(s_base / s_agent)
print(f"1/diag(H): Mean Reduction = {np.mean(reductions):.2f}x")

print("\n=== Test optimal R_diag inverse ===")
reductions = []
for j in range(K):
    pattern = X[j]
    a_star = model.find_equilibrium(pattern)
    H = model.hessian(a_star)
    
    pi_base = np.ones(N)
    s_base = _symmetrised_spread(pi_base, H)
    if s_base is None: continue
    
    # The actual best diagonal preconditioner might just be identity because R is almost diagonal?
    R_diag = np.diag(R)
    pi = 1.0 / np.clip(R_diag, 0.05, None)
    pi = model.clip_and_normalise(pi)
    
    s_agent = _symmetrised_spread(pi, H)
    if s_agent is None: continue
    
    reductions.append(s_base / s_agent)
print(f"1/diag(R): Mean Reduction = {np.mean(reductions):.2f}x")
