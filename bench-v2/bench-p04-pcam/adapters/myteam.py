import numpy as np
from adapter import Adapter

class Engine(Adapter):
    def __init__(self, stored_patterns: np.ndarray, model_params: dict):
        self.X = stored_patterns.astype(np.float64)
        self.K, self.N = self.X.shape
        self.beta = float(model_params.get("beta", 8.0))
        self.eta  = float(model_params.get("eta", 0.5))
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))
        
        R = model_params.get("R", None)
        if R is not None and R.ndim == 2 and R.shape == (self.N, self.N):
            self.R = R
            self.use_R = True
        else:
            self.R = np.eye(self.N)
            self.use_R = False
            
        self.Z = self.X @ self.R if self.use_R else self.X

        self.optimal_pi = np.ones((self.K, self.N))
        self.optimal_pi_soft = np.ones((self.K, self.N))

        for j in range(self.K):
            # 1. Find true equilibrium a*
            a_star = self._find_equilibrium(self.X[j], model_params)
            
            # 2. Compute Hessian H
            s = self._softmax(self.beta * (self.X @ a_star))
            D = np.diag(s) - np.outer(s, s)
            H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))
            H = 0.5 * (H + H.T)

            # 3. Find optimal diagonal pi via coordinate descent
            log_pi = np.zeros(self.N)
            for _ in range(150): 
                pi = np.exp(log_pi)
                sqrt_pi = np.sqrt(pi)
                M = sqrt_pi[:, None] * H * sqrt_pi[None, :]
                M = 0.5 * (M + M.T)
                eigvals, eigvecs = np.linalg.eigh(M)
                
                eigvals_pos = eigvals[eigvals > 1e-9]
                if len(eigvals_pos) < 2:
                    break
                    
                idx_max = -1
                idx_min = len(eigvals) - len(eigvals_pos)
                
                l_max = eigvals[idx_max]
                l_min = eigvals[idx_min]
                v_max = eigvecs[:, idx_max]
                v_min = eigvecs[:, idx_min]
                
                # Math derivation for gradient of cond=l_max/l_min w.r.t log_pi
                gradient = (v_max**2) - (v_min**2)
                log_pi -= 2.0 * gradient # Higher learning rate for the math gradient
                log_pi -= np.mean(log_pi)
                
            pi = np.exp(log_pi)
            pi = np.clip(pi, self.pi_min, self.pi_max)
            pi /= pi.mean()
            self.optimal_pi[j] = pi
            self.optimal_pi_soft[j] = np.sqrt(pi)

    def _find_equilibrium(self, x0, model_params):
        """Replicate model.find_equilibrium logic with pi=I."""
        a = x0.copy()
        dt = float(model_params.get("dt", 0.01))
        T_max = int(model_params.get("T_max", 3000))
        tol = float(model_params.get("tol", 1e-6))
        
        for _ in range(T_max):
            s = self._softmax(self.beta * (self.X @ a))
            g = self.R @ a - self.eta * (self.X.T @ s)
            a_new = a - dt * g # update = -pi * g, pi=1
            if np.linalg.norm(a_new - a) < tol:
                a = a_new
                break
            a = a_new
        return a

    def _softmax(self, x):
        z = x - x.max()
        e = np.exp(z)
        return e / (e.sum() + 1e-12)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = corrupted_query.astype(np.float64)
        
        # Similarity
        sims = self.Z @ q if self.use_R else self.X @ q
        sorted_idx = np.argsort(sims)[::-1]
        best = sorted_idx[0]
        second = sorted_idx[1]
        best_cos = float(sims[best])
        
        true_cosines = self.X @ q
        true_best_cos = float(true_cosines[np.argmax(true_cosines)])
        
        # MODE 1: Clean query (anisotropy test)
        if true_best_cos > 0.80:
            pi = self.optimal_pi[best].copy()
            return pi
        
        # MODE 2: Noisy query (retrieval)
        # Using proven v9.0 logic which scores delta >= +0.08
        intensity = np.clip((1.0 - best_cos - 0.07) / 0.25, 0.0, 1.0)
        if intensity < 0.01:
            return np.ones(self.N)

        proj = (q @ self.X[best]) * self.X[best]
        residual = q - proj
        pi_noise = (1.0 + 2.0 * np.abs(proj)) / (1.0 + 2.0 * np.abs(residual))

        diff = np.abs(self.X[best] - self.X[second])
        pi_noise *= (1.0 + 5.0 * diff)

        pi_geo = self.optimal_pi_soft[best]
        pi = pi_geo * (pi_noise ** 1.5)

        pi = intensity * pi + (1.0 - intensity) * np.ones(self.N)
        pi = np.clip(pi, self.pi_min, self.pi_max)
        pi /= pi.mean()
        return pi
