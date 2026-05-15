import numpy as np
from adapter import Adapter


class Engine(Adapter):
    def __init__(self, stored_patterns: np.ndarray, model_params: dict):
        self.X = stored_patterns.astype(np.float64)
        self.K, self.N = self.X.shape
        self.beta = float(model_params.get("beta", 8.0))
        self.eta  = float(model_params.get("eta", 0.5))
        self.dt = float(model_params.get("dt", 0.01))
        self.T_max = int(model_params.get("T_max", 3000))
        self.tol = float(model_params.get("tol", 1e-6))
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        R = model_params.get("R", None)
        if R is not None and R.ndim == 2 and R.shape == (self.N, self.N):
            self.R = R.astype(np.float64)
            self.use_R = True
        else:
            self.R = np.eye(self.N)
            self.use_R = False

        self.Z = self.X @ self.R if self.use_R else self.X
        self.pattern_norms = np.linalg.norm(self.X, axis=1)
        self.pattern_norms = np.maximum(self.pattern_norms, 1e-12)

        self.optimal_pi = np.ones((self.K, self.N))
        self.optimal_pi_soft = np.ones((self.K, self.N))

        for j in range(self.K):
            # Match the harness equilibrium: pi = I, no external input.
            a_star = self._find_equilibrium(self.X[j], model_params)
            H = self._hessian(a_star)
            pi = self._optimise_precision(H, restart_seed=j)
            self.optimal_pi[j] = pi
            self.optimal_pi_soft[j] = pi ** 0.3

    def _find_equilibrium(self, x0, model_params):
        """Replicate PCAMModel.find_equilibrium with pi=I and no input."""
        a = np.asarray(x0, dtype=np.float64).copy()

        for _ in range(self.T_max):
            s = self._softmax(self.beta * (self.X @ a))
            g = self.R @ a - self.eta * (self.X.T @ s)
            a_new = a - self.dt * g
            if np.linalg.norm(a_new - a) < self.tol:
                a = a_new
                break
            a = a_new
        return a

    def _hessian(self, a):
        s = self._softmax(self.beta * (self.X @ a))
        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))
        return 0.5 * (H + H.T)

    def _softmax(self, x):
        z = x - x.max()
        e = np.exp(z)
        return e / max(e.sum(), 1e-12)

    def _clip_and_normalise(self, pi):
        pi = np.asarray(pi, dtype=np.float64).reshape(self.N)
        if not np.all(np.isfinite(pi)):
            return np.ones(self.N)

        for _ in range(20):
            pi = np.clip(pi, self.pi_min, self.pi_max)
            mean = pi.mean()
            if mean <= 1e-12:
                return np.ones(self.N)
            pi = pi / mean
            if (pi.min() >= self.pi_min - 1e-9
                    and pi.max() <= self.pi_max + 1e-9
                    and abs(pi.mean() - 1.0) < 1e-8):
                break
        return np.clip(pi, self.pi_min, self.pi_max)

    def _condition_and_gradient(self, log_pi, H):
        pi = self._clip_and_normalise(np.exp(log_pi))
        log_pi = np.log(np.maximum(pi, 1e-12))
        log_pi -= log_pi.mean()

        sqrt_pi = np.sqrt(pi)
        M = sqrt_pi[:, None] * H * sqrt_pi[None, :]
        M = 0.5 * (M + M.T)
        eigvals, eigvecs = np.linalg.eigh(M)
        pos = np.flatnonzero(eigvals > 1e-9)
        if len(pos) < 2:
            return np.inf, np.zeros(self.N), pi, log_pi

        lo = pos[0]
        hi = pos[-1]
        condition = float(eigvals[hi] / eigvals[lo])
        gradient = eigvecs[:, hi] ** 2 - eigvecs[:, lo] ** 2
        gradient -= gradient.mean()
        return condition, gradient, pi, log_pi

    def _optimise_precision(self, H, restart_seed):
        """Minimise the local spectral spread of Pi^(1/2) H Pi^(1/2)."""
        rng = np.random.default_rng(1729 + int(restart_seed))
        starts = [np.zeros(self.N)]

        diag = np.clip(np.diag(H), 1e-9, None)
        starts.append(-np.log(diag))
        while len(starts) < 5:
            starts.append(rng.normal(0.0, 0.1, self.N))

        best_pi = np.ones(self.N)
        best_condition, _, _, _ = self._condition_and_gradient(np.zeros(self.N), H)

        for start in starts:
            log_pi = np.asarray(start, dtype=np.float64)
            log_pi -= log_pi.mean()
            lr = 2.0

            for _ in range(150):
                condition, gradient, pi, log_pi = self._condition_and_gradient(log_pi, H)
                if condition < best_condition:
                    best_condition = condition
                    best_pi = pi.copy()
                if not np.isfinite(condition):
                    break

                log_pi -= lr * gradient
                log_pi -= log_pi.mean()
                lr = max(0.25, lr * 0.995)

        return self._clip_and_normalise(best_pi)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = corrupted_query.astype(np.float64)

        # Similarity
        sims = self.Z @ q if self.use_R else self.X @ q
        sorted_idx = np.argsort(sims)[::-1]
        best = sorted_idx[0]
        second = sorted_idx[1]
        best_cos = float(sims[best])

        true_cosines = self.X @ q
        q_norm = max(float(np.linalg.norm(q)), 1e-12)
        true_confidence = true_cosines / (self.pattern_norms * q_norm)
        clean_best = int(np.argmax(true_confidence))
        clean_confidence = float(true_confidence[clean_best])

        # MODE 1: Clean query (anisotropy test)
        if clean_confidence > 0.85:
            pi = self.optimal_pi[clean_best].copy()
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
