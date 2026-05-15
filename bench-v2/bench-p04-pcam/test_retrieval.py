import sys
import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns, make_test_queries
from metrics import retrieval_accuracy
from adapters.dummy import DummyAgent
from adapter import Adapter

seed = 42
X = make_patterns(K=16, N=64, seed=seed, n_clusters=4, intra_sim=0.5)
R = build_default_R(N=64, seed=seed)
model = PCAMModel(X, R)
queries, truths, levels = make_test_queries(X, [0.75, 0.85], 50, seed)
dummy = DummyAgent(X, {'beta': 8.0, 'eta': 0.5, 'R': R})
base_acc = retrieval_accuracy(model, dummy, queries, truths)
print(f"Base accuracy: {base_acc:.3f}")

class PromptRetrieval(Adapter):
    def __init__(self, X, params):
        self.X = X
        self.R = params['R']
        self.Z = self.X @ self.R
        self.beta = params['beta']
        
    def predict_precision(self, q):
        sims = self.Z @ q
        nearest = int(np.argmax(sims))
        
        s = np.exp(self.beta * (sims - sims.max()))
        s /= s.sum()
        target = self.X.T @ s
        pi_noise = np.abs(target) + 0.1
        
        second = int(np.argsort(sims)[-2])
        diff = np.abs(self.X[nearest] - self.X[second])
        pi_noise *= (1.0 + 3.0 * diff)
        
        pi = pi_noise ** 1.5
        pi = np.clip(pi, 0.1, 10.0)
        pi /= pi.mean()
        return pi

agent = PromptRetrieval(X, {'beta': 8.0, 'eta': 0.5, 'R': R})
agent_acc = retrieval_accuracy(model, agent, queries, truths)
print(f"Prompt Retrieval: {agent_acc:.3f} (Delta: {agent_acc - base_acc:.3f})")

class V9Retrieval(Adapter):
    def __init__(self, X, params):
        self.X = X
        self.R = params['R']
        self.Z = self.X @ self.R
        self.N = 64
        self.beta = params['beta']
        
    def predict_precision(self, q):
        sims = self.Z @ q
        sorted_idx = np.argsort(sims)[::-1]
        best = sorted_idx[0]
        second = sorted_idx[1]
        best_cos = float(sims[best])

        intensity = np.clip((1.0 - best_cos - 0.07) / 0.25, 0.0, 1.0)
        if intensity < 0.01:
            return np.ones(self.N)

        proj = (q @ self.X[best]) * self.X[best]
        residual = q - proj
        pi = (1.0 + 2.0 * np.abs(proj)) / (1.0 + 2.0 * np.abs(residual))

        diff = np.abs(self.X[best] - self.X[second])
        pi *= (1.0 + 5.0 * diff)

        pi = intensity * pi + (1.0 - intensity) * np.ones(self.N)
        pi = np.clip(pi, 0.1, 10.0)
        pi /= pi.mean()
        return pi

agent9 = V9Retrieval(X, {'beta': 8.0, 'eta': 0.5, 'R': R})
agent9_acc = retrieval_accuracy(model, agent9, queries, truths)
print(f"V9 Retrieval: {agent9_acc:.3f} (Delta: {agent9_acc - base_acc:.3f})")

