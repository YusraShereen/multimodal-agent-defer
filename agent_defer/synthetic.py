import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_cases(n, seed, n_agents=3):
    # simulated agents with correlated errors on hard cases. Smoke test / demo only.
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    d = rng.beta(2, 2, n)                    # case difficulty in [0,1]
    shared = rng.normal(0, 1, n)             # shared misleading evidence (correlated errors)
    skills = [1.0, 0.9, 0.8, 0.7][:n_agents]
    probs = np.zeros((n, n_agents))
    for k, sk in enumerate(skills):
        logit = (2 * y - 1) * 3.0 * sk * (1 - d) + 1.5 * shared * d + rng.normal(0, 0.8 * (0.5 + d), n)
        probs[:, k] = sigmoid(logit)
    return y, probs
