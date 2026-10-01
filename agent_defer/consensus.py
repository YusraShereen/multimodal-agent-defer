import numpy as np


def entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)    # in [0,1]


def single(probs):
    p = probs[:, 0]
    return p > 0.5, entropy(p)


def consensus_entropy(probs):
    p = probs.mean(1)
    return p > 0.5, entropy(p)


def consensus_disagreement(probs):
    p = probs.mean(1)
    return p > 0.5, probs.std(1) * 2          # std of probabilities scaled to ~[0,1]


def consensus_combined(probs):
    p = probs.mean(1)
    return p > 0.5, 0.5 * entropy(p) + 0.5 * probs.std(1) * 2


STRATEGIES = {
    "single agent": single,
    "consensus (entropy)": consensus_entropy,
    "consensus (disagreement)": consensus_disagreement,
    "consensus (combined)": consensus_combined,
}
