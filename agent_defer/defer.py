import numpy as np


def risk_coverage(unc, correct):
    # accept the most certain cases first; return (coverage, selective_error) arrays
    o = np.argsort(unc)
    c = np.asarray(correct, float)[o]
    k = np.arange(1, len(c) + 1)
    return k / len(c), 1 - np.cumsum(c) / k


def tune_threshold(unc, correct, max_error=0.05):
    # largest threshold (max coverage) whose selective error on the tuning set is <= max_error
    best = unc.min() - 1e-9
    for t in np.unique(unc):
        acc = unc <= t
        if acc.sum() == 0:
            continue
        if 1 - correct[acc].mean() <= max_error:
            best = t
    return best


def defer_stats(unc, correct, tau):
    defer = unc > tau
    n = len(correct)
    wrong = ~correct
    return dict(
        coverage=float((~defer).mean()),
        selective_acc=float(correct[~defer].mean()) if (~defer).any() else float("nan"),
        base_acc=float(correct.mean()),
        defer_precision=float(wrong[defer].mean()) if defer.any() else float("nan"),   # deferred cases that were truly wrong
        error_recall=float(defer[wrong].mean()) if wrong.any() else float("nan"),      # wrong cases that got deferred
        wrong_defer_rate=float((defer & correct).sum() / n),                           # unnecessary deferrals
        missed_error_rate=float(((~defer) & wrong).sum() / n),                         # wrong and auto-accepted
    )
