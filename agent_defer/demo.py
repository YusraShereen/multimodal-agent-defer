import argparse

import numpy as np

from .consensus import consensus_combined
from .defer import tune_threshold
from .synthetic import make_cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    yv, pv = make_cases(1000, a.seed)
    pred, unc = consensus_combined(pv)
    tau = tune_threshold(unc, pred == yv, 0.05)
    y, p = make_cases(a.n, a.seed + 7)
    pr, u = consensus_combined(p)
    print(f"defer threshold tau={tau:.3f}\n")
    print("case | truth | agent probs            | consensus | uncertainty | decision")
    for i in range(a.n):
        decision = "DEFER to human" if u[i] > tau else f"auto: {'anomaly' if pr[i] else 'normal'}"
        print(f"{i:4d} | {'anom' if y[i] else 'norm'}  | {np.round(p[i], 2)} | {p[i].mean():.2f}      | {u[i]:.2f}        | {decision}")


if __name__ == "__main__":
    main()
