import argparse
import os

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .consensus import STRATEGIES  # noqa: E402
from .defer import defer_stats, risk_coverage, tune_threshold  # noqa: E402
from .synthetic import make_cases  # noqa: E402
from .utils import md_table, set_seed  # noqa: E402


def load_probs(cfg):
    # real data: CSV with columns label, agent_0..agent_K-1 (agent probabilities), one row per case
    if cfg.get("synthetic"):
        y_val, p_val = make_cases(cfg["n_val"], cfg["seed"], cfg["n_agents"])
        y_te, p_te = make_cases(cfg["n_test"], cfg["seed"] + 1, cfg["n_agents"])
        return (y_val, p_val), (y_te, p_te)
    d_val = pd.read_csv(cfg["val_csv"])
    d_te = pd.read_csv(cfg["test_csv"])
    cols = [c for c in d_val.columns if c.startswith("agent_")]
    return (d_val.label.values, d_val[cols].values), (d_te.label.values, d_te[cols].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    set_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    (y_val, p_val), (y_te, p_te) = load_probs(cfg)

    rows = []
    plt.figure(figsize=(5.5, 4.5))
    for name, fn in STRATEGIES.items():
        pred_v, unc_v = fn(p_val)
        tau = tune_threshold(unc_v, pred_v == y_val, cfg["max_selective_error"])
        pred_t, unc_t = fn(p_te)
        correct_t = pred_t == y_te
        r = defer_stats(unc_t, correct_t, tau)
        r["strategy"] = name
        rows.append(r)
        cov, err = risk_coverage(unc_t, correct_t)
        plt.plot(cov, err, label=name)
    plt.xlabel("coverage (fraction handled automatically)")
    plt.ylabel("selective error")
    plt.title("Risk-coverage: single vs consensus")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["out_dir"], "risk_coverage.png"), dpi=200)

    res = pd.DataFrame(rows)[["strategy", "coverage", "selective_acc", "base_acc", "defer_precision",
                              "error_recall", "wrong_defer_rate", "missed_error_rate"]]
    res.to_csv(os.path.join(cfg["out_dir"], "ablation.csv"), index=False)
    with open(os.path.join(cfg["out_dir"], "results_table.md"), "w") as f:
        f.write(md_table(res))
    print(md_table(res))


if __name__ == "__main__":
    main()
