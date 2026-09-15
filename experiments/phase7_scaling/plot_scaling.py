"""
Phase 7 headline figure — best validation accuracy vs. training-set size, for the 2x2
{ParT, LorentzParT} x {scratch, JEPA}, with the published ParT accuracy (0.861) as an anchor line.

Reads the per-epoch CSVs the runs already write (no extra eval): best `val_metric` per cell from
`<logs>/<ModelDir>/logging/<model>_<protocol>_<scale>_seed<N>.csv`. `val_metric` is accuracy
(JetClassTrainer metric), directly comparable to published ParT's 0.861. Metric is on val_5M (we
don't hold test_20M) — a held-out proxy for the published test number; report it as such.

    python experiments/phase7_scaling/plot_scaling.py --scales 10m 100m --seeds 42
    python experiments/phase7_scaling/plot_scaling.py --scales 100k 1m 10m 100m --seeds 42 123 456
"""

import argparse
import csv
import os
from collections import defaultdict  # noqa: F401 (kept for future per-seed dumps)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = {"part": "ParticleTransformer", "lorentzpart": "LorentzParT"}
# (model, protocol, label, color, marker) — hybrid=blue, vanilla=orange; JEPA=lighter/square
CELLS = [
    ("part",        "scratch", "ParT (scratch)",        "#ff7f0e", "o"),
    ("part",        "jepa",    "ParT + JEPA",           "#ffbb78", "s"),
    ("lorentzpart", "scratch", "LorentzParT (scratch)", "#1f77b4", "o"),
    ("lorentzpart", "jepa",    "LorentzParT + JEPA",    "#7fbfff", "s"),
]
PUBLISHED_PART_ACC = 0.861   # Qu et al. 2022, ParT-full on JetClass (10-class accuracy)


def scale_to_jets(s: str) -> float:
    s = s.lower().strip()
    if s[-1] in "km":
        return float(s[:-1]) * {"k": 1e3, "m": 1e6}[s[-1]]
    return float(s)


def best_val(path: str):
    if not os.path.exists(path):
        return None
    best = None
    for d in csv.DictReader(open(path)):
        try:
            v = float(d["val_metric"])
        except (KeyError, ValueError, TypeError):
            continue
        best = v if best is None else max(best, v)
    return best


def main():
    p = argparse.ArgumentParser(description="Phase 7 scaling-curve plot")
    p.add_argument("--logs-dir", default="./logs")
    p.add_argument("--scales", nargs="+", default=["100k", "1m", "10m", "100m"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--out", default="experiments/phase7_scaling/results/scaling_accuracy.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    xs = [scale_to_jets(s) for s in args.scales]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    print(f"{'cell':26}" + "".join(f"{s:>10}" for s in args.scales))
    for model, proto, label, color, marker in CELLS:
        means, stds = [], []
        for scale in args.scales:
            vals = [v for seed in args.seeds
                    if (v := best_val(os.path.join(
                        args.logs_dir, MODEL_DIR[model], "logging",
                        f"{model}_{proto}_{scale}_seed{seed}.csv"))) is not None]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        print(f"{label:26}" + "".join(
            f"{m:>10.4f}" if not np.isnan(m) else f"{'—':>10}" for m in means))
        m, s = np.array(means), np.array(stds)
        mask = ~np.isnan(m)
        if mask.any():
            ax.errorbar(np.array(xs)[mask], m[mask], yerr=s[mask], label=label,
                        color=color, marker=marker, lw=2, capsize=3)

    ax.axhline(PUBLISHED_PART_ACC, ls="--", color="0.4", lw=1)
    ax.text(xs[-1], PUBLISHED_PART_ACC, " published ParT (0.861)", color="0.4",
            fontsize=8, va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("training-set size (jets)")
    ax.set_ylabel("best validation accuracy")
    ax.set_title("JetClass classification accuracy vs. training-set size")
    ax.set_xticks(xs); ax.set_xticklabels(args.scales)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(args.out, dpi=200)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
