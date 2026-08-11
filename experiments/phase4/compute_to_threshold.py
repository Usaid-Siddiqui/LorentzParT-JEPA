"""
Phase 4 — the honest finetune-compute metric: compute-to-threshold.

`analyze_phase4.py` charges finetune at its full train_time (time-to-early-stop), which is
biased against pretrained methods (they plateau higher, so they stop later). The fair question
is: does finetuning from a pretrained encoder reach a *fixed target accuracy* with LESS compute
than scratch? This reads the per-epoch finetune convergence CSVs, finds the wall-clock at which
each run first crosses a per-task target (a fraction of the achieved val ceiling), and redraws
the amortization on that metric.

Run on the machine that has the training logs (no need to pull CSVs anywhere):

    python experiments/phase4/compute_to_threshold.py \\
        --logging-dir logs/LorentzParT/logging \\
        --target-frac 0.95 --pretrain-cost-jepa 812 --pretrain-cost-mae 5202

Metric note: the CSV logs *validation accuracy* (the trainer metric), so the target is an
accuracy threshold — a reasonable convergence proxy (AUC curves track it).
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
import tasks as T

ENCODERS = ['scratch', 'jepa', 'mae']
VAL_COLS = ['val_metric', 'val_acc', 'val_accuracy', 'val_metric_acc']
TIME_COLS = ['elapsed_total_s', 'elapsed', 'time_total_s', 'cumulative_time_s', 'time']


def _pick(cols, candidates, kind):
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"no {kind} column found in {cols} (looked for {candidates})")


def load_curve(path):
    """Return (val[], elapsed[]) arrays from a finetune CSV, or None if missing/empty."""
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    vcol = _pick(rows[0].keys(), VAL_COLS, 'val-accuracy')
    tcol = _pick(rows[0].keys(), TIME_COLS, 'elapsed-time')
    val = np.array([float(r[vcol]) for r in rows])
    ela = np.array([float(r[tcol]) for r in rows])
    return val, ela, vcol, tcol


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logging-dir', default='logs/LorentzParT/logging')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--target-frac', type=float, default=0.95,
                   help="target = frac × per-task mean final val accuracy")
    p.add_argument('--pretrain-cost-jepa', type=float, default=812.0)
    p.add_argument('--pretrain-cost-mae', type=float, default=5202.0)
    p.add_argument('--out', default='experiments/phase4/results/amortization_threshold.png')
    args = p.parse_args()
    pre = {'scratch': 0.0, 'jepa': args.pretrain_cost_jepa, 'mae': args.pretrain_cost_mae}

    def path(task, enc, seed):
        return os.path.join(args.logging_dir, f'{task}_finetune_{enc}_seed{seed}.csv')

    # ── load every curve, report what columns were detected ──────────────────
    curves = {}           # (task, enc, seed) -> (val, elapsed)
    detected = None
    for task in T.TASKS:
        for enc in ENCODERS:
            for s in args.seeds:
                c = load_curve(path(task, enc, s))
                if c is None:
                    print(f"[miss] {task}_finetune_{enc}_seed{s}.csv", flush=True)
                    continue
                val, ela, vcol, tcol = c
                curves[(task, enc, s)] = (val, ela)
                detected = (vcol, tcol)
    if not curves:
        print(f"No finetune CSVs found in {args.logging_dir}. Check --logging-dir.")
        return
    print(f"\ndetected columns: val='{detected[0]}'  elapsed='{detected[1]}'")

    # ── per-task target = frac × mean(final val across all cells) ─────────────
    def compute_to_target(task):
        finals = [v[-1] for (t, e, s), (v, _) in curves.items() if t == task]
        target = args.target_frac * float(np.mean(finals))
        out = {}   # enc -> list of compute-to-threshold seconds (per seed)
        for enc in ENCODERS:
            secs = []
            for s in args.seeds:
                if (task, enc, s) not in curves:
                    continue
                val, ela = curves[(task, enc, s)]
                hit = np.where(val >= target)[0]
                secs.append(float(ela[hit[0]]) if len(hit) else float(ela[-1]))  # final if never
            out[enc] = secs
        return target, out

    # ── table + cumulative-compute amortization ──────────────────────────────
    print(f"\ncompute-to-threshold (val acc ≥ {args.target_frac:.2f}×ceiling), seconds, mean±std:")
    print(f"{'task':12}{'target':>8}   " + "".join(f"{e:>16}" for e in ENCODERS)
          + f"{'jepa saves':>12}{'mae saves':>11}")
    cum = {e: [] for e in ENCODERS}
    for task in T.TASKS:
        target, out = compute_to_target(task)
        row = f"{task:12}{target:>8.3f}   "
        means = {}
        for e in ENCODERS:
            m = np.mean(out[e]) if out[e] else np.nan
            sd = np.std(out[e]) if out[e] else np.nan
            means[e] = m
            cum[e].append(m)
            row += f"{f'{m:.0f}±{sd:.0f}':>16}"
        js = 100 * (means['scratch'] - means['jepa']) / means['scratch'] if means['scratch'] else np.nan
        ms = 100 * (means['scratch'] - means['mae']) / means['scratch'] if means['scratch'] else np.nan
        row += f"{js:>11.0f}%{ms:>10.0f}%"
        print(row)

    xs = np.arange(1, len(T.TASKS) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, np.cumsum(cum['scratch']), 'o-', color='0.4', label='scratch (N finetunes)')
    for enc, col in [('jepa', 'C0'), ('mae', 'C1')]:
        ax.plot(xs, pre[enc] + np.cumsum(cum[enc]), 's-', color=col,
                label=f'{enc} (pretrain + N finetunes-to-target)')
    ax.set_xlabel('number of downstream tasks')
    ax.set_ylabel('cumulative compute-to-threshold (s)')
    ax.set_title('Amortization — honest compute-to-threshold metric')
    ax.set_xticks(xs); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, dpi=200)
    print(f"\nwrote {args.out}")
    print("Read: 'jepa/mae saves' = per-task % less finetune compute than scratch to hit target. "
          "On the figure, a pretrained line dipping below grey = amortization; the crossover #tasks "
          "is where pretrain cost is repaid.")


if __name__ == '__main__':
    main()
