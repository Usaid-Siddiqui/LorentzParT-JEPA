"""
Phase 3 speedup bar chart: per-epoch training time by condition.

Reads the seed_*.json written by run_ragged_e2e.py (which stores
pretrain_epoch_s / finetune_epoch_s per condition) and plots median epoch time for
pretrain and finetune, grouped by condition, with the speedup vs stock annotated on
each bar. Per-epoch (not total) time removes the confound of different epoch counts
from early stopping.

    python experiments/phase3/plot_speedup.py --results-dir experiments/phase3/results
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'stock': 'Stock (−1e9)', 'ragged': 'Ragged (padding-aware)', 'fill0': 'Fill-0 (dense BN fix)'}
COLORS = {'stock': '#1f77b4', 'ragged': '#ff7f0e', 'fill0': '#2ca02c'}
PHASES = [('pretrain', 'pretrain_epoch_s'), ('finetune', 'finetune_epoch_s')]


def main():
    p = argparse.ArgumentParser(description="Phase 3 per-epoch speedup bar chart")
    p.add_argument('--results-dir', default='./experiments/phase3/results')
    p.add_argument('--conditions', nargs='+', default=['stock', 'ragged', 'fill0'])
    p.add_argument('--output', default=None)
    args = p.parse_args()
    out = args.output or os.path.join(args.results_dir, 'speedup_bars.png')

    seeds = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.results_dir, 'seed_*.json')))]
    if not seeds:
        raise FileNotFoundError(f"no seed_*.json in {args.results_dir}")

    def agg(cond, key):
        v = [s['conditions'][cond][key] for s in seeds
             if cond in s['conditions'] and s['conditions'][cond].get(key) is not None]
        return (float(np.mean(v)), float(np.std(v))) if v else (float('nan'), 0.0)

    conds = [c for c in args.conditions if any(c in s['conditions'] for s in seeds)]
    ref = {ph: agg('stock', key)[0] for ph, key in PHASES}   # stock is the speedup reference

    x = np.arange(len(PHASES))
    w = 0.8 / len(conds)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, c in enumerate(conds):
        means = [agg(c, key)[0] for _, key in PHASES]
        stds = [agg(c, key)[1] for _, key in PHASES]
        pos = x + (i - (len(conds) - 1) / 2) * w
        ax.bar(pos, means, w, yerr=stds, capsize=4, color=COLORS.get(c),
               edgecolor='black', linewidth=0.6, label=LABELS.get(c, c))
        for xp, m, (ph, _) in zip(pos, means, PHASES):
            sp = ref[ph] / m if m else float('nan')
            ax.text(xp, m, f'{sp:.2f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([ph for ph, _ in PHASES], fontsize=12)
    ax.set_ylabel('median epoch time (s)', fontsize=11)
    ax.set_title(f'Phase 3: per-epoch training time  (100k, n={len(seeds)} seeds, ±1 std)\n'
                 'speedup vs stock annotated', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('wrote', out)


if __name__ == '__main__':
    main()
