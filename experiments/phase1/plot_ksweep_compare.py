"""
Overlay AUC-vs-K for biased vs random masking — the multi-particle-masking
hypothesis test. If random rises with K while biased stays flat, biased masking
front-loads the informative particle; if both decline, context depletion dominates.

K=1 points are anchored from the 2×2 (gate_biased / gate_random, 3 seeds each).
K∈{2,4,8,16} come from the two sweep dirs (biased: k{K}; random: rand_k{K}).
Works with however many seeds are present (reports n per point).

    python experiments/phase1/plot_ksweep_compare.py
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Biased vs random AUC-vs-K overlay")
    p.add_argument('--twobytwo-dir', default='./experiments/phase1/results')
    p.add_argument('--biased-dir',   default='./experiments/phase1/results_ksweep')
    p.add_argument('--random-dir',   default='./experiments/phase1/results_ksweep_random')
    p.add_argument('--output',       default='./experiments/phase1/ksweep_biased_vs_random.png')
    return p.parse_args()


def auc(results_dir, cond):
    """mean, std, n of test_auc for a condition across a dir's seed JSONs."""
    vals = []
    for f in sorted(glob.glob(os.path.join(results_dir, 'seed_*.json'))):
        c = json.load(open(f))['conditions'].get(cond, {})
        if 'test_auc' in c:
            vals.append(c['test_auc'])
    return (np.mean(vals), np.std(vals), len(vals)) if vals else None


def curve(args, masking):
    """Return (Ks, means, stds) for one masking type, anchoring K=1 from the 2×2."""
    if masking == 'biased':
        k1 = auc(args.twobytwo_dir, 'gate_biased')
        rest = {k: auc(args.biased_dir, f'k{k}') for k in (2, 4, 8, 16)}
    else:
        k1 = auc(args.twobytwo_dir, 'gate_random')
        rest = {k: auc(args.random_dir, f'rand_k{k}') for k in (2, 4, 8, 16)}

    pts = {1: k1, **rest}
    Ks, means, stds = [], [], []
    for k in sorted(pts):
        if pts[k] is not None:
            Ks.append(k); means.append(pts[k][0]); stds.append(pts[k][1])
    return Ks, means, stds


def main():
    args = parse_args()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = {'biased': ('#1B5E20', 'o', 'biased masking'),
              'random': ('#B71C1C', 's', 'random masking')}

    for masking, (color, marker, label) in styles.items():
        Ks, means, stds = curve(args, masking)
        if not Ks:
            print(f"[skip] no data for {masking}")
            continue
        ax.errorbar(Ks, means, yerr=stds, marker=marker, markersize=8, capsize=5,
                    linewidth=2, color=color, label=label, markeredgecolor='black')
        for k, m in zip(Ks, means):
            ax.annotate(f'{m:.3f}', (k, m), textcoords='offset points', xytext=(0, 9),
                        ha='center', fontsize=8, color=color)
        print(f"{masking:7s}: " + "  ".join(f"K{k}={m:.4f}" for k, m in zip(Ks, means)))

    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('K  (particles masked per jet)')
    ax.set_ylabel('OVO ROC AUC')
    ax.set_title('Masking-count sweep: biased vs random  (gate on)')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"\nSaved → {args.output}")


if __name__ == '__main__':
    main()
