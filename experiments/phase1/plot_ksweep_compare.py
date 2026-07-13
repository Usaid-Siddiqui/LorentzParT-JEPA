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
    p = argparse.ArgumentParser(description="AUC-vs-K overlay for the masking sweep")
    # Unified single-dir mode (run_ablation_ragged): one dir, names jepa_gate{on,off}_{mask}_k{K}
    p.add_argument('--results-dir', default=None,
                   help="unified mode: one dir with jepa_gate{on,off}_{mask}_k{K} + mae_*/scratch/"
                        "jepa_curriculum conditions. Plots all 4 gate×mask series + reference lines.")
    p.add_argument('--ks', nargs='+', type=int, default=[1, 2, 4, 8],
                   help="K values to plot in unified mode")
    # Legacy 3-dir mode (old fragmented Phase-1 layout)
    p.add_argument('--twobytwo-dir', default='./experiments/phase1/results')
    p.add_argument('--biased-dir',   default='./experiments/phase1/results_ksweep')
    p.add_argument('--random-dir',   default='./experiments/phase1/results_ksweep_random')
    p.add_argument('--output',       default=None,
                   help="defaults to <results-dir>/ksweep_grid.png (unified) or "
                        "experiments/phase1/ksweep_biased_vs_random.png (legacy)")
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


def curve_unified(results_dir, gate, mask, ks):
    """(Ks, means, stds) for one gate×mask series from the unified results dir."""
    Ks, means, stds = [], [], []
    for k in ks:
        r = auc(results_dir, f'jepa_gate{gate}_{mask}_k{k}')
        if r is not None:
            Ks.append(k); means.append(r[0]); stds.append(r[1])
    return Ks, means, stds


def _finish_axes(ax, ks):
    ax.set_xscale('log', base=2)
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('K  (particles masked per jet)')
    ax.set_ylabel('OVO ROC AUC')
    ax.grid(alpha=0.3)


def plot_unified(args):
    """All four gate×mask AUC-vs-K series from one dir + MAE/scratch/curriculum reference lines."""
    fig, ax = plt.subplots(figsize=(9, 6))
    series = [('on',  'biased', '#1B5E20', 'o', 'gate-on · biased'),
              ('on',  'random', '#B71C1C', 's', 'gate-on · random'),
              ('off', 'biased', '#66BB6A', '^', 'gate-off · biased'),
              ('off', 'random', '#EF9A9A', 'D', 'gate-off · random')]
    for gate, mask, color, marker, label in series:
        Ks, means, stds = curve_unified(args.results_dir, gate, mask, args.ks)
        if not Ks:
            print(f"[skip] no data for gate-{gate} {mask}")
            continue
        ax.errorbar(Ks, means, yerr=stds, marker=marker, markersize=7, capsize=4,
                    linewidth=2, color=color, label=label, markeredgecolor='black')
        print(f"gate-{gate:3s} {mask:6s}: " + "  ".join(f"K{k}={m:.4f}" for k, m in zip(Ks, means)))

    # horizontal reference lines for conditions that have no single K
    refs = [('scratch',         '#616161', ':',  'scratch'),
            ('mae_biased',      '#1565C0', '--', 'MAE biased'),
            ('mae_random',      '#42A5F5', '--', 'MAE random'),
            ('jepa_curriculum', '#6A1B9A', '-.', 'curriculum')]
    for cond, color, ls, label in refs:
        r = auc(args.results_dir, cond)
        if r is None:
            continue
        ax.axhline(r[0], color=color, ls=ls, lw=1.3, alpha=0.85)
        ax.annotate(f'{label} {r[0]:.3f}', (max(args.ks), r[0]), textcoords='offset points',
                    xytext=(6, 0), va='center', fontsize=7.5, color=color)
        print(f"{label:12s}: {r[0]:.4f}  (n={r[2]})")

    _finish_axes(ax, args.ks)
    ax.set_title('Masking-count sweep (ragged backbone) · gate × mask × K')
    ax.legend(fontsize=9, loc='best')
    ax.margins(x=0.14)
    out = args.output or os.path.join(args.results_dir, 'ksweep_grid.png')
    fig.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    print(f"\nSaved → {out}")


def plot_legacy(args):
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
    _finish_axes(ax, [1, 2, 4, 8, 16])
    ax.set_title('Masking-count sweep: biased vs random  (gate on)')
    ax.legend(fontsize=10)
    out = args.output or './experiments/phase1/ksweep_biased_vs_random.png'
    fig.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    print(f"\nSaved → {out}")


def main():
    args = parse_args()
    if args.results_dir:
        plot_unified(args)
    else:
        plot_legacy(args)


if __name__ == '__main__':
    main()
