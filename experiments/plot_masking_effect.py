"""
Effect of biased vs random masking on each objective (JEPA and MAE).

The (objective × masking) 2×2, as grouped bars: shows biased masking helps BOTH
objectives — and helps MAE more (it was the one handicapped by random masking).
JEPA cells use the NO-GATE variants so masking is the only variable.

Cells (test AUC, 3 seeds):
    JEPA random  ← phase1/results : no_gate_random
    JEPA biased  ← phase1/results : no_gate_biased
    MAE  random  ← phase0/results : mae_finetune
    MAE  biased  ← phase0/results : mae_biased_finetune

    python experiments/plot_masking_effect.py
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# (group, masking) -> (results_dir_arg, condition_key)
CELLS = {
    ('JEPA', 'random'): ('phase1', 'no_gate_random'),
    ('JEPA', 'biased'): ('phase1', 'no_gate_biased'),
    ('MAE',  'random'): ('phase0', 'mae_finetune'),
    ('MAE',  'biased'): ('phase0', 'mae_biased_finetune'),
}
GROUPS = ['JEPA', 'MAE']
MASKINGS = ['random', 'biased']
COLORS = {'random': '#90CAF9', 'biased': '#1565C0'}


def parse_args():
    p = argparse.ArgumentParser(description="Biased-vs-random masking effect per objective")
    p.add_argument('--phase0-dir', default='./experiments/phase0/results')
    p.add_argument('--phase1-dir', default='./experiments/phase1/results')
    p.add_argument('--output', default='./experiments/masking_effect.png')
    return p.parse_args()


def _collect(results_dir, cond, key, nested=None):
    """Mean, std, n of a metric for a condition across that dir's seed JSONs."""
    vals = []
    for f in sorted(glob.glob(os.path.join(results_dir, 'seed_*.json'))):
        c = json.load(open(f))['conditions'].get(cond, {})
        if nested:
            c = c.get(nested, {})
        if key in c:
            vals.append(c[key])
    if not vals:
        raise ValueError(f"no {key} for {cond} in {results_dir}")
    return np.mean(vals), np.std(vals), len(vals)


def auc(results_dir, cond):
    return _collect(results_dir, cond, 'test_auc')


def main():
    args = parse_args()
    dirs = {'phase0': args.phase0_dir, 'phase1': args.phase1_dir}

    stats = {}
    n = 0
    for (group, masking), (which, cond) in CELLS.items():
        m, s, k = auc(dirs[which], cond)
        stats[(group, masking)] = (m, s)
        n = max(n, k)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = np.arange(len(GROUPS))
    w = 0.36
    for i, masking in enumerate(MASKINGS):
        means = [stats[(g, masking)][0] for g in GROUPS]
        stds  = [stats[(g, masking)][1] for g in GROUPS]
        bars = ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=5,
                      label=f'{masking} masking', color=COLORS[masking],
                      edgecolor='black', linewidth=0.8)
        for b, mv in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                    f'{mv:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Δ (biased − random) annotation per objective
    for xi, g in zip(x, GROUPS):
        d = stats[(g, 'biased')][0] - stats[(g, 'random')][0]
        top = max(stats[(g, 'random')][0], stats[(g, 'biased')][0])
        ax.text(xi, top + 0.045, f'Δ={d:+.3f}', ha='center', fontsize=10,
                bbox=dict(boxstyle='round', fc='#FFF9C4', ec='gray'))

    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, fontsize=12)
    ax.set_ylabel('OVO ROC AUC', fontsize=11)
    ax.set_ylim(0.6, 0.85)
    ax.set_title(f'Biased masking helps both objectives  (gate off, n={n} seeds, ±1 std)',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.35)
    fig.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

    print(f"Saved → {args.output}")
    for g in GROUPS:
        r, b = stats[(g, 'random')][0], stats[(g, 'biased')][0]
        print(f"  {g:4s}  random {r:.4f}  biased {b:.4f}  Δ {b - r:+.4f}")

    # ── Paired embedding-geometry figure (same grouping, 3 metrics) ──────────────
    metrics = [('effective_rank', 'Effective Rank\n(collapse → 1)'),
               ('mean_var',        'Mean Variance\n(collapse → 0)'),
               ('mean_cos_sim',    'Mean Cosine Sim\n(collapse → 1)')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (key, title) in zip(axes, metrics):
        for i, masking in enumerate(MASKINGS):
            means, stds = [], []
            for g in GROUPS:
                which, cond = CELLS[(g, masking)]
                m, s, _ = _collect(dirs[which], cond, key, nested='embedding_stats')
                means.append(m); stds.append(s)
            ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=4,
                   label=f'{masking} masking', color=COLORS[masking],
                   edgecolor='black', linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(GROUPS, fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.35)
    axes[0].legend(fontsize=9)
    fig.suptitle('Representation geometry: objective dominates, masking barely moves it  '
                 f'(gate off, n={n} seeds, ±1 std)', fontsize=11)
    fig.tight_layout()
    embed_out = args.output.replace('.png', '_embedding.png')
    plt.savefig(embed_out, dpi=300)
    plt.close()
    print(f"Saved → {embed_out}")


if __name__ == '__main__':
    main()
