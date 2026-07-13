"""
Faceted ablation figure — one small panel per comparison, each holding the other
factors fixed, from a single run_ablation_ragged results dir (unified naming
jepa_gate{on,off}_{mask}_k{K} + mae_{biased,random} + scratch + jepa_curriculum).

Panels:
  1. MAE: biased vs random                       (bar)
  2. JEPA masking: biased vs random over K        (line, pooled over gate)
  3. JEPA gate: on vs off over K                  (line, pooled over masking)
  4. K-sweep | biased: gate-on vs gate-off        (line)
  5. K-sweep | random: gate-on vs gate-off        (line)
  6. K-sweep | gate-on: biased vs random          (line)
  7. K-sweep | gate-off: biased vs random         (line)
  8. Objectives: scratch / MAE / JEPA-best / curriculum   (bar)

Every panel is zoomed to its own data range (the whole ablation lives in a ~0.004
AUC band) and shows ±1σ across seeds, so you can see whether any split clears noise.

    python experiments/phase1/plot_ablation_facets.py \\
        --results-dir experiments/phase1/results_ragged --ks 1 2 4 8
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
    p = argparse.ArgumentParser(description="Faceted ablation comparison figure")
    p.add_argument('--results-dir', default='experiments/phase1/results_ragged')
    p.add_argument('--ks', nargs='+', type=int, default=[1, 2, 4, 8])
    p.add_argument('--metric', default='test_auc')
    p.add_argument('--output', default=None, help="defaults to <results-dir>/ablation_facets.png")
    return p.parse_args()


def jname(gate, mask, k):
    return f'jepa_gate{gate}_{mask}_k{k}'


def pool(results_dir, conds, metric):
    """(mean, std, n) of `metric` pooled across every seed of every condition in `conds`."""
    vals = []
    for f in sorted(glob.glob(os.path.join(results_dir, 'seed_*.json'))):
        c = json.load(open(f))['conditions']
        for cond in conds:
            if cond in c and metric in c[cond]:
                vals.append(c[cond][metric])
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None


def kcurve(results_dir, ks, name_fn, metric):
    """(Ks, means, stds) where each K pools whatever conditions name_fn(k) returns."""
    Ks, ms, ss = [], [], []
    for k in ks:
        r = pool(results_dir, name_fn(k), metric)
        if r is not None:
            Ks.append(k); ms.append(r[0]); ss.append(r[1])
    return Ks, ms, ss


def _zoom(ax, values, pad=0.0015):
    vals = [v for v in values if v is not None]
    if vals:
        ax.set_ylim(min(vals) - pad, max(vals) + pad)


def _kaxis(ax, ks):
    ax.set_xscale('log', base=2)
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('K'); ax.grid(alpha=0.3)


def line_panel(ax, results_dir, ks, series, title, metric, scratch=None):
    """series: list of (name_fn, color, marker, label)."""
    allv = []
    for name_fn, color, marker, label in series:
        Ks, ms, ss = kcurve(results_dir, ks, name_fn, metric)
        if not Ks:
            continue
        ax.errorbar(Ks, ms, yerr=ss, marker=marker, ms=6, capsize=3, lw=1.8,
                    color=color, label=label, markeredgecolor='black')
        allv += [m + s for m, s in zip(ms, ss)] + [m - s for m, s in zip(ms, ss)]
    if scratch is not None:
        ax.axhline(scratch, ls=':', color='#616161', lw=1.2)
        allv.append(scratch)
    _zoom(ax, allv)
    _kaxis(ax, ks)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5)


def bar_panel(ax, labels, triples, title, colors):
    """triples: list of (mean,std,n) aligned with labels."""
    xs = range(len(labels))
    ms = [t[0] if t else np.nan for t in triples]
    ss = [t[1] if t else 0 for t in triples]
    ax.bar(xs, ms, yerr=ss, capsize=4, color=colors, edgecolor='black', linewidth=0.5)
    for x, t in zip(xs, triples):
        if t:
            ax.annotate(f'{t[0]:.4f}', (x, t[0]), textcoords='offset points',
                        xytext=(0, 4), ha='center', fontsize=7)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=8)
    _zoom(ax, [m + s for m, s in zip(ms, ss)] + [m - s for m, s in zip(ms, ss)], pad=0.002)
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.3, axis='y')


def main():
    args = parse_args()
    d, ks, m = args.results_dir, args.ks, args.metric
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    ax = axes.ravel()
    scratch = pool(d, ['scratch'], m)
    s0 = scratch[0] if scratch else None

    # 1. MAE biased vs random
    bar_panel(ax[0], ['MAE biased', 'MAE random'],
              [pool(d, ['mae_biased'], m), pool(d, ['mae_random'], m)],
              'MAE: biased vs random', ['#1B5E20', '#B71C1C'])

    # 2. JEPA masking (pooled over gate): biased vs random over K
    line_panel(ax[1], d, ks, [
        (lambda k: [jname('on', 'biased', k), jname('off', 'biased', k)], '#1B5E20', 'o', 'biased'),
        (lambda k: [jname('on', 'random', k), jname('off', 'random', k)], '#B71C1C', 's', 'random'),
    ], 'JEPA masking: biased vs random (pooled over gate)', m, s0)

    # 3. JEPA gate (pooled over masking): on vs off over K
    line_panel(ax[2], d, ks, [
        (lambda k: [jname('on', 'biased', k), jname('on', 'random', k)], '#0D47A1', 'o', 'gate-on'),
        (lambda k: [jname('off', 'biased', k), jname('off', 'random', k)], '#F9A825', 's', 'gate-off'),
    ], 'JEPA gate: on vs off (pooled over masking)', m, s0)

    # 4. K-sweep | biased: gate on vs off
    line_panel(ax[3], d, ks, [
        (lambda k: [jname('on', 'biased', k)], '#1B5E20', 'o', 'gate-on'),
        (lambda k: [jname('off', 'biased', k)], '#66BB6A', '^', 'gate-off'),
    ], 'K-sweep | biased', m, s0)

    # 5. K-sweep | random: gate on vs off
    line_panel(ax[4], d, ks, [
        (lambda k: [jname('on', 'random', k)], '#B71C1C', 's', 'gate-on'),
        (lambda k: [jname('off', 'random', k)], '#EF9A9A', 'D', 'gate-off'),
    ], 'K-sweep | random', m, s0)

    # 6. K-sweep | gate-on: biased vs random
    line_panel(ax[5], d, ks, [
        (lambda k: [jname('on', 'biased', k)], '#1B5E20', 'o', 'biased'),
        (lambda k: [jname('on', 'random', k)], '#B71C1C', 's', 'random'),
    ], 'K-sweep | gate-on', m, s0)

    # 7. K-sweep | gate-off: biased vs random
    line_panel(ax[6], d, ks, [
        (lambda k: [jname('off', 'biased', k)], '#66BB6A', '^', 'biased'),
        (lambda k: [jname('off', 'random', k)], '#EF9A9A', 'D', 'random'),
    ], 'K-sweep | gate-off', m, s0)

    # 8. Objectives: scratch / MAE / JEPA-best (gate-on random K1) / curriculum
    bar_panel(ax[7],
              ['scratch', 'MAE\nrandom', 'JEPA\n(best)', 'curric.'],
              [scratch, pool(d, ['mae_random'], m),
               pool(d, [jname('on', 'random', 1)], m), pool(d, ['jepa_curriculum'], m)],
              'Objectives (pretrain recipe)', ['#616161', '#42A5F5', '#1B5E20', '#6A1B9A'])

    fig.suptitle(f'JEPA ablation on ragged backbone ({m}, ±1σ over seeds) — '
                 f'note the whole grid spans <0.005 AUC (within noise)', y=1.01, fontsize=12)
    fig.tight_layout()
    out = args.output or os.path.join(d, 'ablation_facets.png')
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
