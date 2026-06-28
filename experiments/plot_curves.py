"""
Plot training convergence curves from per-epoch CSV logs.

Produces two figures:
  1. Pretrain val loss over wall-clock time, mean ± std shading across seeds
  2. Finetune val accuracy + val loss over epoch, mean ± std shading

CSV formats (all headerless):

  JEPA pretrain (8 cols):
    epoch, embedding_loss, val_loss, lr, ema_momentum, epoch_time_s, elapsed_total_s, best_epoch

  MAE pretrain (7 cols):
    epoch, train_loss, val_loss, lr, epoch_time_s, elapsed_total_s, best_epoch

  Finetune / scratch (6+ cols):
    epoch, train_loss, train_metric, val_loss, val_metric, lr[, elapsed_total_s]

Usage:
    python experiments/plot_curves.py \\
        --pretrain-csvs jepa:path/to/csv mae:path/to/csv \\
        --finetune-csvs jepa_finetune:path/to/csv scratch:path/to/csv \\
        --output-dir ./experiments/phase0/results
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    'jepa':          '#1565C0',
    'mae':           '#E65100',
    'jepa_finetune': '#1565C0',
    'mae_finetune':  '#E65100',
    'scratch':       '#757575',
}
LABELS = {
    'jepa':          'JEPA pretrain',
    'mae':           'MAE pretrain',
    'jepa_finetune': 'JEPA → finetune',
    'mae_finetune':  'MAE → finetune',
    'scratch':       'Scratch',
}
_FALLBACK_COLORS = ['#1B5E20', '#4A148C', '#B71C1C', '#00BCD4']


def _color(name, idx=0):
    return COLORS.get(name, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _label(name):
    return LABELS.get(name, name.replace('_', ' ').title())


def read_pretrain_csv(path):
    """Returns list of (elapsed_s, val_loss) per epoch."""
    rows = []
    with open(path, newline='') as f:
        for line in csv.reader(f):
            try:                 # skip the header row (and any blank rows)
                float(line[0])
            except (ValueError, IndexError):
                continue
            if len(line) == 8:   # JEPA: epoch,emb_loss,val_loss,lr,ema,epoch_t,elapsed,best
                rows.append((float(line[6]), float(line[2])))
            elif len(line) == 7: # MAE: epoch,train_loss,val_loss,lr,epoch_t,elapsed,best
                rows.append((float(line[5]), float(line[2])))
    return rows


def read_finetune_csv(path):
    """Returns list of (epoch, val_acc, val_loss) per epoch."""
    rows = []
    with open(path, newline='') as f:
        for line in csv.reader(f):
            try:                 # skip the header row (and any blank rows)
                float(line[0])
            except (ValueError, IndexError):
                continue
            if len(line) >= 5:  # epoch,train_loss,train_metric,val_loss,val_metric[,lr,...]
                rows.append((int(float(line[0])), float(line[4]), float(line[3])))
    return rows


def parse_csvs(specs):
    """Parse 'name:path' specs → {name: [path, ...]}"""
    grouped = defaultdict(list)
    for spec in specs:
        name, path = spec.split(':', 1)
        grouped[name].append(path)
    return grouped


def _mean_std_curve(curves):
    """
    Given a list of per-seed curves (each a list of (x, y) tuples),
    interpolate onto a common x grid and return (x, mean, std).
    """
    max_len = max(len(c) for c in curves)
    # Use x values from the longest curve as the reference grid
    ref_x = [c[0] for c in max(curves, key=len)]
    ys = []
    for curve in curves:
        xs = [p[0] for p in curve]
        y  = [p[1] for p in curve]
        ys.append(np.interp(ref_x, xs, y))
    return np.array(ref_x), np.mean(ys, axis=0), np.std(ys, axis=0)


def plot_pretrain_curves(grouped, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (name, paths) in enumerate(grouped.items()):
        curves = []
        for path in paths:
            if not os.path.exists(path):
                print(f"  [SKIP] {path} not found")
                continue
            curves.append(read_pretrain_csv(path))
        if not curves:
            continue

        x, mean, std = _mean_std_curve(curves)
        x_min = x / 60  # seconds → minutes
        color = _color(name, idx)
        ax.plot(x_min, mean, label=_label(name), color=color, linewidth=2)
        ax.fill_between(x_min, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel('Wall-clock time (min)', fontsize=11)
    ax.set_ylabel('Val loss', fontsize=11)
    ax.set_title('Pretraining convergence  (mean ± std, 3 seeds)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved → {output_path}")


def plot_finetune_curves(grouped, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for idx, (name, paths) in enumerate(grouped.items()):
        acc_curves, loss_curves = [], []
        for path in paths:
            if not os.path.exists(path):
                print(f"  [SKIP] {path} not found")
                continue
            rows = read_finetune_csv(path)
            acc_curves.append([(r[0], r[1]) for r in rows])
            loss_curves.append([(r[0], r[2]) for r in rows])
        if not acc_curves:
            continue

        color = _color(name, idx)
        lbl   = _label(name)

        x, mean, std = _mean_std_curve(acc_curves)
        axes[0].plot(x, mean, label=lbl, color=color, linewidth=2)
        axes[0].fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

        x, mean, std = _mean_std_curve(loss_curves)
        axes[1].plot(x, mean, label=lbl, color=color, linewidth=2)
        axes[1].fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    for ax, ylabel, title in zip(
        axes,
        ['Val accuracy', 'Val loss'],
        ['Finetune val accuracy', 'Finetune val loss'],
    ):
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{title}  (mean ± std, 3 seeds)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved → {output_path}")


# ── Results-dir mode (modular: overlay any experiment's per-epoch curves) ───────
#
# run_ablation.py embeds each condition's per-epoch finetune curve in the seed
# JSONs under 'training_curve' (epoch, val_acc, val_loss). This mode reads those
# directly — no CSV path juggling — so it works on any run_ablation-style results
# dir (phase1 2×2, K-sweep, ...). Point it at the dir and it discovers conditions.

def _order_conditions(conditions):
    """K-sweep conditions (k2, k4, ...) sort numerically; others alphabetically."""
    if conditions and all(re.fullmatch(r'k\d+', c) for c in conditions):
        return sorted(conditions, key=lambda c: int(c[1:]))
    return sorted(conditions)


def _condition_colors(ordered):
    """Gradient for K-sweeps (reads as a progression); fallback palette otherwise."""
    if ordered and all(re.fullmatch(r'k\d+', c) for c in ordered):
        cmap = plt.get_cmap('viridis')
        return {c: cmap(i / max(1, len(ordered) - 1)) for i, c in enumerate(ordered)}
    return {c: _color(c, i) for i, c in enumerate(ordered)}


def load_curves_from_results(results_dir, conditions=None):
    """{condition: [per-seed [(epoch, val_acc, val_loss)], ...]} from seed JSONs."""
    files = sorted(glob.glob(os.path.join(results_dir, 'seed_*.json')))
    if not files:
        raise FileNotFoundError(f"No seed_*.json in {results_dir}")

    per_cond = defaultdict(list)
    for path in files:
        conds = json.load(open(path)).get('conditions', {})
        for name, c in conds.items():
            if conditions and name not in conditions:
                continue
            curve = c.get('training_curve')
            if not curve:
                continue
            rows = [(e['epoch'], e.get('val_acc'), e.get('val_loss'))
                    for e in curve if 'val_acc' in e and 'val_loss' in e]
            if rows:
                per_cond[name].append(rows)
    return per_cond


def plot_convergence_from_results(results_dir, output_path, conditions=None):
    per_cond = load_curves_from_results(results_dir, conditions)
    if not per_cond:
        print(f"  [SKIP] no embedded training_curve found in {results_dir}")
        return

    ordered = _order_conditions(list(per_cond.keys()))
    colors = _condition_colors(ordered)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    n_seeds = max(len(v) for v in per_cond.values())

    for name in ordered:
        seed_curves = per_cond[name]
        acc = [[(r[0], r[1]) for r in sc] for sc in seed_curves]
        loss = [[(r[0], r[2]) for r in sc] for sc in seed_curves]
        color, lbl = colors[name], _label(name)

        x, mean, std = _mean_std_curve(acc)
        axes[0].plot(x, mean, label=lbl, color=color, linewidth=2)
        axes[0].fill_between(x, mean - std, mean + std, alpha=0.18, color=color)

        x, mean, std = _mean_std_curve(loss)
        axes[1].plot(x, mean, label=lbl, color=color, linewidth=2)
        axes[1].fill_between(x, mean - std, mean + std, alpha=0.18, color=color)

    for ax, ylabel, title in zip(
        axes, ['Val accuracy', 'Val loss'],
        ['Finetune val accuracy', 'Finetune val loss'],
    ):
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{title}  (mean ± std, n={n_seeds})', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved → {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrain-csvs', nargs='+', default=[],
                   metavar='NAME:PATH', help='Pretrain CSVs as name:path pairs')
    p.add_argument('--finetune-csvs', nargs='+', default=[],
                   metavar='NAME:PATH', help='Finetune CSVs as name:path pairs')
    p.add_argument('--results-dir', default=None,
                   help='Overlay finetune convergence for all conditions in a '
                        'run_ablation-style results dir (reads embedded training_curve)')
    p.add_argument('--conditions', nargs='+', default=None,
                   help='Restrict --results-dir mode to these conditions')
    p.add_argument('--output-dir', default='./experiments/phase0/results')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.results_dir:
        plot_convergence_from_results(
            args.results_dir,
            os.path.join(args.output_dir, 'convergence_finetune.png'),
            conditions=args.conditions,
        )

    if args.pretrain_csvs:
        grouped = parse_csvs(args.pretrain_csvs)
        plot_pretrain_curves(grouped, os.path.join(args.output_dir, 'pretrain_curves.png'))

    if args.finetune_csvs:
        grouped = parse_csvs(args.finetune_csvs)
        plot_finetune_curves(grouped, os.path.join(args.output_dir, 'finetune_curves.png'))


if __name__ == '__main__':
    main()
