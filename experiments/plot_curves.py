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
import os
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrain-csvs', nargs='+', default=[],
                   metavar='NAME:PATH', help='Pretrain CSVs as name:path pairs')
    p.add_argument('--finetune-csvs', nargs='+', default=[],
                   metavar='NAME:PATH', help='Finetune CSVs as name:path pairs')
    p.add_argument('--output-dir', default='./experiments/phase0/results')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.pretrain_csvs:
        grouped = parse_csvs(args.pretrain_csvs)
        plot_pretrain_curves(grouped, os.path.join(args.output_dir, 'pretrain_curves.png'))

    if args.finetune_csvs:
        grouped = parse_csvs(args.finetune_csvs)
        plot_finetune_curves(grouped, os.path.join(args.output_dir, 'finetune_curves.png'))


if __name__ == '__main__':
    main()
