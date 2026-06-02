"""
Aggregate Phase 0 multi-seed results and produce a summary table and figures.

Reads all seed_*.json files written by run_phase0.py and computes mean ± std
across seeds for each condition and metric.

Must be run from the LorentzParT_JEPA/ root directory:

    python experiments/phase0/analyze_results.py --results-dir ./experiments/phase0/results
"""

import os
import sys
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CONDITION_ORDER  = ['jepa_finetune', 'mae_finetune', 'scratch', 'jepa_probe', 'mae_probe']
CONDITION_LABELS = {
    'jepa_finetune': 'JEPA → finetune',
    'mae_finetune':  'MAE → finetune',
    'scratch':       'Scratch',
    'jepa_probe':    'JEPA → probe',
    'mae_probe':     'MAE → probe',
}
CONDITION_COLORS = {
    'jepa_finetune': '#1565C0',
    'mae_finetune':  '#E65100',
    'scratch':       '#757575',
    'jepa_probe':    '#90CAF9',
    'mae_probe':     '#FFCC80',
}
CLASS_NAMES = [
    'QCD/Zνν', 'H→bb', 'H→cc', 'H→gg', 'H→4q',
    'H→lνqq', 'Z→qq', 'W→qq', 't→bqq', 't→blν',
]


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate Phase 0 results")
    p.add_argument('--results-dir', default='./experiments/phase0/results')
    p.add_argument('--output-dir',  default=None, help='Defaults to --results-dir')
    return p.parse_args()


def load_results(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, 'seed_*.json')))
    if not files:
        raise FileNotFoundError(f"No seed_*.json found in {results_dir}")
    results = []
    for path in files:
        with open(path) as f:
            results.append(json.load(f))
    return results


def collect(all_results, condition, metric, nested=None):
    """Pull a scalar metric for a condition across all seeds."""
    vals = []
    for r in all_results:
        node = r.get('conditions', {}).get(condition, {})
        if nested:
            node = node.get(nested, {})
        v = node.get(metric)
        if v is not None:
            vals.append(float(v))
    return vals


def collect_per_class(all_results, condition):
    """Return (10, n_seeds) array of per-class accuracies, None where missing."""
    rows = []
    for r in all_results:
        pc = r.get('conditions', {}).get(condition, {}).get('per_class_acc', [])
        if len(pc) == 10:
            rows.append(pc)
    return np.array(rows) if rows else None  # (n_seeds, 10)


# ── Tables ────────────────────────────────────────────────────────────────────

def print_summary_table(all_results):
    n = len(all_results)
    cols = ['test_auc', 'test_acc', 'best_val_acc', 'pretrain_time_s']
    col_w = 22

    header = f"{'Condition':<22}"
    for c in cols:
        header += f"  {c:>{col_w}}"
    sep = '─' * len(header)

    print(f"\n{'Phase 0 Summary':^{len(header)}}")
    print(f"{'(n=' + str(n) + ' seeds)':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)

    for cond in CONDITION_ORDER:
        label = CONDITION_LABELS.get(cond, cond)
        row = f"{label:<22}"
        for metric in cols:
            nested = 'embedding_stats' if metric.startswith('embed') else None
            vals = collect(all_results, cond, metric, nested=nested)
            if vals:
                row += f"  {np.mean(vals):>9.4f} ± {np.std(vals):.4f}   "
            else:
                row += f"  {'N/A':>{col_w}}"
        print(row)

    print(sep)


def collect_per_class_metric(all_results, condition, key='per_class_acc'):
    """Return (n_seeds, 10) array for per-class acc or auc."""
    rows = []
    for r in all_results:
        pc = r.get('conditions', {}).get(condition, {}).get(key, [])
        if len(pc) == 10:
            rows.append(pc)
    return np.array(rows) if rows else None


def print_per_class_table(all_results):
    conds = ['jepa_finetune', 'mae_finetune', 'scratch']

    for metric_key, metric_label in [('per_class_auc', 'AUC'), ('per_class_acc', 'Accuracy')]:
        print(f"\n{'Per-class ' + metric_label + '  (mean ± std across seeds)'}")
        header = f"{'Class':<14}"
        for c in conds:
            header += f"  {CONDITION_LABELS[c]:>20}"
        print('─' * len(header))
        print(header)
        print('─' * len(header))

        arrays = {c: collect_per_class_metric(all_results, c, key=metric_key) for c in conds}

        for i, name in enumerate(CLASS_NAMES):
            row = f"{name:<14}"
            for c in conds:
                arr = arrays[c]
                if arr is not None and arr.shape[0] > 0:
                    m, s = arr[:, i].mean(), arr[:, i].std()
                    row += f"  {m:>8.4f} ± {s:.4f}      "
                else:
                    row += f"  {'N/A':>20}"
            print(row)

        print('─' * len(header))


def print_embedding_table(all_results):
    keys   = ['effective_rank', 'mean_var', 'mean_cos_sim']
    labels = ['Effective Rank', 'Mean Variance', 'Mean Cos Sim']
    print(f"\n{'Embedding collapse diagnostics  (mean ± std)'}")
    header = f"{'Condition':<22}" + "".join(f"  {l:>18}" for l in labels)
    print('─' * len(header))
    print(header)
    print('─' * len(header))
    for cond in ['jepa_finetune', 'mae_finetune']:
        row = f"{CONDITION_LABELS[cond]:<22}"
        for key in keys:
            vals = collect(all_results, cond, key, nested='embedding_stats')
            if vals:
                row += f"  {np.mean(vals):>9.4f} ± {np.std(vals):.4f}  "
            else:
                row += f"  {'N/A':>18}"
        print(row)
    print('─' * len(header))


# ── Figures ───────────────────────────────────────────────────────────────────

def _bar_plot(all_results, metric, ylabel, title, output_path):
    means, stds, labels, colors = [], [], [], []
    for cond in CONDITION_ORDER:
        vals = collect(all_results, cond, metric)
        if not vals:
            continue
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        labels.append(CONDITION_LABELS[cond])
        colors.append(CONDITION_COLORS[cond])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(means))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.8, alpha=0.9)
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.004,
            f'{m:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{title}  (n={len(all_results)} seeds, ±1 std)', fontsize=11)
    ax.set_ylim(0, min(1.0, max(means) * 1.3) if means else 1.0)
    ax.grid(axis='y', alpha=0.35)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved → {output_path}")


def plot_accuracy_bars(all_results, output_path):
    _bar_plot(all_results, 'test_auc', 'Macro-OVO ROC AUC',
              'Phase 0 — ROC AUC by condition', output_path)


def _plot_per_class(all_results, metric_key, ylabel, title, output_path):
    conds = ['jepa_finetune', 'mae_finetune', 'scratch']
    arrays = {c: collect_per_class_metric(all_results, c, key=metric_key) for c in conds}

    x = np.arange(10)
    width = 0.28
    fig, ax = plt.subplots(figsize=(14, 5))

    for k, cond in enumerate(conds):
        arr = arrays[cond]
        if arr is None:
            continue
        means = arr.mean(axis=0)
        stds  = arr.std(axis=0)
        ax.bar(x + (k - 1) * width, means, width,
               yerr=stds, capsize=3,
               label=CONDITION_LABELS[cond],
               color=CONDITION_COLORS[cond],
               edgecolor='black', linewidth=0.5, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}  (mean ± std)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.35)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved → {output_path}")


def plot_per_class_bars(all_results, output_path):
    _plot_per_class(all_results, 'per_class_auc', 'ROC AUC',
                    'Per-class ROC AUC by condition', output_path)


def plot_embedding_stats(all_results, output_path):
    keys   = ['effective_rank', 'mean_var', 'mean_cos_sim']
    titles = ['Effective Rank\n(collapse → 1)', 'Mean Variance\n(collapse → 0)',
              'Mean Cosine Sim\n(collapse → 1)']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, title in zip(axes, keys, titles):
        for cond, color in [('jepa_finetune', '#1565C0'), ('mae_finetune', '#E65100')]:
            vals = collect(all_results, cond, key, nested='embedding_stats')
            if vals:
                ax.bar(
                    [CONDITION_LABELS[cond].replace(' → finetune', '')],
                    [np.mean(vals)], yerr=[np.std(vals)],
                    color=color, capsize=5, edgecolor='black', linewidth=0.8, alpha=0.9,
                )
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.35)

    fig.suptitle('Embedding collapse diagnostics  (mean ± std across seeds)', fontsize=11)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = args.output_dir or args.results_dir

    all_results = load_results(args.results_dir)
    print(f"Loaded {len(all_results)} seed(s) from {args.results_dir}")

    print_summary_table(all_results)
    print_per_class_table(all_results)
    print_embedding_table(all_results)

    plot_accuracy_bars(all_results,  os.path.join(out_dir, 'auc_bars.png'))
    _bar_plot(all_results, 'test_acc', 'Test Accuracy',
              'Phase 0 — Test accuracy by condition',
              os.path.join(out_dir, 'accuracy_bars.png'))
    plot_per_class_bars(all_results, os.path.join(out_dir, 'per_class_auc_bars.png'))
    _plot_per_class(all_results, 'per_class_acc', 'Accuracy',
                    'Per-class accuracy by condition',
                    os.path.join(out_dir, 'per_class_bars.png'))
    plot_embedding_stats(all_results, os.path.join(out_dir, 'embedding_stats.png'))

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == '__main__':
    main()
