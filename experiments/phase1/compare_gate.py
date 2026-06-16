"""
Compare JEPA pretrain + finetune WITH vs WITHOUT the attention gate.

Both runs use BIASED masking and a 15-epoch pretrain (the *_quick configs); only the
attention gate differs. Runs pretrain -> finetune -> eval for each, then writes a
comparison figure (overall AUC/accuracy + per-class AUC).

Run AFTER the AttentionGate fix is applied.

    python experiments/phase1/compare_gate.py --seeds 42 --gpu 0
    python experiments/phase1/compare_gate.py --seeds 42 123 456 --gpu 0   # error bars

Output:
    experiments/phase1/results_gatecompare/seed_*.json
    experiments/phase1/results_gatecompare/gate_comparison.png

Checkpoint names are gate_on / gate_off (fresh) so this does NOT reuse the old
no-op-gate checkpoints — it re-pretrains with the fixed gate.
"""

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# condition_name : pretrain_config (both biased, 15-epoch; differ only in the gate)
CONDITIONS = {
    'gate_on':  'configs/pretrain_jepa_biased_quick.yaml',          # use_attention_gate: true
    'gate_off': 'configs/pretrain_jepa_no_gate_biased_quick.yaml',  # use_attention_gate: false
}
ORDER  = ['gate_on', 'gate_off']
LABELS = {'gate_on': 'Gate ON', 'gate_off': 'Gate OFF'}
COLORS = {'gate_on': '#1B5E20', 'gate_off': '#B71C1C'}
CLASS_NAMES = ['QCD/Zνν', 'H→bb', 'H→cc', 'H→gg', 'H→4q',
               'H→lνqq', 'Z→qq', 'W→qq', 't→bqq', 't→blν']


def parse_args():
    p = argparse.ArgumentParser(description="Gate vs no-gate comparison (biased, 15-ep pretrain)")
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--output-dir', default='./experiments/phase1/results_gatecompare')
    p.add_argument('--skip-run', action='store_true',
                   help='Skip training; just (re)generate the figure from existing JSONs')
    return p.parse_args()


def run_experiment(args):
    conds = [f'{name}:{cfg}' for name, cfg in CONDITIONS.items()]
    cmd = [
        sys.executable, 'experiments/phase1/run_ablation.py',
        '--conditions', *conds,
        '--finetune-config', 'configs/train_lorentz_part.yaml',
        '--data-dir', args.data_dir,
        '--seeds', *[str(s) for s in args.seeds],
        '--gpu', str(args.gpu),
        '--output-dir', args.output_dir,
    ]
    subprocess.run([str(c) for c in cmd], check=True)


def load_results(output_dir, seeds):
    results = []
    for s in seeds:
        path = os.path.join(output_dir, f'seed_{s}.json')
        if os.path.exists(path):
            results.append(json.load(open(path)))
    if not results:
        raise FileNotFoundError(f"No seed JSONs found in {output_dir}")
    return results


def scalar(results, cond, key):
    """Mean, std of a scalar metric across seeds."""
    vals = [r['conditions'][cond][key] for r in results
            if cond in r['conditions'] and key in r['conditions'][cond]]
    return (np.mean(vals), np.std(vals)) if vals else (np.nan, 0.0)


def per_class(results, cond):
    """(n_seeds, 10) array of per-class AUC."""
    rows = [r['conditions'][cond]['per_class_auc'] for r in results
            if cond in r['conditions'] and len(r['conditions'][cond].get('per_class_auc', [])) == 10]
    return np.array(rows) if rows else None


def make_figure(results, output_path):
    n = len(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={'width_ratios': [1, 2.4]})

    # ── Panel 1: overall AUC and accuracy ──────────────────────────────────────
    metrics = [('test_auc', 'OVO ROC AUC'), ('test_acc', 'Accuracy')]
    x = np.arange(len(metrics))
    w = 0.38
    for i, cond in enumerate(ORDER):
        means = [scalar(results, cond, m)[0] for m, _ in metrics]
        stds  = [scalar(results, cond, m)[1] for m, _ in metrics]
        bars = ax1.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=4,
                       label=LABELS[cond], color=COLORS[cond],
                       edgecolor='black', linewidth=0.8, alpha=0.9)
        for b, mval in zip(bars, means):
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                     f'{mval:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in metrics])
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Score')
    ax1.set_title('Overall')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # AUC delta annotation (the headline)
    on, _ = scalar(results, 'gate_on', 'test_auc')
    off, _ = scalar(results, 'gate_off', 'test_auc')
    if not (np.isnan(on) or np.isnan(off)):
        ax1.text(0.5, 0.02, f'ΔAUC (on−off) = {on - off:+.4f}',
                 transform=ax1.transAxes, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', fc='#FFF9C4', ec='gray'))

    # ── Panel 2: per-class AUC ──────────────────────────────────────────────────
    xc = np.arange(10)
    for i, cond in enumerate(ORDER):
        arr = per_class(results, cond)
        if arr is None:
            continue
        m, s = arr.mean(0), arr.std(0)
        ax2.bar(xc + (i - 0.5) * w, m, w, yerr=s, capsize=2,
                label=LABELS[cond], color=COLORS[cond],
                edgecolor='black', linewidth=0.5, alpha=0.9)
    ax2.set_xticks(xc)
    ax2.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('OVO ROC AUC')
    ax2.set_title('Per-class')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Attention gate vs no gate  —  biased masking, 15-epoch pretrain  '
                 f'(n={n} seed{"s" if n > 1 else ""})', fontsize=12)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'\nSaved figure → {output_path}')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_run:
        run_experiment(args)

    results = load_results(args.output_dir, args.seeds)

    # Console summary
    print(f"\n{'=' * 48}\nGATE ON vs OFF  (biased, 15-ep pretrain, n={len(results)})\n{'=' * 48}")
    for m, lbl in [('test_auc', 'AUC'), ('test_acc', 'Acc')]:
        on_m, on_s = scalar(results, 'gate_on', m)
        off_m, off_s = scalar(results, 'gate_off', m)
        print(f"{lbl:4s}  ON {on_m:.4f}±{on_s:.4f}  |  OFF {off_m:.4f}±{off_s:.4f}  |  Δ {on_m - off_m:+.4f}")

    make_figure(results, os.path.join(args.output_dir, 'gate_comparison.png'))


if __name__ == '__main__':
    main()
