"""
Phase 1 K-sweep: how many particles to mask per jet (num_mask = K).

Sweeps K over {2, 4, 8, 16} using the locked canonical config (gate ON + biased
masking, configs/pretrain_jepa_biased.yaml), overriding num_mask per run via the
--num-mask flag. The K=1 point is NOT re-run — it is reused from the 2x2 ablation
(the `gate_biased` condition in experiments/phase1/results/), so the resulting
AUC-vs-K curve includes your base case as its leftmost point.

Stage 1 (find the curve shape, cheap):
    python experiments/phase1/run_ksweep.py --seeds 42 --gpu 0
Stage 2 (error bars at the winning K — resumes, skips seed 42):
    python experiments/phase1/run_ksweep.py --seeds 42 123 456 --k-values 4 --gpu 0

Output:
    experiments/phase1/results_ksweep/seed_*.json   (k2, k4, ... + injected k1)
    experiments/phase1/results_ksweep/ksweep_auc.png

If the curve is inverted-U (rises then falls), that motivates curriculum masking
for Phase 2. If monotonic/flat, lock the best static K and scale up.
"""

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

CONFIG = 'configs/pretrain_jepa_biased.yaml'        # gate ON + biased, 20-epoch
BASE_RESULTS = 'experiments/phase1/results'          # holds gate_biased (= K=1)


def parse_args():
    p = argparse.ArgumentParser(description="K-sweep (num_mask) on gate+biased JEPA")
    p.add_argument('--k-values', nargs='+', type=int, default=[2, 4, 8, 16])
    p.add_argument('--seeds',    nargs='+', type=int, default=[42])
    p.add_argument('--gpu',      type=int, default=0)
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--output-dir', default='./experiments/phase1/results_ksweep')
    p.add_argument('--skip-run', action='store_true',
                   help='Skip training; just (re)inject K=1 and regenerate the figure')
    return p.parse_args()


def run_experiment(args):
    conds = [f'k{k}:{CONFIG}:{k}' for k in args.k_values]
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


def inject_k1(args):
    """Copy the gate_biased (K=1) result from the 2x2 into the sweep JSONs as k1."""
    for s in args.seeds:
        base_path = os.path.join(BASE_RESULTS, f'seed_{s}.json')
        out_path = os.path.join(args.output_dir, f'seed_{s}.json')
        if not (os.path.exists(base_path) and os.path.exists(out_path)):
            continue
        base = json.load(open(base_path))
        if 'gate_biased' not in base.get('conditions', {}):
            print(f"[k1] no gate_biased in {base_path}; skipping injection for seed {s}")
            continue
        out = json.load(open(out_path))
        out['conditions']['k1'] = base['conditions']['gate_biased']
        json.dump(out, open(out_path, 'w'), indent=2)
        print(f"[k1] injected gate_biased -> k1 in {out_path}")


def load_results(output_dir, seeds):
    results = []
    for s in seeds:
        path = os.path.join(output_dir, f'seed_{s}.json')
        if os.path.exists(path):
            results.append(json.load(open(path)))
    if not results:
        raise FileNotFoundError(f"No seed JSONs found in {output_dir}")
    return results


def auc_for_k(results, k):
    """Mean, std, n of test_auc for condition kK across seeds."""
    cond = f'k{k}'
    vals = [r['conditions'][cond]['test_auc'] for r in results
            if cond in r.get('conditions', {}) and 'test_auc' in r['conditions'][cond]]
    if not vals:
        return None
    return np.mean(vals), np.std(vals), len(vals)


def make_figure(results, k_values, output_path):
    ks = sorted(set([1] + list(k_values)))
    xs, means, stds = [], [], []
    for k in ks:
        stat = auc_for_k(results, k)
        if stat is None:
            continue
        xs.append(k)
        means.append(stat[0])
        stds.append(stat[1])

    n = max((auc_for_k(results, k) or (0, 0, 0))[2] for k in ks)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(xs, means, yerr=stds, marker='o', markersize=8, capsize=5,
                linewidth=2, color='#1B5E20', ecolor='gray',
                markerfacecolor='#1B5E20', markeredgecolor='black')
    for x, m in zip(xs, means):
        ax.annotate(f'{m:.3f}', (x, m), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    # mark the base case (K=1)
    if 1 in xs:
        ax.axvline(1, color='#B71C1C', linestyle='--', alpha=0.4)
        ax.annotate('base (2×2)', (1, ax.get_ylim()[0]), color='#B71C1C',
                    fontsize=8, ha='left', va='bottom')

    ax.set_xscale('log', base=2)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('K  (particles masked per jet, num_mask)')
    ax.set_ylabel('OVO ROC AUC')
    ax.set_title(f'K-sweep — gate + biased masking  (n={n} seed{"s" if n > 1 else ""}, ±1 std)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'\nSaved figure → {output_path}')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_run:
        run_experiment(args)

    inject_k1(args)
    results = load_results(args.output_dir, args.seeds)

    print(f"\n{'=' * 40}\nK-SWEEP  (gate + biased, n={len(results)})\n{'=' * 40}")
    for k in sorted(set([1] + list(args.k_values))):
        stat = auc_for_k(results, k)
        if stat:
            m, s, nn = stat
            print(f"K={k:<3}  AUC {m:.4f} ± {s:.4f}  (n={nn})")

    make_figure(results, args.k_values, os.path.join(args.output_dir, 'ksweep_auc.png'))

    # Per-epoch finetune convergence overlay across K (reads embedded training_curve)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from plot_curves import plot_convergence_from_results
    plot_convergence_from_results(
        args.output_dir,
        os.path.join(args.output_dir, 'ksweep_convergence.png'),
    )


if __name__ == '__main__':
    main()
