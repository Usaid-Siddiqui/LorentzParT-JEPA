"""
Phase 4 aggregation — per-task tables + the amortization figure.

Reads the per-cell JSONs written by train_task.py (run_name = {task}_{proto}_{enc}_seed{N}),
aggregates test AUC (and 1/ε_B for binary tasks) across seeds, and — if pretrain costs
are supplied — draws the headline amortization curve: cumulative compute vs. number of
downstream tasks for {scratch: N finetunes} vs {pretrain + N frozen probes / finetunes}.

Compute unit here is per-cell wall-clock (train_time_s) on one GPU — an honest proxy; a
FLOPs-based version can swap in measure_flops numbers later.

    python experiments/phase4/analyze_phase4.py --results-dir experiments/phase4/results \\
        --pretrain-cost-jepa 61200 --pretrain-cost-mae 154800   # seconds, optional
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root (for src.*)
sys.path.insert(0, _HERE)
import tasks as T

ENCODERS = ['scratch', 'jepa', 'mae']
PROTOCOLS = ['linear', 'pool', 'finetune']


def load(results_dir):
    """(task, proto, enc) → {metric: [values across seeds]}, from JSON fields."""
    cells = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(results_dir, '*.json')):
        j = json.load(open(f))
        if not all(key in j for key in ('task', 'protocol', 'encoder_label')):
            continue                      # not a Phase-4 cell JSON
        key3 = (j['task'], j['protocol'], j['encoder_label'])
        for key in ('test_auc', 'test_acc', 'bkg_rej_at_0.5', 'train_time_s'):
            if key in j:
                cells[key3][key].append(j[key])
    return cells


def ms(vals):
    return (np.mean(vals), np.std(vals), len(vals)) if vals else (np.nan, np.nan, 0)


def print_tables(cells):
    for task in T.TASKS:
        kind = T.TASKS[task]['kind']
        print(f"\n{'='*74}\n{task}  ({kind}, classes {T.TASKS[task]['classes']})\n{'='*74}")
        print(f"{'protocol':10} " + "".join(f"{e:>18}" for e in ENCODERS))
        for proto in PROTOCOLS:
            row = f"{proto:10} "
            for enc in ENCODERS:
                mu, sd, n = ms(cells[(task, proto, enc)].get('test_auc', []))
                row += f"{(f'{mu:.4f}±{sd:.4f}' if n else '—'):>18}"
            print(row)
        if kind == 'binary':      # background rejection table
            print(f"{'  (1/εB)':10} " + "".join(f"{e:>18}" for e in ENCODERS))
            for proto in PROTOCOLS:
                row = f"{proto:10} "
                for enc in ENCODERS:
                    mu, sd, n = ms(cells[(task, proto, enc)].get('bkg_rej_at_0.5', []))
                    row += f"{(f'{mu:.1f}±{sd:.1f}' if n else '—'):>18}"
                print(row)


def amortization_figure(cells, pretrain_cost, out):
    """Cumulative wall-clock vs #tasks: scratch finetune baseline vs pretrain+transfer."""
    task_order = [t for t in T.TASKS]

    def cum_time(proto, enc):
        times = []
        for t in task_order:
            mu, _, n = ms(cells[(t, proto, enc)].get('train_time_s', []))
            times.append(mu if n else np.nan)
        return np.cumsum(times)

    xs = np.arange(1, len(task_order) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))

    # baseline: N independent from-scratch finetunes (no pretrain)
    ax.plot(xs, cum_time('finetune', 'scratch'), 'o-', color='0.4', label='scratch (N finetunes)')

    for enc, col in [('jepa', 'C0'), ('mae', 'C1')]:
        off = pretrain_cost.get(enc, 0.0)
        for proto, ls in [('finetune', '-'), ('pool', '--'), ('linear', ':')]:
            y = off + cum_time(proto, enc)
            ax.plot(xs, y, ls, color=col, label=f'{enc} ({proto})')

    ax.set_xlabel('number of downstream tasks')
    ax.set_ylabel('cumulative wall-clock (s)')
    ax.set_title('Multi-task compute amortization')
    ax.set_xticks(xs)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"\nwrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='experiments/phase4/results')
    p.add_argument('--pretrain-cost-jepa', type=float, default=None, help='seconds')
    p.add_argument('--pretrain-cost-mae', type=float, default=None, help='seconds')
    p.add_argument('--out', default='experiments/phase4/results/amortization.png')
    args = p.parse_args()

    cells = load(args.results_dir)
    if not cells:
        print(f"no result JSONs in {args.results_dir}")
        return
    print_tables(cells)

    if args.pretrain_cost_jepa is not None and args.pretrain_cost_mae is not None:
        amortization_figure(
            cells,
            {'jepa': args.pretrain_cost_jepa, 'mae': args.pretrain_cost_mae},
            args.out,
        )
    else:
        print("\n(amortization figure skipped — pass --pretrain-cost-jepa/--mae in seconds)")


if __name__ == '__main__':
    main()
