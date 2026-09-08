"""
Phase 6 convergence curves — globs the per-epoch CSVs and calls experiments/plot_curves.py.

Produces:
  pretrain_curves.png              JEPA vs MAE val loss over wall-clock (mean±std across seeds)
  <task>_finetune_curves.png       val accuracy + val loss over epoch, scratch/jepa/mae (mean±std)

    python experiments/phase6_fullfeature/plot_convergence.py --tasks full10 hbb_hcc wz

Run on the machine with the logs. The MAE hbb_hcc band will be wide — that's the reproducible
seed-42 collapse showing up honestly, not a plotting bug.
"""

import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'experiments'))
import plot_curves as pc  # noqa: E402


def g(pattern):
    return sorted(glob.glob(pattern))


def main():
    p = argparse.ArgumentParser(description="Phase 6 convergence curves")
    p.add_argument('--tasks', nargs='+', default=['full10', 'hbb_hcc', 'wz'])
    p.add_argument('--lp-dir', default='logs/LorentzParT/logging',
                   help='MAE-pretrain + all finetune CSVs live here')
    p.add_argument('--jepa-dir', default='logs/ParticleJEPA/logging')
    p.add_argument('--out', default='experiments/phase6_fullfeature/results')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── pretrain: JEPA vs MAE val loss over wall-clock ──
    pre = {}
    j = g(f'{args.jepa_dir}/jepa_full_seed*.csv')
    m = g(f'{args.lp_dir}/mae_full_seed*.csv')
    if j:
        pre['jepa'] = j
    if m:
        pre['mae'] = m
    if pre:
        pc.plot_pretrain_curves(pre, os.path.join(args.out, 'pretrain_curves.png'))
    else:
        print("[skip] no pretrain CSVs found")

    # ── finetune: one figure per task, scratch/jepa/mae overlaid ──
    for task in args.tasks:
        grouped = {}
        for enc, name in [('scratch', 'scratch'), ('jepa', 'jepa_finetune'), ('mae', 'mae_finetune')]:
            paths = g(f'{args.lp_dir}/{task}_finetune_{enc}_seed*.csv')
            if paths:
                grouped[name] = paths
        if grouped:
            pc.plot_finetune_curves(grouped, os.path.join(args.out, f'{task}_finetune_curves.png'))
        else:
            print(f"[skip] no finetune CSVs for task {task}")


if __name__ == '__main__':
    main()
