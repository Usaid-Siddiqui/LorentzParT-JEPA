"""
Phase 6 pretrain-compute comparison — does JEPA or MAE converge faster, and at what cost?

Reads the per-epoch pretrain CSVs and reports, per method across seeds:
  best_ep    epoch of minimum val_loss (convergence in EPOCHS)
  sec@best   wall-clock (elapsed_total_s) at that epoch — the compute charged for the
             DEPLOYED encoder (JEPA finetunes from its best-val *_best.pt, MAE from its best)
  last_ep    epochs actually run (JEPA = full 30; MAE early-stops on patience)
  sec_total  full-schedule wall-clock

JEPA's val embedding loss is a noisy proxy for representation quality, but best-val is
exactly the checkpoint we transfer, so sec@best is the honest pretrain bill for it.
Wall-clock is noisy on a shared GPU — read the ratio, not the absolute seconds.

    python experiments/phase6_fullfeature/pretrain_compute.py --seeds 42 123 456
"""

import argparse
import csv
import os
import numpy as np


def read_pretrain_csv(path):
    """[(epoch, val_loss, elapsed_s), ...] via header names; keep latest run on resume."""
    rows = []
    with open(path) as f:
        for d in csv.DictReader(f):
            try:
                ep = int(float(d['epoch']))
                val = float(d['val_loss'])
                el = float(d['elapsed_total_s'])
            except (KeyError, ValueError, TypeError):
                continue
            if rows and el < rows[-1][2]:          # elapsed jumped back → resume/append
                rows = []
            rows.append((ep, val, el))
    return rows


def summarize(path):
    rows = read_pretrain_csv(path)
    if not rows:
        return None
    best = min(rows, key=lambda t: t[1])           # (epoch, val, elapsed) at min val_loss
    last = rows[-1]
    return dict(best_epoch=best[0], sec_at_best=best[2],
                last_epoch=last[0], sec_total=last[2])


def agg(paths):
    s = [x for x in (summarize(p) for p in paths) if x]
    if not s:
        return None
    return {k: (np.mean([x[k] for x in s]), np.std([x[k] for x in s]), len(s)) for k in s[0]}


def main():
    p = argparse.ArgumentParser(description="Phase 6 pretrain convergence / compute")
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--jepa-dir', default='./logs/ParticleJEPA/logging')
    p.add_argument('--mae-dir', default='./logs/LorentzParT/logging')
    p.add_argument('--jepa-stem', default='jepa_full_seed{s}')
    p.add_argument('--mae-stem', default='mae_full_seed{s}')
    args = p.parse_args()

    methods = [
        ('JEPA (best-val)', args.jepa_dir, args.jepa_stem),
        ('MAE',             args.mae_dir,  args.mae_stem),
    ]
    print(f"{'method':18s}{'best_ep':>9s}{'sec@best':>11s}{'last_ep':>9s}{'sec_total':>11s}   (n)")
    res = {}
    for name, d, stem in methods:
        paths = [os.path.join(d, stem.format(s=s) + '.csv') for s in args.seeds]
        missing = [p for p in paths if not os.path.exists(p)]
        for mp in missing:
            print(f"  [warn] missing {mp}")
        a = agg([p for p in paths if os.path.exists(p)])
        res[name] = a
        if a is None:
            print(f"{name:18s}  no CSVs found")
            continue
        print(f"{name:18s}{a['best_epoch'][0]:9.1f}{a['sec_at_best'][0]:11.0f}"
              f"{a['last_epoch'][0]:9.1f}{a['sec_total'][0]:11.0f}   (n={a['best_epoch'][2]})")

    j, m = res.get('JEPA (best-val)'), res.get('MAE')
    if j and m and j['sec_at_best'][0] > 0:
        print(f"\nDeployed-encoder pretrain cost: JEPA {j['sec_at_best'][0]:.0f}s @ ep{j['best_epoch'][0]:.0f}"
              f"  vs  MAE {m['sec_at_best'][0]:.0f}s @ ep{m['best_epoch'][0]:.0f}"
              f"  →  JEPA {m['sec_at_best'][0]/j['sec_at_best'][0]:.1f}× cheaper.")


if __name__ == '__main__':
    main()
