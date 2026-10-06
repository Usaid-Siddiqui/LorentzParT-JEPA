"""
Phase 8 revalidation table — does SSL help now that pretraining is not silently crippled?

Reads best `val_metric` from the per-epoch CSVs and prints scratch / JEPA / MAE side by side,
with the fixed-vs-broken delta when both arms exist.

    python experiments/phase8_revalidate/compare_phase8.py --scale 1m --seeds 42 123 456
    python experiments/phase8_revalidate/compare_phase8.py --scale 1m --seeds 42 --models part lorentzpart
"""

import argparse
import csv
import os

import numpy as np

MODEL_DIR = {'part': 'ParticleTransformer', 'lorentzpart': 'LorentzParT'}
PROTOCOLS = ['scratch', 'jepa', 'mae']


def best_val(path):
    if not os.path.exists(path):
        return None
    best = None
    for row in csv.DictReader(open(path)):
        try:
            v = float(row['val_metric'])
        except (KeyError, ValueError, TypeError):
            continue
        best = v if best is None else max(best, v)
    return best


def collect(logs, model, proto, scale, seeds, broken):
    sfx = '_broken' if broken else ''
    out = []
    for s in seeds:
        v = best_val(os.path.join(logs, MODEL_DIR[model], 'logging',
                                  f'{model}_{proto}_{scale}_seed{s}{sfx}.csv'))
        if v is not None:
            out.append(v)
    return out


def main():
    p = argparse.ArgumentParser(description="Phase 8 revalidation comparison")
    p.add_argument('--logs-dir', default='./logs')
    p.add_argument('--scale', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--models', nargs='+', default=['lorentzpart'],
                   choices=['part', 'lorentzpart'])
    args = p.parse_args()

    for model in args.models:
        print(f"\n{'='*74}\n{model}  @ {args.scale}  (best val accuracy, mean±std over "
              f"{len(args.seeds)} seed(s))\n{'='*74}")
        print(f"{'protocol':12}{'FIXED':>22}{'broken (control)':>22}{'Δ':>14}")
        scratch_vals = []
        for proto in PROTOCOLS:
            f = collect(args.logs_dir, model, proto, args.scale, args.seeds, False)
            b = collect(args.logs_dir, model, proto, args.scale, args.seeds, True)
            fs = f"{np.mean(f):.4f}±{np.std(f):.4f}" if f else "—"
            bs = f"{np.mean(b):.4f}±{np.std(b):.4f}" if b else "—"
            d = f"{np.mean(f)-np.mean(b):+.4f}" if (f and b) else "—"
            print(f"{proto:12}{fs:>22}{bs:>22}{d:>14}")
            if proto == 'scratch' and f:
                scratch_vals = f

        if scratch_vals:
            print(f"\n  SSL vs scratch on the FIXED code (the thesis question):")
            print(f"    {'contrast':20}{'Δ':>10}{'SE':>9}{'t':>7}{'95% CI':>22}")
            for proto in ('jepa', 'mae'):
                f = collect(args.logs_dir, model, proto, args.scale, args.seeds, False)
                if not f:
                    print(f"    {proto+' - scratch':20}{'—':>10}   (not run)"); continue
                d = np.mean(f) - np.mean(scratch_vals)
                n_a, n_b = len(scratch_vals), len(f)
                if n_a < 2 or n_b < 2:
                    print(f"    {proto+' - scratch':20}{d:>+10.4f}   (n=1, no error bar)"); continue
                # Welch SE on the difference of means. np.std is population std (ddof=0),
                # so convert to the sample std first. Comparing |delta| against ONE arm's
                # spread is not a test — it ignores the other arm's variance entirely.
                se_a = np.std(scratch_vals, ddof=1) / np.sqrt(n_a)
                se_b = np.std(f, ddof=1) / np.sqrt(n_b)
                se = float(np.hypot(se_a, se_b))
                t = d / se if se > 0 else float('inf')
                crit = 2.78 if min(n_a, n_b) <= 3 else 2.0     # ~t_{0.975}
                lo, hi = d - crit * se, d + crit * se
                flag = '' if lo <= 0 <= hi else '  <-- excludes 0'
                print(f"    {proto+' - scratch':20}{d:>+10.4f}{se:>9.4f}{t:>7.2f}"
                      f"   [{lo:+.4f}, {hi:+.4f}]{flag}")
            print("    (a CI straddling 0 means no detectable difference from scratch)")


if __name__ == '__main__':
    main()
