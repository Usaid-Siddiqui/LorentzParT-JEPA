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
        fixed_scratch = None
        for proto in PROTOCOLS:
            f = collect(args.logs_dir, model, proto, args.scale, args.seeds, False)
            b = collect(args.logs_dir, model, proto, args.scale, args.seeds, True)
            fs = f"{np.mean(f):.4f}±{np.std(f):.4f}" if f else "—"
            bs = f"{np.mean(b):.4f}±{np.std(b):.4f}" if b else "—"
            d = f"{np.mean(f)-np.mean(b):+.4f}" if (f and b) else "—"
            print(f"{proto:12}{fs:>22}{bs:>22}{d:>14}")
            if proto == 'scratch' and f:
                fixed_scratch = np.mean(f)

        if fixed_scratch is not None:
            print(f"\n  SSL vs scratch on the FIXED code (the thesis question):")
            for proto in ('jepa', 'mae'):
                f = collect(args.logs_dir, model, proto, args.scale, args.seeds, False)
                if not f:
                    print(f"    {proto:6} —  (not run)"); continue
                delta = np.mean(f) - fixed_scratch
                # a crude significance hint: seed spread of the SSL arm
                spread = np.std(f) if len(f) > 1 else float('nan')
                verdict = ("within seed noise" if not np.isnan(spread) and abs(delta) < spread
                           else "outside seed spread" if not np.isnan(spread) else "n=1, no error bar")
                print(f"    {proto:6} {delta:+.4f} vs scratch   ({verdict})")


if __name__ == '__main__':
    main()
