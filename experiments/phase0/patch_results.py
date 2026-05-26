"""
One-off script to backfill best_val_acc and pretrain_time_s into existing
seed JSON files that were produced when CSVs had no header row.

Run from LorentzParT_JEPA/ root:
    python experiments/phase0/patch_results.py --results-dir ./results/phase0
"""

import os
import sys
import csv
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_FINETUNE_COLS      = ['epoch', 'train_loss', 'train_metric', 'val_loss',
                       'val_metric', 'learning_rate']
_JEPA_PRETRAIN_COLS = ['epoch', 'embedding_loss', 'val_loss', 'learning_rate',
                       'ema_momentum', 'epoch_time_s', 'elapsed_total_s', 'best_epoch']
_MAE_PRETRAIN_COLS  = ['epoch', 'train_loss', 'val_loss', 'learning_rate',
                       'epoch_time_s', 'elapsed_total_s', 'best_epoch']


def _read_csv_rows(path, fallback_cols=None):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        raw = f.read().strip()
    if not raw:
        return []
    first_field = raw.split('\n')[0].split(',')[0].strip()
    try:
        float(first_field)
        has_header = False
    except ValueError:
        has_header = True

    with open(path, newline='') as f:
        if has_header:
            return list(csv.DictReader(f))
        elif fallback_cols:
            return [dict(zip(fallback_cols, row)) for row in csv.reader(f)]
        else:
            return []


def read_best(path, column, fallback_cols=None):
    rows = _read_csv_rows(path, fallback_cols=fallback_cols)
    vals = [float(r[column]) for r in rows if column in r]
    return max(vals) if vals else None


def read_last(path, column, fallback_cols=None):
    rows = _read_csv_rows(path, fallback_cols=fallback_cols)
    for row in reversed(rows):
        if column in row:
            return float(row[column])
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='./results/phase0')
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.results_dir, 'seed_*.json')))
    if not files:
        print(f"No seed_*.json found in {args.results_dir}")
        return

    for path in files:
        with open(path) as f:
            data = json.load(f)

        seed = data['seed']
        changed = False

        ft_map = {
            'jepa_finetune': f'./logs/LorentzParT/logging/jepa_ft_seed{seed}.csv',
            'mae_finetune':  f'./logs/LorentzParT/logging/mae_ft_seed{seed}.csv',
            'scratch':       f'./logs/LorentzParT/logging/scratch_seed{seed}.csv',
        }
        for label, csv_path in ft_map.items():
            if label not in data.get('conditions', {}):
                continue
            val = read_best(csv_path, 'val_metric', fallback_cols=_FINETUNE_COLS)
            if val is not None and data['conditions'][label].get('best_val_acc') is None:
                data['conditions'][label]['best_val_acc'] = val
                print(f"  seed {seed} | {label}: best_val_acc = {val:.4f}")
                changed = True

        jepa_pt_csv = f'./logs/ParticleJEPA/logging/jepa_seed{seed}.csv'
        mae_pt_csv  = f'./logs/LorentzParT/logging/mae_seed{seed}.csv'

        t = read_last(jepa_pt_csv, 'elapsed_total_s', fallback_cols=_JEPA_PRETRAIN_COLS)
        if t is not None and data['conditions'].get('jepa_finetune', {}).get('pretrain_time_s') is None:
            data['conditions']['jepa_finetune']['pretrain_time_s'] = t
            print(f"  seed {seed} | jepa pretrain_time_s = {t:.1f}s")
            changed = True

        t = read_last(mae_pt_csv, 'elapsed_total_s', fallback_cols=_MAE_PRETRAIN_COLS)
        if t is not None and data['conditions'].get('mae_finetune', {}).get('pretrain_time_s') is None:
            data['conditions']['mae_finetune']['pretrain_time_s'] = t
            print(f"  seed {seed} | mae  pretrain_time_s = {t:.1f}s")
            changed = True

        if changed:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Updated {path}")
        else:
            print(f"  seed {seed}: nothing to patch")


if __name__ == '__main__':
    main()
