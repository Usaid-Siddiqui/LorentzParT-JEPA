"""
General JEPA ablation runner: pretrain → finetune → eval for each condition × seed.

Each condition is a (name, pretrain_config) pair. The finetune config and eval
are shared across all conditions. Output JSONs match the Phase 0 seed_N.json
format so analyze_results.py works on both.

Must be run from the LorentzParT_JEPA/ root directory.

Usage (Phase 1 — 2x2 gate × masking ablation):

    python experiments/phase1/run_ablation.py \\
        --conditions \\
            gate_random:configs/pretrain_jepa.yaml \\
            no_gate_biased:configs/pretrain_jepa_no_gate_biased.yaml \\
            gate_biased:configs/pretrain_jepa_biased.yaml \\
        --finetune-config configs/train_lorentz_part.yaml \\
        --data-dir ./data --seeds 42 123 456 --gpu 0 \\
        --output-dir ./experiments/phase1/results

Use --skip-pretrain or --skip-finetune to resume after preemption.
"""

import csv
import os
import sys
import json
import time
import argparse
import subprocess

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.configs import LorentzParTConfig
from src.models import LorentzParT
from src.utils import set_seed
from src.utils.data import NpyJetClassDataset

NORM_DICT = {
    'pT':     (92.72917175292969,    105.83937072753906),
    'eta':    (0.0005733045982196927, 0.9174848794937134),
    'phi':    (-0.00041169871110469103, 1.8136887550354004),
    'energy': (133.8745574951172,    167.528564453125),
}
NORMALIZE = [True, False, False, True]
CLASS_NAMES = [
    'QCD/Zνν', 'H→bb', 'H→cc', 'H→gg', 'H→4q',
    'H→lνqq', 'Z→qq', 'W→qq', 't→bqq', 't→blν',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--conditions', nargs='+', required=True,
                   metavar='NAME:CONFIG',
                   help='Ablation conditions as name:pretrain_config_path pairs')
    p.add_argument('--finetune-config', default='./configs/train_lorentz_part.yaml')
    p.add_argument('--data-dir',        default='./data')
    p.add_argument('--seeds',           nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--output-dir',      default='./experiments/phase1/results')
    p.add_argument('--gpu',             type=int, default=0)
    p.add_argument('--skip-pretrain',   action='store_true')
    p.add_argument('--skip-finetune',   action='store_true')
    return p.parse_args()


def parse_conditions(specs):
    """Parse 'name:config_path[:num_mask]' strings into an ordered dict.

    Returns name -> (config_path, num_mask) where num_mask is None unless the
    optional third field is given (used for K-sweeps). Neither names nor config
    paths contain ':', so a plain split is unambiguous.
    """
    conditions = {}
    for spec in specs:
        parts = spec.split(':')
        name, config_path = parts[0], parts[1]
        num_mask = int(parts[2]) if len(parts) > 2 else None
        conditions[name] = (config_path, num_mask)
    return conditions


def read_training_curve(csv_path):
    """Read per-epoch val metrics from a finetune CSV. Returns list of dicts."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        curve = []
        for row in reader:
            entry = {'epoch': int(float(row['epoch']))}
            for key, out_key in [
                ('val_metric',      'val_acc'),
                ('val_loss',        'val_loss'),
                ('elapsed_total_s', 'elapsed_s'),
            ]:
                if key in row and row[key] not in ('', None):
                    entry[out_key] = float(row[key])
            curve.append(entry)
    return curve


def run_stage(cmd, env, desc):
    print(f"\n{'=' * 60}")
    print(f"STAGE: {desc}")
    print(f"CMD:   {' '.join(str(c) for c in cmd)}")
    print('=' * 60)
    subprocess.run([str(c) for c in cmd], check=True, env=env)


@torch.no_grad()
def evaluate_classifier(weights_path, data_dir, device, finetune_config):
    with open(finetune_config) as f:
        config = yaml.safe_load(f)
    model_cfg = LorentzParTConfig.from_dict(config['model'])
    model_cfg.inference = True

    model = LorentzParT(config=model_cfg)
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    test_dataset = NpyJetClassDataset(
        os.path.join(data_dir, 'test', 'particles.npy'),
        os.path.join(data_dir, 'test', 'labels.npy'),
        normalize=NORMALIZE, norm_dict=NORM_DICT,
    )
    loader = DataLoader(test_dataset, batch_size=1000, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_pred, all_true = [], []
    for X, y in loader:
        all_pred.append(model(X.to(device)).cpu().numpy())
        all_true.append(y.numpy())

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)

    acc = float((np.argmax(y_pred, 1) == np.argmax(y_true, 1)).mean())
    auc = float(roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo'))
    per_class_acc, per_class_auc = [], []
    for i in range(10):
        mask = np.argmax(y_true, 1) == i
        per_class_acc.append(float((np.argmax(y_pred[mask], 1) == i).mean()) if mask.sum() else 0.0)
        per_class_auc.append(float(roc_auc_score(
            (np.argmax(y_true, 1) == i).astype(int), y_pred[:, i]
        )))

    return {
        'test_acc':      acc,
        'test_auc':      auc,
        'per_class_acc': per_class_acc,
        'per_class_auc': per_class_auc,
    }


def main():
    args = parse_args()
    conditions = parse_conditions(args.conditions)
    python = sys.executable
    os.makedirs(args.output_dir, exist_ok=True)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print(f"Conditions: {list(conditions.keys())}")
    print(f"Seeds: {args.seeds}  |  GPU: {args.gpu}  |  Output: {args.output_dir}")

    for seed in args.seeds:
        set_seed(seed)
        print(f"\n{'#' * 60}\n# SEED {seed}\n{'#' * 60}")
        t0 = time.monotonic()

        # Load existing results for this seed (if any) so we can skip done conditions
        out_path = os.path.join(args.output_dir, f'seed_{seed}.json')
        if os.path.exists(out_path):
            with open(out_path) as f:
                seed_results = json.load(f)
            done = set(seed_results.get('conditions', {}).keys())
            print(f"Loaded existing results: {sorted(done)}")
        else:
            seed_results = {'seed': seed, 'conditions': {}}
            done = set()

        for cond_name, (pretrain_config, num_mask) in conditions.items():
            # Skip if this condition already has eval results
            if cond_name in done:
                print(f"\n--- Condition: {cond_name} — already done, skipping ---")
                continue

            print(f"\n--- Condition: {cond_name} ---")

            pretrain_ckpt = f'./logs/ParticleJEPA/best/{cond_name}_seed{seed}.pt'
            finetune_ckpt = f'./logs/LorentzParT/best/{cond_name}_ft_seed{seed}.pt'

            # 1. Pretrain — skip if checkpoint exists or --skip-pretrain
            if args.skip_pretrain or os.path.exists(pretrain_ckpt):
                if os.path.exists(pretrain_ckpt):
                    print(f"[SKIP pretrain] checkpoint exists: {pretrain_ckpt}")
            else:
                pretrain_cmd = [
                    python, 'scripts/pretrain_jepa.py',
                    '--data-dir', args.data_dir,
                    '--config-path', pretrain_config,
                    '--run-name', f'{cond_name}_seed{seed}',
                    '--seed', seed,
                ]
                if num_mask is not None:
                    pretrain_cmd += ['--num-mask', num_mask]
                run_stage(pretrain_cmd, env=env, desc=f'Pretrain {cond_name} seed={seed}')

            # 2. Finetune — skip if checkpoint exists or --skip-finetune
            if args.skip_finetune or os.path.exists(finetune_ckpt):
                if os.path.exists(finetune_ckpt):
                    print(f"[SKIP finetune] checkpoint exists: {finetune_ckpt}")
            else:
                run_stage(
                    [python, 'scripts/train_lorentz_part.py',
                     '--data-dir', args.data_dir,
                     '--config-path', args.finetune_config,
                     '--weights', pretrain_ckpt,
                     '--run-name', f'{cond_name}_ft_seed{seed}',
                     '--seed', seed],
                    env=env, desc=f'Finetune {cond_name} seed={seed}',
                )

            # 3. Eval
            if not os.path.exists(finetune_ckpt):
                print(f"[SKIP eval] checkpoint not found: {finetune_ckpt}")
                continue

            print(f"Evaluating {finetune_ckpt}...")
            metrics = evaluate_classifier(finetune_ckpt, args.data_dir, device, args.finetune_config)

            ft_csv = f'./logs/LorentzParT/logging/{cond_name}_ft_seed{seed}.csv'
            metrics['training_curve'] = read_training_curve(ft_csv)

            seed_results['conditions'][cond_name] = metrics

            print(f"  test_acc={metrics['test_acc']:.4f}  test_auc={metrics['test_auc']:.4f}  curve_epochs={len(metrics['training_curve'])}")

            # Save after each condition so progress isn't lost
            with open(out_path, 'w') as f:
                json.dump(seed_results, f, indent=2)
            print(f"Saved → {out_path}")

        elapsed = time.monotonic() - t0
        print(f"\nSeed {seed} done in {elapsed / 60:.1f} min")

    print(f"\nAll done. Aggregate with:")
    print(f"  python experiments/analyze_results.py \\")
    print(f"    --results-dir {args.output_dir} \\")
    print(f"    --conditions {' '.join(conditions.keys())}")


if __name__ == '__main__':
    main()
