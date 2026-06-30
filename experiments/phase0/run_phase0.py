"""
Phase 0 multi-seed experiment orchestrator.

Runs JEPA pretraining, MAE pretraining, fine-tuning (3 conditions), and
linear probing (2 conditions) for each seed, then evaluates all models
and writes structured JSON results for downstream analysis.

Must be run from the LorentzParT_JEPA/ root directory:

    python experiments/phase0/run_phase0.py \\
        --data-dir ./data \\
        --seeds 42 123 456 \\
        --gpu 0

Skip flags let you resume mid-experiment:

    --skip-pretrain   use existing checkpoint files, skip training stages
    --skip-finetune
    --skip-probe

Outputs per seed:  experiments/phase0/results/seed_{seed}.json
Summary:           run experiments/analyze_results.py after all seeds complete.
"""

import os
import sys
import csv
import json
import time
import argparse
import subprocess
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
warnings.filterwarnings('ignore')

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.configs import LorentzParTConfig
from src.models import LorentzParT
from src.models.lorentz_part import LorentzParTEncoder
from src.models.processor import ParticleProcessor
from src.utils import set_seed
from src.utils.data import NpyJetClassDataset
from src.utils.embedding_stats import probe_encoder_stats

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


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 multi-seed experiment")
    p.add_argument('--data-dir',         default='./data')
    p.add_argument('--seeds',            nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--output-dir',       default='./experiments/phase0/results')
    p.add_argument('--gpu',              type=int, default=0)
    p.add_argument('--pretrain-config',  default='./configs/pretrain_jepa.yaml')
    p.add_argument('--mae-config',       default='./configs/pretrain_mae.yaml')
    p.add_argument('--finetune-config',  default='./configs/train_lorentz_part.yaml')
    p.add_argument('--probe-config',     default='./experiments/phase0/configs/linear_probe.yaml')
    p.add_argument('--skip-pretrain',    action='store_true')
    p.add_argument('--skip-finetune',    action='store_true')
    p.add_argument('--skip-probe',       action='store_true')
    return p.parse_args()


# ── Subprocess helpers ────────────────────────────────────────────────────────

def run_stage(cmd, env, desc):
    print(f"\n{'=' * 60}")
    print(f"STAGE: {desc}")
    print(f"CMD:   {' '.join(str(c) for c in cmd)}")
    print('=' * 60)
    subprocess.run([str(c) for c in cmd], check=True, env=env)


# ── CSV log readers ───────────────────────────────────────────────────────────

# Known column orders for headerless CSVs (caused by a bug where
# _set_logging_paths unconditionally set _log_header_written = True).
# Order matches the `logs` dict in each trainer's log_csv call.
_FINETUNE_COLS      = ['epoch', 'train_loss', 'train_metric', 'val_loss',
                       'val_metric', 'learning_rate']
_JEPA_PRETRAIN_COLS = ['epoch', 'embedding_loss', 'val_loss', 'learning_rate',
                       'ema_momentum', 'epoch_time_s', 'elapsed_total_s', 'best_epoch']
_MAE_PRETRAIN_COLS  = ['epoch', 'train_loss', 'val_loss', 'learning_rate',
                       'epoch_time_s', 'elapsed_total_s', 'best_epoch']


def _read_csv_rows(path, fallback_cols=None):
    """Read CSV rows as dicts. If no header row is detected and fallback_cols
    is provided, column names are assigned positionally."""
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        raw = f.read().strip()
    if not raw:
        return []
    first_field = raw.split('\n')[0].split(',')[0].strip()
    try:
        float(first_field)      # numeric first field → headerless CSV
        has_header = False
    except ValueError:
        has_header = True       # string first field → proper header present

    with open(path, newline='') as f:
        if has_header:
            return list(csv.DictReader(f))
        elif fallback_cols:
            return [dict(zip(fallback_cols, row)) for row in csv.reader(f)]
        else:
            return []


def read_best_csv(path, column, mode='max', fallback_cols=None):
    rows = _read_csv_rows(path, fallback_cols=fallback_cols)
    vals = [float(r[column]) for r in rows if column in r]
    if not vals:
        return None
    return max(vals) if mode == 'max' else min(vals)


def read_last_csv(path, column, fallback_cols=None):
    rows = _read_csv_rows(path, fallback_cols=fallback_cols)
    for row in reversed(rows):
        if column in row:
            return float(row[column])
    return None


# ── Inline evaluation (avoids subprocess stdout parsing) ─────────────────────

@torch.no_grad()
def evaluate_classifier(weights_path, data_dir, device, finetune_config='./configs/train_lorentz_part.yaml'):
    """Load a fine-tuned LorentzParT model and evaluate on the test set."""
    with open(finetune_config) as f:
        config = yaml.safe_load(f)

    model_cfg = LorentzParTConfig.from_dict(config['model'])
    # NOTE: the LorentzParT constructor ignores config.inference, so the model
    # returns logits. We softmax explicitly below before scoring — do NOT also set
    # inference=True or it would double-softmax.
    model = LorentzParT(config=model_cfg)
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    test_dataset = NpyJetClassDataset(
        os.path.join(data_dir, 'test', 'particles.npy'),
        os.path.join(data_dir, 'test', 'labels.npy'),
        normalize=NORMALIZE, norm_dict=NORM_DICT,
    )
    loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)

    all_pred, all_true = [], []
    for X, y in loader:
        all_pred.append(torch.softmax(model(X.to(device)), dim=1).cpu().numpy())  # probabilities, not logits
        all_true.append(y.numpy())

    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)

    acc = float((np.argmax(y_pred, 1) == np.argmax(y_true, 1)).mean())
    auc = float(roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo'))
    per_class = []
    per_class_auc = []
    for i in range(10):
        mask = np.argmax(y_true, 1) == i
        per_class.append(float((np.argmax(y_pred[mask], 1) == i).mean()) if mask.sum() else 0.0)
        per_class_auc.append(float(roc_auc_score(
            (np.argmax(y_true, 1) == i).astype(int), y_pred[:, i]
        )))

    return {'test_acc': acc, 'test_auc': auc, 'per_class_acc': per_class, 'per_class_auc': per_class_auc}


@torch.no_grad()
def get_embedding_stats(encoder_weights, data_dir, device, jepa_config='./configs/pretrain_jepa.yaml'):
    """Load a pretrained encoder and compute collapse diagnostics on the val set."""
    with open(jepa_config) as f:
        cfg = yaml.safe_load(f)['model']

    encoder = LorentzParTEncoder(
        embed_dim=cfg.get('embed_dim', 128),
        num_heads=cfg.get('num_heads', 8),
        num_layers=cfg.get('num_layers', 8),
        dropout=cfg.get('dropout', 0.1),
        expansion_factor=cfg.get('expansion_factor', 4),
        pair_embed_dims=cfg.get('pair_embed_dims', [64, 64, 64]),
    ).to(device)

    state_dict = torch.load(encoder_weights, map_location='cpu', weights_only=True)
    filtered = {k[len('encoder.'):]: v for k, v in state_dict.items() if k.startswith('encoder.')}
    encoder.load_state_dict(filtered, strict=False)

    processor = ParticleProcessor(to_multivector=True).to(device)
    val_dataset = NpyJetClassDataset(
        os.path.join(data_dir, 'val', 'particles.npy'),
        os.path.join(data_dir, 'val', 'labels.npy'),
        normalize=NORMALIZE, norm_dict=NORM_DICT,
    )
    loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)

    return probe_encoder_stats(encoder, processor, loader, device, max_jets=4096)


# ── Per-seed orchestration ────────────────────────────────────────────────────

def run_seed(seed, args, python, env):
    """Execute all Phase 0 stages for one seed. Returns a results dict."""
    set_seed(seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Deterministic checkpoint paths (follow run_comparison.py convention)
    jepa_ckpt = f'./logs/ParticleJEPA/best/jepa_seed{seed}.pt'
    mae_ckpt  = f'./logs/LorentzParT/best/mae_seed{seed}.pt'
    jepa_ft   = f'./logs/LorentzParT/best/jepa_ft_seed{seed}.pt'
    mae_ft    = f'./logs/LorentzParT/best/mae_ft_seed{seed}.pt'
    scratch   = f'./logs/LorentzParT/best/scratch_seed{seed}.pt'

    jepa_pt_csv = f'./logs/ParticleJEPA/logging/jepa_seed{seed}.csv'
    mae_pt_csv  = f'./logs/LorentzParT/logging/mae_seed{seed}.csv'
    jepa_ft_csv = f'./logs/LorentzParT/logging/jepa_ft_seed{seed}.csv'
    mae_ft_csv  = f'./logs/LorentzParT/logging/mae_ft_seed{seed}.csv'
    scratch_csv = f'./logs/LorentzParT/logging/scratch_seed{seed}.csv'

    probe_jepa_json = os.path.join(args.output_dir, f'probe_jepa_seed{seed}.json')
    probe_mae_json  = os.path.join(args.output_dir, f'probe_mae_seed{seed}.json')

    # ── Pretraining ───────────────────────────────────────────────────────────
    if not args.skip_pretrain:
        run_stage(
            [python, 'scripts/pretrain_jepa.py',
             '--data-dir', args.data_dir, '--config-path', args.pretrain_config,
             '--run-name', f'jepa_seed{seed}', '--seed', seed],
            env=env, desc=f'JEPA pretraining (seed={seed})',
        )
        run_stage(
            [python, 'scripts/pretrain_mae.py',
             '--data-dir', args.data_dir, '--config-path', args.mae_config,
             '--run-name', f'mae_seed{seed}', '--seed', seed],
            env=env, desc=f'MAE pretraining (seed={seed})',
        )

    # ── Fine-tuning ───────────────────────────────────────────────────────────
    if not args.skip_finetune:
        for run_name, weights in [
            (f'jepa_ft_seed{seed}', jepa_ckpt),
            (f'mae_ft_seed{seed}',  mae_ckpt),
            (f'scratch_seed{seed}', None),
        ]:
            cmd = [python, 'scripts/train_lorentz_part.py',
                   '--data-dir', args.data_dir, '--config-path', args.finetune_config,
                   '--run-name', run_name, '--seed', seed]
            if weights:
                cmd += ['--weights', weights]
            run_stage(cmd, env=env, desc=f'Fine-tune {run_name}')

    # ── Linear probing ────────────────────────────────────────────────────────
    if not args.skip_probe:
        for probe_name, ckpt in [
            (f'probe_jepa_seed{seed}', jepa_ckpt),
            (f'probe_mae_seed{seed}',  mae_ckpt),
        ]:
            run_stage(
                [python, 'experiments/phase0/linear_probe.py',
                 '--data-dir', args.data_dir, '--weights', ckpt,
                 '--config-path', args.probe_config,
                 '--run-name', probe_name, '--seed', seed,
                 '--output-dir', args.output_dir],
                env=env, desc=f'Linear probe {probe_name}',
            )

    # ── Embedding collapse diagnostics ────────────────────────────────────────
    jepa_embed_stats, mae_embed_stats = {}, {}
    if os.path.exists(jepa_ckpt):
        print(f"\nComputing embedding stats for JEPA (seed={seed})...")
        jepa_embed_stats = get_embedding_stats(jepa_ckpt, args.data_dir, device)
        print(f"  effective_rank={jepa_embed_stats['effective_rank']:.2f}  "
              f"mean_var={jepa_embed_stats['mean_var']:.4f}  "
              f"mean_cos_sim={jepa_embed_stats['mean_cos_sim']:.4f}")
    if os.path.exists(mae_ckpt):
        print(f"\nComputing embedding stats for MAE (seed={seed})...")
        mae_embed_stats = get_embedding_stats(mae_ckpt, args.data_dir, device)
        print(f"  effective_rank={mae_embed_stats['effective_rank']:.2f}  "
              f"mean_var={mae_embed_stats['mean_var']:.4f}  "
              f"mean_cos_sim={mae_embed_stats['mean_cos_sim']:.4f}")

    # ── Test-set evaluation ───────────────────────────────────────────────────
    conditions = {}
    for label, ft_path, ft_csv in [
        ('jepa_finetune', jepa_ft, jepa_ft_csv),
        ('mae_finetune',  mae_ft,  mae_ft_csv),
        ('scratch',       scratch, scratch_csv),
    ]:
        c = {
            'best_val_acc':    read_best_csv(ft_csv, 'val_metric', mode='max',
                                             fallback_cols=_FINETUNE_COLS),
            'pretrain_time_s': None,
        }
        if os.path.exists(ft_path):
            c.update(evaluate_classifier(ft_path, args.data_dir, device, args.finetune_config))

        print(f"\n{label}: test_acc={c.get('test_acc', 'N/A')}")
        for i, (name, acc) in enumerate(zip(CLASS_NAMES, c.get('per_class_acc', []))):
            print(f"  {name:<12}: {acc:.4f}")

        conditions[label] = c

    # Attach pretraining timing and embedding stats
    conditions['jepa_finetune']['pretrain_time_s'] = read_last_csv(
        jepa_pt_csv, 'elapsed_total_s', fallback_cols=_JEPA_PRETRAIN_COLS)
    conditions['mae_finetune']['pretrain_time_s']  = read_last_csv(
        mae_pt_csv,  'elapsed_total_s', fallback_cols=_MAE_PRETRAIN_COLS)
    conditions['jepa_finetune']['embedding_stats'] = jepa_embed_stats
    conditions['mae_finetune']['embedding_stats']  = mae_embed_stats

    # Attach linear probe results
    for label, json_path in [
        ('jepa_probe', probe_jepa_json),
        ('mae_probe',  probe_mae_json),
    ]:
        if os.path.exists(json_path):
            with open(json_path) as f:
                conditions[label] = json.load(f)
        else:
            conditions[label] = {}

    return {'seed': seed, 'conditions': conditions}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    python = sys.executable
    os.makedirs(args.output_dir, exist_ok=True)

    # Pin a single GPU for all subprocesses (avoids accidental DDP on multi-GPU nodes)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print(f"Phase 0 — {len(args.seeds)} seed(s): {args.seeds}")
    print(f"GPU: {args.gpu}  |  output: {args.output_dir}")

    for seed in args.seeds:
        print(f"\n{'#' * 60}")
        print(f"# SEED {seed}")
        print(f"{'#' * 60}")
        t0 = time.monotonic()

        results = run_seed(seed, args, python, env)

        elapsed = time.monotonic() - t0
        print(f"\nSeed {seed} complete in {elapsed / 60:.1f} min")

        out_path = os.path.join(args.output_dir, f'seed_{seed}.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved → {out_path}")

    print(f"\n{'=' * 60}")
    print(f"All seeds done. Run analyze_results.py to produce summary.")
    print(f"  python experiments/analyze_results.py --results-dir {args.output_dir}")


if __name__ == '__main__':
    main()
