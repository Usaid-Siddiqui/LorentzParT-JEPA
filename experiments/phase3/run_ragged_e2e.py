"""
Phase 3 — end-to-end validation of the ragged interaction embedding (100k).

The decisive gate for whether ragged ships. For each seed x {stock, ragged}, runs the
FULL pipeline — JEPA pretrain -> LorentzParT finetune -> eval — with ragged_pair_embed
OFF vs ON throughout, then reports:
  (1) does ragged preserve accuracy? (softmax OVO AUC delta vs seed variance), and
  (2) the pretrain AND finetune per-epoch speedups (the compute win, where pretraining
      is the bigger prize since the encoder runs the pair embedding every step).

Unlike a finetune-only A/B, this validates ragged in PRETRAINING too — the place the
semantic change (BN over valid pairs only) could break the JEPA pretext task. Run on
the 100k subset (./data): ~10x cheaper than 1M and accuracy-preservation is scale-robust.

Must be run from the LorentzParT_JEPA/ root.

    python experiments/phase3/run_ragged_e2e.py --data-dir ./data --seeds 42 123 456 --gpu 0

Re-run to resume (finished checkpoints are skipped).
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.configs import LorentzParTConfig
from src.models import LorentzParT
from src.utils.data import NpyJetClassDataset

NORM_DICT = {
    'pT':     (92.72917175292969,    105.83937072753906),
    'eta':    (0.0005733045982196927, 0.9174848794937134),
    'phi':    (-0.00041169871110469103, 1.8136887550354004),
    'energy': (133.8745574951172,    167.528564453125),
}
NORMALIZE = [True, False, False, True]

# condition -> (jepa pretrain config, finetune config). Same weights load into both.
CONDITIONS = {
    'stock':  ('./configs/pretrain_jepa.yaml',        './configs/train_lorentz_part.yaml'),
    'ragged': ('./configs/pretrain_jepa_ragged.yaml', './configs/train_lorentz_part_ragged.yaml'),
}


@torch.no_grad()
def evaluate(ckpt, data_dir, device, finetune_config):
    """Softmax OVO AUC + acc. Builds the model from `finetune_config` so the ragged
    flag matches the checkpoint's forward."""
    with open(finetune_config) as f:
        model_cfg = LorentzParTConfig.from_dict(yaml.safe_load(f)['model'])
    model = LorentzParT(config=model_cfg)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True), strict=False)
    model = model.to(device).eval()
    ds = NpyJetClassDataset(
        os.path.join(data_dir, 'test', 'particles.npy'),
        os.path.join(data_dir, 'test', 'labels.npy'),
        normalize=NORMALIZE, norm_dict=NORM_DICT,
    )
    loader = DataLoader(ds, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)
    pred, true = [], []
    for X, y in loader:
        pred.append(torch.softmax(model(X.to(device)), dim=1).cpu().numpy())
        true.append(y.numpy())
    y_pred, y_true = np.concatenate(pred), np.concatenate(true)
    acc = float((np.argmax(y_pred, 1) == np.argmax(y_true, 1)).mean())
    auc = float(roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo'))
    return acc, auc


def median_epoch_s(csv_path):
    """Median per-epoch seconds from elapsed_total_s (col 6) — works for both the JEPA
    pretrain CSV and the finetune CSV."""
    if not os.path.exists(csv_path):
        return None
    el = []
    with open(csv_path) as f:
        for r in csv.reader(f):
            try:
                el.append(float(r[6]))
            except (ValueError, IndexError):
                continue
    if len(el) < 1:
        return None
    per_epoch = [el[0]] + [el[i] - el[i - 1] for i in range(1, len(el))]
    return statistics.median(per_epoch)


def run_stage(cmd, env, desc):
    print(f"\n{'=' * 60}\nSTAGE: {desc}\nCMD:   {' '.join(str(c) for c in cmd)}\n{'=' * 60}")
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def main():
    p = argparse.ArgumentParser(description="End-to-end ragged vs stock validation (100k)")
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()

    python = sys.executable
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    rows = []   # (cond, seed, acc, auc, pre_ep_s, ft_ep_s)
    for seed in args.seeds:
        for cond, (pre_cfg, ft_cfg) in CONDITIONS.items():
            pre_run, ft_run = f'jrag_{cond}_seed{seed}', f'ftrag_{cond}_seed{seed}'
            pre_ckpt = f'./logs/ParticleJEPA/best/{pre_run}.pt'
            ft_ckpt = f'./logs/LorentzParT/best/{ft_run}.pt'

            if not os.path.exists(pre_ckpt):
                run_stage([python, 'scripts/pretrain_jepa.py', '--data-dir', args.data_dir,
                           '--config-path', pre_cfg, '--run-name', pre_run, '--seed', seed],
                          env, f'Pretrain {cond} seed={seed}')
            else:
                print(f"[skip pretrain] {pre_ckpt} exists")

            if not os.path.exists(ft_ckpt):
                run_stage([python, 'scripts/train_lorentz_part.py', '--data-dir', args.data_dir,
                           '--config-path', ft_cfg, '--weights', pre_ckpt,
                           '--run-name', ft_run, '--seed', seed],
                          env, f'Finetune {cond} seed={seed}')
            else:
                print(f"[skip finetune] {ft_ckpt} exists")

            acc, auc = evaluate(ft_ckpt, args.data_dir, device, ft_cfg)
            pre_ep = median_epoch_s(f'./logs/ParticleJEPA/logging/{pre_run}.csv')
            ft_ep = median_epoch_s(f'./logs/LorentzParT/logging/{ft_run}.csv')
            rows.append((cond, seed, acc, auc, pre_ep, ft_ep))
            print(f"  {cond:6s} seed{seed}: acc={acc:.4f} auc={auc:.4f}")

    # ---- report ----
    print(f"\n{'cond':7s}{'seed':>6s}{'acc':>9s}{'auc':>9s}{'pre_ep_s':>10s}{'ft_ep_s':>10s}")
    for cond, seed, acc, auc, pre_ep, ft_ep in rows:
        pe = f'{pre_ep:.1f}' if pre_ep else '--'
        fe = f'{ft_ep:.1f}' if ft_ep else '--'
        print(f"{cond:7s}{seed:>6d}{acc:>9.4f}{auc:>9.4f}{pe:>10s}{fe:>10s}")

    def agg(cond, i):
        v = [r[i] for r in rows if r[0] == cond and r[i] is not None]
        return (statistics.mean(v), statistics.pstdev(v)) if v else (float('nan'), 0.0)

    (sa, ss), (ra, rs) = agg('stock', 3), agg('ragged', 3)
    spe, rpe = agg('stock', 4)[0], agg('ragged', 4)[0]      # pretrain median epoch
    sfe, rfe = agg('stock', 5)[0], agg('ragged', 5)[0]      # finetune median epoch
    d = ra - sa
    within = abs(d) <= (ss + rs)
    print(f"\nstock  AUC {sa:.4f} ± {ss:.4f}")
    print(f"ragged AUC {ra:.4f} ± {rs:.4f}   Δ {d:+.4f}  -> {'WITHIN' if within else 'OUTSIDE'} seed variance")
    if spe and rpe:
        print(f"pretrain median epoch: stock {spe:.1f}s -> ragged {rpe:.1f}s   ({spe / rpe:.2f}x)")
    if sfe and rfe:
        print(f"finetune median epoch: stock {sfe:.1f}s -> ragged {rfe:.1f}s   ({sfe / rfe:.2f}x)")
    print(f"\nVERDICT: ragged {'PRESERVES' if within else 'CHANGES'} AUC"
          + (f"; pretrain {spe / rpe:.2f}x, finetune {sfe / rfe:.2f}x faster/epoch"
             if spe and rpe and sfe and rfe else ""))


if __name__ == '__main__':
    main()
