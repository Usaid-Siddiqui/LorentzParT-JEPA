"""
Phase 3 — validate the padding-aware (ragged) interaction embedding.

Finetunes LorentzParT with ragged_pair_embed OFF (stock) vs ON (ragged) from the
SAME pretrained encoder, across seeds, evaluates softmax OVO AUC, and reports:
  (1) does ragged preserve accuracy? (AUC delta vs seed variance), and
  (2) the end-to-end finetune speedup (median epoch time, stock -> ragged).

Same pretrained weights + same seeds -> the ONLY difference is the pair-embedding
algorithm, so any AUC gap is the semantic change (BN over valid pairs only, which
also removes the stock module's -1e9-padding BN-stat corruption).

Must be run from the LorentzParT_JEPA/ root.

    python experiments/phase3/run_ragged_validation.py \\
        --weights ./logs/ParticleJEPA/best/<pretrain>.pt \\
        --data-dir ./data_1m --seeds 42 123 456 --gpu 0

Omit --weights to compare from scratch (architecture-level check). Re-run to resume
(finished checkpoints are skipped).
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
from src.utils import set_seed
from src.utils.data import NpyJetClassDataset

NORM_DICT = {
    'pT':     (92.72917175292969,    105.83937072753906),
    'eta':    (0.0005733045982196927, 0.9174848794937134),
    'phi':    (-0.00041169871110469103, 1.8136887550354004),
    'energy': (133.8745574951172,    167.528564453125),
}
NORMALIZE = [True, False, False, True]

# (condition name, finetune config) — same weights load into both.
CONDITIONS = [
    ('stock',  './configs/train_lorentz_part.yaml'),
    ('ragged', './configs/train_lorentz_part_ragged.yaml'),
]


@torch.no_grad()
def evaluate(weights_path, data_dir, device, finetune_config):
    """Softmax OVO AUC + accuracy. Builds the model from `finetune_config` so the
    ragged flag matches the checkpoint's forward (critical: a ragged checkpoint must
    be evaluated with the ragged forward)."""
    with open(finetune_config) as f:
        model_cfg = LorentzParTConfig.from_dict(yaml.safe_load(f)['model'])
    model = LorentzParT(config=model_cfg)
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=False)
    model = model.to(device).eval()

    ds = NpyJetClassDataset(
        os.path.join(data_dir, 'test', 'particles.npy'),
        os.path.join(data_dir, 'test', 'labels.npy'),
        normalize=NORMALIZE, norm_dict=NORM_DICT,
    )
    loader = DataLoader(ds, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)
    pred, true = [], []
    for X, y in loader:
        pred.append(torch.softmax(model(X.to(device)), dim=1).cpu().numpy())  # probabilities
        true.append(y.numpy())
    y_pred, y_true = np.concatenate(pred), np.concatenate(true)
    acc = float((np.argmax(y_pred, 1) == np.argmax(y_true, 1)).mean())
    auc = float(roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo'))
    return acc, auc


def ft_times(csv_path):
    """(total finetune seconds, median epoch seconds) from a finetune CSV."""
    if not os.path.exists(csv_path):
        return None, None
    el = []
    with open(csv_path) as f:
        for r in csv.reader(f):
            try:
                el.append(float(r[6]))            # elapsed_total_s
            except (ValueError, IndexError):
                continue
    if not el:
        return None, None
    per_epoch = [el[0]] + [el[i] - el[i - 1] for i in range(1, len(el))]
    return el[-1], statistics.median(per_epoch)


def run_stage(cmd, env, desc):
    print(f"\n{'=' * 60}\nSTAGE: {desc}\nCMD:   {' '.join(str(c) for c in cmd)}\n{'=' * 60}")
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def main():
    p = argparse.ArgumentParser(description="Ragged vs stock finetune A/B")
    p.add_argument('--weights', default=None, help="pretrained encoder; omit for from-scratch")
    p.add_argument('--data-dir', default='./data_1m')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()

    python = sys.executable
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    rows = []   # (cond, seed, acc, auc, total_s, med_epoch_s)
    for seed in args.seeds:
        set_seed(seed)
        for cond, cfg in CONDITIONS:
            run = f'ragval_{cond}_seed{seed}'
            ckpt = f'./logs/LorentzParT/best/{run}.pt'
            if not os.path.exists(ckpt):
                cmd = [python, 'scripts/train_lorentz_part.py',
                       '--data-dir', args.data_dir, '--config-path', cfg,
                       '--run-name', run, '--seed', seed]
                if args.weights:
                    cmd += ['--weights', args.weights]
                run_stage(cmd, env, f'Finetune {cond} seed={seed}')
            else:
                print(f"[skip] {ckpt} exists")
            acc, auc = evaluate(ckpt, args.data_dir, device, cfg)
            total_s, med_s = ft_times(f'./logs/LorentzParT/logging/{run}.csv')
            rows.append((cond, seed, acc, auc, total_s, med_s))
            print(f"  {cond:6s} seed{seed}: acc={acc:.4f} auc={auc:.4f} "
                  f"med_epoch={med_s:.1f}s" if med_s else f"  {cond} seed{seed}: acc={acc:.4f} auc={auc:.4f}")

    # ---- report ----
    print(f"\n{'cond':7s}{'seed':>6s}{'acc':>9s}{'auc':>9s}{'ft_total_h':>12s}{'med_ep_s':>10s}")
    for cond, seed, acc, auc, total_s, med_s in rows:
        th = f'{total_s / 3600:.2f}' if total_s else '--'
        ms = f'{med_s:.1f}' if med_s else '--'
        print(f"{cond:7s}{seed:>6d}{acc:>9.4f}{auc:>9.4f}{th:>12s}{ms:>10s}")

    def agg(cond, i):
        vals = [r[i] for r in rows if r[0] == cond and r[i] is not None]
        return (statistics.mean(vals), statistics.pstdev(vals)) if vals else (float('nan'), float('nan'))

    (sa, ss), (ra, rs) = agg('stock', 3), agg('ragged', 3)     # auc
    (_, _),   (rm, _)  = agg('stock', 5), agg('ragged', 5)     # med epoch
    sm = agg('stock', 5)[0]
    d = ra - sa
    within = abs(d) <= (ss + rs)                                # delta within summed 1-sigma
    print(f"\nstock  AUC {sa:.4f} ± {ss:.4f}")
    print(f"ragged AUC {ra:.4f} ± {rs:.4f}   Δ {d:+.4f}  -> {'WITHIN' if within else 'OUTSIDE'} seed variance")
    if sm and rm:
        print(f"median epoch: stock {sm:.1f}s -> ragged {rm:.1f}s   ({sm / rm:.2f}x faster)")
    print(f"\nVERDICT: ragged {'PRESERVES' if within else 'CHANGES'} AUC"
          + (f" and is {sm / rm:.2f}x faster/epoch" if sm and rm else ""))


if __name__ == '__main__':
    main()
