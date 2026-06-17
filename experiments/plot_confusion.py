"""
Confusion-matrix grid for arbitrary trained classifiers.

Confusion matrices aren't stored in the result JSONs, so this re-runs test-set
inference on each checkpoint. Conditions are given as `label:glob` specs — the glob
expands to one or more checkpoints (e.g. seeds), whose predictions are pooled. Lays
the conditions out in an auto grid of row-normalized 10×10 matrices (diagonal =
per-class recall). Must run where the checkpoints + test data live (GPU).

    # one condition, 3 seeds pooled
    python experiments/plot_confusion.py \\
        --checkpoints "JEPA→FT:logs/LorentzParT/best/jepa_ft_seed*.pt" \\
        --data-dir ./data --output experiments/confusion_jepa.png

    # objective × masking 2×2 (any number of conditions; --ncols controls layout)
    python experiments/plot_confusion.py --ncols 2 --data-dir ./data \\
        --checkpoints \\
            "JEPA·random:logs/LorentzParT/best/jepa_ft_seed*.pt" \\
            "JEPA·biased:logs/LorentzParT/best/no_gate_biased_ft_seed*.pt" \\
            "MAE·random:logs/LorentzParT/best/mae_ft_seed*.pt" \\
            "MAE·biased:logs/LorentzParT/best/mae_biased_ft_seed*.pt"

--config-path gives the model architecture; --model picks the build (lorentz_part
or linear_probe), matching scripts/evaluate.py.
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.configs import LorentzParTConfig
from src.models import LorentzParT, LinearProbeModel
from src.utils.data import NpyJetClassDataset

NORM_DICT = {
    'pT':     (92.72917175292969,    105.83937072753906),
    'eta':    (0.0005733045982196927, 0.9174848794937134),
    'phi':    (-0.00041169871110469103, 1.8136887550354004),
    'energy': (133.8745574951172,    167.528564453125),
}
NORMALIZE = [True, False, False, True]
# LaTeX class labels (matches src/utils/viz/viz.py & Nguyen's formatting)
CLASS_LABELS = [
    "$q/g$", "$H\\to b\\bar{b}$", "$H\\to c\\bar{c}$", "$H\\to gg$", "$H\\to 4q$",
    "$H\\to \\ell\\nu qq'$", "$Z\\to q\\bar{q}$", "$W\\to qq'$", "$t\\to bqq'$", "$t\\to b\\ell\\nu$",
]


def parse_args():
    p = argparse.ArgumentParser(description="Confusion-matrix grid for any classifiers")
    p.add_argument('--checkpoints', nargs='+', required=True, metavar='LABEL:GLOB',
                   help='label:checkpoint-glob specs; glob matches seed checkpoints to pool')
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--config-path', default='./configs/train_lorentz_part.yaml')
    p.add_argument('--model', default='lorentz_part', choices=['lorentz_part', 'linear_probe'])
    p.add_argument('--ncols', type=int, default=2)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', default='./experiments/confusion.png')
    return p.parse_args()


def parse_specs(specs):
    out = []
    for s in specs:
        label, pattern = s.split(':', 1)
        out.append((label, pattern))
    return out


def build_model(model_kind, cfg, ckpt, device):
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    if model_kind == 'linear_probe':
        model = LinearProbeModel(
            encoder_weights=ckpt, embed_dim=cfg.embed_dim, num_classes=cfg.num_classes,
            encoder_kwargs=dict(num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                                dropout=cfg.dropout, expansion_factor=cfg.expansion_factor,
                                pair_embed_dims=cfg.pair_embed_dims),
        )
    else:
        cfg.inference = True
        model = LorentzParT(config=cfg)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, loader, device):
    pred, true = [], []
    for X, y in loader:
        pred.append(model(X.to(device)).argmax(1).cpu().numpy())
        true.append(y.argmax(1).numpy())
    return np.concatenate(true), np.concatenate(pred)


def pooled_confusion(pattern, model_kind, cfg, loader, device):
    """Sum confusion counts over all checkpoints matching the glob, then row-normalize."""
    files = sorted(glob.glob(pattern))
    if not files:
        return None, 0
    total = np.zeros((10, 10), dtype=np.int64)
    for ckpt in files:
        model = build_model(model_kind, cfg, ckpt, device)
        yt, yp = predict(model, loader, device)
        total += confusion_matrix(yt, yp, labels=list(range(10)))
    return total / total.sum(axis=1, keepdims=True).clip(min=1), len(files)


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    cfg = LorentzParTConfig.from_dict(yaml.safe_load(open(args.config_path))['model'])

    ds = NpyJetClassDataset(
        os.path.join(args.data_dir, 'test', 'particles.npy'),
        os.path.join(args.data_dir, 'test', 'labels.npy'),
        normalize=NORMALIZE, norm_dict=NORM_DICT,
    )
    loader = DataLoader(ds, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)

    specs = parse_specs(args.checkpoints)
    n = len(specs)
    ncols = min(args.ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 6.5), squeeze=False)

    for idx, (label, pattern) in enumerate(specs):
        ax = axes[idx // ncols][idx % ncols]
        cm, k = pooled_confusion(pattern, args.model, cfg, loader, device)
        if cm is None:
            ax.set_title(f'{label}  (no checkpoints matched)')
            ax.axis('off')
            print(f"  [skip] no files match {pattern}")
            continue
        sns.heatmap(cm, ax=ax, annot=True, fmt='.2f', annot_kws={'size': 5},
                    cmap='coolwarm', vmin=0, vmax=1, cbar=False, square=True,
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
        ax.set_title(f'{label}   (mean recall {np.diag(cm).mean():.3f}, n={k})', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('Actual', fontsize=9)
        ax.tick_params(axis='x', labelrotation=45, labelsize=7)
        ax.tick_params(axis='y', labelrotation=0, labelsize=7)
        print(f"  {label}: mean recall {np.diag(cm).mean():.3f} (n={k})")

    # hide any unused grid cells
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    fig.suptitle('Confusion matrices (row-normalized = per-class recall)', fontsize=13)
    fig.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()
    print(f"Saved → {args.output}")


if __name__ == '__main__':
    main()
