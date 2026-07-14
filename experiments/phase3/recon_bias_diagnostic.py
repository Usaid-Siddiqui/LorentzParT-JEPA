"""
Momentum-balancing bias in MAE reconstruction — broken vs fixed backbone, biased vs random.

Reproduces Thanh Nguyen's GSoC-2025 diagnostic (src/utils/viz/viz.py::plot_particle_reconstruction):
a 2D histogram of TRUE (x) vs PREDICTED (y) feature values for the reconstructed masked particle,
with a blue y=x diagonal. When the transformer "balances" pT and eta to fake momentum conservation,
the eta density collapses into a horizontal band at predicted-eta ≈ 0 (off the diagonal) instead of
tracking the true value; pT gets inflated to compensate. Thanh's fix was biased masking.

Here we lay his exact cell (hist2d + diagonal, gist_heat_r) into a grid — rows = MAE conditions
(broken/-1e9 vs fixed/ragged × biased vs random), cols = features — so the collapse can be compared
across the interaction-embedding fix. σ_pred/σ_true in each title quantifies it (1.0 = the model
reproduces the true spread; ≪1 = collapsed to the mean). Note: ConservationLoss's extreme-eta reward
is disabled in this repo, so biased masking is the only thing opposing the collapse.

    python experiments/phase3/recon_bias_diagnostic.py --data-dir ./data --features pt eta \\
      --out experiments/phase3/recon_bias.png \\
      --checkpoints \\
        "broken biased:configs/pretrain_mae.yaml:logs/LorentzParT/best/stockmae_biased_seed42.pt:biased" \\
        "broken random:configs/pretrain_mae_random.yaml:logs/LorentzParT/best/stockmae_random_seed42.pt:random" \\
        "fixed biased:experiments/phase1/results_ragged/generated_configs/mae_biased.yaml:logs/LorentzParT/best/mae_biased_seed42.pt:biased" \\
        "fixed random:experiments/phase1/results_ragged/generated_configs/mae_random.yaml:logs/LorentzParT/best/mae_random_seed42.pt:random"

Each --checkpoints entry is  label:config:weights:mask_mode  (label/paths contain no ':').
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.configs import LorentzParTConfig
from src.models import LorentzParT
from src.utils.data import NpyJetClassDataset

NORM_DICT = {
    'pT':     (92.72917175292969,  105.83937072753906),
    'eta':    (0.0005733045982196927, 0.9174848794937134),
    'phi':    (-0.00041169871110469103, 1.8136887550354004),
    'energy': (133.8745574951172, 167.528564453125),
}
NORMALIZE = [True, False, False, True]   # pT, energy scaled; eta, phi raw (matches training)

# feature key -> (column index, axis label), matching Thanh's plot_particle_reconstruction
FEATURES = {'pt': (0, 'scaled pT'), 'eta': (1, 'eta'), 'phi': (2, 'phi'), 'energy': (3, 'scaled energy')}


def parse_args():
    p = argparse.ArgumentParser(description="MAE reconstruction momentum-balancing bias")
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--checkpoints', nargs='+', required=True,
                   metavar='label:config:weights:mask_mode')
    p.add_argument('--features', nargs='+', default=['pt', 'eta'], choices=list(FEATURES),
                   help="which features to plot as columns (default: pt eta — the balancing pair)")
    p.add_argument('--max-jets', type=int, default=50000, help="cap for speed")
    p.add_argument('--out', default='experiments/phase3/recon_bias.png')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


@torch.no_grad()
def collect(cfg_path, weights, data_dir, mask_mode, device, max_jets):
    """Return (pred (N,4), true (N,4)) for the masked particle over the test set."""
    cfg = LorentzParTConfig.from_dict(yaml.safe_load(open(cfg_path))['model'])
    model = LorentzParT(config=cfg)
    sd = torch.load(weights, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    ds = NpyJetClassDataset(
        os.path.join(data_dir, 'test', 'particles.npy'),
        os.path.join(data_dir, 'test', 'labels.npy'),
        normalize=NORMALIZE, norm_dict=NORM_DICT, mask_mode=mask_mode, num_mask=1,
    )
    loader = DataLoader(ds, batch_size=1000, shuffle=False, num_workers=4)

    preds, trues, n = [], [], 0
    for X, y, mask_idx in loader:                          # y = masked_targets (B, 4)
        out = model(X.to(device), mask_idx.to(device).long())
        if out.ndim == 3:                                  # (B, K, 4) -> (B, 4)
            out = out[:, 0, :]
        y = y if y.ndim == 2 else y[:, 0, :]
        preds.append(out.cpu().numpy()); trues.append(y.numpy())
        n += len(X)
        if n >= max_jets:
            break
    return np.concatenate(preds), np.concatenate(trues), (len(missing), len(unexpected))


def hist2d_cell(ax, true, pred, label):
    """Identical to Thanh's cell (viz.py::plot_particle_reconstruction): hist2d(true, pred)
    over the raw data range, gist_heat_r, solid blue y=x diagonal, per-cell colorbar.
    Returns (σ_pred/σ_true, mean(pred)-mean(true), mesh) — the numbers go to the console, not
    the plot, so the figure stays pixel-faithful to his."""
    lo = min(float(true.min()), float(pred.min()))
    hi = max(float(true.max()), float(pred.max()))
    _, _, _, mesh = ax.hist2d(true, pred, bins=50, cmap='gist_heat_r')
    ax.plot([lo, hi], [lo, hi], color='blue', linestyle='-')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f'true {label}'); ax.set_ylabel(f'predicted {label}')
    ax.set_title(f'{label} distribution')
    sr = np.std(pred) / (np.std(true) + 1e-9)                  # spread ratio (1 = matches true spread)
    corr = float(np.corrcoef(true, pred)[0, 1])                # does pred track true? (1 = on diagonal)
    return sr, float(pred.mean() - true.mean()), corr, mesh


def main():
    args = parse_args()
    specs = [tuple(s.split(':')) for s in args.checkpoints]     # (label, cfg, weights, mode)
    device = torch.device(args.device)

    data = {}
    for label, cfg, w, mode in specs:
        print(f"[collect] {label:16s} mode={mode}")
        pred, true, load = collect(cfg, w, args.data_dir, mode, device, args.max_jets)
        data[label] = (pred, true)
        print(f"          load missing/unexpected = {load}  (n={len(pred)})")

    feats = [(FEATURES[f][0], FEATURES[f][1]) for f in args.features]
    nrow, ncol = len(specs), len(feats)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.9 * nrow), squeeze=False)

    print(f"\n{'condition':16s} " + "  ".join(f'{lbl:>16s}' for _, lbl in feats))
    for i, (label, *_ ) in enumerate(specs):
        pred, true = data[label]
        cells = []
        for j, (idx, flabel) in enumerate(feats):
            sr, dmu, corr, mesh = hist2d_cell(axes[i][j], true[:, idx], pred[:, idx], flabel)
            fig.colorbar(mesh, ax=axes[i][j])
            cells.append((flabel, sr, dmu, corr))
        # condition label on the row margin — leaves the cell (his) labels/title untouched
        axes[i][0].text(-0.42, 0.5, label, transform=axes[i][0].transAxes, rotation=90,
                        va='center', ha='center', fontweight='bold', fontsize=11)
        print(f"{label:16s} " + "  ".join(f'{fl}:σr={s:.3f} r={c:+.3f} Δμ={d:+.3f}' for fl, s, d, c in cells))

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
