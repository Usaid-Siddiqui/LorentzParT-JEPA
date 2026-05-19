"""
Linear probe evaluation for SSL representations.

Trains only a linear head on top of a frozen pretrained encoder (mean-pooled
embeddings). This is the standard SSL linear evaluation protocol — a clean
measure of representation quality that is not confounded by fine-tuning.

Must be run from the LorentzParT_JEPA/ root directory:

    python experiments/phase0/linear_probe.py \\
        --data-dir ./data \\
        --weights ./logs/ParticleJEPA/best/jepa_seed42.pt \\
        --run-name jepa_probe_seed42 \\
        --seed 42

Results are written to --output-dir as <run-name>.json.
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.linear_probe import LinearProbeModel
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
    p = argparse.ArgumentParser(description="Linear probe on pretrained LorentzParT encoder")
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--weights', required=True, help='Pretrained encoder checkpoint (.pt)')
    p.add_argument('--config-path', default='experiments/phase0/configs/linear_probe.yaml')
    p.add_argument('--run-name', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', default='./results/phase0')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for X, y in loader:
        logits = model(X.to(device))
        all_pred.append(logits.cpu().numpy())
        all_true.append(y.cpu().numpy())
    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    acc = float((np.argmax(y_pred, 1) == np.argmax(y_true, 1)).mean())
    per_class = []
    for i in range(10):
        mask = np.argmax(y_true, 1) == i
        per_class.append(
            float((np.argmax(y_pred[mask], 1) == i).mean()) if mask.sum() > 0 else 0.0
        )
    return acc, per_class


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)

    encoder_kwargs = {
        'num_heads':        cfg.get('num_heads',        8),
        'num_layers':       cfg.get('num_layers',       8),
        'dropout':          cfg.get('dropout',          0.1),
        'expansion_factor': cfg.get('expansion_factor', 4),
        'pair_embed_dims':  cfg.get('pair_embed_dims',  [64, 64, 64]),
    }
    model = LinearProbeModel(
        encoder_weights=args.weights,
        embed_dim=cfg.get('embed_dim', 128),
        num_classes=10,
        encoder_kwargs=encoder_kwargs,
    ).to(device)

    loader_kw = dict(
        batch_size=cfg.get('batch_size', 1000),
        num_workers=cfg.get('num_workers', 4),
        pin_memory=cfg.get('pin_memory', True),
    )
    train_loader = DataLoader(
        NpyJetClassDataset(
            os.path.join(args.data_dir, 'train', 'particles.npy'),
            os.path.join(args.data_dir, 'train', 'labels.npy'),
            normalize=NORMALIZE, norm_dict=NORM_DICT,
        ),
        shuffle=True, **loader_kw,
    )
    val_loader = DataLoader(
        NpyJetClassDataset(
            os.path.join(args.data_dir, 'val', 'particles.npy'),
            os.path.join(args.data_dir, 'val', 'labels.npy'),
            normalize=NORMALIZE, norm_dict=NORM_DICT,
        ),
        shuffle=False, **loader_kw,
    )
    test_loader = DataLoader(
        NpyJetClassDataset(
            os.path.join(args.data_dir, 'test', 'particles.npy'),
            os.path.join(args.data_dir, 'test', 'labels.npy'),
            normalize=NORMALIZE, norm_dict=NORM_DICT,
        ),
        shuffle=False, **loader_kw,
    )

    opt_cfg = cfg.get('optimizer', {})
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=opt_cfg.get('lr', 1e-3),
        weight_decay=opt_cfg.get('weight_decay', 0.01),
    )
    sched_cfg = cfg.get('scheduler', {})
    epochs = cfg.get('epochs', 10)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg.get('T_max', epochs),
        eta_min=sched_cfg.get('eta_min', 1e-5),
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    t_start = time.monotonic()

    for epoch in range(epochs):
        model.train()
        model.encoder.eval()  # keep BatchNorm running stats frozen

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y.argmax(dim=1))
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_acc, _ = evaluate(model, val_loader, device)
        best_val_acc = max(best_val_acc, val_acc)
        print(f"epoch {epoch + 1:2d}/{epochs} | val_acc: {val_acc:.4f}")

    train_time_s = time.monotonic() - t_start
    test_acc, per_class_acc = evaluate(model, test_loader, device)

    print(f"\nLinear probe results ({args.run_name})")
    print(f"  test_acc:     {test_acc:.4f}")
    print(f"  best_val_acc: {best_val_acc:.4f}")
    print(f"  train_time_s: {train_time_s:.1f}")
    print("\nPer-class accuracy:")
    for name, acc in zip(CLASS_NAMES, per_class_acc):
        print(f"  {name:<12}: {acc:.4f}")

    run_name = args.run_name or f'probe_seed{args.seed}'
    results = {
        'run_name':      run_name,
        'seed':          args.seed,
        'weights':       args.weights,
        'test_acc':      test_acc,
        'best_val_acc':  best_val_acc,
        'per_class_acc': per_class_acc,
        'train_time_s':  train_time_s,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{run_name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
