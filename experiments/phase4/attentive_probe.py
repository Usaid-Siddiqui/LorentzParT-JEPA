"""
Attentive probe evaluation for SSL representations.

Trains only a class-attention head on top of a FROZEN pretrained encoder — the
I-JEPA frozen-evaluation protocol. Strictly more expressive than the linear
probe (mean-pool + linear), which is known to undersell JEPA features, so this
is the fair frozen-encoder ceiling between the linear probe and full fine-tuning.

Mirrors experiments/phase0/linear_probe.py (same harness, metrics, JSON output);
only the model (AttentiveProbeModel) and the trained params (the class-attention
head) differ.

    python experiments/phase4/attentive_probe.py \\
        --data-dir ./data_1m \\
        --weights ./logs/ParticleJEPA/best/jepa_1m_ragged_seed42_best.pt \\
        --run-name attn_jepa_1m_seed42 \\
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
from sklearn.metrics import roc_auc_score

from src.models.attentive_probe import AttentiveProbeModel
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
    p = argparse.ArgumentParser(description="Attentive probe on pretrained LorentzParT encoder")
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--weights', default=None,
                   help='Pretrained encoder checkpoint (.pt). Omit for the random-feature (scratch) control.')
    p.add_argument('--config-path', default='experiments/phase4/configs/attentive_probe.yaml')
    p.add_argument('--run-name', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', default='./experiments/phase4/results')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for X, y in loader:
        logits = model(X.to(device))
        all_pred.append(torch.softmax(logits, dim=1).cpu().numpy())  # score probabilities, not logits
        all_true.append(y.cpu().numpy())
    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    acc = float((np.argmax(y_pred, 1) == np.argmax(y_true, 1)).mean())
    auc = float(roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo'))
    per_class = []
    per_class_auc = []
    for i in range(10):
        mask = np.argmax(y_true, 1) == i
        per_class.append(
            float((np.argmax(y_pred[mask], 1) == i).mean()) if mask.sum() > 0 else 0.0
        )
        per_class_auc.append(float(roc_auc_score(
            (np.argmax(y_true, 1) == i).astype(int), y_pred[:, i]
        )))
    return acc, auc, per_class, per_class_auc


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)

    encoder_kwargs = {
        'num_layers':       cfg.get('num_layers',       8),
        'dropout':          cfg.get('dropout',          0.1),
        'expansion_factor': cfg.get('expansion_factor', 4),
        'pair_embed_dims':  cfg.get('pair_embed_dims',  [64, 64, 64]),
        'ragged_pair_embed': cfg.get('ragged_pair_embed', False),
    }
    model = AttentiveProbeModel(
        encoder_weights=args.weights,
        embed_dim=cfg.get('embed_dim', 128),
        num_classes=10,
        num_heads=cfg.get('num_heads', 8),
        num_cls_layers=cfg.get('num_cls_layers', 2),
        hidden_dim=cfg.get('hidden_dim', 256),
        num_mlp_layers=cfg.get('num_mlp_layers', 0),
        expansion_factor=cfg.get('expansion_factor', 4),
        dropout=cfg.get('dropout', 0.1),
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
        model.head_parameters(),
        lr=opt_cfg.get('lr', 1e-3),
        weight_decay=opt_cfg.get('weight_decay', 0.01),
    )
    sched_cfg = cfg.get('scheduler', {})
    epochs = cfg.get('epochs', 20)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg.get('T_max', epochs),
        eta_min=sched_cfg.get('eta_min', 1e-5),
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_val_auc = 0.0
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
        val_acc, val_auc, _, _ = evaluate(model, val_loader, device)
        best_val_acc = max(best_val_acc, val_acc)
        best_val_auc = max(best_val_auc, val_auc)
        print(f"epoch {epoch + 1:2d}/{epochs} | val_acc: {val_acc:.4f}  val_auc: {val_auc:.4f}")

    train_time_s = time.monotonic() - t_start
    test_acc, test_auc, per_class_acc, per_class_auc = evaluate(model, test_loader, device)

    print(f"\nAttentive probe results ({args.run_name})")
    print(f"  test_acc:     {test_acc:.4f}")
    print(f"  test_auc:     {test_auc:.4f}")
    print(f"  best_val_acc: {best_val_acc:.4f}")
    print(f"  best_val_auc: {best_val_auc:.4f}")
    print(f"  train_time_s: {train_time_s:.1f}")
    print("\nPer-class accuracy / AUC:")
    for name, acc, auc in zip(CLASS_NAMES, per_class_acc, per_class_auc):
        print(f"  {name:<12}: acc={acc:.4f}  auc={auc:.4f}")

    run_name = args.run_name or f'attn_probe_seed{args.seed}'
    results = {
        'run_name':       run_name,
        'seed':           args.seed,
        'weights':        args.weights,
        'test_acc':       test_acc,
        'test_auc':       test_auc,
        'best_val_acc':   best_val_acc,
        'best_val_auc':   best_val_auc,
        'per_class_acc':  per_class_acc,
        'per_class_auc':  per_class_auc,
        'train_time_s':   train_time_s,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{run_name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
