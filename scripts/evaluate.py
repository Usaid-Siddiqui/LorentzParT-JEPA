"""
Evaluation script — measures classification performance on the test set.

Produces:
  - Accuracy and per-class accuracy
  - Macro-averaged OVO ROC AUC (primary metric)
  - ROC curve plot
  - Confusion matrix
  - JSON summary (if --json-out is specified)

Usage:
    # Full LorentzParT model (finetune / scratch checkpoints)
    python scripts/evaluate.py \\
        --data-dir ./data \\
        --weights ./logs/LorentzParT/best/jepa_finetune_seed42.pt \\
        --run-name jepa_finetune_seed42 \\
        --json-out ./results/jepa_finetune_seed42.json

    # Linear probe checkpoint
    python scripts/evaluate.py \\
        --data-dir ./data \\
        --weights ./logs/LinearProbe/best/jepa_probe_seed42.pt \\
        --model linear_probe \\
        --run-name jepa_probe_seed42 \\
        --json-out ./results/jepa_probe_seed42.json
"""

import json
import os
import sys
import yaml
import argparse
import warnings

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs import LorentzParTConfig, TrainConfig
from src.engine import JetClassTrainer
from src.models import LorentzParT, LinearProbeModel
from src.utils import accuracy_metric_ce, set_seed
from src.utils.data import NpyJetClassDataset
from src.utils.viz import plot_roc_curve, plot_confusion_matrix

warnings.filterwarnings('ignore')


NORM_DICT = {
    'pT':     (92.72917175292969,  105.83937072753906),
    'eta':    (0.0005733045982196927, 0.9174848794937134),
    'phi':    (-0.00041169871110469103, 1.8136887550354004),
    'energy': (133.8745574951172, 167.528564453125),
}
NORMALIZE = [True, False, False, True]

CLASS_NAMES = [
    'QCD/Z→νν', 'H→bb̄', 'H→cc̄', 'H→gg', 'H→4q',
    'H→ℓνqq′', 'Z→qq̄', 'W→qq′', 't→bqq′', 't→bℓν'
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a LorentzParT classifier")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--config-path', type=str, default='./configs/train_lorentz_part.yaml')
    parser.add_argument('--weights', type=str, required=True,
                        help="Path to model weights (.pt)")
    parser.add_argument('--model', type=str, default='lorentz_part',
                        choices=['lorentz_part', 'linear_probe'],
                        help="Model type to evaluate")
    parser.add_argument('--run-name', type=str, default='eval')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outputs-dir', type=str, default='./outputs')
    parser.add_argument('--json-out', type=str, default=None,
                        help="Path to save JSON results (optional)")
    return parser.parse_args()


def load_model(args, model_cfg, device):
    state_dict = torch.load(args.weights, map_location='cpu')

    if args.model == 'linear_probe':
        # LinearProbeModel state dict contains head.* and encoder.* keys
        # We reconstruct from scratch and load the full state dict
        model = LinearProbeModel(
            encoder_weights=args.weights,  # ignored — we load full state dict below
            embed_dim=model_cfg.embed_dim,
            num_classes=model_cfg.num_classes,
            encoder_kwargs=dict(
                num_heads=model_cfg.num_heads,
                num_layers=model_cfg.num_layers,
                dropout=model_cfg.dropout,
                expansion_factor=model_cfg.expansion_factor,
                pair_embed_dims=model_cfg.pair_embed_dims,
            ),
        )
        # Overwrite with the saved full state dict if it has head.* keys
        if any(k.startswith('head.') for k in state_dict):
            model.load_state_dict(state_dict, strict=False)
    else:
        model_cfg.inference = True
        model = LorentzParT(config=model_cfg)
        model.load_state_dict(state_dict, strict=False)

    return model.to(device)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = LorentzParTConfig.from_dict(config['model'])
    train_cfg = TrainConfig.from_dict(config['train'])

    test_dataset = NpyJetClassDataset(
        particles_path=os.path.join(args.data_dir, 'test', 'particles.npy'),
        labels_path=os.path.join(args.data_dir, 'test', 'labels.npy'),
        normalize=NORMALIZE,
        norm_dict=NORM_DICT,
        mask_mode=None,
    )
    # Dummy train dataset — Trainer requires it but we only call evaluate()
    train_dataset = NpyJetClassDataset(
        particles_path=os.path.join(args.data_dir, 'train', 'particles.npy'),
        labels_path=os.path.join(args.data_dir, 'train', 'labels.npy'),
        normalize=NORMALIZE,
        norm_dict=NORM_DICT,
        mask_mode=None,
    )

    model = load_model(args, model_cfg, device)

    trainer = JetClassTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        test_dataset=test_dataset,
        device=device,
        metric=accuracy_metric_ce,
        config=train_cfg,
    )
    trainer._set_logging_paths(args.run_name)

    os.makedirs(args.outputs_dir, exist_ok=True)
    trainer.outputs_dir = args.outputs_dir

    test_loss, test_acc, y_true, y_pred = trainer.evaluate(
        loss_type='cross_entropy',
        plot=[plot_roc_curve, plot_confusion_matrix]
    )

    # Primary metric: macro-averaged OVO ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo')

    # Per-class accuracy
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true_labels == i
        if mask.sum() == 0:
            continue
        per_class_acc[name] = float((y_pred_labels[mask] == i).mean())

    print(f"\n{'=' * 50}")
    print(f"Run:      {args.run_name}")
    print(f"Weights:  {args.weights}")
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"ROC AUC (OVO): {roc_auc:.4f}")
    print(f"{'=' * 50}")
    print("\nPer-class accuracy:")
    for name, acc in per_class_acc.items():
        print(f"  {name:14s}: {acc:.4f}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
        results = {
            'run_name': args.run_name,
            'weights': args.weights,
            'model': args.model,
            'seed': args.seed,
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'roc_auc_ovo': float(roc_auc),
            'per_class_acc': per_class_acc,
        }
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.json_out}")

    return test_loss, test_acc, roc_auc


if __name__ == '__main__':
    main()
