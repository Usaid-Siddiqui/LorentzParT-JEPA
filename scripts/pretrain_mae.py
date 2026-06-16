"""
MAE pretraining script for LorentzParT on the 100k JetClass subset.

This is the baseline approach: the model reconstructs raw particle 4-vector
features (pT, η, φ, E) for a single randomly masked particle.

Usage:
    python scripts/pretrain_mae.py \\
        --data-dir ./data \\
        --config-path ./configs/pretrain_mae.yaml \\
        --seed 42

The best model is saved to ./logs/LorentzParT/best/<run>.pt
Weights are stored with the existing state_dict convention (no prefix),
so finetune.py loads them via the 'weights' parameter directly.
"""

import os
import sys
import yaml
import argparse
import warnings

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs import LorentzParTConfig, TrainConfig
from src.engine import MaskedModelTrainer
from src.models import LorentzParT
from src.utils import set_seed, setup_ddp, cleanup_ddp
from src.utils.data import NpyJetClassDataset

warnings.filterwarnings('ignore')


NORM_DICT = {
    'pT':     (92.72917175292969,  105.83937072753906),
    'eta':    (0.0005733045982196927, 0.9174848794937134),
    'phi':    (-0.00041169871110469103, 1.8136887550354004),
    'energy': (133.8745574951172, 167.528564453125),
}
NORMALIZE = [True, False, False, True]


def parse_args():
    parser = argparse.ArgumentParser(description="MAE pretraining on 100k JetClass subset")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--config-path', type=str, default='./configs/pretrain_mae.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--run-name', type=str, default=None,
                        help="Fixed run name for deterministic save paths (e.g. mae_pretrain)")
    return parser.parse_args()


def main(rank, world_size, seed, config_path, data_dir, checkpoint_path=None, run_name=None):
    set_seed(seed)
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = LorentzParTConfig.from_dict(config['model'])
    train_cfg = TrainConfig.from_dict(config['train'])

    train_dataset = NpyJetClassDataset(
        particles_path=os.path.join(data_dir, 'train', 'particles.npy'),
        labels_path=os.path.join(data_dir, 'train', 'labels.npy'),
        normalize=NORMALIZE,
        norm_dict=NORM_DICT,
        mask_mode=model_cfg.mask_mode,
    )
    val_dataset = NpyJetClassDataset(
        particles_path=os.path.join(data_dir, 'val', 'particles.npy'),
        labels_path=os.path.join(data_dir, 'val', 'labels.npy'),
        normalize=NORMALIZE,
        norm_dict=NORM_DICT,
        mask_mode=model_cfg.mask_mode,
    )

    model = LorentzParT(config=model_cfg).to(device)

    trainer = MaskedModelTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=train_cfg,
    )

    if run_name is not None and rank == 0:
        trainer._set_logging_paths(run_name)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)

    history, _ = trainer.train()

    cleanup_ddp()

    if rank == 0:
        print(f"\nBest model saved to: {trainer.best_model_path}")


if __name__ == '__main__':
    args = parse_args()
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            main,
            args=(world_size, args.seed, args.config_path, args.data_dir, args.checkpoint_path, args.run_name),
            nprocs=world_size,
        )
    else:
        main(
            rank=0,
            world_size=1,
            seed=args.seed,
            config_path=args.config_path,
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint_path,
            run_name=args.run_name,
        )
