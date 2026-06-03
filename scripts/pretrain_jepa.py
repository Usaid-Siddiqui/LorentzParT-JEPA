"""
JEPA pretraining script for ParticleJEPA on the 100k JetClass subset.

Run after scripts/prepare_data.py has created the data splits.

Usage:
    python scripts/pretrain_jepa.py \\
        --data-dir ./data \\
        --config-path ./configs/pretrain_jepa.yaml \\
        --seed 42

The best model is saved to ./logs/ParticleJEPA/best/<run>.pt
Context encoder weights are stored with the "encoder." prefix, making them
directly compatible with the fine-tuning scripts (finetune.py).
"""

import os
import sys
import yaml
import argparse
import warnings

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs import JEPAConfig, TrainConfig
from src.engine import JEPATrainer
from src.models import ParticleJEPA
from src.utils import set_seed, setup_ddp, cleanup_ddp
from src.utils.data import NpyJetClassDataset
from src.utils.viz import plot_jepa_history

warnings.filterwarnings('ignore')


# Normalization statistics computed from the JetClass training set
NORM_DICT = {
    'pT':     (92.72917175292969,  105.83937072753906),
    'eta':    (0.0005733045982196927, 0.9174848794937134),
    'phi':    (-0.00041169871110469103, 1.8136887550354004),
    'energy': (133.8745574951172, 167.528564453125),
}
NORMALIZE = [True, False, False, True]


def parse_args():
    parser = argparse.ArgumentParser(description="JEPA pretraining on 100k JetClass subset")
    parser.add_argument('--data-dir', type=str, default='./data',
                        help="Root data directory (must contain train/ and val/ sub-dirs)")
    parser.add_argument('--config-path', type=str, default='./configs/pretrain_jepa.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help="Resume from a checkpoint file")
    parser.add_argument('--run-name', type=str, default=None,
                        help="Fixed run name for deterministic save paths (e.g. jepa_pretrain)")
    return parser.parse_args()


def main(rank, world_size, seed, config_path, data_dir, checkpoint_path=None, run_name=None):
    set_seed(seed)
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = JEPAConfig.from_dict(config['model'])
    train_cfg = TrainConfig.from_dict(config['train'])

    # Datasets
    train_dataset = NpyJetClassDataset(
        particles_path=os.path.join(data_dir, 'train', 'particles.npy'),
        labels_path=os.path.join(data_dir, 'train', 'labels.npy'),
        normalize=NORMALIZE,
        norm_dict=NORM_DICT,
        mask_mode='random',
    )
    val_dataset = NpyJetClassDataset(
        particles_path=os.path.join(data_dir, 'val', 'particles.npy'),
        labels_path=os.path.join(data_dir, 'val', 'labels.npy'),
        normalize=NORMALIZE,
        norm_dict=NORM_DICT,
        mask_mode='random',
    )

    # Model
    model = ParticleJEPA(
        embed_dim=model_cfg.embed_dim,
        num_heads=model_cfg.num_heads,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        expansion_factor=model_cfg.expansion_factor,
        pair_embed_dims=model_cfg.pair_embed_dims,
        predictor_dim=model_cfg.predictor_dim,
        predictor_heads=model_cfg.predictor_heads,
        predictor_layers=model_cfg.predictor_layers,
        predictor_dropout=model_cfg.predictor_dropout,
        max_num_particles=model_cfg.max_num_particles,
        ema_momentum=model_cfg.ema_momentum_start,
        use_attention_gate=model_cfg.use_attention_gate,
    ).to(device)

    # Trainer
    trainer = JEPATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=train_cfg,
        ema_momentum_start=model_cfg.ema_momentum_start,
        ema_momentum_end=model_cfg.ema_momentum_end,
    )

    # Set a fixed run name so the best model is always at a predictable path
    if run_name is not None and rank == 0:
        trainer._set_logging_paths(run_name)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)

    history, _ = trainer.train()

    cleanup_ddp()

    if rank == 0:
        output_path = (
            os.path.join(trainer.outputs_dir, f"{trainer.run_name}_jepa_history.png")
            if train_cfg.save_fig else None
        )
        plot_jepa_history(history, save_fig=output_path)
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
