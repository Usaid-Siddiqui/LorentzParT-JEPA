"""
Fine-tuning script — unified for all three comparison conditions.

Loads the LorentzParT encoder, optionally initialises it from a pretrained
checkpoint, and fine-tunes on the 100k JetClass subset classification task.

Usage:
    # JEPA pretrained
    python scripts/finetune.py \\
        --data-dir ./data \\
        --weights ./logs/ParticleJEPA/best/<run>.pt \\
        --run-name jepa_finetune \\
        --seed 42

    # MAE pretrained
    python scripts/finetune.py \\
        --data-dir ./data \\
        --weights ./logs/LorentzParT/best/<run>.pt \\
        --run-name mae_finetune \\
        --seed 42

    # From scratch (no weights)
    python scripts/finetune.py \\
        --data-dir ./data \\
        --run-name scratch \\
        --seed 42

Trained classifiers are saved to ./logs/LorentzParT/best/<run-name>.pt
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
from src.engine import JetClassTrainer
from src.models import LorentzParT
from src.utils import accuracy_metric_ce, set_seed, setup_ddp, cleanup_ddp
from src.utils.data import NpyJetClassDataset
from src.utils.viz import plot_history

warnings.filterwarnings('ignore')


NORM_DICT = {
    'pT':     (92.72917175292969,  105.83937072753906),
    'eta':    (0.0005733045982196927, 0.9174848794937134),
    'phi':    (-0.00041169871110469103, 1.8136887550354004),
    'energy': (133.8745574951172, 167.528564453125),
}
NORMALIZE = [True, False, False, True]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LorentzParT for jet classification")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--config-path', type=str, default='./configs/finetune.yaml')
    parser.add_argument('--weights', type=str, default=None,
                        help="Path to pretrained encoder weights (.pt file)")
    parser.add_argument('--run-name', type=str, default=None,
                        help="Custom run name for log files (optional)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    return parser.parse_args()


def main(rank, world_size, seed, config_path, data_dir, weights, run_name, checkpoint_path=None):
    set_seed(seed)
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = LorentzParTConfig.from_dict(config['model'])
    train_cfg = TrainConfig.from_dict(config['train'])

    # Override weights from command line
    if weights is not None:
        model_cfg.weights = weights

    train_dataset = NpyJetClassDataset(
        particles_path=os.path.join(data_dir, 'train', 'particles.npy'),
        labels_path=os.path.join(data_dir, 'train', 'labels.npy'),
        normalize=NORMALIZE,
        norm_dict=NORM_DICT,
        mask_mode=None,
    )
    val_dataset = NpyJetClassDataset(
        particles_path=os.path.join(data_dir, 'val', 'particles.npy'),
        labels_path=os.path.join(data_dir, 'val', 'labels.npy'),
        normalize=NORMALIZE,
        norm_dict=NORM_DICT,
        mask_mode=None,
    )
    test_dataset = NpyJetClassDataset(
        particles_path=os.path.join(data_dir, 'test', 'particles.npy'),
        labels_path=os.path.join(data_dir, 'test', 'labels.npy'),
        normalize=NORMALIZE,
        norm_dict=NORM_DICT,
        mask_mode=None,
    )

    model = LorentzParT(config=model_cfg).to(device)

    trainer = JetClassTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        metric=accuracy_metric_ce,
        config=train_cfg,
    )

    # Override run name for cleaner file naming
    if run_name is not None and rank == 0:
        trainer._set_logging_paths(run_name)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)

    history, _ = trainer.train()

    cleanup_ddp()

    if rank == 0:
        output_path = (
            os.path.join(trainer.outputs_dir, f"{trainer.run_name}_finetune_history.png")
            if train_cfg.save_fig else None
        )
        plot_history(history, save_fig=output_path)
        print(f"\nBest fine-tuned model saved to: {trainer.best_model_path}")
        print(f"Final val accuracy: {max(history.get('val_metric', [0])):.4f}")


if __name__ == '__main__':
    args = parse_args()
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            main,
            args=(
                world_size, args.seed, args.config_path, args.data_dir,
                args.weights, args.run_name, args.checkpoint_path
            ),
            nprocs=world_size,
        )
    else:
        main(
            rank=0,
            world_size=1,
            seed=args.seed,
            config_path=args.config_path,
            data_dir=args.data_dir,
            weights=args.weights,
            run_name=args.run_name,
            checkpoint_path=args.checkpoint_path,
        )
