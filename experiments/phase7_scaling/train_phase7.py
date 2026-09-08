"""
Phase 7 training entry — one cell of the 2x2 {part, lorentzpart} x {scratch, jepa}, streaming JetClass.

torchrun-compatible (reads RANK/LOCAL_RANK/WORLD_SIZE from env → DDP), so the dual-GPU workflow is:
    torchrun --standalone --nproc_per_node=2 experiments/phase7_scaling/train_phase7.py \
        --model lorentzpart --protocol scratch \
        --train-dir /data/jetclass/train_100M --val-dir /data/jetclass/val_5M \
        --config experiments/phase7_scaling/configs/phase7_supervised.yaml \
        --run-name lorentzpart_scratch_100m_seed42 --seed 42

Protocols: scratch (supervised), jepa_pretrain (SSL), jepa_finetune (supervised from a JEPA encoder
via --weights). Encoder architecture is FIXED across all cells (parity); only the train recipe (yaml)
and the model/protocol vary. Features default to the full 14 (4-vector + 4 displacement + 6 PID).
"""

import argparse
import os
import sys

import torch
import yaml
from torch.distributed import init_process_group

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

from src.configs import TrainConfig
from src.engine import JetClassTrainer, JEPATrainer
from src.models import LorentzParT, ParticleTransformer, ParticleJEPA
from src.utils import set_seed, accuracy_metric_ce, cleanup_ddp
from src.utils.data.streaming_jetclass import StreamingJetClassDataset, KINEMATIC

DISPLACEMENT = ['part_d0val', 'part_d0err', 'part_dzval', 'part_dzerr']
PID = ['part_charge', 'part_isChargedHadron', 'part_isNeutralHadron',
       'part_isPhoton', 'part_isElectron', 'part_isMuon']
FEATURES = {4: KINEMATIC, 14: KINEMATIC + DISPLACEMENT + PID}

NORM_DICT = {'pT': (92.72917175292969, 105.83937072753906),
             'eta': (0.0005733045982196927, 0.9174848794937134),
             'phi': (-0.00041169871110469103, 1.8136887550354004),
             'energy': (133.8745574951172, 167.528564453125)}
NORMALIZE = [True, False, False, True]

# Shared encoder architecture — IDENTICAL for every cell so params match (parity is the whole point).
ARCH = dict(embed_dim=128, num_heads=8, num_layers=8)
# JEPA predictor/EMA — the corrected phase-2/6 recipe.
JEPA_EXTRA = dict(predictor_dim=64, predictor_heads=4, predictor_layers=4,
                  ema_momentum=0.999, use_attention_gate=True)


def init_ddp():
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        init_process_group('nccl')
        return local_rank, int(os.environ['WORLD_SIZE'])
    return 0, 1


def parse_args():
    p = argparse.ArgumentParser(description="Phase 7 — one 2x2 cell")
    p.add_argument('--model', required=True, choices=['part', 'lorentzpart'])
    p.add_argument('--protocol', required=True, choices=['scratch', 'jepa_pretrain', 'jepa_finetune'])
    p.add_argument('--train-dir', required=True)
    p.add_argument('--val-dir', required=True)
    p.add_argument('--config', required=True, help='train-recipe yaml (train: section)')
    p.add_argument('--run-name', required=True)
    p.add_argument('--features', type=int, default=14, choices=[4, 14])
    p.add_argument('--weights', default=None, help='JEPA encoder checkpoint (jepa_finetune only)')
    p.add_argument('--num-classes', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-epochs', type=int, default=None, help='override config')
    p.add_argument('--steps-per-epoch', type=int, default=None, help='override config')
    return p.parse_args()


def build_model(args, n_extra):
    encoder_type = 'part' if args.model == 'part' else 'lorentz'
    if args.protocol == 'jepa_pretrain':
        return ParticleJEPA(**ARCH, **JEPA_EXTRA, ragged_pair_embed=True,
                            num_extra_features=n_extra, encoder_type=encoder_type)
    # supervised (scratch or jepa_finetune): classification model, optionally encoder-initialized
    cls = ParticleTransformer if args.model == 'part' else LorentzParT
    return cls(**ARCH, num_cls_layers=2, num_classes=args.num_classes,
               ragged_pair_embed=True, num_extra_features=n_extra,
               weights=args.weights if args.protocol == 'jepa_finetune' else None)


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    local_rank, world_size = init_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    feats = FEATURES[args.features]
    n_extra = len(feats) - 4
    is_jepa = args.protocol == 'jepa_pretrain'
    mask_mode = 'biased' if is_jepa else None

    ds_kw = dict(particle_features=feats, norm_dict=NORM_DICT, normalize=NORMALIZE,
                 mask_mode=mask_mode, seed=args.seed)
    train_ds = StreamingJetClassDataset(args.train_dir, **ds_kw)
    val_ds = StreamingJetClassDataset(args.val_dir, **ds_kw)

    with open(args.config) as f:
        tcfg = TrainConfig.from_dict(yaml.safe_load(f)['train'])
    if args.num_epochs is not None:
        tcfg.num_epochs = args.num_epochs
    if args.steps_per_epoch is not None:
        tcfg.steps_per_epoch = args.steps_per_epoch

    model = build_model(args, n_extra).to(device)

    common = dict(model=model, train_dataset=train_ds, val_dataset=val_ds, device=device, config=tcfg)
    if is_jepa:
        trainer = JEPATrainer(**common, ema_momentum_start=0.999, ema_momentum_end=1.0)
    else:
        trainer = JetClassTrainer(**common, metric=accuracy_metric_ce)
    trainer._set_logging_paths(args.run_name)

    if local_rank == 0:
        print(f"[phase7] {args.run_name}: model={args.model} protocol={args.protocol} "
              f"features={args.features} (n_extra={n_extra}) world_size={world_size} "
              f"steps/epoch={tcfg.steps_per_epoch} epochs={tcfg.num_epochs} amp={tcfg.amp}", flush=True)

    trainer.train()
    cleanup_ddp()
    if local_rank == 0:
        print(f"[phase7] done → {trainer.best_model_path}", flush=True)


if __name__ == '__main__':
    main()
