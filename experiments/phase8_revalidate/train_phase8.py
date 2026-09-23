"""
Phase 8 revalidation — one cell of {ParT, LorentzParT} x {scratch, JEPA, MAE}, streaming JetClass.

Same machinery as Phase 7, extended with the MAE protocols and with the Phase 8 fixes ON by
default. The point is to re-ask "does SSL help?" now that pretraining is not silently crippled:

  * the interaction matrix was information-free during JEPA/MAE pretraining (and only then),
  * the JEPA target encoder was nondeterministic (dropout + batch-stat BN),
  * ln_m2 was a dead constant for ~100% of pairs,
  * npy 'biased' masking was uniform (streaming was already correct).

Run the SAME cell with --broken to reproduce the pre-fix behaviour for a paired A/B.

    torchrun --standalone --nproc_per_node=1 experiments/phase8_revalidate/train_phase8.py \
        --model lorentzpart --protocol mae_pretrain \
        --train-dir <train_10M> --val-dir <val_5M> \
        --config experiments/phase8_revalidate/configs/phase8_mae_pretrain.yaml \
        --run-name lorentzpart_maepre_1m_seed42 --steps-per-epoch 2000 --num-epochs 4
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
sys.path.insert(0, os.path.join(_REPO, "experiments", "phase7_scaling"))

import train_phase7 as T7                      # ARCH / FEATURES / NORM_DICT(_COMMON) / NORMALIZE
from src.configs import TrainConfig
from src.engine import JetClassTrainer, JEPATrainer, MaskedModelTrainer
from src.models import LorentzParT, ParticleTransformer, ParticleJEPA
from src.utils import set_seed, accuracy_metric_ce, cleanup_ddp
from src.utils.data.streaming_jetclass import StreamingJetClassDataset

PROTOCOLS = ['scratch', 'jepa_pretrain', 'jepa_finetune', 'mae_pretrain', 'mae_finetune']


def init_ddp():
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        init_process_group('nccl')
        return local_rank, int(os.environ['WORLD_SIZE'])
    return 0, 1


def parse_args():
    p = argparse.ArgumentParser(description="Phase 8 — one revalidation cell")
    p.add_argument('--model', required=True, choices=['part', 'lorentzpart'])
    p.add_argument('--protocol', required=True, choices=PROTOCOLS)
    p.add_argument('--train-dir', required=True)
    p.add_argument('--val-dir', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--features', type=int, default=4, choices=[4, 14])
    p.add_argument('--weights', default=None, help='encoder checkpoint (for *_finetune)')
    p.add_argument('--num-classes', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-epochs', type=int, default=None)
    p.add_argument('--steps-per-epoch', type=int, default=None)
    p.add_argument('--mask-mode', default='biased', choices=['random', 'biased', 'first'])
    p.add_argument('--num-mask', type=int, default=1)
    p.add_argument('--broken', action='store_true',
                   help='reproduce pre-Phase-8 feature scaling (the A/B control arm)')
    return p.parse_args()


def build_model(args, n_extra):
    fixed = not args.broken
    is_lorentz = args.model == 'lorentzpart'
    mv = dict(cartesian_mv=fixed) if is_lorentz else {}

    if args.protocol == 'jepa_pretrain':
        return ParticleJEPA(**T7.ARCH, **T7.JEPA_EXTRA, ragged_pair_embed=True,
                            num_extra_features=n_extra,
                            encoder_type='lorentz' if is_lorentz else 'part',
                            cartesian_mv=fixed and is_lorentz)

    cls = LorentzParT if is_lorentz else ParticleTransformer
    if args.protocol == 'mae_pretrain':
        # MAE reconstructs the masked particle's 4-vector: same backbone, mask=True head.
        return cls(**T7.ARCH, num_cls_layers=2, num_classes=args.num_classes,
                   ragged_pair_embed=True, num_extra_features=n_extra, mask=True, **mv)

    return cls(**T7.ARCH, num_cls_layers=2, num_classes=args.num_classes,
               ragged_pair_embed=True, num_extra_features=n_extra, **mv,
               weights=args.weights if args.protocol.endswith('_finetune') else None)


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    local_rank, world_size = init_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    feats = T7.FEATURES[args.features]
    n_extra = len(feats) - 4
    pretrain = args.protocol in ('jepa_pretrain', 'mae_pretrain')
    mask_mode = args.mask_mode if pretrain else None
    norm = T7.NORM_DICT if args.broken else T7.NORM_DICT_COMMON

    ds_kw = dict(particle_features=feats, norm_dict=norm, normalize=T7.NORMALIZE,
                 mask_mode=mask_mode, num_mask=args.num_mask, seed=args.seed)
    train_ds = StreamingJetClassDataset(args.train_dir, **ds_kw)
    val_ds = StreamingJetClassDataset(args.val_dir, **ds_kw)

    with open(args.config) as f:
        tcfg = TrainConfig.from_dict(yaml.safe_load(f)['train'])
    if args.num_epochs is not None:
        tcfg.num_epochs = args.num_epochs
        # Keep the cosine schedule in step with the ACTUAL epoch count. The configs ship
        # T_max: 30 to match their default num_epochs; overriding epochs without this left
        # T_max at 30, so a 4-epoch run annealed the LR by ~4% (0.000997 -> 0.000957) and
        # effectively trained at a constant peak LR.
        sch = getattr(tcfg, 'scheduler', None)
        if isinstance(sch, dict) and 'T_max' in sch:
            sch['T_max'] = args.num_epochs
    if args.steps_per_epoch is not None:
        tcfg.steps_per_epoch = args.steps_per_epoch

    model = build_model(args, n_extra).to(device)
    common = dict(model=model, train_dataset=train_ds, val_dataset=val_ds,
                  device=device, config=tcfg)

    if args.protocol == 'jepa_pretrain':
        trainer = JEPATrainer(**common, ema_momentum_start=0.999, ema_momentum_end=1.0)
    elif args.protocol == 'mae_pretrain':
        trainer = MaskedModelTrainer(**common)
    else:
        trainer = JetClassTrainer(**common, metric=accuracy_metric_ce)
    trainer._set_logging_paths(args.run_name)

    if local_rank == 0:
        print(f"[phase8] {args.run_name}: model={args.model} protocol={args.protocol} "
              f"features={args.features} FIXED={not args.broken} mask_mode={mask_mode} "
              f"world_size={world_size} steps/epoch={tcfg.steps_per_epoch} "
              f"epochs={tcfg.num_epochs} amp={tcfg.amp}", flush=True)

    trainer.train()
    cleanup_ddp()
    if local_rank == 0:
        print(f"[phase8] done → {trainer.best_model_path}", flush=True)


if __name__ == '__main__':
    main()
