"""
Measure training FLOPs per jet for each objective, so the Phase-2 cost figure
can use a compute axis that is immune to cluster wall-clock jitter.

Wall-clock on a shared/preemptible GPU swings ~2x within a single run, which
makes time-based cost comparisons noisy. FLOPs are deterministic: total training
compute = (FLOPs per jet) x (jets seen) = (FLOPs/jet) x N_train x epochs. This
script measures the (FLOPs/jet) constant for the three steps that matter:

    jepa_pretrain  ParticleJEPA step: context encoder (fwd+bwd) + EMA target
                   encoder (fwd only, no grad) + predictor (fwd+bwd)
    mae_pretrain   LorentzParT(mask=True) step: encoder + reconstruction head
    finetune       LorentzParT(mask=False) step: encoder + class-attention head
                   (shared by JEPA-ft, MAE-ft, and scratch)

Each is measured by running one real training step (forward + backward) under
torch's FlopCounterMode, which counts matmul / bmm / attention FLOPs (the
standard convention; elementwise ops are excluded, as in every FLOP report).
The target encoder runs under no_grad inside the JEPA forward, so it is charged
its forward pass but no backward -- exactly the real training cost.

We use a scalar surrogate loss (mean of squared outputs) instead of wiring up the
real criterion + targets: the loss's own FLOPs are negligible next to the
network, and this keeps the script decoupled from the loss/target plumbing.

Run ON A MACHINE WITH THE REPO'S TORCH (CPU is fine -- no GPU needed):

    python experiments/phase2/measure_flops.py --json-out experiments/phase2/flops.json

FLOPs/jet is batch-independent (matmuls are linear in batch), so a small batch
is measured and divided by batch size.
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.configs import JEPAConfig, LorentzParTConfig
from src.models import ParticleJEPA, LorentzParT

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError as e:  # pragma: no cover
    raise SystemExit("FlopCounterMode unavailable -- needs torch >= 1.13 "
                     "(2.1+ recommended). Got: %s" % e)


def count_step_flops(model, inputs, device, batch):
    """FLOPs for one fwd+bwd training step, returned per jet.

    `inputs` is the positional-arg tuple passed to model(...). A scalar surrogate
    loss (mean of squared outputs) drives the backward; its own FLOPs are
    negligible next to the network's."""
    model = model.to(device).train()
    fc = FlopCounterMode(display=False)
    with fc:
        out = model(*inputs)
        pred = out[0] if isinstance(out, (tuple, list)) else out   # JEPA returns (pred, target)
        loss = pred.float().pow(2).mean()
        loss.backward()
    return fc.get_total_flops() / batch


def make_batch(batch, n_particles, num_mask, device):
    """Synthetic (X, mask_idx) with no padded slots so all N particles are active
    (the model computes over every slot regardless of padding -> upper bound)."""
    x = torch.randn(batch, n_particles, 4, device=device)
    x[..., 3] = x[..., 3].abs() + 1.0          # energy != 0  -> nothing read as padding
    mask_idx = torch.randint(0, n_particles, (batch, num_mask), device=device)
    return x, mask_idx


def load(cfg_path):
    with open(cfg_path) as f:
        return yaml.safe_load(f)['model']


def main():
    p = argparse.ArgumentParser(description="Measure training FLOPs/jet per objective")
    p.add_argument('--config-jepa', default='./configs/pretrain_jepa.yaml')
    p.add_argument('--config-mae',  default='./configs/pretrain_mae.yaml')
    p.add_argument('--config-clf',  default='./configs/train_lorentz_part.yaml')
    p.add_argument('--batch',       type=int, default=32)
    p.add_argument('--device',      default='cpu')
    p.add_argument('--json-out',    default='./experiments/phase2/flops.json')
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    results = {}

    # ---- JEPA pretrain step ----
    jc = JEPAConfig.from_dict(load(args.config_jepa))
    jepa = ParticleJEPA(
        embed_dim=jc.embed_dim, num_heads=jc.num_heads, num_layers=jc.num_layers,
        dropout=jc.dropout, expansion_factor=jc.expansion_factor,
        pair_embed_dims=jc.pair_embed_dims,
        predictor_dim=jc.predictor_dim, predictor_heads=jc.predictor_heads,
        predictor_layers=jc.predictor_layers, predictor_dropout=jc.predictor_dropout,
        max_num_particles=jc.max_num_particles, ema_momentum=jc.ema_momentum_start,
        use_attention_gate=jc.use_attention_gate,
    )
    x, mask_idx = make_batch(args.batch, jc.max_num_particles, jc.num_mask, device)
    results['jepa_pretrain'] = count_step_flops(jepa, (x, mask_idx), device, args.batch)

    # ---- MAE pretrain step (LorentzParT, mask=True) ----
    mc = LorentzParTConfig.from_dict(load(args.config_mae))
    mae = LorentzParT(config=mc)
    x, mask_idx = make_batch(args.batch, mc.max_num_particles, 1, device)
    results['mae_pretrain'] = count_step_flops(mae, (x, mask_idx[:, 0]), device, args.batch)

    # ---- Finetune / scratch step (LorentzParT, mask=False) ----
    cc = LorentzParTConfig.from_dict(load(args.config_clf))
    clf = LorentzParT(config=cc)
    x, _ = make_batch(args.batch, cc.max_num_particles, 1, device)
    results['finetune'] = count_step_flops(clf, (x,), device, args.batch)

    print(f"\n{'step':16s}{'GFLOPs/jet (fwd+bwd)':>24s}")
    for k in ('jepa_pretrain', 'mae_pretrain', 'finetune'):
        print(f"{k:16s}{results[k] / 1e9:>24.4f}")
    print(f"\njepa_pretrain / finetune = {results['jepa_pretrain'] / results['finetune']:.2f}x")
    print(f"mae_pretrain  / finetune = {results['mae_pretrain'] / results['finetune']:.2f}x")

    os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
    with open(args.json_out, 'w') as f:
        json.dump({'flops_per_jet': results, 'batch': args.batch}, f, indent=2)
    print(f"\nwrote {args.json_out}")


if __name__ == '__main__':
    main()
