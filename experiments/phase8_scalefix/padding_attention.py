"""
Phase 8 — how much attention do TRAINED models send to padded particles?

The models build their padding mask as a float, and PyTorch ADDS a float key_padding_mask to the
attention scores (only a bool mask excludes keys). So padded slots were never masked: each got +1
on its score, and at initialisation ~86% of a real particle's attention lands on padding. Padded
slots all hold the same constant vector, so a trained network CAN learn to suppress them — this
measures whether it did, on the real checkpoints, without any retraining.

For every attention layer it reports the mean attention mass that REAL particles (and the class
token) put on PADDED keys, next to the "uniform" mass a content-blind layer would give (the
fraction of keys that are padding). Mass far below uniform = the model learned to ignore padding.

    python experiments/phase8_scalefix/padding_attention.py --val-dir <val_5M> --scale 100m \
        --seeds 42 --tag fixed --common-scale --cartesian-mv

The flags must match how the runs were trained (same as eval_auc.py).
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "experiments", "phase7_scaling"))

import eval_auc as EA                       # load_model / MODEL_DIR / CELLS / T7 — eval == train
from src.models.particle_transformer import ParticleAttentionBlock
from src.models.classifier import ClassAttentionBlock
from src.utils.data.streaming_jetclass import StreamingJetClassDataset


class PaddingMassMeter:
    """Forward hooks on every attention layer; accumulates attention mass on padded keys."""

    def __init__(self, model):
        self.valid = None                                   # (B, N) bool, set before each forward
        self.mass = defaultdict(list)                       # layer -> per-jet mass on padding
        self.uniform = defaultdict(list)                    # layer -> per-jet padding fraction
        self.handles = []
        n_enc = n_cls = 0
        for mod in model.modules():
            if isinstance(mod, ParticleAttentionBlock):
                n_enc += 1
                self.handles.append(mod.pmha.register_forward_hook(self._hook(f"enc{n_enc}", cls=False)))
            elif isinstance(mod, ClassAttentionBlock):
                n_cls += 1
                self.handles.append(mod.mha.register_forward_hook(self._hook(f"cls{n_cls}", cls=True)))

    def _hook(self, name, cls):
        def fn(module, args, output):
            w = output[1]                                   # (B, L, S), averaged over heads
            if w is None:
                return
            w = w.float()
            pad = ~self.valid                               # (B, N) padded particles
            if cls:                                         # keys = [class token] + particles
                padk = torch.cat([torch.zeros_like(pad[:, :1]), pad], dim=1).float()
                m = (w[:, 0, :] * padk).sum(-1)             # (B,)
                u = padk.mean(-1)
            else:                                           # queries: real particles only
                m_q = (w * pad[:, None, :].float()).sum(-1)               # (B, N)
                real = self.valid.float()
                m = (m_q * real).sum(-1) / real.sum(-1).clamp(min=1)
                u = pad.float().mean(-1)
            self.mass[name].extend(m.cpu().tolist())
            self.uniform[name].extend(u.cpu().tolist())
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def measure(model, val_dir, feats, device, norm_dict, max_jets, bs, workers):
    meter = PaddingMassMeter(model)
    ds = StreamingJetClassDataset(val_dir, particle_features=feats, norm_dict=norm_dict,
                                  normalize=EA.T7.NORMALIZE, mask_mode=None, seed=0)
    n = 0
    for X, _ in DataLoader(ds, batch_size=bs, num_workers=workers):
        X = X.to(device)
        meter.valid = X[..., 3] > 0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model(X)
        n += len(X)
        if n >= max_jets:
            break
    meter.remove()
    return meter, n


def main():
    p = argparse.ArgumentParser(description="Attention mass on padded particles in trained models")
    p.add_argument("--val-dir", required=True)
    p.add_argument("--logs-dir", default="./logs")
    p.add_argument("--scale", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--tag", default=None)
    p.add_argument("--features", type=int, default=14, choices=[4, 14])
    p.add_argument("--common-scale", action="store_true", help="MUST match training")
    p.add_argument("--cartesian-mv", action="store_true", help="MUST match training")
    p.add_argument("--bool-padding-mask", action="store_true", help="MUST match training")
    p.add_argument("--max-jets", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = EA.T7.FEATURES[args.features]
    n_extra = len(feats) - 4
    sfx = f"_{args.tag}" if args.tag else ""
    norm = EA.T7.NORM_DICT_COMMON if args.common_scale else EA.T7.NORM_DICT

    for model_name, proto in EA.CELLS:
        for seed in args.seeds:
            run = f"{model_name}_{proto}_{args.scale}_seed{seed}{sfx}"
            ckpt = os.path.join(args.logs_dir, EA.MODEL_DIR[model_name], "best", f"{run}.pt")
            if not os.path.exists(ckpt):
                print(f"\n{run}: no checkpoint — skipped")
                continue
            model = EA.load_model(model_name, ckpt, n_extra, device, args.cartesian_mv,
                                  args.bool_padding_mask)
            meter, n = measure(model, args.val_dir, feats, device, norm, args.max_jets,
                               args.batch_size, args.num_workers)
            print(f"\n{run}  ({n} val jets)")
            print(f"  {'layer':7}{'mass on padding':>17}{'uniform would be':>18}")
            enc = []
            for layer in meter.mass:
                m, u = np.mean(meter.mass[layer]), np.mean(meter.uniform[layer])
                print(f"  {layer:7}{m:>17.4f}{u:>18.4f}")
                if layer.startswith("enc"):
                    enc.append(m)
            mean_enc = float(np.mean(enc)) if enc else float("nan")
            verdict = ("bool mask: padding excluded by construction" if args.bool_padding_mask else
                       "learned to ignore padding — effectively cosmetic" if mean_enc < 0.01 else
                       "SUBSTANTIAL attention still on padding" if mean_enc > 0.10 else
                       "moderate attention on padding")
            print(f"  mean over encoder layers: {mean_enc:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
