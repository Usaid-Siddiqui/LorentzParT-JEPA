"""
Phase 7 — compute ROC AUC (+ accuracy) for trained cells by inference over the val set.

The trainers log only accuracy (val_metric); this loads each cell's best checkpoint, streams the val
set, and computes macro one-vs-one ROC AUC — directly comparable to published ParT's 0.9877. Mirrors
train_phase7's model + dataset construction exactly (same ARCH, features, NORM_DICT), so numbers match.

    python experiments/phase7_scaling/eval_auc.py --val-dir /home/jovyan/LPTJ/data/val_5M/val_5M \
        --scale 100m --seeds 42                       # add 123 456 as they finish
    python experiments/phase7_scaling/eval_auc.py --val-dir <val> --scale 10m --seeds 42 --max-jets 200000

Reads checkpoints from logs/<ModelDir>/best/<model>_<protocol>_<scale>_seed<N>.pt. Needs scikit-learn.
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import train_phase7 as T7   # reuse FEATURES / NORM_DICT / NORMALIZE / ARCH so eval == train
from src.models import ParticleTransformer, LorentzParT
from src.utils.data.streaming_jetclass import StreamingJetClassDataset

MODEL_DIR = {"part": "ParticleTransformer", "lorentzpart": "LorentzParT"}
MODEL_CLS = {"part": ParticleTransformer, "lorentzpart": LorentzParT}
CELLS = [("part", "scratch"), ("part", "jepa"), ("lorentzpart", "scratch"), ("lorentzpart", "jepa")]


def load_model(model, ckpt, n_extra, device):
    m = MODEL_CLS[model](**T7.ARCH, num_cls_layers=2, num_classes=10,
                         ragged_pair_embed=True, num_extra_features=n_extra)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    return m.to(device).eval()


@torch.no_grad()
def evaluate(m, val_dir, feats, device, max_jets, bs, workers):
    ds = StreamingJetClassDataset(val_dir, particle_features=feats, norm_dict=T7.NORM_DICT,
                                  normalize=T7.NORMALIZE, mask_mode=None, seed=0)
    dl = DataLoader(ds, batch_size=bs, num_workers=workers)
    probs, trues, n = [], [], 0
    for X, y in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = m(X.to(device))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        trues.append(y.argmax(1).numpy())
        n += len(y)
        if n >= max_jets:
            break
    from sklearn.metrics import roc_auc_score, accuracy_score
    p, t = np.concatenate(probs), np.concatenate(trues)
    return roc_auc_score(t, p, multi_class="ovo", average="macro"), accuracy_score(t, p.argmax(1)), len(t)


def main():
    p = argparse.ArgumentParser(description="Phase 7 ROC AUC eval")
    p.add_argument("--val-dir", required=True)
    p.add_argument("--logs-dir", default="./logs")
    p.add_argument("--scale", required=True, help="100m / 10m / ...")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--features", type=int, default=14, choices=[4, 14])
    p.add_argument("--max-jets", type=int, default=500000, help="cap val jets for a stable AUC")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = T7.FEATURES[args.features]
    n_extra = len(feats) - 4

    print(f"{'cell':26}{'seed':>6}{'AUC(ovo)':>11}{'acc':>9}{'jets':>10}")
    agg = {}
    for model, proto in CELLS:
        aucs, accs = [], []
        for seed in args.seeds:
            run = f"{model}_{proto}_{args.scale}_seed{seed}"
            ckpt = os.path.join(args.logs_dir, MODEL_DIR[model], "best", f"{run}.pt")
            if not os.path.exists(ckpt):
                print(f"{model+'_'+proto:26}{seed:>6}{'— (no ckpt)':>30}")
                continue
            auc, acc, n = evaluate(load_model(model, ckpt, n_extra, device),
                                   args.val_dir, feats, device, args.max_jets,
                                   args.batch_size, args.num_workers)
            aucs.append(auc); accs.append(acc)
            print(f"{model+'_'+proto:26}{seed:>6}{auc:>11.4f}{acc:>9.4f}{n:>10}", flush=True)
        if aucs:
            agg[(model, proto)] = (np.mean(aucs), np.std(aucs), len(aucs))

    if any(len(v) and v[2] > 1 for v in agg.values()):
        print(f"\n{'cell':26}{'AUC mean±std (n)':>24}")
        for (model, proto), (mu, sd, k) in agg.items():
            print(f"{model+'_'+proto:26}{f'{mu:.4f}±{sd:.4f} (n={k})':>24}")


if __name__ == "__main__":
    main()
