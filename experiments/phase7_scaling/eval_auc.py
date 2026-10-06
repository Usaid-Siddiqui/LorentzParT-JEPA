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
import csv
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


def load_model(model, ckpt, n_extra, device, cartesian_mv=False):
    # cartesian_mv has NO parameters, so load_state_dict cannot detect a mismatch — it must
    # mirror the flag the run was trained with, or the predictions are silently wrong.
    extra = dict(cartesian_mv=cartesian_mv) if model == 'lorentzpart' else {}
    m = MODEL_CLS[model](**T7.ARCH, num_cls_layers=2, num_classes=10,
                         ragged_pair_embed=True, num_extra_features=n_extra, **extra)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    return m.to(device).eval()


@torch.no_grad()
def evaluate(m, val_dir, feats, device, max_jets, bs, workers, norm_dict):
    ds = StreamingJetClassDataset(val_dir, particle_features=feats, norm_dict=norm_dict,
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


def best_logged_val(logs_dir, model, run):
    """Best val_metric in the run's training CSV (None if the log is missing)."""
    path = os.path.join(logs_dir, MODEL_DIR[model], 'logging', f'{run}.csv')
    if not os.path.exists(path):
        return None
    vals = []
    for row in csv.DictReader(open(path)):
        try:
            vals.append(float(row['val_metric']))
        except (KeyError, ValueError, TypeError):
            pass
    return max(vals) if vals else None


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
    p.add_argument("--tag", default=None, help="run-name suffix used at training time, e.g. 'fixed'")
    p.add_argument("--common-scale", action="store_true",
                   help="MUST match training: Phase 8 pT/E common-scale normalisation")
    p.add_argument("--cartesian-mv", action="store_true",
                   help="MUST match training: Cartesian embed_vector for lorentzpart")
    args = p.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    norm = T7.NORM_DICT_COMMON if args.common_scale else T7.NORM_DICT
    print(f"[eval] tag={args.tag} common_scale={args.common_scale} cartesian_mv={args.cartesian_mv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = T7.FEATURES[args.features]
    n_extra = len(feats) - 4

    print(f"{'cell':26}{'seed':>6}{'AUC(ovo)':>11}{'acc':>9}{'jets':>10}")
    agg = {}
    for model, proto in CELLS:
        aucs, accs = [], []
        for seed in args.seeds:
            run = f"{model}_{proto}_{args.scale}_seed{seed}{sfx}"
            ckpt = os.path.join(args.logs_dir, MODEL_DIR[model], "best", f"{run}.pt")
            if not os.path.exists(ckpt):
                print(f"{model+'_'+proto:26}{seed:>6}{'— (no ckpt)':>30}")
                continue
            auc, acc, n = evaluate(load_model(model, ckpt, n_extra, device, args.cartesian_mv),
                                   args.val_dir, feats, device, args.max_jets,
                                   args.batch_size, args.num_workers, norm)
            aucs.append(auc); accs.append(acc)
            print(f"{model+'_'+proto:26}{seed:>6}{auc:>11.4f}{acc:>9.4f}{n:>10}", flush=True)
            # Guard: eval accuracy should reproduce the training log's best val accuracy. A large
            # gap almost always means the eval flags don't match how the run was trained.
            logged = best_logged_val(args.logs_dir, model, run)
            if logged is not None and abs(acc - logged) > 0.02:
                print(f"    [WARN] eval acc {acc:.4f} vs training best val {logged:.4f} - check that "
                      f"--common-scale / --cartesian-mv / --tag match the training run", flush=True)
        if aucs:
            agg[(model, proto)] = (np.mean(aucs), np.std(aucs), len(aucs))

    if any(len(v) and v[2] > 1 for v in agg.values()):
        print(f"\n{'cell':26}{'AUC mean±std (n)':>24}")
        for (model, proto), (mu, sd, k) in agg.items():
            print(f"{model+'_'+proto:26}{f'{mu:.4f}±{sd:.4f} (n={k})':>24}")


if __name__ == "__main__":
    main()
