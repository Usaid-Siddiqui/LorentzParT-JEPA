"""
Phase 7 — compute ROC AUC, accuracy and per-class background rejection for trained cells by inference.

The trainers log only accuracy (val_metric); this loads each cell's best checkpoint, streams the val
set, and computes macro one-vs-one ROC AUC — directly comparable to published ParT's 0.9877. Mirrors
train_phase7's model + dataset construction exactly (same ARCH, features, NORM_DICT), so numbers match.

Background rejection follows Qu et al. 2022 Eq. 1-2: Rej_X = 1/FPR at TPR = X, for each signal class S
against QCD, with score p(S) / (p(S) + p(QCD)). X = 50%, except 99% for H->lvqq' and 99.5% for t->blv.
A rejection is only as good as the number of QCD jets that pass the cut, so that count is reported too:
with n jets passing, the relative uncertainty is about 1/sqrt(n). Rejections near 30000 therefore need
the full test set (2M QCD jets); point --val-dir at test_20M and raise --max-jets for those.

    python experiments/phase7_scaling/eval_auc.py --val-dir /home/jovyan/LPTJ/data/val_5M/val_5M \
        --scale 100m --seeds 42                       # add 123 456 as they finish
    python experiments/phase7_scaling/eval_auc.py --val-dir <val> --scale 10m --seeds 42 --max-jets 200000

Reads checkpoints from logs/<ModelDir>/best/<model>_<protocol>_<scale>_seed<N>.pt. Needs scikit-learn.
Writes results/eval_<scale><_tag>_<data dir name>.csv, one row per cell and seed.
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

# (name, index in streaming_jetclass.LABELS, signal efficiency), in the column order of Qu et al. Table 1
QCD = 0
SIGNALS = [("Hbb", 1, 0.50), ("Hcc", 2, 0.50), ("Hgg", 3, 0.50), ("H4q", 4, 0.50), ("Hlvqq", 5, 0.99),
           ("tbqq", 8, 0.50), ("tblv", 9, 0.995), ("Wqq", 7, 0.50), ("Zqq", 6, 0.50)]
# Qu et al. 2022 Table 1: ParT, 100M training, 17 features, evaluated on the 20M test set
PUBLISHED = dict(auc=0.9877, acc=0.861, Hbb=10638, Hcc=4149, Hgg=123, H4q=1864, Hlvqq=5479,
                 tbqq=32787, tblv=15873, Wqq=543, Zqq=402)


def background_rejection(t, p):
    """{signal: (rejection, QCD jets passing)} — Rej_X = 1/FPR at TPR = X, score p(S)/(p(S)+p(QCD))."""
    out = {}
    for name, idx, eff in SIGNALS:
        sig, bkg = p[t == idx], p[t == QCD]
        if len(sig) == 0 or len(bkg) == 0:
            out[name] = (float("nan"), 0)
            continue
        score = lambda a: a[:, idx] / np.clip(a[:, idx] + a[:, QCD], 1e-30, None)
        cut = np.quantile(score(sig), 1.0 - eff)
        passing = int((score(bkg) > cut).sum())
        out[name] = (len(bkg) / passing if passing else float("inf"), passing)
    return out


def load_model(model, ckpt, n_extra, device, cartesian_mv=False, bool_padding_mask=False):
    # cartesian_mv has NO parameters, so load_state_dict cannot detect a mismatch — it must
    # mirror the flag the run was trained with, or the predictions are silently wrong.
    extra = dict(cartesian_mv=cartesian_mv) if model == 'lorentzpart' else {}
    extra['bool_padding_mask'] = bool_padding_mask     # changes outputs; also has no parameters
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
    return (roc_auc_score(t, p, multi_class="ovo", average="macro"), accuracy_score(t, p.argmax(1)),
            len(t), background_rejection(t, p), int((t == QCD).sum()))


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
    p.add_argument("--bool-padding-mask", action="store_true",
                   help="MUST match training: boolean (true) padding mask")
    args = p.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    norm = T7.NORM_DICT_COMMON if args.common_scale else T7.NORM_DICT
    print(f"[eval] tag={args.tag} common_scale={args.common_scale} cartesian_mv={args.cartesian_mv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = T7.FEATURES[args.features]
    n_extra = len(feats) - 4

    names = [n for n, _, _ in SIGNALS]
    header = f"{'cell':22}{'seed':>6}{'AUC':>8}{'acc':>8}" + "".join(f"{n:>9}" for n in names)
    print(f"background rejection vs QCD at 50% signal efficiency (Hlvqq 99%, tblv 99.5%)\n{header}")
    print(f"{'published ParT (test)':22}{'':>6}{PUBLISHED['auc']:>8.4f}{PUBLISHED['acc']:>8.4f}"
          + "".join(f"{PUBLISHED[n]:>9}" for n in names))
    rows, agg = [], {}
    for model, proto in CELLS:
        aucs, accs = [], []
        for seed in args.seeds:
            run = f"{model}_{proto}_{args.scale}_seed{seed}{sfx}"
            ckpt = os.path.join(args.logs_dir, MODEL_DIR[model], "best", f"{run}.pt")
            if not os.path.exists(ckpt):
                print(f"{model+'_'+proto:26}{seed:>6}{'— (no ckpt)':>30}")
                continue
            auc, acc, n, rej, n_qcd = evaluate(load_model(model, ckpt, n_extra, device, args.cartesian_mv,
                                              args.bool_padding_mask),
                                   args.val_dir, feats, device, args.max_jets,
                                   args.batch_size, args.num_workers, norm)
            aucs.append(auc); accs.append(acc)
            print(f"{model+'_'+proto:22}{seed:>6}{auc:>8.4f}{acc:>8.4f}"
                  + "".join(f"{rej[k][0]:>9.0f}" for k in names), flush=True)
            print(f"{'  QCD jets passing':22}{'':>6}{'':>8}{'':>8}"
                  + "".join(f"{rej[k][1]:>9}" for k in names) + f"   (of {n_qcd} QCD, {n} jets)", flush=True)
            rows.append(dict(model=model, protocol=proto, seed=seed, auc=auc, acc=acc, jets=n, n_qcd=n_qcd,
                             **{f"rej_{k}": rej[k][0] for k in names},
                             **{f"pass_{k}": rej[k][1] for k in names}))
            # Guard: eval accuracy should reproduce the training log's best val accuracy. A large
            # gap almost always means the eval flags don't match how the run was trained.
            logged = best_logged_val(args.logs_dir, model, run)
            if logged is not None and abs(acc - logged) > 0.02:
                print(f"    [WARN] eval acc {acc:.4f} vs training best val {logged:.4f} - check that "
                      f"--common-scale / --cartesian-mv / --tag match the training run", flush=True)
        if aucs:
            agg[(model, proto)] = (np.mean(aucs), np.std(aucs), len(aucs))

    if rows:
        tag_dir = os.path.basename(os.path.normpath(args.val_dir))
        out = os.path.join(_HERE, "results", f"eval_{args.scale}{sfx}_{tag_dir}.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out}")

    if any(len(v) and v[2] > 1 for v in agg.values()):
        print(f"\n{'mean±std over seeds':22}{'AUC':>18}{'acc':>18}" + "".join(f"{n:>14}" for n in names))
        for (model, proto), (mu, sd, k) in agg.items():
            rs = [r for r in rows if (r["model"], r["protocol"]) == (model, proto)]
            line = f"{model+'_'+proto:22}{f'{mu:.4f}±{sd:.4f}':>18}"
            accs = [r["acc"] for r in rs]
            line += f"{f'{np.mean(accs):.4f}±{np.std(accs):.4f}':>18}"
            for n in names:
                vals = [r[f"rej_{n}"] for r in rs]
                line += f"{f'{np.mean(vals):.0f}±{np.std(vals):.0f}':>14}"
            print(line + f"  (n={k})")


if __name__ == "__main__":
    main()
