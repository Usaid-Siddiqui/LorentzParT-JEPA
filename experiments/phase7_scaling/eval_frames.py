"""
Phase 7 — do trained models keep their predictions when the same jets are seen from another frame?

ParT and LorentzParT tie at 100M on in-distribution validation jets. This asks whether the Lorentz-
equivariant input layer buys robustness instead: every particle four-momentum in a set of validation
jets is transformed, the jets are re-normalised exactly as in training, and each model is scored.

  rotation      azimuthal rotation by angle a. Control: the collider is phi-symmetric and training covers
                every phi, so both models should be insensitive.
  long_boost    boost along the beam by rapidity y. Shifts every particle's rapidity; larger y pushes jets
                outside JetClass's |eta_jet| < 2 acceptance.
  transv_boost  boost along x with velocity beta. Changes the jet pT, moving jets out of the 500-1000 GeV
                training range. Not a symmetry of collider data — a stress test.

Reported per model and setting: AUC (one-vs-one macro), accuracy, and the FLIP RATE — the fraction of
jets whose predicted class differs from the same model's prediction on the untransformed jets (the
direct measure of sensitivity). Alongside, the fraction of jets each setting moves outside the training
acceptance, so robustness and out-of-distribution are kept apart.

Only the four-vector is transformed. The 10 extra per-particle features (track displacement, PID) are
left as they are — there is no consistent way to boost impact parameters — so this probes the
kinematic pathway, which is where the two architectures differ.

Before evaluating, the transform code must pass self-checks on real jets: a zero-size transform is
exact, each boost preserves every jet's invariant mass, and the coordinate round trip is lossless.

    python experiments/phase7_scaling/eval_frames.py --val-dir <val_5M> --scale 100m --seeds 42 \
        --tag fixed --common-scale --cartesian-mv
    # flags must match training, exactly as for eval_auc.py
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import eval_auc as EA                       # load_model / MODEL_DIR / CELLS / T7 — eval == train
from src.utils.data.streaming_jetclass import StreamingJetClassDataset

JET_PT_RANGE = (500.0, 1000.0)              # JetClass jet selection (GeV)
JET_ETA_MAX = 2.0


# ----------------------------------------------------------------------- Lorentz transforms
# P: (..., 4) Cartesian four-momenta (E, px, py, pz), float64
def rot_z(P, a):
    E, px, py, pz = P.unbind(-1)
    c, s = math.cos(a), math.sin(a)
    return torch.stack([E, c * px - s * py, s * px + c * py, pz], -1)


def boost_z(P, y):
    E, px, py, pz = P.unbind(-1)
    ch, sh = math.cosh(y), math.sinh(y)
    return torch.stack([ch * E + sh * pz, px, py, sh * E + ch * pz], -1)


def boost_x(P, b):
    E, px, py, pz = P.unbind(-1)
    g = 1.0 / math.sqrt(1.0 - b * b)
    return torch.stack([g * (E - b * px), g * (px - b * E), py, pz], -1)


FAMILIES = {
    "rotation":     (rot_z,   [0.5, 1.0, math.pi / 2, math.pi],        "azimuthal angle a (rad)"),
    "long_boost":   (boost_z, [0.25, 0.5, 1.0, 1.5, 2.0, 3.0],         "boost rapidity y"),
    "transv_boost": (boost_x, [0.1, 0.2, 0.3, 0.5],                     "boost velocity beta"),
}


def to_cart(x4):
    """(pT, eta, phi, E) -> (E, px, py, pz)."""
    pT, eta, phi, E = x4.unbind(-1)
    return torch.stack([E, pT * torch.cos(phi), pT * torch.sin(phi), pT * torch.sinh(eta)], -1)


def to_coords(P):
    """(E, px, py, pz) -> (pT, eta, phi, E); padded (all-zero) rows stay zero."""
    E, px, py, pz = P.unbind(-1)
    pT = torch.hypot(px, py)
    eta = torch.asinh(pz / pT.clamp(min=1e-12))
    return torch.stack([pT, eta, torch.atan2(py, px), E], -1)


def jet_kinematics(P, valid):
    """Jet four-vector = sum of its valid particles -> (pT, eta, m^2)."""
    J = (P * valid[..., None]).sum(1)
    E, px, py, pz = J.unbind(-1)
    pT = torch.hypot(px, py)
    return pT, torch.asinh(pz / pT.clamp(min=1e-12)), E ** 2 - px ** 2 - py ** 2 - pz ** 2


def model_input(raw, P, valid, norm_dict):
    """Rebuild the normalised model input exactly as the dataset's _norm() does."""
    x = raw.clone()
    x[..., :4] = to_coords(P)
    x[~valid] = 0.0                                    # padding stays exactly zero (all features)
    x[..., 0] = x[..., 0] / norm_dict['pT'][0]         # NORMALIZE = [True, False, False, True]:
    x[..., 3] = x[..., 3] / norm_dict['energy'][0]     # pT and E divided by their means, eta/phi raw
    return x.float()


# ----------------------------------------------------------------------- self-checks
def self_check(raw, device):
    """Refuse to run if the transform code is wrong — a plausible table from a buggy boost is worse
    than no table."""
    x = raw[:256].to(device, torch.float64)
    valid = x[..., 3] > 0
    P = to_cart(x[..., :4])
    back = to_coords(P)
    rt = (back[valid] - x[..., :4][valid]).abs()
    rt[:, 2] = torch.remainder(rt[:, 2] + math.pi, 2 * math.pi) - math.pi   # phi is periodic
    assert rt.abs().max() < 1e-6, f"coordinate round trip error {rt.abs().max():.2e}"
    _, _, m2 = jet_kinematics(P, valid)
    for name, (fn, values, _) in FAMILIES.items():
        assert torch.equal(fn(P, 0.0), P), f"{name}: zero-size transform is not the identity"
        for v in values:
            _, _, m2t = jet_kinematics(fn(P, v), valid)
            rel = ((m2t - m2).abs() / m2.abs().clamp(min=1e-6)).max().item()
            assert rel < 1e-6, f"{name}({v}): jet invariant mass changed by {rel:.2e}"
    print("[frames] self-checks passed: exact identity, lossless coordinates, jet mass invariant")


# ----------------------------------------------------------------------- evaluation
def load_raw(val_dir, feats, norm, max_jets, workers):
    ds = StreamingJetClassDataset(val_dir, particle_features=feats, norm_dict=norm,
                                  normalize=[False, False, False, False], mask_mode=None, seed=0)
    xs, ys, n = [], [], 0
    for X, y in DataLoader(ds, batch_size=512, num_workers=workers):
        xs.append(X); ys.append(y.argmax(1)); n += len(X)
        if n >= max_jets:
            break
    return torch.cat(xs)[:max_jets], torch.cat(ys)[:max_jets].numpy()


@torch.no_grad()
def predict(model, raw, norm, device, bs, fn=None, v=None):
    probs = []
    for i in range(0, len(raw), bs):
        x = raw[i:i + bs].to(device, torch.float64)
        valid = x[..., 3] > 0
        P = to_cart(x[..., :4])
        if fn is not None:
            P = fn(P, v)
        inp = model_input(x, P, valid, norm).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(inp)
        probs.append(torch.softmax(logits.float(), 1).cpu())
    return torch.cat(probs).numpy()


@torch.no_grad()
def acceptance(raw, device, bs, fn=None, v=None):
    """Fraction of jets outside the training acceptance after the transform (model-independent)."""
    eta_out = pt_out = 0
    for i in range(0, len(raw), bs):
        x = raw[i:i + bs].to(device, torch.float64)
        valid = x[..., 3] > 0
        P = to_cart(x[..., :4])
        if fn is not None:
            P = fn(P, v)
        pT, eta, _ = jet_kinematics(P, valid)
        eta_out += (eta.abs() > JET_ETA_MAX).sum().item()
        pt_out += ((pT < JET_PT_RANGE[0]) | (pT > JET_PT_RANGE[1])).sum().item()
    return eta_out / len(raw), pt_out / len(raw)


def scores(y, probs, base_pred):
    from sklearn.metrics import roc_auc_score
    pred = probs.argmax(1)
    auc = (roc_auc_score(y, probs, multi_class="ovo", average="macro")
           if len(np.unique(y)) == probs.shape[1] else float("nan"))   # needs every class present
    return auc, float((pred == y).mean()), float((pred != base_pred).mean())


def main():
    p = argparse.ArgumentParser(description="Robustness of trained models to frame changes")
    p.add_argument("--val-dir", required=True)
    p.add_argument("--logs-dir", default="./logs")
    p.add_argument("--scale", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--tag", default=None)
    p.add_argument("--models", nargs="+", default=["part", "lorentzpart"], choices=["part", "lorentzpart"])
    p.add_argument("--protocols", nargs="+", default=["scratch", "jepa"], choices=["scratch", "jepa"])
    p.add_argument("--features", type=int, default=14, choices=[4, 14])
    p.add_argument("--common-scale", action="store_true", help="MUST match training")
    p.add_argument("--cartesian-mv", action="store_true", help="MUST match training")
    p.add_argument("--bool-padding-mask", action="store_true", help="MUST match training")
    p.add_argument("--max-jets", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", default=None, help="CSV path (default results/frames_<scale><_tag>.csv)")
    args = p.parse_args()

    assert list(EA.T7.NORMALIZE) == [True, False, False, True], \
        "model_input() replicates NORMALIZE=[T,F,F,T]; update it if the training normalisation changed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = EA.T7.FEATURES[args.features]
    n_extra = len(feats) - 4
    norm = EA.T7.NORM_DICT_COMMON if args.common_scale else EA.T7.NORM_DICT
    sfx = f"_{args.tag}" if args.tag else ""
    out = args.out or os.path.join(_HERE, "results", f"frames_{args.scale}{sfx}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    raw, y = load_raw(args.val_dir, feats, norm, args.max_jets, args.num_workers)
    print(f"[frames] {len(raw)} validation jets  common_scale={args.common_scale} "
          f"cartesian_mv={args.cartesian_mv} bool_padding_mask={args.bool_padding_mask}")
    self_check(raw, device)

    settings = [("none", None, 0.0)] + [(fam, fn, v) for fam, (fn, vals, _) in FAMILIES.items()
                                        for v in vals]
    ctx = {(fam, v): acceptance(raw, device, args.batch_size, fn, v) for fam, fn, v in settings}

    rows = []
    for model_name in args.models:
        for proto in args.protocols:
            for seed in args.seeds:
                run = f"{model_name}_{proto}_{args.scale}_seed{seed}{sfx}"
                ckpt = os.path.join(args.logs_dir, EA.MODEL_DIR[model_name], "best", f"{run}.pt")
                if not os.path.exists(ckpt):
                    print(f"{run}: no checkpoint — skipped"); continue
                model = EA.load_model(model_name, ckpt, n_extra, device, args.cartesian_mv,
                                      args.bool_padding_mask)
                base = predict(model, raw, norm, device, args.batch_size)
                base_pred = base.argmax(1)
                for fam, fn, v in settings:
                    probs = base if fn is None else predict(model, raw, norm, device,
                                                            args.batch_size, fn, v)
                    auc, acc, flip = scores(y, probs, base_pred)
                    eta_out, pt_out = ctx[(fam, v)]
                    rows.append(dict(model=model_name, protocol=proto, seed=seed, family=fam, value=v,
                                     auc=auc, acc=acc, flip_rate=flip,
                                     frac_eta_out=eta_out, frac_pt_out=pt_out))
                print(f"[frames] done {run}", flush=True)

    if not rows:
        sys.exit("no checkpoints found — check --scale/--tag/--logs-dir")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    # ---------------- readable summary ----------------
    cells = sorted({(r["model"], r["protocol"]) for r in rows})
    base_auc = {(r["model"], r["protocol"]): r["auc"] for r in rows if r["family"] == "none"}
    print(f"\nAUC (change vs untransformed) and flip rate — {len(raw)} jets\n")
    for fam, (_, vals, label) in [("none", (None, [0.0], "—"))] + list(FAMILIES.items()):
        print(f"== {fam}  ({label})")
        head = f"  {'value':>7} {'|eta|>2':>8} {'pT out':>7}"
        print(head + "".join(f"  {m + '+' + p:>26}" for m, p in cells))
        for v in vals:
            sel = {(r["model"], r["protocol"]): r for r in rows if r["family"] == fam and r["value"] == v}
            eta_out, pt_out = ctx[(fam, v)]
            line = f"  {v:>7.3g} {eta_out:>8.1%} {pt_out:>7.1%}"
            for c in cells:
                r = sel.get(c)
                if r is None:
                    line += f"  {'—':>26}"
                else:
                    d = r["auc"] - base_auc[c]
                    line += f"  {r['auc']:>8.4f} ({d:+.4f}) {r['flip_rate']:>6.1%}"
            print(line)
        print()
    print(f"wrote {out}")
    plot(rows, cells, base_auc, out.replace(".csv", ".png"))


def plot(rows, cells, base_auc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    titles = {"rotation": "ROC AUC under azimuthal rotation",
              "long_boost": "ROC AUC under longitudinal boost",
              "transv_boost": "ROC AUC under transverse boost"}
    for ax, (fam, (_, vals, label)) in zip(axes, FAMILIES.items()):
        for m, p in cells:
            pts = sorted((r["value"], r["auc"]) for r in rows
                         if r["model"] == m and r["protocol"] == p and r["family"] == fam)
            xs = [0.0] + [v for v, _ in pts]; ys = [base_auc[(m, p)]] + [a for _, a in pts]
            ax.plot(xs, ys, marker="o", lw=1.8, ls="-" if p == "scratch" else "--",
                    label=f"{'ParT' if m == 'part' else 'LorentzParT'} ({p})")
        ax.set_title(titles[fam]); ax.set_xlabel(label); ax.grid(alpha=0.3)
    axes[0].set_ylabel("ROC AUC (one-vs-one, macro)")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
