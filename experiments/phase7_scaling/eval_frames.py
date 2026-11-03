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
acceptance, so robustness and out-of-distribution are kept apart. The INSIDE-ACCEPTANCE flip rate and
AUC change repeat this on only the jets inside |eta_jet| <= 2 and 500-1000 GeV both before and after
the transform, so a model is not penalised for jets the transform moved outside the training region.

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
    """Per jet, whether it is inside the training |eta| and pT acceptance after the transform
    (model-independent). Returns two bool arrays (eta_ok, pt_ok)."""
    eta_ok, pt_ok = [], []
    for i in range(0, len(raw), bs):
        x = raw[i:i + bs].to(device, torch.float64)
        valid = x[..., 3] > 0
        P = to_cart(x[..., :4])
        if fn is not None:
            P = fn(P, v)
        pT, eta, _ = jet_kinematics(P, valid)
        eta_ok.append((eta.abs() <= JET_ETA_MAX).cpu())
        pt_ok.append(((pT >= JET_PT_RANGE[0]) & (pT <= JET_PT_RANGE[1])).cpu())
    return torch.cat(eta_ok).numpy(), torch.cat(pt_ok).numpy()


def scores(y, probs, base_pred):
    from sklearn.metrics import roc_auc_score
    pred = probs.argmax(1)
    auc = (roc_auc_score(y, probs, multi_class="ovo", average="macro")
           if len(np.unique(y)) == probs.shape[1] else float("nan"))   # needs every class present
    return auc, float((pred == y).mean()), float((pred != base_pred).mean())


def main():
    p = argparse.ArgumentParser(description="Robustness of trained models to frame changes")
    p.add_argument("--val-dir", default=None)
    p.add_argument("--logs-dir", default="./logs")
    p.add_argument("--scale", default=None)
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
    p.add_argument("--replot", default=None,
                   help="redraw the figure from an existing results CSV, without evaluating any model")
    args = p.parse_args()

    if args.replot:
        agg = aggregate(load_rows(args.replot))
        print_summary(agg)
        plot(agg, args.replot.replace(".csv", ".png"))
        return
    if not (args.val_dir and args.scale):
        p.error("--val-dir and --scale are required unless --replot is given")

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
    # inside[(fam, v)]: jets inside the acceptance both before and after the transform, so the
    # "inside acceptance" metrics compare the same jets with and without the transform
    ok = {(fam, v): acceptance(raw, device, args.batch_size, fn, v) for fam, fn, v in settings}
    ok0 = ok[("none", 0.0)][0] & ok[("none", 0.0)][1]
    inside = {k: ok0 & e & p for k, (e, p) in ok.items()}

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
                    eta_ok, pt_ok = ok[(fam, v)]
                    m = inside[(fam, v)]
                    auc_in, _, flip_in = scores(y[m], probs[m], base_pred[m])
                    auc_in0 = scores(y[m], base[m], base_pred[m])[0]
                    rows.append(dict(model=model_name, protocol=proto, seed=seed, family=fam, value=v,
                                     auc=auc, acc=acc, flip_rate=flip,
                                     frac_eta_out=float((~eta_ok).mean()),
                                     frac_pt_out=float((~pt_ok).mean()),
                                     frac_inside=float(m.mean()), drop_inside=auc_in - auc_in0,
                                     flip_inside=flip_in))
                print(f"[frames] done {run}", flush=True)

    if not rows:
        sys.exit("no checkpoints found — check --scale/--tag/--logs-dir")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    agg = aggregate(rows)
    print(f"\n{len(raw)} jets")
    print_summary(agg)
    print(f"wrote {out}")
    plot(agg, out.replace(".csv", ".png"))


def aggregate(rows):
    """One entry per (model, protocol, family, value), averaged over seeds. The AUC drop is computed
    within each seed against that seed's own untransformed AUC, then averaged."""
    base = {(r["model"], r["protocol"], r["seed"]): r["auc"] for r in rows if r["family"] == "none"}
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["protocol"], r["family"], r["value"]), []).append(r)
    agg = {}
    for key, rs in groups.items():
        drops = [r["auc"] - base[(r["model"], r["protocol"], r["seed"])] for r in rs]
        flips = [r["flip_rate"] for r in rs]
        aucs = [r["auc"] for r in rs]
        agg[key] = dict(n=len(rs), auc=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                        drop=float(np.mean(drops)), drop_std=float(np.std(drops)),
                        flip=float(np.mean(flips)), flip_std=float(np.std(flips)),
                        eta_out=rs[0]["frac_eta_out"], pt_out=rs[0]["frac_pt_out"])
        if "flip_inside" in rs[0]:                  # absent in CSVs written before this metric
            fi = [r["flip_inside"] for r in rs]
            di = [r["drop_inside"] for r in rs]
            agg[key].update(inside=rs[0]["frac_inside"],
                            flip_in=float(np.mean(fi)), flip_in_std=float(np.std(fi)),
                            drop_in=float(np.mean(di)), drop_in_std=float(np.std(di)))
    return agg


def print_summary(agg):
    cells = sorted({k[:2] for k in agg})
    seeds = {c: agg[(*c, "none", 0.0)]["n"] for c in cells}
    print("seeds per cell: " + ", ".join(f"{m}+{p}={seeds[(m, p)]}" for m, p in cells))
    print("each cell: AUC (change vs untransformed) flip rate; +/- = std over seeds\n")
    for fam, (_, vals, label) in [("none", (None, [0.0], "—"))] + list(FAMILIES.items()):
        print(f"== {fam}  ({label})")
        print(f"  {'value':>7} {'|eta|>2':>8} {'pT out':>7}" + "".join(f"  {m + '+' + p:>36}" for m, p in cells))
        for v in vals:
            first = next(agg[(*c, fam, v)] for c in cells if (*c, fam, v) in agg)
            line = f"  {v:>7.3g} {first['eta_out']:>8.1%} {first['pt_out']:>7.1%}"
            for c in cells:
                e = agg.get((*c, fam, v))
                if e is None:
                    line += f"  {'—':>36}"
                else:
                    text = (f"{e['auc']:.4f} ({e['drop']:+.4f}±{e['drop_std']:.4f}) "
                            f"{100 * e['flip']:.1f}±{100 * e['flip_std']:.1f}%")
                    line += f"  {text:>36}"
            print(line)
        print()

    if not all("flip_in" in e for e in agg.values()):
        return
    print("INSIDE ACCEPTANCE: only jets inside |eta_jet| <= 2 and 500-1000 GeV both before and after "
          "the transform\neach cell: AUC change vs the same jets untransformed, flip rate\n")
    for fam, (_, vals, label) in FAMILIES.items():
        print(f"== {fam}  ({label})")
        print(f"  {'value':>7} {'inside':>8}" + "".join(f"  {m + '+' + p:>27}" for m, p in cells))
        for v in vals:
            first = next(agg[(*c, fam, v)] for c in cells if (*c, fam, v) in agg)
            line = f"  {v:>7.3g} {first['inside']:>8.1%}"
            for c in cells:
                e = agg.get((*c, fam, v))
                text = "—" if e is None else (f"{e['drop_in']:+.4f}±{e['drop_in_std']:.4f} "
                                              f"{100 * e['flip_in']:.1f}±{100 * e['flip_in_std']:.1f}%")
                line += f"  {text:>27}"
            print(line)
        print()


def load_rows(path):
    num = {"seed": int, "value": float, "auc": float, "acc": float, "flip_rate": float,
           "frac_eta_out": float, "frac_pt_out": float, "frac_inside": float,
           "drop_inside": float, "flip_inside": float}
    with open(path, newline="") as f:
        return [{k: num.get(k, str)(v) for k, v in r.items()} for r in csv.DictReader(f)]


# same colours as plot_scaling.py: ParT orange, LorentzParT blue; JEPA lighter, dashed, squares
STYLE = {("part", "scratch"): ("#ff7f0e", "-", "o"), ("part", "jepa"): ("#ffbb78", "--", "s"),
         ("lorentzpart", "scratch"): ("#1f77b4", "-", "o"), ("lorentzpart", "jepa"): ("#7fbfff", "--", "s")}
NAMES = {"rotation": "azimuthal rotation", "long_boost": "longitudinal boost",
         "transv_boost": "transverse boost"}


def plot(agg, path):
    cells = sorted({k[:2] for k in agg})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    has_in = all("flip_in" in e for e in agg.values())
    rows = 3 if has_in else 2
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.1 * rows), sharex="col", sharey="row")
    for j, (fam, (_, vals, label)) in enumerate(FAMILIES.items()):
        for m, p in cells:
            base = agg[(m, p, "none", 0.0)]
            pts = sorted((k[3], e) for k, e in agg.items() if k[:3] == (m, p, fam))
            xs = [0.0] + [v for v, _ in pts]
            color, ls, marker = STYLE.get((m, p), (None, "-", "o"))
            kw = dict(color=color, ls=ls, marker=marker, lw=1.8, capsize=3,
                      label=f"{'ParT' if m == 'part' else 'LorentzParT'} ({p})")
            axes[0, j].errorbar(xs, [base["auc"]] + [e["auc"] for _, e in pts],
                                yerr=[base["auc_std"]] + [e["auc_std"] for _, e in pts], **kw)
            axes[1, j].errorbar(xs, [0.0] + [100 * e["flip"] for _, e in pts],
                                yerr=[0.0] + [100 * e["flip_std"] for _, e in pts], **kw)
            if has_in:
                axes[2, j].errorbar(xs, [0.0] + [100 * e["flip_in"] for _, e in pts],
                                    yerr=[0.0] + [100 * e["flip_in_std"] for _, e in pts], **kw)
        axes[0, j].set_title(f"ROC AUC under {NAMES[fam]}")
        axes[1, j].set_title(f"Predictions changed under {NAMES[fam]}")
        if has_in:
            axes[2, j].set_title("Predictions changed, jets inside acceptance")
        axes[-1, j].set_xlabel(label)
        for ax in axes[:, j]:
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("ROC AUC (one-vs-one, macro)")
    axes[1, 0].set_ylabel("jets whose predicted class changed (%)")
    if has_in:
        axes[2, 0].set_ylabel("changed, inside |η|≤2, 500–1000 GeV (%)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
