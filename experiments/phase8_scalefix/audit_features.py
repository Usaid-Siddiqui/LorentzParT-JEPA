"""
Phase 8 — feature liveness audit: does every engineered feature actually CARRY INFORMATION?

Every bug we have found in this codebase has the same signature: a physics quantity is computed,
then silently collapsed to a constant by a clamp / relu / padding-fill / degenerate norm. Nothing
crashes, no NaN fires, the loss goes down, accuracy looks plausible. Known instances:

  * Phase 3: padded pairs filled with -1e9 BEFORE BatchNorm  -> every valid feature crushed to ~2.795
  * Phase 8: pT/E scaled by different means -> ln_m2 clamped to log(1e-8) for 99.99% of pairs
  * (peer repo) global m2 via LayerNorm(1) -> identically 0 for every jet

Reading code did not catch any of them; checking VARIANCE does. This asserts that each input
channel, each interaction feature, each gate output and each multivector slot still varies on real
data. A channel with ~zero variance, or one pinned to a single value, is dead weight or a bug.

    python experiments/phase8_scalefix/audit_features.py --data-dir <JetClass ROOT dir>
    python experiments/phase8_scalefix/audit_features.py --data-dir <dir> --common-scale

Exit code 1 if anything is flagged DEAD, so it can gate CI or a pre-run check.
"""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "experiments", "phase7_scaling"))

from src.models.processor import ParticleProcessor
from src.models.attention_gate import AttentionGate
from src.utils.data.streaming_jetclass import StreamingJetClassDataset

U_NAMES = ["ln_delta", "ln_kT", "ln_z", "ln_m2"]
DEAD_STD = 1e-6          # below this = constant
PINNED_FRAC = 0.90       # this fraction at one value = effectively constant


MIN_MINORITY = 30        # a rare BINARY flag is fine if the minority class is this well populated


def stats(v: torch.Tensor):
    """(std, fraction at the most common value, min, max, n_unique, minority_count)."""
    v = v.flatten().float()
    if v.numel() == 0:
        return 0.0, 1.0, 0.0, 0.0, 0, 0
    # round so float noise does not hide a pinned value
    vals, counts = torch.unique((v * 1e4).round(), return_counts=True)
    top = counts.max().item()
    return (v.std().item(), top / v.numel(), v.min().item(), v.max().item(),
            len(vals), int(v.numel() - top))


def verdict(std, pinned, n_unique=None, minority=None):
    """Is this channel information-free?

    A channel pinned to one value is dead — EXCEPT a genuinely rare binary flag. Jet
    constituents are ~99% not-an-electron and ~99.4% not-a-muon, so part_isElectron /
    part_isMuon sit at one value almost always yet still carry real signal. For a binary
    channel judge the ABSOLUTE size of the minority class, not its fraction.
    """
    if std < DEAD_STD:
        return "DEAD (constant)"
    if n_unique is not None and n_unique <= 2:
        return "ok (rare flag)" if minority >= MIN_MINORITY else f"DEAD (only {minority} minority)"
    if pinned > PINNED_FRAC:
        return f"DEAD ({pinned*100:.1f}% one value)"
    return "ok"


def main():
    p = argparse.ArgumentParser(description="Phase 8 feature liveness audit")
    p.add_argument('--data-dir', required=True)
    p.add_argument('--features', type=int, default=14, choices=[4, 14])
    p.add_argument('--common-scale', action='store_true')
    p.add_argument('--cartesian-mv', action='store_true')
    p.add_argument('--jets', type=int, default=2048)
    args = p.parse_args()

    import train_phase7 as T7
    feats = T7.FEATURES[args.features]
    nd = T7.NORM_DICT_COMMON if args.common_scale else T7.NORM_DICT

    ds = StreamingJetClassDataset(args.data_dir, particle_features=feats, norm_dict=nd,
                                  normalize=T7.NORMALIZE, mask_mode=None, seed=0)
    xs, n = [], 0
    for X, _ in torch.utils.data.DataLoader(ds, batch_size=256):
        xs.append(X); n += len(X)
        if n >= args.jets:
            break
    x = torch.cat(xs)[:args.jets]
    valid = x[..., 3] > 0
    print(f"jets={len(x)}  particles/jet(mean)={valid.float().sum(1).mean():.1f}  "
          f"common_scale={args.common_scale} cartesian_mv={args.cartesian_mv}\n")

    bad = []
    hdr = f"{'tensor':26}{'std':>12}{'pinned':>10}{'min':>12}{'max':>12}  verdict"

    # ---- 1. input channels (valid particles only) ----
    print("1. INPUT CHANNELS"); print(hdr)
    for i, name in enumerate(feats):
        s, pin, lo, hi, nu, mino = stats(x[..., i][valid])
        v = verdict(s, pin, nu, mino); bad += [f"input:{name}"] if v.startswith("DEAD") else []
        print(f"{name:26}{s:>12.4f}{pin*100:>9.1f}%{lo:>12.3f}{hi:>12.3f}  {v}")

    # ---- 2. interaction features (valid off-diagonal pairs only) ----
    proc = ParticleProcessor(to_multivector=True, cartesian_mv=args.cartesian_mv)
    with torch.no_grad():
        mv, U = proc(x[..., :4])
    vp = valid.unsqueeze(2) & valid.unsqueeze(1)
    vp &= ~torch.eye(x.shape[1], dtype=torch.bool).unsqueeze(0)
    print("\n2. INTERACTION FEATURES U"); print(hdr)
    for i, name in enumerate(U_NAMES):
        s, pin, lo, hi, nu, mino = stats(U[..., i][vp])
        v = verdict(s, pin, nu, mino); bad += [f"U:{name}"] if v.startswith("DEAD") else []
        print(f"{name:26}{s:>12.4f}{pin*100:>9.1f}%{lo:>12.3f}{hi:>12.3f}  {v}")

    # ---- 3. multivector slots ----
    print("\n3. MULTIVECTOR SLOTS (nonzero only)"); print(hdr)
    for i in range(mv.shape[-1]):
        col = mv[..., i][valid]
        if col.abs().max() == 0:
            continue
        s, pin, lo, hi, nu, mino = stats(col)
        v = verdict(s, pin, nu, mino); bad += [f"mv:slot{i}"] if v.startswith("DEAD") else []
        print(f"{'slot '+str(i):26}{s:>12.4f}{pin*100:>9.1f}%{lo:>12.3f}{hi:>12.3f}  {v}")

    # ---- 4. attention gate output ----
    gate = AttentionGate().eval()
    with torch.no_grad():
        g = gate(U, valid)
    s, pin, lo, hi, nu, mino = stats(g[valid])
    v = verdict(s, pin, nu, mino); bad += ["gate"] if v.startswith("DEAD") else []
    print("\n4. ATTENTION GATE (random init)"); print(hdr)
    print(f"{'gate':26}{s:>12.4f}{pin*100:>9.1f}%{lo:>12.3f}{hi:>12.3f}  {v}")

    print("\n" + "=" * 78)
    if bad:
        print(f"FLAGGED {len(bad)}: " + ", ".join(bad))
        sys.exit(1)
    print("All channels carry variance.")


if __name__ == '__main__':
    main()
