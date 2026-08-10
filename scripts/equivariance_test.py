"""
Architectural Lorentz-invariance test for the LorentzParT classifier.

For a classifier the correct symmetry property is INVARIANCE, not equivariance:
viewing the same jet from a boosted/rotated frame must not change the predicted
class. Invariance is a property of the *architecture*, not the weights, so this runs
on a random-init model (no data, no GPU, no training) — pass --weights to also confirm
on a trained checkpoint.

Method: build massless jet constituents in Cartesian 4-momenta (E, px, py, pz), apply a
Lorentz transform, reconvert to the model's (pT, η, φ, E) input, normalize, and compare
the softmax outputs. A PERMUTATION of the particles is included as a control — the model
*is* permutation-invariant (attention pooling), so it should read ~0 and proves the test
can detect an invariance that is genuinely present.

    max|Δ softmax-prob|  ~1e-6 = invariant (rounding floor);  >>0 = broken.

Finding: LorentzParT is not exactly Lorentz-invariant — by design. It is a HYBRID: a ParT
core "nudged in the right physical direction by a small dose of LGATr's Lorentz structure"
(Nguyen's GSoC-2025 article), which deliberately avoids the "computational overhead" of strict
equivariance. This test measures how far that nudge lands from exact invariance: ~3e-3 on
rotation/boost, vs a full-L-GATr model's ~1e-7 (see experiments/phase5_equivariance/). The
`self.proj = nn.Linear(16, embed_dim)` after EquiLinear is the bridge from the geometric-algebra
embedding to the efficient scalar transformer core — the mechanism of the tradeoff, not a bug.

    python scripts/equivariance_test.py                 # random init
    python scripts/equivariance_test.py --weights <.pt>  # trained checkpoint
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import LorentzParT

# training NORM_DICT means (pT, E scaled by mean; η, φ raw)
PT_MEAN, E_MEAN = 92.72917175292969, 133.8745574951172
N_MAX = 128
THRESH = 1e-4          # above this = invariance broken (rounding floor is ~1e-6)


def parse_args():
    p = argparse.ArgumentParser(description="Lorentz-invariance test for LorentzParT")
    p.add_argument('--weights', default=None, help="Optional trained checkpoint (.pt)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--particles', type=int, default=30)
    return p.parse_args()


def to_coords(P):                       # (E,px,py,pz) -> (pT,eta,phi,E)
    E, px, py, pz = [P[..., i] for i in range(4)]
    pT = np.hypot(px, py)
    eta = np.arcsinh(pz / np.clip(pT, 1e-9, None))
    phi = np.arctan2(py, px)
    return np.stack([pT, eta, phi, E], -1)


def model_input(P):                     # Cartesian -> normalized (pT,eta,phi,E), padded
    x = to_coords(P).astype(np.float32)
    x[..., 0] /= PT_MEAN
    x[..., 3] /= E_MEAN
    pad = np.zeros((x.shape[0], N_MAX - x.shape[1], 4), np.float32)
    return torch.from_numpy(np.concatenate([x, pad], 1))


# Lorentz transforms on Cartesian 4-momenta (E, px, py, pz)
def boost_z(P, y):
    E, px, py, pz = [P[..., i] for i in range(4)]
    ch, sh = np.cosh(y), np.sinh(y)
    return np.stack([E * ch + pz * sh, px, py, pz * ch + E * sh], -1)


def rot_z(P, a):
    E, px, py, pz = [P[..., i] for i in range(4)]
    return np.stack([E, px * np.cos(a) - py * np.sin(a),
                     px * np.sin(a) + py * np.cos(a), pz], -1)


def boost_x(P, b):
    g = 1.0 / np.sqrt(1.0 - b * b)
    E, px, py, pz = [P[..., i] for i in range(4)]
    return np.stack([g * (E - b * px), g * (px - b * E), py, pz], -1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    B, M = args.batch, args.particles

    model = LorentzParT(num_classes=10, ragged_pair_embed=True, mask=False)
    if args.weights:
        sd = torch.load(args.weights, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {args.weights}  (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()

    # massless constituents: random 3-momenta, E = |p|
    p3 = np.random.randn(B, M, 3) * 30.0
    P = np.concatenate([np.linalg.norm(p3, axis=-1, keepdims=True), p3], -1)

    @torch.no_grad()
    def probs(Pin):
        return torch.softmax(model(model_input(Pin)), dim=1).numpy()

    base = probs(P)
    print(f"baseline softmax spread: min={base.min():.3f} max={base.max():.3f}  "
          f"(random init ~0.1 = uniform)\n")

    perm = np.stack([np.random.permutation(M) for _ in range(B)])
    P_perm = np.take_along_axis(P, perm[:, :, None], axis=1)

    cases = [
        ("PERMUTATION (control)",     P_perm),
        ("azimuthal rotation a=0.9",  rot_z(P, 0.9)),
        ("longitudinal boost y=0.6",  boost_z(P, 0.6)),
        ("transverse boost   b=0.4",  boost_x(P, 0.4)),
    ]
    for name, Pt in cases:
        d = float(np.abs(probs(Pt) - base).max())
        print(f"  {name:28s} max|Δprob| = {d:.2e}   "
              f"[{'INVARIANT' if d < THRESH else 'BROKEN'}]")


if __name__ == '__main__':
    main()
