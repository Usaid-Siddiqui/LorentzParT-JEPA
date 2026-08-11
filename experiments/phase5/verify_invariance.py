"""
Phase 5 (equivariance) — M1: does a *properly* invariant architecture pass the test?

Prototype of the fixed encoder that keeps the L-GATr backbone but removes the two
things that break invariance in LorentzParT/LorentzGATr:
  1. NO dense Linear(16, embed) over raw multivector components. Instead read out
     Lorentz INVARIANTS: extract_scalar (grade-0) of the output mv channels + the
     scalar-channel outputs. A Linear on invariants is still invariant.
  2. embed_vector is fed the CARTESIAN 4-momentum (E, px, py, pz), not (pT,η,φ,E).
  3. A single global scale for pT and E (Lorentz-covariant rescaling) instead of the
     per-feature normalization (pT/92.7, E/133.9), which itself distorts the 4-vector
     and would break invariance regardless of architecture.

Runs the same invariance test as scripts/equivariance_test.py. Expectation: all Lorentz
transforms drop to the ~1e-6 rounding floor (vs ~3e-3 for the broken model), confirming
the fix. Random init, no data, no GPU.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lgatr import LGATr
from lgatr.interface import embed_vector, extract_scalar

SCALE = 100.0          # single global scale (covariant); keeps magnitudes O(1)
THRESH = 1e-4


def to_cartesian(x):   # (...,4) pT,eta,phi,E -> (...,4) E,px,py,pz
    pT, eta, phi, E = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    return torch.stack([E, pT*torch.cos(phi), pT*torch.sin(phi), pT*torch.sinh(eta)], -1)


class InvariantGATr(nn.Module):
    """L-GATr backbone + invariant readout (extract_scalar + scalar channels)."""
    def __init__(self, num_classes=10, num_blocks=4, hidden_mv=8, hidden_s=16,
                 out_mv=16, out_s=32, embed_dim=128):
        super().__init__()
        self.gatr = LGATr(num_blocks=num_blocks, in_mv_channels=1, out_mv_channels=out_mv,
                          hidden_mv_channels=hidden_mv, in_s_channels=None, out_s_channels=out_s,
                          hidden_s_channels=hidden_s, attention={}, mlp={})
        self.proj = nn.Linear(out_mv + out_s, embed_dim)   # linear on INVARIANTS = invariant
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):                      # x: (B,N,4) = (pT,eta,phi,E), globally scaled
        p = to_cartesian(x).unsqueeze(-2)      # (B,N,1,4) Cartesian 4-momentum
        mv = embed_vector(p)                   # (B,N,1,16)
        mv_out, s_out = self.gatr(mv)          # (B,N,out_mv,16), (B,N,out_s)
        inv = torch.cat([extract_scalar(mv_out).squeeze(-1), s_out], -1)  # (B,N,out_mv+out_s)
        feat = self.proj(inv)                  # (B,N,embed_dim) — invariant per particle
        return self.head(feat.mean(1))         # mean-pool (perm-inv) -> (B,num_classes)


def boost_z(P, y):
    E, px, py, pz = [P[..., i] for i in range(4)]
    ch, sh = np.cosh(y), np.sinh(y)
    return np.stack([E*ch + pz*sh, px, py, pz*ch + E*sh], -1)

def rot_z(P, a):
    E, px, py, pz = [P[..., i] for i in range(4)]
    return np.stack([E, px*np.cos(a) - py*np.sin(a), px*np.sin(a) + py*np.cos(a), pz], -1)

def boost_x(P, b):
    g = 1/np.sqrt(1 - b*b); E, px, py, pz = [P[..., i] for i in range(4)]
    return np.stack([g*(E - b*px), g*(px - b*E), py, pz], -1)


def coords(P):                                 # Cartesian -> (pT,eta,phi,E), globally scaled
    E, px, py, pz = [P[..., i] for i in range(4)]
    pT = np.hypot(px, py)
    x = np.stack([pT, np.arcsinh(pz/np.clip(pT, 1e-9, None)), np.arctan2(py, px), E], -1)
    x = x.astype(np.float32); x[..., 0] /= SCALE; x[..., 3] /= SCALE
    return torch.from_numpy(x)


def main():
    torch.manual_seed(0); np.random.seed(0)
    B, M = 16, 30
    p3 = np.random.randn(B, M, 3) * 30.0
    P = np.concatenate([np.linalg.norm(p3, axis=-1, keepdims=True), p3], -1)

    model = InvariantGATr().eval()

    @torch.no_grad()
    def probs(Pin):
        return torch.softmax(model(coords(Pin)), dim=1).numpy()

    base = probs(P)
    perm = np.stack([np.random.permutation(M) for _ in range(B)])
    P_perm = np.take_along_axis(P, perm[:, :, None], axis=1)

    print("InvariantGATr (L-GATr backbone + invariant readout):\n")
    for name, Pt in [("PERMUTATION (control)", P_perm),
                     ("azimuthal rotation a=0.9", rot_z(P, 0.9)),
                     ("longitudinal boost y=0.6", boost_z(P, 0.6)),
                     ("transverse boost   b=0.4", boost_x(P, 0.4))]:
        d = float(np.abs(probs(Pt) - base).max())
        print(f"  {name:28s} max|Δprob| = {d:.2e}   [{'INVARIANT' if d < THRESH else 'BROKEN'}]")


if __name__ == '__main__':
    main()
