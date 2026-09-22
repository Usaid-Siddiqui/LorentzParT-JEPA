"""
Phase 8 — diagnose the pT/E scale mismatch: does it kill ln_m2 and break the EquiLinear embedding?

The dataset divides pT by its mean (92.73) and E by a DIFFERENT mean (133.87), so energy and
momentum carry different units downstream. Two consequences: (a) m2 = E_sum^2 - |p_sum|^2 is no
longer the pair invariant mass, goes negative for every in-cone pair and clamps to log(1e-8);
(b) the 4 numbers handed to `embed_vector` are not a Lorentz vector, so EquiLinear's equivariance
is never engaged. This measures both, current vs. fixed, WITHOUT modifying any source file — the
fix is applied by monkeypatching ParticleProcessor.forward, so nothing here commits us to it.

The "fix" = put E on the same scale as pT, and hand embed_vector Cartesian (E, px, py, pz)
(i.e. restore the conversion commented out at processor.py:93-98).

    python experiments/phase8_scalefix/diagnose_scaling.py                      # checks 1+2, no data
    python experiments/phase8_scalefix/diagnose_scaling.py --weights <ckpt.pt>  # check 2, trained
    python experiments/phase8_scalefix/diagnose_scaling.py --data-dir <ROOT>    # + check 3, real data

Expected if the diagnosis holds: ln_m2 ~100% clamped now / ~0% fixed; layer equivariance error
~1e-1 now / ~1e-8 fixed.
"""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))          # equivariance_test
sys.path.insert(0, os.path.join(_REPO, "experiments", "phase7_scaling"))   # NORM_DICT

from lgatr.interface import embed_vector
from lgatr.layers import EquiLinear

import equivariance_test as EQ          # reuse its transforms / model_input / to_coords
from src.models.processor import ParticleProcessor

PT_MEAN, E_MEAN = EQ.PT_MEAN, EQ.E_MEAN
E_OVER_PT = E_MEAN / PT_MEAN
LN_EPS = float(np.log(1e-8))


# --------------------------------------------------------------------------- the fix
_ORIG_FORWARD = ParticleProcessor.forward


def _patched_forward(self, x):
    """E rescaled onto the pT scale, and embed_vector fed Cartesian (E, px, py, pz)."""
    x = x.clone()
    x[..., 3] = x[..., 3] * E_OVER_PT                    # common scale -> m2 is a true invariant
    B, N, F = x.shape
    U = self._get_interaction(x)
    if self.to_multivector:
        pT, eta, phi, E = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        cart = torch.stack([E, pT * torch.cos(phi), pT * torch.sin(phi), pT * torch.sinh(eta)], -1)
        x = embed_vector(cart.view(B, N, 1, 4)).view(B, N, 16)
    return x, U


class fixed:
    """`with fixed():` runs the enclosed block against the patched processor."""
    def __enter__(self):
        ParticleProcessor.forward = _patched_forward

    def __exit__(self, *a):
        ParticleProcessor.forward = _ORIG_FORWARD


# ------------------------------------------------------------------ 1. layer equivariance
def check_layer_equivariance(seed=0):
    print("=" * 78)
    print("1. EquiLinear equivariance:  is  f(Lx) == L f(x)  ?   (isolated layer, no model)")
    print("=" * 78)
    torch.manual_seed(seed)
    lin = EquiLinear(in_mv_channels=1, out_mv_channels=1).eval()
    rng = np.random.default_rng(seed)
    n = 128
    pt = rng.pareto(2.0, n) * 8 + 0.5
    eta, phi = rng.normal(0, 0.25, n), rng.uniform(-np.pi, np.pi, n)
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    P = np.stack([np.sqrt(px**2 + py**2 + pz**2), px, py, pz], -1)   # massless jet

    def as_current(P):                                   # (pT/muP, eta, phi, E/muE)
        c = EQ.to_coords(P).copy(); c[..., 0] /= PT_MEAN; c[..., 3] /= E_MEAN
        return c

    def as_cartesian(P, escale):                         # Cartesian, E on the given scale
        c = EQ.to_coords(P)
        q, e, f, en = c[..., 0] / PT_MEAN, c[..., 1], c[..., 2], c[..., 3] / escale
        return np.stack([en, q * np.cos(f), q * np.sin(f), q * np.sinh(e)], -1)

    @torch.no_grad()
    def vec(a):
        mv = embed_vector(torch.from_numpy(np.ascontiguousarray(a)).float().view(1, -1, 1, 4))
        return lin(mv)[0][..., 1:5].squeeze().numpy()

    variants = [("A current  (pT,eta,phi,E)",   as_current),
                ("B uncomment only",            lambda P: as_cartesian(P, E_MEAN)),
                ("C uncomment + common scale",  lambda P: as_cartesian(P, PT_MEAN))]
    print(f"{'variant':32}{'transform':24}{'||f(Lx)-Lf(x)||/||Lf(x)||':>28}")
    for name, vf in variants:
        for tname, tf, arg in [("long. boost y=0.6", EQ.boost_z, 0.6),
                               ("transv. boost b=0.4", EQ.boost_x, 0.4)]:
            lhs, rhs = vec(vf(tf(P, arg))), tf(vec(vf(P)), arg)
            err = np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)
            print(f"{name:32}{tname:24}{err:>28.3e}")
    print("  -> C at ~1e-8 means BOTH changes are needed; uncommenting alone is not enough.\n")


# ------------------------------------------------------- 2. end-to-end model invariance
def check_model_invariance(weights=None, seed=0, batch=16, particles=30):
    print("=" * 78)
    print("2. End-to-end LorentzParT invariance:  max|delta softmax-prob| under a transform")
    print("=" * 78)
    from src.models import LorentzParT

    def run():
        torch.manual_seed(seed); np.random.seed(seed)
        model = LorentzParT(num_classes=10, ragged_pair_embed=True, mask=False)
        if weights:
            sd = torch.load(weights, map_location='cpu', weights_only=True)
            model.load_state_dict(sd, strict=False)
        model.eval()
        p3 = np.random.randn(batch, particles, 3) * 30.0
        P = np.concatenate([np.linalg.norm(p3, axis=-1, keepdims=True), p3], -1)
        with torch.no_grad():
            base = torch.softmax(model(EQ.model_input(P)), 1).numpy()
            return {nm: float(np.abs(torch.softmax(model(EQ.model_input(Pt)), 1).numpy() - base).max())
                    for nm, Pt in [("azimuthal rotation a=0.9", EQ.rot_z(P, 0.9)),
                                   ("longitudinal boost y=0.6", EQ.boost_z(P, 0.6)),
                                   ("transverse boost  b=0.4",  EQ.boost_x(P, 0.4))]}

    cur = run()
    with fixed():
        fix = run()
    print(f"{'transform':30}{'current':>14}{'fixed':>14}")
    for k in cur:
        print(f"{k:30}{cur[k]:>14.2e}{fix[k]:>14.2e}")
    print("  -> rot_z / boost_z are ALREADY near-invariant because dEta, dPhi and pT are")
    print("     invariant under them. The transverse boost is the discriminating one.\n")


# ----------------------------------------------------------- 3. real-data ln_m2 clamp rate
def check_ln_m2_on_data(data_dir, max_jets=4096):
    print("=" * 78)
    print(f"3. ln_m2 on REAL JetClass ({data_dir})")
    print("=" * 78)
    from torch.utils.data import DataLoader
    from src.utils.data.streaming_jetclass import StreamingJetClassDataset, KINEMATIC
    import train_phase7 as T7

    ds = StreamingJetClassDataset(data_dir, particle_features=KINEMATIC, norm_dict=T7.NORM_DICT,
                                  normalize=T7.NORMALIZE, mask_mode=None, seed=0)
    xs, n = [], 0
    for X, _ in DataLoader(ds, batch_size=256):
        xs.append(X); n += len(X)
        if n >= max_jets:
            break
    x = torch.cat(xs)[:max_jets]
    proc = ParticleProcessor(to_multivector=False)

    def clamp_rate(xin):
        U = proc._get_interaction(xin)
        valid = xin[..., 3] > 0
        vp = valid.unsqueeze(2) & valid.unsqueeze(1)
        off = ~torch.eye(xin.shape[1], dtype=torch.bool).unsqueeze(0)
        sel = U[..., 3][vp & off]
        return (sel <= LN_EPS + 1e-4).float().mean().item() * 100, sel.mean().item(), sel.numel()

    cur_r, cur_m, npairs = clamp_rate(x)
    xf = x.clone(); xf[..., 3] *= E_OVER_PT
    fix_r, fix_m, _ = clamp_rate(xf)
    print(f"jets={len(x)}  valid off-diagonal pairs={npairs}")
    print(f"{'':10}{'% ln_m2 clamped to log(1e-8)':>32}{'mean ln_m2':>14}")
    print(f"{'current':10}{cur_r:>31.2f}%{cur_m:>14.3f}")
    print(f"{'fixed':10}{fix_r:>31.2f}%{fix_m:>14.3f}")
    print("  -> a high current rate means the m2 channel is a dead constant in every run.\n")


def main():
    p = argparse.ArgumentParser(description="Phase 8 — diagnose the pT/E scale mismatch")
    p.add_argument('--weights', default=None, help="optional trained LorentzParT checkpoint")
    p.add_argument('--data-dir', default=None, help="JetClass ROOT dir; enables check 3")
    p.add_argument('--max-jets', type=int, default=4096)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    check_layer_equivariance(args.seed)
    check_model_invariance(args.weights, args.seed)
    if args.data_dir:
        check_ln_m2_on_data(args.data_dir, args.max_jets)
    else:
        print("(skipping check 3 — pass --data-dir <JetClass ROOT dir> to run it on real data)")


if __name__ == '__main__':
    main()
