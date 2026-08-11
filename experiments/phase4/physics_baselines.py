"""
Phase 4 ceiling probe (2/2) — deep model vs. known physics observables (binary tasks).

If the deep model's AUC is no better than the AUC from simple, physically-motivated
observables computed straight from the 4-vectors, then the model has extracted all the
known-physics information and the ceiling IS the overlap of those observables — a
physicist-legible ceiling. If the model clearly beats them, it is finding extra structure.

Observables per jet (from raw, un-normalized 4-vectors): invariant mass, jet pT, energy,
constituent multiplicity, girth (pT-weighted ΔR spread), and leading-pT fraction. For each:
single-observable ROC AUC (direction-agnostic) + a logistic combination of all of them
(fit on half, scored on the other half). Compared to the finetune model AUC from the sweep.

    python experiments/phase4/physics_baselines.py --data-dir ./data_1m \\
        --results-dir experiments/phase4/results
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
import tasks


def observables(P):
    """P: (M, 4, 128) raw (pT, eta, phi, E) per particle -> dict of jet-level observables."""
    x = np.transpose(P, (0, 2, 1)).astype(np.float64)       # (M, 128, 4)
    pT, eta, phi, E = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    valid = E > 0
    px = pT * np.cos(phi); py = pT * np.sin(phi); pz = pT * np.sinh(eta)
    for a in (px, py, pz, E, pT):
        a[~valid] = 0.0

    Et, pxt, pyt, pzt = E.sum(1), px.sum(1), py.sum(1), pz.sum(1)
    mass = np.sqrt(np.clip(Et**2 - (pxt**2 + pyt**2 + pzt**2), 0, None))
    jpt = np.hypot(pxt, pyt)
    mult = valid.sum(1).astype(np.float64)

    eta_j = np.arcsinh(pzt / np.clip(jpt, 1e-9, None))
    phi_j = np.arctan2(pyt, pxt)
    dphi = (phi - phi_j[:, None] + np.pi) % (2 * np.pi) - np.pi
    deta = eta - eta_j[:, None]
    dR = np.sqrt(deta**2 + dphi**2); dR[~valid] = 0.0
    sumpt = pT.sum(1) + 1e-9
    girth = (pT * dR).sum(1) / sumpt
    lead = pT.max(1) / sumpt
    return {'mass': mass, 'jet_pT': jpt, 'energy': Et,
            'multiplicity': mult, 'girth': girth, 'lead_pT_frac': lead}


def auc_dir(y, s):
    a = roc_auc_score(y, s)
    return max(a, 1 - a)          # direction-agnostic discriminating power


def model_finetune_auc(results_dir, task):
    """Mean finetune test AUC across encoders/seeds (they tie) — the deep-model reference."""
    vals = []
    for f in glob.glob(os.path.join(results_dir, f'{task}_finetune_*_seed*.json')):
        vals.append(json.load(open(f)).get('test_auc'))
    vals = [v for v in vals if v is not None]
    return (np.mean(vals), len(vals)) if vals else (np.nan, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./data_1m')
    p.add_argument('--results-dir', default='experiments/phase4/results')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    particles = np.load(os.path.join(args.data_dir, 'test', 'particles.npy'), mmap_mode='r')
    lab10 = np.load(os.path.join(args.data_dir, 'test', 'labels.npy')).argmax(1)

    binary = [t for t, c in tasks.TASKS.items() if c['kind'] == 'binary']
    for task in binary:
        cfg = tasks.TASKS[task]
        keep = np.where(np.isin(lab10, cfg['classes']))[0]
        P = np.ascontiguousarray(particles[keep])
        y = (lab10[keep] == cfg['signal']).astype(int)
        obs = observables(P)

        print(f"\n{'='*60}\n{task}  (signal = class {cfg['signal']}, N={len(y)})\n{'='*60}")
        print(f"{'observable':16}{'AUC':>8}")
        feats = []
        for name, v in obs.items():
            feats.append(v)
            print(f"{name:16}{auc_dir(y, v):>8.4f}")

        # logistic combination of all observables (fit on half, score on the other half)
        Xf = StandardScaler().fit_transform(np.stack(feats, 1))
        Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.5, random_state=args.seed,
                                              stratify=y)
        lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        comb = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
        m_auc, n = model_finetune_auc(args.results_dir, task)
        print(f"{'-'*24}")
        print(f"{'ALL observables':16}{comb:>8.4f}   (logistic, held-out)")
        print(f"{'deep model (ft)':16}{m_auc:>8.4f}   (n={n} finetune cells)")
        gap = m_auc - comb
        print(f"{'model - obs':16}{gap:>+8.4f}   "
              f"{'→ near observable ceiling' if abs(gap) < 0.02 else '→ model finds extra structure'}")


if __name__ == '__main__':
    main()
