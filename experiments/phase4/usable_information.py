"""
Phase 4 ceiling probe (1/2) — usable information from finetune cross-entropy.

For discrete labels, I(X;Y) = H(Y) - H(Y|X), and any classifier's test cross-entropy
upper-bounds H(Y|X). So I(X;Y) >= H(Y) - CE_model (bits) — a LOWER bound on the label
information the input carries, tightening as the model improves. Two readings:
  - absolute: "the kinematics carry >= N bits about this label" (objective, per task).
  - ties: if scratch / jepa / mae have equal CE (not just equal AUC — CE is calibrated),
    all methods extract the same information content -> evidence of a shared ceiling.

Runs against the finetune checkpoints saved by the Phase-4 sweep. No retraining.

    python experiments/phase4/usable_information.py --data-dir ./data_1m \\
        --ckpt-dir logs/LorentzParT/best --seeds 42 123 456
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import tasks
import protocols

NORM_DICT = {
    'pT':     (92.72917175292969,    105.83937072753906),
    'eta':    (0.0005733045982196927, 0.9174848794937134),
    'phi':    (-0.00041169871110469103, 1.8136887550354004),
    'energy': (133.8745574951172,    167.528564453125),
}
NORMALIZE = [True, False, False, True]
ENCODERS = ['scratch', 'jepa', 'mae']


def make_test_loader(task, data_dir, batch, workers):
    cfg = tasks.TASKS[task]
    ds = tasks.TaskDataset(
        os.path.join(data_dir, 'test', 'particles.npy'),
        os.path.join(data_dir, 'test', 'labels.npy'),
        cfg['classes'], normalize=NORMALIZE, norm_dict=NORM_DICT,
        max_samples=None, subsample_seed=0)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers), len(cfg['classes'])


@torch.no_grad()
def ce_bits(model, loader, device):
    """Mean test cross-entropy in bits (natural log / ln2)."""
    model.eval()
    tot, n = 0.0, 0
    for X, y in loader:
        logp = torch.log_softmax(model(X.to(device)), dim=1)
        yt = y.argmax(1).to(device)
        tot += (-logp.gather(1, yt[:, None]).squeeze(1)).sum().item()
        n += len(yt)
    return (tot / n) / np.log(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./data_1m')
    p.add_argument('--ckpt-dir', default='logs/LorentzParT/best')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--batch-size', type=int, default=1000)
    p.add_argument('--num-workers', type=int, default=2)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'task':11}{'H(Y)':>7}   " + "".join(f"{e+' CE':>16}" for e in ENCODERS)
          + "     I_min (bits)")
    print("-" * 78)
    for task in tasks.TASKS:
        loader, k = make_test_loader(task, args.data_dir, args.batch_size, args.num_workers)
        HY = float(np.log2(k))
        ce = {}
        for enc in ENCODERS:
            vals = []
            for s in args.seeds:
                ckpt = os.path.join(args.ckpt_dir, f'{task}_finetune_{enc}_seed{s}.pt')
                if not os.path.exists(ckpt):
                    print(f"  [miss] {os.path.basename(ckpt)}", flush=True)
                    continue
                model = protocols.build_model('finetune', k, encoder_weights=None)
                model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
                model.to(device)
                vals.append(ce_bits(model, loader, device))
            ce[enc] = vals
        row = f"{task:11}{HY:>7.3f}   "
        for enc in ENCODERS:
            v = ce[enc]
            row += f"{(f'{np.mean(v):.3f}±{np.std(v):.3f}' if v else '—'):>16}"
        # I_min = H(Y) - min CE over encoders (tightest bound); also flag CE spread
        allmeans = [np.mean(ce[e]) for e in ENCODERS if ce[e]]
        if allmeans:
            imin = HY - min(allmeans)
            spread = max(allmeans) - min(allmeans)
            row += f"     {imin:.3f}   (CE spread {spread:.3f})"
        print(row, flush=True)

    print("\nI_min = H(Y) - min CE = tightest lower bound on I(X;Y) in bits.")
    print("Small CE spread across encoders => all methods extract the same information "
          "=> evidence of a shared (input-set) ceiling.")


if __name__ == '__main__':
    main()
