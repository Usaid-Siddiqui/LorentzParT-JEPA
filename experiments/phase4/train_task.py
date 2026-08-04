"""
Phase 4 — train ONE (task, protocol, encoder, seed) cell and write its metrics JSON.

The single k-class-aware entry point. `run_phase4.py` shells out to this per cell.
  - finetune : full LorentzParT via JetClassTrainer (same recipe as train_lorentz_part.py),
               then evaluate the best checkpoint on the test set.
  - linear / pool : frozen encoder + trainable head, simple best-val head-training loop.

All datasets are TaskDataset (relabel the 10-class arrays, train capped to --max-train,
balanced, deterministic subset). Metrics come from metrics.classification_metrics
(binary → AUC + 1/ε_B@0.5; multiclass → OVO AUC + per-class).

    python experiments/phase4/train_task.py --task top_chan --protocol pool \\
        --data-dir ./data_1m --weights ./logs/ParticleJEPA/best/jepa_1m_ragged_seed42_best.pt \\
        --run-name top_chan_pool_jepa_seed42 --seed 42 --output-dir experiments/phase4/results
"""

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.configs import TrainConfig
from src.engine import JetClassTrainer
from src.utils import accuracy_metric_ce, set_seed, setup_ddp, cleanup_ddp

import tasks
import protocols
import metrics as m4

warnings.filterwarnings('ignore')

NORM_DICT = {
    'pT':     (92.72917175292969,    105.83937072753906),
    'eta':    (0.0005733045982196927, 0.9174848794937134),
    'phi':    (-0.00041169871110469103, 1.8136887550354004),
    'energy': (133.8745574951172,    167.528564453125),
}
NORMALIZE = [True, False, False, True]

# fixed subset seed per task (independent of the training seed) so every
# encoder/protocol/seed on a task sees the identical training jets
SUBSAMPLE_SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 — train one task/protocol/encoder cell")
    p.add_argument('--task', required=True, choices=list(tasks.TASKS))
    p.add_argument('--protocol', required=True, choices=list(protocols.PROTOCOLS))
    p.add_argument('--data-dir', default='./data_1m')
    p.add_argument('--weights', default=None, help='Encoder checkpoint; omit for scratch/random')
    p.add_argument('--encoder-label', default=None,
                   help="Clean encoder name for aggregation (jepa/mae/scratch); inferred if omitted")
    p.add_argument('--run-name', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', default='experiments/phase4/results')
    p.add_argument('--max-train', type=int, default=100_000, help='Cap on training jets (balanced)')
    p.add_argument('--finetune-config', default='configs/train_lorentz_part_ragged.yaml')
    p.add_argument('--probe-epochs', type=int, default=20)
    p.add_argument('--probe-lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=1000)
    p.add_argument('--num-workers', type=int, default=4)
    return p.parse_args()


def make_dataset(split, data_dir, classes, max_samples):
    return tasks.TaskDataset(
        os.path.join(data_dir, split, 'particles.npy'),
        os.path.join(data_dir, split, 'labels.npy'),
        classes, normalize=NORMALIZE, norm_dict=NORM_DICT,
        max_samples=max_samples, subsample_seed=SUBSAMPLE_SEED,
    )


@torch.no_grad()
def eval_on_test(model, loader, device, kind, signal_idx):
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        logits = model(X.to(device))
        preds.append(torch.softmax(logits, dim=1).cpu().numpy())
        trues.append(y.numpy())
    return m4.classification_metrics(
        np.concatenate(trues), np.concatenate(preds), kind, signal_idx)


@torch.no_grad()
def _val_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for X, y in loader:
        pred = model(X.to(device)).argmax(1).cpu()
        correct += (pred == y.argmax(1)).sum().item()
        total += len(y)
    return correct / total


def train_probe(model, train_loader, val_loader, device, epochs, lr):
    """Frozen encoder + head loop, keep best-val head state."""
    opt = torch.optim.AdamW(model.head_parameters(), lr=lr, weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.CrossEntropyLoss()
    best_val, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        model.encoder.eval()                          # freeze BatchNorm running stats
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            crit(model(X), y.argmax(1)).backward()
            opt.step()
        sched.step()
        va = _val_acc(model, val_loader, device)
        if va > best_val:
            best_val = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep+1:2d}/{epochs} | val_acc {va:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_finetune(args, task_cfg, k, device):
    """Full-model finetune via JetClassTrainer (single-GPU), then load best checkpoint."""
    setup_ddp(0, 1)
    torch.set_float32_matmul_precision('high')
    train_ds = make_dataset('train', args.data_dir, task_cfg['classes'], args.max_train)
    val_ds   = make_dataset('val',   args.data_dir, task_cfg['classes'], None)
    test_ds  = make_dataset('test',  args.data_dir, task_cfg['classes'], None)

    model = protocols.build_model('finetune', k, encoder_weights=args.weights).to(device)
    with open(args.finetune_config) as f:
        train_cfg = TrainConfig.from_dict(yaml.safe_load(f)['train'])
    trainer = JetClassTrainer(model=model, train_dataset=train_ds, val_dataset=val_ds,
                              test_dataset=test_ds, device=device,
                              metric=accuracy_metric_ce, config=train_cfg)
    trainer._set_logging_paths(args.run_name)
    history, _ = trainer.train()
    best_path = trainer.best_model_path
    cleanup_ddp()

    # evaluate the best checkpoint on the test set with the k-class metrics
    eval_model = protocols.build_model('finetune', k, encoder_weights=None).to(device)
    eval_model.load_state_dict(torch.load(best_path, map_location=device))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    signal_idx = tasks.signal_index(args.task) if task_cfg['kind'] == 'binary' else None
    result = eval_on_test(eval_model, test_loader, device, task_cfg['kind'], signal_idx)
    result['best_val_acc'] = float(max(history.get('val_metric', [0])))
    return result


def run_probe(args, task_cfg, k, device):
    train_ds = make_dataset('train', args.data_dir, task_cfg['classes'], args.max_train)
    val_ds   = make_dataset('val',   args.data_dir, task_cfg['classes'], None)
    test_ds  = make_dataset('test',  args.data_dir, task_cfg['classes'], None)
    lk = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, shuffle=True, **lk)
    val_loader   = DataLoader(val_ds, shuffle=False, **lk)
    test_loader  = DataLoader(test_ds, shuffle=False, **lk)

    model = protocols.build_model(args.protocol, k, encoder_weights=args.weights).to(device)
    model = train_probe(model, train_loader, val_loader, device, args.probe_epochs, args.probe_lr)
    signal_idx = tasks.signal_index(args.task) if task_cfg['kind'] == 'binary' else None
    result = eval_on_test(model, test_loader, device, task_cfg['kind'], signal_idx)
    result['best_val_acc'] = _val_acc(model, val_loader, device)
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_cfg = tasks.TASKS[args.task]
    k = len(task_cfg['classes'])
    run_name = args.run_name or f'{args.task}_{args.protocol}_seed{args.seed}'

    print(f"[cell] task={args.task} k={k} protocol={args.protocol} "
          f"encoder={'scratch' if args.weights is None else os.path.basename(args.weights)} "
          f"seed={args.seed}", flush=True)

    t0 = time.monotonic()
    if args.protocol == 'finetune':
        result = run_finetune(args, task_cfg, k, device)
    else:
        result = run_probe(args, task_cfg, k, device)
    result['train_time_s'] = time.monotonic() - t0

    if args.encoder_label is not None:
        enc_label = args.encoder_label
    elif args.weights is None:
        enc_label = 'scratch'
    else:
        w = args.weights.lower()
        enc_label = 'jepa' if 'jepa' in w else 'mae' if 'mae' in w else 'enc'
    result.update({
        'task': args.task, 'protocol': args.protocol,
        'encoder': 'scratch' if args.weights is None else args.weights,
        'encoder_label': enc_label,
        'seed': args.seed, 'k': k, 'run_name': run_name,
    })

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f'{run_name}.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    rej = f"  rej@0.5={result['bkg_rej_at_0.5']:.1f}" if 'bkg_rej_at_0.5' in result else ''
    print(f"[done] {run_name}: test_auc={result['test_auc']:.4f} acc={result['test_acc']:.4f}{rej}\n"
          f"       → {out}", flush=True)


if __name__ == '__main__':
    main()
