"""
Phase 7 orchestrator — the 2x2 {part, lorentzpart} x {scratch, jepa} at one data scale, on 2 GPUs.

Shells `torchrun --nproc_per_node=N` per cell (resumable: a cell whose best checkpoint exists is
skipped; a failure is logged and the sweep continues). JEPA cells run two stages: pretrain the
encoder, then finetune from its best-val checkpoint. Climb the data ladder by pointing --train-dir
at successively larger subsets (100k → 1M → 10M → 100M) and setting --steps-per-epoch/--num-epochs.

    python experiments/phase7_scaling/run_phase7.py \
        --train-dir /data/jetclass/train_100M --val-dir /data/jetclass/val_5M \
        --scale 100m --seeds 42 --nproc 2 --steps-per-epoch 20000 --num-epochs 10

Phase 8 fixed-code rerun — ALWAYS tag it, or it collides with the old run names and every cell
is skipped as "exists" (0 launches, reported as success):

    python experiments/phase7_scaling/run_phase7.py ... --common-scale --cartesian-mv --tag fixed

Aggregate accuracies with a per-scale table (reuse analyze_phase4-style tooling on the logs).
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

SUP_CFG = 'experiments/phase7_scaling/configs/phase7_supervised.yaml'
JEPA_CFG = 'experiments/phase7_scaling/configs/phase7_jepa_pretrain.yaml'
ENTRY = 'experiments/phase7_scaling/train_phase7.py'
# model → (classification-log dir for the supervised best checkpoint)
CLS_LOGDIR = {'part': 'logs/ParticleTransformer/best', 'lorentzpart': 'logs/LorentzParT/best'}
JEPA_BEST = 'logs/ParticleJEPA/best'


def epochs_logged(best_dir, run_name):
    """Distinct epochs in a run's training CSV (0 if there is no log)."""
    path = os.path.join(_REPO, os.path.dirname(best_dir), 'logging', f'{run_name}.csv')
    if not os.path.exists(path):
        return 0
    seen = set()
    with open(path, newline='') as f:
        for row in csv.reader(f):
            try:
                seen.add(int(float(row[0])))
            except (ValueError, IndexError):
                pass                              # header / malformed line
    return len(seen)


def target_epochs(args, cfg):
    if args.num_epochs is not None:
        return args.num_epochs
    with open(os.path.join(_REPO, cfg)) as f:
        return int(yaml.safe_load(f)['train']['num_epochs'])


def is_complete(args, best_dir, run_name, cfg, artifact):
    """A cell is done only when it has logged EVERY epoch and its artifact exists.

    The old test was "the best checkpoint exists" — but that file is written the first time
    validation improves, i.e. after epoch 1. A cell killed mid-training therefore looked finished,
    and a relaunch would have skipped it and reported a 4-epoch model as the 30-epoch result.
    """
    return os.path.exists(artifact) and epochs_logged(best_dir, run_name) >= target_epochs(args, cfg)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 7 2x2 sweep at one data scale")
    p.add_argument('--train-dir', required=True)
    p.add_argument('--val-dir', required=True)
    p.add_argument('--scale', required=True, help="label for run names, e.g. 100k/1m/10m/100m")
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--models', nargs='+', default=['part', 'lorentzpart'], choices=['part', 'lorentzpart'])
    p.add_argument('--protocols', nargs='+', default=['scratch', 'jepa'], choices=['scratch', 'jepa'])
    p.add_argument('--features', type=int, default=14, choices=[4, 14])
    p.add_argument('--nproc', type=int, default=2, help="GPUs (torchrun --nproc_per_node)")
    p.add_argument('--steps-per-epoch', type=int, default=None, help="override configs for this scale")
    p.add_argument('--num-epochs', type=int, default=None)
    p.add_argument('--common-scale', action='store_true',
                   help='Phase 8 fix: one scale for pT and E (use a distinct --scale label)')
    p.add_argument('--cartesian-mv', action='store_true',
                   help='Phase 8 fix: Cartesian embed_vector (lorentzpart only, needs --common-scale)')
    p.add_argument('--bool-padding-mask', action='store_true',
                   help='Phase 8: exclude padded slots from attention (use with a NEW --tag)')
    p.add_argument('--tag', default=None,
                   help="suffix for every run name, e.g. 'fixed' -> lorentzpart_scratch_100m_seed42_fixed. "
                        "REQUIRED when re-running a scale that already has checkpoints: cells whose "
                        "checkpoint exists are skipped, so an untagged rerun silently does nothing.")
    return p.parse_args()


def note_partial(args, best_dir, run_name, cfg):
    n = epochs_logged(best_dir, run_name)
    if n:
        print(f"[partial] {run_name}: {n}/{target_epochs(args, cfg)} epochs logged — "
              f"the trainer will resume from its checkpoint", flush=True)


def torchrun(args, model, protocol, run_name, seed, weights=None):
    cfg = JEPA_CFG if protocol == 'jepa_pretrain' else SUP_CFG
    cmd = ['torchrun', '--standalone', f'--nproc_per_node={args.nproc}', ENTRY,
           '--model', model, '--protocol', protocol,
           '--train-dir', args.train_dir, '--val-dir', args.val_dir,
           '--config', cfg, '--run-name', run_name,
           '--features', str(args.features), '--seed', str(seed)]
    if weights:
        cmd += ['--weights', weights]
    if args.steps_per_epoch is not None:
        cmd += ['--steps-per-epoch', str(args.steps_per_epoch)]
    if args.num_epochs is not None:
        cmd += ['--num-epochs', str(args.num_epochs)]
    if args.common_scale:
        cmd += ['--common-scale']
    if args.cartesian_mv:
        cmd += ['--cartesian-mv']
    if args.bool_padding_mask:
        cmd += ['--bool-padding-mask']
    print(f"\n[run] {run_name}\n  {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, cwd=_REPO, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {run_name}: {e} — logged, continuing.", flush=True)
        return False


def main():
    args = parse_args()
    sfx = f'_{args.tag}' if args.tag else ''
    n_done = n_skip = n_fail = 0
    t0 = time.monotonic()
    for seed in args.seeds:
        for model in args.models:
            for protocol in args.protocols:
                if protocol == 'scratch':
                    rn = f'{model}_scratch_{args.scale}_seed{seed}{sfx}'
                    best = os.path.join(_REPO, CLS_LOGDIR[model], f'{rn}.pt')
                    if is_complete(args, CLS_LOGDIR[model], rn, SUP_CFG, best):
                        print(f"[skip] {rn} — complete", flush=True); n_skip += 1; continue
                    note_partial(args, CLS_LOGDIR[model], rn, SUP_CFG)
                    ok = torchrun(args, model, 'scratch', rn, seed)
                    n_done += ok; n_fail += (not ok)
                else:  # jepa: pretrain then finetune
                    ft = f'{model}_jepa_{args.scale}_seed{seed}{sfx}'
                    ft_best = os.path.join(_REPO, CLS_LOGDIR[model], f'{ft}.pt')
                    if is_complete(args, CLS_LOGDIR[model], ft, SUP_CFG, ft_best):
                        print(f"[skip] {ft} — complete", flush=True); n_skip += 1; continue
                    pre = f'{model}_jepapre_{args.scale}_seed{seed}{sfx}'
                    pre_best = os.path.join(_REPO, JEPA_BEST, f'{pre}_best.pt')
                    if not is_complete(args, JEPA_BEST, pre, JEPA_CFG, pre_best):
                        note_partial(args, JEPA_BEST, pre, JEPA_CFG)
                        ok = torchrun(args, model, 'jepa_pretrain', pre, seed)
                        n_done += ok; n_fail += (not ok)
                        if not ok:
                            continue
                    else:
                        print(f"[skip] {pre} — encoder complete", flush=True); n_skip += 1
                    note_partial(args, CLS_LOGDIR[model], ft, SUP_CFG)
                    ok = torchrun(args, model, 'jepa_finetune', ft, seed, weights=pre_best)
                    n_done += ok; n_fail += (not ok)

    print(f"\n{'='*60}\nPhase 7 [{args.scale}{sfx}]: {n_done} run, {n_skip} skipped, {n_fail} failed "
          f"in {(time.monotonic()-t0)/60:.1f} min.", flush=True)


if __name__ == '__main__':
    main()
