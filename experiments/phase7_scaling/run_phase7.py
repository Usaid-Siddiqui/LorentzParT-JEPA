"""
Phase 7 orchestrator — the 2x2 {part, lorentzpart} x {scratch, jepa} at one data scale, on 2 GPUs.

Shells `torchrun --nproc_per_node=N` per cell (resumable: a cell whose best checkpoint exists is
skipped; a failure is logged and the sweep continues). JEPA cells run two stages: pretrain the
encoder, then finetune from its best-val checkpoint. Climb the data ladder by pointing --train-dir
at successively larger subsets (100k → 1M → 10M → 100M) and setting --steps-per-epoch/--num-epochs.

    python experiments/phase7_scaling/run_phase7.py \
        --train-dir /data/jetclass/train_100M --val-dir /data/jetclass/val_5M \
        --scale 100m --seeds 42 --nproc 2 --steps-per-epoch 20000 --num-epochs 10

Aggregate accuracies with a per-scale table (reuse analyze_phase4-style tooling on the logs).
"""

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

SUP_CFG = 'experiments/phase7_scaling/configs/phase7_supervised.yaml'
JEPA_CFG = 'experiments/phase7_scaling/configs/phase7_jepa_pretrain.yaml'
ENTRY = 'experiments/phase7_scaling/train_phase7.py'
# model → (classification-log dir for the supervised best checkpoint)
CLS_LOGDIR = {'part': 'logs/ParticleTransformer/best', 'lorentzpart': 'logs/LorentzParT/best'}
JEPA_BEST = 'logs/ParticleJEPA/best'


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
    return p.parse_args()


def torchrun(args, model, protocol, run_name, weights=None):
    cfg = JEPA_CFG if protocol == 'jepa_pretrain' else SUP_CFG
    cmd = ['torchrun', '--standalone', f'--nproc_per_node={args.nproc}', ENTRY,
           '--model', model, '--protocol', protocol,
           '--train-dir', args.train_dir, '--val-dir', args.val_dir,
           '--config', cfg, '--run-name', run_name,
           '--features', str(args.features), '--seed', str(run_name.split('seed')[-1])]
    if weights:
        cmd += ['--weights', weights]
    if args.steps_per_epoch is not None:
        cmd += ['--steps-per-epoch', str(args.steps_per_epoch)]
    if args.num_epochs is not None:
        cmd += ['--num-epochs', str(args.num_epochs)]
    print(f"\n[run] {run_name}\n  {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, cwd=_REPO, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {run_name}: {e} — logged, continuing.", flush=True)
        return False


def main():
    args = parse_args()
    n_done = n_skip = n_fail = 0
    t0 = time.monotonic()
    for seed in args.seeds:
        for model in args.models:
            for protocol in args.protocols:
                if protocol == 'scratch':
                    rn = f'{model}_scratch_{args.scale}_seed{seed}'
                    best = os.path.join(_REPO, CLS_LOGDIR[model], f'{rn}.pt')
                    if os.path.exists(best):
                        print(f"[skip] {rn} — exists", flush=True); n_skip += 1; continue
                    ok = torchrun(args, model, 'scratch', rn)
                    n_done += ok; n_fail += (not ok)
                else:  # jepa: pretrain then finetune
                    ft = f'{model}_jepa_{args.scale}_seed{seed}'
                    ft_best = os.path.join(_REPO, CLS_LOGDIR[model], f'{ft}.pt')
                    if os.path.exists(ft_best):
                        print(f"[skip] {ft} — exists", flush=True); n_skip += 1; continue
                    pre = f'{model}_jepapre_{args.scale}_seed{seed}'
                    pre_best = os.path.join(_REPO, JEPA_BEST, f'{pre}_best.pt')
                    if not os.path.exists(pre_best):
                        ok = torchrun(args, model, 'jepa_pretrain', pre)
                        n_done += ok; n_fail += (not ok)
                        if not ok:
                            continue
                    else:
                        print(f"[skip] {pre} — encoder exists", flush=True); n_skip += 1
                    ok = torchrun(args, model, 'jepa_finetune', ft, weights=pre_best)
                    n_done += ok; n_fail += (not ok)

    print(f"\n{'='*60}\nPhase 7 [{args.scale}]: {n_done} run, {n_skip} skipped, {n_fail} failed "
          f"in {(time.monotonic()-t0)/60:.1f} min.", flush=True)


if __name__ == '__main__':
    main()
