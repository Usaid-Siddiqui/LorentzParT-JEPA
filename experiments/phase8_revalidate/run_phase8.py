"""
Phase 8 revalidation sweep — {part, lorentzpart} x {scratch, jepa, mae} at one data scale.

Re-asks the thesis question ("does SSL help?") on the FIXED code. Every SSL arm runs two stages
(pretrain -> finetune from the best-val encoder); scratch is one stage. Resumable: a cell whose
best checkpoint already exists is skipped.

    # fixed arm (default) at 1M, three seeds
    python experiments/phase8_revalidate/run_phase8.py \
        --train-dir <train_10M> --val-dir <val_5M> --scale 1m \
        --seeds 42 123 456 --nproc 1 --steps-per-epoch 2000 --num-epochs 4

    # paired control on the PRE-FIX behaviour (adds a _broken suffix to every run name)
    python experiments/phase8_revalidate/run_phase8.py ... --broken

Compare with `python experiments/phase8_revalidate/compare_phase8.py --scale 1m --seeds 42 123 456`.
"""

import argparse
import os
import subprocess
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

ENTRY = 'experiments/phase8_revalidate/train_phase8.py'
CFG = {
    'supervised': 'experiments/phase8_revalidate/configs/phase8_supervised.yaml',
    'jepa_pretrain': 'experiments/phase8_revalidate/configs/phase8_jepa_pretrain.yaml',
    'mae_pretrain': 'experiments/phase8_revalidate/configs/phase8_mae_pretrain.yaml',
}
CLS_LOGDIR = {'part': 'logs/ParticleTransformer/best', 'lorentzpart': 'logs/LorentzParT/best'}
JEPA_BEST = 'logs/ParticleJEPA/best'


def parse_args():
    p = argparse.ArgumentParser(description="Phase 8 revalidation sweep")
    p.add_argument('--train-dir', required=True)
    p.add_argument('--val-dir', required=True)
    p.add_argument('--scale', required=True, help='label for run names, e.g. 100k/1m')
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--models', nargs='+', default=['lorentzpart'], choices=['part', 'lorentzpart'])
    p.add_argument('--protocols', nargs='+', default=['scratch', 'jepa', 'mae'],
                   choices=['scratch', 'jepa', 'mae'])
    p.add_argument('--features', type=int, default=4, choices=[4, 14])
    p.add_argument('--nproc', type=int, default=1)
    p.add_argument('--steps-per-epoch', type=int, default=None)
    p.add_argument('--num-epochs', type=int, default=None)
    p.add_argument('--pretrain-epochs', type=int, default=None,
                   help='epochs for the SSL PRETRAIN stage (default: same as --num-epochs). '
                        'SSL normally needs far more pretraining than finetuning; the original '
                        'recipes used 20-30 pretrain epochs.')
    p.add_argument('--pretrain-steps-per-epoch', type=int, default=None,
                   help='steps/epoch for the SSL PRETRAIN stage (default: same as --steps-per-epoch)')
    p.add_argument('--mask-mode', default='biased')
    p.add_argument('--broken', action='store_true',
                   help='pre-Phase-8 behaviour; run names get a _broken suffix')
    return p.parse_args()


def torchrun(args, model, protocol, run_name, weights=None):
    cfg = CFG.get(protocol, CFG['supervised'])
    # SSL pretraining gets its own budget when asked for; every other stage uses the shared one.
    is_pre = protocol.endswith('_pretrain')
    epochs = (args.pretrain_epochs if is_pre and args.pretrain_epochs is not None
              else args.num_epochs)
    steps = (args.pretrain_steps_per_epoch if is_pre and args.pretrain_steps_per_epoch is not None
             else args.steps_per_epoch)
    cmd = ['torchrun', '--standalone', f'--nproc_per_node={args.nproc}', ENTRY,
           '--model', model, '--protocol', protocol,
           '--train-dir', args.train_dir, '--val-dir', args.val_dir,
           '--config', cfg, '--run-name', run_name,
           '--features', str(args.features), '--mask-mode', args.mask_mode,
           '--seed', str(run_name.split('seed')[-1].split('_')[0])]
    if weights:
        cmd += ['--weights', weights]
    if steps is not None:
        cmd += ['--steps-per-epoch', str(steps)]
    if epochs is not None:
        cmd += ['--num-epochs', str(epochs)]
    if args.broken:
        cmd += ['--broken']
    print(f"\n[run] {run_name}\n  {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, cwd=_REPO, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {run_name}: {e} — logged, continuing.", flush=True)
        return False


def main():
    args = parse_args()
    sfx = '_broken' if args.broken else ''
    n_done = n_skip = n_fail = 0
    t0 = time.monotonic()

    for seed in args.seeds:
        for model in args.models:
            for proto in args.protocols:
                if proto == 'scratch':
                    rn = f'{model}_scratch_{args.scale}_seed{seed}{sfx}'
                    if os.path.exists(os.path.join(_REPO, CLS_LOGDIR[model], f'{rn}.pt')):
                        print(f"[skip] {rn} — exists", flush=True); n_skip += 1; continue
                    ok = torchrun(args, model, 'scratch', rn)
                    n_done += ok; n_fail += (not ok)
                    continue

                # SSL: pretrain then finetune
                ft = f'{model}_{proto}_{args.scale}_seed{seed}{sfx}'
                if os.path.exists(os.path.join(_REPO, CLS_LOGDIR[model], f'{ft}.pt')):
                    print(f"[skip] {ft} — exists", flush=True); n_skip += 1; continue

                pre = f'{model}_{proto}pre_{args.scale}_seed{seed}{sfx}'
                pre_dir = JEPA_BEST if proto == 'jepa' else CLS_LOGDIR[model]
                # JEPATrainer writes <run>_best.pt; MaskedModelTrainer writes <run>.pt
                pre_best = os.path.join(_REPO, pre_dir,
                                        f'{pre}_best.pt' if proto == 'jepa' else f'{pre}.pt')
                if not os.path.exists(pre_best):
                    ok = torchrun(args, model, f'{proto}_pretrain', pre)
                    n_done += ok; n_fail += (not ok)
                    if not ok:
                        continue
                else:
                    print(f"[skip] {pre} — encoder exists", flush=True); n_skip += 1
                if not os.path.exists(pre_best):
                    print(f"[FAILED] {pre}: no checkpoint at {pre_best}", flush=True)
                    n_fail += 1; continue
                ok = torchrun(args, model, f'{proto}_finetune', ft, weights=pre_best)
                n_done += ok; n_fail += (not ok)

    print(f"\n{'='*66}\nPhase 8 [{args.scale}{sfx}]: {n_done} run, {n_skip} skipped, "
          f"{n_fail} failed in {(time.monotonic()-t0)/60:.1f} min.", flush=True)


if __name__ == '__main__':
    main()
