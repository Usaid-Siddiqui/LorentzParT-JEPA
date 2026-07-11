"""
Full 100k ablation on the FIXED (ragged) backbone — a clean redo of the Phase-1
exploration now that the interaction embedding is correct.

Sweeps the full grid, all with ``ragged_pair_embed: true``:

    gate       ∈ {on, off}                (--gate)
    mask_mode  ∈ {biased, random}         (--masks)
    num_mask K ∈ {1, 2, 4}                (--ks)

= every gate × mask × K JEPA combination, PLUS a now-correct MAE (biased/random)
and a from-scratch floor — each × 3 seeds. Every condition's config is generated
from the ragged base configs, so the ONLY differences between runs are the swept
knobs (nothing else drifts). Finetune loads the BEST-VAL encoder by default.

Resumable: each condition's eval is written into seed_N.json as it finishes, and
a re-run skips anything already there (and skips pretrain/finetune whose checkpoint
exists). A single failed condition is logged and the sweep continues.

    python experiments/phase1/run_ablation_ragged.py \\
        --data-dir ./data --seeds 42 123 456 --gpu 0 \\
        --output-dir experiments/phase1/results_ragged

Aggregate:
    python experiments/analyze_results.py --results-dir experiments/phase1/results_ragged
"""

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import time

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))              # experiments/phase1
_REPO = os.path.dirname(os.path.dirname(_HERE))                 # repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)                                       # for run_ablation helpers

import torch

from run_ablation import evaluate_classifier, read_training_curve, run_stage
from src.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Full ragged-backbone 100k ablation")
    p.add_argument('--data-dir',    default='./data')
    p.add_argument('--seeds',       nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--gate',        choices=['both', 'on', 'off'], default='both')
    p.add_argument('--masks',       nargs='+', default=['biased', 'random'])
    p.add_argument('--ks',          nargs='+', type=int, default=[1, 2, 4, 8],
                   help="num_mask values for the multiparticle sweep (e.g. 1 2 4 8 16)")
    p.add_argument('--base-jepa-config', default='configs/pretrain_jepa_ragged.yaml')
    p.add_argument('--base-mae-config',  default='configs/pretrain_mae.yaml')
    p.add_argument('--finetune-config',  default='configs/train_lorentz_part_ragged.yaml')
    p.add_argument('--curriculum-config', default='configs/pretrain_jepa_curriculum_ragged.yaml',
                   help="static config for the one curriculum-masking JEPA condition")
    p.add_argument('--no-curriculum', action='store_true', help="skip the curriculum condition")
    p.add_argument('--encoder',     choices=['best', 'final'], default='best',
                   help="which pretrained encoder to finetune: best-val (default) or final-epoch")
    p.add_argument('--no-mae',      action='store_true', help="skip the MAE comparison conditions")
    p.add_argument('--no-scratch',  action='store_true', help="skip the from-scratch floor")
    p.add_argument('--output-dir',  default='experiments/phase1/results_ragged')
    p.add_argument('--gpu',         type=int, default=0)
    p.add_argument('--skip-pretrain', action='store_true')
    p.add_argument('--skip-finetune', action='store_true')
    return p.parse_args()


def _dump(cfg, path):
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def gen_jepa_config(base, gate, mask, k, out):
    """Ragged JEPA config with only gate / mask_mode / num_mask overridden."""
    cfg = copy.deepcopy(base)
    cfg['model']['ragged_pair_embed'] = True
    cfg['model']['jepa']['use_attention_gate'] = gate
    cfg['model']['jepa']['mask_mode'] = mask
    cfg['model']['jepa']['num_mask'] = k
    return _dump(cfg, out)


def gen_mae_config(base, mask, out):
    """Ragged MAE config with only mask_mode overridden (MAE has no gate/K axis)."""
    cfg = copy.deepcopy(base)
    cfg['model']['ragged_pair_embed'] = True
    cfg['model']['mask_mode'] = mask
    return _dump(cfg, out)


def build_conditions(args, gcfg_dir):
    """Ordered list of {name, obj, config} for the whole grid + MAE + scratch."""
    base_jepa = yaml.safe_load(open(args.base_jepa_config))
    base_mae = yaml.safe_load(open(args.base_mae_config))
    gates = {'both': [('on', True), ('off', False)],
             'on':   [('on', True)],
             'off':  [('off', False)]}[args.gate]

    conds = []
    for (gname, gval), mask, k in itertools.product(gates, args.masks, args.ks):
        name = f'jepa_gate{gname}_{mask}_k{k}'
        conds.append({'name': name, 'obj': 'jepa',
                      'config': gen_jepa_config(base_jepa, gval, mask, k,
                                                os.path.join(gcfg_dir, name + '.yaml'))})
    if not args.no_curriculum:
        conds.append({'name': 'jepa_curriculum', 'obj': 'jepa', 'config': args.curriculum_config})
    if not args.no_mae:
        for mask in args.masks:
            name = f'mae_{mask}'
            conds.append({'name': name, 'obj': 'mae',
                          'config': gen_mae_config(base_mae, mask,
                                                   os.path.join(gcfg_dir, name + '.yaml'))})
    if not args.no_scratch:
        conds.append({'name': 'scratch', 'obj': 'scratch', 'config': None})
    return conds


def encoder_path(pre_dir, name, seed, which):
    """Pretrained encoder to finetune. jepa_trainer overwrites '<run>.pt' with the
    final encoder and preserves best-val as '<run>_best.pt'; prefer best-val unless
    asked for final, and fall back to whatever exists."""
    final = os.path.join(pre_dir, 'best', f'{name}_seed{seed}.pt')
    best = os.path.join(pre_dir, 'best', f'{name}_seed{seed}_best.pt')
    if which == 'best' and os.path.exists(best):
        return best
    return final


def main():
    args = parse_args()
    python = sys.executable
    gcfg_dir = os.path.join(args.output_dir, 'generated_configs')
    os.makedirs(gcfg_dir, exist_ok=True)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    conds = build_conditions(args, gcfg_dir)
    print(f"Ragged ablation | {len(conds)} conditions × {len(args.seeds)} seeds "
          f"| gate={args.gate} masks={args.masks} ks={args.ks} | encoder={args.encoder}")
    for c in conds:
        print(f"  - {c['name']} ({c['obj']})")

    for seed in args.seeds:
        set_seed(seed)
        print(f"\n{'#' * 60}\n# SEED {seed}\n{'#' * 60}", flush=True)
        t0 = time.monotonic()

        out_path = os.path.join(args.output_dir, f'seed_{seed}.json')
        if os.path.exists(out_path):
            seed_results = json.load(open(out_path))
        else:
            seed_results = {'seed': seed, 'conditions': {}}
        done = set(seed_results['conditions'].keys())

        for c in conds:
            name, obj, cfg = c['name'], c['obj'], c['config']
            if name in done:
                print(f"[skip] {name} — already in seed_{seed}.json", flush=True)
                continue
            print(f"\n--- {name} (seed {seed}) ---", flush=True)

            pre_dir = './logs/ParticleJEPA' if obj == 'jepa' else './logs/LorentzParT'
            ft_ckpt = f'./logs/LorentzParT/best/{name}_ft_seed{seed}.pt'
            ft_csv = f'./logs/LorentzParT/logging/{name}_ft_seed{seed}.csv'

            try:
                # 1. Pretrain (jepa / mae) ---------------------------------------
                if obj in ('jepa', 'mae'):
                    final_ckpt = f'{pre_dir}/best/{name}_seed{seed}.pt'
                    script = 'scripts/pretrain_jepa.py' if obj == 'jepa' else 'scripts/pretrain_mae.py'
                    if not (args.skip_pretrain or os.path.exists(final_ckpt)):
                        run_stage([python, script, '--data-dir', args.data_dir,
                                   '--config-path', cfg,
                                   '--run-name', f'{name}_seed{seed}', '--seed', seed],
                                  env, f'Pretrain {name} seed={seed}')
                    weights = encoder_path(pre_dir, name, seed, args.encoder)
                else:
                    weights = None

                # 2. Finetune ---------------------------------------------------
                if not (args.skip_finetune or os.path.exists(ft_ckpt)):
                    if weights is not None and not os.path.exists(weights):
                        print(f"[skip] {name} finetune — encoder missing ({weights})", flush=True)
                        continue
                    cmd = [python, 'scripts/train_lorentz_part.py', '--data-dir', args.data_dir,
                           '--config-path', args.finetune_config,
                           '--run-name', f'{name}_ft_seed{seed}', '--seed', seed]
                    if weights is not None:
                        cmd += ['--weights', weights]
                    run_stage(cmd, env, f'Finetune {name} seed={seed}')

                # 3. Eval -------------------------------------------------------
                if not os.path.exists(ft_ckpt):
                    print(f"[skip eval] {name} — no finetune checkpoint", flush=True)
                    continue
                metrics = evaluate_classifier(ft_ckpt, args.data_dir, device, args.finetune_config)
                metrics['training_curve'] = read_training_curve(ft_csv)
                seed_results['conditions'][name] = metrics
                with open(out_path, 'w') as f:
                    json.dump(seed_results, f, indent=2)
                print(f"  {name}: test_acc={metrics['test_acc']:.4f} "
                      f"test_auc={metrics['test_auc']:.4f} → saved", flush=True)

            except subprocess.CalledProcessError as e:
                print(f"\n[STAGE FAILED] {name} seed={seed}: {e}\n  --> logged, continuing.\n", flush=True)

        print(f"\nSeed {seed} done in {(time.monotonic() - t0) / 60:.1f} min", flush=True)

    print(f"\n{'=' * 60}\nAll conditions attempted. Aggregate with:\n"
          f"  python experiments/analyze_results.py --results-dir {args.output_dir}", flush=True)


if __name__ == '__main__':
    main()
