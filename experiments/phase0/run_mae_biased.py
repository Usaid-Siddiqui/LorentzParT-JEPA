"""
Augment Phase 0 with a fair (biased-masking) MAE baseline.

The original Phase 0 MAE used random masking, which handicaps it: Nguyen's
LorentzParT MAE design relies on biased (high-pT weighted) masking. This runner
re-pretrains MAE with biased masking, finetunes, evaluates, and probes — then
merges the new `mae_biased_finetune` / `mae_biased_probe` conditions into the
existing experiments/phase0/results/seed_{seed}.json files WITHOUT touching the
original conditions (jepa_finetune, mae_finetune, scratch, *_probe).

New run names (mae_biased_*) so the original random-MAE checkpoints are untouched.
Resumable: skips any stage whose checkpoint already exists.

    python experiments/phase0/run_mae_biased.py --data-dir ./data --seeds 42 123 456 --gpu 0

Aggregate the full (objective × masking) picture with:
    python experiments/analyze_results.py --results-dir experiments/phase0/results \\
        --conditions jepa_finetune mae_finetune mae_biased_finetune scratch \\
                     jepa_probe mae_probe mae_biased_probe
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # for run_phase0 helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from run_phase0 import (
    run_stage, evaluate_classifier, get_embedding_stats,
    read_best_csv, read_last_csv,
    _FINETUNE_COLS, _MAE_PRETRAIN_COLS,
)
from src.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 MAE-biased augmentation")
    p.add_argument('--data-dir',        default='./data')
    p.add_argument('--mae-config',      default='./configs/pretrain_mae.yaml')
    p.add_argument('--finetune-config', default='./configs/train_lorentz_part.yaml')
    p.add_argument('--probe-config',    default='./experiments/phase0/configs/linear_probe.yaml')
    p.add_argument('--results-dir',     default='./experiments/phase0/results')
    p.add_argument('--seeds',           nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--gpu',             type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    python = sys.executable
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    for seed in args.seeds:
        set_seed(seed)
        print(f"\n{'#' * 60}\n# SEED {seed} — MAE biased\n{'#' * 60}")

        pretrain_ckpt = f'./logs/LorentzParT/best/mae_biased_seed{seed}.pt'
        finetune_ckpt = f'./logs/LorentzParT/best/mae_biased_ft_seed{seed}.pt'
        pretrain_csv  = f'./logs/LorentzParT/logging/mae_biased_seed{seed}.csv'
        finetune_csv  = f'./logs/LorentzParT/logging/mae_biased_ft_seed{seed}.csv'
        probe_json    = os.path.join(args.results_dir, f'probe_mae_biased_seed{seed}.json')

        # 1. Pretrain (biased masking, set in pretrain_mae.yaml)
        if os.path.exists(pretrain_ckpt):
            print(f"[SKIP pretrain] {pretrain_ckpt}")
        else:
            run_stage(
                [python, 'scripts/pretrain_mae.py',
                 '--data-dir', args.data_dir, '--config-path', args.mae_config,
                 '--run-name', f'mae_biased_seed{seed}', '--seed', seed],
                env=env, desc=f'MAE-biased pretrain seed={seed}',
            )

        # 2. Finetune
        if os.path.exists(finetune_ckpt):
            print(f"[SKIP finetune] {finetune_ckpt}")
        else:
            run_stage(
                [python, 'scripts/train_lorentz_part.py',
                 '--data-dir', args.data_dir, '--config-path', args.finetune_config,
                 '--weights', pretrain_ckpt,
                 '--run-name', f'mae_biased_ft_seed{seed}', '--seed', seed],
                env=env, desc=f'MAE-biased finetune seed={seed}',
            )

        # 3. Linear probe (frozen encoder)
        if os.path.exists(probe_json):
            print(f"[SKIP probe] {probe_json}")
        else:
            run_stage(
                [python, 'experiments/phase0/linear_probe.py',
                 '--data-dir', args.data_dir, '--weights', pretrain_ckpt,
                 '--config-path', args.probe_config,
                 '--run-name', f'probe_mae_biased_seed{seed}', '--seed', seed,
                 '--output-dir', args.results_dir],
                env=env, desc=f'MAE-biased probe seed={seed}',
            )

        # 4. Eval + embedding stats
        finetune = {
            'best_val_acc':    read_best_csv(finetune_csv, 'val_metric', mode='max',
                                             fallback_cols=_FINETUNE_COLS),
            'pretrain_time_s': read_last_csv(pretrain_csv, 'elapsed_total_s',
                                             fallback_cols=_MAE_PRETRAIN_COLS),
        }
        finetune.update(evaluate_classifier(finetune_ckpt, args.data_dir, device, args.finetune_config))
        finetune['embedding_stats'] = get_embedding_stats(pretrain_ckpt, args.data_dir, device)
        print(f"  mae_biased_finetune: test_acc={finetune['test_acc']:.4f}  "
              f"test_auc={finetune['test_auc']:.4f}")

        with open(probe_json) as f:
            probe = json.load(f)

        # 5. Merge into existing phase0 seed JSON (augment, don't overwrite)
        seed_path = os.path.join(args.results_dir, f'seed_{seed}.json')
        with open(seed_path) as f:
            seed_results = json.load(f)
        seed_results['conditions']['mae_biased_finetune'] = finetune
        seed_results['conditions']['mae_biased_probe'] = probe
        with open(seed_path, 'w') as f:
            json.dump(seed_results, f, indent=2)
        print(f"Merged mae_biased_* into {seed_path}")

    print("\nAll seeds done. Aggregate with:")
    print("  python experiments/analyze_results.py --results-dir experiments/phase0/results \\")
    print("    --conditions jepa_finetune mae_finetune mae_biased_finetune scratch \\")
    print("                 jepa_probe mae_probe mae_biased_probe")


if __name__ == '__main__':
    main()
