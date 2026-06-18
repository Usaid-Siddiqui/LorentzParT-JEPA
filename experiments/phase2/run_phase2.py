"""
Phase 2 — 1M-scale headline run, hardened for long unattended execution.

Same comparison as Phase 0 (JEPA gate+biased+K1 vs MAE-biased vs scratch, + probes,
+ embedding stats), but with three robustness fixes over run_phase0.py for a multi-day
run you can't babysit:

  1. ERROR ISOLATION — every stage is wrapped; a failed stage (OOM, transient error)
     is logged and skipped, and the run CONTINUES to the next stage/seed instead of
     killing everything. A bad finetune at seed 1 no longer loses seeds 2..N.
  2. RESUME/SKIP — each stage skips if its checkpoint/JSON already exists, so a
     crash-and-restart picks up where it left off (no recomputation).
  3. NO CHECKPOINT COLLISION — checkpoints are tagged (default '1m'), e.g.
     jepa_1m_seed42.pt, so the 100k Phase 0 checkpoints (jepa_seed42.pt) are NOT
     overwritten.

Output JSONs use the SAME condition keys as Phase 0 (jepa_finetune / mae_finetune /
scratch / jepa_probe / mae_probe) so analyze_results.py works unchanged.

    python experiments/phase2/run_phase2.py \\
        --data-dir ./data_1m \\
        --pretrain-config configs/pretrain_jepa_biased_1m.yaml \\
        --mae-config configs/pretrain_mae_1m.yaml \\
        --finetune-config configs/train_lorentz_part_1m.yaml \\
        --seeds 42 123 456 7 21 --output-dir experiments/phase2/results

Resume after an interruption: re-run the exact same command — completed stages skip.
"""

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))           # experiments/phase2
_REPO = os.path.dirname(os.path.dirname(_HERE))              # repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'experiments', 'phase0'))

import torch

from run_phase0 import (
    run_stage, evaluate_classifier, get_embedding_stats,
    read_best_csv, read_last_csv,
    _FINETUNE_COLS, _JEPA_PRETRAIN_COLS, _MAE_PRETRAIN_COLS,
    CLASS_NAMES,
)
from src.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 — hardened 1M-scale headline run")
    p.add_argument('--data-dir',        default='./data_1m')
    p.add_argument('--pretrain-config', default='./configs/pretrain_jepa_biased_1m.yaml')
    p.add_argument('--mae-config',      default='./configs/pretrain_mae_1m.yaml')
    p.add_argument('--finetune-config', default='./configs/train_lorentz_part_1m.yaml')
    p.add_argument('--probe-config',    default='./experiments/phase0/configs/linear_probe.yaml')
    p.add_argument('--seeds',           nargs='+', type=int, default=[42, 123, 456, 7, 21])
    p.add_argument('--output-dir',      default='./experiments/phase2/results')
    p.add_argument('--gpu',             type=int, default=0)
    p.add_argument('--tag',             default='1m',
                   help="Checkpoint name tag (keeps these distinct from 100k Phase 0)")
    p.add_argument('--skip-probe',      action='store_true')
    return p.parse_args()


def safe_stage(cmd, env, desc):
    """run_stage with error isolation — log failure and continue instead of crashing."""
    try:
        run_stage(cmd, env, desc)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[STAGE FAILED] {desc}\n  {e}\n  --> logged, continuing.\n", flush=True)
        return False


def run_seed(seed, args, python, env, device):
    tag = args.tag
    # tagged checkpoint paths (distinct from 100k Phase 0)
    jepa_ckpt = f'./logs/ParticleJEPA/best/jepa_{tag}_seed{seed}.pt'
    mae_ckpt  = f'./logs/LorentzParT/best/mae_{tag}_seed{seed}.pt'
    jepa_ft   = f'./logs/LorentzParT/best/jepa_{tag}_ft_seed{seed}.pt'
    mae_ft    = f'./logs/LorentzParT/best/mae_{tag}_ft_seed{seed}.pt'
    scratch   = f'./logs/LorentzParT/best/scratch_{tag}_seed{seed}.pt'
    jepa_pt_csv = f'./logs/ParticleJEPA/logging/jepa_{tag}_seed{seed}.csv'
    mae_pt_csv  = f'./logs/LorentzParT/logging/mae_{tag}_seed{seed}.csv'
    jepa_ft_csv = f'./logs/LorentzParT/logging/jepa_{tag}_ft_seed{seed}.csv'
    mae_ft_csv  = f'./logs/LorentzParT/logging/mae_{tag}_ft_seed{seed}.csv'
    scratch_csv = f'./logs/LorentzParT/logging/scratch_{tag}_seed{seed}.csv'
    probe_jepa_json = os.path.join(args.output_dir, f'probe_jepa_{tag}_seed{seed}.json')
    probe_mae_json  = os.path.join(args.output_dir, f'probe_mae_{tag}_seed{seed}.json')

    def done(path, what):
        if os.path.exists(path):
            print(f"[skip] {what} — exists: {path}", flush=True)
            return True
        return False

    # ── Pretraining (skip if checkpoint exists) ─────────────────────────────────
    if not done(jepa_ckpt, 'JEPA pretrain'):
        safe_stage([python, 'scripts/pretrain_jepa.py', '--data-dir', args.data_dir,
                    '--config-path', args.pretrain_config,
                    '--run-name', f'jepa_{tag}_seed{seed}', '--seed', seed],
                   env, f'JEPA pretrain seed={seed}')
    if not done(mae_ckpt, 'MAE pretrain'):
        safe_stage([python, 'scripts/pretrain_mae.py', '--data-dir', args.data_dir,
                    '--config-path', args.mae_config,
                    '--run-name', f'mae_{tag}_seed{seed}', '--seed', seed],
                   env, f'MAE pretrain seed={seed}')

    # ── Fine-tuning (skip if done; skip pretrained variants if encoder missing) ──
    for run_name, weights, ft_ckpt in [
        (f'jepa_{tag}_ft_seed{seed}',   jepa_ckpt, jepa_ft),
        (f'mae_{tag}_ft_seed{seed}',    mae_ckpt,  mae_ft),
        (f'scratch_{tag}_seed{seed}',   None,      scratch),
    ]:
        if done(ft_ckpt, f'finetune {run_name}'):
            continue
        if weights and not os.path.exists(weights):
            print(f"[skip] finetune {run_name} — encoder missing (pretrain failed?)", flush=True)
            continue
        cmd = [python, 'scripts/train_lorentz_part.py', '--data-dir', args.data_dir,
               '--config-path', args.finetune_config, '--run-name', run_name, '--seed', seed]
        if weights:
            cmd += ['--weights', weights]
        safe_stage(cmd, env, f'Finetune {run_name}')

    # ── Linear probing ──────────────────────────────────────────────────────────
    if not args.skip_probe:
        for probe_name, ckpt, pj in [
            (f'probe_jepa_{tag}_seed{seed}', jepa_ckpt, probe_jepa_json),
            (f'probe_mae_{tag}_seed{seed}',  mae_ckpt,  probe_mae_json),
        ]:
            if done(pj, f'probe {probe_name}'):
                continue
            if not os.path.exists(ckpt):
                print(f"[skip] probe {probe_name} — encoder missing", flush=True)
                continue
            safe_stage([python, 'experiments/phase0/linear_probe.py',
                        '--data-dir', args.data_dir, '--weights', ckpt,
                        '--config-path', args.probe_config, '--run-name', probe_name,
                        '--seed', seed, '--output-dir', args.output_dir],
                       env, f'Probe {probe_name}')

    # ── Embedding stats (guarded) ───────────────────────────────────────────────
    def embed(ckpt):
        if not os.path.exists(ckpt):
            return {}
        try:
            return get_embedding_stats(ckpt, args.data_dir, device, jepa_config=args.pretrain_config)
        except Exception as e:                              # noqa: BLE001 — never let stats kill the run
            print(f"[warn] embedding stats failed for {ckpt}: {e}", flush=True)
            return {}
    jepa_embed = embed(jepa_ckpt)
    mae_embed  = embed(mae_ckpt)

    # ── Test-set eval (guarded) → Phase-0-compatible condition keys ─────────────
    conditions = {}
    for label, ft_path, ft_csv in [
        ('jepa_finetune', jepa_ft, jepa_ft_csv),
        ('mae_finetune',  mae_ft,  mae_ft_csv),
        ('scratch',       scratch, scratch_csv),
    ]:
        c = {'best_val_acc': read_best_csv(ft_csv, 'val_metric', mode='max',
                                           fallback_cols=_FINETUNE_COLS),
             'pretrain_time_s': None}
        if os.path.exists(ft_path):
            try:
                c.update(evaluate_classifier(ft_path, args.data_dir, device, args.finetune_config))
                print(f"{label}: test_acc={c.get('test_acc')}  test_auc={c.get('test_auc')}", flush=True)
            except Exception as e:                          # noqa: BLE001
                print(f"[warn] eval failed for {label}: {e}", flush=True)
        else:
            print(f"[warn] {label} checkpoint missing — recorded empty", flush=True)
        conditions[label] = c

    conditions['jepa_finetune']['pretrain_time_s'] = read_last_csv(
        jepa_pt_csv, 'elapsed_total_s', fallback_cols=_JEPA_PRETRAIN_COLS)
    conditions['mae_finetune']['pretrain_time_s'] = read_last_csv(
        mae_pt_csv, 'elapsed_total_s', fallback_cols=_MAE_PRETRAIN_COLS)
    conditions['jepa_finetune']['embedding_stats'] = jepa_embed
    conditions['mae_finetune']['embedding_stats'] = mae_embed

    for label, jp in [('jepa_probe', probe_jepa_json), ('mae_probe', probe_mae_json)]:
        conditions[label] = json.load(open(jp)) if os.path.exists(jp) else {}

    return {'seed': seed, 'conditions': conditions}


def main():
    args = parse_args()
    python = sys.executable
    os.makedirs(args.output_dir, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print(f"Phase 2 (tag={args.tag}) — seeds {args.seeds} | GPU {args.gpu} | out {args.output_dir}", flush=True)

    for seed in args.seeds:
        print(f"\n{'#' * 60}\n# SEED {seed}\n{'#' * 60}", flush=True)
        t0 = time.monotonic()
        set_seed(seed)
        try:
            results = run_seed(seed, args, python, env, device)
            out_path = os.path.join(args.output_dir, f'seed_{seed}.json')
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSeed {seed} done in {(time.monotonic()-t0)/60:.1f} min → {out_path}", flush=True)
        except Exception as e:                              # noqa: BLE001 — isolate per-seed failures
            print(f"\n[SEED {seed} FAILED] {e}\n  --> continuing to next seed.\n", flush=True)

    print(f"\n{'=' * 60}\nAll requested seeds attempted. Aggregate with:\n"
          f"  python experiments/analyze_results.py --results-dir {args.output_dir}", flush=True)


if __name__ == '__main__':
    main()
