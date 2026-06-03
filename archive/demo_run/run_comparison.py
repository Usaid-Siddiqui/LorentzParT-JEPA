"""
Run all three comparison experiments and produce a summary.

Experiments:
  1. JEPA pretrain  → fine-tune → evaluate
  2. MAE pretrain   → fine-tune → evaluate
  3. From scratch   → fine-tune → evaluate

Save paths are all deterministic and fixed:
  Pretraining:
    ./logs/ParticleJEPA/best/jepa_pretrain.pt   ← JEPA encoder weights
    ./logs/LorentzParT/best/mae_pretrain.pt      ← MAE encoder weights
  Fine-tuning (full model state dicts):
    ./logs/LorentzParT/best/jepa_finetune.pt
    ./logs/LorentzParT/best/mae_finetune.pt
    ./logs/LorentzParT/best/scratch.pt
  Evaluation plots:
    ./outputs/{jepa_finetune,mae_finetune,scratch}_{roc_curve,confusion_matrix}.png
  CSV logs:
    ./logs/ParticleJEPA/logging/jepa_pretrain.csv
    ./logs/LorentzParT/logging/{mae_pretrain,jepa_finetune,mae_finetune,scratch}.csv

Usage:
    python scripts/run_comparison.py --data-dir ./data --seed 42

Resume flags (skip a stage if already done):
    --skip-jepa-pretrain  --jepa-weights ./logs/ParticleJEPA/best/jepa_pretrain.pt
    --skip-mae-pretrain   --mae-weights  ./logs/LorentzParT/best/mae_pretrain.pt
"""

import os
import sys
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.viz import plot_pretraining_comparison


# Deterministic paths — no glob, no modification-time guessing
JEPA_PRETRAIN_WEIGHTS = './logs/ParticleJEPA/best/jepa_pretrain.pt'
MAE_PRETRAIN_WEIGHTS  = './logs/LorentzParT/best/mae_pretrain.pt'
JEPA_FT_WEIGHTS       = './logs/LorentzParT/best/jepa_finetune.pt'
MAE_FT_WEIGHTS        = './logs/LorentzParT/best/mae_finetune.pt'
SCRATCH_FT_WEIGHTS    = './logs/LorentzParT/best/scratch.pt'
OUTPUTS_DIR           = './outputs'

# CSV logs written by trainers — used for the convergence comparison plot
JEPA_PRETRAIN_CSV = './logs/ParticleJEPA/logging/jepa_pretrain.csv'
MAE_PRETRAIN_CSV  = './logs/LorentzParT/logging/mae_pretrain.csv'


def parse_args():
    parser = argparse.ArgumentParser(description="Run full JEPA vs MAE comparison")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-jepa-pretrain', action='store_true')
    parser.add_argument('--skip-mae-pretrain', action='store_true')
    parser.add_argument('--jepa-weights', type=str, default=JEPA_PRETRAIN_WEIGHTS)
    parser.add_argument('--mae-weights', type=str, default=MAE_PRETRAIN_WEIGHTS)
    return parser.parse_args()


def run(cmd: list, desc: str):
    print(f"\n{'=' * 60}")
    print(f"STAGE: {desc}")
    print(f"CMD:   {' '.join(cmd)}")
    print('=' * 60)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    python = sys.executable

    # ------------------------------------------------------------------ #
    # Stage 1: JEPA pretraining                                            #
    # ------------------------------------------------------------------ #
    if not args.skip_jepa_pretrain:
        run(
            [python, 'scripts/pretrain_jepa.py',
             '--data-dir', args.data_dir,
             '--config-path', 'configs/pretrain_jepa.yaml',
             '--run-name', 'jepa_pretrain',
             '--seed', str(args.seed)],
            "JEPA Pretraining"
        )
    jepa_weights = args.jepa_weights
    if not os.path.exists(jepa_weights):
        raise FileNotFoundError(
            f"JEPA pretrained weights not found: {jepa_weights}\n"
            f"Run without --skip-jepa-pretrain or provide --jepa-weights."
        )
    print(f"\nJEPA encoder weights: {jepa_weights}")

    # ------------------------------------------------------------------ #
    # Stage 2: MAE pretraining                                             #
    # ------------------------------------------------------------------ #
    if not args.skip_mae_pretrain:
        run(
            [python, 'scripts/pretrain_mae.py',
             '--data-dir', args.data_dir,
             '--config-path', 'configs/pretrain_mae.yaml',
             '--run-name', 'mae_pretrain',
             '--seed', str(args.seed)],
            "MAE Pretraining"
        )
    mae_weights = args.mae_weights
    if not os.path.exists(mae_weights):
        raise FileNotFoundError(
            f"MAE pretrained weights not found: {mae_weights}\n"
            f"Run without --skip-mae-pretrain or provide --mae-weights."
        )
    print(f"\nMAE encoder weights: {mae_weights}")

    # ------------------------------------------------------------------ #
    # Pretraining convergence comparison plot                              #
    # ------------------------------------------------------------------ #
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    if os.path.exists(JEPA_PRETRAIN_CSV) and os.path.exists(MAE_PRETRAIN_CSV):
        convergence_plot = os.path.join(OUTPUTS_DIR, 'pretrain_convergence_comparison.png')
        plot_pretraining_comparison(
            jepa_csv=JEPA_PRETRAIN_CSV,
            mae_csv=MAE_PRETRAIN_CSV,
            save_fig=convergence_plot,
        )
        print(f"\nPretraining convergence plot saved to: {convergence_plot}")
    else:
        print("\nSkipping convergence plot — one or both CSV logs not found.")

    # ------------------------------------------------------------------ #
    # Stage 3a: Fine-tune with JEPA pretrained weights                     #
    # ------------------------------------------------------------------ #
    run(
        [python, 'scripts/finetune.py',
         '--data-dir', args.data_dir,
         '--config-path', 'configs/finetune.yaml',
         '--weights', jepa_weights,
         '--run-name', 'jepa_finetune',
         '--seed', str(args.seed)],
        "Fine-tune (JEPA pretrained)"
    )

    # ------------------------------------------------------------------ #
    # Stage 3b: Fine-tune with MAE pretrained weights                      #
    # ------------------------------------------------------------------ #
    run(
        [python, 'scripts/finetune.py',
         '--data-dir', args.data_dir,
         '--config-path', 'configs/finetune.yaml',
         '--weights', mae_weights,
         '--run-name', 'mae_finetune',
         '--seed', str(args.seed)],
        "Fine-tune (MAE pretrained)"
    )

    # ------------------------------------------------------------------ #
    # Stage 3c: Fine-tune from scratch                                     #
    # ------------------------------------------------------------------ #
    run(
        [python, 'scripts/finetune.py',
         '--data-dir', args.data_dir,
         '--config-path', 'configs/finetune.yaml',
         '--run-name', 'scratch',
         '--seed', str(args.seed)],
        "Fine-tune (from scratch)"
    )

    # Verify fine-tuned weights exist before evaluation
    for label, path in [
        ('JEPA fine-tuned', JEPA_FT_WEIGHTS),
        ('MAE fine-tuned',  MAE_FT_WEIGHTS),
        ('Scratch',         SCRATCH_FT_WEIGHTS),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} weights not found at {path}. "
                f"Did fine-tuning complete and did --run-name match?"
            )

    # ------------------------------------------------------------------ #
    # Stage 4: Evaluation                                                  #
    # ------------------------------------------------------------------ #
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    for run_name, weights in [
        ('jepa_finetune', JEPA_FT_WEIGHTS),
        ('mae_finetune',  MAE_FT_WEIGHTS),
        ('scratch',       SCRATCH_FT_WEIGHTS),
    ]:
        run(
            [python, 'scripts/evaluate.py',
             '--data-dir', args.data_dir,
             '--weights', weights,
             '--run-name', run_name,
             '--outputs-dir', OUTPUTS_DIR],
            f"Evaluate ({run_name})"
        )

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"\nPlots saved to: {OUTPUTS_DIR}/")
    print("  pretrain_convergence_comparison.png  ← val loss vs epoch + wall-clock time")
    print("  jepa_finetune_roc_curve.png")
    print("  jepa_finetune_confusion_matrix.png")
    print("  mae_finetune_roc_curve.png")
    print("  mae_finetune_confusion_matrix.png")
    print("  scratch_roc_curve.png")
    print("  scratch_confusion_matrix.png")
    print(f"\nModel weights:")
    print(f"  JEPA pretrained  → {jepa_weights}")
    print(f"  MAE pretrained   → {mae_weights}")
    print(f"  JEPA fine-tuned  → {JEPA_FT_WEIGHTS}")
    print(f"  MAE fine-tuned   → {MAE_FT_WEIGHTS}")
    print(f"  Scratch          → {SCRATCH_FT_WEIGHTS}")


if __name__ == '__main__':
    main()
