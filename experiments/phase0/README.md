# Phase 0 — Multi-Seed Rigorous Baseline

Establishes credible baseline numbers for JEPA vs MAE vs scratch training before any novel architectural work. All results are aggregated over 3 random seeds (42, 123, 456) on a balanced 100k-jet subset of JetClass.

## Conditions

| Condition | Description |
|-----------|-------------|
| `jepa_finetune` | Pretrain with JEPA (20 epochs), then fine-tune encoder + head |
| `mae_finetune` | Pretrain with MAE (20 epochs), then fine-tune encoder + head |
| `scratch` | Train encoder + head from random init (no pretraining) |
| `jepa_probe` | Pretrain with JEPA, freeze encoder, train linear head only |
| `mae_probe` | Pretrain with MAE, freeze encoder, train linear head only |

## Key Results (mean ± std, 3 seeds)

| Condition | Test AUC | Test Accuracy |
|-----------|:--------:|:-------------:|
| JEPA → finetune | **—** | **29.56%** |
| MAE → finetune | — | 24.06% |
| Scratch | — | 23.64% |

JEPA's largest per-class gain is on t→bqq′ (+38 pp over MAE, +46 pp over scratch), consistent with learning inter-particle relational structure rather than per-particle feature statistics.

## Files

| File | Purpose |
|------|---------|
| `run_phase0.py` | Orchestrates all 5 conditions across multiple seeds; writes `results/seed_{seed}.json` |
| `analyze_results.py` | Aggregates seed JSONs → summary tables + bar chart figures |
| `linear_probe.py` | Standalone linear probe evaluation (frozen encoder + linear head) |
| `patch_results.py` | One-off backfill script for fixing missing fields in existing JSONs |
| `configs/linear_probe.yaml` | Hyperparameters for linear probe (AdamW, CosineAnnealingLR, 10 epochs) |
| `results/` | Seed JSONs (`seed_*.json`, `probe_*.json`) and output figures |

## Reproducing

```bash
# Run all 5 conditions for 3 seeds (requires GPU, ~1-2 hrs per seed on A100)
python experiments/phase0/run_phase0.py \
    --data-dir ./data --seeds 42 123 456 --gpu 0

# Aggregate results and generate figures
python experiments/phase0/analyze_results.py \
    --results-dir experiments/phase0/results
```

Use `--skip-pretrain`, `--skip-finetune`, `--skip-probe` to resume after preemption.
