# Phase 0 — Multi-Seed Rigorous Baseline

Establishes credible baseline numbers for JEPA vs MAE vs scratch training before any novel architectural work. All results are aggregated over **3 random seeds (42, 123, 456)** on a balanced **100k-jet subset** of JetClass (10 classes, 8k/1k/1k train/val/test per class).

---

## Conditions

| Condition | Description |
|-----------|-------------|
| `jepa_finetune` | Pretrain with JEPA (20 epochs), then fine-tune encoder + head |
| `mae_finetune` | Pretrain with MAE (20 epochs), then fine-tune encoder + head |
| `scratch` | Train encoder + head from random init, no pretraining |
| `jepa_probe` | Pretrain with JEPA, freeze encoder, train linear head only |
| `mae_probe` | Pretrain with MAE, freeze encoder, train linear head only |

---

## Results (mean ± std, 3 seeds)

**Primary metric: macro-averaged OVO ROC AUC** (one-vs-one over all 45 class pairs).

| Condition | OVO ROC AUC | Test Accuracy |
|-----------|:-----------:|:-------------:|
| **JEPA → finetune** | **0.762 ± 0.014** | **31.25% ± 1.96%** |
| Scratch | 0.743 ± 0.016 | 28.26% ± 1.65% |
| MAE → finetune | 0.722 ± 0.009 | 24.76% ± 1.50% |
| MAE → probe | 0.686 ± 0.003 | 22.28% ± 0.20% |
| JEPA → probe | 0.542 ± 0.010 | 19.02% ± 0.27% |

<p align="center">
  <img src="results/auc_bars.png" width="70%">
</p>

Key findings:

- **JEPA finetune beats scratch by +1.9pp AUC and beats MAE finetune by +4.0pp AUC.** JEPA learns relational structure in latent space rather than per-particle feature statistics, which transfers better to classification.
- **Scratch outperforms MAE finetune** — negative transfer. The MAE reconstruction objective (predict raw pT/η/φ/E) encourages the encoder to memorise low-level feature statistics, the wrong inductive bias for jet topology classification.
- **JEPA probe is weak (0.542 — barely above random).** JEPA representations are not linearly separable; finetuning is essential to unlock them. This is expected for abstract latent-space SSL.
- **MAE probe > JEPA probe (0.686 vs 0.542)** — MAE encodes raw feature correlations a linear head can exploit directly; JEPA does not.

---

## Per-Class AUC

<p align="center">
  <img src="results/per_class_auc_bars.png" width="90%">
</p>

| Class | JEPA → finetune | MAE → finetune | Scratch |
|-------|:---------------:|:--------------:|:-------:|
| QCD/Z→νν | 0.679 ± 0.021 | 0.622 ± 0.012 | 0.641 ± 0.024 |
| H→bb̄ | 0.669 ± 0.013 | 0.639 ± 0.006 | 0.647 ± 0.008 |
| H→cc̄ | 0.680 ± 0.019 | 0.638 ± 0.002 | 0.653 ± 0.026 |
| H→gg | 0.864 ± 0.005 | 0.846 ± 0.003 | 0.858 ± 0.007 |
| H→4q | 0.767 ± 0.014 | 0.735 ± 0.009 | 0.750 ± 0.011 |
| H→ℓνqq′ | 0.780 ± 0.001 | 0.779 ± 0.001 | 0.778 ± 0.001 |
| Z→qq̄ | 0.677 ± 0.014 | 0.641 ± 0.006 | 0.668 ± 0.024 |
| W→qq′ | 0.749 ± 0.019 | 0.701 ± 0.012 | 0.730 ± 0.027 |
| **t→bqq′** | **0.907 ± 0.026** | 0.789 ± 0.043 | 0.863 ± 0.038 |
| t→bℓν | 0.846 ± 0.008 | 0.835 ± 0.003 | 0.842 ± 0.005 |

JEPA's largest margin is on **t→bqq′ (+11.8pp over MAE, +4.4pp over scratch)** — a three-body hadronic top decay requiring reasoning about the relationship between the b-jet and two light quarks from W→qq′. This is precisely the kind of inter-particle relational structure JEPA is designed to capture.

H→ℓνqq′ is notably flat across all conditions (≈0.779) — this decay has a neutrino which is invisible to the detector, making it structurally harder to distinguish regardless of pretraining.

---

## Per-Class Accuracy

<p align="center">
  <img src="results/per_class_bars.png" width="90%">
</p>

| Class | JEPA → finetune | MAE → finetune | Scratch |
|-------|:---------------:|:--------------:|:-------:|
| QCD/Z→νν | 0.258 ± 0.047 | 0.168 ± 0.034 | 0.179 ± 0.045 |
| H→bb̄ | 0.104 ± 0.006 | 0.166 ± 0.039 | 0.096 ± 0.006 |
| H→cc̄ | 0.184 ± 0.007 | 0.160 ± 0.045 | 0.155 ± 0.033 |
| H→gg | 0.554 ± 0.004 | 0.500 ± 0.034 | 0.573 ± 0.036 |
| H→4q | 0.279 ± 0.043 | 0.217 ± 0.104 | 0.267 ± 0.003 |
| H→ℓνqq′ | 0.237 ± 0.022 | 0.199 ± 0.093 | 0.245 ± 0.035 |
| Z→qq̄ | 0.139 ± 0.014 | 0.066 ± 0.056 | 0.110 ± 0.030 |
| W→qq′ | 0.406 ± 0.038 | 0.366 ± 0.069 | 0.451 ± 0.041 |
| **t→bqq′** | **0.649 ± 0.055** | 0.340 ± 0.129 | 0.446 ± 0.094 |
| t→bℓν | 0.314 ± 0.041 | 0.293 ± 0.130 | 0.306 ± 0.031 |

---

## Embedding Collapse Diagnostics

<p align="center">
  <img src="results/embedding_stats.png" width="60%">
</p>

| Metric | JEPA | MAE |
|--------|:----:|:---:|
| Effective rank | ~7 | ~25 |
| Mean variance | ~48 | ~1.8 |
| Mean cosine sim | ~0.21 | ~0.67 |

MAE embeddings have high effective rank but high cosine similarity — many dimensions used but pointing similarly, suggesting spread without diversity. JEPA has lower effective rank but much higher variance and lower cosine similarity — fewer but more distinct, well-separated embedding directions.

---

## Files

| File | Purpose |
|------|---------|
| `run_phase0.py` | Orchestrates all 5 conditions across multiple seeds; writes `results/seed_{seed}.json` |
| `../analyze_results.py` | Aggregates seed JSONs → summary tables + figures (shared across phases) |
| `linear_probe.py` | Standalone linear probe (frozen encoder + linear head) |
| `patch_results.py` | One-off backfill script for fixing missing fields in existing JSONs |
| `configs/linear_probe.yaml` | Hyperparameters for linear probe (AdamW, CosineAnnealingLR, 10 epochs) |
| `results/` | Seed JSONs and output figures |

---

## Reproducing

```bash
# Run all 5 conditions for 3 seeds (~1-2 hrs per seed on A100)
python experiments/phase0/run_phase0.py \
    --data-dir ./data --seeds 42 123 456 --gpu 0

# Re-eval only (existing checkpoints, adds AUC to JSONs)
python experiments/phase0/run_phase0.py \
    --data-dir ./data --seeds 42 123 456 --gpu 0 \
    --skip-pretrain --skip-finetune

# Aggregate and plot
python experiments/analyze_results.py \
    --results-dir ./experiments/phase0/results
```

Use `--skip-pretrain`, `--skip-finetune`, `--skip-probe` to resume after preemption.
