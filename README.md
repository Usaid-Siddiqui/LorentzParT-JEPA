# LorentzParT-JEPA

**GSoC 2026 · ML4Sci — Self-Supervised Pretraining for Jet Classification**

This repo implements JEPA-style (Joint-Embedding Predictive Architecture) self-supervised pretraining for the LorentzParT model, applied to jet classification on the JetClass benchmark. The core claim: predicting in **latent embedding space** (JEPA) yields more transferable representations of jet physics than reconstructing raw particle features (MAE).

Built on top of [LorentzParT](https://medium.com/@thanhnguyen14401/gsoc-2025-with-ml4sci-event-classification-with-masked-transformer-autoencoders-6da369d42140) (Thanh Nguyen, GSoC 2025), which wraps a ParT-style transformer with L-GATr's Lorentz-equivariant EquiLinear layers.

---

## Results (Phase 0 — 3 seeds, 100k JetClass subset)

| Condition | Test Accuracy |
|-----------|:-------------:|
| **JEPA → finetune** | **29.56%** |
| MAE → finetune | 24.06% |
| From scratch | 23.64% |

JEPA's largest gains are on structurally complex decays: t→bqq′ (+38 pp over MAE, +46 pp over scratch), consistent with learning particle *relationships* in latent space rather than per-particle feature statistics.

See [`experiments/phase0/`](experiments/phase0/) for full per-class results and figures.

---

## Architecture

**Pretraining — ParticleJEPA**
```
Input (pT, η, φ, E)
  └── ParticleProcessor
        ├── 16-dim Lorentz multivectors  (via L-GATr embed_vector)
        └── pairwise interaction matrix U  (ln Δ, ln kT, ln z, ln m²)

  ┌── AttentionGate (Phase 1)
  │     U_context → mean over neighbors → MLP → scalar gate ∈ [0,1] per particle
  │     multivectors *= gate  (downweight irrelevant particles)
  │
  ├── Context Encoder  [trainable]  — sees gated, zeroed masked particle
  ├── Target Encoder   [EMA copy]   — sees full unmasked input (frozen)
  └── Predictor        [bottleneck transformer, 64-dim]
        Linear(128→64) → pos embed → 4× TransformerBlock → Linear(64→128)

EMA update: θ_target ← m·θ_target + (1−m)·θ_context,  m: 0.996 → 1.0
Loss: LayerNorm-MSE between predictor output and target encoder embedding
```

**Fine-tuning — LorentzParT**
```
Context encoder weights → LorentzParTEncoder
  └── CLS token → 2× ClassAttentionBlock → Linear(128→10)
```

---

## Repository Layout

```
LorentzParT_JEPA/
├── src/
│   ├── models/
│   │   ├── jepa.py              # ParticleJEPA — full JEPA pretraining model
│   │   ├── attention_gate.py    # AttentionGate — learned per-particle scalar gate
│   │   ├── predictor.py         # ParticlePredictor — bottleneck transformer
│   │   ├── lorentz_part.py      # LorentzParT + LorentzParTEncoder
│   │   ├── particle_transformer.py
│   │   ├── classifier.py
│   │   ├── processor.py         # ParticleProcessor + InteractionEmbedding
│   │   └── linear_probe.py
│   ├── engine/
│   │   ├── jepa_trainer.py      # JEPATrainer: EMA schedule, DDP support
│   │   ├── jetclass_trainer.py
│   │   ├── mm_trainer.py        # MAE trainer
│   │   └── trainer.py
│   ├── loss/
│   │   ├── embedding_loss.py    # LayerNorm-MSE JEPA loss
│   │   └── conservation_loss.py
│   ├── optim/
│   │   └── lookahead.py
│   └── utils/
│       ├── data/
│       ├── embedding_stats.py   # collapse diagnostics (effective rank, variance, cosine sim)
│       └── ...
├── scripts/
│   ├── prepare_data.py          # extract balanced subset from ROOT files → numpy
│   ├── dry_run.py               # 30 diagnostic checks, no GPU needed
│   ├── pretrain_jepa.py
│   ├── pretrain_mae.py
│   ├── finetune.py
│   ├── evaluate.py
│   └── run_comparison.py        # single-seed end-to-end demo
├── configs/
│   ├── pretrain_jepa.yaml
│   ├── pretrain_mae.yaml
│   ├── finetune.yaml
│   └── evaluate.yaml
├── experiments/
│   └── phase0/                  # multi-seed baseline (complete)
│       ├── results/             # seed JSONs + figures
│       └── README.md
└── archive/
    └── demo_run/                # single-seed demo outputs (proof-of-concept)
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify pipeline (no GPU needed)
python scripts/dry_run.py   # expect: 30/30 checks passed
```

## Data

Download the JetClass val_5M split (~7.6 GB) from [Zenodo](https://zenodo.org/records/6619768):

```bash
wget -P /path/to/val_5M \
  "https://zenodo.org/records/6619768/files/JetClass_Pythia_val_5M.tar?download=1"
tar -xf /path/to/val_5M/JetClass_Pythia_val_5M.tar -C /path/to/val_5M/

# Extract 100k balanced subset
python scripts/prepare_data.py --data-dir /path/to/val_5M --output-dir ./data --seed 42
```

---

## Key Dependencies

- `torch>=2.0.0`
- `lgatr>=1.4.0` — Lorentz-equivariant geometric algebra layers (`pip install lgatr`)
- `uproot`, `awkward`, `vector` — HEP ROOT file reading
- `einops`

---

## References

1. Qu et al. "Particle Transformer for Jet Tagging." *ICML 2022*. arXiv:2202.03772
2. Spinner et al. "Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics." *NeurIPS 2024*. arXiv:2405.14806
3. Assran et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." *CVPR 2023*. arXiv:2301.08243
4. He et al. "Masked Autoencoders Are Scalable Vision Learners." *CVPR 2022*. arXiv:2111.06377
5. Nguyen, T.P. "GSoC 2025: Event Classification with Masked Transformer Autoencoders." *Medium 2025*
