# LorentzParT-JEPA

Self-supervised pretraining for jet classification using a Joint-Embedding Predictive Architecture (JEPA). Built on [LorentzParT](https://medium.com/@thanhnguyen14401/gsoc-2025-with-ml4sci-event-classification-with-masked-transformer-autoencoders-6da369d42140) (Thanh Nguyen, GSoC 2025) — a Lorentz-equivariant particle transformer for the JetClass benchmark.


---


## Architecture

```
Input (pT, η, φ, E)  →  ParticleProcessor
                              ├── 16-dim Lorentz multivectors  (L-GATr embed_vector)
                              └── pairwise interaction matrix U  (ln Δ, ln kT, ln z, ln m²)

AttentionGate: U → mean over neighbors → MLP → scalar weight per particle
Gated multivectors *= gate

Context Encoder  [trainable]  ←  gated, masked input
Target Encoder   [EMA copy]   ←  full unmasked input  (frozen)
Predictor        [bottleneck, 64-dim]  →  predicted embedding at masked position

Loss: LayerNorm-MSE(predicted, target)
EMA: θ_target ← m·θ_target + (1−m)·θ_context,  m: 0.996 → 1.0
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/dry_run.py   # 30 checks, no GPU needed
```

**Dependencies:** `torch>=2.0`, `lgatr>=1.4` (`pip install lgatr`), `uproot`, `awkward`, `einops`

## Data

Download JetClass val_5M (~7.6 GB) from [Zenodo](https://zenodo.org/records/6619768), then:

```bash
python scripts/prepare_data.py --data-dir /path/to/val_5M --output-dir ./data --seed 42
```

---

## References

1. Qu et al. "Particle Transformer for Jet Tagging." *ICML 2022*. arXiv:2202.03772
2. Spinner et al. "Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics." *NeurIPS 2024*. arXiv:2405.14806
3. Assran et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." *CVPR 2023*. arXiv:2301.08243
4. He et al. "Masked Autoencoders Are Scalable Vision Learners." *CVPR 2022*. arXiv:2111.06377
5. Nguyen, T.P. "GSoC 2025: Event Classification with Masked Transformer Autoencoders." *Medium 2025*
