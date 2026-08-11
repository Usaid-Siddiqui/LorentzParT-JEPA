# Phase 5 — Lorentz equivariance: the hybrid tradeoff, quantified

**Status: complete at M1 (the informative part). M2/M3 parked (see end).**

This phase started from a question — *"is LorentzParT actually Lorentz-equivariant, and if we
made it exactly so, would anything change?"* — and answered the cheap, decisive half of it. Kept
isolated so it never disturbs the Phase 0–4 results (which run the efficient hybrid).

## The question and the answer up front
LorentzParT is a **hybrid by design**, not a fully equivariant model. Per Nguyen's GSoC-2025
article, the ParT core is *"nudged in the right physical direction by a small dose of LGATr's
Lorentz structure,"* and strict equivariance was deliberately avoided because it *"introduces
computational overhead."* The EquiLinear layer *"might encourage the encoder to learn
Lorentz-equivariant interactions"* — aspiration, not guarantee.

We put numbers on both halves of that stated tradeoff:
- **How much of a nudge?** The classifier is ~3e-3 off invariance (vs a full-L-GATr's ~1e-7) — a
  measurable lean toward physical structure, not a symmetry constraint.
- **What does exact equivariance cost?** ~2.5× compute (with 7× fewer parameters).

So this is not a bug report. It's the quantitative evidence for the design choice the article
described qualitatively, and it places LorentzParT on a clean spectrum:

| model | Lorentz invariance | compute | params |
|---|---|---|---|
| vanilla ParT | none | — | — |
| **LorentzParT (hybrid)** | **~3e-3 nudge** | **1.0×** | 2.27M |
| full L-GATr (`InvariantGATr`) | ~1e-7 (exact) | 2.5× | 0.31M |

## M1 — the invariance measurement (`verify_invariance.py`, `scripts/equivariance_test.py`)
Invariance is architectural, so both tests run on random-init models (no data, no GPU). Apply a
Lorentz transform to the input 4-momenta and measure `max|Δ softmax-prob|`; a particle-permutation
control (which the model *is* invariant to) validates that the test detects invariance when present.

`max|Δprob|`:

| transform | LorentzParT (hybrid) | `InvariantGATr` (exact) |
|---|---|---|
| permutation (control) | 2.2e-07 (invariant) | 1.5e-08 (invariant) |
| azimuthal rotation | **3.3e-03** | **1.4e-07** |
| longitudinal boost | **2.4e-03** | **9.5e-07** |
| transverse boost | **1.3e-03** | **4.3e-07** |

The hybrid breaks invariance by ~4 orders of magnitude over the control floor — the "nudge." The
three things that make it inexact, all consistent with the hybrid intent:
1. a dense `Linear(16, embed_dim)` bridging the geometric-algebra embedding to the scalar transformer
   core (mixes GA grades — the mechanism of the tradeoff);
2. `embed_vector` fed `(pT,η,φ,E)` rather than Cartesian `(E,px,py,pz)`;
3. per-feature normalization (pT/92.7, E/133.9), which itself distorts the 4-vector.

`InvariantGATr` fixes all three — LGATr backbone kept in multivector space, invariant readout
(`extract_scalar` + scalar channels, no dense `Linear(16,·)`), Cartesian embedding, single global
scale — and reaches exact full SO(1,3) invariance (~1e-7).

## M1 — the compute measurement (CPU, B=32, N=128)
| model | params | forward (ms) | × hybrid |
|---|---|---|---|
| LorentzParT (hybrid, 8 layers) | 2.27M | 162 | 1.0 |
| `InvariantGATr` (8 blocks) | 0.31M | 410 | **2.5×** |

Geometric-algebra profile: parameter-efficient but compute-dense (multivector contractions cost
more per parameter). Exact equivariance is ~2.5× the compute. The 7× param-efficiency only helps in
a data/memory-limited regime — not ours (labels free, compute-bound) — so it doesn't tip the balance.

## Conclusion
The hybrid is the **compute-efficient middle** of the equivariance spectrum, and that's a deliberate,
now-validated choice: it captures physics-informed structure at ~40% the compute of exact
equivariance. Given the Phase-1 ablation washout (the symmetry prior never visibly helps accuracy),
the expected verdict on exact equivariance is **a 2.5× compute tax for no accuracy gain** — which
*supports* the hybrid rather than indicting it. This dovetails with the project's compute-efficiency
thesis: more symmetry is more compute, and here it doesn't buy accuracy.

## M2/M3 — parked
Building a trainable `InvariantGATr` (class-attention head on invariants, padding-masked LGATr,
single-scale pipeline) and running the 100k/1M accuracy comparison would turn "likely ties" into a
measured fact and complete the spectrum table. Deferred because the outcome is predictable and the
GPU is better spent finishing the Phase-4 compute story. If revived: decide full-Lorentz vs
beam-spurion subgroup (`get_spurions`) and supervised-scratch-first vs SSL. Adopting it for real
means re-pretraining + re-running the sweep (results not comparable to Phase 0–4).
