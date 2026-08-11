# Phase 5 — True Lorentz equivariance (alternate branch)

Isolated phase: build the *fully* Lorentz-equivariant version of the model and measure whether
exact equivariance changes anything — in accuracy AND compute. Kept separate so it never disturbs
the published Phase 0–4 results (which run the efficient hybrid architecture).

## Background
LorentzParT is a **hybrid by design**, not a fully equivariant model: a ParT core "nudged in the
right physical direction by a small dose of LGATr's Lorentz structure" (Nguyen's GSoC-2025 article),
which deliberately avoids the "computational overhead" of strict equivariance. `scripts/equivariance_test.py`
quantifies how far that nudge lands from exact invariance: rotation/boost move the softmax output by
~3e-3 (vs a ~2e-7 permutation-control floor) — a lean toward invariance, not a guarantee. The three
things that make it inexact (all consistent with the hybrid intent): a dense `Linear(16, embed_dim)`
bridging the geometric-algebra embedding to the scalar transformer core, `embed_vector` fed `(pT,η,φ,E)`
instead of Cartesian, and the per-feature normalization (pT/92.7, E/133.9) which distorts the 4-vector.
This phase builds the exact-equivariant endpoint to price the tradeoff. See memory `equivariance-broken`.

## Milestones
- **M1 — verify the fix is exactly invariant. ✅ DONE.**
  `verify_invariance.py` — `InvariantGATr`: LGATr backbone + invariant readout
  (`extract_scalar` on output mv channels + scalar-channel outputs; a Linear on invariants
  stays invariant), Cartesian `embed_vector`, single global scale. Invariance test residuals
  drop to the ~1e-7 floor on rotation, longitudinal boost, and transverse boost (vs ~3e-3
  broken). Full SO(1,3) invariance, random init, no GPU.
- **M2 — 100k performance comparison.** Promote `InvariantGATr` to a trainable model and
  compare fixed-equivariant vs. broken LorentzParT (scratch, 10-class). Question: does exact
  equivariance help, hurt, or tie? (Prior: likely ties — the Phase-1 washout suggests the
  symmetry prior doesn't help here — which would be a strong "even *real* equivariance
  doesn't help" result.)
- **M3 — 1M run**, only if M2 is promising.

## Open design decisions for M2 (not yet made)
- **Full Lorentz vs collider subgroup.** M1 is full SO(1,3)-invariant (no spurions). Jet
  physics has a preferred frame (beam axis), so the physically-motivated choice may be the
  beam-preserving subgroup via L-GATr spurions (`get_spurions`). Compare both, or pick.
- **Head.** M1 uses mean-pool for the test; a class-attention head on the *invariant* features
  stays invariant and matches the other models — use that for training.
- **Padding.** M1 tests all-valid jets; real jets need padded particles masked in LGATr attention.
- **Normalization.** Must be a single global scale (covariant), not per-feature — changes the
  data pipeline for this model only.
- **SSL.** Whether to also pretrain JEPA/MAE on the invariant backbone, or first answer the
  supervised scratch question.

## Note
Adopting this for real means re-pretraining + re-running the sweep (architecture change →
results not comparable to Phase 0–4). M1 is the cheap proof; M2+ is the expensive part —
gated on whether an equivariance claim is worth making.
