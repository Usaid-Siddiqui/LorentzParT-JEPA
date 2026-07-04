# Phase 3 — Optimizing (and fixing) the LorentzParT backbone

**Goal:** make the backbone meaningfully faster by profiling it, finding the real
hotspot, and attacking it — a de-risked deliverable that speeds up every later run
(pretraining + all Phase 4 finetunes).

**Summary:** profiling showed the *pairwise interaction embedding* — not the equivariant
ops — dominates training (~55%). Making it **padding-aware ("ragged")** gives **2×
end-to-end training / 5.6× memory**, and it removes a `-1e9` BatchNorm-corruption bug in
that embedding for free, which recovers **+0.15 AUC** at 100k. A decomposition control
proves the two are independent: the +0.15 is a *correctness* fix, the 2× is the *speed*
optimization. The bug is a reimplementation regression in Thanh's code, **not** a flaw in
the original ParT. Remaining work: re-validate at 1M on the finalized backbone (Phase 3.5).

---

## Why: the hotspot (`profile_backbone.py`)

H200, batch 1000, fwd+bwd, ranked by self-CUDA time:

| op | % CUDA | subsystem |
|---|---:|---|
| `cudnn_batch_norm_backward` | 35.3% | interaction embedding |
| `cudnn_batch_norm` (fwd) | 12.5% | interaction embedding |
| `mm` | 7.3% | linears (EquiLinear/dense) |
| `convolution_backward` | 4.6% | interaction embedding |
| `bmm` (attention) | 3.8% | attention |

**BatchNorm ≈ 48%, the whole pair-interaction path ≈ 55%.** The equivariant `lgatr` ops we
*assumed* would dominate don't appear; attention is only ~4%. It's memory-bandwidth-bound
(BN 124 GB / GELU 141 GB / conv-bwd 119 GB summed), and **~90% of the N² pairs are padding**
for typical jets — so the fix is to stop paying for the padding.

---

## Changes

### 1. Ragged (padding-aware) interaction embedding — the *speed* win
**What.** `RaggedInteractionEmbedding` (`src/models/processor.py`): gather only valid pairs
(`index_select`), run the **same** `BatchNorm1d + [Conv1d(k=1)+BN+GELU]×4` MLP on the
`(1, C, M)` gathered pairs, scatter back out-of-place (`index_copy`) to `(B·H, N, N)`. Uses
`Conv1d` (not `Linear`) so weights are byte-identical to stock → a pretrained checkpoint
loads with **0 missing / 0 unexpected**. Behind flag `ragged_pair_embed` (default off),
threaded through `LorentzParT` and `ParticleJEPA` (both context + target encoders).

**Why.** ~90% of the N² pairs are padding; the BN-dominated cost scales with pair count, so
processing only valid pairs (~12%) removes most of it.

**Results.**
- **Kernel benchmark** (`bench_interaction.py`, H200 batch 1000, train fwd+bwd, ~12% valid):
  baseline **219.8 ms / 43.3 GB → ragged 15.8 ms (13.9×) / 7.7 GB (5.6×)**. `torch.compile`
  alone is 2.75× (ragged beats it ~5×; compile graph-breaks on ragged's dynamic gather, so
  a production ragged path needs static indexing, not compile).
- **End-to-end (100k):** **1.95× pretrain, 2.10× finetune** per epoch.
- **Correctness:** float64 `gradcheck` passes w.r.t. both the input and the MLP params
  (`test_ragged_gradcheck.py`) — the backward is *correct*, not merely finite.

### 2. Remove the `-1e9` padding fill — the *correctness* win
**What.** The stock embedding fills padded pairs with `-1e9` *before* its leading
`BatchNorm1d`. Ragged never sees padded pairs; the `fill0` control fills with `0.0` in the
dense path. Both remove `-1e9` from the BN statistics (flag `pad_fill_zero`,
`ParticleProcessor.pad_fill`).

**Why.** Verified from trained checkpoints: stock's first BN has `running_mean ≈ −8.86e8,
running_var ≈ 1.0e17`. Every valid feature then normalizes to `(x + 8.86e8)/3.17e8 ≈ 2.795`
*regardless of x* → the interaction bias (ParT's core mechanism) carries ~zero information.

**Results — 100k end-to-end A/B** (3 seeds, softmax OVO AUC; configs differ *only* by the flag):

| condition | AUC | Δ vs stock | speed (pre / ft) |
|---|---:|---:|---:|
| stock (dense, −1e9) | 0.7862 ± 0.0222 | — | 1.0× / 1.0× |
| **ragged** (skip padding) | 0.9345 ± 0.0020 | **+0.148** | **1.95× / 2.10×** |
| **fill0** (dense, fill 0) | 0.9402 ± 0.0006 | **+0.154** | 1.0× / 1.0× |

**Decomposition:** `fill0` recovers **104%** of ragged's gain at 1.0× speed → the +0.15 AUC
is **entirely the BN-corruption fix**, and ragged's speed is fully orthogonal. (fill0 edges
ragged by a consistent ~0.006 — likely because ragged's *variable*-M valid-pair BN has
noisier batch stats than fill0's fixed-N² BN; minor, doesn't change the recommendation:
ragged for speed+accuracy, fill0 only if you want the last 0.006 without the speed.)

### 3. Enable TF32 — free matmul speedup
**What.** Enable TensorFloat-32 (`torch.set_float32_matmul_precision('high')`) in the
training scripts — it was **off**, so every matmul ran full fp32.

**Why.** TF32 gives ~1.5–2× on GEMMs (`mm`/`addmm`/`bmm`/EquiLinear) on A100/H100 with
negligible accuracy impact; it stacks with ragged (different subsystem) and is standard
practice (the peer repo has it on).

**Results.** Measured speedup pending (part of the re-profile / Phase 3.5). Needs a quick
100k accuracy-neutral check since it changes numerics slightly.

---

## Root cause: a reimplementation bug, not a ParT bug

Checked the reference `weaver-core` ParT (`weaver/nn/model/ParticleTransformer.py`):
- `PairEmbed` defaults `sparse_eval=(True, True)` → `forward` uses **`_forward_sparse`**
  (train + eval) whenever a mask is present, which masks padded pairs OUT
  (`pair_mask.nonzero()` → gather valid → BN over valid → scatter) — i.e. **the original
  ParT already does exactly our "ragged" fix by default.**
- grep for `1e9 / -inf / full_like / fill_value` → **none**; the only floors are `eps=1e-8`
  clamps, so even the dense fallback gives finite (~−18) padded values, never `-1e9`.

**So the original ParT is correct.** Thanh's reimplementation regressed by (a) *adding* the
`-1e9` fill (`processor.py:70`) and (b) *dropping* the masked/sparse path. Conceptual error:
`-1e9` is an *attention-mask* value (correct as a pre-softmax bias to kill padded keys) that
was misapplied to the *BatchNorm inputs* of the interaction features.

**Framing consequences:** our "ragged" method is **not novel** (it re-derives weaver's
`_forward_sparse`), and this is **not a ParT bug**. It's a correctness fix for our pipeline
+ a candidate PR to the ML4Sci repo. Paper wording: *"corrected the interaction embedding to
match the reference weaver ParT (masks padded pairs from normalization)."* Do **not** claim
"beat ParT by 0.15" or "found a ParT bug."

### Open question: how did Thanh's Hybrid "outperform" ParT with this bug?
The bug lives in a **shared** component: Thanh's ParT baseline imports the *same* buggy
`InteractionEmbedding` (`particle_transformer.py:8,66`); LGATr (`lorentz_gatr.py`) has no
interaction embedding (immune). So every in-repo comparison was broken-vs-broken or
broken-vs-immune — nothing looked anomalous. Any "Hybrid > ParT" result was **Hybrid vs
Thanh's own equally-broken ParT**, which the Hybrid can win purely on its Lorentz-equivariant
`EquiLinear` layers (orthogonal to the dead pair bias). It did **not** show the Hybrid beats
a *correctly-implemented* ParT — no controlled run exists, and a working pair bias would
likely close or reverse the gap. A clean claim requires re-running both with the fix, and the
conclusion could move.

---

## Files

| file | what it does |
|---|---|
| `profile_backbone.py` | profile `LorentzParT` fwd+bwd → ranked op table + `--plot-out` hotspot chart |
| `bench_interaction.py` | kernel A/B: baseline / torch.compile / ragged on the interaction embedding |
| `test_ragged_gradcheck.py` | float64 gradcheck of `RaggedInteractionEmbedding` (input + params) |
| `run_ragged_e2e.py` | end-to-end 100k A/B {stock, ragged, fill0}: AUC + speedup + decomp; writes `results/seed_*.json` |
| `plot_speedup.py` | per-epoch training-time bar chart (pretrain+finetune), speedup vs stock |
| `../analyze_results.py` | AUC / accuracy / per-class bar charts from the seed JSONs |
| `run_ragged_validation.py` | 1M finetune-only follow-up (reuses an existing encoder) |

**Model changes:** `RaggedInteractionEmbedding` + `pad_fill` (`src/models/processor.py`),
encoder wiring (`src/models/lorentz_part.py`, `src/models/jepa.py`), flags
`ragged_pair_embed` / `pad_fill_zero` (`src/configs/model_config.py`). **Configs:**
`{pretrain_jepa,train_lorentz_part}_{ragged,fill0}.yaml`.

**Reproduce the figures (100k):**
```
python experiments/phase3/run_ragged_e2e.py --data-dir ./data --seeds 42 123 456   # → results/seed_*.json
python experiments/analyze_results.py --results-dir experiments/phase3/results --conditions stock ragged fill0
python experiments/phase3/plot_speedup.py --results-dir experiments/phase3/results
```

---

## Re-profile after ragged + TF32 (`--realistic-padding`, batch 1000)

Total self-CUDA time **3.43s → 1.24s (~2.76×)**; BN's share **48% → 16.6%** (~8× less BN in
absolute terms — the padding skip). TF32 confirmed live (`sm90 … tf32` / `cutlass tensorop`
GEMM kernels). The profile is now **balanced — no dominant op:**

| subsystem | self-CUDA % |
|---|---:|
| interaction-embed BN | 16.6 |
| LayerNorm (fwd+bwd) | 12.1 |
| attention (bmm+baddbmm+softmax) | ~11.5 |
| memory copies (`copy_`, incl. ragged gather/scatter) | 10.2 |
| dropout | 7.5 |
| GEMMs (`mm`) | 5.8 |

**Decision: stop optimizing, proceed to Phase 3.5.** No 40%-style hotspot remains — it's
diminishing returns. Optional single win if desired: **upper-triangle symmetry** on the pair
path (U is symmetric → process only `i<j`), which ~halves the remaining BN *and* the `copy_`
overhead (~10% of total) at low effort and accuracy-neutral. `torch.compile` (~1.3× on the
elementwise/LN/dropout pile) and a fused-attention kernel are higher-effort for a balanced
profile — future work, not now.

## Next step — Phase 3.5: 1M re-validation

On the finalized backbone (ragged + TF32): stock vs ragged at 1M → regenerate the accuracy +
speedup figures and the corrected Phase 2 numbers. This decides how much of Phase 2 (run on
the broken, slower backbone) needs redoing — likely the relative rankings hold but the
absolute numbers rise.
