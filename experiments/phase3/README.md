# Phase 3 — CUDA / kernel optimization of the LorentzParT backbone

**Goal:** make the LorentzParT backbone meaningfully faster (fwd + bwd) by profiling
it, finding the real hotspot, and attacking it. A faster backbone speeds up *every*
later run (pretrain + all Phase 4 finetunes), and it's a de-risked deliverable —
valuable regardless of the ML results.

**Status (2026-07-02):** hotspot identified (pairwise interaction embedding, ~55% of
training compute — *not* the equivariant ops we assumed). Padding-aware "ragged"
interaction embedding built, gradcheck-verified, wired behind a flag, and **validated
100k end-to-end**. Result exceeded expectations: **2× faster AND +0.15 AUC** — because
the stock embedding has a `-1e9` BatchNorm-corruption bug that ragged fixes. **Open:**
decomposition control + 1M re-validation before paper-claiming the accuracy gain.

> This README is a running log — every change, why we made it, and script results.

---

## 🔑 Key finding — the hotspot is the pairwise interaction embedding

Profiling (not assumption) shows the cost is the **ParT-style pairwise
`InteractionEmbedding`** (`src/models/processor.py`): `BatchNorm1d → [Conv1d(k=1) →
BatchNorm1d → GELU] × 4` over the flattened `(B, C, N²)` pair tensor. The equivariant
`lgatr` ops (EquiLinear, geometric product) — which we *expected* to dominate — don't
even appear. BatchNorm alone is ~48% of training compute; the whole pair path ~55%.
And ~90% of the N² pairs are padding for typical jets.

**Optimization taken:** process only the **valid pairs** (ragged), skipping the ~90%
padding. Beats `torch.compile` fusion decisively on both time and memory (below).

---

## Progress log

### 2026-07-02 — `profile_backbone.py`: profiled the backbone (H200, batch 1000, fwd+bwd)
Built `LorentzParT` at the true training shape, ran fwd+bwd under `torch.profiler`,
ranked ops by self-CUDA time. **Result (top ops):**

| op | % CUDA | subsystem |
|---|---:|---|
| `cudnn_batch_norm_backward` | 35.3% | interaction embedding |
| `cudnn_batch_norm` (fwd) | 12.5% | interaction embedding |
| `mm` | 7.3% | linears (EquiLinear/dense) |
| `convolution_backward` | 4.6% | interaction embedding |
| `bmm` | 3.8% | attention |
| `cudnn_convolution` (fwd) | 2.4% | interaction embedding |

**Takeaway:** BatchNorm ≈ 48%, whole pair-interaction path ≈ 55%. Equivariant/`lgatr`
ops absent; attention only ~4%. This flipped the target away from the assumed
geometric-product kernel. Memory footprints (BN 124 GB, GELU 141 GB, conv-bwd 119 GB
summed) confirm it's memory-bandwidth-bound.
- Later fix: added `_short_label()` — raw CUDA kernel names are full C++ template
  signatures (e.g. `void cudnn::bn_bw_1C11_kernel_new<float, …>`) that overflowed the
  `--plot-out` y-axis; now shortened to ~44 chars (categorization still uses full names).

### 2026-07-02 — competitive analysis (context, not our code)
A peer GSoC repo (`github.com/omasho-codes/ml4sci_cms_e2e26`) optimized the *identical*
models. They independently confirmed this hotspot, but their fused pair-MLP kernel is
**inference-only** (BN folded from running stats) — at *training* time they run the
stock BN/conv MLP, and they never skip padded pairs. So the training-time pair-embed
cost and the padding waste are left open → **that's our differentiation** (padding-aware
training-time optimization + real gradcheck; they only checked gradient finiteness).
Reusable from them (with attribution): their `compile_patches.py` graph-break fix.

### 2026-07-02 — `bench_interaction.py`: A/B baseline vs torch.compile vs ragged (H200, batch 1000)
TRAIN mode (BN batch-stats + backward), ~12% valid pairs. **Result:**

| variant | fwd+bwd | speedup | peak mem |
|---|---:|---:|---:|
| baseline | 219.8 ms | 1.0× | 43.3 GB |
| torch.compile | 79.9 ms | 2.75× | 31.0 GB |
| **ragged (skip padding)** | **15.8 ms** | **13.9×** | **7.7 GB** |
| ragged + compile | 15.8 ms | 13.9× | 7.7 GB |

**Takeaways:** (1) `torch.compile` gives 2.75× for free (Inductor fuses the memory-bound
BN/conv/GELU chains). (2) Ragged gives **13.9× and 5.6× less memory** — ~5× more
impactful than compile, and it beats *proportional* scaling (12% valid ⇒ predicted 8×)
because baseline also wastes BN bandwidth on the `-1e9` padding. (3) `torch.compile`
**graph-breaks on the ragged path** (boolean-mask gather/scatter → data-dependent
`aten.nonzero`), so ragged+compile ≈ ragged — a production ragged path needs static
integer indexing or a custom kernel, not compile. **Caveat:** ragged is a *semantic*
change (BN over valid pairs only), so the speed number is only real once AUC is
re-validated.

### 2026-07-02 — `RaggedInteractionEmbedding` module + `test_ragged_gradcheck.py`
Added `RaggedInteractionEmbedding` to `src/models/processor.py`: gathers valid pairs via
static integer indexing (`index_select`), runs the **same** `BatchNorm1d + [Conv1d(k=1)
+ BatchNorm1d + GELU]` stack on the `(1, C, M)` gathered pairs, scatters back
out-of-place (`index_copy`, autograd-clean) to the same `(B·H, N, N)` shape.
- Uses **Conv1d (not Linear)** so its `embed` state_dict is byte-identical to the stock
  module's → a pretrained/stock checkpoint loads with **0 missing / 0 unexpected** keys
  (verified). This enables a same-weights ragged-vs-stock A/B and finetuning the existing
  1M encoder with ragged on.
- **gradcheck (float64) — both PASS:** w.r.t. `U` (train mode; exercises the risky
  gather/scatter data path) and w.r.t. a Conv1d weight (via `functional_call`, eval mode;
  confirms grads reach the MLP params through the scatter). Stronger than the peer's
  finiteness-only check.
- NOT an output-parity test vs stock — ragged is a deliberate semantic change, so its
  output differs by design (also sidesteps the stock module's `-1e9`-padding BN-stat
  corruption).

### 2026-07-02 — wired `ragged_pair_embed` flag into the model
- `LorentzParTConfig.ragged_pair_embed: bool = False` (`from_dict` filters unknown keys,
  so old YAMLs default to `False`).
- Threaded through `LorentzParT.__init__` → `LorentzParTEncoder`, which builds
  `RaggedInteractionEmbedding` vs `InteractionEmbedding` on the flag. In the encoder
  forward, `valid_pairs` is derived from the `padding_mask` it already has
  (`padding_mask == 0`; masked particles count as valid).
- **Smoke test:** stock checkpoint → ragged model loads `missing=0, unexpected=0`; output
  shapes match in both classification and MAE modes; outputs differ (intended semantic
  change); default flag is `False` (no behavior change for existing configs).

### 2026-07-02 — validation harness: `run_ragged_validation.py` + `train_lorentz_part_ragged.yaml`
The A/B that gates whether ragged ships. `configs/train_lorentz_part_ragged.yaml` is
`train_lorentz_part.yaml` + `ragged_pair_embed: true`. The runner finetunes stock vs
ragged from the SAME pretrained encoder across seeds, evaluates softmax OVO AUC, and
reports (1) AUC delta vs seed variance (does accuracy hold) and (2) median-epoch
speedup (the end-to-end training win). Eval builds each model from its own config so
the ragged checkpoint is scored with the ragged forward. Verified: stock config →
`InteractionEmbedding`, ragged config → `RaggedInteractionEmbedding`. (This is the
*1M finetune-only* confirmation for later — see the end-to-end harness below for the
primary gate.)

### 2026-07-02 — JEPA wiring + **100k end-to-end** validation harness (the primary gate)
Threaded `ragged_pair_embed` into `JEPAConfig` → `ParticleJEPA` → `LorentzParTEncoder`
(both context and target encoders; target is a deepcopy so it inherits ragged);
`pretrain_jepa.py` passes the flag. New `configs/pretrain_jepa_ragged.yaml`
(= pretrain_jepa.yaml + ragged flag). New `run_ragged_e2e.py`: per seed × {stock,
ragged}, **JEPA pretrain → finetune → eval**, reporting AUC delta + **pretrain AND
finetune** per-epoch speedups.
- **Why 100k end-to-end** (decided): a finetune-only A/B never tests ragged
  *pretraining* — where most compute lives and where the semantic change could break
  the JEPA pretext task. 100k is ~10× cheaper than 1M and accuracy-preservation is
  scale-robust, so it's the right gate; a 1M finetune-only run (existing
  `run_ragged_validation.py`, reusing the current 1M encoder) is the optional tight-eval
  follow-up.
- Verified: stock JEPA config → `InteractionEmbedding` (ctx+tgt), ragged →
  `RaggedInteractionEmbedding` (ctx+tgt).

### 2026-07-02 — 🔑 RESULT: ragged is 2× faster AND +0.15 AUC (fixes a BN-corruption bug)
100k end-to-end A/B (3 seeds, softmax OVO AUC), configs differ ONLY by the ragged flag:

| | AUC | acc | pretrain ep | finetune ep |
|---|---|---|---|---|
| stock | 0.7862 ± 0.0222 | ~0.31 | 41.5s | 29.1s |
| **ragged** | **0.9345 ± 0.0020** | ~0.64 | **21.2s (1.95×)** | **13.8s (2.10×)** |

Δ AUC **+0.148**, ragged also far tighter across seeds. Stock matches the established
Phase-0 100k baseline (~0.31 acc / ~0.78 AUC), so stock is *normal* and ragged is the
gain. **Root cause (verified from checkpoints):** the stock embedding fills padded
pairs with `-1e9` before its first `BatchNorm1d`, giving `running_mean≈-8.86e8,
running_var≈1.0e17`; every valid feature then normalizes to `(x+8.86e8)/3.17e8 ≈ 2.795`
regardless of `x` → the interaction bias carries ~zero information. Ragged normalizes
over valid pairs only (clean O(1) stats). So ragged = **2 separable wins**: skip-padding
speed (2×/5.6× mem) + a BN-correctness fix (+0.15 AUC).
- **Implication:** the bug predates us → all prior Phase 0/1/2 (and the peer repo) ran a
  partly-broken embedding; relative rankings likely hold, absolute numbers suppressed.
- **Controls still owed:** (1) the fill0 decomposition (below); (2) 1M re-validation.

### 2026-07-02 — decomposition control built (`fill0`): isolate the BN fix from ragged speed
Added a `pad_fill_zero` flag (`ParticleProcessor.pad_fill`, threaded through
`LorentzParTConfig`/`JEPAConfig` -> `LorentzParT`/`ParticleJEPA` -> `pretrain_jepa.py`):
fills padded pairs with `0.0` instead of `-1e9`, so the stock (dense) BatchNorm stats
aren't corrupted -- the BN fix WITHOUT the ragged reorg. New configs
`{pretrain_jepa,train_lorentz_part}_fill0.yaml`, and `fill0` added as a third condition
in `run_ragged_e2e.py`. The runner now prints a 3-way table + a `DECOMP:` line reporting
what fraction of ragged's AUC gain `fill0` recovers:
- **fill0 ~ ragged** => the +0.15 is *purely* the BN-corruption fix (ragged's speed is
  orthogonal) -- the clean, separable story.
- fill0 between stock and ragged => ragged's valid-only normalization adds something the
  0-fill approximation doesn't (0-fill still dilutes BN stats with the padded zeros).
Verified builds: stock/fill0 -> `InteractionEmbedding` (pad_fill -1e9 / 0.0), ragged ->
`RaggedInteractionEmbedding`. **Run pending** (re-run the same e2e command; stock+ragged
checkpoints are skipped, only the 6 new fill0 stages train).

---

## Files

| file | what it does | run |
|---|---|---|
| `profile_backbone.py` | profile LorentzParT fwd+bwd → ranked op table + `--plot-out` hotspot chart | `python experiments/phase3/profile_backbone.py --batch 1000 --iters 10 --plot-out experiments/phase3/hotspots_gpu.png` |
| `bench_interaction.py` | A/B baseline / torch.compile / ragged on the interaction embedding (uses the production `RaggedInteractionEmbedding`) | `python experiments/phase3/bench_interaction.py --batch 1000 --plot-out experiments/phase3/interaction_bench.png` |
| `test_ragged_gradcheck.py` | float64 gradcheck of `RaggedInteractionEmbedding` (w.r.t. input + params) | `python experiments/phase3/test_ragged_gradcheck.py` |
| `run_ragged_e2e.py` | end-to-end 100k A/B, 3 conditions {stock, ragged, fill0}: AUC + speedup + BN-decomp readout | `python experiments/phase3/run_ragged_e2e.py --data-dir ./data --seeds 42 123 456` |
| `run_ragged_validation.py` | 1M finetune-only follow-up: AUC preservation + epoch speedup (reuses existing encoder) | `python experiments/phase3/run_ragged_validation.py --weights <encoder.pt> --data-dir ./data_1m --seeds 42 123 456` |
| `hotspots_gpu.png` | profiling hotspot chart | — |

Model changes live in `src/models/processor.py` (`RaggedInteractionEmbedding`),
`src/models/lorentz_part.py` (encoder wiring), `src/configs/model_config.py` (flag).

---

## Open questions / next steps

1. **Train-validation (the gate).** Finetune the existing 1M encoder (or a short pretrain)
   with `ragged_pair_embed: true` vs `false`, confirm **AUC holds within seed variance**,
   and check whether fixing the `-1e9` BN-stat corruption nudges it. Only then is the
   13.9× a real optimization rather than a faster op.
2. End-to-end training speedup: measure whole-model epoch time / peak memory with the flag
   on (the 55% pair-embed cost should translate to a large end-to-end win).
3. If ragged validates, optionally also raggedify the pairwise *feature* computation
   (`_get_interaction`) so the O(N²) feature build also skips padding.
4. Free win to stack independently: `torch.compile` (+ the peer's compile-patch technique
   for the `lgatr` graph breaks) on the rest of the model.
