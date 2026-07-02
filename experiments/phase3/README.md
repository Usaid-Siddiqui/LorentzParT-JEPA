# Phase 3 — CUDA / kernel optimization of the LorentzParT backbone

**Goal:** make the LorentzParT backbone meaningfully faster (fwd + bwd) by profiling
it, finding the real hotspot, and attacking it. A faster backbone speeds up *every*
later run (pretrain + all Phase 4 finetunes), and it's a de-risked deliverable —
valuable regardless of the ML results.

**Status (2026-07-02):** hotspot identified (pairwise interaction embedding, ~55% of
training compute — *not* the equivariant ops we assumed). Padding-aware "ragged"
interaction embedding built, gradcheck-verified, and wired into the model behind a
flag. **Open:** train-validation that AUC survives the semantic change.

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
  `RaggedInteractionEmbedding` (ctx+tgt). **Run pending (H200).**

---

## Files

| file | what it does | run |
|---|---|---|
| `profile_backbone.py` | profile LorentzParT fwd+bwd → ranked op table + `--plot-out` hotspot chart | `python experiments/phase3/profile_backbone.py --batch 1000 --iters 10 --plot-out experiments/phase3/hotspots_gpu.png` |
| `bench_interaction.py` | A/B baseline / torch.compile / ragged on the interaction embedding (uses the production `RaggedInteractionEmbedding`) | `python experiments/phase3/bench_interaction.py --batch 1000 --plot-out experiments/phase3/interaction_bench.png` |
| `test_ragged_gradcheck.py` | float64 gradcheck of `RaggedInteractionEmbedding` (w.r.t. input + params) | `python experiments/phase3/test_ragged_gradcheck.py` |
| `run_ragged_e2e.py` | **primary gate:** end-to-end 100k A/B (pretrain→finetune→eval), AUC + pretrain & finetune speedup | `python experiments/phase3/run_ragged_e2e.py --data-dir ./data --seeds 42 123 456` |
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
