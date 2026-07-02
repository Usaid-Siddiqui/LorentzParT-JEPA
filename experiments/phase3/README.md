# Phase 3 — CUDA / kernel optimization of the LorentzParT backbone

**Goal:** make the LorentzParT backbone meaningfully faster (fwd + bwd) by profiling
it, finding the real hotspot, and writing a fused kernel for it. A faster backbone
speeds up *every* subsequent run (pretrain + all Phase 4 finetunes), and a kernel is
a **de-risked deliverable** — it's valuable regardless of how the ML results land.

## Why this backbone
The equivariance comes from the external **`lgatr`** package, not hand-rolled code:
- `lgatr.layers.EquiLinear` — equivariant linear layers (used throughout
  `LorentzParTEncoder` and `src/models/lorentz_part.py`).
- `lgatr.interface.embed_vector` / `extract_vector` — 4-vector ↔ 16-dim multivector.
- the geometric-product-based equivariant attention inside `lgatr`.

These Clifford-algebra ops are **bilinear products over sparse structure-constant
tensors**, usually implemented as `einsum` — which lowers to many small unfused
matmuls/broadcasts. That's the classic profile-and-fuse target. **But we profile
first — do not assume the hotspot.**

## Plan (each step gates the next)
1. **Profile** the full `LorentzParT` fwd+bwd at the real training shape
   (batch 1000, 128 particles, embed 128 / 8L / 8H) on the H200.
   Tools: `torch.profiler` (op-level, CUDA time + memory), `nsys`/`ncu` for kernel
   detail. → verify: a ranked op table; identify the single dominant op(s).
2. **Isolate** the hotspot into a standalone microbenchmark (inputs + shapes matching
   the profile) with a correct reference (the current `lgatr` op). → verify: microbench
   reproduces the op's share of runtime.
3. **Write the kernel** (Triton first — portable, autotunable) fusing the hot op
   (candidate: geometric product / `EquiLinear`). Either shim it in locally or, if
   clean, prep an upstream `lgatr` contribution. → verify: **numerical parity** vs the
   reference (fwd + bwd grads within a tight `atol/rtol`) across random inputs.
4. **Benchmark** the kernel vs baseline: op-level speedup, then end-to-end
   epoch-time / FLOPs-throughput on a real pretrain step. → verify: measured wall-clock
   reduction with parity preserved.

## Success criteria
- Bit-close numerical parity (fwd output + parameter grads) with the `lgatr` op.
- A reported **op-level speedup** and **end-to-end training-step speedup** at the real
  training shape, on the H200.
- No accuracy regression on a short sanity pretrain/finetune (loss curve matches).

## Deliverables
- `profile_backbone.py` — profiling harness + ranked op report.
- `bench_<op>.py` — standalone microbenchmark (baseline vs kernel).
- `kernels/` — the Triton (or CUDA) kernel + a `torch.autograd.Function` wrapper.
- `test_parity.py` — parity test (fwd + bwd) gating any use of the kernel.
- results: profile tables + before/after benchmark numbers.

## Notes
- The hot op lives in the `lgatr` dependency, so the fix is either a local shim
  (monkeypatch / subclass) or an upstream PR — decide after profiling shows the target.
- Keep the kernel behind a flag so we can A/B and fall back to `lgatr` for correctness.
