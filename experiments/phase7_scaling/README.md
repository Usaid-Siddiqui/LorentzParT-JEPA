# Phase 7 — Scaling to full JetClass (100M): the headline experiment

**Goal.** A fair, param-matched scaling study of equivariance × self-supervision on the full
JetClass benchmark, with published Particle Transformer (ParT) as the external anchor. The paper's
headline is decided by *how the four models scale with data* up to 100M jets.

> Status: PLAN + scaffold (2026-09-04). Nothing here has run. Grounded in measured Phase-6
> throughput and a real param audit (§2).
>
> **Decisions resolved (2026-09-04):** hardware = **1 node, 2 GPUs** (target 2×H100, fallback
> 2×A100-80GB); headline = **model-accuracy comparison at 100M** (the four-cell endpoint table +
> published-ParT anchor; a light 10M/1M scaling trend for context, not a full multi-seed curve);
> data pipeline = **weaver** (reader only — see §4); features = **full 14** (4-vector + displacement
> + PID). Label-efficiency axis is dropped for now.

---

## 1. Scientific design — a 2×2 × data-scale

A clean **2×2** isolates the two levers, benchmarked against published ParT:

| | **scratch** | **+ JEPA pretrain** |
|---|---|---|
| **ParT** (vanilla) | ParT-scratch (reproduces published → pipeline sanity + control) | ParT + JEPA |
| **LorentzParT** (hybrid, equivariance nudge) | LorentzParT-scratch | LorentzParT + JEPA |

- **Row effect** = the equivariance nudge (LorentzParT vs ParT).
- **Column effect** = self-supervision (JEPA vs scratch).
- **External anchor** = published ParT-full on JetClass (acc 0.861 / AUC 0.9877); our ParT-scratch
  must reproduce it, validating the whole pipeline.

**Primary axis — data scaling (full supervision):** train every cell at
{100k, 1M, 10M, 100M} training jets; plot accuracy/AUC vs data size, one line per model. Two
pre-registered hypotheses, both consistent with our prior findings and both publishable either way:
1. **Equivariance helps most in the low-data regime and washes out by 100M** (equivariant inductive
   bias substitutes for data; scale substitutes for the bias). This is the classic, testable
   scaling story for physics-informed architectures.
2. **JEPA adds no accuracy at any scale** under full supervision (our input-limited + label-abundant
   thesis, now tested at 100× the scale). A null here at 100M is a strong, defensible result.

**Optional second axis — label efficiency (the SSL money plot).** Pretrain JEPA once on the full
100M (labels ignored), then finetune on {0.1%, 1%, 10%, 100%} labels vs scratch at the same budgets.
This is the *one* regime our thesis predicts SSL should win (label-scarce), so it closes the loop:
"SSL doesn't help when labels are abundant (Phases 2/4/6) — does it, at full pretraining scale, when
they're scarce?" **Recommended as a stretch goal** — it is the highest-upside result but adds compute
(§6) and depends on a positive framing. See §9 for the decision.

---

## 2. Model parity (measured, not estimated)

Param counts at the shared config (embed_dim 128, 8 heads, 8 encoder layers, 2 class-attn layers):

| model | total | **encoder** | classification-relevant |
|---|---|---|---|
| ParT (vanilla) | 2.073 M | **1.607 M** | ~2.01 M |
| LorentzParT (hybrid) | 2.271 M | **1.608 M** | ~2.01 M |
| published ParT-full | ~2.14 M | — | — |

**The encoders are within 0.1%** — the L-GATr `EquiLinear` nudge is nearly free in parameters, so
parity is automatic at equal depth/width/heads. The ~0.2M total gap is LorentzParT's unused MAE
reconstruction head (`fc`), irrelevant to classification. **No width tuning is needed for a fair
comparison** — matched (embed_dim, layers, heads) *is* matched params here. Keep all four cells at
this exact config; report the param table in the paper.

---

## 3. Matching ParT for a fair benchmark

To make our ParT-scratch reproduce the published number (and so anchor the 2×2), match ParT's setup:

- **Input features.** Published ParT-full uses **17** per-particle features (7 kinematic + 6 PID +
  4 trajectory-displacement). Our pipeline already extracts the informative 14 (4-vector + 10
  extras = displacement + PID); the remaining 3 published "kinematic" features (Δη, Δφ, ΔR-type) are
  deterministic functions of the 4-vector our processor already ingests, so **14 of ours ≡ 17 of
  theirs in information**. Decision (§9): feed all four cells the full 14-feature input for a true
  benchmark, or run a controlled 4-vector-only comparison and cite published ParT as reference only.
  Recommendation: **full 14 features** so ParT-scratch can actually hit ~0.861 and validate the pipe.
- **Interaction features.** Identical already: ln Δ, ln kT, ln z, ln m² → the same 3-layer pointwise
  embedding → attention bias (`InteractionEmbedding` / `RaggedInteractionEmbedding`). Use the ragged
  (padding-aware) variant everywhere — the Phase-3 fix (+0.15 AUC, 2× faster).
- **Blocks.** 8 particle-attention + 2 class-attention, embed 128, 8 heads, FFN×4 — already matches.
- **Training recipe.** Pull the exact optimizer/LR/batch/epoch schedule from the official
  `weaver-core` JetClass ParT config and mirror it (they use RAdam+Lookahead ≈ our "ranger"-style
  setup already; lr 1e-3, cosine). Matching the recipe is as important as the architecture for the
  benchmark to be credible. **Action:** copy the published `data/JetClass/*.yaml` + training flags.

---

## 4. Dataset — download, storage, processing

**Source.** Full JetClass: Zenodo record `6619768` (`JetClass_Pythia_train_100M.tar`, `val_5M`,
`test_20M`), or the HuggingFace mirror `jet-universe/JetClass`. Train = 100M jets (10 classes ×
10M), val = 5M, test = 20M.

**Storage budget.** Compressed ROOT train set is order **several hundred GB** (verify exact size on
Zenodo before committing disk). Budget **≥ 1 TB** working disk for raw ROOT + working space. Do
**not** attempt to densify to `.npy`: 100M × 14 × 128 × float32 ≈ **870 GB per copy** and cripples
I/O — the in-memory `NpyJetClassDataset` path used at 100k/1M is dead at ≥10M.

**Processing — three options, in order of fairness/effort:**
1. **Reuse weaver's ROOT dataloader (recommended for parity).** The official ParT uses
   `weaver-core`, which already streams ROOT with correct per-class balancing, preprocessing, and
   feature standardization matching the published results. Wrap it to feed our trainers. Highest
   fairness, least reinvention; cost = a weaver dependency + an adapter.
2. **Our streaming `IterableDataset` (`src/utils/data/streaming_jetclass.py`, scaffolded).** Streams
   ROOT shards with DDP+worker file-sharding, round-robin across the 10 class-file streams for
   balanced batches, and a shuffle buffer. No conversion, no extra storage. Cost = we own a subtle
   data pipeline; **must be validated against `val_5M` before trusting** (untested — no local data).
3. **One-time convert to sharded float16 (webdataset/HDF5).** ~435 GB for 14 feat; fastest repeated
   epochs, clean DDP sharding. Cost = conversion time + storage. Worth it only if we do many epochs.

**CHOSEN: weaver (option 1), as a READER ONLY.** Critical integration note: our
`ParticleProcessor` recomputes the interaction matrix and the L-GATr multivector from the **raw**
4-vector (pt, eta, phi, E), so configure weaver's data YAML to emit the **raw branches** we need
(the 4-vector + the 10 extra scalars), NOT weaver's default pre-engineered/normalized ParT features.
Weaver gives us its battle-tested **balanced streaming ROOT reader**; our processor + models do the
physics unchanged. Deliverables: a weaver data YAML (raw 14-branch), and a thin adapter mapping
weaver's batch dict → our `(B, N, 14)` tensor + one-hot label, feeding `JEPATrainer` /
`JetClassTrainer`. This supersedes the custom `streaming_jetclass.py` (kept as a no-weaver fallback).

**Class balance.** JetClass files are single-class. Balanced batches require interleaving class
streams (weaver does this; our loader round-robins across the 10 class groups). A naive
one-file-at-a-time stream yields class-homogeneous batches — a correctness bug, not just a nicety.

---

## 5. Training organization (multi-GPU / multi-node)

- **Parallelism.** DDP data-parallel (the model is 2M params — no model/tensor parallel needed).
  Near-linear scaling to a node of 8 GPUs; multi-node via `torchrun --nnodes`. Our trainers already
  do single-node DDP (`setup_ddp`, `JetClassDistributedSampler`) — extend the sampler/loader to the
  streaming/IterableDataset path (§7 M3) and add `torchrun`/SLURM launch wrappers.
- **Global batch.** Keep effective batch comparable to ParT (~512–1024). With DDP, per-GPU batch ×
  #GPUs = global; scale LR with global batch (linear or sqrt rule) and add warmup.
- **Precision.** TF32 already on; add **bf16 autocast** for the 100M runs (throughput, memory) — a
  small change, big at scale. Keep fp32 master weights.
- **Checkpointing & resumption.** 100M runs span days — checkpoint every N steps (not just per
  epoch), and make runs resumable (the trainers already checkpoint per epoch; add step-level for the
  big runs). Snapshot to durable storage; assume preemption.
- **Determinism / seeds.** 1 seed at 100M (cost), 3 seeds at ≤ 10M for error bars. Fixed data
  sharding seed so all four cells at a scale see the same jets.

---

## 6. Compute budget (grounded in measured throughput)

Measured Phase-6 throughput (their GPU, ~2M-param model, batch 1000, TF32): **~350 s per 1M-jet
epoch**. Linear in jets → **~9.7 hr per 100M-jet epoch on one GPU**. Assumed epochs: supervised 10,
JEPA pretrain 8, JEPA finetune 10 (tune down with early-stop — likely fewer suffice at 100M).

**100M points (1 seed), single-GPU-equivalent:**

| cell | work | GPU-hours |
|---|---|---|
| ParT-scratch | 10 ep supervised | ~97 |
| LorentzParT-scratch | 10 ep supervised | ~97 |
| ParT + JEPA | 8 ep pretrain + 10 ep finetune | ~175 |
| LorentzParT + JEPA | 8 ep pretrain + 10 ep finetune | ~175 |
| **100M subtotal** | | **~544 GPU-hr ≈ 23 GPU-days** |

Smaller scales: 10M ≈ 0.1× (×3 seeds ≈ 7 GPU-days), 1M/100k ≈ negligible (~1 GPU-day). Optional
label-efficiency axis reuses the 100M JEPA pretrains (already paid) + cheap small-label finetunes
(~1–2 GPU-days).

**On the chosen 2-GPU node** (DDP ~linear → 2× the single-GPU throughput above). Four 100M cells =
2×10 (scratch) + 2×18 (JEPA pre+ft) = **56 epochs @ 100M**:

| scenario | 100M epoch | four 100M cells (1 seed) |
|---|---|---|
| 2×A100-80GB, measured-baseline throughput | ~4.9 hr | ~11 days |
| 2×A100, reduced epochs (scratch 6 / JEPA 5+6) | ~4.9 hr | ~7 days |
| 2×H100 + bf16 (~1.5–2× A100) | ~2.5–3.5 hr | **~4–7 days** |

Add ~0.5 day for a light 10M/1M scaling trend. **So the 100M headline is ~1 week on 2×H100,
~1.5 weeks on 2×A100** — feasible, but the single biggest lever is per-epoch throughput, which
depends on the *actual* GPU and bf16.

⚠️ **Calibrate first.** The ~350 s/1M-epoch baseline is from an unknown Phase-6 GPU. Before
budgeting, run **one 1M-jet epoch on the real 2×H100 with bf16** and re-extrapolate — the estimate
could easily halve. Also tune the 100M epoch count down (10 is likely more than needed; early-stop
on val plateau governs).

**De-risking order (do NOT start at 100M):** 100k → 1M → 10M → 100M. Validate the weaver reader +
2×2 trends at 10M (a few GPU-hours), confirm **ParT-scratch tracks published ~0.861** there, then
spend the 100M budget. Given 2 GPUs, keep multi-seed error bars at ≤10M; run **1 seed at 100M**.

---

## 7. Required code changes before launch (the real work)

| id | change | why | status |
|---|---|---|---|
| **M1** | `ParticleTransformer` + `num_extra_features` + `ragged_pair_embed` (proj `Linear(4+E,…)`, 4-vec/extras split, padding-aware embedding) | vanilla ParT ingests the full 14 features; both backbones share the ragged embedding | ✅ **DONE + smoke-tested** |
| **M2** | `ParticleJEPA` **encoder-pluggable** via `encoder_type='part'\|'lorentz'` (part → `ParticleTransformerEncoder`, `to_multivector=False`, gate over the 4-vec) | build **ParT + JEPA** | ✅ **DONE + smoke-tested** |
| **M3** | streaming loader (own, not weaver) + `IterableDataset` trainer branch + step-epochs + bf16 autocast + capped streaming val; resumability via per-(step-)epoch checkpoints | any run ≥ 10M | ✅ **DONE** (loader validated on val_5M; trainer plumbing CPU-tested) |
| **M4** | `train_phase7.py` (torchrun entry, all 4 cells), 2 train-recipe configs, `run_phase7.py` orchestrator (JEPA two-stage, resumable) | orchestrate the 2×2 sweep | ✅ **DONE** (syntax + config + mask-contract tested) |

All of M1–M4 are code-complete and verified as far as this no-GPU/no-ROOT box allows. **Cluster-only
validations remaining before the headline runs:** (a) bf16 on real CUDA, (b) actual 2-GPU DDP,
(c) JEPA-streaming end-to-end on real ROOT, (d) the throughput calibration (§6). Architecture parity
holds at **0.10%** (ParT 1.608M vs LorentzParT 1.609M encoders, full features).

Design notes: streaming uses **step-based epochs** (`steps_per_epoch`) — a full 100M pass is ~5 hr,
so an "epoch" is a fixed number of optimizer steps; this makes the `len()`-less `IterableDataset`
work, bounds checkpoint/validation cadence, and makes the existing per-epoch checkpointing double as
**resumability** (resume restarts the current step-epoch; each reshuffles fresh data, so nothing is
lost). Map-style datasets (phases 0–6) pass `steps_per_epoch=None` → behavior byte-identical.

---

## 8. Experiment matrix, naming, orchestration

- Cells: `{part, lorentzpart} × {scratch, jepa}` × scale `{100k, 1m, 10m, 100m}` × seed.
- Run-name convention: `<model>_<protocol>_<scale>_seed<N>` (e.g. `lorentzpart_jepa_100m_seed42`).
- JEPA encoders reused across the finetune + label-efficiency axes (pretrain once per scale/seed).
- Orchestrator: mirror `run_phase6.py` (resumable, error-isolated, `--stage pretrain|downstream`,
  reads scale + feature count from a data manifest) → `experiments/phase7_scaling/run_phase7.py`.
- Analysis: reuse `analyze_phase4.py` (tables), `plot_curves.py` (convergence),
  `pretrain_compute.py` + `compute_to_threshold.py` (compute), plus a **new scaling-curve plot**
  (accuracy vs data size, 4 lines + published-ParT reference).

## 9. Decisions needed before building (see the questions accompanying this plan)

1. **Compute/hardware available** (GPU count/type, cluster scheduler, disk) — gates feasibility,
   time, and the data-pipeline choice.
2. **Headline axis:** data-scaling only (primary) vs + label-efficiency (stretch money-plot).
3. **Data pipeline:** weaver (max fairness) vs our streaming loader (native integration).
4. **Feature set:** full 14 (benchmark-comparable to published ParT) vs 4-vector-only (controlled,
   cites ParT as reference).

## 10. Risks / intricacies

- **Data-loader class balance** — the #1 correctness risk (§4); validate empirically on `val_5M`.
- **JEPA at 100× scale** — EMA momentum and LR schedules are step-count-based; retune for the far
  larger step count (Phase-2 lesson: too-fast EMA target → divergence). Monitor representation
  collapse (effective rank, `embedding_stats.py`) during the long pretrain.
- **Benchmark credibility** — if ParT-scratch doesn't reproduce ~0.861, the pipeline (features,
  preprocessing, recipe) is off; fix before trusting the 2×2. This is why weaver parity matters.
- **Cost control** — always climb the scale ladder; never launch 100M without the 10M trends in hand.
- **Reproducibility** — fixed data-shard seed across cells; log exact feature set, recipe, and
  per-step throughput for the paper's compute table.

## 11. Dual-GPU workflow (2×H100 / 2×A100-80GB)

Each cell runs under `torchrun` on one node; the streaming loader shards internally by
(rank × worker), so no `DistributedSampler` and identical code in 1- vs 2-GPU. Global batch =
`batch_size` × 2. LR/`steps_per_epoch` are set for the *global* batch (scale LR if you change GPU count).

**Step 0 — data** (once): `scripts/download_jetclass.py --split train/val`. Make the scale ladder as
subdirectories of ROOT files (100k/1M/10M = fewer files per class; 100M = all).

**Step 1 — validate the loader** on the real val set (done once):
`python -m src.utils.data.streaming_jetclass /path/to/val_5M` → expect ~5000/class.

**Step 2 — calibrate throughput** (before budgeting 100M): one short run on 2 GPUs at 1M, read the
per-epoch seconds from the CSV, extrapolate ×100.
```
torchrun --standalone --nproc_per_node=2 experiments/phase7_scaling/train_phase7.py \
  --model lorentzpart --protocol scratch --train-dir DATA/train_1M --val-dir DATA/val_5M \
  --config experiments/phase7_scaling/configs/phase7_supervised.yaml \
  --run-name calib_lorentzpart_1m --num-epochs 2 --steps-per-epoch 500 --seed 42
```

**Step 3 — climb the ladder** with the orchestrator (runs all 4 cells; JEPA cells auto pretrain→finetune):
```
# small scales for trends + error bars (cheap, 3 seeds), confirm ParT-scratch tracks published:
python experiments/phase7_scaling/run_phase7.py --train-dir DATA/train_10M --val-dir DATA/val_5M \
  --scale 10m --seeds 42 123 456 --nproc 2 --steps-per-epoch 6000 --num-epochs 15
# headline endpoint (1 seed, more steps/epoch, fewer epochs):
python experiments/phase7_scaling/run_phase7.py --train-dir DATA/train_100M --val-dir DATA/val_5M \
  --scale 100m --seeds 42 --nproc 2 --steps-per-epoch 20000 --num-epochs 10
```

**Resumability / preemption:** every step-epoch writes a checkpoint
(`logs/<Model>/checkpoints/<run>.pt`). To resume a killed run, the trainer's `load_checkpoint`
restores model/opt/scheduler/epoch; re-invoking the same `run-name` cell is skipped once its `best/`
checkpoint exists, so `run_phase7.py` is safe to re-run after a crash. Keep `logs/` on durable storage.

**Monitoring:** per-epoch CSVs in `logs/<Model>/logging/<run>.csv` (val_metric / val_loss /
elapsed_total_s). Reuse `experiments/plot_curves.py` for convergence and a per-scale accuracy table
for the headline figure.
