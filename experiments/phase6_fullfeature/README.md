# Phase 6 — Full-information rerun (does SSL help when the model has everything?)

**Question.** Phases 2/4 found SSL (JEPA/MAE) gives **no finetune-accuracy advantage** over
scratch, and the input-ceiling test proved the 4-vector ceiling is **input-limited** (adding
4 displacement features took H→bb/cc from 0.79→0.976). Both were run on a 4-vector (or
4-vector + displacement) input. Phase 6 asks the clean follow-up: **give the model the full
information JetClass provides and re-run the SSL-vs-scratch comparison.** If the model is
capped by feature choice, a different training paradigm can't beat that cap — so we test it.

**Prediction on record.** scratch / JEPA / MAE rise together and **still tie**. The regime is
input-limited + label-abundant (free MC labels), so SSL's data-efficiency lever stays moot.
Phase 6 is the decisive falsification test: if SSL *did* open a gap in the feature-rich regime,
the "no advantage" thesis would need revising.

## What "every feature" means (and what it deliberately excludes)

JetClass ships 17 ParT inputs, but they are not 17 independent pieces of information for *our*
model. The engineered kinematics (Δη, Δφ, log pT, log E, log pT/pTjet, log E/Ejet, ΔR) are
**deterministic functions of the 4-vector the model already ingests** → zero added information.
The information beyond (px, py, pz, E) is:

| group | features | added information |
|---|---|---|
| **displacement** (4) | d0val, d0err, dzval, dzerr | track impact params → b/c flavor, vertexing |
| **PID** (6) | charge, isChargedHadron, isNeutralHadron, isPhoton, isElectron, isMuon | particle identity → lepton signatures, charged/neutral mix |

So the full-information input = **4-vector + 10 extra scalars** (`num_extra_features=10`), not a
padded 17-vector. This is the complete non-redundant information content of JetClass.

## Design decisions

1. **Extras ride the `num_extra_features` path** — concatenated into the encoder `proj`
   (16 → 128 becomes 26 → 128). The 4-vector still drives the Lorentz processor + interaction
   matrix, preserving the equivariance nudge; PID/displacement enter as scalars, exactly how
   ParT treats them.
2. **Displacement z-scored** (train stats, padded rows zeroed); **PID left raw**
   (charge ∈ {−1,0,1}, flags ∈ {0,1}; padded rows already zero) — ParT convention.
3. **MAE pretext stays 4-vector reconstruction** (conservation, E²=p²+m²). The encoder *sees*
   all features but reconstructs only the 4-momentum — extras are inputs, not targets — so MAE
   stays comparable to its 4-vector self. Reconstructing PID would be a different pretext = confound.
   **JEPA is latent → unchanged** (predicts target-encoder embeddings).
4. **Pretrain feature-rich**, then finetune feature-rich, so the *entire* encoder (incl. `proj`)
   transfers. A 4-vector pretrain → feature-rich finetune would silently drop `proj` via
   `strict=False` = partial-transfer confound.

## Files

- `configs/pretrain_jepa_full.yaml`, `configs/pretrain_mae_full.yaml` — the corrected-backbone
  (ragged, biased, K=1) 1M pretrain recipes + `num_extra_features: 10`.
- `run_phase6.py` — orchestrator: `pretrain` (JEPA + MAE per seed) then `downstream`
  (scratch/jepa/mae finetune per task × seed). Resumable, error-isolated. Reads
  `num_extra_features` from the data dir's `features.json`.
- Reuses `scripts/pretrain_{jepa,mae}.py`, `experiments/phase4/train_task.py`
  (`--num-extra-features`), `analyze_phase4.py`, and the `phase4/tasks.py` registry
  (added `full10` = all 10 classes, the Phase-2 headline analog).

## Run order

```bash
# 1. Build the feature-rich data once (writes features.json with num_extra_features=10):
python scripts/prepare_data.py --data-dir /path/to/val_5M --output-dir ./data_1m_full \
    --with-displacement --with-pid \
    --train-per-class 100000 --val-per-class 10000 --test-per-class 10000

# 2. Run the phase (pretrain + downstream, 3 seeds):
python experiments/phase6_fullfeature/run_phase6.py --data-dir ./data_1m_full --seeds 42 123 456

# 3. Aggregate:
python experiments/phase4/analyze_phase4.py --results-dir experiments/phase6_fullfeature/results
```

Default downstream tasks = `full10` (Phase-2 headline analog) + `hbb_hcc` + `wz` (the ceiling
anchors: displacement-recoverable vs mass-limited). Compare feature-rich finetune AUC/acc to
the 4-vector Phase-4 numbers — the tie (or its breaking) is the result.

## Not done here (scoped out; green-light before building)

- **Frozen probes on feature-rich encoders.** `LinearProbeModel` / `AttentivePoolProbeModel`
  don't yet thread `num_extra_features` into their encoder (proj shape + a 4-vector/extras split
  in their `forward`). Finetune is the decisive cut, so probes are deferred. To add: give both
  probe models a `num_extra_features` arg → encoder `proj` and a `x[...,:4] / x[...,4:]` split,
  matching `LorentzParT.forward`.
- The MAE conservation pretext still targets the 4-vector only (by design, decision 3). A
  variant that reconstructs displacement/PID is a *different* experiment, not this one.
