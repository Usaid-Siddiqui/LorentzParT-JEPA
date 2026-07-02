# Phase 4 — Downstream transfer & multi-task compute amortization

**Thesis (compute, not data-efficiency):** labels in HEP are ~free (Monte-Carlo
simulation); **compute is the bottleneck.** One JEPA pretrain amortizes across many
downstream jet tasks — each reached with less finetune compute than training from
scratch — so total compute savings **grow with the number of downstream tasks**.

## Protocols (three points per task, an accuracy-vs-compute trade)
Reuse the SAME 1M encoders (JEPA best-val, MAE, + scratch as the baseline):
- **Linear probe** (frozen encoder + linear head) — floor. Already have it for the
  10-class task (`LinearProbeModel`).
- **Attentive probe** (frozen encoder + small attention-pool head) — the frozen
  *ceiling*. This is I-JEPA's headline protocol precisely because linear probes
  undersell JEPA. NEW.
- **Full finetune** (encoder + head both train) — performance ceiling. Have it for
  the 10-class task (Phase 2).

Frozen protocols are the amortization headline (one encoder, N cheap heads); full
finetune is the per-task ceiling. Expect frozen < full, and note the frozen-linear
headline may favor MAE (0.749) over JEPA (0.711) while full finetune favors JEPA —
report both honestly.

## Tasks
1. **10-class classification** ("task 0") — start here: reuses existing encoders +
   data, and the attentive probe directly fills the gap between the known linear
   (0.711) and finetune (0.934) endpoints while validating the head.
2. **Binary taggings** (relabel existing `labels.npy`, no re-extraction):
   top-tag `{TTBar, TTBarLep}` vs QCD, W-tag `{WToQQ}` vs QCD, H→bb `{HToBB}` vs QCD.
3. **Jet-mass regression** — different task *type* (continuous target); needs the
   `prepare_data` extension to extract `jet_sdmass` from the ROOT files and re-produce
   the 1M/100k subsets. Strongest "the representation is general" evidence; gated on
   data-prep so it comes after the taggings.

## Metrics
- Classification: OVO/binary ROC AUC + **background rejection at fixed signal
  efficiency** (1/ε_B at ε_S=0.5 — the tagger metric HEP reviewers expect).
- Regression: resolution (RMS / IQR of predicted−true mass), and a response plot.
- Compute per task: finetune epochs + FLOPs (via `phase2/measure_flops.py` +
  `flops.json`), for each protocol.

## Headline figure — amortization
Cumulative training FLOPs vs. number of downstream tasks:
- **Scratch:** N independent full trainings (steep slope).
- **JEPA:** one pretrain (~28 PFLOPs best-val) + N cheap finetunes/probes (small
  offset, shallow slope).
Lines cross at ~1–2 tasks; JEPA's lead widens with N. Plot both frozen and
full-finetune variants.

## Build order
1. Attentive-probe head + runner; run on the 10-class task (JEPA/MAE/scratch).
2. Binary-tagging relabeling + run the three protocols per tagger.
3. `prepare_data` mass-regression extension → regression head → run.
4. Aggregate → amortization figure.

## Depends on
Phase 3 (faster backbone) is not a blocker but will cut every finetune/probe run here.
