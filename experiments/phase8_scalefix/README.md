# Phase 8 — energy/momentum scale mismatch

**Status: DIAGNOSED, not fixed. No source file changed, nothing re-run.**

One root cause with two downstream consequences, found while auditing `processor.py`.

## Root cause

The dataset normalizes the 4-vector by dividing each component by its own mean
([`streaming_jetclass.py:119`](../../src/utils/data/streaming_jetclass.py#L119),
identically in [`jetclass.py:82`](../../src/utils/data/jetclass.py#L82)):

```
pT -> pT / 92.73        E -> E / 133.87
```

pT and E have the same dimension, so dividing them by *different* constants means energy and
momentum no longer share units downstream. Everything below follows from that one line.

## Consequence 1 — `ln_m2` is a dead constant

`m2 = energy_sum**2 - momentum_sum.norm()**2` ([`processor.py:41`](../../src/models/processor.py#L41))
becomes `(E_sum/muE)^2 - |p_sum|^2/muPT^2`, which is not the pair invariant mass. It is positive
only when `E_sum/|p_sum| > muE/muPT = 1.444`, i.e. for light particles an opening angle **> ~92 deg**.
Nothing inside an R=0.8 jet cone qualifies, so `m2 < 0` for every in-cone pair, `torch.clamp(m2, min=eps)`
silently floors it, and `ln_m2 = log(1e-8) = -18.42` always. The `isnan` guards never fire, because a
negative number is not NaN.

Measured on synthetic collimated jets through the real `ParticleProcessor`:

| feature | raw mean | normalized mean | std(diff) | % clamped |
|---|---|---|---|---|
| `ln_delta` | -0.976 | -0.976 | 0.000000 | 0% |
| `ln_kT` | -0.250 | -4.779 | 0.000000 | 0% |
| `ln_z` | -1.591 | -1.591 | 0.000000 | 0% |
| **`ln_m2`** | **+0.780** | **-18.421** | **2.06** | **100%** |

`ln_delta` and `ln_z` are exact (the scale cancels in `z = min_pT/pT_sum`), `ln_kT` takes a constant
offset that BatchNorm absorbs. Only `m2` is destroyed. So ParT's 4-feature interaction matrix has been
running on 3 features. Same failure class as the Phase-3 `-1e9` BatchNorm bug: a channel that looks
present but carries no information.

Note *which* feature died: `m2` is the pair mass, the feature encoding 2-prong/3-prong structure
(W/Z/H/top). Worth reading next to the `input_ceiling` finding that `wz` was the task nothing improved —
`wz` is a mass-discrimination task.

## Consequence 2 — `embed_vector` is not receiving a Lorentz vector

`embed_vector` does no validation; it assigns `multivector[..., 1:5] = vector` assuming `(E, px, py, pz)`.
It is handed `(pT, eta, phi, E)`, because the conversion at
[`processor.py:93-98`](../../src/models/processor.py#L93) is commented out — so `pT` lands in the time
slot and `E` in the z slot.

`EquiLinear` itself is fine: fed a genuine massive 4-vector it preserves the Lorentz invariant to 2.3e-7.
The input is the problem. Testing equivariance directly, `f(Lx) == L f(x)`:

| variant | long. boost y=0.6 | transv. boost b=0.4 |
|---|---|---|
| A — current `(pT,eta,phi,E)` | 3.4e-01 | 1.9e-01 |
| B — uncomment conversion only | 1.4e-01 | 1.2e-01 |
| C — uncomment + common scale | **3.4e-08** | **4.6e-08** |

**Uncommenting alone does not fix it** (variant B) — with pT and E on different scales the reassembled
`(E, px, py, pz)` is still not a 4-vector. Both changes are required, and they share the same root cause.

## What is actually new here

**Consequence 2 is not new.** Phase 5 already diagnosed both halves of it and fixed them in
`InvariantGATr` ([`verify_invariance.py:9-11`](../phase5/verify_invariance.py#L9)), which explicitly
lists Cartesian `embed_vector` and "a single global scale for pT and E ... instead of the per-feature
normalization (pT/92.7, E/133.9), which itself distorts the 4-vector" as its causes 2 and 3.
`equivariance_broken` also flags the mis-fed `embed_vector` as a "compounding suspect". **That fix was
never back-ported to production `processor.py`**, so LorentzParT still carries it — but the diagnosis
was already on record. Phase 8 only adds the quantification that uncommenting *alone* is insufficient
(variant B).

**Consequence 1 is new**, and it is the more consequential of the two, because Phase 5 was looking only
at the multivector/equivariance pathway and never checked the interaction features. `ln_m2` being dead:

- affects the **interaction matrix**, not the multivector pathway;
- therefore hits **vanilla ParT too** — every cell of the Phase 7 2x2, not just the Lorentz models;
- is invisible to the invariance tests, which is why Phase 5 missed it.

On the ~3e-3: `equivariance_broken`'s mechanism section already attributes it correctly to the
multivector pathway (under rotation the interaction features are exactly invariant, so the residual
must come from `EquiLinear` + `proj`). Phase 8 does not overturn that. The one wrinkle worth noting is
that because `ln_m2` is clamped dead, `U` is *more* invariant than it should be — the core looks
cleaner than it is.

## Honest limit of the fix

Applying both changes does **not** make the model Lorentz-invariant end to end (random init):

| transform | current | fixed |
|---|---|---|
| azimuthal rotation a=0.9 | 3.33e-03 | 8.05e-04 |
| longitudinal boost y=0.6 | 2.37e-03 | 2.22e-03 |
| transverse boost b=0.4 | 1.30e-03 | 9.83e-04 |

Better, but still ~1e-3, because `self.proj = nn.Linear(16, embed_dim)` after EquiLinear collapses the
geometric algebra into scalars regardless. The hybrid remains a hybrid. The value of this fix is
**physics correctness of `m2`** (large) and **giving EquiLinear a meaningful input** (moderate) — not
end-to-end invariance. Phase 8 should not be written up as "we made it Lorentz-invariant."

## Scope

`_get_interaction` is **byte-identical** between our `processor.py` and Thanh's — the only diff in that
file is the `pad_fill` arg and the appended `RaggedInteractionEmbedding`. The commented-out conversion
is present in `Initial Commit`. Both issues are **inherited**, and affect every phase run so far plus
the peer repo.

## The fix (not applied)

1. Give pT and E a common scale in the norm pipeline.
2. Restore the `(pT,eta,phi,E) -> (E,px,py,pz)` conversion before `embed_vector`.

## How to check

```bash
python experiments/phase8_scalefix/diagnose_scaling.py
```

Runs checks 1 and 2 with no data and no GPU. The fix is applied by monkeypatching
`ParticleProcessor.forward`, so running this commits us to nothing.

```bash
python experiments/phase8_scalefix/diagnose_scaling.py --data-dir <JetClass ROOT dir>
```

Adds check 3: the real-data `ln_m2` clamp rate, current vs fixed. **This is the one that still needs
running** — the 100% clamp rate above is from synthetic collimated jets, and should be confirmed on
real JetClass before acting.

```bash
python experiments/phase8_scalefix/diagnose_scaling.py --weights <ckpt.pt>
```

Runs check 2 against a trained checkpoint instead of random init.

## Open decisions

- Confirm the clamp rate on real data before changing anything.
- The in-flight 100M seed-42 run and all prior phase numbers are affected. Decide whether to finish
  the current scaling study on the known-broken features (internally consistent — every cell shares
  the same processor, so the 2x2 comparison is still apples-to-apples) or restart on corrected ones.
- Phase 5's `InvariantGATr` fix (Cartesian embed + single global scale) should be back-ported to
  production `processor.py`, not left in a side experiment.
