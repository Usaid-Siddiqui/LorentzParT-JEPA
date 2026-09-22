"""
Phase 8 regression tests — one test per audited bug. CPU-only, no data, ~30s.

Each test FAILS on the pre-fix code and PASSES after. Run before any re-training:

    python experiments/phase8_scalefix/test_fixes.py
"""
import os
import sys
import contextlib
import io

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# ---------------------------------------------------------------- 1. interaction matrix alive
def test_interaction_alive():
    from src.models.processor import RaggedInteractionEmbedding
    from src.models import ParticleJEPA
    torch.manual_seed(0)
    B, N = 4, 128
    m = ParticleJEPA(ragged_pair_embed=True, encoder_type='lorentz')
    bn = m.context_encoder.interaction_embed.embed[0]
    seen = {}
    h = bn.register_forward_pre_hook(lambda mod, i: seen.update(
        frac=(i[0] < -1e8).float().mean().item()))
    x = torch.rand(B, N, 4) + 0.5
    x[:, 40:, :] = 0.0
    m.train()
    with torch.no_grad():
        m(x, torch.zeros(B, dtype=torch.long))
    h.remove()
    check("JEPA: no -1e9 reaches interaction BatchNorm", seen.get('frac', 1.0) == 0.0,
          f"fraction={seen.get('frac'):.4f}")

    # sensitivity: does changing valid-pair inputs still change the output?
    emb = RaggedInteractionEmbedding(4, [64, 64, 64, 8]).train()
    real = torch.zeros(B, N, dtype=torch.bool); real[:, :40] = True
    vp = real[:, :, None] & real[:, None, :]
    tv = vp.clone(); tv[:, 0, :] = False; tv[:, :, 0] = False
    Ua = torch.randn(B, N, N, 4) * 1.5
    Ub = Ua.clone(); Ub[tv] = torch.randn(int(tv.sum()), 4) * 1.5
    outs = []
    for U in (Ua, Ub):
        Uc = U.clone(); Uc[:, 0, :, :] = -1e9; Uc[:, :, 0, :] = -1e9
        vpair = vp.clone(); vpair[:, 0, :] = False; vpair[:, :, 0] = False   # energy-consistent
        o = emb(Uc, vpair).view(B, 8, N, N).permute(0, 2, 3, 1)
        outs.append(o[tv])
    rel = (outs[0] - outs[1]).abs().mean() / outs[0].abs().mean()
    check("Ragged embed stays input-sensitive when masked pairs excluded", rel > 0.1,
          f"relative Δ={rel*100:.1f}%")


# ---------------------------------------------------------------- 2. target encoder determinism
def test_target_deterministic():
    from src.models import ParticleJEPA
    torch.manual_seed(0)
    m = ParticleJEPA(ragged_pair_embed=True, encoder_type='lorentz')
    x = torch.rand(8, 128, 4) + 0.5; x[:, 40:, :] = 0.0
    mi = torch.zeros(8, dtype=torch.long)
    m.train()                                   # trainer calls .train() on the whole module
    with torch.no_grad():
        t1 = m(x, mi)[1]; t2 = m(x, mi)[1]
    cos = torch.nn.functional.cosine_similarity(
        t1.reshape(8, -1), t2.reshape(8, -1), dim=-1).mean().item()
    check("JEPA target encoder deterministic under model.train()", cos > 0.9999,
          f"cosine={cos:.4f}")


# ---------------------------------------------------------------- 3. EMA covers buffers
def test_ema_buffers():
    from src.models import ParticleJEPA
    torch.manual_seed(0)
    m = ParticleJEPA(ragged_pair_embed=True, encoder_type='lorentz')
    for b in m.context_encoder.buffers():
        if b.dtype.is_floating_point:
            b.add_(torch.randn_like(b) * 0.5)   # make context buffers differ
    before = [b.clone() for b in m.target_encoder.buffers() if b.dtype.is_floating_point]
    m.update_target_encoder(0.5)
    after = [b for b in m.target_encoder.buffers() if b.dtype.is_floating_point]
    moved = any(not torch.equal(a, b) for a, b in zip(before, after))
    check("EMA updates target-encoder buffers (BN stats)", moved,
          f"{len(before)} float buffers")


# ---------------------------------------------------------------- 4. predictor honours padding
def test_predictor_padding():
    from src.models.predictor import ParticlePredictor
    torch.manual_seed(0)
    p = ParticlePredictor(encoder_dim=128, predictor_dim=64).eval()
    B, N = 4, 128
    enc = torch.randn(B, N, 128)
    pad = torch.zeros(B, N); pad[:, 40:] = 1.0        # last 88 are padding
    mi = torch.zeros(B, 1, dtype=torch.long)
    import inspect
    takes_mask = 'padding_mask' in inspect.signature(p.forward).parameters
    if not takes_mask:
        check("Predictor accepts a padding mask", False, "forward() has no padding_mask arg")
        return
    with torch.no_grad():
        a = p(enc, mi, pad)
        enc2 = enc.clone(); enc2[:, 40:] = torch.randn(B, N - 40, 128) * 10   # perturb padding only
        b = p(enc2, mi, pad)
    d = (a - b).abs().max().item()
    check("Predictor output independent of padded positions", d < 1e-5, f"max Δ={d:.2e}")


# ---------------------------------------------------------------- 5. biased masking direction
def test_biased_masking():
    from src.utils.data.jetclass import NpyJetClassDataset as _ds
    import src.utils.data.jetclass as jc
    part = np.zeros((128, 4), dtype=np.float32)
    part[:60, 0] = np.sort(np.random.pareto(2., 60))[::-1] + 1.0
    part[:60, 3] = part[:60, 0] * 1.2
    obj = object.__new__(jc.NpyJetClassDataset)
    draws = []
    for _ in range(4000):
        _, _, mi = obj._mask_particle(part.copy(), 'biased', 1)
        draws.append(int(mi[0]))
    draws = np.array(draws)
    frac_top = float(np.mean(draws < 10))
    check("npy 'biased' masking favours high-pT (low index)", frac_top > 0.35,
          f"{frac_top*100:.1f}% in idx 0-9 (uniform would be ~17%)")


# ---------------------------------------------------------------- 6. norm stats correctness
def test_norm_stats():
    from src.utils.data.normalize import compute_norm_stats
    rng = np.random.default_rng(0)
    J, P = 500, 128
    pt = np.zeros((J, P)); eta = np.zeros((J, P)); phi = np.zeros((J, P)); E = np.zeros((J, P))
    for j in range(J):
        n = rng.integers(25, 60)
        v = np.sort(rng.pareto(2., n) * 8 + .5)[::-1]
        pt[j, :n] = v; eta[j, :n] = rng.normal(0, .9, n)
        phi[j, :n] = rng.uniform(-np.pi, np.pi, n); E[j, :n] = v * np.cosh(eta[j, :n])
    X = np.stack([pt, eta, phi, E], axis=1)
    real = pt > 0
    with contextlib.redirect_stdout(io.StringIO()):
        out = compute_norm_stats(X.copy())
    err = abs(out['pT'][0] - pt[real].mean()) / pt[real].mean()
    check("compute_norm_stats returns the true mean pT", err < 0.02,
          f"got {out['pT'][0]:.3f}, true {pt[real].mean():.3f} ({err*100:.1f}% off)")
    # and survives a feature count that does not divide the particle count
    X14 = np.concatenate([X, rng.normal(5, 1, (J, 10, P))], axis=1)
    with contextlib.redirect_stdout(io.StringIO()):
        o14 = compute_norm_stats(X14.copy())
    err14 = abs(o14['pT'][0] - pt[real].mean()) / pt[real].mean()
    check("compute_norm_stats correct with 14 features", err14 < 0.02,
          f"got {o14['pT'][0]:.3f}, true {pt[real].mean():.3f}")


# ---------------------------------------------------------------- 7. callback kwargs
def test_callback_kwargs():
    from src.utils.get_config import get_callbacks_from_config
    from src.utils.callbacks import CALLBACK_REGISTRY
    cbs = get_callbacks_from_config(
        [{'name': 'early_stopping', 'patience': 7, 'min_delta': 1e-6}], CALLBACK_REGISTRY)
    check("Callback params read from flat YAML keys", cbs[0].patience == 7,
          f"patience={cbs[0].patience} (want 7)")


# ---------------------------------------------------------------- 8. class-attention query norm
def test_cls_query_norm():
    from src.models.classifier import ClassAttentionBlock
    torch.manual_seed(0)
    blk = ClassAttentionBlock(embed_dim=32, num_heads=4).eval()
    seen = {}
    def hook(mod, args, kwargs):
        q = args[0] if args else kwargs['query']
        seen['q_std'] = q.std().item()
    blk.mha.register_forward_pre_hook(hook, with_kwargs=True)
    x = torch.randn(2, 16, 32) * 50 + 10          # deliberately un-normalised scale
    x_cls = torch.randn(2, 1, 32) * 50 + 10
    with torch.no_grad():
        blk(x, x_cls, torch.zeros(2, 16))
    check("Class-attention query is LayerNorm'd", seen.get('q_std', 99) < 3.0,
          f"query std={seen.get('q_std'):.2f} (raw input std ~50)")


# ---------------------------------------------------------------- 9. conservation loss knobs
def test_conservation_loss():
    from src.loss.conservation_loss import ConservationLoss
    l = ConservationLoss(alpha=0.3, reduction='mean')
    check("ConservationLoss stores alpha", getattr(l, 'alpha', None) == 0.3,
          f"alpha={getattr(l, 'alpha', None)}")


# ---------------------------------------------------------------- 10. lookahead checkpointing
def test_lookahead_state():
    from src.optim.lookahead import Lookahead
    p = torch.nn.Parameter(torch.randn(4))
    opt = Lookahead(torch.optim.SGD([p], lr=0.1), la_steps=3)
    sd = opt.state_dict()
    check("Lookahead state_dict carries slow weights", 'la_state' in sd or 'cached_params' in str(sd.keys()),
          f"keys={list(sd.keys())}")


# ---------------------------------------------------------------- 11. fp16 needs a scaler
def test_fp16_scaler():
    import src.engine.trainer as T
    src = open(T.__file__).read()
    check("fp16 path has a GradScaler (or is rejected)",
          'GradScaler' in src or "fp16" not in src or 'raise' in src.split('fp16')[1][:400],
          "no GradScaler found" if 'GradScaler' not in src else "present")


# ---------------------------------------------------------------- 12. 14-feature probe paths
def test_probe_14_features():
    from src.models.linear_probe import LinearProbeModel
    torch.manual_seed(0)
    try:
        m = LinearProbeModel(num_classes=10, num_extra_features=10)
        with torch.no_grad():
            out = m(torch.rand(2, 128, 14) + 0.5)
        ok = out.shape == (2, 10)
        detail = f"output {tuple(out.shape)}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {str(e)[:60]}"
    check("LinearProbeModel works with 14 features", ok, detail)


def main():
    tests = [
        ("interaction matrix liveness", test_interaction_alive),
        ("JEPA target determinism", test_target_deterministic),
        ("EMA buffer sync", test_ema_buffers),
        ("predictor padding", test_predictor_padding),
        ("biased masking direction", test_biased_masking),
        ("norm stats", test_norm_stats),
        ("callback kwargs", test_callback_kwargs),
        ("class-attention query norm", test_cls_query_norm),
        ("conservation loss knobs", test_conservation_loss),
        ("lookahead state", test_lookahead_state),
        ("fp16 scaler", test_fp16_scaler),
        ("14-feature probes", test_probe_14_features),
    ]
    for title, fn in tests:
        print(f"\n== {title} ==")
        try:
            fn()
        except Exception as e:
            check(title, False, f"{type(e).__name__}: {str(e)[:100]}")
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{'='*70}\n{n_pass}/{len(RESULTS)} checks passed")
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == '__main__':
    main()
