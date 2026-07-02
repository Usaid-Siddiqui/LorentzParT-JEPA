"""
Phase 3, step 2 — microbenchmark the pairwise interaction-embedding hotspot.

Our GPU profile put ~48% of TRAINING compute in this module's BatchNorm (the
`BatchNorm1d -> [Conv1d(k=1) -> BatchNorm1d -> GELU] x4` MLP over the flattened
(B, C, N*N) pair tensor), and ~90% of those N*N pairs are padding for typical
jets. This A/Bs three ways of running it, in TRAIN mode (BN batch stats + backward
— the regime that matters for us), at the real training shape:

  baseline        stock InteractionEmbedding, eager.
  compile         torch.compile(baseline) — what Inductor fuses for free (the
                  memory-bound BN/conv/GELU elementwise chains).
  ragged          padding-aware: gather only VALID pairs -> (M, 4), run the MLP
                  on M << B*N*N rows, scatter back. Skips the ~90% padded pairs.
                  NOTE this is an ALGORITHMIC change, not a drop-in: BatchNorm
                  now normalizes over valid pairs only (which also sidesteps the
                  stock module's -1e9-padding BN-stat corruption). It is NOT
                  expected to match baseline numerically — it needs accuracy
                  re-validation, so this script measures speed/memory only.

Reports mean fwd and fwd+bwd time and peak memory for each, with speedups.
Run ON THE H200 (a GPU number is the only meaningful one).

    python experiments/phase3/bench_interaction.py --batch 1000
    # local smoke (CPU): --batch 16 --device cpu --no-compile
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.processor import ParticleProcessor, InteractionEmbedding


def make_inputs(batch, n_particles, device, seed=0):
    """Realistic padded jets -> the stock -1e9-filled U (B,N,N,4) and the valid-pair
    mask. Per-jet valid-particle count ~ N(40, 18) clamped to [5, N] (median ~40 ->
    ~90% of the 128x128 pairs are padding), matching the 100k data distribution."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    n_valid = torch.normal(40.0, 18.0, (batch,), generator=g).round().clamp(5, n_particles).long()
    idx = torch.arange(n_particles)
    mask = (idx[None, :] < n_valid[:, None]).to(device)          # (B, N) bool, valid particles

    x = torch.randn(batch, n_particles, 4, generator=g).to(device) * mask[..., None]
    x[..., 3] = torch.where(mask, x[..., 3].abs() + 0.5, torch.zeros_like(x[..., 3]))  # energy>0 iff valid

    proc = ParticleProcessor(to_multivector=False).to(device)
    _, U = proc(x)                                                # U: (B, N, N, 4), padded pairs = -1e9
    valid_pairs = mask[:, :, None] & mask[:, None, :]             # (B, N, N) bool
    return U, valid_pairs


class RaggedInteractionEmbedding(nn.Module):
    """Padding-aware equivalent of InteractionEmbedding: run the same
    BN -> [Linear -> BN -> GELU]*L stack on only the valid pairs. Conv1d(k=1) over
    (B,C,L) is per-position Linear over channels, so this is the same MLP structure;
    the difference is it processes M valid pairs instead of B*N*N."""

    def __init__(self, num_features=4, pair_embed_dims=(64, 64, 64, 8)):
        super().__init__()
        self.in_bn = nn.BatchNorm1d(num_features)
        layers, inp = [], num_features
        for d in pair_embed_dims:
            layers += [nn.Linear(inp, d), nn.BatchNorm1d(d), nn.GELU()]
            inp = d
        self.mlp = nn.Sequential(*layers)
        self.out_dim = pair_embed_dims[-1]

    def forward(self, U, valid_pairs):
        B, N, _, _ = U.shape
        sel = U[valid_pairs]                      # (M, 4) — gather valid pairs only
        h = self.mlp(self.in_bn(sel))             # (M, H)
        out = U.new_zeros(B, N, N, self.out_dim)
        out[valid_pairs] = h                      # scatter back
        return out.permute(0, 3, 1, 2).reshape(B * self.out_dim, N, N)


def bench(run, iters, warmup, cuda):
    """Mean ms/iter and peak MB for `run` (already closes over fwd or fwd+bwd)."""
    for _ in range(warmup):
        run()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    if cuda:
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    mem = torch.cuda.max_memory_allocated() / 1e6 if cuda else float('nan')
    return ms, mem


def main():
    p = argparse.ArgumentParser(description="Benchmark the interaction-embedding hotspot")
    p.add_argument('--batch', type=int, default=1000)
    p.add_argument('--particles', type=int, default=128)
    p.add_argument('--dims', type=int, nargs='+', default=[64, 64, 64, 8],
                   help="pair_embed_dims + [num_heads]; model default [64,64,64]+[8]")
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no-compile', action='store_true', help="skip the torch.compile variants")
    p.add_argument('--plot-out', default=None, help="write a fwd+bwd time / peak-mem bar chart")
    args = p.parse_args()

    device = torch.device(args.device)
    cuda = device.type == 'cuda'
    torch.manual_seed(0)

    U, valid_pairs = make_inputs(args.batch, args.particles, device)
    frac = valid_pairs.float().mean().item()
    print(f"interaction embedding | batch={args.batch} particles={args.particles} "
          f"dims={args.dims} | device={device} | TRAIN mode (BN stats + backward)")
    print(f"valid pairs: {frac * 100:.1f}% of {args.particles}^2 "
          f"({int(valid_pairs.sum().item())} of {args.batch * args.particles**2}) "
          f"-> ragged skips ~{(1 - frac) * 100:.0f}%\n")

    baseline = InteractionEmbedding(4, list(args.dims)).to(device).train()
    ragged   = RaggedInteractionEmbedding(4, tuple(args.dims)).to(device).train()

    def fb(mod, *a):                              # fwd+bwd closure (train-relevant)
        def run():
            mod.zero_grad(set_to_none=True)
            (mod(*a).float().pow(2).mean()).backward()
        return run

    def fw(mod, *a):                              # fwd-only closure
        def run():
            with torch.no_grad():
                mod(*a)
        return run

    variants = [
        ('baseline',        baseline, (U,)),
        ('ragged',          ragged,   (U, valid_pairs)),
    ]
    if not args.no_compile:
        variants += [
            ('compile',        torch.compile(baseline), (U,)),
            ('ragged+compile', torch.compile(ragged),   (U, valid_pairs)),
        ]

    rows = []
    for name, mod, a in variants:
        f_ms, f_mem   = bench(fw(mod, *a), args.iters, args.warmup, cuda)
        fb_ms, fb_mem = bench(fb(mod, *a), args.iters, args.warmup, cuda)
        rows.append((name, f_ms, fb_ms, fb_mem))

    base_fb = rows[0][2]
    print(f"{'variant':16s}{'fwd ms':>10s}{'fwd+bwd ms':>13s}{'speedup':>10s}{'peak MB':>11s}")
    for name, f_ms, fb_ms, fb_mem in rows:
        sp = base_fb / fb_ms if fb_ms else float('nan')
        print(f"{name:16s}{f_ms:10.2f}{fb_ms:13.2f}{sp:9.2f}x{fb_mem:11.1f}")

    if args.plot_out:
        _plot(rows, args.plot_out, args.batch, args.particles, frac)


def _plot(rows, out_path, batch, particles, frac):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = [r[0] for r in rows]
    fb_ms = [r[2] for r in rows]
    mem   = [r[3] for r in rows]
    colors = ['#7f7f7f', '#d62728', '#1f77b4', '#9467bd'][:len(rows)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.bar(names, fb_ms, color=colors, edgecolor='black', lw=0.5)
    a1.set_ylabel('fwd+bwd ms/iter'); a1.set_title('Interaction embedding — train step time')
    for i, v in enumerate(fb_ms):
        a1.text(i, v, f'{v:.1f}\n({fb_ms[0] / v:.2f}x)', ha='center', va='bottom', fontsize=8)
    a2.bar(names, mem, color=colors, edgecolor='black', lw=0.5)
    a2.set_ylabel('peak MB'); a2.set_title('peak memory')
    for i, v in enumerate(mem):
        a2.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
    for ax in (a1, a2):
        ax.tick_params(axis='x', labelrotation=20)
    fig.suptitle(f'batch {batch} · {particles} particles · {frac * 100:.0f}% valid pairs', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
