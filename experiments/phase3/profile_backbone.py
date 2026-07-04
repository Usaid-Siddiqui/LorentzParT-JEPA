"""
Phase 3, step 1 — profile the LorentzParT backbone to find the real hotspot.

Everything else in Phase 3 gates on this: we do NOT assume the geometric product
(or EquiLinear, or attention) dominates — we measure. Builds LorentzParT at the true
training shape, runs fwd+bwd under torch.profiler, and prints a ranked op table
(CUDA time + memory). The equivariant ops come from the `lgatr` package and are
usually einsum-based; in the table watch for `aten::einsum` / `aten::bmm` /
`aten::mul` + `aten::sum` (geometric product), `aten::addmm` / `aten::linear`
(EquiLinear / dense), and attention (`aten::scaled_dot_product_attention` or a
bmm+softmax+bmm cluster).

Run ON THE H200 — a CUDA profile is the only meaningful one (CPU works too but only
shows host-side cost). Paste the printed table back to pick the fuse target.

    python experiments/phase3/profile_backbone.py --batch 1000 --iters 10
    # local smoke (CPU): --batch 8 --iters 2

Optional: --trace-out trace.json  -> load in chrome://tracing / perfetto for detail.
"""

import argparse
import os
import sys

import torch
import yaml
from torch.profiler import profile, ProfilerActivity, schedule

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.configs import LorentzParTConfig
from src.models import LorentzParT


def make_batch(batch, n_particles, device, realistic_padding=False):
    """(B, N, 4) inputs.

    Default: no padded slots (all N active) — the backbone's upper-bound cost, and
    padding-independent (fine for profiling the STOCK dense path, which always processes
    all N^2 pairs). realistic_padding=True: per-jet valid count ~ N(40, 18) clamped to
    [5, N] (median ~40 -> ~90% padding) — REQUIRED to profile the RAGGED backbone, which
    only skips work when padding is actually present."""
    if realistic_padding:
        g = torch.Generator(device='cpu').manual_seed(0)
        n_valid = torch.normal(40.0, 18.0, (batch,), generator=g).round().clamp(5, n_particles).long()
        mask = (torch.arange(n_particles)[None, :] < n_valid[:, None]).to(device)
        x = torch.randn(batch, n_particles, 4, device=device) * mask[..., None]
        x[..., 3] = torch.where(mask, x[..., 3].abs() + 0.5, torch.zeros_like(x[..., 3]))
        return x
    x = torch.randn(batch, n_particles, 4, device=device)
    x[..., 3] = x[..., 3].abs() + 1.0
    return x


# Heuristic op-name -> (subsystem, colour), checked in order (an aid, not ground truth).
_CATEGORIES = [
    ('attention',        ('#1f77b4', ('scaled_dot_product', 'sdpa', 'baddbmm', 'bmm', 'softmax'))),
    ('equivariant/lgatr',('#d62728', ('einsum', 'linalg'))),
    ('pair-interaction', ('#ff7f0e', ('conv', 'batch_norm'))),
    ('linear/dense',     ('#2ca02c', ('addmm', 'linear', 'matmul', '::mm'))),
    ('elementwise/norm', ('#9467bd', ('gelu', 'layer_norm', 'aten::mul', 'aten::add', 'copy_'))),
]


def _categorize(name):
    for cat, (colour, keys) in _CATEGORIES:
        if any(k in name for k in keys):
            return cat, colour
    return 'other', '#7f7f7f'


def _short_label(name, max_len=44):
    """Shorten a profiler op/kernel name for a y-axis label. CUDA kernel names are
    often full C++ template signatures (e.g. 'void cudnn::bn_bw_1C11_kernel_new<
    float, float, floa...'); this drops the 'void' prefix and template args, then
    keeps the most specific trailing '::'-segments that fit in max_len."""
    n = name.strip()
    if n.startswith('void '):
        n = n[5:]
    n = n.split('<', 1)[0].strip().rstrip(':').strip()   # drop template args
    if len(n) > max_len:
        parts = n.split('::')
        kept = parts[-1]
        for p in reversed(parts[:-1]):
            if len(kept) + len(p) + 2 > max_len - 1:
                break
            kept = f'{p}::{kept}'
        n = kept
    if len(n) > max_len:
        n = n[:max_len - 1] + '…'
    return n


def _self_time_us(e, cuda):
    """Self device (CUDA) time if profiling GPU, else self CPU time — robust across
    torch versions (self_device_time_total is the newer name for self_cuda_time_total)."""
    if cuda:
        for a in ('self_device_time_total', 'self_cuda_time_total'):
            v = getattr(e, a, None)
            if v:
                return float(v)
        return 0.0
    return float(getattr(e, 'self_cpu_time_total', 0) or 0)


def plot_top_ops(events, cuda, out_path, batch, n_particles, top_n):
    """Horizontal bar chart of the top ops by self-time, coloured by subsystem, with a
    rolled-up '(other)' bar. Shows at a glance which subsystem dominates the backbone."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = sorted(((e.key, _self_time_us(e, cuda)) for e in events),
                  key=lambda r: r[1], reverse=True)
    total = sum(t for _, t in rows) or 1.0
    top = rows[:top_n]
    rest = sum(t for _, t in rows[top_n:])

    names = [k for k, _ in top]                       # full names, kept for categorization
    vals  = [t / 1e3 for _, t in top]                 # ms
    cols  = [_categorize(k)[1] for k in names]
    if rest > 0:
        names.append(f'(other · {len(rows) - top_n} ops)'); vals.append(rest / 1e3); cols.append('#cccccc')
    labels = [_short_label(n) for n in names]         # shortened, for display only

    y = range(len(names))[::-1]                        # largest at top
    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(names) + 1)))
    ax.barh(list(y), vals, color=cols, edgecolor='black', linewidth=0.4)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f' {v:.1f}ms ({100 * v * 1e3 / total:.0f}%)', va='center', fontsize=8)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=8, fontfamily='monospace')
    ax.set_xlabel(f"self {'CUDA' if cuda else 'CPU'} time (ms, summed over profiled steps)")
    ax.set_title(f"LorentzParT backbone hotspots · batch {batch} · {n_particles} particles · "
                 f"{'GPU' if cuda else 'CPU'}")
    cat_color = {cat: col for cat, (col, _) in _CATEGORIES}
    seen = dict.fromkeys(_categorize(n)[0] for n in names if not n.startswith('(other'))
    ax.legend(handles=[Patch(color=cat_color.get(cat, '#7f7f7f'), label=cat) for cat in seen],
              fontsize=8, loc='lower right', title='subsystem (heuristic)')
    ax.margins(x=0.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"wrote hotspot chart -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Profile the LorentzParT backbone (fwd+bwd)")
    p.add_argument('--config-path', default='./configs/train_lorentz_part.yaml',
                   help="classification config; its [model] section builds the backbone.")
    p.add_argument('--batch',    type=int, default=1000, help="real training batch is 1000")
    p.add_argument('--particles', type=int, default=None, help="default = config max_num_particles")
    p.add_argument('--iters',    type=int, default=10, help="profiled (active) steps")
    p.add_argument('--warmup',   type=int, default=3,  help="warmup steps (compile/caches)")
    p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--fwd-only', action='store_true', help="profile forward only (skip backward)")
    p.add_argument('--sort-by',  default=None,
                   help="key_averages sort key; default self_cuda_time_total on CUDA, "
                        "else self_cpu_time_total")
    p.add_argument('--row-limit', type=int, default=30)
    p.add_argument('--by-shape',  action='store_true',
                   help="group rows by input shape (finds which op instances are hot)")
    p.add_argument('--trace-out', default=None, help="write a chrome trace to this path")
    p.add_argument('--plot-out', default=None,
                   help="write a hotspot bar chart (self-time per op, coloured by subsystem)")
    p.add_argument('--plot-top', type=int, default=15, help="ops to show in --plot-out")
    p.add_argument('--realistic-padding', action='store_true',
                   help="variable per-jet valid particles (~90%% padding); REQUIRED to profile "
                        "a ragged config (else no padding to skip -> ragged looks like dense)")
    args = p.parse_args()

    device = torch.device(args.device)
    cuda = device.type == 'cuda'
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('high')  # match training (TF32 on) so the profile reflects real GEMMs

    with open(args.config_path) as f:
        cfg = LorentzParTConfig.from_dict(yaml.safe_load(f)['model'])
    n_particles = args.particles or cfg.max_num_particles

    model = LorentzParT(config=cfg).to(device).train()
    n_params = sum(p.numel() for p in model.parameters())
    x = make_batch(args.batch, n_particles, device, realistic_padding=args.realistic_padding)

    print(f"LorentzParT backbone | params={n_params/1e6:.2f}M | device={device} | "
          f"batch={args.batch} particles={n_particles} | "
          f"{'fwd' if args.fwd_only else 'fwd+bwd'} | "
          f"padding={'realistic ~90%' if args.realistic_padding else 'none (upper-bound)'} | "
          f"ragged={'on' if getattr(cfg, 'ragged_pair_embed', False) else 'off'}")

    def step():
        model.zero_grad(set_to_none=True)
        out = model(x)                              # (B, num_classes)
        if not args.fwd_only:
            (out.float().pow(2).mean()).backward()  # surrogate loss; network bwd is what we profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    sched = schedule(wait=1, warmup=args.warmup, active=args.iters, repeat=1)
    total = 1 + args.warmup + args.iters

    with profile(activities=acts, schedule=sched, record_shapes=args.by_shape,
                 profile_memory=True) as prof:
        for _ in range(total):
            step()
            if cuda:
                torch.cuda.synchronize()
            prof.step()

    sort_by = args.sort_by or ('self_cuda_time_total' if cuda else 'self_cpu_time_total')
    print(f"\n===== ranked ops (sort_by={sort_by}) =====")
    print(prof.key_averages(group_by_input_shape=args.by_shape).table(
        sort_by=sort_by, row_limit=args.row_limit))

    if args.plot_out:
        plot_top_ops(prof.key_averages(), cuda, args.plot_out,
                     args.batch, n_particles, args.plot_top)

    if args.trace_out:
        prof.export_chrome_trace(args.trace_out)
        print(f"\nwrote chrome trace -> {args.trace_out}")


if __name__ == '__main__':
    main()
