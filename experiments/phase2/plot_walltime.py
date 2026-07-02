"""
Cost-vs-accuracy figures for Phase 2 (1M).

At full labels JEPA only ties scratch on AUC, so the honest comparison is
compute. scratch pays a single training run; JEPA/MAE pay pretraining PLUS
finetuning. This produces a two-panel convergence figure:

  (left)  FINETUNE-only cost  — pretrained models' head start / faster
                                convergence, ignoring the pretrain bill.
  (right) TOTAL cost (pretrain + finetune) — the honest bill: scratch trains
                                while JEPA/MAE are still pretraining. Vertical
                                dotted lines mark when each pretrained method
                                starts finetuning; the dashed line is scratch's
                                final accuracy. Each curve end is annotated with
                                its final test AUC.

Two cost axes (--cost-axis):
  walltime  (default)  hours from the finetune CSVs; pretrain hours from the
                       seed JSONs (or the best-val stopping point). Noisy on a
                       shared/preemptible GPU -> run ON THE CLUSTER.
  flops                deterministic training FLOPs = (FLOPs/jet from
                       measure_flops.py) x N_train x epochs. Immune to cluster
                       jitter; the recommended headline. Reads flops.json.

--jepa-encoder final (default) charges JEPA the full pretrain; --jepa-encoder
best uses the best-val encoder (pretrain charged only to the val-loss minimum +
--jepa-patience epochs, finetune = *_bestft_*).

    # deterministic compute axis, best-val JEPA encoder (recommended):
    python experiments/phase2/measure_flops.py --json-out experiments/phase2/flops.json
    python experiments/phase2/plot_walltime.py --tag 1m --seeds 42 123 456 \\
        --cost-axis flops --jepa-encoder best

Output: {walltime,flops}_convergence[_bestval].png.
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Base (label, JSON condition key, finetune-CSV stem, colour) for MAE/scratch.
# The JEPA row is built in main() so --jepa-encoder can switch it between the
# final encoder and the (cheaper, ~equal-accuracy) best-val encoder.
BASE_METHODS = [
    ('MAE',     'mae_finetune',  'mae_{tag}_ft_seed{s}',   '#2ca02c'),
    ('Scratch', 'scratch',       'scratch_{tag}_seed{s}',  '#ff7f0e'),
]


def _jepa_pretrain_rows(pretrain_csv):
    """[(epoch, val_loss, elapsed), ...] from a JEPA pretrain CSV, keeping only
    the latest run if the file was appended across a resume. JEPA pretrain cols:
    epoch,emb_loss,val_loss,lr,ema,epoch_t,elapsed,best."""
    if not os.path.exists(pretrain_csv):
        return []
    rows = []
    with open(pretrain_csv) as f:
        for r in csv.reader(f):
            try:
                ep, val, elapsed = int(float(r[0])), float(r[2]), float(r[6])
            except (ValueError, IndexError):
                continue
            if rows and elapsed < rows[-1][2]:      # resume/append -> keep latest run
                rows = []
            rows.append((ep, val, elapsed))
    return rows


def bestval_pretrain_seconds(pretrain_csv, patience):
    """Elapsed_total_s at the val-loss minimum + `patience` epochs — the honest
    stopping point for taking the best-val encoder (you must train a little past
    the minimum to confirm it)."""
    rows = _jepa_pretrain_rows(pretrain_csv)
    if not rows:
        return None
    best_ep = min(rows, key=lambda t: t[1])[0]
    cand = [e for e in rows if e[0] >= best_ep + patience]
    return (cand[0][2] if cand else rows[-1][2])


def jepa_best_epoch(pretrain_csv):
    """Epoch of minimum val_loss in the JEPA pretrain CSV (None if absent)."""
    rows = _jepa_pretrain_rows(pretrain_csv)
    return min(rows, key=lambda t: t[1])[0] if rows else None


def pretrain_last_epoch(pretrain_csv):
    """Highest epoch in any pretrain CSV (keeps only the latest run on resume).
    Column 0 is the epoch for both the JEPA and MAE pretrain logs."""
    if not os.path.exists(pretrain_csv):
        return None
    eps = []
    with open(pretrain_csv) as f:
        for r in csv.reader(f):
            try:
                e = int(float(r[0]))
            except (ValueError, IndexError):
                continue
            if eps and e < eps[-1]:                 # resume/append -> keep latest run
                eps = []
            eps.append(e)
    return max(eps) if eps else None


def bestft_auc(diag_dir, seed):
    """Best-val finetune OVO AUC from diag_bestft (roc_auc_ovo); None if absent."""
    for name in (f'seed{seed}.json', f'seed_{seed}.json'):
        p = os.path.join(diag_dir, name)
        if os.path.exists(p):
            try:
                return float(json.load(open(p)).get('roc_auc_ovo'))
            except Exception:
                return None
    return None


def read_ft_csv(path):
    """(val_metric, elapsed_s, epoch) from a finetune CSV; skips the header, and
    if the CSV was appended across a resume (elapsed jumps backward), keeps only
    the latest run so the curve stays monotonic in wall-clock."""
    vm, el, ep = [], [], []
    with open(path) as f:
        for r in csv.reader(f):
            try:
                e0 = int(float(r[0]))               # epoch; skips header / blank rows
            except (ValueError, IndexError):
                continue
            e = float(r[6])                          # elapsed_total_s
            if el and e < el[-1]:                    # resume/append -> keep latest run
                vm, el, ep = [], [], []
            vm.append(float(r[4]))                   # val_metric
            el.append(e)
            ep.append(e0)
    return np.array(vm), np.array(el), np.array(ep, dtype=float)


def mean_curve(seed_curves, n=200):
    """seed_curves: [(x, y), ...]. Interpolate onto the shared x-overlap, average."""
    if not seed_curves:
        return None, None
    lo = max(x[0] for x, _ in seed_curves)
    hi = min(x[-1] for x, _ in seed_curves)
    if hi <= lo:                                # no overlap → use the longest curve
        x, y = max(seed_curves, key=lambda c: c[0][-1])
        return x, y
    grid = np.linspace(lo, hi, n)
    ys = np.array([np.interp(grid, x, y) for x, y in seed_curves])
    return grid, ys.mean(axis=0)


def main():
    p = argparse.ArgumentParser(description="Phase 2 cost-vs-accuracy figures")
    p.add_argument('--results-dir', default='./experiments/phase2/results')
    p.add_argument('--logs-dir',    default='./logs/LorentzParT/logging')
    p.add_argument('--seeds',       nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--tag',         default='1m')
    p.add_argument('--output-dir',  default=None, help='defaults to --results-dir')
    p.add_argument('--jepa-encoder', choices=['final', 'best'], default='final',
                   help="'final': full-schedule encoder (pretrain = full run). "
                        "'best': best-val encoder (pretrain charged only to the val-loss "
                        "minimum + patience) — ~equal accuracy, far cheaper.")
    p.add_argument('--jepa-pretrain-dir', default='./logs/ParticleJEPA/logging',
                   help="JEPA pretrain CSVs (to time/epoch the best-val stop).")
    p.add_argument('--jepa-patience', type=int, default=5,
                   help="epochs past the val-loss minimum to charge as pretrain "
                        "for --jepa-encoder best (honest deployable stop).")
    p.add_argument('--jepa-bestft-json', default='./experiments/phase2/diag_bestft',
                   help="dir with best-val finetune JSONs (roc_auc_ovo) for the AUC label.")
    p.add_argument('--cost-axis', choices=['walltime', 'flops'], default='walltime',
                   help="'walltime': hours from the CSVs (noisy on a shared GPU). "
                        "'flops': deterministic training FLOPs (recommended headline).")
    p.add_argument('--flops-json', default='./experiments/phase2/flops.json',
                   help="FLOPs/jet per step from measure_flops.py (for --cost-axis flops).")
    p.add_argument('--n-train', type=int, default=1_000_000,
                   help="training jets per epoch (for --cost-axis flops).")
    p.add_argument('--mae-pretrain-dir', default='./logs/LorentzParT/logging',
                   help="MAE pretrain CSVs (to count MAE pretrain epochs for the FLOPs offset).")
    args = p.parse_args()
    out = args.output_dir or args.results_dir
    os.makedirs(out, exist_ok=True)

    # JEPA row depends on the chosen encoder; best-val uses the *_bestft_* finetune.
    jbest  = args.jepa_encoder == 'best'
    jlabel = 'JEPA (best-val)' if jbest else 'JEPA'
    jstem  = 'jepa_{tag}_bestft_seed{s}' if jbest else 'jepa_{tag}_ft_seed{s}'
    METHODS = [(jlabel, 'jepa_finetune', jstem, '#1f77b4')] + BASE_METHODS

    # FLOPs axis: PFLOPs per finetune / pretrain epoch = (FLOPs/jet) x N_train.
    flops_axis = args.cost_axis == 'flops'
    if flops_axis:
        fp = json.load(open(args.flops_json))['flops_per_jet']
        PFLOP = 1e15
        ft_pe   = fp['finetune']      * args.n_train / PFLOP
        jpre_pe = fp['jepa_pretrain'] * args.n_train / PFLOP
        mpre_pe = fp['mae_pretrain']  * args.n_train / PFLOP
    unit = 'PFLOPs' if flops_axis else 'hours'

    agg = {m[0]: {'pre': [], 'ft': [], 'auc': []} for m in METHODS}   # values already in `unit`
    ft_curves    = defaultdict(list)   # label -> [(finetune_cost, val_metric), ...]
    total_curves = defaultdict(list)   # label -> [(total_cost,    val_metric), ...]

    for s in args.seeds:
        jpath = os.path.join(args.results_dir, f'seed_{s}.json')
        if not os.path.exists(jpath):
            print(f'[warn] missing {jpath}, skipping seed {s}')
            continue
        cond = json.load(open(jpath))['conditions']
        jpc  = os.path.join(args.jepa_pretrain_dir, f'jepa_{args.tag}_seed{s}.csv')
        for label, key, stem, _ in METHODS:
            c = cond.get(key, {})
            auc = c.get('test_auc')
            if label == jlabel and jbest:                    # best-val JEPA uses its own AUC
                ba = bestft_auc(args.jepa_bestft_json, s)
                if ba is not None:
                    auc = ba
            csvp = os.path.join(args.logs_dir, stem.format(tag=args.tag, s=s) + '.csv')
            if not os.path.exists(csvp):
                print(f'[warn] missing finetune CSV {csvp}')
                continue
            vm, el, ep = read_ft_csv(csvp)
            if el.size == 0:
                print(f'[warn] empty CSV {csvp}')
                continue

            if flops_axis:
                ft_x = ep * ft_pe                            # cumulative PFLOPs by epoch
                if label == jlabel:
                    be = jepa_best_epoch(jpc) if jbest else pretrain_last_epoch(jpc)
                    if be is None:
                        print(f'[warn] no JEPA pretrain CSV {jpc} -> pretrain FLOPs charged 0')
                    pe = ((be or 0) + args.jepa_patience) if jbest else (be or 0)
                    pre = jpre_pe * pe
                elif label == 'MAE':
                    mpc = os.path.join(args.mae_pretrain_dir, f'mae_{args.tag}_seed{s}.csv')
                    me = pretrain_last_epoch(mpc)
                    if me is None:
                        print(f'[warn] no MAE pretrain CSV {mpc} -> pretrain FLOPs charged 0')
                    pre = mpre_pe * (me or 0)
                else:
                    pre = 0.0
            else:
                ft_x = el / 3600.0                           # hours
                if label == jlabel and jbest:
                    bp = bestval_pretrain_seconds(jpc, args.jepa_patience)
                    pre = (bp if bp is not None else 0.0) / 3600.0
                else:
                    pre = (c.get('pretrain_time_s') or 0.0) / 3600.0

            agg[label]['pre'].append(pre)
            agg[label]['ft'].append(ft_x[-1])
            if auc is not None:
                agg[label]['auc'].append(auc)
            ft_curves[label].append((ft_x, vm))
            total_curves[label].append((ft_x + pre, vm))

    labels = [m[0] for m in METHODS]
    colors = {m[0]: m[3] for m in METHODS}
    pre_c = {l: (np.mean(agg[l]['pre']) if agg[l]['pre'] else 0.0) for l in labels}
    ft_c  = {l: (np.mean(agg[l]['ft'])  if agg[l]['ft']  else 0.0) for l in labels}
    auc   = {l: (np.mean(agg[l]['auc']) if agg[l]['auc'] else float('nan')) for l in labels}

    # ---- convergence, two panels ----
    scratch_acc = (np.mean([c[1][-1] for c in ft_curves['Scratch']])
                   if ft_curves['Scratch'] else None)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, curves, title, xlabel in [
        (axL, ft_curves,    'Finetune-only cost (head start)',           f'finetune {unit}'),
        (axR, total_curves, 'Total cost = pretrain + finetune (bill)',   f'cumulative {unit}'),
    ]:
        for label in labels:
            for cx, cy in curves[label]:
                ax.plot(cx, cy, color=colors[label], alpha=0.22, lw=1)
            mx, my = mean_curve(curves[label])
            if mx is not None:
                ax.plot(mx, my, color=colors[label], lw=2.2, label=label)
                if not np.isnan(auc[label]):    # tie convergence curve back to headline AUC
                    ax.annotate(f'AUC {auc[label]:.3f}', (mx[-1], my[-1]),
                                color=colors[label], fontsize=8, fontweight='bold',
                                xytext=(4, 0), textcoords='offset points', va='center')
        if scratch_acc is not None:
            ax.axhline(scratch_acc, ls='--', color='#9e9e9e', lw=1)
            ax.text(ax.get_xlim()[1], scratch_acc, ' scratch final',
                    color='#666', fontsize=8, va='bottom', ha='right')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.legend(loc='lower right')
    # mark where each pretrained method begins finetuning on the total-cost panel
    for label in (jlabel, 'MAE'):
        if pre_c[label] > 0:
            axR.axvline(pre_c[label], ls=':', color=colors[label], lw=1.2)
            axR.text(pre_c[label], axR.get_ylim()[0], f' {label} ft start',
                     color=colors[label], fontsize=8, rotation=90, va='bottom', ha='left')
    axL.set_ylabel('val metric (accuracy)')
    fig.suptitle(f'Time-to-accuracy: finetune-only vs. total (pretrain-charged) {unit}', y=1.02)
    fig.tight_layout()
    stem = 'flops_convergence' if flops_axis else 'walltime_convergence'
    fname = f'{stem}_bestval.png' if jbest else f'{stem}.png'
    fig.savefig(os.path.join(out, fname), dpi=300, bbox_inches='tight')
    print('wrote', fname)

    # ---- table ----
    u = unit[:8]
    print(f"\n{'method':16s}{'pretrain_' + u:>16s}{'finetune_' + u:>16s}{'total_' + u:>14s}{'auc':>9s}")
    for l in labels:
        print(f"{l:16s}{pre_c[l]:16.2f}{ft_c[l]:16.2f}{pre_c[l] + ft_c[l]:14.2f}{auc[l]:9.4f}")


if __name__ == '__main__':
    main()
