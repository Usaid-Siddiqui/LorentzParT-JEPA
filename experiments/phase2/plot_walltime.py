"""
Wall-clock figures for Phase 2 (1M).

At full labels JEPA only ties scratch on AUC, so the honest comparison is
compute. scratch pays a single training run; JEPA/MAE pay pretraining PLUS
finetuning. This produces:

  walltime_convergence.png  two panels of val-metric vs wall-clock (a line
                            graph subsumes a cost bar: total time = curve end,
                            pretrain time = curve start, finetune = span,
                            final accuracy = plateau):
                            (left)  FINETUNE-only clock — pretrained models'
                                    head start / faster finetune convergence.
                            (right) TOTAL clock (pretrain + finetune) — the
                                    honest cost: scratch trains while JEPA is
                                    still pretraining. Vertical dotted lines
                                    mark when each pretrained method starts
                                    finetuning; the dashed line is scratch's
                                    final accuracy, so you can read off how
                                    long anything takes to reach it. Each curve
                                    end is annotated with its final test AUC.
                            Exact pretrain/finetune/total hours print as a table.

Pretrain time comes from the seed JSONs (pretrain_time_s); finetune wall-clock
comes from the finetune CSVs in logs/, so run this ON THE CLUSTER.

    python experiments/phase2/plot_walltime.py \\
        --results-dir experiments/phase2/results --tag 1m --seeds 42 123 456 \\
        --jepa-encoder best      # charge JEPA only to its best-val epoch (~1h), not the full 4h

--jepa-encoder final (default) charges JEPA the full pretrain and uses the final
encoder's finetune; --jepa-encoder best uses the best-val encoder (pretrain timed
to the val-loss minimum + --jepa-patience epochs, finetune = *_bestft_*). Output
is walltime_convergence.png / walltime_convergence_bestval.png respectively.
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


def bestval_pretrain_seconds(pretrain_csv, patience):
    """Elapsed_total_s at the val-loss minimum + `patience` epochs — the honest
    stopping point for taking the best-val encoder (you must train a little past
    the minimum to confirm it). JEPA pretrain CSV cols:
    epoch,emb_loss,val_loss,lr,ema,epoch_t,elapsed,best."""
    if not os.path.exists(pretrain_csv):
        return None
    rows = []
    with open(pretrain_csv) as f:
        for r in csv.reader(f):
            try:
                rows.append((int(float(r[0])), float(r[2]), float(r[6])))  # epoch, val_loss, elapsed
            except (ValueError, IndexError):
                continue
    if not rows:
        return None
    best_ep = min(rows, key=lambda t: t[1])[0]
    cand = [e for e in rows if e[0] >= best_ep + patience]
    return (cand[0][2] if cand else rows[-1][2])


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
    """(val_metric, elapsed_s) arrays from a finetune CSV; skips the header."""
    vm, el = [], []
    with open(path) as f:
        for r in csv.reader(f):
            try:
                int(float(r[0]))                # skip header / blank rows
            except (ValueError, IndexError):
                continue
            vm.append(float(r[4]))              # val_metric
            el.append(float(r[6]))              # elapsed_total_s
    return np.array(vm), np.array(el)


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
    p = argparse.ArgumentParser(description="Phase 2 wall-clock figures")
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
                   help="JEPA pretrain CSVs (to time the best-val epoch).")
    p.add_argument('--jepa-patience', type=int, default=5,
                   help="epochs past the val-loss minimum to charge as pretrain "
                        "for --jepa-encoder best (honest deployable stop).")
    p.add_argument('--jepa-bestft-json', default='./experiments/phase2/diag_bestft',
                   help="dir with best-val finetune JSONs (roc_auc_ovo) for the AUC label.")
    args = p.parse_args()
    out = args.output_dir or args.results_dir
    os.makedirs(out, exist_ok=True)

    # JEPA row depends on the chosen encoder; best-val uses the *_bestft_* finetune.
    jbest  = args.jepa_encoder == 'best'
    jlabel = 'JEPA (best-val)' if jbest else 'JEPA'
    jstem  = 'jepa_{tag}_bestft_seed{s}' if jbest else 'jepa_{tag}_ft_seed{s}'
    METHODS = [(jlabel, 'jepa_finetune', jstem, '#1f77b4')] + BASE_METHODS

    agg = {m[0]: {'pre': [], 'ft': [], 'auc': []} for m in METHODS}
    ft_curves    = defaultdict(list)   # label -> [(finetune_hours, val_metric), ...]
    total_curves = defaultdict(list)   # label -> [(total_hours,    val_metric), ...]

    for s in args.seeds:
        jpath = os.path.join(args.results_dir, f'seed_{s}.json')
        if not os.path.exists(jpath):
            print(f'[warn] missing {jpath}, skipping seed {s}')
            continue
        cond = json.load(open(jpath))['conditions']
        for label, key, stem, _ in METHODS:
            c = cond.get(key, {})
            pre = c.get('pretrain_time_s') or 0.0
            auc = c.get('test_auc')
            if label == jlabel and jbest:                    # best-val JEPA: cheaper pretrain + its own AUC
                bp = bestval_pretrain_seconds(
                    os.path.join(args.jepa_pretrain_dir, f'jepa_{args.tag}_seed{s}.csv'),
                    args.jepa_patience)
                if bp is not None:
                    pre = bp
                ba = bestft_auc(args.jepa_bestft_json, s)
                if ba is not None:
                    auc = ba
            csvp = os.path.join(args.logs_dir, stem.format(tag=args.tag, s=s) + '.csv')
            if not os.path.exists(csvp):
                print(f'[warn] missing finetune CSV {csvp}')
                continue
            vm, el = read_ft_csv(csvp)
            if el.size == 0:
                print(f'[warn] empty CSV {csvp}')
                continue
            agg[label]['pre'].append(pre)
            agg[label]['ft'].append(el[-1])
            if auc is not None:
                agg[label]['auc'].append(auc)
            ft_curves[label].append((el / 3600.0, vm))
            total_curves[label].append(((el + pre) / 3600.0, vm))

    labels = [m[0] for m in METHODS]
    colors = {m[0]: m[3] for m in METHODS}
    pre_h = {l: (np.mean(agg[l]['pre']) / 3600 if agg[l]['pre'] else 0.0) for l in labels}
    ft_h  = {l: (np.mean(agg[l]['ft'])  / 3600 if agg[l]['ft']  else 0.0) for l in labels}
    auc   = {l: (np.mean(agg[l]['auc']) if agg[l]['auc'] else float('nan')) for l in labels}

    # ---- convergence, two clocks ----
    scratch_acc = (np.mean([c[1][-1] for c in ft_curves['Scratch']])
                   if ft_curves['Scratch'] else None)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, curves, title, xlabel in [
        (axL, ft_curves,    'Finetune-only clock (head start)',          'finetune wall-clock (hours)'),
        (axR, total_curves, 'Total clock = pretrain + finetune (cost)',  'cumulative wall-clock (hours)'),
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
    # mark where each pretrained method begins finetuning on the total-clock panel
    for label in (jlabel, 'MAE'):
        if pre_h[label] > 0:
            axR.axvline(pre_h[label], ls=':', color=colors[label], lw=1.2)
            axR.text(pre_h[label], axR.get_ylim()[0], f' {label} ft start',
                     color=colors[label], fontsize=8, rotation=90, va='bottom', ha='left')
    axL.set_ylabel('val metric (accuracy)')
    fig.suptitle('Time-to-accuracy: finetune-only vs. total (pretrain-charged) wall-clock', y=1.02)
    fig.tight_layout()
    fname = 'walltime_convergence_bestval.png' if jbest else 'walltime_convergence.png'
    fig.savefig(os.path.join(out, fname), dpi=300, bbox_inches='tight')
    print('wrote', fname)

    # ---- table ----
    print(f"\n{'method':10s}{'pretrain_h':>12s}{'finetune_h':>12s}{'total_h':>10s}{'auc':>9s}")
    for l in labels:
        print(f"{l:10s}{pre_h[l]:12.2f}{ft_h[l]:12.2f}{pre_h[l] + ft_h[l]:10.2f}{auc[l]:9.4f}")


if __name__ == '__main__':
    main()
