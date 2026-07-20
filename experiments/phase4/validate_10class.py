"""
Phase 4 — step 0: validate the attentive probe on the existing 10-class task.

Runs, for each seed and each encoder {jepa, mae, scratch}, the two FROZEN protocols
(linear + attentive) with the RAGGED forward — matching how the 1M encoders were
trained — then tabulates them against the known full-finetune references. Scratch is
the random-feature control (no --weights). Resumable: a cell whose JSON already exists
is skipped.

Success criterion: the attentive probe lands between the linear floor and the finetune
ceiling, ideally JEPA >= MAE (linear undersells JEPA; the attentive head should close or
flip that gap). This gates whether the attentive protocol is carried into the 5-task matrix.

    python experiments/phase4/validate_10class.py \\
        --data-dir ./data_1m --seeds 42 123 456

Adjust --jepa-tmpl / --mae-tmpl if your 1M checkpoint names differ (use {seed}).
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

# Known full-finetune references (1M ragged, best-val encoder) — for the comparison band.
FINETUNE_REF = {'jepa': 0.9497, 'mae': 0.9501}
# Historical linear-probe numbers were run NON-ragged (undersell); shown only as context.
LINEAR_NONRAGGED_REF = {'jepa': 0.711, 'mae': 0.749}


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 step 0 — attentive-probe 10-class validation")
    p.add_argument('--data-dir', default='./data_1m')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--jepa-tmpl', default='./logs/ParticleJEPA/best/jepa_1m_ragged_seed{seed}_best.pt',
                   help='JEPA best-val encoder path template ({seed} placeholder)')
    p.add_argument('--mae-tmpl', default='./logs/LorentzParT/best/mae_1m_ragged_seed{seed}.pt',
                   help='MAE encoder path template ({seed} placeholder)')
    p.add_argument('--linear-config', default='experiments/phase4/configs/linear_probe.yaml')
    p.add_argument('--attn-config', default='experiments/phase4/configs/attentive_probe.yaml')
    p.add_argument('--output-dir', default='experiments/phase4/results_10class')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--skip-linear', action='store_true')
    p.add_argument('--skip-attn', action='store_true')
    return p.parse_args()


def safe_stage(cmd, env, desc):
    try:
        print(f"\n[run] {desc}\n  {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[STAGE FAILED] {desc}\n  {e}\n  --> logged, continuing.\n", flush=True)
        return False


def run(args, python, env):
    os.makedirs(args.output_dir, exist_ok=True)

    for seed in args.seeds:
        encoders = [
            ('jepa',    args.jepa_tmpl.format(seed=seed)),
            ('mae',     args.mae_tmpl.format(seed=seed)),
            ('scratch', None),   # random-feature control
        ]
        for enc, weights in encoders:
            if weights is not None and not os.path.exists(weights):
                print(f"[skip] {enc} seed {seed} — encoder missing: {weights}", flush=True)
                continue

            for proto, script, cfg, skip in [
                ('linear', 'experiments/phase0/linear_probe.py',   args.linear_config, args.skip_linear),
                ('attn',   'experiments/phase4/attentive_probe.py', args.attn_config,   args.skip_attn),
            ]:
                if skip:
                    continue
                run_name = f'{proto}_{enc}_seed{seed}'
                out_json = os.path.join(args.output_dir, f'{run_name}.json')
                if os.path.exists(out_json):
                    print(f"[skip] {run_name} — exists", flush=True)
                    continue
                cmd = [python, script, '--data-dir', args.data_dir,
                       '--config-path', cfg, '--run-name', run_name,
                       '--seed', str(seed), '--output-dir', args.output_dir]
                if weights is not None:
                    cmd += ['--weights', weights]
                safe_stage(cmd, env, f'{proto} probe — {enc} seed {seed}')


def aggregate(args):
    """Mean +/- std test_auc across seeds, per (protocol, encoder)."""
    def collect(proto, enc):
        vals = []
        for seed in args.seeds:
            p = os.path.join(args.output_dir, f'{proto}_{enc}_seed{seed}.json')
            if os.path.exists(p):
                vals.append(json.load(open(p))['test_auc'])
        return vals

    print("\n" + "=" * 72)
    print("Phase 4 step 0 — 10-class validation (test AUC, ragged forward)")
    print("=" * 72)
    print(f"{'encoder':<9} {'linear (ragged)':<20} {'attentive (ragged)':<22} {'finetune ref':<12}")
    print("-" * 72)
    for enc in ['jepa', 'mae', 'scratch']:
        cells = []
        for proto in ['linear', 'attn']:
            v = collect(proto, enc)
            cells.append(f"{np.mean(v):.4f} ± {np.std(v):.4f} (n={len(v)})" if v else "—")
        ref = f"{FINETUNE_REF[enc]:.4f}" if enc in FINETUNE_REF else "— (rand)"
        print(f"{enc:<9} {cells[0]:<20} {cells[1]:<22} {ref:<12}")
    print("-" * 72)
    print("context: historical NON-ragged linear refs (undersell) — "
          f"jepa {LINEAR_NONRAGGED_REF['jepa']}, mae {LINEAR_NONRAGGED_REF['mae']}")
    print("success: attentive between linear and finetune, ideally jepa >= mae.")


def main():
    args = parse_args()
    python = sys.executable
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    t0 = time.monotonic()
    run(args, python, env)
    aggregate(args)
    print(f"\nDone in {(time.monotonic() - t0) / 60:.1f} min → {args.output_dir}", flush=True)


if __name__ == '__main__':
    main()
