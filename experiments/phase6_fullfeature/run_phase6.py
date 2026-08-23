"""
Phase 6 orchestrator — full-information rerun (4-vector + 10 extra scalars: displacement + PID).

Two stages, both resumable + error-isolated (mirrors run_phase4.py):
  pretrain   : JEPA + MAE on the feature-rich data, per seed.
  downstream : finetune (scratch / jepa / mae) per task × seed, with --num-extra-features
               read from the data dir's features.json.

The scientific question: does SSL (JEPA/MAE) buy any accuracy in the FEATURE-RICH regime,
or does the Phase-4 "no SSL accuracy advantage" hold once the model has the full information?
Prediction on record (input-ceiling memo): scratch/JEPA/MAE rise together and still TIE —
the regime is input-limited + label-abundant, so SSL's data-efficiency lever stays moot.

    # 1. build data (once): scripts/prepare_data.py --with-displacement --with-pid \
    #        --output-dir ./data_1m_full --train-per-class 100000 --val-per-class 10000 --test-per-class 10000
    # 2. run the phase:
    python experiments/phase6_fullfeature/run_phase6.py --data-dir ./data_1m_full --seeds 42 123 456

Default downstream tasks = full10 (Phase-2 headline analog) + hbb_hcc + wz (the ceiling
anchors). Frozen probes (linear/pool) are NOT run here — they don't yet thread
num_extra_features through their encoder (see README TODO); finetune is the decisive cut.
"""

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'experiments', 'phase4'))

import tasks  # noqa: E402  (from phase4)

_JEPA_CFG = 'experiments/phase6_fullfeature/configs/pretrain_jepa_full.yaml'
_MAE_CFG = 'experiments/phase6_fullfeature/configs/pretrain_mae_full.yaml'
_FINETUNE_CFG = 'configs/train_lorentz_part_1m_ragged.yaml'


def parse_args():
    p = argparse.ArgumentParser(description="Phase 6 full-information rerun")
    p.add_argument('--data-dir', default='./data_1m_full')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--tasks', nargs='+', default=['full10', 'hbb_hcc', 'wz'],
                   choices=list(tasks.TASKS))
    p.add_argument('--stage', default='all', choices=['pretrain', 'downstream', 'all'])
    p.add_argument('--num-extra-features', type=int, default=None,
                   help='Override; otherwise read from <data-dir>/features.json.')
    p.add_argument('--jepa-tmpl', default='./logs/ParticleJEPA/best/jepa_full_seed{seed}_best.pt')
    p.add_argument('--mae-tmpl', default='./logs/LorentzParT/best/mae_full_seed{seed}.pt')
    p.add_argument('--output-dir', default='experiments/phase6_fullfeature/results')
    p.add_argument('--max-train', type=int, default=100_000)
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()


def resolve_num_extra(args):
    if args.num_extra_features is not None:
        return args.num_extra_features
    meta_path = os.path.join(args.data_dir, 'features.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return int(json.load(f)['num_extra_features'])
    raise SystemExit(f"No {meta_path}; pass --num-extra-features explicitly "
                     f"(build data with prepare_data.py --with-displacement --with-pid).")


def run(cmd, env, label):
    print(f"\n[run] {label}\n  {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {label}: {e} — logged, continuing.", flush=True)
        return False


def stage_pretrain(args, python, env):
    n_done = n_skip = n_fail = 0
    for seed in args.seeds:
        # JEPA — best-val encoder is the downstream init (jepa_full_seed{seed}_best.pt)
        jepa_ckpt = args.jepa_tmpl.format(seed=seed)
        if os.path.exists(jepa_ckpt):
            print(f"[skip] jepa_full_seed{seed} — exists", flush=True); n_skip += 1
        else:
            ok = run([python, 'scripts/pretrain_jepa.py', '--data-dir', args.data_dir,
                      '--config-path', _JEPA_CFG, '--seed', str(seed),
                      '--run-name', f'jepa_full_seed{seed}'], env, f'pretrain jepa seed{seed}')
            n_done += ok; n_fail += (not ok)

        # MAE
        mae_ckpt = args.mae_tmpl.format(seed=seed)
        if os.path.exists(mae_ckpt):
            print(f"[skip] mae_full_seed{seed} — exists", flush=True); n_skip += 1
        else:
            ok = run([python, 'scripts/pretrain_mae.py', '--data-dir', args.data_dir,
                      '--config-path', _MAE_CFG, '--seed', str(seed),
                      '--run-name', f'mae_full_seed{seed}'], env, f'pretrain mae seed{seed}')
            n_done += ok; n_fail += (not ok)
    return n_done, n_skip, n_fail


def stage_downstream(args, python, env, n_extra):
    n_done = n_skip = n_fail = 0
    os.makedirs(args.output_dir, exist_ok=True)
    for seed in args.seeds:
        encoders = [
            ('jepa',    args.jepa_tmpl.format(seed=seed)),
            ('mae',     args.mae_tmpl.format(seed=seed)),
            ('scratch', None),
        ]
        for task in args.tasks:
            for enc, weights in encoders:
                run_name = f'{task}_finetune_{enc}_seed{seed}'
                out_json = os.path.join(args.output_dir, f'{run_name}.json')
                if os.path.exists(out_json):
                    print(f"[skip] {run_name} — exists", flush=True); n_skip += 1; continue
                if weights is not None and not os.path.exists(weights):
                    print(f"[skip] {run_name} — encoder missing: {weights}", flush=True)
                    n_skip += 1; continue
                cmd = [python, 'experiments/phase4/train_task.py',
                       '--task', task, '--protocol', 'finetune',
                       '--data-dir', args.data_dir, '--run-name', run_name,
                       '--encoder-label', enc, '--seed', str(seed),
                       '--output-dir', args.output_dir, '--max-train', str(args.max_train),
                       '--num-extra-features', str(n_extra),
                       '--finetune-config', _FINETUNE_CFG]
                if weights is not None:
                    cmd += ['--weights', weights]
                ok = run(cmd, env, run_name)
                n_done += ok; n_fail += (not ok)
    return n_done, n_skip, n_fail


def main():
    args = parse_args()
    python = sys.executable
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    n_extra = resolve_num_extra(args)
    print(f"Phase 6: num_extra_features={n_extra}, data={args.data_dir}, "
          f"seeds={args.seeds}, tasks={args.tasks}, stage={args.stage}", flush=True)

    t0 = time.monotonic()
    totals = [0, 0, 0]
    if args.stage in ('pretrain', 'all'):
        for i, v in enumerate(stage_pretrain(args, python, env)):
            totals[i] += v
    if args.stage in ('downstream', 'all'):
        for i, v in enumerate(stage_downstream(args, python, env, n_extra)):
            totals[i] += v

    n_done, n_skip, n_fail = totals
    print(f"\n{'='*60}\nPhase 6: {n_done} run, {n_skip} skipped, {n_fail} failed "
          f"in {(time.monotonic()-t0)/60:.1f} min.\n"
          f"Aggregate with: python experiments/phase4/analyze_phase4.py "
          f"--results-dir {args.output_dir}", flush=True)


if __name__ == '__main__':
    main()
