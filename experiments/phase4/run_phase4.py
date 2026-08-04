"""
Phase 4 orchestrator — task × protocol × encoder × seed, resumable + error-isolated.

Shells out to train_task.py per cell. A cell whose JSON exists is skipped (resume after
a crash — see the pod-restart lesson); a failed cell is logged and the sweep continues.
Mirrors run_phase2.py's hardening.

    python experiments/phase4/run_phase4.py --data-dir ./data_1m --seeds 42 123 456

By default runs the frozen protocols (linear + pool) and finetune for the 3 encoders.
Drop 'pool' with --protocols finetune linear if the attentive gate failed.
"""

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)   # for src.* (tasks.py imports it)
sys.path.insert(0, _HERE)

import tasks
import protocols


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 downstream transfer sweep")
    p.add_argument('--data-dir', default='./data_1m')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--tasks', nargs='+', default=list(tasks.TASKS), choices=list(tasks.TASKS))
    p.add_argument('--protocols', nargs='+', default=['finetune', 'linear', 'pool'],
                   choices=list(protocols.PROTOCOLS))
    p.add_argument('--jepa-tmpl', default='./logs/ParticleJEPA/best/jepa_1m_ragged_seed{seed}_best.pt')
    p.add_argument('--mae-tmpl', default='./logs/LorentzParT/best/mae_1m_ragged_seed{seed}.pt')
    p.add_argument('--output-dir', default='experiments/phase4/results')
    p.add_argument('--max-train', type=int, default=100_000)
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    python = sys.executable
    os.makedirs(args.output_dir, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    n_done = n_skip = n_fail = 0
    t0 = time.monotonic()
    for seed in args.seeds:
        encoders = [
            ('jepa',    args.jepa_tmpl.format(seed=seed)),
            ('mae',     args.mae_tmpl.format(seed=seed)),
            ('scratch', None),
        ]
        for task in args.tasks:
            for proto in args.protocols:
                for enc, weights in encoders:
                    run_name = f'{task}_{proto}_{enc}_seed{seed}'
                    out_json = os.path.join(args.output_dir, f'{run_name}.json')
                    if os.path.exists(out_json):
                        print(f"[skip] {run_name} — exists", flush=True)
                        n_skip += 1
                        continue
                    if weights is not None and not os.path.exists(weights):
                        print(f"[skip] {run_name} — encoder missing: {weights}", flush=True)
                        n_skip += 1
                        continue
                    cmd = [python, 'experiments/phase4/train_task.py',
                           '--task', task, '--protocol', proto,
                           '--data-dir', args.data_dir, '--run-name', run_name,
                           '--encoder-label', enc,
                           '--seed', str(seed), '--output-dir', args.output_dir,
                           '--max-train', str(args.max_train)]
                    if weights is not None:
                        cmd += ['--weights', weights]
                    print(f"\n[run] {run_name}\n  {' '.join(cmd)}", flush=True)
                    try:
                        subprocess.run(cmd, env=env, check=True)
                        n_done += 1
                    except subprocess.CalledProcessError as e:
                        print(f"[FAILED] {run_name}: {e} — logged, continuing.", flush=True)
                        n_fail += 1

    print(f"\n{'='*60}\nPhase 4 sweep: {n_done} run, {n_skip} skipped, {n_fail} failed "
          f"in {(time.monotonic()-t0)/60:.1f} min.\n"
          f"Aggregate with: python experiments/phase4/analyze_phase4.py "
          f"--results-dir {args.output_dir}", flush=True)


if __name__ == '__main__':
    main()
