"""
Post-hoc: add embedding-collapse diagnostics to every condition of a run_ablation
results dir, so collapse geometry can be compared ACROSS the ablation (not just the
one inherited Phase 0 cell).

run_ablation.py doesn't compute embedding stats; the pretrain encoders are already
saved, so this just loads each and measures effective rank / variance / cosine sim
on the val set, then merges `embedding_stats` into the matching condition in the
seed JSONs. No retraining.

Note: this measures the ENCODER representation (gate is applied at JEPA runtime,
not inside LorentzParTEncoder) — i.e. "what the encoder learned", evaluated on full
unmasked jets. State that caveat if you report gate-vs-no-gate geometry.

    python experiments/phase1/add_embedding_stats.py --results-dir experiments/phase1/results --seeds 42 123 456 --gpu 0

Idempotent: skips conditions that already have embedding_stats (use --force to recompute).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))           # repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'phase0'))

import torch

from run_phase0 import get_embedding_stats

# Pretrain-encoder checkpoint stem per condition. no_gate_random is the reused
# Phase 0 JEPA pretrain (its encoder lives under jepa_seed*, not no_gate_random_seed*).
CKPT_STEM = {
    'no_gate_random': 'jepa',
    'no_gate_biased': 'no_gate_biased',
    'gate_random':    'gate_random',
    'gate_biased':    'gate_biased',
}


def parse_args():
    p = argparse.ArgumentParser(description="Add embedding-collapse stats to a 2x2 results dir")
    p.add_argument('--results-dir', default='./experiments/phase1/results')
    p.add_argument('--data-dir',    default='./data')
    p.add_argument('--seeds',       nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--gpu',         type=int, default=0)
    p.add_argument('--force', action='store_true', help='Recompute even if stats exist')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    for seed in args.seeds:
        path = os.path.join(args.results_dir, f'seed_{seed}.json')
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found")
            continue
        results = json.load(open(path))

        for cond, c in results['conditions'].items():
            if 'embedding_stats' in c and not args.force:
                print(f"seed {seed}  {cond}: stats present, skipping")
                continue
            stem = CKPT_STEM.get(cond, cond)
            ckpt = f'./logs/ParticleJEPA/best/{stem}_seed{seed}.pt'
            if not os.path.exists(ckpt):
                print(f"seed {seed}  {cond}: [SKIP] no pretrain ckpt {ckpt}")
                continue
            stats = get_embedding_stats(ckpt, args.data_dir, device)
            c['embedding_stats'] = stats
            print(f"seed {seed}  {cond}: eff_rank={stats['effective_rank']:.2f}  "
                  f"var={stats['mean_var']:.3f}  cos_sim={stats['mean_cos_sim']:.3f}")

        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Updated {path}\n")

    print("Done. Re-run analyze_results.py to regenerate the embedding_stats table/figure.")


if __name__ == '__main__':
    main()
