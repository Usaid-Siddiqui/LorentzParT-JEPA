"""
Download the full JetClass dataset for the Phase 7 (100M) scaling study.

Primary backend is the HuggingFace mirror `jet-universe/JetClass` (robust, resumable). The
Zenodo record (6619768) is the canonical source if you prefer wget/zenodo_get.

    # full training set (100M jets, ~hundreds of GB — verify free disk first):
    python scripts/download_jetclass.py --split train --out-dir /data/jetclass

    # smaller splits for pipeline validation:
    python scripts/download_jetclass.py --split val  --out-dir /data/jetclass   # 5M
    python scripts/download_jetclass.py --split test --out-dir /data/jetclass   # 20M

Verifies the ROOT file count per split afterwards. Budget >= 1 TB working disk for train + space.
"""

import argparse
import glob
import os
import shutil
import sys

# expected ROOT-file counts (100k jets/file) — sanity check after download
EXPECTED = {'train': 1000, 'val': 50, 'test': 200}
HF_REPO = 'jet-universe/JetClass'
HF_SUBDIR = {'train': 'train_100M', 'val': 'val_5M', 'test': 'test_20M'}


def parse_args():
    p = argparse.ArgumentParser(description="Download JetClass for Phase 7")
    p.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    p.add_argument('--out-dir', required=True, help='destination root (a <split>/ subdir is created)')
    p.add_argument('--backend', default='hf', choices=['hf', 'zenodo'])
    p.add_argument('--workers', type=int, default=8, help='parallel download workers (hf)')
    return p.parse_args()


def check_disk(out_dir, split):
    need_gb = {'train': 900, 'val': 50, 'test': 190}[split]   # rough; verify on Zenodo
    free_gb = shutil.disk_usage(os.path.dirname(os.path.abspath(out_dir)) or '.').free / 1e9
    print(f"free disk: {free_gb:.0f} GB · rough need for '{split}': ~{need_gb} GB")
    if free_gb < need_gb * 1.1:
        print("  [warn] low free disk for this split — confirm before proceeding.")


def download_hf(split, out_dir, workers):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed → `pip install huggingface_hub`, or use --backend zenodo.")
    subdir = HF_SUBDIR[split]
    print(f"downloading {HF_REPO}:{subdir}/ → {out_dir} (resumable)")
    snapshot_download(repo_id=HF_REPO, repo_type='dataset', local_dir=out_dir,
                      allow_patterns=[f"{subdir}/*"], max_workers=workers)
    return os.path.join(out_dir, subdir)


def download_zenodo(split, out_dir):
    print("Zenodo path (manual): record 6619768.")
    print("  pip install zenodo_get && zenodo_get 6619768 -o", out_dir)
    print(f"  then extract JetClass_Pythia_{split}_*.tar into {out_dir}/{split}/")
    sys.exit(0)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    check_disk(args.out_dir, args.split)

    if args.backend == 'zenodo':
        download_zenodo(args.split, args.out_dir)
    data_dir = download_hf(args.split, args.out_dir, args.workers)

    n = len(glob.glob(os.path.join(data_dir, '*.root')))
    exp = EXPECTED[args.split]
    print(f"\n{args.split}: {n} ROOT files in {data_dir}  (expected ~{exp})")
    if n < exp:
        print(f"  [warn] fewer files than expected — download may be incomplete; re-run (resumable).")
    else:
        print("  OK.")


if __name__ == '__main__':
    main()
