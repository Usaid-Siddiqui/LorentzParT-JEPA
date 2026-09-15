"""
Download a JetClass split from Zenodo (record 6619768) — the canonical source.

The HuggingFace repo `jet-universe/JetClass` is a README-only stub (no data files), so Zenodo is
the real source. JetClass train is 10 tarballs (~15 GB each ≈ 150 GB); val/test are single tarballs.
Zenodo throttles each connection to ~0.5 MB/s, so we download the tarballs IN PARALLEL (N streams ≈
N×), then extract and delete each tar. Resumable (`wget -c`) — re-run to continue after an interruption.

    python scripts/download_jetclass.py --split train --out-dir /data/jetclass   # ~150 GB, ~1000 .root
    python scripts/download_jetclass.py --split val   --out-dir /data/jetclass   # 5M jets
    python scripts/download_jetclass.py --split test  --out-dir /data/jetclass   # 20M jets

Needs `wget` and `tar` on PATH (no Python deps). Budget >= ~1.2× the split size in free disk (the
tar is deleted right after its extraction, so peak ≈ final size + one tar).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

RECORD = "6619768"
SPLIT_KEY = {"train": "train_100M", "val": "val_5M", "test": "test_20M"}   # substring in the tar name
EXPECTED_ROOT = {"train": 1000, "val": 50, "test": 200}                    # ~100k jets/file


def parse_args():
    p = argparse.ArgumentParser(description="Download a JetClass split from Zenodo")
    p.add_argument("--split", required=True, choices=list(SPLIT_KEY))
    p.add_argument("--out-dir", required=True, help="root; a <split_key>/ subdir gets the .root files")
    p.add_argument("--record", default=RECORD)
    p.add_argument("--jobs", type=int, default=10, help="parallel downloads (Zenodo throttles per stream)")
    p.add_argument("--no-extract", action="store_true", help="download tarballs only, don't extract")
    return p.parse_args()


def zenodo_tarballs(record, split):
    """[(filename, content-url), ...] for the split, from the Zenodo record API."""
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}") as r:
        rec = json.load(r)
    key = SPLIT_KEY[split]
    out = []
    for f in rec["files"]:
        name = f["key"]
        if key in name and name.endswith(".tar"):
            out.append((name, f"https://zenodo.org/api/records/{record}/files/{name}/content"))
    return sorted(out)


def wget(url, dest):
    subprocess.run(["wget", "-c", "-q", "-O", dest, url], check=True)
    return dest


def main():
    args = parse_args()
    key = SPLIT_KEY[args.split]
    split_dir = os.path.join(args.out_dir, key)
    os.makedirs(split_dir, exist_ok=True)

    files = zenodo_tarballs(args.record, args.split)
    if not files:
        sys.exit(f"No '{key}' tarballs in Zenodo record {args.record}.")
    print(f"{args.split}: {len(files)} tarball(s) → {args.out_dir} (parallel x{args.jobs}, resumable)")

    # ---- parallel download ----
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(wget, url, os.path.join(args.out_dir, name)): name for name, url in files}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                fut.result(); print(f"  downloaded {name}", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"  [FAILED] {name}: {e} — re-run to resume", flush=True)

    if args.no_extract:
        print("--no-extract set; tarballs left in", args.out_dir); return

    # ---- extract + clean (delete each tar after extraction to cap peak disk) ----
    for name, _ in files:
        tar = os.path.join(args.out_dir, name)
        if not os.path.exists(tar):
            print(f"  [skip extract] {name} missing"); continue
        print(f"  extracting {name} ...", flush=True)
        subprocess.run(["tar", "xf", tar, "-C", split_dir], check=True)
        os.remove(tar)

    # ---- flatten any nested dirs (loader globs *.root non-recursively) + verify ----
    for f in glob.glob(os.path.join(split_dir, "**", "*.root"), recursive=True):
        if os.path.dirname(f) != split_dir:
            shutil.move(f, os.path.join(split_dir, os.path.basename(f)))
    n = len(glob.glob(os.path.join(split_dir, "*.root")))
    exp = EXPECTED_ROOT[args.split]
    print(f"\n{args.split}: {n} ROOT files in {split_dir} (expected ~{exp})")
    print("  OK." if n >= exp else "  [warn] fewer than expected — re-run (resumable) to finish.")


if __name__ == "__main__":
    main()
