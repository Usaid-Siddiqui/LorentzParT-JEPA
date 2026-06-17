"""
Prepare a balanced JetClass subset (numpy) for the JEPA vs MAE experiments.

Reads ROOT files per class from the val_5M directory, takes a configurable number
of events per class, and splits into balanced train / val / test sets. Each class
has several files (e.g. HToBB_120.root … HToBB_124.root, 100k events each); this
reads as many as needed, in sorted order, to cover the requested per-class count.

Defaults reproduce the original 100k subset (8k/1k/1k per class = 80k/10k/10k):
    python scripts/prepare_data.py --data-dir /path/to/val_5M --output-dir ./data

1M subset (100k/10k/10k per class = 1M/100k/100k, reads ~2 files/class):
    python scripts/prepare_data.py --data-dir /path/to/val_5M --output-dir ./data_1m \\
        --train-per-class 100000 --val-per-class 10000 --test-per-class 10000

Output: numpy .npy files in <output-dir>/{train,val,test}/
  - particles.npy  : (N, 4, 128) float32
  - labels.npy     : (N, 10)     int32
"""

import os
import glob
import argparse
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.data.dataloader import read_file


# Maps class index → source ROOT filename prefix in val_5M.
# Each prefix expands to {prefix}_*.root (sorted); the '_' guards TTBar vs TTBarLep.
CLASS_PREFIXES = {
    0: "ZJetsToNuNu",   # label_QCD  (background, q/g jets)
    1: "HToBB",          # label_Hbb
    2: "HToCC",          # label_Hcc
    3: "HToGG",          # label_Hgg
    4: "HToWW4Q",        # label_H4q
    5: "HToWW2Q1L",      # label_Hqql
    6: "ZToQQ",          # label_Zqq
    7: "WToQQ",          # label_Wqq
    8: "TTBar",          # label_Tbqq
    9: "TTBarLep",       # label_Tbl
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a balanced JetClass numpy subset")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to val_5M directory containing ROOT files")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output directory for processed numpy files")
    parser.add_argument("--train-per-class", type=int, default=8_000)
    parser.add_argument("--val-per-class",   type=int, default=1_000)
    parser.add_argument("--test-per-class",  type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")
    parser.add_argument("--max-particles", type=int, default=128,
                        help="Maximum number of particles per jet")
    return parser.parse_args()


def load_class(data_dir: str, prefix: str, n_events: int, max_particles: int):
    """Read sorted {prefix}_*.root files until n_events collected; return (particles, label)."""
    files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.root")))
    if not files:
        raise FileNotFoundError(f"No files matching {prefix}_*.root in {data_dir}")

    xs, ys, collected = [], [], 0
    for fp in files:
        if collected >= n_events:
            break
        print(f"  Reading {os.path.basename(fp)} ...", flush=True)
        x_particles, _, y = read_file(fp, max_num_particles=max_particles)
        xs.append(x_particles.astype(np.float32))
        ys.append(y.astype(np.int32))
        collected += len(x_particles)

    x = np.concatenate(xs, axis=0)[:n_events]
    y = np.concatenate(ys, axis=0)[:n_events]
    if len(x) < n_events:
        raise ValueError(
            f"{prefix}: only {len(x)} events available across {len(files)} file(s), "
            f"need {n_events}. Add more source files or lower the per-class counts.")
    print(f"    {prefix}: {len(x)} events from {len(xs)} file(s), shape {x.shape}")
    return x, y


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    events_per_class = args.train_per_class + args.val_per_class + args.test_per_class

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

    all_train_x, all_train_y = [], []
    all_val_x, all_val_y = [], []
    all_test_x, all_test_y = [], []

    print(f"\nLoading {events_per_class} events per class from {args.data_dir}")
    print(f"Split: {args.train_per_class} train / {args.val_per_class} val / "
          f"{args.test_per_class} test per class\n")

    tr, va = args.train_per_class, args.val_per_class
    for class_idx, prefix in CLASS_PREFIXES.items():
        x, y = load_class(args.data_dir, prefix, events_per_class, args.max_particles)

        # Shuffle within class for random, disjoint train/val/test slices
        perm = rng.permutation(len(x))
        x, y = x[perm], y[perm]

        all_train_x.append(x[:tr]);            all_train_y.append(y[:tr])
        all_val_x.append(x[tr:tr + va]);       all_val_y.append(y[tr:tr + va])
        all_test_x.append(x[tr + va:]);        all_test_y.append(y[tr + va:])

    # Concatenate and shuffle each split globally
    for split_name, x_list, y_list in [
        ("train", all_train_x, all_train_y),
        ("val", all_val_x, all_val_y),
        ("test", all_test_x, all_test_y),
    ]:
        X = np.concatenate(x_list, axis=0)
        Y = np.concatenate(y_list, axis=0)
        perm = rng.permutation(len(X))
        X, Y = X[perm], Y[perm]

        out_dir = os.path.join(args.output_dir, split_name)
        np.save(os.path.join(out_dir, "particles.npy"), X)
        np.save(os.path.join(out_dir, "labels.npy"), Y)

        print(f"\n{split_name:5s}: {len(X):7d} jets  "
              f"| particles {X.shape}  | labels {Y.shape}")
        print(f"       Saved to {out_dir}/")

    print("\nDone. Class distribution (train set):")
    Y_train = np.concatenate(all_train_y, axis=0)
    class_names = [
        "QCD/ZNuNu", "H→bb", "H→cc", "H→gg", "H→4q",
        "H→ℓνqq", "Z→qq", "W→qq", "t→bqq", "t→bℓν"
    ]
    for i, name in enumerate(class_names):
        print(f"  Class {i:2d} ({name:10s}): {int(Y_train[:, i].sum()):6d}")


if __name__ == "__main__":
    main()
