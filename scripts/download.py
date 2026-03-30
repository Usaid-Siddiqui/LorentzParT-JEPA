"""
Download the JetClass val_5M dataset (7.6 GB) from Zenodo.

This is the only split needed for the JEPA vs MAE demo — prepare_data.py
carves the 100k subset directly from these ROOT files.

Usage:
    python scripts/download.py --output-dir ./val_5M
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data import download_jetclass_data, extract_tar

VAL_URL = 'https://zenodo.org/records/6619768/files/JetClass_Pythia_val_5M.tar?download=1'


def parse_args():
    parser = argparse.ArgumentParser(description="Download JetClass val_5M dataset")
    parser.add_argument('--output-dir', type=str, default='./val_5M',
                        help="Directory to download and extract the dataset into")
    parser.add_argument('--timeout', type=int, default=7200,
                        help="Download timeout in seconds (default 2 hours)")
    parser.add_argument('--chunk-size', type=int, default=1024 * 1024 * 10,
                        help="Download chunk size in bytes (default 10 MB)")
    parser.add_argument('--keep-tar', action='store_true',
                        help="Keep the .tar archive after extraction")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading JetClass val_5M (~7.6 GB) to {args.output_dir}/")
    tar_path = download_jetclass_data(
        VAL_URL, args.output_dir, args.timeout, args.chunk_size
    )
    extract_tar(tar_path, args.output_dir, remove_tar=not args.keep_tar)

    print(f"\nDone. ROOT files are in: {args.output_dir}/")
    print("Next step:")
    print(f"  python scripts/prepare_data.py --data-dir {args.output_dir} --output-dir ./data")


if __name__ == '__main__':
    main()
