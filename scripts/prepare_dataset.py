"""Prepare a compressed .npz dataset with spectral, wavelet, and labels arrays.

This script loads one or more `X_*.npy` and matching `y_*.npy` files (or a single pair),
concatenates them, computes CWT wavelet images using PyWavelets, and writes a
`dataset.npz` file consumable by the training pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from typing import List

from raman_amr.data import save_npz_with_wavelets


def find_pairs_in_dir(directory: Path) -> List[tuple[Path, Path]]:
    xs = sorted(directory.glob("X_*.npy"))
    ys = sorted(directory.glob("y_*.npy"))
    if len(xs) != len(ys):
        raise SystemExit("Mismatched number of X_*.npy and y_*.npy files in directory")
    return list(zip(xs, ys))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare .npz dataset with CWT wavelet images.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing X_*.npy and y_*.npy files (will concatenate)")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path")
    args = parser.parse_args()

    pairs = find_pairs_in_dir(args.input_dir)
    spectra_list = []
    labels_list = []
    for x_path, y_path in pairs:
        X = np.load(x_path)
        y = np.load(y_path)
        spectra_list.append(X)
        labels_list.append(y)

    spectra = np.concatenate(spectra_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    print(f"Loaded {len(spectra)} spectra. Computing wavelet images and saving to {args.output}...")
    save_npz_with_wavelets(spectra, labels, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
