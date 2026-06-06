"""Minimal demo runner for quick smoke tests using the repository's Raman `.npy` files.

This version removes unused imports and makes label-path handling robust.
"""

import os
import argparse
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ramanspy")


def load_pair(prefix):
    """Load `X` and corresponding `y` file (if present) for a given `X_*.npy` prefix.

    Uses a reliable basename transformation to locate the `y_*.npy` file.
    """
    x_path = os.path.join(DATA_DIR, prefix)
    if not os.path.exists(x_path):
        return None, None
    # derive y filename from the X_ prefix
    y_fname = prefix.replace("X_", "y_", 1)
    y_path = os.path.join(DATA_DIR, y_fname)
    try:
        X = np.load(x_path)
    except Exception:
        return None, None
    y = np.load(y_path) if os.path.exists(y_path) else None
    return X, y


def find_datasets():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted([f for f in os.listdir(DATA_DIR) if f.startswith("X_") and f.endswith(".npy")])


def nearest_centroid_baseline(X, y):
    classes = np.unique(y)
    centroids = {c: X[y == c].mean(axis=0) for c in classes}
    # predict by finding nearest centroid for each sample
    preds = np.array([min(classes, key=lambda c: np.linalg.norm(x - centroids[c])) for x in X])
    acc = (preds == y).mean()
    return acc


def run_demo(dataset_prefix=None):
    prefixes = find_datasets()
    if not prefixes:
        print("No datasets found in", DATA_DIR)
        return
    if dataset_prefix is None:
        for prefer in ("X_reference.npy", "X_2018clinical.npy", prefixes[0]):
            if prefer in prefixes:
                dataset_prefix = prefer
                break
    if dataset_prefix not in prefixes:
        print("Dataset not found; available:")
        for p in prefixes:
            print(" -", p)
        return
    X, y = load_pair(dataset_prefix)
    if X is None:
        print("Failed to load dataset:", dataset_prefix)
        return
    print(f"Loaded {dataset_prefix}: X shape={X.shape}, y present={y is not None}")
    if y is None:
        print("No labels available for this dataset; demo ends.")
        return
    print("Running nearest-centroid baseline...")
    acc = nearest_centroid_baseline(X, y)
    print(f"Nearest-centroid accuracy (same-split): {acc:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Novel research demo runner")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--demo", action="store_true", help="Run smoke demo on a dataset")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset filename to use (e.g. X_reference.npy)")
    args = parser.parse_args()

    if args.list:
        prefixes = find_datasets()
        if not prefixes:
            print("No datasets found in", DATA_DIR)
            return
        print("Found datasets:")
        for p in prefixes:
            print(" -", p)
        return
    if args.demo:
        run_demo(args.dataset)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
