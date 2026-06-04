"""
Simple runner for the novel-research scaffold.
Usage:
  - Load available Raman datasets from `data/ramanspy/`.
  - Run a quick nearest-centroid baseline for a smoke test.

This file is intentionally minimal and dependency-light.
"""

import os
import argparse
import numpy as np
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ramanspy")


def load_pair(prefix):
    """Load X and y files with a common prefix (e.g. 'X_2018clinical.npy'/'y_2018clinical.npy')."""
    x_path = os.path.join(DATA_DIR, prefix)
    if not os.path.exists(x_path):
        return None, None
    # infer y filename by replacing leading X_ with y_
    y_path = x_path.replace("/X_", "/y_").replace("\\X_", "\\y_")
    try:
        X = np.load(x_path)
    except Exception:
        return None, None
    if os.path.exists(y_path):
        y = np.load(y_path)
    else:
        y = None
    return X, y


def find_datasets():
    """Return list of dataset file basenames found in DATA_DIR."""
    if not os.path.isdir(DATA_DIR):
        return []
    files = os.listdir(DATA_DIR)
    prefixes = set()
    for f in files:
        if f.startswith("X_") and f.endswith(".npy"):
            prefixes.add(f)
    return sorted(prefixes)


def nearest_centroid_baseline(X, y):
    """Train and evaluate a nearest-centroid classifier on (X,y)."""
    classes = np.unique(y)
    centroids = {}
    for c in classes:
        centroids[c] = X[y == c].mean(axis=0)
    preds = np.array([min(classes, key=lambda c: np.linalg.norm(x - centroids[c])) for x in X])
    acc = (preds == y).mean()
    return acc


def run_demo(dataset_prefix=None):
    """Load a dataset and run the smoke baseline; print summary."""
    prefixes = find_datasets()
    if not prefixes:
        print("No datasets found in", DATA_DIR)
        return
    if dataset_prefix is None:
        # prefer reference or 2018clinical if present
        for prefer in ("X_reference.npy", "X_2018clinical.npy", prefixes[0]):
            if prefer in prefixes:
                dataset_prefix = prefer
                break
    if dataset_prefix not in prefixes:
        print(f"Dataset {dataset_prefix} not found; available: {prefixes}")
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
