"""Example: train a simple classifier and explain one instance with SHAP.

Usage:
    python scripts/explain_shap.py --index 0
"""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression

from explainable_ai.shap_explainer import explain_instance_shap, top_shap_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="which test instance to explain")
    parser.add_argument("--nsamples", type=int, default=200, help="number of SHAP samples for KernelExplainer")
    args = parser.parse_args()

    # load data
    X = np.load("data/ramanspy/X_test.npy")
    y = np.load("data/ramanspy/y_test.npy")
    wavenumbers = np.load("data/ramanspy/wavenumbers.npy")

    # simple classifier trained on the test set (for demonstration only)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)

    idx = args.index
    instance = X[idx]

    # predict_fn should return probability for each class
    def predict_fn(arr):
        return clf.predict_proba(arr)

    # use a small background set sampled from X
    bg = X[np.random.RandomState(0).choice(len(X), size=min(50, len(X)), replace=False)]

    explanation = explain_instance_shap(predict_fn, bg, instance, explainer="auto", nsamples=args.nsamples)
    tf = top_shap_features(explanation, k=10)

    print(f"Explaining instance {idx}")
    print("Top features by absolute SHAP value (idx, wavenumber, score):")
    for i, score in zip(tf["top_indices"], tf["scores"]):
        print(i, wavenumbers[i], score)


if __name__ == "__main__":
    main()
