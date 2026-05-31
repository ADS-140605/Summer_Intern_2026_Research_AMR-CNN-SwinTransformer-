"""Example: train a simple classifier and explain one instance with the tiny LIME.

Usage:
    python scripts/explain_lime.py --index 0
"""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression

from explainable_ai.lime import explain_instance, top_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="which test instance to explain")
    parser.add_argument("--num-samples", type=int, default=800, help="number of perturbations")
    args = parser.parse_args()

    # load data
    X = np.load("data/ramanspy/X_test.npy")
    y = np.load("data/ramanspy/y_test.npy")
    wavenumbers = np.load("data/ramanspy/wavenumbers.npy")

    # simple classifier trained on the test set (for demonstration only)
    # in a real workflow, replace with your trained model
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)

    idx = args.index
    instance = X[idx]

    # define predict function that returns probabilities
    def predict_fn(arr):
        return clf.predict_proba(arr)

    explanation = explain_instance(instance, predict_fn, num_samples=args.num_samples, random_state=0)
    pred = clf.predict_proba(instance.reshape(1, -1))[0]
    predicted_class = int(np.argmax(pred))

    print(f"Explaining instance {idx}, predicted class: {predicted_class}, probs: {pred}")

    tf = top_features(explanation, k=10)

    print("Top positive contributions (feature idx, wavenumber, coef):")
    for i in tf["positive"]:
        print(i, wavenumbers[i], explanation["coefficients"][i])

    print("Top negative contributions (feature idx, wavenumber, coef):")
    for i in tf["negative"]:
        print(i, wavenumbers[i], explanation["coefficients"][i])


if __name__ == "__main__":
    main()
