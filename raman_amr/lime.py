"""A very basic LIME-like explainer for 1D spectral data.

This implements a minimal local surrogate linear model by perturbing an
input instance with Gaussian noise, weighting perturbed samples by
an exponential kernel of the distance to the instance, and fitting a
weighted linear model to approximate the model's output locally.

This is intentionally small and dependency-light for demonstration and
educational use.
"""
from typing import Callable, Optional, Dict, Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import euclidean_distances


def explain_instance(
    instance: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    num_samples: int = 500,
    kernel_width: Optional[float] = None,
    feature_std: Optional[float] = None,
    random_state: Optional[int] = None,
    target: Optional[int] = None,
    ridge_alpha: float = 1.0,
) -> Dict[str, Any]:
    """Explain a single instance using a simple LIME-style surrogate.

    Args:
        instance: 1D array of shape (n_features,) to explain.
        predict_fn: function mapping an array of shape (N, n_features)
            to either shape (N,) (regression) or (N, n_classes)
            (classification probabilities or scores).
        num_samples: number of perturbed samples to generate.
        kernel_width: kernel width for the exponential kernel. If None,
            set to sqrt(n_features) * 0.75.
        feature_std: standard deviation for Gaussian perturbation. If None,
            set to 0.1 * (max(instance)-min(instance)).
        random_state: seed for reproducibility.
        target: for multi-class outputs, the class index to explain. If None,
            and predict_fn returns multi-column output, the predicted class
            for the instance is used.
        ridge_alpha: regularization strength for the local surrogate.

    Returns:
        A dict with keys: `coefficients`, `intercept`, `local_prediction`,
        `weights`, and `samples` for optional inspection.
    """
    rng = np.random.RandomState(random_state)
    instance = np.asarray(instance).ravel()
    n_features = instance.shape[0]

    if feature_std is None:
        span = np.max(instance) - np.min(instance)
        feature_std = span * 0.1 if span > 0 else 1e-6

    # create perturbed samples around the instance
    samples = np.repeat(instance[np.newaxis, :], num_samples, axis=0)
    noise = rng.normal(loc=0.0, scale=feature_std, size=samples.shape)
    samples = samples + noise

    # distances and kernel weights
    distances = euclidean_distances(samples, instance.reshape(1, -1)).ravel()
    if kernel_width is None:
        kernel_width = np.sqrt(n_features) * 0.75
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

    # get the model outputs for the perturbed samples
    preds = predict_fn(samples)
    preds = np.asarray(preds)

    # choose target dimension if needed
    if preds.ndim == 2:
        if target is None:
            # pick the predicted class for the original instance
            orig_pred = predict_fn(instance.reshape(1, -1))
            orig_pred = np.asarray(orig_pred)
            if orig_pred.ndim == 2:
                target = int(np.argmax(orig_pred[0]))
            else:
                # fallback to first column
                target = 0
        y = preds[:, target]
    else:
        y = preds.ravel()

    # fit a weighted linear surrogate (Ridge) on the raw features
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(samples, y, sample_weight=weights)

    # local prediction at the instance (surrogate)
    local_pred = ridge.predict(instance.reshape(1, -1))[0]

    return {
        "coefficients": ridge.coef_.ravel(),
        "intercept": float(ridge.intercept_),
        "local_prediction": float(local_pred),
        "weights": weights,
        "samples": samples,
    }


def top_features(explanation: Dict[str, Any], k: int = 10) -> Dict[str, np.ndarray]:
    """Return indices of top positive and negative contributing features.

    Args:
        explanation: dict returned by `explain_instance`.
        k: number of top features to return for positive and negative.

    Returns:
        Dict with keys `positive` and `negative` containing feature indices.
    """
    coefs = np.asarray(explanation["coefficients"]).ravel()
    pos_idx = np.argsort(-coefs)[:k]
    neg_idx = np.argsort(coefs)[:k]
    return {"positive": pos_idx, "negative": neg_idx}
