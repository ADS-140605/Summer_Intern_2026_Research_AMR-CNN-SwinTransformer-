"""Robust SHAP utilities for spectral data (Kernel SHAP only).

This rewrite keeps behavior predictable for high-dimensional multi-class data
by always explaining a single target output (class probability).
"""
from typing import Callable, Optional, Dict, Any, Sequence

import numpy as np


def _pick_background(background: np.ndarray, max_background: int, random_state: Optional[int]) -> np.ndarray:
    """Downsample background for stable Kernel SHAP runtime."""
    background = np.asarray(background)
    if background.ndim != 2:
        raise ValueError("background must be 2D: (n_samples, n_features)")
    if background.shape[0] <= max_background:
        return background

    rng = np.random.RandomState(random_state)
    idx = rng.choice(background.shape[0], size=max_background, replace=False)
    return background[idx]


def explain_instance_shap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    background: np.ndarray,
    instance: np.ndarray,
    nsamples: int = 300,
    class_idx: Optional[int] = None,
    max_background: int = 30,
    random_state: Optional[int] = 0,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Explain one instance with Kernel SHAP.

    Args:
        predict_fn: function mapping (N, n_features) -> (N,) or (N, n_classes).
        background: background samples.
        instance: one sample with shape (n_features,) or (1, n_features).
        nsamples: Kernel SHAP sample budget.
        class_idx: target class for multi-class outputs. If None, predicted class is used.
        max_background: maximum number of background samples retained.
        random_state: random seed for background downsampling.
        feature_names: optional names for features.

    Returns:
        Dict containing one-dimensional SHAP attribution for the instance.
    """
    try:
        import shap
    except Exception as e:
        raise RuntimeError("shap package is required for SHAP explanations") from e

    x = np.atleast_2d(np.asarray(instance))
    if x.shape[0] != 1:
        raise ValueError("instance must represent exactly one sample")

    bg = _pick_background(np.asarray(background), max_background=max_background, random_state=random_state)

    # infer output shape on the target instance
    pred_x = np.asarray(predict_fn(x))
    if pred_x.ndim == 1:
        target_idx = None

        def wrapped_predict(arr: np.ndarray) -> np.ndarray:
            p = np.asarray(predict_fn(arr))
            return p.ravel()

    elif pred_x.ndim == 2:
        if class_idx is None:
            target_idx = int(np.argmax(pred_x[0]))
        else:
            target_idx = int(class_idx)

        def wrapped_predict(arr: np.ndarray) -> np.ndarray:
            p = np.asarray(predict_fn(arr))
            return p[:, target_idx]

    else:
        raise ValueError("predict_fn output must be 1D or 2D")

    explainer = shap.KernelExplainer(wrapped_predict, bg)
    shap_values = explainer.shap_values(x, nsamples=nsamples)
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 2:
        # expected shape for one instance is (1, n_features)
        values_1d = shap_values[0]
    else:
        values_1d = shap_values.ravel()

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = float(np.asarray(expected_value).ravel()[0])

    return {
        "shap_values": values_1d,
        "expected_value": float(expected_value),
        "feature_names": list(feature_names) if feature_names is not None else None,
        "instance": x[0],
        "class_idx": target_idx,
        "prediction": pred_x[0] if pred_x.ndim == 2 else float(pred_x.ravel()[0]),
    }


def top_shap_features(explanation: Dict[str, Any], k: int = 10) -> Dict[str, np.ndarray]:
    """Return top-k features by absolute SHAP value."""
    sv = np.asarray(explanation["shap_values"]).ravel()
    abs_sv = np.abs(sv)
    idx = np.argsort(-abs_sv)[:k]
    return {
        "top_indices": idx,
        "scores": sv[idx],
        "abs_scores": abs_sv[idx],
    }
