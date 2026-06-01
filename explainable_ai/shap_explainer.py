"""Simple SHAP wrapper utilities for spectral data.

Provides a small convenience function around the `shap` library to
explain individual instances using `KernelExplainer` (model-agnostic)
or `shap.Explainer` when available.
"""
from typing import Callable, Optional, Dict, Any, Sequence

import numpy as np


def explain_instance_shap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    background: np.ndarray,
    instance: np.ndarray,
    explainer: str = "auto",
    nsamples: int = 100,
    random_state: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Explain `instance` using SHAP.

    Args:
        predict_fn: function mapping (N, n_features) -> (N,) or (N, n_classes).
        background: background dataset for the explainer (e.g., subset of X).
        instance: single instance or array of instances to explain.
        explainer: 'auto', 'kernel', or 'shap' to pick specific explainer.
        nsamples: number of samples for Kernel SHAP (if used).
        random_state: seed passed to SHAP explainer where supported.
        feature_names: optional list of feature names.

    Returns:
        Dict with keys: `shap_values`, `expected_value`, `feature_names`, `instance`.
    """
    background = np.asarray(background)
    instance = np.atleast_2d(instance)

    try:
        import shap
    except Exception as e:
        raise RuntimeError("shap package is required for SHAP explanations") from e

    # prefer the unified shap.Explainer when available and requested
    use_shap_explainer = False
    if explainer == "shap":
        use_shap_explainer = True
    elif explainer == "auto":
        # shap.Explainer is available in newer versions; try to use it
        use_shap_explainer = hasattr(shap, "Explainer")

    if use_shap_explainer:
        # let shap choose best explainer for the model + data
        expl = shap.Explainer(predict_fn, background)
        shap_values = expl(instance)
        # shap_values may be a shap._explanation.Explanation object
        result = {
            "shap_values": shap_values.values if hasattr(shap_values, "values") else shap_values,
            "expected_value": getattr(shap_values, "base_values", None),
            "feature_names": feature_names,
            "instance": instance,
        }
        return result

    # fallback to KernelExplainer
    if not hasattr(shap, "KernelExplainer"):
        raise RuntimeError("shap.KernelExplainer not available in this shap installation")

    # shap.KernelExplainer expects a function that maps samples to a 1D/2D array.
    masker = None
    try:
        # try to use maskers if available (shap 0.39+)
        masker = shap.maskers.Independent(background)
    except Exception:
        masker = background

    expl = shap.KernelExplainer(predict_fn, masker)
    # nsamples controls runtime; higher -> more accurate
    shap_values = expl.shap_values(instance, nsamples=nsamples)

    return {
        "shap_values": shap_values,
        "expected_value": expl.expected_value,
        "feature_names": feature_names,
        "instance": instance,
    }


def top_shap_features(explanation: Dict[str, Any], k: int = 10) -> Dict[str, np.ndarray]:
    """Return indices of top-k features by absolute SHAP value for the first instance.

    Handles both single-output and multi-output SHAP values.
    """
    sv = explanation["shap_values"]
    # convert to numpy array of shape (n_outputs, n_instances, n_features) or (n_instances, n_features)
    if isinstance(sv, list):
        # list per class
        sv_arr = np.array([np.asarray(s) for s in sv])
        # pick first output/class for top features
        vals = sv_arr[:, 0, :].mean(axis=0)
    else:
        sv_arr = np.asarray(sv)
        if sv_arr.ndim == 3:
            # (n_instances, n_outputs, n_features) or (n_outputs, n_instances, n_features)
            # try to find features for first instance
            vals = np.abs(sv_arr[0]).mean(axis=0) if sv_arr.shape[0] > 0 else np.abs(sv_arr).mean(axis=0)
        elif sv_arr.ndim == 2:
            vals = np.abs(sv_arr[0])
        else:
            vals = np.abs(sv_arr).ravel()

    idx = np.argsort(-vals)[:k]
    return {"top_indices": idx, "scores": vals[idx]}
