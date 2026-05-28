from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def flatten_multimodal_features(spectral: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
    """Flatten and concatenate the 1D and 2D representations.

    Expected shapes:
    - spectral: (n_samples, n_features)
    - wavelet: (n_samples, height, width, channels) or (n_samples, channels, height, width)
    """

    spectral = np.asarray(spectral)
    wavelet = np.asarray(wavelet)

    if wavelet.ndim == 4 and wavelet.shape[1] in {1, 3}:
        wavelet = np.moveaxis(wavelet, 1, -1)

    spectral_flat = spectral.reshape(spectral.shape[0], -1)
    wavelet_flat = wavelet.reshape(wavelet.shape[0], -1)
    return np.concatenate([spectral_flat, wavelet_flat], axis=1)


@dataclass
class RamanSVMClassifier:
    """RBF-kernel SVM baseline from the paper."""

    c: float = 10.0
    gamma: str = "scale"
    probability: bool = False

    def __post_init__(self) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svm", SVC(C=self.c, gamma=self.gamma, kernel="rbf", probability=self.probability)),
            ]
        )

    def fit(self, spectral: np.ndarray, wavelet: np.ndarray, labels: np.ndarray) -> "RamanSVMClassifier":
        features = flatten_multimodal_features(spectral, wavelet)
        self.pipeline.fit(features, labels)
        return self

    def predict(self, spectral: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
        features = flatten_multimodal_features(spectral, wavelet)
        return self.pipeline.predict(features)

    def score(self, spectral: np.ndarray, wavelet: np.ndarray, labels: np.ndarray) -> float:
        features = flatten_multimodal_features(spectral, wavelet)
        return self.pipeline.score(features, labels)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "RamanSVMClassifier":
        instance = cls()
        instance.pipeline = joblib.load(str(path))
        return instance
