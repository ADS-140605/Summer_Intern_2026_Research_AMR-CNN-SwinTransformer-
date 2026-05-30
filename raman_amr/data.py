from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

try:
    import pywt
except Exception:  # pragma: no cover - optional dependency
    pywt = None


class RamanNpzDataset(Dataset):
    """Dataset backed by a `.npz` file with Raman tensors and labels.

    Expected arrays:
    - spectral: shape (N, 1000) or (N, 1, 1000)
    - wavelet: shape (N, 3, 224, 224) or (N, 224, 224, 3)
    - labels: shape (N,)
    """

    def __init__(
        self,
        npz_path: str | Path,
        spectral_key: str = "spectral",
        wavelet_key: str = "wavelet",
        labels_key: str = "labels",
    ) -> None:
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(path)

        data = np.load(path, allow_pickle=False)
        missing_keys = [key for key in (spectral_key, wavelet_key, labels_key) if key not in data]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise KeyError(f"Missing keys in {path}: {missing}")

        self.spectral = np.asarray(data[spectral_key], dtype=np.float32)
        self.wavelet = np.asarray(data[wavelet_key], dtype=np.float32)
        self.labels = np.asarray(data[labels_key], dtype=np.int64)

        if len(self.spectral) != len(self.wavelet) or len(self.spectral) != len(self.labels):
            raise ValueError("Spectral, wavelet, and label arrays must have the same length")

        if self.wavelet.ndim == 4 and self.wavelet.shape[-1] in {1, 3}:
            self.wavelet = np.moveaxis(self.wavelet, -1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        spectral = torch.from_numpy(self.spectral[index])
        wavelet = torch.from_numpy(self.wavelet[index])
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return spectral, wavelet, label


def stratified_split(
    dataset: Dataset,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset[Dataset], Subset[Dataset], Subset[Dataset]]:
    """Split a dataset into train/validation/test subsets with stratification."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    labels = np.asarray([int(dataset[index][2]) for index in range(len(dataset))])
    indices = np.arange(len(dataset))

    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )

    temp_labels = labels[temp_indices]
    if validation_ratio == 0:
        return Subset(dataset, train_indices.tolist()), Subset(dataset, [],), Subset(dataset, temp_indices.tolist())

    validation_size = validation_ratio / (1.0 - train_ratio)
    validation_indices, test_indices = train_test_split(
        temp_indices,
        train_size=validation_size,
        random_state=seed,
        stratify=temp_labels,
    )

    return (
        Subset(dataset, train_indices.tolist()),
        Subset(dataset, validation_indices.tolist()),
        Subset(dataset, test_indices.tolist()),
    )


def collate_multimodal(batch):
    spectral, wavelet, labels = zip(*batch)
    return torch.stack(list(spectral)), torch.stack(list(wavelet)), torch.stack(list(labels))


def compute_cwt_image(
    spectrum: np.ndarray, *,
    scales: int = 224,
    out_width: int = 224,
    wavelet: str = "morl",
) -> np.ndarray:
    """Compute a magnitude CWT image for a single 1D spectrum.

    Returns an array with shape (3, out_height, out_width) with channel-first layout
    and dtype float32. The 2D magnitude map is linearly scaled to [0, 1]
    and replicated across three channels to be compatible with the 2D branch.
    """
    if pywt is None:
        raise RuntimeError("PyWavelets is required for CWT computation. Install PyWavelets.")

    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.ndim != 1:
        spectrum = spectrum.flatten()

    # logarithmically spaced scales from 1..scales
    scales_arr = np.logspace(0, np.log10(scales), num=scales)
    coeffs, _ = pywt.cwt(spectrum, scales_arr, wavelet)
    magnitude = np.abs(coeffs)

    # resample time axis from original length to out_width
    orig_len = magnitude.shape[1]
    if orig_len == out_width:
        img = magnitude
    else:
        x_old = np.linspace(0.0, 1.0, orig_len)
        x_new = np.linspace(0.0, 1.0, out_width)
        img = np.vstack([np.interp(x_new, x_old, row) for row in magnitude])

    # normalize to [0,1]
    maxv = img.max() if img.size else 1.0
    img = img.astype(np.float32) / (float(maxv) + 1e-8)

    # ensure shape (H, W) -> (3, H, W)
    img3 = np.stack([img, img, img], axis=0).astype(np.float32)
    return img3


def build_wavelet_stack(spectra: np.ndarray, *, batch_size: int = 256) -> np.ndarray:
    """Compute CWT images for a batch of spectra.

    Returns an array of shape (N, 3, H, W) dtype float32.
    """
    spectra = np.asarray(spectra)
    images = []
    for i in range(0, len(spectra), batch_size):
        batch = spectra[i : i + batch_size]
        for spec in batch:
            images.append(compute_cwt_image(spec))

    return np.stack(images, axis=0)


def save_npz_with_wavelets(spectral: np.ndarray, labels: np.ndarray, out_path: str | Path) -> None:
    """Compute wavelet images for `spectral` and save a compressed .npz with keys
    `spectral`, `wavelet`, and `labels`.
    """
    spectral = np.asarray(spectral, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    wavelet = build_wavelet_stack(spectral)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, spectral=spectral, wavelet=wavelet, labels=labels)
