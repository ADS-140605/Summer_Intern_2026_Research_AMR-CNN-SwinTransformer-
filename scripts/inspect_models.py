from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raman_amr import MultimodalRamanCNN, RamanOnlyCNN, WaveletOnlyCNN, count_parameters


def main() -> None:
    multimodal = MultimodalRamanCNN()
    spectral_only = RamanOnlyCNN()
    wavelet_only = WaveletOnlyCNN()

    spectral = torch.randn(2, 1000)
    wavelet = torch.randn(2, 3, 224, 224)

    print("Multimodal logits:", multimodal(spectral, wavelet).shape)
    print("Raman-only logits:", spectral_only(spectral).shape)
    print("Wavelet-only logits:", wavelet_only(wavelet).shape)
    print("Multimodal parameters:", count_parameters(multimodal))
    print("Raman-only parameters:", count_parameters(spectral_only))
    print("Wavelet-only parameters:", count_parameters(wavelet_only))


if __name__ == "__main__":
    main()
