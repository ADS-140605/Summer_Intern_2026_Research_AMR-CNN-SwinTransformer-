from __future__ import annotations

import sys
from pathlib import Path

import torch
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raman_amr import MultimodalRamanCNN, RamanOnlyCNN, WaveletOnlyCNN, count_parameters


def main() -> None:
    multimodal = MultimodalRamanCNN()
    spectral_only = RamanOnlyCNN()
    wavelet_only = WaveletOnlyCNN()

    spectral = torch.randn(2, 1000)
    wavelet = torch.randn(2, 3, 224, 224)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Multimodal logits: %s", multimodal(spectral, wavelet).shape)
    logging.info("Raman-only logits: %s", spectral_only(spectral).shape)
    logging.info("Wavelet-only logits: %s", wavelet_only(wavelet).shape)
    logging.info("Multimodal parameters: %d", count_parameters(multimodal))
    logging.info("Raman-only parameters: %d", count_parameters(spectral_only))
    logging.info("Wavelet-only parameters: %d", count_parameters(wavelet_only))


if __name__ == "__main__":
    main()
