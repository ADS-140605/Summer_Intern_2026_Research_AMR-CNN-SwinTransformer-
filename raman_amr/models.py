from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def count_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class Conv1DBranch(nn.Module):
    """Lightweight 1D encoder for raw Raman spectra."""

    def __init__(self, in_channels: int = 1, feature_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        return x.flatten(1)


class Conv2DBranch(nn.Module):
    """Lightweight 2D encoder for wavelet images."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.features(x)
        return x.flatten(1)


class MultimodalRamanCNN(nn.Module):
    """Dual-branch multimodal CNN from the paper."""

    def __init__(
        self,
        num_classes: int = 3,
        spectral_channels: int = 1,
        wavelet_channels: int = 3,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        dropout_branch: float = 0.2,
        dropout_head: float = 0.4,
        dropout_logits: float = 0.3,
    ) -> None:
        super().__init__()
        self.spectral_branch = Conv1DBranch(spectral_channels, feature_dim, dropout_branch)
        self.wavelet_branch = Conv2DBranch(wavelet_channels, feature_dim, dropout_branch)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_head),
            nn.Dropout(dropout_logits),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, spectral: torch.Tensor, wavelet: torch.Tensor) -> torch.Tensor:
        spectral_features = self.spectral_branch(spectral)
        wavelet_features = self.wavelet_branch(wavelet)
        fused = torch.cat([spectral_features, wavelet_features], dim=1)
        return self.classifier(fused)


class RamanOnlyCNN(nn.Module):
    """Single-modality baseline using only raw spectra."""

    def __init__(self, num_classes: int = 3, spectral_channels: int = 1, feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = Conv1DBranch(spectral_channels, feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, spectral: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(spectral))


class WaveletOnlyCNN(nn.Module):
    """Single-modality baseline using only CWT images."""

    def __init__(self, num_classes: int = 3, wavelet_channels: int = 3, feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = Conv2DBranch(wavelet_channels, feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, wavelet: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(wavelet))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameters: int
