"""
Evaluation metrics for the 1D Swin Transformer classifier.
Includes accuracy, macro-F1, and Expected Calibration Error (ECE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict


class ECELoss(nn.Module):
    """
    Expected Calibration Error (ECE).

    Bins predictions by confidence and computes the weighted average of the
    absolute difference between accuracy and confidence within each bin.

    Args:
        n_bins: Number of equal-width confidence bins.
    """

    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins = n_bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.register_buffer("bin_lowers", bin_boundaries[:-1])
        self.register_buffer("bin_uppers", bin_boundaries[1:])

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ECE from raw logits and ground-truth labels.

        Args:
            logits: (N, C) raw model outputs.
            labels: (N,) ground-truth class indices.

        Returns:
            Scalar tensor with the ECE value.
        """
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.squeeze()


def compute_metrics(
    all_logits: torch.Tensor,
    all_labels: torch.Tensor,
    n_bins: int = 15,
) -> Dict[str, float]:
    """
    Compute accuracy, macro-F1, and ECE from accumulated logits and labels.

    Args:
        all_logits: (N, C) tensor of raw logits.
        all_labels: (N,) tensor of ground-truth labels.
        n_bins: Number of bins for ECE calculation.

    Returns:
        Dictionary with keys 'accuracy', 'macro_f1', 'ece'.
    """
    preds = all_logits.argmax(dim=1).cpu().numpy()
    labels_np = all_labels.cpu().numpy()

    acc = float(accuracy_score(labels_np, preds))
    macro_f1 = float(f1_score(labels_np, preds, average="macro", zero_division=0))

    ece_fn = ECELoss(n_bins=n_bins)
    # Move buffers to same device as logits
    ece_fn = ece_fn.to(all_logits.device)
    ece_val = float(ece_fn(all_logits, all_labels).item())

    return {"accuracy": acc, "macro_f1": macro_f1, "ece": ece_val}
