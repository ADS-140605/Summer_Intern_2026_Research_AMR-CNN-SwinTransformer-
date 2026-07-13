"""
Training and evaluation utilities for the Plain 1D Swin Transformer.

Provides:
    - train_one_epoch: single-epoch training loop
    - evaluate_model: evaluation loop returning loss, accuracy, macro-F1, ECE
    - Trainer: high-level wrapper for multi-epoch train/val with checkpointing
"""

import os
import json
import time
import random
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Hyperparameter config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All tuneable knobs live here -- nothing is hardcoded in the loops."""
    # Optimiser
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Schedule
    scheduler: str = "cosine"          # "cosine" | "step" | "none"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    step_size: int = 10
    step_gamma: float = 0.5

    # Training
    epochs: int = 50
    seed: int = 42

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True

    # Model architecture (forwarded to PlainSwin1D)
    patch_size: int = 10
    embed_dim: int = 32
    depths: List[int] = field(default_factory=lambda: [2, 2])
    num_heads: List[int] = field(default_factory=lambda: [2, 4])
    window_size: int = 8
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    # Data
    batch_size: int = 64
    data_dir: str = r"d:\AMR\data\ramanspy"
    num_classes: int = 30

    # Fine-tune
    finetune_epochs: int = 20
    finetune_lr: float = 5e-4


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Train / evaluate primitives
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one epoch of training.

    Args:
        model: The model to train.
        loader: Training DataLoader yielding (tokens, labels).
        optimizer: Optimiser instance.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for tokens, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(tokens)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on a dataset.

    Args:
        model: The model to evaluate.
        loader: Evaluation DataLoader.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Tuple of (avg_loss, accuracy, macro_f1, ece).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits = []
    all_labels = []

    for tokens, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        logits = model(tokens)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits)
        all_labels.append(labels)

    avg_loss = total_loss / max(n_batches, 1)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(all_logits, all_labels)
    return avg_loss, metrics["accuracy"], metrics["macro_f1"], metrics["ece"]


# ---------------------------------------------------------------------------
# Learning-rate scheduling helpers
# ---------------------------------------------------------------------------

def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig, n_epochs: int):
    """Build a learning-rate scheduler from config."""
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=cfg.min_lr
        )
    elif cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.step_gamma
        )
    return None


# ---------------------------------------------------------------------------
# High-level training driver
# ---------------------------------------------------------------------------

def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    tag: str = "train",
) -> Dict:
    """
    Full training loop with validation, LR schedule, and checkpointing.

    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        cfg: Training configuration.
        device: Device to run on.
        tag: Prefix for checkpoint filenames.

    Returns:
        Dictionary with training history and best metrics.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    lr = cfg.lr if tag == "train" else cfg.finetune_lr
    n_epochs = cfg.epochs if tag == "train" else cfg.finetune_epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg, n_epochs)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_ece": []}
    best_val_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_ece = evaluate_model(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{tag}] Epoch {epoch:3d}/{n_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  val_ece={val_ece:.4f}  "
            f"lr={current_lr:.2e}  ({elapsed:.1f}s)"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_ece"].append(val_ece)

        if cfg.save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"{tag}_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved best checkpoint ({val_acc:.4f}) -> {ckpt_path}")

    return history
