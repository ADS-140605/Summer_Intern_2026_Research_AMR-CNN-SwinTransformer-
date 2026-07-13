#!/usr/bin/env python
"""
Phase 1 baseline: train PlainSwin1D on Bacteria-ID reference splits,
fine-tune on the fine-tune split, evaluate on test, and save metrics.

Usage:
    python run_phase1_baseline.py
    python run_phase1_baseline.py --epochs 10 --finetune_epochs 5
"""

import argparse
import json
import os
import sys
import time

import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import get_dataloaders
from src.model import PlainSwin1D
from src.train import TrainConfig, set_seed, run_training, evaluate_model
from src.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 baseline training")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    p.add_argument("--finetune_epochs", type=int, default=None, help="Override fine-tune epochs")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    p.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Config ──────────────────────────────────────────────────────────
    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.finetune_epochs is not None:
        cfg.finetune_epochs = args.finetune_epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.seed is not None:
        cfg.seed = args.seed

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: epochs={cfg.epochs}, finetune_epochs={cfg.finetune_epochs}, "
          f"lr={cfg.lr}, batch_size={cfg.batch_size}, seed={cfg.seed}")

    # ── Data ────────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    ref_train_loader, ref_val_loader, ft_val_loader, test_loader, spectral_axis = (
        get_dataloaders(cfg.data_dir, cfg.patch_size, cfg.batch_size, cfg.seed)
    )

    # Infer input_dim from first batch
    sample_tokens, _ = next(iter(ref_train_loader))
    input_dim = sample_tokens.shape[-1]  # patch_size
    num_patches = sample_tokens.shape[1]
    print(f"  Tokens per spectrum: {num_patches},  Patch dim: {input_dim}")
    print(f"  Reference train batches: {len(ref_train_loader)}")
    print(f"  Reference val batches:   {len(ref_val_loader)}")
    print(f"  Fine-tune batches:       {len(ft_val_loader)}")
    print(f"  Test batches:            {len(test_loader)}")

    # ── Model ───────────────────────────────────────────────────────────
    print("\n=== Building model ===")
    model = PlainSwin1D(
        input_dim=input_dim,
        embed_dim=cfg.embed_dim,
        depths=cfg.depths,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
        mlp_ratio=cfg.mlp_ratio,
        drop_rate=cfg.drop_rate,
        attn_drop_rate=cfg.attn_drop_rate,
        drop_path_rate=cfg.drop_path_rate,
        num_classes=cfg.num_classes,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Stage 1: Train on reference splits ──────────────────────────────
    print("\n=== Stage 1: Training on reference train/val ===")
    t0 = time.time()
    train_history = run_training(
        model, ref_train_loader, ref_val_loader, cfg, device, tag="train"
    )
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")

    # Load best checkpoint for fine-tuning
    best_ckpt = os.path.join(cfg.checkpoint_dir, "train_best.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        print(f"  Loaded best training checkpoint from {best_ckpt}")

    # ── Stage 2: Fine-tune on fine-tune split ───────────────────────────
    print("\n=== Stage 2: Fine-tuning on fine-tune split ===")
    t0 = time.time()
    ft_history = run_training(
        model, ft_val_loader, ref_val_loader, cfg, device, tag="finetune"
    )
    ft_time = time.time() - t0
    print(f"  Fine-tuning completed in {ft_time:.1f}s")

    # Load best fine-tune checkpoint for evaluation
    best_ft_ckpt = os.path.join(cfg.checkpoint_dir, "finetune_best.pt")
    if os.path.exists(best_ft_ckpt):
        model.load_state_dict(torch.load(best_ft_ckpt, map_location=device, weights_only=True))
        print(f"  Loaded best fine-tune checkpoint from {best_ft_ckpt}")

    # ── Stage 3: Evaluate on held-out test set ──────────────────────────
    print("\n=== Stage 3: Evaluating on test set ===")
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    test_loss, test_acc, test_f1, test_ece = evaluate_model(
        model, test_loader, criterion, device
    )
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Macro-F1: {test_f1:.4f}")
    print(f"  Test ECE:      {test_ece:.4f}")

    # ── Save results ────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    results = {
        "model": "PlainSwin1D",
        "config": {
            "embed_dim": cfg.embed_dim,
            "depths": cfg.depths,
            "num_heads": cfg.num_heads,
            "window_size": cfg.window_size,
            "patch_size": cfg.patch_size,
            "mlp_ratio": cfg.mlp_ratio,
            "drop_path_rate": cfg.drop_path_rate,
            "lr": cfg.lr,
            "finetune_lr": cfg.finetune_lr,
            "epochs": cfg.epochs,
            "finetune_epochs": cfg.finetune_epochs,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
        },
        "parameters": n_params,
        "test_metrics": {
            "accuracy": test_acc,
            "macro_f1": test_f1,
            "ece": test_ece,
            "loss": test_loss,
        },
        "train_history": train_history,
        "finetune_history": ft_history,
        "training_time_s": train_time,
        "finetuning_time_s": ft_time,
    }

    out_path = os.path.join("results", "phase1_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {out_path}")


if __name__ == "__main__":
    main()
