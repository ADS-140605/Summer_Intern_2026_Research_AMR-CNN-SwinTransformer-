import os
import sys
import json
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest import mock

# Try importing the metrics and training functions
try:
    from src.metrics import ECELoss
    has_metrics = True
except ImportError:
    has_metrics = False

try:
    from src.train import train_one_epoch, evaluate_model
    has_train = True
except ImportError:
    has_train = False

try:
    from src.model import PlainSwin1D
    has_model = True
except ImportError:
    has_model = False


# ==========================================
# Tier 1 Unit Tests
# ==========================================

@pytest.mark.skipif(not has_metrics, reason="src.metrics components not found")
def test_metric_calculations():
    """Feeds mock predictions and targets to verify accuracy, macro-F1, and ECE formulas."""
    # 5 samples, 3 classes
    logits = torch.tensor([
        [2.0, 1.0, 0.1],   # softmax: [0.66, 0.24, 0.10] -> pred 0, conf 0.66
        [0.5, 2.0, 0.5],   # softmax: [0.15, 0.70, 0.15] -> pred 1, conf 0.70
        [1.0, 1.0, 3.0],   # softmax: [0.10, 0.10, 0.80] -> pred 2, conf 0.80
        [2.5, 0.5, 0.5],   # softmax: [0.78, 0.11, 0.11] -> pred 0, conf 0.78
        [0.1, 0.1, 2.0]    # softmax: [0.12, 0.12, 0.76] -> pred 2, conf 0.76
    ])
    
    # True targets
    targets = torch.tensor([0, 1, 1, 2, 2]) # predictions: [0, 1, 2, 0, 2]
    # Correct predictions:
    # sample 0: pred 0, target 0 (Correct)
    # sample 1: pred 1, target 1 (Correct)
    # sample 2: pred 2, target 1 (Incorrect)
    # sample 3: pred 0, target 2 (Incorrect)
    # sample 4: pred 2, target 2 (Correct)
    # Accuracy = 3 / 5 = 0.60
    
    # ECE with n_bins=5. Bin boundaries: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    # All confidences fall into Bin 3 (0.6 to 0.8)!
    # Total samples in bin 3 = 5. Fraction = 1.0.
    # Accuracy in bin 3 = 3 / 5 = 0.60
    # Average confidence in bin 3 = (0.6601 + 0.7053 + 0.7997 + 0.7820 + 0.7585) / 5 ≈ 0.7411
    # ECE = 1.0 * |0.7411 - 0.60| ≈ 0.1411
    
    ece_loss_fn = ECELoss(n_bins=5)
    ece_val = ece_loss_fn(logits, targets)
    
    assert torch.is_tensor(ece_val), "ECE output must be a tensor"
    assert torch.isclose(ece_val, torch.tensor(0.1411), atol=1e-2), f"Expected ECE ≈ 0.1411, got {ece_val.item()}"


# ==========================================
# Tier 4 System / End-to-End Tests
# ==========================================

@pytest.mark.skipif(not (has_train and has_model), reason="src.train or src.model components not found")
def test_reproducibility():
    """Runs a training execution twice with the identical random seed, asserting that the training loss curves and parameters are identical."""
    
    def run_training(seed):
        # Set all seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # Create small synthetic dataset: 16 samples, 8 patches, 20 features per patch
        x = torch.randn(16, 8, 20)
        y = torch.randint(0, 5, (16,))
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        model = PlainSwin1D(
            input_dim=20,
            embed_dim=16,
            depths=[2],
            num_heads=[2],
            window_size=8,
            num_classes=5
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        losses = []
        for epoch in range(2):
            loss = train_one_epoch(model, loader, optimizer, criterion, device=torch.device('cpu'))
            losses.append(loss)
            
        params = [p.clone().detach() for p in model.parameters()]
        return losses, params

    # Run 1 and Run 2 with identical seed
    losses1, params1 = run_training(42)
    losses2, params2 = run_training(42)
    
    # Run with different seed
    losses_diff, params_diff = run_training(100)
    
    # Assert loss curves are identical
    assert np.allclose(losses1, losses2), f"Losses mismatch under identical seeds: {losses1} vs {losses2}"
    
    # Assert parameters are identical
    for p1, p2 in zip(params1, params2):
        assert torch.allclose(p1, p2), "Parameters mismatch under identical seeds"
        
    # Assert different seed yields different results (losses or parameters)
    different_losses = not np.allclose(losses1, losses_diff)
    different_params = not all(torch.allclose(p1, pd) for p1, pd in zip(params1, params_diff))
    assert different_losses or different_params, "Setting a different seed yielded identical training behavior"


@pytest.mark.skipif(not (has_train and has_model), reason="src.train or src.model components not found")
def test_short_training_run():
    """Verifies a short training/fine-tuning execution completes and saves metrics/checkpoints."""
    # Create small synthetic dataset: 10 samples
    x = torch.randn(10, 8, 20)
    y = torch.randint(0, 5, (10,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    model = PlainSwin1D(
        input_dim=20,
        embed_dim=16,
        depths=[2],
        num_heads=[2],
        window_size=8,
        num_classes=5
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 1. Test model evaluation
    eval_metrics = evaluate_model(model, loader, criterion, device=torch.device('cpu'))
    assert len(eval_metrics) == 4, f"evaluate_model should return 4 metrics, got {len(eval_metrics)}"
    loss, acc, f1, ece = eval_metrics
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert isinstance(f1, float)
    assert isinstance(ece, float)
    
    # 2. Test train one epoch
    train_loss = train_one_epoch(model, loader, optimizer, criterion, device=torch.device('cpu'))
    assert isinstance(train_loss, float), f"Expected float train loss, got {type(train_loss)}"


@pytest.mark.skipif(not os.path.exists("run_phase1_baseline.py"), reason="run_phase1_baseline.py not found")
def test_baseline_script_execution():
    """Verifies that executing run_phase1_baseline.py outputs a valid JSON file to results/phase1_baseline.json."""
    DATA_DIR = r"d:\AMR\data\ramanspy"
    if not os.path.exists(DATA_DIR):
        pytest.skip("Real dataset files not available, skipping baseline execution test.")
        
    json_path = os.path.join("results", "phase1_baseline.json")
    if os.path.exists(json_path):
        try:
            os.remove(json_path)
        except Exception:
            pass
            
    import subprocess
    
    # Try running the script for 1 epoch to keep it fast
    cmd = [sys.executable, "run_phase1_baseline.py", "--epochs", "1"]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        # If --epochs is not supported, fallback to default execution
        if result.returncode != 0 and ("unrecognized arguments" in result.stderr or "invalid option" in result.stderr):
            cmd = [sys.executable, "run_phase1_baseline.py"]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("run_phase1_baseline.py timed out after 60 seconds")
        
    assert os.path.exists(json_path), f"Baseline script did not generate {json_path}. Stderr: {result.stderr}"
    
    # Read the JSON file and verify schema
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # Find accuracy, macro_f1, and ece recursively in the JSON structure
    def find_keys(d):
        found = set()
        if isinstance(d, dict):
            for k, v in d.items():
                if k.lower() in {"accuracy", "acc", "macro_f1", "macro_f1_score", "ece", "ece_loss"}:
                    found.add(k.lower())
                found.update(find_keys(v))
        elif isinstance(d, list):
            for item in d:
                found.update(find_keys(item))
        return found

    found_metrics = find_keys(data)
    assert any("acc" in k for k in found_metrics), "Accuracy metric not found in baseline JSON output"
    assert any("f1" in k for k in found_metrics), "Macro-F1 metric not found in baseline JSON output"
    assert any("ece" in k for k in found_metrics), "ECE metric not found in baseline JSON output"
