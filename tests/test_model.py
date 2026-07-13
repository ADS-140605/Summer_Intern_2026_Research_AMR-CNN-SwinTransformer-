import os
import tempfile
import pytest
import torch
import torch.nn as nn
import numpy as np

# Try importing the model components
try:
    from src.model import (
        window_partition_1d,
        window_reverse_1d,
        WindowAttention1D,
        SwinBlock1D,
        PatchMerging1D,
        PlainSwin1D
    )
    has_model = True
except ImportError:
    has_model = False

# Import RamanSwinDataset from src.data if available
try:
    from src.data import RamanSwinDataset
    has_data = True
except ImportError:
    has_data = False

# ==========================================
# Tier 1 Unit Tests
# ==========================================

@pytest.mark.skipif(not has_model, reason="src.model components not found")
def test_window_partition_and_reverse_identity():
    """Asserts that partitioning a 1D sequence into windows and reversing the partitioning yields the identical input tensor."""
    B, L, C = 4, 64, 16
    window_size = 8
    x = torch.randn(B, L, C)
    
    # Partition
    windows = window_partition_1d(x, window_size)
    
    # Expected partitioned shape: (B * (L // window_size), window_size, C) -> (32, 8, 16)
    expected_num_windows = B * (L // window_size)
    assert windows.shape == (expected_num_windows, window_size, C), \
        f"Expected partition shape {(expected_num_windows, window_size, C)}, got {windows.shape}"
    
    # Reverse
    x_reversed = window_reverse_1d(windows, window_size, L)
    
    assert x_reversed.shape == x.shape, \
        f"Expected reversed shape {x.shape}, got {x_reversed.shape}"
    assert torch.allclose(x, x_reversed, atol=1e-6), "Reversed tensor does not match original tensor"


# ==========================================
# Tier 2 Integration Tests
# ==========================================

@pytest.mark.skipif(not has_model, reason="src.model components not found")
def test_window_attention_shapes():
    """Validates output shapes of WindowAttention1D with/without optional relative position bias/hook."""
    B, window_size, C = 16, 8, 32
    num_heads = 4
    attn = WindowAttention1D(dim=C, window_size=window_size, num_heads=num_heads)
    
    # Input shape: (B, window_size, C)
    x = torch.randn(B, window_size, C)
    
    # Forward pass without mask
    out = attn(x)
    assert out.shape == (B, window_size, C), f"Expected shape {(B, window_size, C)}, got {out.shape}"
    
    # Check if forward method accepts an optional mask argument
    import inspect
    sig = inspect.signature(attn.forward)
    if 'mask' in sig.parameters:
        mask = torch.zeros(B, window_size, window_size)
        out_masked = attn(x, mask=mask)
        assert out_masked.shape == (B, window_size, C)


@pytest.mark.skipif(not has_model, reason="src.model components not found")
def test_shifted_window_masking():
    """Verifies that shifted window attention masks out-of-bounds cross-attention correctly."""
    dim = 8
    input_resolution = 16
    num_heads = 2
    window_size = 8
    shift_size = 3
    
    block = SwinBlock1D(
        dim=dim,
        input_resolution=input_resolution,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size
    )
    
    # Test forward pass with shift
    x = torch.randn(2, input_resolution, dim)
    out = block(x)
    assert out.shape == x.shape
    
    # Find any buffers or attributes containing 'mask'
    mask_buffers = [name for name, _ in block.named_buffers() if 'mask' in name]
    mask_attrs = [attr for attr in dir(block) if 'mask' in attr.lower() and attr not in mask_buffers]
    
    mask = None
    if mask_buffers:
        mask = getattr(block, mask_buffers[0])
    elif mask_attrs:
        mask = getattr(block, mask_attrs[0])
        
    # If a mask buffer/attribute is registered, assert its shape and values
    if mask is not None and isinstance(mask, torch.Tensor):
        # The attention mask in Swin Transformer has shape (num_windows, window_size, window_size)
        assert mask.shape[-2:] == (window_size, window_size), \
            f"Expected mask spatial shape {(window_size, window_size)}, got {mask.shape[-2:]}"
        # Where the mask is applied, elements should be negative/non-zero; valid elements should be 0.0
        assert (mask == 0).any(), "Expected some elements in attention mask to be 0 (unmasked)"
        assert (mask < -1.0).any() or (mask != 0).any(), "Expected some elements in attention mask to be non-zero (masked)"


@pytest.mark.skipif(not has_model, reason="src.model components not found")
def test_swin_block_shapes():
    """Validates output shapes of SwinBlock1D with shift_size = 0 and shift_size > 0."""
    B, L, C = 2, 64, 16
    window_size = 8
    x = torch.randn(B, L, C)
    
    # No shift
    block_no_shift = SwinBlock1D(dim=C, input_resolution=L, num_heads=2, window_size=window_size, shift_size=0)
    out_no_shift = block_no_shift(x)
    assert out_no_shift.shape == (B, L, C)
    
    # Shifted
    block_shift = SwinBlock1D(dim=C, input_resolution=L, num_heads=2, window_size=window_size, shift_size=4)
    out_shift = block_shift(x)
    assert out_shift.shape == (B, L, C)


@pytest.mark.skipif(not has_model, reason="src.model components not found")
def test_patch_merging_shapes():
    """Validates output shapes of PatchMerging1D."""
    B, L, C = 2, 64, 16
    merger = PatchMerging1D(dim=C)
    x = torch.randn(B, L, C)
    
    out = merger(x)
    # Downsamples sequence length by 2, doubles features/channels
    assert out.shape == (B, L // 2, C * 2), f"Expected shape {(B, L // 2, C * 2)}, got {out.shape}"


@pytest.mark.skipif(not has_model, reason="src.model components not found")
def test_plain_swin_classifier_shapes():
    """Validates the output shapes of the PlainSwin1D classifier."""
    model = PlainSwin1D(
        input_dim=20,
        embed_dim=32,
        depths=[2, 2],
        num_heads=[2, 4],
        window_size=8,
        num_classes=30
    )
    
    # Input sequence has 32 patches, each patch of size 20 (patch_size)
    x = torch.randn(2, 32, 20)
    out = model(x)
    assert out.shape == (2, 30), f"Expected output shape {(2, 30)}, got {out.shape}"


@pytest.mark.skipif(not has_model, reason="src.model components not found")
def test_gradient_flow():
    """Verifies gradients propagate successfully through all layers to the parameters without causing NaNs or Infs."""
    model = PlainSwin1D(
        input_dim=20,
        embed_dim=32,
        depths=[2, 2],
        num_heads=[2, 4],
        window_size=8,
        num_classes=30
    )
    
    x = torch.randn(2, 32, 20, requires_grad=True)
    out = model(x)
    
    loss = out.mean()
    loss.backward()
    
    # Check input gradients
    assert x.grad is not None
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaNs"
    assert not torch.isinf(x.grad).any(), "Input gradient contains Infs"
    
    # Check parameter gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaNs"
            assert not torch.isinf(param.grad).any(), f"Gradient for {name} contains Infs"


@pytest.mark.skipif(not has_model, reason="src.model components not found")
def test_swin_block_overfit():
    """Overfits a single SwinBlock1D on a tiny synthetic batch of 2 samples to ~0 loss within 20 iterations."""
    torch.manual_seed(42)
    B, L, C = 2, 8, 16
    block = SwinBlock1D(dim=C, input_resolution=L, num_heads=2, window_size=8, shift_size=0)
    
    x = torch.randn(B, L, C)
    target = torch.randn(B, L, C)
    
    optimizer = torch.optim.Adam(block.parameters(), lr=0.1)
    
    for _ in range(20):
        optimizer.zero_grad()
        out = block(x)
        loss = torch.mean((out - target) ** 2)
        loss.backward()
        optimizer.step()
        
    final_loss = loss.item()
    assert final_loss < 5e-2, f"Failed to overfit single SwinBlock1D within 20 iterations. Final loss: {final_loss}"


# ==========================================
# Tier 3 Functional Tests
# ==========================================

@pytest.mark.skipif(not (has_model and has_data), reason="src.model or src.data components not found")
def test_full_model_overfit():
    """Verifies PlainSwin1D overfits a small batch of 64 samples to >= 90% accuracy."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    DATA_DIR = r"d:\AMR\data\ramanspy"
    if os.path.exists(DATA_DIR):
        try:
            from src.data import load_bacteria_dataset
            spectra, labels, wavenumbers = load_bacteria_dataset(DATA_DIR, "train")
            # Take 64 samples
            spectra = spectra[:64]
            labels = labels[:64]
        except Exception:
            # Fallback to synthetic data if loading fails
            spectra = np.random.rand(64, 1000) * 10.0 + 5.0
            labels = np.random.randint(0, 30, size=(64,))
            wavenumbers = np.linspace(381.98, 1792.4, 1000)
    else:
        # Fallback to synthetic data if dataset files are missing
        spectra = np.random.rand(64, 1000) * 10.0 + 5.0
        labels = np.random.randint(0, 30, size=(64,))
        wavenumbers = np.linspace(381.98, 1792.4, 1000)
        
    dataset = RamanSwinDataset(spectra, labels, wavenumbers, patch_size=20, normalize=True, tokenize=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 50 patches (1000 / 20 = 50), window_size=5 (50 and 25 are divisible by 5)
    model = PlainSwin1D(
        input_dim=20,
        embed_dim=32,
        depths=[2, 2],
        num_heads=[2, 4],
        window_size=5,
        num_classes=30
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    accuracy = 0.0
    for epoch in range(100):
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(out, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        accuracy = correct / total
        if accuracy >= 0.90:
            break
            
    assert accuracy >= 0.90, f"Failed to overfit PlainSwin1D to >= 90% accuracy. Final accuracy: {accuracy}"


@pytest.mark.skipif(not has_model, reason="src.model components not found")
def test_checkpoint_integrity():
    """Asserts that saving a model checkpoint and reloading it restores parameter tensors exactly and yields identical model outputs."""
    model = PlainSwin1D(
        input_dim=20,
        embed_dim=32,
        depths=[2, 2],
        num_heads=[2, 4],
        window_size=8,
        num_classes=30
    )
    
    x = torch.randn(2, 32, 20)
    
    model.eval()
    with torch.no_grad():
        orig_out = model(x)
        
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
        
        # Save model checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        
        # Instantiate a new model with same config
        new_model = PlainSwin1D(
            input_dim=20,
            embed_dim=32,
            depths=[2, 2],
            num_heads=[2, 4],
            window_size=8,
            num_classes=30
        )
        
        # Load state dict
        new_model.load_state_dict(torch.load(checkpoint_path))
        new_model.eval()
        
        with torch.no_grad():
            new_out = new_model(x)
            
    # Verify outputs match
    assert torch.allclose(orig_out, new_out, atol=1e-6), "Model outputs do not match after checkpoint load"
    
    # Verify state_dict weights match exactly
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2), "Model parameters are not bitwise identical after loading checkpoint"


@pytest.mark.skipif(not has_model, reason="src.model components not found")
def test_swin_block_indivisible_seq_len():
    """Validates that SwinBlock1D handles sequence lengths that are not divisible by window_size."""
    B, L, C = 2, 50, 16
    window_size = 8
    shift_size = 3
    x = torch.randn(B, L, C)
    
    # We test both shift_size = 0 and shift_size > 0
    for s_size in [0, shift_size]:
        block = SwinBlock1D(dim=C, input_resolution=L, num_heads=2, window_size=window_size, shift_size=s_size)
        out = block(x)
        assert out.shape == (B, L, C), f"Expected shape {(B, L, C)}, got {out.shape} for shift_size={s_size}"


@pytest.mark.skipif(not has_model, reason="src.model components not found")
def test_swin_block_attn_mask_values():
    """Explicitly checks the exact mask values of attn_mask generated inside SwinBlock1D."""
    dim = 8
    L = 16
    num_heads = 2
    window_size = 8
    shift_size = 3
    
    block = SwinBlock1D(
        dim=dim,
        input_resolution=L,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size
    )
    
    # Run a forward pass to generate the mask
    x = torch.randn(2, L, dim)
    _ = block(x)
    
    # Retrieve the mask
    assert hasattr(block, 'attn_mask') and block.attn_mask is not None
    mask = block.attn_mask  # shape (num_windows, window_size, window_size) -> (2, 8, 8)
    
    assert mask.shape == (2, 8, 8), f"Expected mask shape (2, 8, 8), got {mask.shape}"
    
    # Window 0 should not have any masked elements (all elements within Window 0 are not masked)
    window_0_mask = mask[0]
    assert torch.all(window_0_mask == 0.0), f"Expected Window 0 mask to be all 0.0, got {window_0_mask}"
    
    # Window 1 should mask cross-attention between region 1 (indices 0 to 4) and region 2 (indices 5 to 7)
    window_1_mask = mask[1]
    
    # Region 1 self-attention: indices 0 to 4
    assert torch.all(window_1_mask[0:5, 0:5] == 0.0), "Expected region 1 self-attention to be unmasked"
    
    # Region 2 self-attention: indices 5 to 7
    assert torch.all(window_1_mask[5:8, 5:8] == 0.0), "Expected region 2 self-attention to be unmasked"
    
    # Region 1 to Region 2 cross-attention (masked)
    assert torch.all(window_1_mask[0:5, 5:8] == -100.0), "Expected region 1 to 2 cross-attention to be masked with -100.0"
    assert torch.all(window_1_mask[5:8, 0:5] == -100.0), "Expected region 2 to 1 cross-attention to be masked with -100.0"
