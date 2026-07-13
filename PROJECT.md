# Project: Plain 1D Swin Transformer Classifier for Raman Spectroscopy

## Architecture
A modular architecture separating data ingestion, model blocks, training pipeline, and verification.

```
+-------------------------------------------------------------+
|                        RamanSPy API                         |
+------------------------------+------------------------------+
                               |
                               v
+------------------------------+------------------------------+
|                         data.py                             |
|  - Load Bacteria-ID (reference train, fine-tune, test)       |
|  - Preprocessing: Per-spectrum Min-Max normalizer           |
|  - Train/Val Split (80/20, fixed seed)                      |
|  - Tokenization: Spectrum to 1D patches + wavenumber book   |
|  - PyTorch Dataset & DataLoader                             |
+------------------------------+------------------------------+
                               |
                               v
+------------------------------+------------------------------+
|                         model.py                            |
|  - 1D Window Partitioning & Reversing                       |
|  - 1D Window Attention (with optional position bias hook)   |
|  - 1D Shifted Window cyclic partitioning & masking          |
|  - 1D Patch Merging (downsamples by 2, doubles channels)    |
|  - SwinBlock1D                                              |
|  - PlainSwin1D Classifier                                   |
+------------------------------+------------------------------+
                               |
                               v
+------------------------------+------------------------------+
|                train.py & metrics.py                        |
|  - training & validation loops                              |
|  - metrics: accuracy, macro-F1, ECE                         |
|  - run_phase1_baseline.py entrypoint                        |
+-------------------------------------------------------------+
```

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 0 | Exploration & Setup | Environment setup, API verification | none | DONE (Verified RamanSPy, Python 3.11.0, PyTorch 2.12.0) |
| 1 | Data Preprocessing | Normalization, tokenization, dataset, loader | M0 | PLANNED |
| 2 | Swin Blocks | Partitioning, attention, merging, SwinBlock1D | M1 | PLANNED |
| 3 | PlainSwin1D Model | PlainSwin1D implementation, parameter count check | M2 | PLANNED |
| 4 | Training & Baseline | Train/val loops, metrics, run_phase1_baseline.py | M1, M3 | PLANNED |
| 5 | E2E Testing Track | Independent opaque-box test suite | none | PLANNED |

## Interface Contracts

### Dataset Locations
- Primary Data Directory: `d:\AMR\data\ramanspy` (contains `X_reference.npy`, `y_reference.npy`, `X_finetune.npy`, `y_finetune.npy`, `X_test.npy`, `y_test.npy`, `wavenumbers.npy`).
- Alternative Data Directory: `d:\amr-data-analysis\bacteria_data`

### Data Loader Interface (`src/data.py`)
- `load_bacteria_dataset(data_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`:
  Loads the spectra (`X`), labels (`y`), and wavenumber axis. `split` is one of `'train'`, `'val'`, or `'test'` (loaded via `ramanspy.datasets.bacteria(split, folder=data_dir)`).
- `class RamanSwinDataset(torch.utils.data.Dataset)`:
  - Custom dataset accepting spectrum arrays, labels, wavenumbers, patch_size, and boolean for normalization/tokenization.
  - Normalizes spectra per-spectrum to range `[0, 1]`.
  - Tokenizes 1D spectra into non-overlapping patches.
  - Computes and stores the representative wavenumber coordinate for each patch.
- `get_dataloaders(data_dir: str, patch_size: int, batch_size: int, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, np.ndarray]`:
  - Splits the loaded reference `'train'` dataset into an 80/20 train/validation split with a fixed random seed.
  - Creates PyTorch DataLoaders for: reference train (80%), reference validation (20%), fine-tune validation (the original `'val'` split), and test (the original `'test'` split).
  - Returns: `(ref_train_loader, ref_val_loader, ft_val_loader, test_loader, spectral_axis)`.

### Swin Transformer Interface (`src/model.py`)
- `window_partition_1d(x: Tensor, window_size: int) -> Tensor`:
  - Partition sequence `x` of shape `(B, L, C)` into windows of shape `(num_windows * B, window_size, C)`.
- `window_reverse_1d(windows: Tensor, window_size: int, seq_len: int) -> Tensor`:
  - Reconstruct original sequence of shape `(B, L, C)` from windows.
- `class WindowAttention1D(nn.Module)`:
  - Computes 1D window-based Multi-head Self-Attention (W-MSA).
  - Supports optional position bias input.
- `class SwinBlock1D(nn.Module)`:
  - Computes `LayerNorm -> Attention1D -> Residual -> LayerNorm -> MLP -> Residual`.
  - Supports shifted window partitioning (`shift_size > 0`) using 1D cyclic shift and attention masking.
- `class PatchMerging1D(nn.Module)`:
  - Downsamples 1D token sequence length by 2, doubling channels.
- `class PlainSwin1D(nn.Module)`:
  - Entire Plain Swin-1D model: patch embedding, stages of SwinBlock1D and PatchMerging1D, average pooling, and linear head. Fully configuration-driven.

## Code Layout
- `src/`
  - `src/__init__.py`
  - `src/data.py`
  - `src/model.py`
  - `src/train.py`
  - `src/metrics.py`
- `tests/`
  - `tests/test_data.py`
  - `tests/test_model.py`
  - `tests/test_train.py`
- `results/`
  - `results/phase1_baseline.json`
- `run_phase1_baseline.py`
- `requirements.txt`
