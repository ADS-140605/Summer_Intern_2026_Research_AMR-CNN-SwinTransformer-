# Original User Request

## Initial Request — 2026-07-13T12:29:26Z

Phase 1 of a research codebase implementing and evaluating a "plain" 1D Swin Transformer classifier (no relative position bias) for Raman spectroscopy. The model is trained and evaluated on the Bacteria-ID dataset loaded via RamanSPy. The codebase must be designed with modular interfaces to support future extensions for position-bias variants and drift evaluations.

Working directory: d:\AMR-Swin Transformer
Integrity mode: development

## Requirements

### R1. Data Acquisition and Loading
- Use RamanSPy to load the standard Bacteria-ID dataset splits (reference train, fine-tune val, and test).
- Implement a helper to inspect and log dataset statistics (number of spectra, wavenumber range, class distributions).

### R2. Preprocessing & Tokenization
- Implement per-spectrum min-max normalization to [0,1].
- Split the reference dataset into an 80/20 train/val split with a fixed random seed.
- Tokenize 1D spectra into non-overlapping patches (length-L spectrum to tokens of shape `(num_patches, patch_size)`).
- Implement coordinate bookkeeping to calculate representative wavenumbers for each token.
- Wrap the preprocessing and tokenization in a PyTorch Dataset and DataLoader.

### R3. Core 1D Swin Blocks
- Implement 1D window partitioning and reversing.
- Implement 1D Window Attention with an optional position bias input argument (commented hook for future Phase 2 CPB variants).
- Implement 1D shifted-window cyclic partitioning and masking at the boundary.
- Implement 1D Patch Merging (downsamples sequence length by 2, doubles channels).
- Implement the SwinBlock1D (LayerNorm -> WindowAttention1D -> Residual -> LayerNorm -> MLP -> Residual).

### R4. Full Plain Swin-1D Classifier
- Implement `PlainSwin1D` containing patch embedding, stages of SwinBlock1D and PatchMerging1D, global average pooling, and a linear head.
- Make the architecture fully configuration-driven (no hardcoded hyperparameters in the class).
- Add a sanity script to run a single forward pass and print parameter count.

### R5. Reproducible Training and Evaluation
- Implement a reproducible training loop with an Adam optimizer, configurable LR schedule, checkpointing, and metric logging.
- Implement evaluation metrics: accuracy, macro-F1, and Expected Calibration Error (ECE).
- Create a baseline script running the end-to-end pipeline: load data, train on reference splits, fine-tune on the fine-tune split, evaluate on test, and save metrics to `results/phase1_baseline.json`.

## Verification Resources and Test Suite
The implementation must include a `pytest`-based test suite verifying each phase:
- **Data Loading**: Ensure shapes are correct, wavenumber axis matches (381.98–1792.4 cm⁻¹ with 1000 points), and label counts per class match documented values (2000 per isolate for reference; 100 for fine-tune and test).
- **Preprocessing**: Normalized spectra have min 0, max 1; tokenize reconstructs the original spectrum; token wavenumbers are monotonically increasing.
- **Swin Blocks**: Window partition/reverse round-trip is identity; forward shapes are correct; gradients backpropagate without NaNs; block overfits on a small synthetic batch.
- **Full Model**: Forward pass works on a real preprocessed batch; parameter count is logged; classifier overfits to ~90%+ training accuracy on a small subset (~64 samples) of real reference data.
- **Training/Eval**: Short training run completes and saves checkpoint; evaluate returns valid metrics in range; seeds ensure reproducible runs.

## Acceptance Criteria

### Execution & Cleanliness
- [ ] No hardcoded model architecture parameters inside the class.
- [ ] All code contains docstrings and type hints.
- [ ] Requirements.txt contains correct pinned versions for libraries.

### Verification Success
- [ ] All unit and sanity tests in `tests/` pass under `pytest`.
- [ ] `run_phase1_baseline.py` runs successfully, logs training/fine-tuning history, and outputs classification metrics (accuracy, macro-F1, ECE) to `results/phase1_baseline.json`.
