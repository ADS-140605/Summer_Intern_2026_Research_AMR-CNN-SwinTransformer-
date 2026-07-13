# Test Infrastructure Specification

This document details the test runner configuration, dependency requirements, and design specifications for the 4-Tier test suite of the Swin-1D classifier.

## 1. Test Environment & Setup

### Requirements
The test suite runs under the project virtual environment. The following dependencies are required:
- **Python**: `3.11.0` (located at `venv/Scripts/python.exe`)
- **PyTorch**: `2.12.0`
- **RamanSPy**: `0.2.10`
- **scikit-learn**: `1.8.0`
- **pytest**: Required for test execution (must be installed via `pip install pytest`).

### Configuration
A `pytest.ini` or configuration section should configure `pytest` to:
- Trace test files in the `tests/` directory.
- Allow stdout capturing for debugging (`-s` flag).
- Enable verbose output (`-v` flag).

---

## 2. The 4-Tier Test Suite Design

The tests are organized into four sequential tiers to isolate logical errors quickly.

```
+-------------------------------------------------------------+
|                     Tier 1: Unit Tests                      |
|  - Math roundtrips (partition/reverse)                      |
|  - Data Min-Max normalizer correctness                      |
|  - Tokenization coordinate bookkeeping                      |
|  - Metrics formulas (ECE, Macro-F1, Accuracy)               |
+------------------------------+------------------------------+
                               |
                               v
+------------------------------+------------------------------+
|             Tier 2: Integration & Model Sanity              |
|  - Layer/Block forward pass shapes                          |
|  - WindowAttention1D masking and relative position hook     |
|  - Gradient backpropagation (no NaNs check)                 |
|  - SwinBlock1D overfitting on a synthetic batch             |
+------------------------------+------------------------------+
                               |
                               v
+------------------------------+------------------------------+
|            Tier 3: Functional / Overfitting                 |
|  - Data loading and class distribution validation           |
|  - Stratified 80/20 train/val split check                   |
|  - PlainSwin1D overfitting (90%+ accuracy on 64 samples)    |
|  - Checkpoint save/load integrity                           |
+------------------------------+------------------------------+
                               |
                               v
+------------------------------+------------------------------+
|                Tier 4: System / End-to-End                  |
|  - Reproducible short training and fine-tuning runs         |
|  - Seed verification across run instances                   |
|  - run_phase1_baseline.py output validation                 |
+-------------------------------------------------------------+
```

### Tier 1: Unit Tests
Focuses on pure mathematical functions, data normalization, tokenization mechanics, and metric calculations.
*   **Test Cases (`tests/test_data.py`)**:
    *   `test_normalization_min_max`: Asserts that spectra are scaled to `[0, 1]` and checks division-by-zero safety on flat lines.
    *   `test_tokenization_shapes`: Verifies that a spectrum of length $L$ is tokenized into $\lfloor L / \text{patch\_size} \rfloor$ tokens of size `patch_size`.
    *   `test_wavenumber_bookkeeping`: Verifies that representative wavenumbers are the mean of wavenumber coordinates within each patch and are monotonically increasing.
*   **Test Cases (`tests/test_model.py`)**:
    *   `test_window_partition_and_reverse_identity`: Asserts that partitioning a 1D sequence into windows and reversing the partitioning yields the identical input tensor.
*   **Test Cases (`tests/test_train.py` / `tests/test_metrics.py`)**:
    *   `test_metric_calculations`: Feeds mock predictions and targets to verify accuracy, macro-F1, and Expected Calibration Error (ECE) formulas match exact pre-calculated values.

### Tier 2: Integration & Model Sanity Tests
Validates block interactions, gradient flows, and shape matching across layers.
*   **Test Cases (`tests/test_model.py`)**:
    *   `test_attention_masking`: Verifies that shifted window attention masks out-of-bounds cross-attention correctly.
    *   `test_layer_shapes`: Validates output shapes of `WindowAttention1D`, `SwinBlock1D`, and `PatchMerging1D`.
    *   `test_gradient_flow`: Verifies gradients propagate successfully through all layers to the inputs and parameters without causing NaNs or Infs.
    *   `test_swin_block_overfit`: Overfits a single `SwinBlock1D` on a tiny synthetic batch of 2 samples to ~0 loss within 20 iterations.

### Tier 3: Functional / Statistical Tests
Validates loading operations from files, data distribution properties, and full-model capabilities.
*   **Test Cases (`tests/test_data.py`)**:
    *   `test_dataset_shapes_and_classes`: Verifies that the Bacteria-ID dataset splits loaded from `d:\AMR\data\ramanspy` contain the correct wavenumber axis (`381.98–1792.4 cm⁻¹` with 1000 points) and target label counts (2000 per isolate for reference; 100 for fine-tune/test).
    *   `test_stratified_split`: Verifies `get_dataloaders` performs an 80/20 train/validation split that preserves class proportions under a fixed random seed.
*   **Test Cases (`tests/test_model.py`)**:
    *   `test_full_model_overfit`: Verifies that `PlainSwin1D` classifier successfully overfits a small subset of 64 real reference samples to $\ge 90\%$ training accuracy under a reasonable epoch limit.
    *   `test_checkpoint_integrity`: Asserts that saving a model checkpoint and reloading it restores parameter tensors exactly and yields identical model outputs.

### Tier 4: System / End-to-End Tests
Validates the entire pipeline, logging outputs, and reproducibility across script boundaries.
*   **Test Cases (`tests/test_train.py`)**:
    *   `test_reproducibility`: Runs a training execution twice with the identical random seed, asserting that the training loss curves and parameters are bitwise identical.
    *   `test_short_training_run`: Verifies a short training/fine-tuning execution completes and saves metrics/checkpoints.
    *   `test_baseline_script_execution`: Verifies that executing `run_phase1_baseline.py` outputs a valid JSON file to `results/phase1_baseline.json` matching the schema containing accuracy, macro-F1, and ECE.

---

## 3. How to Run the Tests

To run the entire suite, execute:
```bash
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Run all tests
pytest -v -s
```

To run a specific tier or module (e.g., data loading unit tests):
```bash
pytest tests/test_data.py -k "normalization"
```
