# Raman AMR Models

PyTorch implementations of the multimodal CNN described in the paper and the comparison models it evaluates

## Included models

- `MultimodalRamanCNN`: raw spectrum branch + CWT branch with late concatenation fusion.
- `RamanOnlyCNN`: 1D baseline that uses only raw spectra.
- `WaveletOnlyCNN`: 2D baseline that uses only the wavelet image.
- `RamanSVMClassifier`: scikit-learn RBF SVM baseline using concatenated spectral and wavelet features.

## Usage

```bash
pip install -r requirements.txt
python scripts/inspect_models.py
```

## Input shapes

- Spectral input: `(batch, 1000)` or `(batch, 1, 1000)`
- Wavelet input: `(batch, 3, 224, 224)`

The code focuses on the model definitions. You can plug these into your own training loop or dataset loader.

## Training and evaluation

The repository now includes simple CLI scripts for `.npz` datasets with `spectral`, `wavelet`, and `labels` arrays.

Train a CNN:

```bash
python scripts/train.py --data path/to/dataset.npz --model multimodal --epochs 40
```

Evaluate a trained CNN:

```bash
python scripts/evaluate.py --data path/to/dataset.npz --model multimodal --checkpoint outputs/multimodal_best.pt
```

Evaluate the SVM baseline:

```bash
python scripts/evaluate.py --data path/to/dataset.npz --model svm
```

## Novel Research Scaffold

A small scaffold for exploring the proposed neuro-symbolic few-shot idea is included in the `novel research` folder.

- **Files:** `novel research/main.py` (small CLI demo) and `research_statement.txt` (plain-language research statement).
- **Purpose:** quick smoke tests on the existing `data/ramanspy` `.npy` arrays and a place to add prototypes for the idea.

Usage examples:

```bash
# list datasets found under data/ramanspy
python "novel research/main.py" --list

# run the smoke nearest-centroid demo (quotes needed because folder name contains a space)
python "novel research/main.py" --demo
```

If you'd like, I can expand the scaffold with training/evaluation scripts, a README inside `novel research`, or a one-page summary.

**How to run the novel research demo**

- **List datasets:**

```bash
python "novel research/main.py" --list
```

- **Run the smoke demo:**

```bash
python "novel research/main.py" --demo
```

- **Notes:**
	- On Windows PowerShell or cmd, keep the quotes around the path because the folder name contains a space.
	- The demo uses the `.npy` files under `data/ramanspy` (e.g., `X_reference.npy`, `X_2018clinical.npy`). If labels (`y_*.npy`) are not present the demo will only show dataset shapes.
	- See the plain-language statement at `research_statement.txt` for the project summary.

Files to inspect:

- `novel research/main.py` — small CLI demo that lists datasets and runs a nearest-centroid baseline.
- `research_statement.txt` — simple research statement and next steps.

