import os
import pytest
import numpy as np
import torch
from unittest import mock
from src.data import load_bacteria_dataset, RamanSwinDataset, get_dataloaders

# Define the local data directory based on constraints
DATA_DIR = r"d:\AMR\data\ramanspy"

@pytest.fixture
def synthetic_data():
    """Generates synthetic spectra, labels, and wavenumbers for fast isolated testing."""
    np.random.seed(42)
    num_samples = 100
    spectrum_length = 1000
    # generate random values in range [5.0, 15.0]
    spectra = np.random.rand(num_samples, spectrum_length) * 10.0 + 5.0
    labels = np.random.randint(0, 30, size=(num_samples,))
    wavenumbers = np.linspace(381.98, 1792.4, spectrum_length)
    return spectra, labels, wavenumbers

def test_load_bacteria_dataset_invalid_split():
    """Ensures that invalid splits raise ValueError."""
    with pytest.raises(ValueError, match="split must be one of"):
        load_bacteria_dataset(DATA_DIR, "invalid_split")

@pytest.mark.skipif(not os.path.exists(DATA_DIR), reason="Real dataset files not available in test environment")
def test_load_bacteria_dataset_real():
    """Verifies shapes, wavenumber range, and class distributions on the real dataset files."""
    # 1. Test train split
    X, y, wavenumbers = load_bacteria_dataset(DATA_DIR, "train")
    assert X.shape == (60000, 1000)
    assert y.shape == (60000,)
    assert wavenumbers.shape == (1000,)
    assert np.isclose(wavenumbers[0], 381.98, atol=1e-2)
    assert np.isclose(wavenumbers[-1], 1792.4, atol=1e-2)
    
    unique_classes, counts = np.unique(y, return_counts=True)
    assert len(unique_classes) == 30
    assert np.all(counts == 2000)

    # 2. Test val split
    X_val, y_val, wavenumbers_val = load_bacteria_dataset(DATA_DIR, "val")
    assert X_val.shape == (3000, 1000)
    assert y_val.shape == (3000,)
    assert np.all(np.unique(y_val, return_counts=True)[1] == 100)

    # 3. Test test split
    X_test, y_test, wavenumbers_test = load_bacteria_dataset(DATA_DIR, "test")
    assert X_test.shape == (3000, 1000)
    assert y_test.shape == (3000,)
    assert np.all(np.unique(y_test, return_counts=True)[1] == 100)

def test_dataset_normalization(synthetic_data):
    """Verifies per-spectrum Min-Max normalization and division by zero guard."""
    spectra, labels, wavenumbers = synthetic_data
    
    # 1. Normal normalization
    dataset = RamanSwinDataset(spectra, labels, wavenumbers, patch_size=20, normalize=True, tokenize=False)
    spectra_norm = dataset.spectra.numpy()
    assert np.allclose(spectra_norm.min(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(spectra_norm.max(axis=-1), 1.0, atol=1e-6)
    
    # 2. Raw spectra (normalize=False)
    dataset_raw = RamanSwinDataset(spectra, labels, wavenumbers, patch_size=20, normalize=False, tokenize=False)
    spectra_raw = dataset_raw.spectra.numpy()
    assert not np.allclose(spectra_raw.min(axis=-1), 0.0, atol=1e-6)
    
    # 3. Flat spectrum (guard for division by zero)
    flat_spectra = np.ones((5, 1000)) * 7.5
    dataset_flat = RamanSwinDataset(flat_spectra, labels[:5], wavenumbers, patch_size=20, normalize=True, tokenize=False)
    spectra_flat = dataset_flat.spectra.numpy()
    assert not np.isnan(spectra_flat).any()
    assert np.allclose(spectra_flat, 0.0, atol=1e-6)

def test_dataset_tokenization(synthetic_data):
    """Verifies reshaping, reconstruction, and representative wavenumbers monotonicity/correctness."""
    spectra, labels, wavenumbers = synthetic_data
    patch_size = 20
    
    dataset = RamanSwinDataset(spectra, labels, wavenumbers, patch_size=patch_size, normalize=False, tokenize=True)
    
    # 1000 / 20 = 50 patches
    assert dataset.spectra.shape == (100, 50, 20)
    assert dataset.representative_wavenumbers.shape == (50,)
    
    # Reconstruction test: reshape back to (100, 1000) and compare with original
    reconstructed = dataset.spectra.numpy().reshape(100, 1000)
    assert np.allclose(reconstructed, spectra)
    
    # Check monotonicity of representative wavenumbers
    assert np.all(np.diff(dataset.representative_wavenumbers) > 0)
    
    # Check representative wavenumber value correctness (mean of patch wavenumbers)
    expected_first_rep = wavenumbers[:20].mean()
    assert np.isclose(dataset.representative_wavenumbers[0], expected_first_rep)
    
    expected_last_rep = wavenumbers[-20:].mean()
    assert np.isclose(dataset.representative_wavenumbers[-1], expected_last_rep)

def test_dataset_tokenization_truncation_warning():
    """Ensures a warning is issued and correct truncation is done for non-divisible spectrum lengths."""
    spectra = np.random.rand(5, 15)
    labels = np.array([0, 1, 2, 0, 1])
    wavenumbers = np.linspace(100, 200, 15)
    
    # 15 is not divisible by 4. Truncates to 3 patches of size 4 (length 12)
    with pytest.warns(UserWarning, match="Truncating to length 12"):
        dataset = RamanSwinDataset(spectra, labels, wavenumbers, patch_size=4, normalize=False, tokenize=True)
        
    assert dataset.spectra.shape == (5, 3, 4)
    # Check wavenumbers shape and representative values
    assert dataset.representative_wavenumbers.shape == (3,)
    expected_first_rep = wavenumbers[:4].mean()
    assert np.isclose(dataset.representative_wavenumbers[0], expected_first_rep)

@mock.patch('src.data.load_bacteria_dataset')
def test_get_dataloaders_split_and_reproducibility(mock_load):
    """Tests splitting proportions, stratification, DataLoader shapes, and split reproducibility."""
    # Setup mock to return 100 training samples (5 classes, 20 samples each)
    num_samples = 100
    spectra_train = np.random.rand(num_samples, 1000)
    labels_train = np.repeat(np.arange(5), 20)
    wavenumbers = np.linspace(381.98, 1792.4, 1000)
    
    spectra_val = np.random.rand(20, 1000)
    labels_val = np.repeat(np.arange(5), 4)
    
    spectra_test = np.random.rand(20, 1000)
    labels_test = np.repeat(np.arange(5), 4)

    def side_effect(data_dir, split):
        if split == 'train':
            return spectra_train, labels_train, wavenumbers
        elif split == 'val':
            return spectra_val, labels_val, wavenumbers
        elif split == 'test':
            return spectra_test, labels_test, wavenumbers
        raise ValueError("Invalid split")
        
    mock_load.side_effect = side_effect
    
    # 1. Load DataLoaders
    train_loader, val_loader, ft_val_loader, test_loader, spectral_axis = get_dataloaders(
        data_dir="dummy_dir", patch_size=20, batch_size=8, seed=42
    )
    
    # 80/20 train/validation split of 100 samples -> 80 train, 20 validation
    assert len(train_loader.dataset) == 80
    assert len(val_loader.dataset) == 20
    assert len(ft_val_loader.dataset) == 20
    assert len(test_loader.dataset) == 20
    assert spectral_axis.shape == (1000,)
    
    # Check loader outputs (tensor types and shapes)
    batch_spectra, batch_labels = next(iter(train_loader))
    # Shape: (batch_size, num_patches, patch_size) -> (8, 50, 20)
    assert batch_spectra.shape == (8, 50, 20)
    assert batch_labels.shape == (8,)
    assert batch_spectra.dtype == torch.float32
    assert batch_labels.dtype == torch.long
    
    # Check stratification:
    # 80 train samples split equally across 5 classes -> 16 samples per class
    train_y = train_loader.dataset.labels.numpy()
    classes, counts = np.unique(train_y, return_counts=True)
    assert len(classes) == 5
    assert np.all(counts == 16)
    
    # 20 val samples split equally across 5 classes -> 4 samples per class
    val_y = val_loader.dataset.labels.numpy()
    val_classes, val_counts = np.unique(val_y, return_counts=True)
    assert len(val_classes) == 5
    assert np.all(val_counts == 4)
    
    # Check seed reproducibility
    train_loader_2, val_loader_2, _, _, _ = get_dataloaders(
        data_dir="dummy_dir", patch_size=20, batch_size=8, seed=42
    )
    train_y_2 = train_loader_2.dataset.labels.numpy()
    val_y_2 = val_loader_2.dataset.labels.numpy()
    
    np.testing.assert_array_equal(train_y, train_y_2)
    np.testing.assert_array_equal(val_y, val_y_2)
    
    # Check that a different seed produces a different split
    train_loader_diff, val_loader_diff, _, _, _ = get_dataloaders(
        data_dir="dummy_dir", patch_size=20, batch_size=8, seed=100
    )
    train_y_diff = train_loader_diff.dataset.labels.numpy()
    # While they could theoretically be equal by chance, with 100 samples and random seed, it is highly unlikely
    assert not np.array_equal(train_y, train_y_diff)
