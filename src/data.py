import os
import warnings
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import ramanspy as rp

def load_bacteria_dataset(data_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the spectra (X), labels (y), and wavenumber axis for a given split using RamanSPy.
    
    Args:
        data_dir: Path to directory containing the .npy files.
        split: Split to load ('train', 'val', or 'test').
        
    Returns:
        X: numpy array of shape (N, L) containing the spectra.
        y: numpy array of shape (N,) containing the integer labels.
        wavenumbers: numpy array of shape (L,) containing the wavenumber axis.
    """
    valid_splits = ['train', 'val', 'test']
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}, got '{split}'")
        
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")
        
    try:
        # ramanspy.datasets.bacteria returns (SpectralContainer, y)
        X_container, y = rp.datasets.bacteria(split, folder=data_dir)
        
        spectra = X_container.spectral_data
        wavenumbers = X_container.spectral_axis
        return spectra, y.astype(np.int64), wavenumbers
    except Exception as e:
        raise RuntimeError(f"Error loading split '{split}' from '{data_dir}': {e}")

class RamanSwinDataset(Dataset):
    """
    Custom PyTorch Dataset for 1D Raman Spectroscopy data.
    Performs per-spectrum Min-Max normalization and tokenization.
    """
    def __init__(
        self,
        spectra: np.ndarray,
        labels: np.ndarray,
        wavenumbers: np.ndarray,
        patch_size: int,
        normalize: bool = True,
        tokenize: bool = True
    ):
        """
        Args:
            spectra: Spectra array of shape (N, L).
            labels: Label array of shape (N,).
            wavenumbers: Wavenumber axis of shape (L,).
            patch_size: Size of non-overlapping 1D patches.
            normalize: Whether to apply per-spectrum min-max normalization.
            tokenize: Whether to apply 1D patch tokenization.
        """
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
            
        self.patch_size = patch_size
        self.normalize = normalize
        self.tokenize = tokenize
        
        # Make copies of arrays to avoid side effects
        X = np.array(spectra, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)
        w = np.array(wavenumbers, dtype=np.float32)
        
        N, L = X.shape
        
        # 1. Per-spectrum Min-Max normalization to [0, 1] with epsilon guard
        if self.normalize:
            spectra_min = X.min(axis=-1, keepdims=True)
            spectra_max = X.max(axis=-1, keepdims=True)
            diff = spectra_max - spectra_min
            safe_diff = np.where(diff > 1e-9, diff, 1.0)
            X = (X - spectra_min) / safe_diff
            
        # 2. Tokenization and Wavenumber Bookkeeping
        if self.tokenize:
            if L < patch_size:
                raise ValueError(f"Spectrum length {L} is smaller than patch size {patch_size}")
                
            num_patches = L // patch_size
            truncated_L = num_patches * patch_size
            
            if L % patch_size != 0:
                warnings.warn(
                    f"Spectrum length {L} is not divisible by patch size {patch_size}. "
                    f"Truncating to length {truncated_L}."
                )
                
            # Truncate and reshape to (N, num_patches, patch_size)
            self.spectra = torch.tensor(X[:, :truncated_L].reshape(N, num_patches, patch_size), dtype=torch.float32)
            
            # Bookkeeping: representative wavenumbers (mean of each patch)
            truncated_w = w[:truncated_L]
            rep_w = truncated_w.reshape(num_patches, patch_size).mean(axis=-1)
            self.representative_wavenumbers = rep_w
            self.wavenumbers = rep_w
        else:
            self.spectra = torch.tensor(X, dtype=torch.float32)
            self.representative_wavenumbers = w
            self.wavenumbers = w
            
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spectra[idx], self.labels[idx]

def get_dataloaders(
    data_dir: str,
    patch_size: int,
    batch_size: int,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Splits reference train set into 80/20 train/validation splits.
    Creates and returns PyTorch DataLoaders and the original spectral axis.
    
    Args:
        data_dir: Directory containing the .npy files.
        patch_size: Size of 1D patches.
        batch_size: Batch size for dataloaders.
        seed: Random seed for reproducibility of splits.
        
    Returns:
        ref_train_loader: DataLoader for 80% reference train (shuffled).
        ref_val_loader: DataLoader for 20% reference validation (unshuffled).
        ft_val_loader: DataLoader for fine-tune validation (unshuffled).
        test_loader: DataLoader for test split (unshuffled).
        spectral_axis: Original wavenumber axis of the spectra.
    """
    # Load Reference Train data
    X_ref, y_ref, spectral_axis = load_bacteria_dataset(data_dir, 'train')
    
    # 80/20 stratified split with fixed random seed
    X_ref_train, X_ref_val, y_ref_train, y_ref_val = train_test_split(
        X_ref, y_ref, test_size=0.2, random_state=seed, stratify=y_ref
    )
    
    # Load Fine-tune Validation and Test data
    X_ft_val, y_ft_val, _ = load_bacteria_dataset(data_dir, 'val')
    X_test, y_test, _ = load_bacteria_dataset(data_dir, 'test')
    
    # Create Dataset instances
    train_dataset = RamanSwinDataset(X_ref_train, y_ref_train, spectral_axis, patch_size, normalize=True, tokenize=True)
    val_dataset = RamanSwinDataset(X_ref_val, y_ref_val, spectral_axis, patch_size, normalize=True, tokenize=True)
    ft_val_dataset = RamanSwinDataset(X_ft_val, y_ft_val, spectral_axis, patch_size, normalize=True, tokenize=True)
    test_dataset = RamanSwinDataset(X_test, y_test, spectral_axis, patch_size, normalize=True, tokenize=True)
    
    # Create DataLoaders
    ref_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    ref_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    ft_val_loader = DataLoader(ft_val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return ref_train_loader, ref_val_loader, ft_val_loader, test_loader, spectral_axis
