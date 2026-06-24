import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Any, Dict, Optional

# Get paths relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
stats_path = os.path.join(current_dir, 'darcy_1d_normalization_stats.json')

class DarcyNormDataset(Dataset):
    """Custom dataset for Normalized Darcy 1D equation data."""

    def __init__(self, dataset, stats, device="cpu"):
        """
        Initialize from HuggingFace dataset with normalization.

        Args:
            dataset: HuggingFace dataset with 'X', 'u', 'Y', 's' fields
            stats: Dictionary containing mean and std for 'u' and 's'
            device: The device to put tensors on
        """
        self.device = device
        self.n_samples = len(dataset)
        self.stats = stats

        # Load data as tensors
        self.X = torch.tensor(dataset["X"], device=device)
        self.u = torch.tensor(dataset["u"], device=device)
        self.Y = torch.tensor(dataset["Y"], device=device)
        self.s = torch.tensor(dataset["s"], device=device)

        # Ensure correct dimensions
        if self.X.dim() == 2:
            self.X = self.X.unsqueeze(-1)
        if self.u.dim() == 2:
            self.u = self.u.unsqueeze(-1)
        if self.Y.dim() == 2:
            self.Y = self.Y.unsqueeze(-1)
        if self.s.dim() == 2:
            self.s = self.s.unsqueeze(-1)

        # Apply normalization
        self.u = self._normalize(self.u, stats['u_mean'], stats['u_std'])
        self.s = self._normalize(self.s, stats['s_mean'], stats['s_std'])

    def _normalize(self, tensor, mean, std):
        """Normalize tensor using mean and std."""
        return (tensor - mean) / std

    def _denormalize(self, tensor, mean, std):
        """Denormalize tensor using mean and std."""
        return tensor * std + mean

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (X, u, Y, s)
        """
        return (self.X[idx], self.u[idx], self.Y[idx], self.s[idx])

    def get_info(self):
        """Extract info from model dataset."""
        return {
            # Basic info
            "X_size": self.X.shape[-1],
            "u_size": self.u.shape[-1],
            "Y_size": self.Y.shape[-1],
            "s_size": self.s.shape[-1],
            "X_len": self.X.shape[0],
            "u_len": self.u.shape[0],
            "Y_len": self.Y.shape[0],
            "s_len": self.s.shape[0],

            # iFNO spatial info (hardcoded for Darcy 1D)
            "input_spatial_dims": (101,),       # 1D spatial domain
            "output_spatial_dims": (101,),      # Same for symmetric problem
            "input_function_channels": 1,        # Scalar permeability field
            "output_function_channels": 1,       # Scalar pressure field
            "coordinate_dim": 1,                 # 1D spatial coordinates
            
            # Normalization info
            "normalization_stats": self.stats
        }

def compute_stats(dataset):
    """Compute mean and std for u and s fields in the dataset."""
    u_data = np.array(dataset["u"])
    s_data = np.array(dataset["s"])
    
    return {
        "u_mean": float(np.mean(u_data)),
        "u_std": float(np.std(u_data)),
        "s_mean": float(np.mean(s_data)),
        "s_std": float(np.std(s_data))
    }

def load_data(params, device, split="train"):
    """
    Load a normalized dataset from a specific split.

    Args:
        params: Parameters for processing
        device: The device to use
        split: The dataset split to load (default: "train")

    Returns:
        DarcyNormDataset instance
    """
    
    # Load statistics or compute them from training set if missing
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    else:
        print(f"Normalization stats not found at {stats_path}. Computing from training set...")
        # Load training set to compute stats
        train_ds = load_dataset("ajthor/darcy_1d", split="train")
        stats = compute_stats(train_ds)
        
        # Save stats
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to {stats_path}")

    # Load the requested split
    ds = load_dataset("ajthor/darcy_1d", split=split)

    # Create normalized dataset
    model_dataset = DarcyNormDataset(ds, stats, device=device)

    return model_dataset


class DarcyCpuDataset(Dataset):
    """Darcy dataset that keeps tensors on host memory until the training loop."""

    def __init__(self, dataset, stats: Dict[str, Any]):
        self.dataset = dataset
        self.stats = stats

    def __len__(self):
        return len(self.dataset)

    def _normalize(self, tensor, mean, std):
        return (tensor - mean) / std

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        X = torch.as_tensor(sample["X"], dtype=torch.float32)
        u = torch.as_tensor(sample["u"], dtype=torch.float32)
        Y = torch.as_tensor(sample["Y"], dtype=torch.float32)
        s = torch.as_tensor(sample["s"], dtype=torch.float32)
        if X.dim() == 1:
            X = X.unsqueeze(-1)
        if u.dim() == 1:
            u = u.unsqueeze(-1)
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        u = self._normalize(u, self.stats["u_mean"], self.stats["u_std"])
        s = self._normalize(s, self.stats["s_mean"], self.stats["s_std"])
        return X, u, Y, s

    def get_info(self):
        X, u, Y, s = self[0]
        return {
            "X_size": X.shape[-1],
            "u_size": u.shape[-1],
            "Y_size": Y.shape[-1],
            "s_size": s.shape[-1],
            "X_len": X.shape[0],
            "u_len": u.shape[0],
            "Y_len": Y.shape[0],
            "s_len": s.shape[0],
            "input_spatial_dims": (X.shape[0],),
            "output_spatial_dims": (Y.shape[0],),
            "input_function_channels": u.shape[-1],
            "output_function_channels": s.shape[-1],
            "coordinate_dim": X.shape[-1],
            "normalization_stats": self.stats,
        }


def _load_or_compute_stats(source: str) -> Dict[str, Any]:
    if os.path.exists(stats_path):
        with open(stats_path, "r") as handle:
            return json.load(handle)
    train_ds = load_dataset(source, split="train")
    return compute_stats(train_ds)


def _materialize(source: str, split: str, sample_limit: Optional[int]):
    if sample_limit is None:
        return load_dataset(source, split=split)
    stream = load_dataset(source, split=split, streaming=True)
    samples = []
    for idx, sample in enumerate(stream):
        if idx >= sample_limit:
            break
        samples.append(dict(sample))
    return samples


def load_darcy_dataset(
    *,
    split: str,
    source: str = "ajthor/darcy_1d",
    sample_limit: Optional[int] = None,
) -> DarcyCpuDataset:
    stats = _load_or_compute_stats(source)
    dataset = _materialize(source, split, sample_limit)
    return DarcyCpuDataset(dataset, stats)
