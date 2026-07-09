import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Any, Dict, Optional


current_dir = os.path.dirname(os.path.abspath(__file__))
stats_path = os.path.join(current_dir, "wave_scattering_normalization_stats.json")


class WaveScatteringDataset(Dataset):
    """Custom dataset for wave scattering data."""

    def __init__(self, dataset, stats, device="cpu"):
        """
        Initialize the dataset by extracting 'theta' (input params), 'u' (input values),
        and 's' (output values) and creating coordinate grid.

        Args:
            dataset: HuggingFace dataset with 'theta', 'u', and 's' fields
            device: The device to put tensors on
        """
        self.device = device
        self.n_samples = len(dataset)
        self.stats = stats

        # Extract theta (X), u, and s values from dataset
        self.X = torch.tensor(dataset["theta"], device=device)  # Input parameters
        self.u = torch.tensor(dataset["u"], device=device)  # Input function values
        self.s = torch.tensor(dataset["s"], device=device)

        # X is in radians, so we convert it to Cartesian coordinates
        self.X = torch.stack([torch.cos(self.X), torch.sin(self.X)], dim=-1)

        # s is [1000, 200, 200]. Flatten it to [1000, 40000]
        self.s = self.s.view(self.s.shape[0], -1)

        # Ensure correct dimensions
        if self.X.dim() == 2:
            self.X = self.X.unsqueeze(-1)
        if self.u.dim() == 2:
            self.u = self.u.unsqueeze(-1)
        if self.s.dim() == 2:
            self.s = self.s.unsqueeze(-1)

        # Normalize u and s
        self.u = self._normalize(self.u, stats["u_mean"], stats["u_std"])
        self.s = self._normalize(self.s, stats["s_mean"], stats["s_std"])

        # Create a meshgrid for Y coordinates
        grid_size = 200
        x = torch.linspace(0, 1, grid_size, device=device)
        y = torch.linspace(0, 1, grid_size, device=device)
        X_grid, Y_grid = torch.meshgrid(x, y, indexing="ij")

        self.Y = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)
        self.Y = self.Y.unsqueeze(0).expand(self.s.shape[0], -1, -1)

    @staticmethod
    def _normalize(tensor, mean, std):
        mean = torch.as_tensor(mean, device=tensor.device, dtype=tensor.dtype)
        std = torch.as_tensor(std, device=tensor.device, dtype=tensor.dtype).clamp(min=1e-6)
        return (tensor - mean) / std

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (X, u, Y, s) where:
            - X is the input parameters (theta)
            - u is the input function values
            - Y is the grid coordinates
            - s is the output function values
        """
        return (
            self.X[idx],
            self.u[idx],
            self.Y[idx],
            self.s[idx],
        )

    def get_info(self):
        """Extract info from model dataset."""
        # Get actual dimensions from data
        n_input_points = self.X.shape[1]  # Number of input sampling points
        n_output_points = self.s.shape[1]  # Total output points (flattened if 2D)

        # Infer output spatial dims from flattened size (assume square grid)
        import math
        output_grid_size = int(math.sqrt(n_output_points))

        return {
            # Basic info (existing)
            "X_size": self.X.shape[-1],
            "u_size": self.u.shape[-1],
            "Y_size": self.Y.shape[-1],
            "s_size": self.s.shape[-1],
            "X_len": self.X.shape[0],
            "u_len": self.u.shape[0],
            "Y_len": self.Y.shape[0],
            "s_len": self.s.shape[0],
            # iFNO spatial info (derived from actual data - asymmetric)
            "input_spatial_dims": (n_input_points,),  # 1D angular domain
            "output_spatial_dims": (output_grid_size, output_grid_size),  # 2D spatial domain
            "input_function_channels": self.u.shape[-1],  # Actual number of function channels in u
            "output_function_channels": self.s.shape[-1],  # Actual number of function channels in s
            "coordinate_dim": 2,  # Coordinates are 2D (x, y)
            "normalization_stats": self.stats,
        }


def compute_stats(dataset):
    """Compute mean and std for u and s fields in the dataset."""
    u_data = np.array(dataset["u"])
    s_data = np.array(dataset["s"])
    return {
        "u_mean": float(np.mean(u_data)),
        "u_std": float(np.std(u_data)),
        "s_mean": float(np.mean(s_data)),
        "s_std": float(np.std(s_data)),
    }


def load_data(params, device, split="train"):
    """
    Load a dataset from a specific split.

    Args:
        params: Parameters for processing
        device: The device to use
        split: The dataset split to load (default: "train")

    Returns:
        A tuple containing the datasets and info for the specified split
    """

    # Load stats (compute from train if missing)
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        print(f"Normalization stats not found at {stats_path}. Computing from training set...")
        train_ds = load_dataset("ajthor/wave_scattering", split="train")
        stats = compute_stats(train_ds)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to {stats_path}")

    ds = load_dataset("ajthor/wave_scattering", split=split)

    model_dataset = WaveScatteringDataset(ds, stats=stats, device=device)

    return model_dataset


class WaveScatteringCpuDataset(Dataset):
    """Wave scattering dataset that keeps tensors on host until the training loop."""

    def __init__(self, dataset, stats: Dict[str, Any]):
        self.dataset = dataset
        self.stats = stats
        grid_size = 200
        x = torch.linspace(0, 1, grid_size, dtype=torch.float32)
        y = torch.linspace(0, 1, grid_size, dtype=torch.float32)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        self.output_coordinates = torch.stack(
            [x_grid.flatten(), y_grid.flatten()],
            dim=1,
        )

    @staticmethod
    def _normalize(tensor, mean, std):
        mean = torch.as_tensor(mean, dtype=tensor.dtype)
        std = torch.as_tensor(std, dtype=tensor.dtype).clamp(min=1e-6)
        return (tensor - mean) / std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        theta = torch.as_tensor(sample["theta"], dtype=torch.float32)
        X = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        u = torch.as_tensor(sample["u"], dtype=torch.float32)
        s = torch.as_tensor(sample["s"], dtype=torch.float32).reshape(-1)
        if u.dim() == 1:
            u = u.unsqueeze(-1)
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        u = self._normalize(u, self.stats["u_mean"], self.stats["u_std"])
        s = self._normalize(s, self.stats["s_mean"], self.stats["s_std"])
        return X, u, self.output_coordinates, s

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
            "output_spatial_dims": (200, 200),
            "input_function_channels": u.shape[-1],
            "output_function_channels": s.shape[-1],
            "coordinate_dim": 2,
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


def load_wave_scattering_dataset(
    *,
    split: str,
    source: str = "ajthor/wave_scattering",
    sample_limit: Optional[int] = None,
) -> WaveScatteringCpuDataset:
    stats = _load_or_compute_stats(source)
    dataset = _materialize(source, split, sample_limit)
    return WaveScatteringCpuDataset(dataset, stats)
