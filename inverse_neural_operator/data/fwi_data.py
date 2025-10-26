import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from datasets import load_dataset
import json
import os
from tqdm import tqdm


class FWIDataset(Dataset):
    """Custom dataset for Full Waveform Inversion (FWI) data from HuggingFace."""

    def __init__(self, dataset, device="cpu", stats=None):
        """
        Initialize dataset from pre-processed HuggingFace dataset.

        Args:
            dataset: HuggingFace dataset with 'models' and 'transforms' fields
            device: Device to load data on
            stats: Normalization statistics (dict with mean, std, min, max)
        """
        self.device = device
        self.dataset = dataset
        self.n_samples = len(dataset)
        self.stats = stats

        # Setup coordinate grid templates (only once, not per sample)
        # Input coordinates (24x48 velocity model grid)
        x_input = torch.linspace(0, 1, 24)
        y_input = torch.linspace(0, 1, 48)
        X_input, Y_input = torch.meshgrid(x_input, y_input, indexing="ij")
        self.X_template = torch.stack([X_input.flatten(), Y_input.flatten()], dim=1)

        # Output coordinates (400x76 seismic transform grid)
        x_output = torch.linspace(0, 1, 400)
        y_output = torch.linspace(0, 1, 76)
        X_output, Y_output = torch.meshgrid(x_output, y_output, indexing="ij")
        self.Y_template = torch.stack([X_output.flatten(), Y_output.flatten()], dim=1)

        print(f"FWI dataset initialized with {self.n_samples} samples")

    def __len__(self):
        """Return the dataset size."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (X, u, Y, s) where:
            - X is the input coordinates (24x48 flattened)
            - u is the input function values (velocity model)
            - Y is the output coordinates (400x76 flattened)
            - s is the output function values (seismic transform)
        """
        # Get pre-normalized data from HuggingFace dataset
        sample = self.dataset[idx]

        # Convert to tensors and reshape
        u = torch.tensor(sample["models"], dtype=torch.float32).view(-1, 1)
        s = torch.tensor(sample["transforms"], dtype=torch.float32).view(-1, 1)

        return (
            self.X_template.to(self.device),
            u.to(self.device),
            self.Y_template.to(self.device),
            s.to(self.device),
        )

    def get_info(self):
        """Extract info from model dataset."""
        return {
            "X_size": self.X_template.shape[-1],
            "u_size": 1,
            "Y_size": self.Y_template.shape[-1],
            "s_size": 1,
            "X_len": self.X_template.shape[0],
            "u_len": 24 * 48,
            "Y_len": self.Y_template.shape[0],
            "s_len": 400 * 76,
            # iFNO spatial info
            "input_spatial_dims": (24, 48),
            "output_spatial_dims": (400, 76),
            "input_function_channels": 1,
            "output_function_channels": 1,
            "coordinate_dim": 2,
        }


def _create_linear_gradient():
    """
    Create a linear gradient array for bias subtraction.
    Gradient ranges from 100 (top) to 900 (bottom) with shape (24, 48).
    """
    gradient_1d = np.linspace(100.0, 900.0, 24)
    gradient = np.tile(gradient_1d.reshape(-1, 1), (1, 48))
    return gradient


def _compute_global_stats(dataset):
    """Compute global statistics (min, max, mean, std) using Welford's algorithm."""
    print("Computing global normalization statistics...")

    models_min = float("inf")
    models_max = float("-inf")
    transforms_min = float("inf")
    transforms_max = float("-inf")

    # Welford's algorithm for running mean and variance
    models_count = 0
    models_mean = 0.0
    models_m2 = 0.0
    transforms_count = 0
    transforms_mean = 0.0
    transforms_m2 = 0.0

    for sample in tqdm(dataset, desc="Computing normalization stats"):
        models = np.array(sample["models"])
        transforms = np.array(sample["transforms"])

        # Update min/max
        models_min = min(models_min, models.min())
        models_max = max(models_max, models.max())
        transforms_min = min(transforms_min, transforms.min())
        transforms_max = max(transforms_max, transforms.max())

        # Welford's algorithm for models
        for value in models.flat:
            models_count += 1
            delta = value - models_mean
            models_mean += delta / models_count
            delta2 = value - models_mean
            models_m2 += delta * delta2

        # Welford's algorithm for transforms
        for value in transforms.flat:
            transforms_count += 1
            delta = value - transforms_mean
            transforms_mean += delta / transforms_count
            delta2 = value - transforms_mean
            transforms_m2 += delta * delta2

    models_std = np.sqrt(models_m2 / models_count) if models_count > 1 else 0.0
    transforms_std = (
        np.sqrt(transforms_m2 / transforms_count) if transforms_count > 1 else 0.0
    )

    return {
        "models_min": float(models_min),
        "models_max": float(models_max),
        "models_mean": float(models_mean),
        "models_std": float(models_std),
        "transforms_min": float(transforms_min),
        "transforms_max": float(transforms_max),
        "transforms_mean": float(transforms_mean),
        "transforms_std": float(transforms_std),
    }


def load_data(params, device, split="train"):
    """
    Load a dataset from a specific split with caching.

    Args:
        params: Parameters containing dataset information (unused for HuggingFace)
        device (str): The device to load the data on, e.g., 'cpu' or 'cuda'
        split (str): The split of the dataset to load, either 'train' or 'test'

    Returns:
        FWIDataset: An instance of the FWIDataset class containing the dataset
    """
    # Load HuggingFace dataset
    print(f"Loading FWI dataset (split={split})...")
    ds = load_dataset("ajthor/fwi", split=split, streaming=False)

    # Determine stats file path (same directory as this file)
    # Always use train split for normalization stats
    stats_file = os.path.join(os.path.dirname(__file__), "fwi_stats.json")

    # Load or compute normalization statistics (always from train split)
    if os.path.exists(stats_file):
        print(f"Loading cached normalization statistics from {stats_file}")
        with open(stats_file, "r") as f:
            stats = json.load(f)
    else:
        # Compute statistics using streaming on train split only
        print(f"Computing normalization statistics from train split (first run)...")
        ds_streaming = load_dataset("ajthor/fwi", split="train", streaming=True)
        stats = _compute_global_stats(ds_streaming)

        # Save statistics to cache
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved normalization statistics to {stats_file}")

    # Create linear gradient for bias subtraction (created once and reused)
    gradient = _create_linear_gradient()
    stats["gradient"] = gradient
    print(f"Created linear gradient bias: [{gradient.min():.1f}, {gradient.max():.1f}]")

    def _normalize_transform(batch, stats):
        """Apply normalization transform on-the-fly."""
        # Subtract linear gradient bias from models (so model learns residuals)
        # models come as (batch_size, 24, 48, 1) from HuggingFace dataset
        models = np.array(batch["models"])
        models = models.squeeze(-1)  # Remove channel dim -> (batch_size, 24, 48)

        # # Expand gradient to (1, 24, 48) for broadcasting over batch dimension
        # gradient_expanded = np.expand_dims(stats["gradient"], axis=0)
        # models = models - gradient_expanded

        # Normalize residuals to [-1, 1] using adjusted stats
        # After subtracting gradient [100, 900]:
        # - Min residual: models_min - gradient_max = 100 - 900 = -800
        # - Max residual: models_max - gradient_min = 879.6 - 100 = 779.6
        residual_min = stats["models_min"] - 900.0
        residual_max = stats["models_max"] - 100.0
        if residual_max > residual_min:
            models = 2 * (models - residual_min) / (residual_max - residual_min) - 1
        batch["models"] = models

        # Normalize transforms to [-1, 1] using global stats
        transforms = np.array(batch["transforms"])
        if stats["transforms_max"] > stats["transforms_min"]:
            transforms = (
                2
                * (transforms - stats["transforms_min"])
                / (stats["transforms_max"] - stats["transforms_min"])
                - 1
            )
        batch["transforms"] = transforms

        return batch

    # Apply transform for on-the-fly normalization
    ds.set_transform(lambda batch: _normalize_transform(batch, stats))

    return FWIDataset(dataset=ds, device=device, stats=stats)


def plot_input(ax, _, y):
    """
    Plot the input data (velocity model).

    Args:
        ax: The axis to plot on
        _: Unused x-coordinates (kept for compatibility)
        y: The y-coordinates of the data (velocity values)
    """
    # Reshape to 2D grid for visualization (24x48)
    y_2d = y.reshape(24, 48)

    # Use extent to match notebook visualization: [0,48,24,0]
    im = ax.imshow(y_2d, extent=[0, 48, 24, 0], aspect="equal", cmap="magma_r")
    ax.set_title("Velocity Model")

    # Add colorbar
    plt.colorbar(im, ax=ax)
    return im


def plot_output(ax, _, y):
    """
    Plot the output data (seismic transform).

    Args:
        ax: The axis to plot on
        _: Unused x-coordinates (kept for compatibility)
        y: The y-coordinates of the data (seismic transform values)
    """
    # Reshape to 2D grid for visualization (400x76)
    y_2d = y.reshape(400, 76)

    # Use contour plot with frequency/time axes like in notebook
    # Frequency range: 5-81 Hz, Time range: 100-1000 ms
    freq_range = np.arange(5, 81, 1)  # 76 frequency points
    time_range = np.arange(100, 1000, 2.25)  # 400 time points

    im = ax.contourf(freq_range, time_range, y_2d, cmap="jet")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Seismic Transform")

    # Add colorbar
    plt.colorbar(im, ax=ax)
    return im
