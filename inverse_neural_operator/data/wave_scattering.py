import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from datasets import load_dataset


class WaveScatteringDataset(Dataset):
    """Custom dataset for wave scattering data."""

    def __init__(self, dataset, device="cpu"):
        """
        Initialize the dataset by extracting 'theta' (input params), 'u' (input values),
        and 's' (output values) and creating coordinate grid.

        Args:
            dataset: HuggingFace dataset with 'theta', 'u', and 's' fields
            device: The device to put tensors on
        """
        self.device = device
        self.n_samples = len(dataset)

        # Extract theta, u, and s values from dataset
        # theta is a list of 100 angle values (the measurement directions)
        self.theta = torch.tensor(dataset["theta"], device=device)  # [n_samples, 100]
        self.u = torch.tensor(dataset["u"], device=device)  # [n_samples, 100, 2] - complex measurements
        self.s = torch.tensor(dataset["s"], device=device)  # [n_samples, 200, 200]

        # s is [n_samples, 200, 200]. Flatten it to [n_samples, 40000]
        self.s = self.s.view(self.s.shape[0], -1)

        # Add channel dimension to s: [n_samples, 40000, 1]
        if self.s.dim() == 2:
            self.s = self.s.unsqueeze(-1)

        # Create 1D spatial coordinates for input domain (100 measurement points)
        # X represents angular positions (0 to 2π) for the 100 measurement directions
        input_coords = torch.linspace(0, 1, 100, device=device)
        self.X = input_coords.unsqueeze(0).unsqueeze(-1).expand(self.n_samples, -1, -1)  # [n_samples, 100, 1]

        # Create a 2D meshgrid for Y coordinates (output domain)
        grid_size = 200
        x = torch.linspace(0, 1, grid_size, device=device)
        y = torch.linspace(0, 1, grid_size, device=device)
        X_grid, Y_grid = torch.meshgrid(x, y, indexing="ij")

        self.Y = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)
        self.Y = self.Y.unsqueeze(0).expand(self.s.shape[0], -1, -1)  # [n_samples, 40000, 2]

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
            
            # iFNO spatial info (hardcoded for Wave Scattering - asymmetric)
            "input_spatial_dims": (100,),        # 1D input domain (100 measurement directions)
            "output_spatial_dims": (200, 200),   # 2D output domain (200x200 spatial grid)
            "input_function_channels": 2,        # 2-channel input (real/imaginary parts of scattered wave)
            "output_function_channels": 1,       # Scalar output field
            "coordinate_dim": 2,                 # For asymmetric, actual coord dims inferred from spatial_dims
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

    ds = load_dataset("ajthor/wave_scattering", split=split)

    model_dataset = WaveScatteringDataset(ds, device=device)

    return model_dataset
