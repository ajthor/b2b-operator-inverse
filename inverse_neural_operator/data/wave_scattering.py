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

        # Create a meshgrid for Y coordinates
        grid_size = 200
        x = torch.linspace(0, 1, grid_size, device=device)
        y = torch.linspace(0, 1, grid_size, device=device)
        X_grid, Y_grid = torch.meshgrid(x, y, indexing="ij")

        self.Y = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)
        self.Y = self.Y.unsqueeze(0).expand(self.s.shape[0], -1, -1)

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
            "input_spatial_dims": (200,),        # 1D angular domain  
            "output_spatial_dims": (200, 200),   # 2D spatial domain
            "input_function_channels": 1,        # Scalar wave parameters
            "output_function_channels": 1,       # Scalar scattered field
            "coordinate_dim": 2,                 # Output coordinates are 2D
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
