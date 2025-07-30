import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from datasets import load_dataset


class ParametricHeatDataset(Dataset):
    """Custom dataset for parametric heat equation data."""

    def __init__(self, dataset, grid_size=51, device="cpu"):
        """
        Initialize the dataset by extracting 'u' and 's' values and creating coordinate grid.

        Args:
            dataset: HuggingFace dataset with 'u' and 's' fields
            grid_size: Size of the square grid (default: 51)
            device: The device to put tensors on
        """
        self.device = device
        self.n_samples = len(dataset)

        self.u = torch.tensor(dataset["u"], device=device)
        self.s = torch.tensor(dataset["s"], device=device)

        # Ensure correct dimensions
        if self.u.dim() == 2:  # [batch, values]
            self.u = self.u.unsqueeze(-1)  # [batch, values, 1]
        if self.s.dim() == 2:  # [batch, values]
            self.s = self.s.unsqueeze(-1)  # [batch, values, 1]

        # Create a meshgrid for X and Y coordinates
        x = torch.linspace(0, 1, grid_size, device=device)
        y = torch.linspace(0, 1, grid_size, device=device)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        self.X = torch.stack([X.flatten(), Y.flatten()], dim=1)
        self.X = self.X.unsqueeze(0).expand(self.u.shape[0], -1, -1)

        self.Y = self.X  # Y coordinates are the same as X coordinates

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (X, u, Y, s) where:
            - X is the grid coordinates
            - u is the input function values
            - Y is the same grid coordinates (for this dataset)
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
            
            # iFNO spatial info (hardcoded for Parametric Heat)
            "input_spatial_dims": (51, 51),      # 51x51 grid
            "output_spatial_dims": (51, 51),     # Same for symmetric problem
            "input_function_channels": 1,        # Scalar parameter field
            "output_function_channels": 1,       # Scalar temperature field
            "coordinate_dim": 2,                 # 2D spatial coordinates
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

    ds = load_dataset("ajthor/parametric_heat", split=split)

    model_dataset = ParametricHeatDataset(ds, grid_size=51, device=device)

    return model_dataset
