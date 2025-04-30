import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from datasets import load_dataset

from data.process_data import (
    InputFunctionEncoderDataset,
    OutputFunctionEncoderDataset,
)


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
            "X_size": self.X.shape[-1],
            "u_size": self.u.shape[-1],
            "Y_size": self.Y.shape[-1],
            "s_size": self.s.shape[-1],
            "X_len": self.X.shape[0],
            "u_len": self.u.shape[0],
            "Y_len": self.Y.shape[0],
            "s_len": self.s.shape[0],
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
    input_fe_dataset = InputFunctionEncoderDataset(model_dataset, device=device)
    output_fe_dataset = OutputFunctionEncoderDataset(model_dataset, device=device)

    return (model_dataset, input_fe_dataset, output_fe_dataset)


def plot_instance(dataset, idx, axs=None):
    """Plot a single instance from the Parametric Heat dataset."""
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    X, u, Y, s = dataset[idx]

    # For 2D heat equation, reshape the data into a grid
    grid_size = int(np.sqrt(u.shape[0]))
    u_grid = u.reshape(grid_size, grid_size).cpu().numpy()
    s_grid = s.reshape(grid_size, grid_size).cpu().numpy()

    # Create 2D heatmaps for input and output functions
    im0 = axs[0].imshow(u_grid, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    im1 = axs[1].imshow(s_grid, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")

    # Add color bars
    fig = plt.gcf()
    fig.colorbar(im0, ax=axs[0], shrink=0.8)
    fig.colorbar(im1, ax=axs[1], shrink=0.8)

    # Add a single legend at the top center
    fig.legend(
        ["Input", "Output"], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2
    )

    plt.tight_layout()

    return axs


def plot_evaluation(result, dataset, idx, axs=None):
    """Plot the evaluation results against the ground truth for Parametric Heat."""
    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()

    X, u, Y, s = dataset[idx]

    # For 2D heat equation, reshape the data into a grid
    grid_size = int(np.sqrt(u.shape[0]))
    u_grid = u.reshape(grid_size, grid_size).cpu().numpy()
    s_grid = s.reshape(grid_size, grid_size).cpu().numpy()
    result_grid = result.reshape(grid_size, grid_size).cpu().numpy()

    # Plot ground truth input
    im0 = axs[0].imshow(u_grid, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    fig = plt.gcf()
    fig.colorbar(im0, ax=axs[0], shrink=0.8)

    # Plot predicted input
    im1 = axs[1].imshow(
        result_grid, origin="lower", extent=[0, 1, 0, 1], cmap="viridis"
    )
    fig.colorbar(im1, ax=axs[1], shrink=0.8)

    # Plot output
    im2 = axs[2].imshow(s_grid, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    fig.colorbar(im2, ax=axs[2], shrink=0.8)

    # Plot difference between ground truth and prediction
    diff = u_grid - result_grid
    im3 = axs[3].imshow(diff, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm")
    fig.colorbar(im3, ax=axs[3], shrink=0.8)

    # Add a single legend at the top center
    fig.legend(
        ["Ground Truth", "Prediction", "Output", "Difference"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
    )

    plt.tight_layout()

    return axs
