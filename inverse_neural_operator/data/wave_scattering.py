import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from datasets import load_dataset

from data.process_data import (
    InputFunctionEncoderDataset,
    OutputFunctionEncoderDataset,
)


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

    ds = load_dataset("ajthor/wave_scattering", split=split)

    model_dataset = WaveScatteringDataset(ds, device=device)
    input_fe_dataset = InputFunctionEncoderDataset(model_dataset, device=device)
    output_fe_dataset = OutputFunctionEncoderDataset(model_dataset, device=device)

    return (model_dataset, input_fe_dataset, output_fe_dataset)


def plot_instance(dataset, idx, axs=None):
    """Plot a single instance from the Wave Scattering dataset."""
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    X, u, Y, s = dataset[idx]

    # Wave scattering has theta (X) as input parameters, u as input function
    # X is the parameters, u is the function values
    theta = X.cpu().numpy()
    u_values = u.cpu().numpy()

    # For the wave scattering problem, s is a 2D field
    grid_size = int(np.sqrt(s.shape[0]))
    s_grid = s.reshape(grid_size, grid_size).cpu().numpy()

    # Plot input parameters
    axs[0].bar(range(len(theta)), theta, color="blue")

    # Plot input function
    if u_values.size > 1:  # If there's an input function
        axs[1].plot(u_values, color="red")
    else:
        axs[1].text(
            0.5,
            0.5,
            "No input function data",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Plot output field as a 2D heatmap
    im = axs[2].imshow(s_grid, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    fig = plt.gcf()
    fig.colorbar(im, ax=axs[2], shrink=0.8)

    # Add a single legend at the top center
    fig.legend(
        ["Parameters", "Input Function", "Output Field"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
    )

    plt.tight_layout()

    return axs


def plot_evaluation(result, dataset, idx, axs=None):
    """Plot the evaluation results against the ground truth for Wave Scattering."""
    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()

    X, u, Y, s = dataset[idx]

    # Wave scattering has theta (X) as input parameters
    theta = X.cpu().numpy()
    result = result.cpu().numpy()

    # For 2D wave scattering, s is the 2D field
    grid_size = int(np.sqrt(s.shape[0]))
    s_grid = s.reshape(grid_size, grid_size).cpu().numpy()

    # Plot ground truth parameters and prediction
    axs[0].bar(range(len(theta)), theta, color="blue", alpha=0.6)
    axs[0].bar(range(len(result)), result, color="red", alpha=0.6)

    # Plot absolute difference between ground truth and prediction
    diff = np.abs(theta - result)
    axs[1].bar(range(len(diff)), diff, color="purple")

    # Plot output field
    im = axs[2].imshow(s_grid, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    fig = plt.gcf()
    fig.colorbar(im, ax=axs[2], shrink=0.8)

    # Plot scatter of prediction vs ground truth
    axs[3].scatter(theta, result, color="green", alpha=0.7)
    # Add perfect prediction line
    min_val = min(np.min(theta), np.min(result))
    max_val = max(np.max(theta), np.max(result))
    axs[3].plot([min_val, max_val], [min_val, max_val], "k--")

    # Add a single legend at the top center
    fig.legend(
        ["Ground Truth", "Prediction", "Difference", "Output", "Perfect Match"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
    )

    plt.tight_layout()

    return axs
