import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from data.process_data import (
    InputFunctionEncoderDataset,
    OutputFunctionEncoderDataset,
)


class WaveScatteringDataset(Dataset):
    """Custom dataset for wave scattering data."""

    def __init__(self, hf_dataset, grid_size=200, device="cpu"):
        """
        Initialize the dataset by extracting 'theta' (input params), 'u' (input values), 
        and 's' (output values) and creating coordinate grid.

        Args:
            hf_dataset: HuggingFace dataset with 'theta', 'u', and 's' fields
            grid_size: Size of the grid (default: 200 for 200x200 grid)
            device: The device to put tensors on
        """
        self.device = device

        # Extract theta (X), u, and s values from dataset
        self.theta = torch.tensor(hf_dataset["theta"], device=device)  # Input parameters
        self.u = torch.tensor(hf_dataset["u"], device=device)  # Input function values
        self.s = torch.tensor(hf_dataset["s"], device=device)  # Output field

        # Ensure correct dimensions
        if self.theta.dim() == 1:  # [batch]
            self.theta = self.theta.unsqueeze(-1)  # [batch, 1]
        if self.u.dim() == 2:  # [batch, values]
            self.u = self.u.unsqueeze(-1)  # [batch, values, 1]
        if self.s.dim() == 2:  # [batch, values]
            self.s = self.s.unsqueeze(-1)  # [batch, values, 1]

        # Create a meshgrid for the output spatial domain (Y coordinates)
        x = torch.linspace(0, 1, grid_size, device=device)
        y = torch.linspace(0, 1, grid_size, device=device)
        X_grid, Y_grid = torch.meshgrid(x, y, indexing="ij")
        
        # Create flattened spatial coordinates for the output domain
        self.grid_coords = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)

    def __len__(self):
        return len(self.theta)

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
            self.theta[idx],
            self.u[idx],
            self.grid_coords,
            self.s[idx],
        )

    def get_info(self):
        """Extract info from the dataset."""

        input_size = self.theta.shape[-1]
        output_size = self.s.shape[-1]
        input_len = self.theta.shape[0]
        output_len = self.grid_coords.shape[0]

        return {
            "input_size": input_size,
            "output_size": output_size,
            "input_len": input_len,
            "output_len": output_len,
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
    hf_dataset = load_dataset("ajthor/wave_scattering", split=split)

    # Create the main dataset
    model_dataset = WaveScatteringDataset(hf_dataset, grid_size=200, device=device)
    model_info = model_dataset.get_info()

    # Create the function encoder datasets
    input_fe_dataset = InputFunctionEncoderDataset(model_dataset, device=device)
    input_info = input_fe_dataset.get_info()

    output_fe_dataset = OutputFunctionEncoderDataset(model_dataset, device=device)
    output_info = output_fe_dataset.get_info()

    return (
        model_dataset,
        input_fe_dataset,
        output_fe_dataset,
        input_info,
        output_info,
        model_info,
    )
