import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from data.process_data import (
    InputFunctionEncoderDataset,
    OutputFunctionEncoderDataset,
)


class ParametricHeatDataset(Dataset):
    """Custom dataset for parametric heat equation data."""

    def __init__(self, hf_dataset, grid_size=51, device="cpu"):
        """
        Initialize the dataset by extracting 'u' and 's' values and creating coordinate grid.

        Args:
            hf_dataset: HuggingFace dataset with 'u' and 's' fields
            grid_size: Size of the square grid (default: 51)
            device: The device to put tensors on
        """
        self.device = device

        self.u = torch.tensor(hf_dataset["u"], device=device)
        self.s = torch.tensor(hf_dataset["s"], device=device)

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
        return len(self.u)

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
        """Extract info from the dataset."""

        input_size = self.X.shape[-1]
        output_size = self.Y.shape[-1]
        input_len = self.X.shape[0]
        output_len = self.Y.shape[0]

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

    ds = load_dataset("ajthor/parametric_heat", split=split)

    model_dataset = ParametricHeatDataset(ds, grid_size=51, device=device)
    model_info = model_dataset.get_info()

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
