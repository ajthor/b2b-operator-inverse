import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from datasets import load_dataset


class BurgersDataset(Dataset):
    """Custom dataset for Burgers 1D equation data."""

    def __init__(self, dataset, device="cpu"):
        """
        Initialize from HuggingFace dataset.

        Args:
            dataset: HuggingFace dataset with 'X', 'u', 'Y', 's' fields
            device: The device to put tensors on
        """
        self.device = device
        self.n_samples = len(dataset)

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
            # Basic info (existing)
            "X_size": self.X.shape[-1],
            "u_size": self.u.shape[-1],
            "Y_size": self.Y.shape[-1],
            "s_size": self.s.shape[-1],
            "X_len": self.X.shape[0],
            "u_len": self.u.shape[0],
            "Y_len": self.Y.shape[0],
            "s_len": self.s.shape[0],

            # iFNO spatial info (hardcoded for Burgers 1D)
            "input_spatial_dims": (101,),       # 1D spatial domain
            "output_spatial_dims": (101,),      # Same for symmetric problem
            "input_function_channels": 1,        # Scalar initial condition
            "output_function_channels": 1,       # Scalar solution
            "coordinate_dim": 1,                 # 1D spatial coordinates
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

    ds = load_dataset("ajthor/burgers_1d", split=split)

    model_dataset = BurgersDataset(ds, device=device)

    return model_dataset
