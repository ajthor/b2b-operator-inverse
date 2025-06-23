import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from datasets import load_dataset


def log_transform(data, k=1, c=0):
    """Apply log transform to data."""
    return (np.log1p(np.abs(k * data) + c)) * np.sign(data)


def minmax_normalize(data, datamin, datamax, scale=2):
    """Min-max normalize data to [-1, 1] if scale=2, else [0, 1]."""
    data = data - datamin
    data = data / (datamax - datamin)
    if scale == 2:
        return (data - 0.5) * 2
    else:
        return data


def process_data_batch(inputs, outputs):
    """Process a batch of data with transforms."""
    # Apply log transform and minmax normalization to inputs
    inputs = log_transform(inputs, k=1)
    inputs = minmax_normalize(inputs, log_transform(-61, k=1), log_transform(120, k=1))
    # Apply minmax normalization to outputs
    outputs = minmax_normalize(outputs, 2000, 6000)
    return inputs, outputs


class FWIData(Dataset):
    """Custom dataset for Full Waveform Inversion (FWI) data with dynamic loading."""

    def __init__(self, dataset: str, device="cpu", split: str = "train"):
        """
        Initialize dataset with file paths instead of loading all data.

        The data is stored in 60 npy files. Each file has 500 samples.
        Input data is (5,1000,70)
        Output data is (1,70,70)

        The first 50 files (25k samples) are used for training, the last 10 for testing.
        """
        self.device = device
        self.split = split

        # Set base path based on dataset
        if dataset == "fwi_flat":
            self.basepath = "/store/at46867/fwi_data/flat_vel"
        elif dataset == "fwi_curve":
            self.basepath = "/store/at46867/fwi_data/curve_vel"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # Define file ranges for splits
        if split == "train":
            self.file_range = range(1, 51)  # Files 1-50
        elif split == "test":
            self.file_range = range(51, 61)  # Files 51-60
        else:
            raise ValueError(f"Unknown split: {split}")

        # Each file contains 500 samples
        self.samples_per_file = 500
        self.n_files = len(self.file_range)
        self.n_samples = self.n_files * self.samples_per_file

        # Cache for currently loaded file
        self._current_file_idx = None
        self._current_data = None
        self._current_targets = None

        # Setup coordinate grids directly in __init__
        # Create an ndgrid for X coordinates (input space)
        s = torch.linspace(0, 1, 5)
        t = torch.linspace(0, 1, 1000)
        d = torch.linspace(0, 1, 70)
        _S, _T, _G = torch.meshgrid(s, t, d, indexing="ij")

        self.X_template = torch.stack([_S.flatten(), _T.flatten(), _G.flatten()], dim=1)

        # Create a meshgrid for Y coordinates (output space)
        x = torch.linspace(0, 1, 70)
        y = torch.linspace(0, 1, 70)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        self.Y_template = torch.stack([X.flatten(), Y.flatten()], dim=1)

    def _load_file(self, file_idx):
        """Load a specific file and cache it."""
        if self._current_file_idx == file_idx:
            return  # Already loaded

        file_num = list(self.file_range)[file_idx]

        # Load input data
        input_file = os.path.join(self.basepath, f"data{file_num}.npy")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} does not exist.")

        # Load output data
        output_file = os.path.join(self.basepath, f"model{file_num}.npy")
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file {output_file} does not exist.")

        # Load and process data
        inputs = np.load(input_file)  # Shape: (500, 5, 1000, 70)
        outputs = np.load(output_file)  # Shape: (500, 1, 70, 70)

        # Apply transforms
        inputs, outputs = process_data_batch(inputs, outputs)

        # Convert to tensors and reshape
        inputs = torch.tensor(inputs, dtype=torch.float32)
        outputs = torch.tensor(outputs, dtype=torch.float32)

        # Reshape for the neural operator
        inputs = inputs.view(inputs.shape[0], -1, 1)  # [batch, values, 1]
        outputs = outputs.view(outputs.shape[0], -1, 1)  # [batch, values, 1]

        # Cache the data
        self._current_file_idx = file_idx
        self._current_data = inputs
        self._current_targets = outputs

        print(f"Loaded file {file_num} with {inputs.shape[0]} samples")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (X, u, Y, s) where:
            - X is the input coordinates
            - u is the input function values
            - Y is the grid coordinates
            - s is the output function values
        """
        # Determine which file and which sample within that file
        file_idx = idx // self.samples_per_file
        sample_idx = idx % self.samples_per_file

        # Load the file if not already loaded
        self._load_file(file_idx)

        # Get the sample
        u = self._current_data[sample_idx]
        s = self._current_targets[sample_idx]

        # Return with coordinate grids
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
            "u_size": 1,  # After reshaping
            "Y_size": self.Y_template.shape[-1],
            "s_size": 1,  # After reshaping
            "X_len": self.X_template.shape[0],
            "u_len": 5 * 1000 * 70,  # Total flattened input size
            "Y_len": self.Y_template.shape[0],
            "s_len": 70 * 70,  # Total flattened output size
        }


def load_data(params, device, split="train"):
    """
    Load a dataset from a specific split.

    Args:
        params (dict): Parameters containing dataset information.
        device (str): The device to load the data on, e.g., 'cpu' or 'cuda'.
        split (str): The split of the dataset to load, either 'train' or 'test'.

    Returns:
        FWIData: An instance of the FWIData class containing the dataset.
    """

    return FWIData(dataset=params.dataset, device=device, split=split)
