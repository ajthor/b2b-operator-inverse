import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import transforms as T
from datasets import load_dataset

# Data is located in /store/at46867/fwi_data/{curve_vel,flat_vel}/data{1-60}.npy
device = "cuda:1" if torch.cuda.is_available() else "cpu"


def load_fwi_data(basepath, split="train"):
    """Load the FWI data from the specified base path."""

    # If the split is train, we load the first 25k samples
    if split == "train":
        split_range = range(1, 51)
    if split == "test":
        split_range = range(51, 61)

    # Assuming the input data files are named data1.npy, data2.npy, ..., data60.npy
    input_files = [os.path.join(basepath, f"data{i}.npy") for i in split_range]
    inputs = []
    for input_file in input_files:
        if os.path.exists(input_file):
            contents = np.load(input_file)
            print(f"Loaded {input_file} with shape {contents.shape}")
            inputs.append(contents)
        else:
            print(f"File {input_file} does not exist.")

    # Assuming the output data files are named model1.npy, model2.npy, ..., model60.npy
    output_files = [os.path.join(basepath, f"model{i}.npy") for i in split_range]
    outputs = []
    for output_file in output_files:
        if os.path.exists(output_file):
            output = np.load(output_file)
            print(f"Loaded {output_file} with shape {output.shape}")
            outputs.append(output)
        else:
            print(f"File {output_file} does not exist.")

    # Convert lists to numpy arrays
    inputs = np.array(inputs)  # Shape: (n_files, 500, 5, 1000, 70)
    outputs = np.array(outputs)  # Shape: (n_files, 500, 1, 70, 70)

    # Stack inputs and outputs along the first dimension
    inputs = np.concatenate(inputs, axis=0)  # Shape: (n_samples, 5, 1000, 70)
    outputs = np.concatenate(outputs, axis=0)  # Shape: (n_samples, 1, 70, 70)

    transform_data = Compose(
        [
            T.LogTransform(k=1),
            T.MinMaxNormalize(T.log_transform(-61, k=1), T.log_transform(120, k=1)),
        ]
    )
    transform_label = Compose([T.MinMaxNormalize(2000, 6000)])

    inputs = transform_data(inputs)
    outputs = transform_label(outputs)

    return inputs, outputs


def load_fwi_flat_vel(split="train"):
    """Load the FWI flat velocity dataset."""
    basepath = "/store/at46867/fwi_data/flat_vel"
    inputs, outputs = load_fwi_data(basepath, split=split)

    return inputs, outputs


def load_fwi_curve_vel(split="train"):
    """Load the FWI curve velocity dataset."""
    basepath = "/store/at46867/fwi_data/curve_vel"
    inputs, outputs = load_fwi_data(basepath, split=split)

    return inputs, outputs


class FWIData(Dataset):
    """Custom dataset for Full Waveform Inversion (FWI) data."""

    def __init__(self, fwi_dataset: str, device="cpu", split: str = "train"):
        """
        Extract 'u' and 's' values from the data.

        The data is stored in 60 npy files. Each file has 500 samples.
        Input data is (5,1000,70)
        Output data is (1,70,70)

        The first 25k samples are used for training, the next 5k for validation.

        """
        self.device = device

        if fwi_dataset == "flat_vel":
            u, s = load_fwi_flat_vel(split=split)
        elif fwi_dataset == "curve_vel":
            u, s = load_fwi_curve_vel(split=split)
        else:
            raise ValueError(f"Unknown dataset: {fwi_dataset}")

        self.u = torch.tensor(u, device=device)  # Input function values
        self.s = torch.tensor(s, device=device)  # Output function values

        self.n_samples = self.u.shape[0]  # Number of samples

        # Reshape u to [batch, values, 1]
        self.u = self.u.view(self.u.shape[0], -1, 1)
        # Reshape s to [batch, values, 1]
        self.s = self.s.view(self.s.shape[0], -1, 1)

        # Create an ndgrid for X coordinates
        s = torch.linspace(0, 1, 5, device=device)
        t = torch.linspace(0, 1, 1000, device=device)
        d = torch.linspace(0, 1, 70, device=device)
        _S, _T, _G = torch.meshgrid(s, t, d, indexing="ij")

        self.X = torch.stack([_S.flatten(), _T.flatten(), _G.flatten()], dim=1)
        self.X = self.X.unsqueeze(0).expand(self.u.shape[0], -1, -1)

        # Create a meshgrid for Y coordinates
        x = torch.linspace(0, 1, 70, device=device)
        y = torch.linspace(0, 1, 70, device=device)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        self.Y = torch.stack([X.flatten(), Y.flatten()], dim=1)
        self.Y = self.Y.unsqueeze(0).expand(self.u.shape[0], -1, -1)

    def __len__(self):
        """Return the number of samples in the dataset."""
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


def load_data(fwi_dataset="flat_vel", device=device, split="train"):
    """
    Load a dataset from a specific split.

    Args:
        fwi_dataset (str): The dataset to load, either 'flat_vel' or 'curve_vel'.
        device (str): The device to load the data on, e.g., 'cpu' or 'cuda'.
        split (str): The split of the dataset to load, either 'train' or 'test'.

    Returns:
        FWIData: An instance of the FWIData class containing the dataset.
    """
    return FWIData(fwi_dataset=fwi_dataset, device=device, split=split)


load_data("flat_vel", device=device, split="test")
