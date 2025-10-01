import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from datasets import load_dataset


class FWIDataset(Dataset):
    """Custom dataset for Full Waveform Inversion (FWI) data from HuggingFace."""

    def __init__(self, dataset, device="cpu"):
        """
        Initialize dataset from HuggingFace dataset.

        Args:
            dataset: HuggingFace dataset with 'models' and 'transforms' fields
            device: Device to load data on
        """
        self.device = device
        self.n_samples = len(dataset)

        # Setup coordinate grids
        # Input coordinates (24x48 velocity model grid)
        x_input = torch.linspace(0, 1, 24)
        y_input = torch.linspace(0, 1, 48)
        X_input, Y_input = torch.meshgrid(x_input, y_input, indexing="ij")
        X_template = torch.stack([X_input.flatten(), Y_input.flatten()], dim=1)

        # Output coordinates (400x76 seismic transform grid)
        x_output = torch.linspace(0, 1, 400)
        y_output = torch.linspace(0, 1, 76)
        X_output, Y_output = torch.meshgrid(x_output, y_output, indexing="ij")
        Y_template = torch.stack([X_output.flatten(), Y_output.flatten()], dim=1)

        # Load data directly into tensors like other datasets
        models = torch.tensor(dataset["models"], dtype=torch.float32)
        transforms = torch.tensor(dataset["transforms"], dtype=torch.float32)

        # Apply global normalization to entire tensors
        models = self._normalize_tensor(models)
        transforms = self._normalize_tensor(transforms)

        # Flatten spatial dimensions and add channel dimension
        self.u = models.view(self.n_samples, -1, 1)  # (batch, 24*48, 1)
        self.s = transforms.view(self.n_samples, -1, 1)  # (batch, 400*76, 1)

        # Expand coordinate templates to match batch size
        self.X = X_template.unsqueeze(0).expand(self.n_samples, -1, -1)
        self.Y = Y_template.unsqueeze(0).expand(self.n_samples, -1, -1)

        print(
            f"FWI dataset loaded with shapes: X={self.X.shape}, u={self.u.shape}, Y={self.Y.shape}, s={self.s.shape}"
        )

    def __len__(self):
        """Return the dataset size."""
        return self.n_samples

    def _normalize_tensor(self, tensor):
        """Apply global min-max normalization to [-1, 1] across entire tensor."""
        tensor_min = torch.min(tensor)
        tensor_max = torch.max(tensor)
        if tensor_max > tensor_min:
            tensor = 2 * (tensor - tensor_min) / (tensor_max - tensor_min) - 1
        return tensor

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (X, u, Y, s) where:
            - X is the input coordinates (24x48 flattened)
            - u is the input function values (velocity model)
            - Y is the output coordinates (400x76 flattened)
            - s is the output function values (seismic transform)
        """
        return (
            self.X[idx].to(self.device),
            self.u[idx].to(self.device),
            self.Y[idx].to(self.device),
            self.s[idx].to(self.device),
        )

    def get_info(self):
        """Extract info from model dataset."""
        return {
            "X_size": self.X.shape[-1],
            "u_size": self.u.shape[-1],
            "Y_size": self.Y.shape[-1],
            "s_size": self.s.shape[-1],
            "X_len": self.X.shape[1],
            "u_len": self.u.shape[1],
            "Y_len": self.Y.shape[1],
            "s_len": self.s.shape[1],
            # iFNO spatial info
            "input_spatial_dims": (24, 48),
            "output_spatial_dims": (400, 76),
            "input_function_channels": 1,
            "output_function_channels": 1,
            "coordinate_dim": 2,
        }


def load_data(params, device, split="train"):
    """
    Load a dataset from a specific split.

    Args:
        params: Parameters containing dataset information (unused for HuggingFace)
        device (str): The device to load the data on, e.g., 'cpu' or 'cuda'
        split (str): The split of the dataset to load, either 'train' or 'test'

    Returns:
        FWIDataset: An instance of the FWIDataset class containing the dataset
    """
    # Load HuggingFace dataset
    ds = load_dataset("ajthor/fwi", split=split)

    return FWIDataset(dataset=ds, device=device)


def plot_input(ax, _, y):
    """
    Plot the input data (velocity model).

    Args:
        ax: The axis to plot on
        _: Unused x-coordinates (kept for compatibility)
        y: The y-coordinates of the data (velocity values)
    """
    # Reshape to 2D grid for visualization (24x48)
    y_2d = y.reshape(24, 48)

    # Use extent to match notebook visualization: [0,48,24,0]
    im = ax.imshow(y_2d, extent=[0, 48, 24, 0], aspect="equal", cmap="magma_r")
    ax.set_title("Velocity Model")

    # Add colorbar
    plt.colorbar(im, ax=ax)
    return im


def plot_output(ax, _, y):
    """
    Plot the output data (seismic transform).

    Args:
        ax: The axis to plot on
        _: Unused x-coordinates (kept for compatibility)
        y: The y-coordinates of the data (seismic transform values)
    """
    # Reshape to 2D grid for visualization (400x76)
    y_2d = y.reshape(400, 76)

    # Use contour plot with frequency/time axes like in notebook
    # Frequency range: 5-81 Hz, Time range: 100-1000 ms
    freq_range = np.arange(5, 81, 1)  # 76 frequency points
    time_range = np.arange(100, 1000, 2.25)  # 400 time points

    im = ax.contourf(freq_range, time_range, y_2d, cmap="jet")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Seismic Transform")

    # Add colorbar
    plt.colorbar(im, ax=ax)
    return im
