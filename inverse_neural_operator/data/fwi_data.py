import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class FWIDataset(Dataset):
    """Custom dataset for Full Waveform Inversion (FWI) data with HDF5 loading."""

    def __init__(self, data_path, device="cpu", split="train"):
        """
        Initialize dataset with HDF5 file loading.

        The data is stored in HDF5 files with structure:
        - models: velocity models (batch_size, 24, 48, 1)  
        - transforms: seismic transforms (batch_size, 400, 76, 1)
        
        Args:
            data_path: Path to directory containing HDF5 files
            device: Device to load data on
            split: 'train' or 'test' split
        """
        self.device = device
        self.split = split
        
        # Set data path based on split
        if split == "train":
            self.data_path = os.path.join(data_path, "training_dataset")
        elif split == "test":
            self.data_path = os.path.join(data_path, "testing_dataset")
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Define batch configuration
        self.batch_size = 500
        
        # For training: assume 128 batches (0-127), for testing: assume 32 batches (0-31)
        if split == "train":
            self.batch_range = range(0, 128)
        else:  # test
            self.batch_range = range(0, 32)
        
        self.n_batches = len(self.batch_range)
        self.n_samples = self.n_batches * self.batch_size
        
        # Cache for currently loaded batch
        self._current_batch_idx = None
        self._current_models = None
        self._current_transforms = None
        
        # Setup coordinate grids
        # Input coordinates (24x48 velocity model grid)
        x_input = torch.linspace(0, 1, 24)
        y_input = torch.linspace(0, 1, 48)
        X_input, Y_input = torch.meshgrid(x_input, y_input, indexing="ij")
        self.X_template = torch.stack([X_input.flatten(), Y_input.flatten()], dim=1)
        
        # Output coordinates (400x76 seismic transform grid)
        x_output = torch.linspace(0, 1, 400)
        y_output = torch.linspace(0, 1, 76)
        X_output, Y_output = torch.meshgrid(x_output, y_output, indexing="ij")
        self.Y_template = torch.stack([X_output.flatten(), Y_output.flatten()], dim=1)
        
        print(f"Initialized FWI dataset: {self.n_samples} {split} samples")

    def _load_batch(self, batch_idx):
        """Load a specific batch from HDF5 file and cache it."""
        if self._current_batch_idx == batch_idx:
            return  # Already loaded
        
        batch_num = list(self.batch_range)[batch_idx]
        filepath = os.path.join(self.data_path, f"new_sor_{self.batch_size}sim_b{batch_num}.hdf5")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"HDF5 file {filepath} does not exist.")
        
        # Load data from HDF5 file
        with h5py.File(filepath, "r") as f:
            models = f["models"][:]  # Shape: (500, 24, 48, 1)
            transforms = f["transforms"][:]  # Shape: (500, 400, 76)
            
            # Reshape transforms to add channel dimension
            transforms = transforms.reshape(self.batch_size, 400, 76, 1)
        
        # Apply basic normalization
        models = self._normalize_data(models)
        transforms = self._normalize_data(transforms)
        
        # Convert to tensors and flatten spatial dimensions
        models = torch.tensor(models, dtype=torch.float32)
        transforms = torch.tensor(transforms, dtype=torch.float32)
        
        # Flatten spatial dimensions: (batch, H, W, 1) -> (batch, H*W, 1)
        models = models.view(models.shape[0], -1, 1)
        transforms = transforms.view(transforms.shape[0], -1, 1)
        
        # Cache the data
        self._current_batch_idx = batch_idx
        self._current_models = models
        self._current_transforms = transforms
        
        print(f"Loaded batch {batch_num} with {models.shape[0]} samples")

    def _normalize_data(self, data):
        """Apply basic min-max normalization to [-1, 1]."""
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            data = 2 * (data - data_min) / (data_max - data_min) - 1
        return data

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.n_samples

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
        # Determine which batch and which sample within that batch
        batch_idx = idx // self.batch_size
        sample_idx = idx % self.batch_size
        
        # Load the batch if not already loaded
        self._load_batch(batch_idx)
        
        # Get the sample
        u = self._current_models[sample_idx]  # Velocity model
        s = self._current_transforms[sample_idx]  # Seismic transform
        
        # Return with coordinate grids
        return (
            self.X_template.to(self.device),  # Input coordinates
            u.to(self.device),                # Input function (velocity model)
            self.Y_template.to(self.device),  # Output coordinates
            s.to(self.device),                # Output function (seismic transform)
        )

    def get_info(self):
        """Extract info from model dataset."""
        return {
            # Basic info
            "X_size": self.X_template.shape[-1],    # 2 (x,y coordinates)
            "u_size": 1,                            # 1 (scalar velocity values)
            "Y_size": self.Y_template.shape[-1],    # 2 (x,y coordinates)
            "s_size": 1,                            # 1 (scalar transform values)
            "X_len": self.X_template.shape[0],      # 24*48 = 1152 points
            "u_len": 24 * 48,                       # Input spatial points
            "Y_len": self.Y_template.shape[0],      # 400*76 = 30400 points
            "s_len": 400 * 76,                      # Output spatial points
            
            # iFNO spatial info
            "input_spatial_dims": (24, 48),         # 2D velocity model space
            "output_spatial_dims": (400, 76),       # 2D seismic transform space
            "input_function_channels": 1,           # Scalar velocity field
            "output_function_channels": 1,          # Scalar seismic data
            "coordinate_dim": 2,                    # 2D coordinates
        }


def load_data(params, device, split="train"):
    """
    Load a dataset from a specific split.

    Args:
        params (dict): Parameters containing dataset information, must have 'data_path'
        device (str): The device to load the data on, e.g., 'cpu' or 'cuda'
        split (str): The split of the dataset to load, either 'train' or 'test'

    Returns:
        FWIDataset: An instance of the FWIDataset class containing the dataset
    """
    # Get data path from params, with fallback to default store path
    data_path = getattr(params, 'data_path', '/store/at46867/fwi_data')
    
    return FWIDataset(data_path=data_path, device=device, split=split)


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
    im = ax.imshow(y_2d, extent=[0, 48, 24, 0], aspect="equal", cmap='magma_r')
    ax.set_title('Velocity Model')
    
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
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Seismic Transform')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    return im