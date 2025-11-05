#!/usr/bin/env python
"""
Generate Elastic Plate Dataset from MATLAB Data
- Download MATLAB .mat file from Google Drive
- Process and normalize the data
- Convert to point cloud format for SetONet training
- Save in HuggingFace dataset format
- Split into training and testing samples
"""

import numpy as np
import requests
import os
from scipy.io import loadmat
from datasets import Dataset as HFDataset, DatasetDict, load_from_disk
import tqdm
import time
import json
import torch
from torch.utils.data import Dataset

# Get paths relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))


class ElasticPlateDataset(Dataset):
    """Custom dataset for Elastic Plate equation data."""

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
        """Extract info from model dataset including spatial dimensions for iFNO."""
        # Get actual dimensions from data
        n_force_points = self.X.shape[1]  # Number of boundary points where force is applied
        n_mesh_points = self.Y.shape[1]   # Number of mesh points for displacement field

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

            # iFNO spatial info (hardcoded for Elastic Plate)
            "input_spatial_dims": (n_force_points,),    # 1D along boundary edge
            "output_spatial_dims": (n_mesh_points,),    # Flattened 2D mesh
            "input_function_channels": 1,                # Scalar force field
            "output_function_channels": 1,               # Scalar displacement field (x-direction)
            "coordinate_dim": 2,                          # 2D spatial coordinates (x, y)
        }


def normalize_data(data, axis=None):
    """
    Normalize data to zero mean and unit variance.
    
    Args:
        data: numpy array to normalize
        axis: axis along which to compute statistics (None for global stats)
    
    Returns:
        normalized_data: normalized array
        mean: mean values used for normalization
        std: standard deviation values used for normalization
    """
    if axis is None:
        mean = np.mean(data)
        std = np.std(data)
    else:
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
    
    # Avoid division by zero
    std = np.where(std < 1e-8, 1.0, std)
    normalized_data = (data - mean) / std
    
    return normalized_data, mean.squeeze() if hasattr(mean, 'squeeze') else mean, std.squeeze() if hasattr(std, 'squeeze') else std

def download_matlab_data(data_path):
    """Download the elastic plate dataset from Google Drive."""
    url = "https://drive.usercontent.google.com/download?id=1m7cjNjyVJG2sNvpDnsKXw51HVOY8liDs&export=download&authuser=0&confirm=t&uuid=6f2a5c6b-b75e-4a54-acc3-53240983f7a1&at=AKSUxGNxL9MgUY7SlGHH_1kG9n60:1761768154410"
    
    print(f"Downloading elastic plate dataset to {data_path}...")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(data_path, 'wb') as f:
        with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Download complete: {data_path}")

def generate_elastic_data():
    """Generate elastic plate dataset in SetONet format."""
    
    print("Starting elastic plate data processing...")
    
    # Download data if needed
    data_path = os.path.join(current_dir, "Dataset_1Circle.mat")
    if not os.path.exists(data_path):
        download_matlab_data(data_path)
    
    # Load MATLAB data
    print(f"Loading MATLAB data from {data_path}...")
    mat_data = loadmat(data_path)
    
    # Extract data arrays
    f_train = mat_data['f_train']
    f_test = mat_data['f_test']
    x_coords = mat_data['x'].flatten()
    y_coords = mat_data['y'].flatten()
    ux_train = mat_data['ux_train']
    ux_test = mat_data['ux_test']
    
    # Combine train and test data for processing
    f_all = np.concatenate([f_train, f_test], axis=0)
    ux_all = np.concatenate([ux_train, ux_test], axis=0)
    
    # Store dimensions
    n_total = f_all.shape[0]
    n_train = f_train.shape[0]
    n_test = f_test.shape[0]
    n_force_points = f_all.shape[1]
    n_mesh_points = ux_all.shape[1]
    
    print(f"Dataset dimensions:")
    print(f"  - Total samples: {n_total}")
    print(f"  - Training samples: {n_train}")
    print(f"  - Test samples: {n_test}")
    print(f"  - Force points: {n_force_points}")
    print(f"  - Mesh points: {n_mesh_points}")
    
    # Setup coordinate systems
    print("Setting up coordinate systems...")
    
    # Force coordinates: boundary edge at x=1, y from 0 to 1 (matching displacement field domain)
    force_y_coords = np.linspace(0, 1, n_force_points, dtype=np.float32)
    force_x_coords = np.ones_like(force_y_coords, dtype=np.float32)
    
    # Mesh coordinates for displacement output
    mesh_coords = np.column_stack([x_coords, y_coords]).astype(np.float32)
    
    # Use only x-direction displacement (ux)
    displacement_x = ux_all
    
    print("Preparing data for SetONet format...")
    
    # Prepare data arrays in point cloud format
    raw_data = {
        "X": np.zeros((n_total, n_force_points, 2), dtype=np.float32),
        "u": np.zeros((n_total, n_force_points), dtype=np.float32),
        "Y": np.zeros((n_total, n_mesh_points, 2), dtype=np.float32),
        "s": np.zeros((n_total, n_mesh_points), dtype=np.float32),
    }
    
    # Force coordinates (same for all samples)
    force_coords = np.column_stack([force_x_coords, force_y_coords])
    
    print("Structuring data...")
    for k in tqdm.tqdm(range(n_total), desc="Processing samples"):
        raw_data["X"][k] = force_coords
        raw_data["Y"][k] = mesh_coords
        raw_data["u"][k] = f_all[k]
        raw_data["s"][k] = displacement_x[k]
    
    # Apply normalization
    print("Computing normalization statistics...")
    
    u_normalized, u_mean, u_std = normalize_data(raw_data["u"])
    s_normalized, s_mean, s_std = normalize_data(raw_data["s"])
    
    # No coordinate normalization needed - coordinates are already in good ranges
    X_normalized = raw_data["X"]
    Y_normalized = raw_data["Y"]
    
    # Create dataset with both normalized and original data
    full_dataset = {
        "X": X_normalized.tolist(),
        "u": u_normalized.tolist(),
        "Y": Y_normalized.tolist(),
        "s": s_normalized.tolist(),
        "X_orig": raw_data["X"].tolist(),
        "u_orig": raw_data["u"].tolist(),
        "Y_orig": raw_data["Y"].tolist(),
        "s_orig": raw_data["s"].tolist(),
    }
    
    # Save normalization statistics
    normalization_stats = {
        "u_mean": float(u_mean),
        "u_std": float(u_std),
        "s_mean": float(s_mean),
        "s_std": float(s_std),
    }
    
    # Create and save dataset
    print("Creating HuggingFace dataset...")
    ds = HFDataset.from_dict(full_dataset)
    
    # Split into train/test maintaining original split
    train_indices = list(range(n_train))
    test_indices = list(range(n_train, n_total))
    
    train_ds = ds.select(train_indices)
    test_ds = ds.select(test_indices)
    
    # Create train/test split
    final_ds = {
        'train': train_ds,
        'test': test_ds
    }
    
    # Save dataset using relative paths
    dataset_path = os.path.join(current_dir, "elastic_dataset")
    os.makedirs(dataset_path, exist_ok=True)
    
    # Save dataset
    from datasets import DatasetDict
    final_dataset = DatasetDict({'train': train_ds, 'test': test_ds})
    final_dataset.save_to_disk(dataset_path)
    
    # Save normalization statistics
    stats_path = os.path.join(current_dir, 'elastic_normalization_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(normalization_stats, f, indent=2)
    
    print(f"Dataset saved: {len(final_ds['train'])} train, {len(final_ds['test'])} test samples")
    print(f"Dataset path: {dataset_path}")
    print(f"Normalization stats saved to: {stats_path}")
    
    # Save original arrays for reference
    original_path = os.path.join(current_dir, 'ElasticData_original.npz')
    np.savez_compressed(original_path,
                       f_train=f_train, f_test=f_test,
                       ux_train=ux_train, ux_test=ux_test,
                       x_coords=x_coords, y_coords=y_coords,
                       force_x_coords=force_x_coords, force_y_coords=force_y_coords,
                       mesh_coords=mesh_coords,
                       n_train=n_train, n_test=n_test,
                       n_force_points=n_force_points, n_mesh_points=n_mesh_points,
                       u_mean=u_mean, u_std=u_std, s_mean=s_mean, s_std=s_std)
    
    print(f"Original data saved to: {original_path}")
    
    return final_dataset

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

    dataset_path = os.path.join(current_dir, 'elastic_dataset')

    # Check if dataset exists, if not generate it
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Generating elastic plate data...")
        generate_elastic_data()

    # Load the dataset
    ds = load_from_disk(dataset_path)

    # Get the requested split
    if split in ds:
        split_ds = ds[split]
    else:
        raise ValueError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")

    # Create the ElasticPlateDataset
    model_dataset = ElasticPlateDataset(split_ds, device=device)

    return model_dataset


def load_elastic_data():
    """Load the generated elastic data in SetONet format (legacy function)."""
    try:
        dataset_path = os.path.join(current_dir, 'elastic_dataset')
        ds = load_from_disk(dataset_path)
        return ds
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Run: python inverse_neural_operator/data/elastic_plate.py")
        return None

def load_elastic_normalization_stats():
    """Load the normalization statistics for the elastic dataset."""
    stats_path = os.path.join(current_dir, 'elastic_normalization_stats.json')
    with open(stats_path, 'r') as f:
        return json.load(f)

def load_elastic_original():
    """Load the original elastic data arrays."""
    original_path = os.path.join(current_dir, 'ElasticData_original.npz')
    data = np.load(original_path)
    return {key: data[key] for key in data.keys()}

if __name__ == "__main__":
    start_time = time.time()
    
    # Generate the data
    print("Starting elastic plate data generation...")
    ds = generate_elastic_data()
    
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.1f} seconds")
    print("Ready!")