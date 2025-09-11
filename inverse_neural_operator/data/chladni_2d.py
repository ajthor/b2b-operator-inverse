"""
Chladni plate dataset for B2B operator inverse problems.
Generates normalized force and displacement data for neural operator training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, load_from_disk
from scipy.integrate import quad
import tqdm
import time
import os


class ChladniDataset(Dataset):
    """Custom dataset for Chladni plate data."""

    def __init__(self, dataset, device="cpu"):
        """
        Initialize the dataset by extracting spatial coordinates and function values.

        Args:
            dataset: HuggingFace dataset with new field structure
            device: The device to put tensors on
        """
        self.device = device
        self.n_samples = len(dataset)

        # Preload all data to GPU for fast training
        print(f"📊 Loading {self.n_samples} samples to {device}...")
        
        # Check if using new format or old format
        if "spatial_coordinates" in dataset.features:
            # New HuggingFace format
            print("Using new HuggingFace dataset format with spatial_coordinates")
            
            # spatial_coordinates is the same for all samples, so we can use the first sample
            # and expand it to all samples
            spatial_coords = torch.tensor(dataset["spatial_coordinates"], device=device, dtype=torch.float32)
            
            # spatial_coords should be [batch_size, num_points, 2] but each sample has same coordinates
            # So we take the first sample's coordinates and expand
            if len(spatial_coords.shape) == 3:  # [batch_size, num_points, 2]
                self.X = spatial_coords  # Input coordinates
                self.Y = spatial_coords  # Output coordinates (same for Chladni)
            elif len(spatial_coords.shape) == 2:  # [num_points, 2] - single set of coordinates
                # Expand to all samples
                spatial_coords = spatial_coords.unsqueeze(0).expand(self.n_samples, -1, -1)
                self.X = spatial_coords  # Input coordinates  
                self.Y = spatial_coords  # Output coordinates (same for Chladni)
            
            # Function values - flattened forcing (S) and displacement (Z) 
            self.u = torch.tensor(dataset["S"], device=device, dtype=torch.float32)  # Flattened forcing
            self.s = torch.tensor(dataset["Z"], device=device, dtype=torch.float32)  # Flattened displacement
            
        else:
            # Fallback to old format
            print("Using legacy dataset format")
            self.X = torch.tensor(dataset["X"], device=device, dtype=torch.float32)  # Input coordinates
            self.u = torch.tensor(dataset["u"], device=device, dtype=torch.float32)  # Input function (forces) 
            self.Y = torch.tensor(dataset["Y"], device=device, dtype=torch.float32)  # Output coordinates
            self.s = torch.tensor(dataset["s"], device=device, dtype=torch.float32)  # Output function (displacements)
        
        # Ensure correct dimensions
        if self.u.dim() == 2:  # [batch, values]
            self.u = self.u.unsqueeze(-1)  # [batch, values, 1]
        if self.s.dim() == 2:  # [batch, values]
            self.s = self.s.unsqueeze(-1)  # [batch, values, 1]
            
        print(f"✅ Dataset loaded: {self.n_samples} samples, {self.X.shape[1]} points")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (X, u, Y, s) where:
            - X is the input grid coordinates (normalized)
            - u is the input function values (forces, normalized)
            - Y is the output grid coordinates (normalized, same as X)
            - s is the output function values (displacements, normalized)
        """
        return (
            self.X[idx],
            self.u[idx],
            self.Y[idx], 
            self.s[idx],
        )

    def get_info(self):
        """Extract info from model dataset."""
        # Try to infer grid size from coordinates
        n_points = self.X.shape[1]
        grid_size = int(np.sqrt(n_points))  # Assume square grid
        
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
            
            # iFNO spatial info (inferred from data)
            "input_spatial_dims": (grid_size, grid_size),      # Square grid
            "output_spatial_dims": (grid_size, grid_size),     # Same for symmetric problem
            "input_function_channels": 1,        # Scalar force field
            "output_function_channels": 1,       # Scalar displacement field
            "coordinate_dim": 2,                 # 2D spatial coordinates
        }


def load_data(params=None, device="cpu", split="train"):
    """
    Load Chladni dataset from a specific split.

    Args:
        params: Parameters for processing (unused, for compatibility)
        device: The device to use
        split: The dataset split to load (default: "train")

    Returns:
        A ChladniDataset instance for the specified split
    """
    try:
        # Load from new HuggingFace dataset
        from datasets import load_dataset
        ds = load_dataset("ajthor/chladni", split=split)
        model_dataset = ChladniDataset(ds, device=device)
        return model_dataset
    except Exception as e:
        print(f"Error loading Chladni dataset from HuggingFace: {e}")
        # Fallback to local dataset
        try:
            ds = load_from_disk('Data/chladni_dataset')
            model_dataset = ChladniDataset(ds[split], device=device)
            return model_dataset
        except Exception as e2:
            print(f"Error loading local Chladni dataset: {e2}")
            print("Please ensure the dataset is available from HuggingFace or run generate_chladni_data() to create local dataset.")
            raise


def generate_chladni_data():
    """Generate Chladni plate simulation data in HuggingFace format."""
    
    print("Starting Chladni plate data generation...")
    
    # Create Data directory if it doesn't exist
    os.makedirs('Data', exist_ok=True)
    
    # 1) Basic Setup
    L = 8.75 * 0.0254   # Dimensions in meters
    M = 8.75 * 0.0254
    omega = 500 * np.pi / M  # Frequency
    t_fixed = 6             # Time at which to evaluate the solution
    
    gamma = 0.02  # damping_adjustment
    v = 0.5
    
    numPoints = 25
    x = np.linspace(0, L, numPoints)
    y = np.linspace(0, M, numPoints)
    
    n_range = 6
    m_range = 6
    
    N_total = 11000   # total number of samples
    N_train = 10000   # number of training samples
    N_test = 1000     # number of testing samples
    
    # Initialize storage arrays
    alpha_full = np.zeros((n_range, m_range, N_total))
    S_full = np.zeros((numPoints, numPoints, N_total))
    Z_full = np.zeros((numPoints, numPoints, N_total))
    
    print("Setup complete. Starting precomputation...")
    
    # 2) Precompute Terms That Don't Depend on alpha
    # ------------------------------------------------
    
    # (A) Wave numbers mu(n), lambda(m)
    mu_vals = np.arange(1, n_range + 1) * np.pi / L  # shape: (n_range,)
    lam_vals = np.arange(1, m_range + 1) * np.pi / M  # shape: (m_range,)
    
    # (B) cosX(n,i) = cos(mu_vals(n) * x(i))
    cosX = np.cos(mu_vals[:, None] * x[None, :])  # shape: (n_range, numPoints)
    
    # (C) cosY(m,j) = cos(lam_vals(m) * y(j))
    cosY = np.cos(lam_vals[:, None] * y[None, :])  # shape: (m_range, numPoints)
    
    # (D) centerFactor(n,m) = cos(mu_n*(L/2)) * cos(lam_m*(M/2))
    centerFactor = np.cos(mu_vals[:, None] * (L/2)) * np.cos(lam_vals[None, :] * (M/2))
    
    # (E) beta(n,m) = sqrt(mu^2 + lam^2 + 3*v^2 - gamma^4)
    mu_squared = mu_vals[:, None]**2  # shape: (n_range, 1)
    lam_squared = lam_vals[None, :]**2  # shape: (1, m_range)
    beta_nm = np.sqrt(mu_squared + lam_squared + 3*v**2 - gamma**4)
    
    print("Computing time integrals...")
    
    # (F) timeInt(n,m) = integral of integrand from 0 to t_fixed
    timeInt = np.zeros((n_range, m_range))
    
    for n in range(n_range):
        for m in range(m_range):
            current_beta = beta_nm[n, m]
            
            def integrand(tau):
                return (np.sin(omega * (tau - t_fixed)) * 
                       np.exp(-gamma**2 + v**2 * tau) * 
                       np.sin(current_beta * tau))
            
            timeInt[n, m], _ = quad(integrand, 0, t_fixed)
    
    # (G) modeFactor(n,m)
    modeFactor = (v**2 / beta_nm) * timeInt * (4/(L*M)) * centerFactor
    
    print("Precomputation complete. Generating samples...")
    
    # 3) Main Loop: Generate Data
    # ------------------------------------------------------------------------
    
    for k in range(N_total):
        if k % 1000 == 0:
            print(f"Generated {k}/{N_total} samples")
        
        # (A) Generate random alpha-coefficients
        alpha_k = 0.01 * np.random.randn(n_range, m_range)
        alpha_full[:, :, k] = alpha_k
        
        # (B) Compute S(i,j) vectorized
        S_k = np.einsum('nm,ni,mj->ij', alpha_k, cosX, cosY)
        S_full[:, :, k] = S_k
        
        # (C) Compute Z(i,j) vectorized  
        alpha_weighted = alpha_k * modeFactor
        Z_k = np.einsum('nm,ni,mj->ij', alpha_weighted, cosX, cosY)
        Z_full[:, :, k] = Z_k
    
    print("Sample generation complete. Preparing HuggingFace format...")
    
    # 4) Prepare data for HuggingFace format
    # ------------------------------------
    
    # Create coordinate meshgrid
    X_coords, Y_coords = np.meshgrid(x, y, indexing='ij')
    coords_flat = np.stack([X_coords.flatten(), Y_coords.flatten()], axis=1)
    n_points = numPoints * numPoints
    
    # Prepare data arrays
    setONet_data = {
        "X": np.zeros((N_total, n_points, 2), dtype=np.float32),  # Input coordinates
        "u": np.zeros((N_total, n_points), dtype=np.float32),     # Input function (S forces)
        "Y": np.zeros((N_total, n_points, 2), dtype=np.float32),  # Output coordinates
        "s": np.zeros((N_total, n_points), dtype=np.float32),     # Output function (Z displacements)
    }
    
    print("Flattening and structuring data...")
    for k in tqdm.tqdm(range(N_total), desc="Processing samples"):
        # Input and output coordinates are the same for our problem
        setONet_data["X"][k] = coords_flat
        setONet_data["Y"][k] = coords_flat
        
        # Flatten the 2D force and displacement fields
        setONet_data["u"][k] = S_full[:, :, k].flatten()  # S forces (input)
        setONet_data["s"][k] = Z_full[:, :, k].flatten()  # Z displacements (output)
    
    # Normalize data for training
    print("Normalizing data...")
    
    # Compute normalization statistics
    u_mean = setONet_data["u"].mean()
    u_std = setONet_data["u"].std()
    s_mean = setONet_data["s"].mean() 
    s_std = setONet_data["s"].std()
    xy_mean = setONet_data["X"].mean(axis=(0,1))
    xy_std = setONet_data["X"].std(axis=(0,1)) + 1e-8
    
    # Apply normalization
    setONet_data["u"] = (setONet_data["u"] - u_mean) / (u_std + 1e-8)
    setONet_data["s"] = (setONet_data["s"] - s_mean) / (s_std + 1e-8)
    setONet_data["X"] = (setONet_data["X"] - xy_mean) / xy_std
    setONet_data["Y"] = (setONet_data["Y"] - xy_mean) / xy_std
    
    # Convert to HuggingFace dataset format
    print("Converting to HuggingFace format...")
    hf_ready = {k: v.tolist() for k, v in setONet_data.items()}
    ds = HFDataset.from_dict(hf_ready)
    
    # Create train/test split
    ds = ds.train_test_split(test_size=N_test, shuffle=False)
    
    # Save dataset
    dataset_path = "Data/chladni_dataset"
    ds.save_to_disk(dataset_path)
    
    print("Data saved in HuggingFace format!")
    print(f"Training samples: {len(ds['train'])}")
    print(f"Testing samples: {len(ds['test'])}")
    print(f"Grid size: {numPoints}x{numPoints} = {n_points} points")
    print(f"Dataset saved to: {dataset_path}")
    
    # Save normalization statistics for potential denormalization
    np.savez_compressed('Data/ChladniData_normalization.npz',
                       u_mean=u_mean, u_std=u_std,
                       s_mean=s_mean, s_std=s_std,
                       xy_mean=xy_mean, xy_std=xy_std)
    
    # Also save the original arrays for reference
    np.savez_compressed('Data/ChladniData_original.npz',
                       alpha_full=alpha_full,
                       S_full=S_full,
                       Z_full=Z_full,
                       x=x, y=y,
                       L=L, M=M, omega=omega, t_fixed=t_fixed,
                       gamma=gamma, v=v, numPoints=numPoints,
                       n_range=n_range, m_range=m_range)
    
    return ds


def load_chladni_data():
    """Load the generated Chladni data in HuggingFace format."""
    try:
        # Try new HuggingFace dataset first
        from datasets import load_dataset
        return load_dataset("ajthor/chladni")
    except:
        try:
            # Fallback to local dataset
            return load_from_disk('Data/chladni_dataset')
        except:
            # Last resort - load original arrays
            data = np.load('Data/ChladniData_original.npz')
            return {key: data[key] for key in data.keys()}


def load_chladni_original():
    """Load the original Chladni data arrays."""
    data = np.load('Data/ChladniData_original.npz')
    return {key: data[key] for key in data.keys()}


def load_normalization_stats():
    """Load the normalization statistics for denormalization if needed."""
    data = np.load('Data/ChladniData_normalization.npz')
    return {key: data[key] for key in data.keys()}


def plot_input(ax, x, y):
    """
    Plot the input data (forces).

    Args:
        ax: The axis to plot on
        x: The x-coordinates of the data  
        y: The y-coordinates of the data (force values)
    """
    # Placeholder - reshape y to 2D grid for visualization
    grid_size = int(np.sqrt(len(y)))
    y_2d = y.reshape(grid_size, grid_size)
    ax.imshow(y_2d, cmap='viridis')
    ax.set_title('Input Forces')


def plot_output(ax, x, y):
    """
    Plot the output data (displacements).

    Args:
        ax: The axis to plot on
        x: The x-coordinates of the data
        y: The y-coordinates of the data (displacement values)
    """
    # Placeholder - reshape y to 2D grid for visualization  
    grid_size = int(np.sqrt(len(y)))
    y_2d = y.reshape(grid_size, grid_size)
    ax.contour(y_2d, levels=[0], colors=['gold'], linewidths=2)
    ax.set_title('Output Displacements')


if __name__ == "__main__":
    start_time = time.time()
    
    # Generate the data
    ds = generate_chladni_data()
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    # Print dataset info
    print("\nDataset Summary:")
    print(f"Training samples: {len(ds['train'])}")
    print(f"Testing samples: {len(ds['test'])}")
    print(f"Input/Output dimensions: 2D coordinates")
    print("Ready for neural operator training!") 