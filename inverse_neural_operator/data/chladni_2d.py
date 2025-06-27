"""
Generate 11000 Forcing Samples for Chladni Plate (Python Version)
- Precompute everything that does NOT depend on alpha(n,m).
- Split into 10000 Training Samples + 1000 Testing Samples.
- Save to ChladniData.npz, including S(x,y) arrays.
- Optimized using NumPy vectorization for better performance.
"""

import numpy as np
from scipy.integrate import quad
from datasets import Dataset
import tqdm
import time
import os


def generate_chladni_data():
    """Generate Chladni plate simulation data."""
    
    print("Starting Chladni plate data generation...")
    
    # Create Data directory if it doesn't exist
    os.makedirs('Data', exist_ok=True)
    
    # 1) Basic Setup
    L = 8.75 * 0.0254   
    M = 8.75 * 0.0254
    omega = 55 * np.pi / M  # Frequency
    t_fixed = 4             # Time at which to evaluate the solution
    gamma = 0.02  # damping_adjustment
    v = 0.5
    numPoints = 25
    x = np.linspace(0, L, numPoints)
    y = np.linspace(0, M, numPoints)
    n_range = 10
    m_range = 10

    N_total = 11000   # total number of samples
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
    # Using vectorized operations for significant speedup
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
    
    print("Sample generation complete. Splitting and saving data...")
    
    # 4) Prepare data for SetONet training
    # ------------------------------------
    print("Preparing data for SetONet format...")
    
    # Create coordinate meshgrid
    X_coords, Y_coords = np.meshgrid(x, y, indexing='ij')
    coords_flat = np.stack([X_coords.flatten(), Y_coords.flatten()], axis=1)
    n_points = numPoints * numPoints
    
    # Prepare data arrays for SetONet format
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
    
    # Convert to Hugging Face dataset format
    print("Converting to Hugging Face format...")
    hf_ready = {k: v.tolist() for k, v in setONet_data.items()}
    ds = Dataset.from_dict(hf_ready)
    
    # Create train/test split
    ds = ds.train_test_split(test_size=N_test, shuffle=False)
    
    # Save dataset
    dataset_path = "Data/chladni_dataset"
    ds.save_to_disk(dataset_path)
    
    print("Data saved in SetONet format!")
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
    """Load the generated Chladni data in SetONet format."""
    from datasets import load_from_disk
    try:
        return load_from_disk('Data/chladni_dataset')
    except:
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


if __name__ == "__main__":
    start_time = time.time()
    
    # Generate the data
    ds = generate_chladni_data()

    # Print dataset info
    print("\nDataset Summary:")
    print(f"Training samples: {len(ds['train'])}")
    print(f"Testing samples: {len(ds['test'])}")
    print(f"Input/Output dimensions: 2D coordinates")
    print("Ready for SetONet training!") 