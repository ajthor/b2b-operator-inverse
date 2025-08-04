#!/usr/bin/env python3
"""
Standalone script to visualize IFNO results on Darcy 1D dataset.
This doesn't interfere with the existing plotting infrastructure.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from inverse_neural_operator.models.ifno import create_model
from inverse_neural_operator.data.darcy_1d import load_data

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = "cuda:1" if torch.cuda.is_available() else "cpu"

def visualize_ifno_results(model_path, n_samples=3, save_dir="./ifno_visualization/"):
    """Visualize IFNO results on Darcy 1D dataset."""
    
    print("Loading Darcy 1D test dataset...")
    test_dataset = load_data(None, device=device, split="test")
    dataset_info = test_dataset.get_info()
    
    print(f"Dataset info: {dataset_info}")
    
    print("Creating IFNO model...")
    model = create_model(
        input_size=None,  # Not used by IFNO
        hidden_sizes=[256, 256, 256],  # Not used by IFNO
        n_coupling_layers=2,
        modes1=16,
        modes2=16, 
        width=64,
        beta=2.0,
        n_layers=4,
        padding=20,
        vae_latent_dim=24,
        intermediate_dim=64,
        # IFNO-specific parameters from dataset info
        input_spatial_dims=dataset_info["input_spatial_dims"],
        output_spatial_dims=dataset_info["output_spatial_dims"], 
        input_function_channels=dataset_info["input_function_channels"],
        output_function_channels=dataset_info["output_function_channels"],
        coordinate_dim=dataset_info["coordinate_dim"],
    ).to(device)
    
    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Select random samples
    test_indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))
    
    print(f"Generating {n_samples} visualization plots...")
    
    for i, idx in enumerate(test_indices):
        sample = test_dataset[idx]
        plot_sample(model, sample, idx, save_dir)
    
    print(f"Visualization completed! Plots saved to: {save_dir}")

def plot_sample(model, sample, sample_idx, save_dir):
    """Plot a single Darcy 1D sample with IFNO results."""
    
    X, u_true, Y, s = sample
    
    # Get IFNO prediction (inverse: s -> u)
    with torch.no_grad():
        # For inverse problem: given observed output s, predict input u
        Y_batch = Y.unsqueeze(0)  # Add batch dimension
        s_batch = s.unsqueeze(0)  # Add batch dimension
        s_input = torch.cat([Y_batch, s_batch], dim=-1)
        
        # Use IFNO inverse function directly
        result = model.inverse(s_input)
        
        # Handle tuple return (IFNO returns tuple for symmetric models)
        if isinstance(result, tuple):
            u_pred, _ = result  # Extract prediction, ignore reconstruction loss
        else:
            u_pred = result
            
        # Extract function values only (remove coordinates if present)
        if u_pred.shape[-1] > u_true.unsqueeze(0).shape[-1]:
            u_pred = u_pred[..., -u_true.unsqueeze(0).shape[-1]:]  # Take last channels (function values)
            
        u_pred = u_pred.squeeze(0)
    
    # Convert to numpy for plotting
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    u_pred_np = u_pred.squeeze(-1).cpu().numpy()  
    s_np = s.squeeze(-1).cpu().numpy()
    X_np = X.squeeze(-1).cpu().numpy()
    
    # Create 1D plot with 2 subplots (matching plot_darcy.py design)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract x coordinates for 1D plotting
    if X_np.ndim == 1:
        x_coords = X_np  # Already 1D coordinates
    else:
        x_coords = X_np[:, 0]  # Use the first coordinate (x)
    
    # Plot 1: Observed output function (what we can measure)
    axes[0].plot(x_coords, s_np, 'g-', label='Observed Output Function')
    axes[0].set_title('Observed Output Function s(x)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('s(x)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Input function comparison (what we want to predict)
    axes[1].plot(x_coords, u_true_np, 'b-', label='True Input', alpha=0.7)
    axes[1].plot(x_coords, u_pred_np, 'r--', label='Predicted Input', alpha=0.7)
    axes[1].set_title('Input Function: True vs Predicted u(x)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u(x)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'ifno_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize IFNO results on Darcy 1D dataset')
    parser.add_argument('--model_path', type=str, 
                       default='./logs/darcy_1d_ifno_standalone/ifno_model.pth',
                       help='Path to trained IFNO model')
    parser.add_argument('--n_samples', type=int, default=3, 
                       help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='./ifno_visualization/',
                       help='Directory to save visualization plots')
    
    args = parser.parse_args()
    
    print("IFNO Darcy 1D Visualization")
    print("=" * 50)
    
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("Please train the model first using: python train_ifno_standalone.py")
        exit(1)
    
    visualize_ifno_results(
        model_path=args.model_path,
        n_samples=args.n_samples,
        save_dir=args.save_dir
    )