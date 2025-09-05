#!/usr/bin/env python3
"""
Plotting script for B2B forward model on Burgers 1D dataset.
Visualizes forward predictions (u -> s) for the nonlinear forward model.
To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_burgers_fwd
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

# Add project root to path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.load_dataset import load_dataset
from models.load_model import load_forward_models


def plot_forward_sample(
    model, 
    sample, 
    input_encoder, 
    output_encoder,
    evaluate_fn,
    sample_idx,
    save_path,
    device="cpu"
):
    """Plot a single forward prediction sample for 1D Burgers equation."""
    
    model.eval()
    
    X, u_true, Y, s_true = sample
    
    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_true = s_true.to(device)
    
    # Add batch dimension for evaluation
    X_batch = X.unsqueeze(0)
    u_batch = u_true.unsqueeze(0)
    Y_batch = Y.unsqueeze(0)
    s_batch = s_true.unsqueeze(0)
    
    # Get forward prediction using the evaluate function
    with torch.no_grad():
        s_pred = evaluate_fn(model, (X_batch, u_batch, Y_batch, s_batch), 
                           input_encoder, output_encoder)
        s_pred = s_pred.squeeze(0)
    
    # Convert to numpy for plotting
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    s_true_np = s_true.squeeze(-1).cpu().numpy()
    s_pred_np = s_pred.squeeze(-1).cpu().numpy()
    X_np = X.squeeze(-1).cpu().numpy()
    Y_np = Y.squeeze(-1).cpu().numpy()
    
    # Extract x coordinates for 1D plotting
    if X_np.ndim == 1:
        x_coords = X_np
        y_coords = Y_np
    else:
        x_coords = X_np[:, 0]
        y_coords = Y_np[:, 0]
    
    # Compute error
    error = np.abs(s_true_np - s_pred_np)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Input function u(x)
    axes[0, 0].plot(x_coords, u_true_np, 'b-', label='Input u(x)', linewidth=2)
    axes[0, 0].set_title('Input Function u(x)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('u(x)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: True output function s(y)
    axes[0, 1].plot(y_coords, s_true_np, 'g-', label='True s(y)', linewidth=2)
    axes[0, 1].set_title('True Output Function s(y)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('y')
    axes[0, 1].set_ylabel('s(y)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Predicted output function ŝ(y)
    axes[1, 0].plot(y_coords, s_pred_np, 'r--', label='Predicted ŝ(y)', linewidth=2)
    axes[1, 0].set_title('Predicted Output Function ŝ(y)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('y')
    axes[1, 0].set_ylabel('ŝ(y)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Comparison and error
    axes[1, 1].plot(y_coords, s_true_np, 'g-', label='True s(y)', linewidth=2)
    axes[1, 1].plot(y_coords, s_pred_np, 'r--', label='Predicted ŝ(y)', linewidth=2)
    axes[1, 1].fill_between(y_coords, s_true_np - error, s_true_np + error, 
                           alpha=0.3, color='red', label=f'Error (MAE: {np.mean(error):.4f})')
    axes[1, 1].set_title('Comparison: True vs Predicted', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('y')
    axes[1, 1].set_ylabel('s(y)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add overall title with error metrics
    mse = np.mean((s_true_np - s_pred_np) ** 2)
    mae = np.mean(error)
    max_error = np.max(error)
    
    fig.suptitle(f'Forward Model Prediction - Sample {sample_idx}\n'
                f'MSE: {mse:.6f} | MAE: {mae:.6f} | Max Error: {max_error:.6f}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return mse, mae, max_error


def main():
    parser = argparse.ArgumentParser(description="Plot forward model results on Burgers 1D dataset")
    parser.add_argument(
        "--model", 
        type=str, 
        default="b2b_nonlinear_fwd",
        help="Model name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="burgers_1d",
        help="Dataset name"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="/store/at46867/b2b_operator_inverse/burgers_1d/shared/seed_1",
        help="Log directory containing trained forward model"
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="results/burgers_1d_fwd",
        help="Directory to save results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of random samples to plot"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)"
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Setup paths
    log_dir = args.log_dir  # Use the direct path provided
    results_dir = os.path.join(args.results_dir, args.model)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading from: {log_dir}")
    print(f"Saving to: {results_dir}")
    
    # Load params
    params_path = os.path.join(log_dir, "params.pth")
    if not os.path.exists(params_path):
        print(f"Error: params.pth not found at {params_path}")
        print("Please train the forward model first using train_forward_model.py")
        return
        
    params = torch.load(params_path, weights_only=False)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset, dataset_info = load_dataset(params.dataset, params, args.device, split="test", return_info=True)
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load forward models
    print("Loading forward models...")
    input_encoder, output_encoder, model, evaluate_fn = load_forward_models(
        log_dir=log_dir, 
        dataset_info=dataset_info, 
        params=params, 
        device=args.device
    )
    
    # Select random samples to plot
    print(f"\nSelecting {args.n_samples} random samples...")
    test_indices = random.sample(
        range(len(test_dataset)), 
        min(args.n_samples, len(test_dataset))
    )
    
    # Track error statistics
    all_mse = []
    all_mae = []
    all_max_error = []
    
    # Plot each sample
    for i, idx in enumerate(test_indices):
        print(f"Plotting sample {i+1}/{args.n_samples} (index {idx})...")
        save_path = os.path.join(results_dir, f"b2b_nonlinear_fwd_forward_sample_{idx}.png")
        
        mse, mae, max_error = plot_forward_sample(
            model=model,
            sample=test_dataset[idx],
            input_encoder=input_encoder,
            output_encoder=output_encoder,
            evaluate_fn=evaluate_fn,
            sample_idx=idx,
            save_path=save_path,
            device=args.device
        )
        
        all_mse.append(mse)
        all_mae.append(mae)
        all_max_error.append(max_error)
        
        print(f"  → MSE: {mse:.6f}, MAE: {mae:.6f}, Max Error: {max_error:.6f}")
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics across {args.n_samples} samples:")
    print(f"  MSE  → Mean: {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
    print(f"  MAE  → Mean: {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
    print(f"  Max  → Mean: {np.mean(all_max_error):.6f} ± {np.std(all_max_error):.6f}")
    
    print(f"\n✅ Plotted {args.n_samples} samples → {results_dir}")


if __name__ == "__main__":
    main()