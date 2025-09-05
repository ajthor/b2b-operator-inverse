#!/usr/bin/env python3
"""
Plotting script for B2B forward model on Chladni 2D dataset.
Visualizes forward predictions (u -> s) for the nonlinear forward model.
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
from models.load_model import load_models


def plot_forward_sample(
    model, 
    sample, 
    input_encoder, 
    output_encoder,
    evaluate_fn,
    sample_idx,
    save_path,
    dataset_info
):
    """Plot a single forward prediction sample."""
    
    X, u_true, Y, s_true = sample
    h, w = dataset_info["input_spatial_dims"]
    
    # Add batch dimension
    X_batch = X.unsqueeze(0)
    u_batch = u_true.unsqueeze(0)
    Y_batch = Y.unsqueeze(0)
    s_batch = s_true.unsqueeze(0)
    
    # Get forward prediction
    with torch.no_grad():
        s_pred = evaluate_fn(model, (X_batch, u_batch, Y_batch, s_batch), 
                         input_encoder, output_encoder)
        s_pred = s_pred.squeeze(0)
    
    # Convert to 2D arrays
    u_true_2d = u_true.squeeze(-1).detach().cpu().numpy().reshape(h, w)
    s_true_2d = s_true.squeeze(-1).detach().cpu().numpy().reshape(h, w)
    s_pred_2d = s_pred.squeeze(-1).detach().cpu().numpy().reshape(h, w)
    
    # Compute error
    error_2d = np.abs(s_true_2d - s_pred_2d)
    
    # Create figure with 4 subplots in one row
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot input force
    im0 = axes[0].contourf(u_true_2d, levels=20, cmap="RdBu_r")
    axes[0].set_title("Input Force u(x,y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot true output
    im1 = axes[1].contourf(s_true_2d, levels=20, cmap="viridis")
    axes[1].set_title("True Displacement s(x,y)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot predicted output
    im2 = axes[2].contourf(s_pred_2d, levels=20, cmap="viridis")
    axes[2].set_title("Predicted Displacement ŝ(x,y)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Plot error
    im3 = axes[3].contourf(error_2d, levels=20, cmap="hot")
    axes[3].set_title("Absolute Error |s - ŝ|")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")
    axes[3].set_aspect("equal")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Add overall title
    fig.suptitle(f"Forward Model Prediction - Sample {sample_idx}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot forward model results on Chladni 2D dataset")
    parser.add_argument(
        "--model", 
        type=str, 
        default="b2b_nonlinear_fwd",
        help="Model name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="chladni_2d",
        help="Dataset name"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="/workspaces/b2b-operator-inverse/logs_chladni_2d",
        help="Base log directory"
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="results/chladni_2d_fwd",
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
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Setup paths
    log_dir = os.path.join(args.log_dir, args.dataset, args.model, f"seed_{args.seed}")
    results_dir = os.path.join(args.results_dir, args.model)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading from: {log_dir}")
    print(f"Saving to: {results_dir}")
    
    # Load params
    params_path = os.path.join(log_dir, "params.pth")
    if not os.path.exists(params_path):
        print(f"Error: params.pth not found at {params_path}")
        print("Please train the model first using run_Exp.sh")
        return
        
    params = torch.load(params_path, weights_only=False)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset, dataset_info = load_dataset(params.dataset, params, args.device, split="test", return_info=True)
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load models
    print("Loading models...")
    input_encoder, output_encoder, model, evaluate_fn = load_models(
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
    
    # Plot each sample
    for i, idx in enumerate(test_indices):
        print(f"Plotting sample {i+1}/{args.n_samples} (index {idx})...")
        plot_forward_sample(
            model=model,
            sample=test_dataset[idx],
            input_encoder=input_encoder,
            output_encoder=output_encoder,
            evaluate_fn=evaluate_fn,
            sample_idx=idx,
            save_path=os.path.join(results_dir, f"b2b_nonlinear_fwd_forward_sample_{idx}.png"),
            dataset_info=dataset_info
        )
    
    print(f"\n✅ Plotted {args.n_samples} samples → {results_dir}")


if __name__ == "__main__":
    main()