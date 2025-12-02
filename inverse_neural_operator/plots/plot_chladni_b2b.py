"""
Plot B2B model performance for Chladni 2D dataset.
Shows function encoder realizations and forward model predictions.

To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_chladni_b2b
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random

import torch

from b2b.load_model import (
    load_function_encoders,
    load_forward_model,
)
from data.load_dataset import load_dataset
from data.process_data import InputFunctionEncoderDataset, OutputFunctionEncoderDataset

device = "cpu"


def plot_input_function_encoder_realizations(
    input_function_encoder,
    input_encoder_dataset,
    n_samples=9,
    seed=42,
    save_path=None,
):
    """
    Plot realizations from input function encoder using actual dataset samples.
    Uses the function encoder dataset to properly split example and spatial points.
    For Chladni (2D): plots 3x3 grid of 2D plate vibration modes.
    """
    input_function_encoder.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Randomly select samples from input encoder dataset
    indices = random.sample(
        range(len(input_encoder_dataset)), min(n_samples, len(input_encoder_dataset))
    )

    with torch.no_grad():
        # Collect coefficients, spatial coordinates, and ground truth from actual data samples
        alphas = []
        all_xs = []
        all_us = []
        for idx in indices:
            # Get split data from function encoder dataset (for example points)
            example_xs, example_ys, xs, ys = input_encoder_dataset[idx]

            # Get full spatial coordinates and ground truth from base dataset
            X, u, Y, s = input_encoder_dataset.dataset[idx]

            # Add batch dimension
            example_xs = example_xs.unsqueeze(0).to(device)
            example_ys = example_ys.unsqueeze(0).to(device)
            X = X.unsqueeze(0).to(device)

            # Compute coefficients from example points
            alpha, _ = input_function_encoder.compute_coefficients(
                example_xs, example_ys
            )
            alphas.append(alpha)
            all_xs.append(X)
            all_us.append(u)

        # Stack all alphas and full spatial coordinates
        alphas = torch.cat(alphas, dim=0)
        all_xs = torch.cat(all_xs, dim=0)

        # Evaluate function encoder at full spatial coordinates using computed coefficients
        functions = input_function_encoder(all_xs, alphas)

    # Convert to numpy for plotting
    functions_np = functions.cpu().numpy()

    # Determine grid size from first sample
    n_points = all_xs.shape[1]
    grid_size = int(np.sqrt(n_points))

    # Create 3x3 grid for difference maps
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for idx in range(n_samples):
        ax = axes[idx]
        function_values = functions_np[idx].squeeze()
        ground_truth = all_us[idx].cpu().numpy().squeeze()

        # Compute MSE
        mse = np.mean((ground_truth - function_values) ** 2)

        # Reshape to 2D
        diff = (ground_truth - function_values).reshape(grid_size, grid_size)
        vmax = max(abs(diff.min()), abs(diff.max()))

        # Plot difference map
        im = ax.imshow(
            diff,
            cmap="seismic",
            origin="lower",
            extent=[0, 1, 0, 1],
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"Sample {indices[idx]}\nMSE: {mse:.2e}", fontsize=9)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Input Function Encoder - Reconstruction Error (GT - Recon) (seed={seed})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")

    plt.close()


def plot_output_function_encoder_realizations(
    output_function_encoder,
    output_encoder_dataset,
    n_samples=9,
    seed=42,
    save_path=None,
):
    """
    Plot realizations from output function encoder using actual dataset samples.
    Uses the function encoder dataset to properly split example and spatial points.
    For Chladni (2D): plots 3x3 grid of 2D output patterns.
    """
    output_function_encoder.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Randomly select samples from output encoder dataset (using same seed as input encoder)
    indices = random.sample(
        range(len(output_encoder_dataset)), min(n_samples, len(output_encoder_dataset))
    )

    with torch.no_grad():
        # Collect coefficients, spatial coordinates, and ground truth from actual data samples
        betas = []
        all_ys = []
        all_ss = []
        for idx in indices:
            # Get split data from function encoder dataset (for example points)
            example_xs, example_ys, xs, ys = output_encoder_dataset[idx]

            # Get full spatial coordinates and ground truth from base dataset
            X, u, Y, s = output_encoder_dataset.dataset[idx]

            # Add batch dimension
            example_xs = example_xs.unsqueeze(0).to(device)
            example_ys = example_ys.unsqueeze(0).to(device)
            Y = Y.unsqueeze(0).to(device)

            # Compute coefficients from example points
            beta, _ = output_function_encoder.compute_coefficients(
                example_xs, example_ys
            )
            betas.append(beta)
            all_ys.append(Y)
            all_ss.append(s)

        # Stack all betas and full spatial coordinates
        betas = torch.cat(betas, dim=0)
        all_ys = torch.cat(all_ys, dim=0)

        # Evaluate function encoder at full spatial coordinates using computed coefficients
        functions = output_function_encoder(all_ys, betas)

    # Convert to numpy for plotting
    functions_np = functions.cpu().numpy()

    # Determine grid size from first sample
    n_points = all_ys.shape[1]
    grid_size = int(np.sqrt(n_points))

    # Create 3x3 grid for difference maps
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for idx in range(n_samples):
        ax = axes[idx]
        function_values = functions_np[idx].squeeze()
        ground_truth = all_ss[idx].cpu().numpy().squeeze()

        # Compute MSE
        mse = np.mean((ground_truth - function_values) ** 2)

        # Reshape to 2D
        diff = (ground_truth - function_values).reshape(grid_size, grid_size)
        vmax = max(abs(diff.min()), abs(diff.max()))

        # Plot difference map
        im = ax.imshow(
            diff,
            cmap="seismic",
            origin="lower",
            extent=[0, 1, 0, 1],
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"Sample {indices[idx]}\nMSE: {mse:.2e}", fontsize=9)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Output Function Encoder - Reconstruction Error (GT - Recon) (seed={seed})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")

    plt.close()


def plot_forward_model_performance(
    forward_model,
    input_function_encoder,
    output_function_encoder,
    test_dataset,
    model_name,
    n_samples=9,
    seed=42,
    save_path=None,
):
    """
    Plot forward model predictions vs true outputs for random test samples in a 3x3 grid.
    For Chladni (2D): shows 9 samples with predicted vs true output patterns as difference maps.
    """
    forward_model.eval()
    input_function_encoder.eval()
    output_function_encoder.eval()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Randomly select samples from test dataset
    indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))

    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for plot_idx, sample_idx in enumerate(indices):
        ax = axes[plot_idx]

        # Get sample
        X, u, Y, s_true = test_dataset[sample_idx]
        X = X.to(device)
        u = u.to(device)
        Y = Y.to(device)
        s_true = s_true.to(device)

        with torch.no_grad():
            # Add batch dimension
            X_batch = X.unsqueeze(0)
            u_batch = u.unsqueeze(0)
            Y_batch = Y.unsqueeze(0)

            # Compute alpha coefficients from input
            alpha, _ = input_function_encoder.compute_coefficients(X_batch, u_batch)

            # Forward pass through model
            beta_pred = forward_model.forward(alpha)

            # Reconstruct predicted output
            s_pred = output_function_encoder(Y_batch, beta_pred)
            s_pred = s_pred.squeeze(0)

        # Convert to numpy
        s_true_np = s_true.squeeze().cpu().numpy()
        s_pred_np = s_pred.squeeze().cpu().numpy()

        # Compute error
        mse = np.mean((s_true_np - s_pred_np) ** 2)

        # Determine grid size
        n_points = len(s_true_np)
        grid_size = int(np.sqrt(n_points))

        # Reshape to 2D
        diff = (s_true_np - s_pred_np).reshape(grid_size, grid_size)
        vmax = max(abs(diff.min()), abs(diff.max()))

        # Plot difference
        im = ax.imshow(
            diff,
            cmap="seismic",
            origin="lower",
            extent=[0, 1, 0, 1],
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"Sample {sample_idx}\nMSE: {mse:.2e}", fontsize=9)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{model_name} Forward Model - Prediction Error (True - Pred) (seed={seed})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")

    plt.close()


# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Plot B2B model performance for Chladni 2D dataset."
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="/store/at46867/b2b_operator_inverse",
    help="Base log directory containing dataset subdirectories",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="runs",
    help="Base results directory for saving plots",
)
parser.add_argument(
    "--forward_model",
    type=str,
    choices=["b2b_linear", "b2b_nonlinear", "all"],
    default="all",
    help="Forward model to plot (default: all)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=9,
    help="Number of samples to plot (default: 9 for 3x3 grid)",
)

args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Dataset name
dataset_name = "chladni_2d"

# Construct paths
shared_log_dir = os.path.join(args.log_dir, dataset_name, "shared", "seed_1")
shared_results_dir = os.path.join(args.results_dir, dataset_name, "shared")

# Check if shared directory exists
if not os.path.exists(os.path.join(shared_log_dir, "input_function_encoder.pth")):
    print(f"✗ Function encoders not found at {shared_log_dir}")
    exit(1)

print(f"Loading function encoders and dataset...")
# Load params
params = torch.load(os.path.join(shared_log_dir, "params.pth"), weights_only=False)

# Load dataset
test_dataset, dataset_info = load_dataset(
    dataset_name, params, device, split="test", return_info=True
)
print(f"✓ Loaded {len(test_dataset)} test samples")

# Create function encoder datasets
input_encoder_dataset = InputFunctionEncoderDataset(test_dataset, device=device)
output_encoder_dataset = OutputFunctionEncoderDataset(test_dataset, device=device)
print(f"✓ Created function encoder datasets")

# Load function encoders
input_function_encoder, output_function_encoder = load_function_encoders(
    log_dir=shared_log_dir,
    dataset_info=dataset_info,
    params=params,
    device=device,
)
print(f"✓ Loaded function encoders")

# Create results directory
os.makedirs(shared_results_dir, exist_ok=True)

# Plot input function encoder realizations
print(f"Generating input function encoder realizations...")
plot_input_function_encoder_realizations(
    input_function_encoder=input_function_encoder,
    input_encoder_dataset=input_encoder_dataset,
    n_samples=args.n_samples,
    seed=args.seed,
    save_path=os.path.join(shared_results_dir, "input_encoder_realizations.png"),
)

# Plot output function encoder realizations
print(f"Generating output function encoder realizations...")
plot_output_function_encoder_realizations(
    output_function_encoder=output_function_encoder,
    output_encoder_dataset=output_encoder_dataset,
    n_samples=args.n_samples,
    seed=args.seed,
    save_path=os.path.join(shared_results_dir, "output_encoder_realizations.png"),
)

# Determine which forward models to plot
forward_models = (
    ["b2b_linear", "b2b_nonlinear"]
    if args.forward_model == "all"
    else [args.forward_model]
)

# Plot forward model performance
for forward_model_name in forward_models:
    try:
        print(f"Loading {forward_model_name} forward model...")
        forward_model = load_forward_model(
            log_dir=shared_log_dir,
            forward_model_name=forward_model_name,
            device=device,
        )
        print(f"✓ Loaded {forward_model_name}")

        print(f"Generating {forward_model_name} performance plot...")
        plot_forward_model_performance(
            forward_model=forward_model,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
            test_dataset=test_dataset,
            model_name=forward_model_name,
            n_samples=args.n_samples,
            seed=args.seed,
            save_path=os.path.join(
                shared_results_dir, f"{forward_model_name}_performance.png"
            ),
        )
    except FileNotFoundError as e:
        print(f"  ⚠ {forward_model_name} not found, skipping...")
        continue

print(f"✓ All B2B performance plots generated → {shared_results_dir}")
