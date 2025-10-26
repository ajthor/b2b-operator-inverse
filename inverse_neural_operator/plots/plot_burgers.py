"""
Plot the results of the Burgers 1D dataset for all models.
To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_burgers
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random

import torch

from inverse_neural_operator.b2b.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)

from data.load_dataset import load_dataset
from models.load_model import load_models
from b2b.load_model import load_forward_model
from plots.plot_utils import find_best_worst_samples

device = "cpu"

# Available models to plot - includes forward model
MODELS = [
    "b2b_linear",
    "b2b_nonlinear",
    "b2b_nonlinear_fwd",
    "variational_autoencoder",
    "invertible_network",
    "realnvp",
    "deeponet",
]


def plot_function_encoder_realizations(
    function_encoder,
    coordinates,
    n_samples=9,
    seed=42,
    title_prefix="Input",
    save_path=None,
):
    """
    Plot random realizations from a function encoder by sampling basis coefficients.
    For Burgers (1D): plots 3x3 grid of 1D function realizations.
    """
    function_encoder.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)

    with torch.no_grad():
        # Get number of basis functions
        n_basis = function_encoder.basis_functions.num_heads

        # Sample random coefficients from standard normal distribution
        alpha = torch.randn(n_samples, n_basis, device=device)

        # Add batch dimension to coordinates and repeat for all samples
        coords_batch = coordinates.unsqueeze(0).repeat(n_samples, 1, 1).to(device)

        # Evaluate function encoder at coordinates
        functions = function_encoder(coords_batch, alpha)

    # Convert to numpy for plotting
    functions_np = functions.cpu().numpy()
    coords_np = coordinates.cpu().numpy()

    # Extract x coordinates
    if coords_np.shape[1] == 1:
        x_coords = coords_np[:, 0]
    else:
        x_coords = coords_np[:, 0]

    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for idx in range(n_samples):
        ax = axes[idx]
        function_values = functions_np[idx].squeeze()

        ax.plot(x_coords, function_values, "b-", linewidth=1.5)
        ax.set_title(f"Realization {idx + 1}", fontsize=10)
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("value", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{title_prefix} Function Encoder - Random Realizations (seed={seed})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()


def plot_forward_model_comparison_grid(
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
    For Burgers (1D): shows 9 samples with predicted vs true output.
    """
    forward_model.eval()
    input_function_encoder.eval()
    output_function_encoder.eval()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Randomly select samples from test dataset
    indices = torch.randperm(len(test_dataset))[:n_samples].tolist()

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
        Y_np = Y.cpu().numpy()

        # Extract coordinates
        if Y_np.shape[1] == 1:
            y_coords = Y_np[:, 0]
        else:
            y_coords = Y_np[:, 0]

        # Compute error
        mse = np.mean((s_true_np - s_pred_np) ** 2)

        # Plot
        ax.plot(y_coords, s_true_np, "b-", label="True", linewidth=1.5, alpha=0.7)
        ax.plot(y_coords, s_pred_np, "r--", label="Predicted", linewidth=1.5, alpha=0.7)
        ax.set_title(f"Sample {sample_idx}\nMSE: {mse:.2e}", fontsize=9)
        ax.set_xlabel("y", fontsize=8)
        ax.set_ylabel("s(y)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{model_name} Forward Model - Predictions vs True Outputs (seed={seed})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()


def plot_forward_model_sample(
    model,
    input_function_encoder,
    output_function_encoder,
    sample,
    sample_idx,
    model_name,
    save_dir=None,
):
    """
    Plot a single sample for the forward model (alpha -> beta -> s prediction).
    """
    model.eval()

    X, u_true, Y, s_true = sample

    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_true = s_true.to(device)

    with torch.no_grad():
        # Add batch dimension for encoders
        X_batch = X.unsqueeze(0)
        u_batch = u_true.unsqueeze(0)
        Y_batch = Y.unsqueeze(0)

        # Compute alpha from true input
        alpha, _ = input_function_encoder.compute_coefficients(X_batch, u_batch)

        # Forward pass through model
        beta_pred = model.forward(alpha)

        # Reconstruct predicted output
        s_pred = output_function_encoder(Y_batch, beta_pred)
        s_pred = s_pred.squeeze(0)  # Remove batch dimension

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

    # Create plot with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Input function (what we start with)
    axes[0].plot(x_coords, u_true_np, "b-", label="Input u(x)", linewidth=2)
    axes[0].set_title("Input Function u(x)", fontsize=12)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Output function comparison
    axes[1].plot(y_coords, s_true_np, "g-", label="True s(y)", linewidth=2)
    axes[1].plot(y_coords, s_pred_np, "r--", label="Predicted s(y)", linewidth=2)
    axes[1].set_title("Forward Model: Output Prediction", fontsize=12)
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("s(y)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{model_name}_forward_sample_{sample_idx}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved forward model plot → {save_path}")

    plt.close()


def plot_burgers_sample(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    sample,
    sample_idx,
    model_name,
    save_dir=None,
):
    """
    Plot a single Burgers sample with observed output, true vs predicted input, and re-simulation.
    """
    model.eval()
    forward_model.eval()

    X, u_true, Y, s_observed = sample

    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_observed = s_observed.to(device)

    # Get model prediction for input
    with torch.no_grad():
        point = (
            X.unsqueeze(0),
            u_true.unsqueeze(0),
            Y.unsqueeze(0),
            s_observed.unsqueeze(0),
        )
        u_pred, _ = evaluate_fn(
            model, point, input_function_encoder, output_function_encoder
        )
        u_pred = u_pred.squeeze(0)

    # Re-simulate using forward model
    with torch.no_grad():
        # Add batch dimension for forward model
        X_batch = X.unsqueeze(0)
        u_pred_batch = u_pred.unsqueeze(0)
        Y_batch = Y.unsqueeze(0)

        # Compute alpha coefficients from predicted input
        alpha, _ = input_function_encoder.compute_coefficients(X_batch, u_pred_batch)

        # Forward pass through model to get beta coefficients
        beta_pred = forward_model.forward(alpha)

        # Reconstruct re-simulation output
        s_resim = output_function_encoder(Y_batch, beta_pred)
        s_resim = s_resim.squeeze(0)  # Remove batch dimension

    # Convert to numpy for plotting
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    u_pred_np = u_pred.squeeze(-1).cpu().numpy()
    s_observed_np = s_observed.squeeze(-1).cpu().numpy()
    s_resim_np = s_resim.squeeze(-1).cpu().numpy()
    X_np = X.squeeze(-1).cpu().numpy()
    Y_np = Y.squeeze(-1).cpu().numpy()

    # Try to determine grid size (assuming square grid)
    n_points = len(X_np)
    grid_size = int(np.sqrt(n_points))

    # If not a perfect square, use the data as-is for 1D case
    if grid_size * grid_size != n_points:
        # Create 1D plot with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Extract x coordinates for 1D plotting
        if X_np.ndim == 1:
            x_coords = X_np  # Already 1D coordinates
            y_coords = Y_np
        else:
            x_coords = X_np[:, 0]  # Use the first coordinate (x)
            y_coords = Y_np[:, 0]  # Use the first coordinate (y)

        # Plot 1: Observed output function (what we can measure)
        axes[0].plot(
            y_coords, s_observed_np, "g-", label="Observed Output s(y)", linewidth=2
        )
        axes[0].set_title("Observed Output Function s(y)", fontsize=12)
        axes[0].set_xlabel("y")
        axes[0].set_ylabel("s(y)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Input function comparison (what we want to predict)
        axes[1].plot(
            x_coords, u_true_np, "b-", label="True Input u(x)", linewidth=2, alpha=0.8
        )
        axes[1].plot(
            x_coords,
            u_pred_np,
            "r--",
            label="Predicted Input û(x)",
            linewidth=2,
            alpha=0.8,
        )
        axes[1].set_title("Input Function: True vs Predicted", fontsize=12)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("u(x)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Re-simulation comparison
        axes[2].plot(
            y_coords,
            s_observed_np,
            "g-",
            label="True Observed s(y)",
            linewidth=2,
            alpha=0.8,
        )
        axes[2].plot(
            y_coords,
            s_resim_np,
            "m--",
            label="Re-simulated ŝ(y)",
            linewidth=2,
            alpha=0.8,
        )

        # Calculate and display error metrics
        mse_resim = np.mean((s_observed_np - s_resim_np) ** 2)
        mae_resim = np.mean(np.abs(s_observed_np - s_resim_np))

        axes[2].set_title(
            f"Re-simulation vs Observed\nMSE: {mse_resim:.6f}, MAE: {mae_resim:.6f}",
            fontsize=12,
        )
        axes[2].set_xlabel("y")
        axes[2].set_ylabel("s(y)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    else:
        # 2D visualization
        # Reshape to 2D grids
        u_true_2d = u_true_np.reshape(grid_size, grid_size)
        u_pred_2d = u_pred_np.reshape(grid_size, grid_size)
        s_observed_2d = s_observed_np.reshape(grid_size, grid_size)
        s_resim_2d = s_resim_np.reshape(grid_size, grid_size)

        # Calculate errors
        input_error_2d = np.abs(u_pred_2d - u_true_2d)
        resim_error_2d = np.abs(s_resim_2d - s_observed_2d)

        # Create the plot with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Physical domain extent (assuming normalized coordinates)
        extent = [0, 1, 0, 1]

        # Plot 1: Observed output function
        im1 = axes[0].imshow(
            s_observed_2d, cmap="viridis", extent=extent, origin="lower"
        )
        axes[0].set_title("Observed Output Function s(x,y)", fontsize=12)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Plot 2: Input prediction error
        im2 = axes[1].imshow(input_error_2d, cmap="Reds", extent=extent, origin="lower")
        axes[1].set_title("Input Prediction Error |û - u|", fontsize=12)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        # Plot 3: Re-simulation error
        im3 = axes[2].imshow(
            resim_error_2d, cmap="Blues", extent=extent, origin="lower"
        )
        mse_resim = np.mean((s_observed_2d - s_resim_2d) ** 2)
        axes[2].set_title(
            f"Re-simulation Error |ŝ - s|\nMSE: {mse_resim:.6f}", fontsize=12
        )
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        plt.colorbar(im3, ax=axes[2], fraction=0.046)

    plt.tight_layout()

    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_name}_sample_{sample_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_multiple_samples(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    test_dataset,
    model_name,
    n_samples=3,
    save_dir=None,
):
    """Plot multiple random samples from the test set."""

    # Select random samples
    test_indices = random.sample(
        range(len(test_dataset)), min(n_samples, len(test_dataset))
    )

    for i, idx in enumerate(test_indices):
        sample = test_dataset[idx]
        plot_burgers_sample(
            model,
            evaluate_fn,
            input_function_encoder,
            output_function_encoder,
            forward_model,
            sample,
            idx,
            model_name,
            save_dir,
        )


def plot_model_results(
    model_name, log_dir, results_dir, test_dataset, dataset_info, n_samples=3, seed=1
):
    """Plot results for a single model.

    Args:
        model_name: Name of the model
        log_dir: Complete path to the model directory (e.g., /path/to/logs/dataset/model/seed_1)
        results_dir: Directory to save results
        test_dataset: Test dataset
        dataset_info: Dataset information
        n_samples: Number of samples to plot
        seed: Random seed (unused, kept for compatibility)
    """

    # log_dir is now the complete path to the model directory
    model_log_dir = log_dir

    # Check if model exists
    if not os.path.exists(os.path.join(model_log_dir, "params.pth")):
        return False

    # Load model parameters
    params = torch.load(os.path.join(model_log_dir, "params.pth"), weights_only=False)

    # Load models
    input_function_encoder, output_function_encoder, model, evaluate_fn = load_models(
        log_dir=model_log_dir,
        dataset_info=dataset_info,
        params=params,
        device=device,
    )

    # Load forward model for re-simulation
    forward_model = load_forward_model(
        log_dir=model_log_dir, forward_model_name="b2b_nonlinear", device=device
    )

    # Use results_dir directly (already includes dataset/model path from plot_all.sh)
    # model_results_dir = os.path.join(results_dir, model_name)

    # Handle forward model separately
    if model_name == "b2b_nonlinear_fwd":
        # Select random samples for forward model plots
        test_indices = random.sample(
            range(len(test_dataset)), min(n_samples, len(test_dataset))
        )

        for i, idx in enumerate(test_indices):
            sample = test_dataset[idx]
            plot_forward_model_sample(
                model=model,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
                sample=sample,
                sample_idx=idx,
                model_name=model_name,
                save_dir=results_dir,
            )
    else:
        # Plot results for inverse models
        plot_multiple_samples(
            model=model,
            evaluate_fn=evaluate_fn,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
            forward_model=forward_model,
            test_dataset=test_dataset,
            model_name=model_name,
            n_samples=n_samples,
            save_dir=results_dir,
        )

        # Find and plot best/worst case samples
        print(f"Finding best and worst case samples for {model_name}...")
        best_idx, worst_idx, best_mse, worst_mse = find_best_worst_samples(
            model=model,
            evaluate_fn=evaluate_fn,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
            forward_model=forward_model,
            test_dataset=test_dataset,
            device=device,
        )

        # Plot best case
        print(f"Plotting best case (MSE: {best_mse:.6e})...")
        best_sample = test_dataset[best_idx]
        plot_burgers_sample(
            model=model,
            evaluate_fn=evaluate_fn,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
            forward_model=forward_model,
            sample=best_sample,
            sample_idx=f"best_{best_idx}",
            model_name=model_name,
            save_dir=results_dir,
        )

        # Plot worst case
        print(f"Plotting worst case (MSE: {worst_mse:.6e})...")
        worst_sample = test_dataset[worst_idx]
        plot_burgers_sample(
            model=model,
            evaluate_fn=evaluate_fn,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
            forward_model=forward_model,
            sample=worst_sample,
            sample_idx=f"worst_{worst_idx}",
            model_name=model_name,
            save_dir=results_dir,
        )

    return True


# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot Burgers 1D results for all models.")
parser.add_argument(
    "--log_dir",
    type=str,
    default="/store/at46867/b2b_operator_inverse",
    help="Complete path to model directory (e.g., /path/to/logs/dataset/model/seed_1)",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="results/burgers_plots",
    help="Results directory for saving plots",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=5,
    help="Number of random samples to plot per model",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Model name to plot results for",
)

args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# log_dir is now the complete path to the model directory
log_dir = args.log_dir
results_dir = args.results_dir
model_name = args.model

# Check if model directory exists
if not os.path.exists(os.path.join(log_dir, "params.pth")):
    print(f"✗ Model not found at {log_dir}")
    exit(1)

print(f"Loading model parameters and dataset...")
# Load dataset using model's parameters
params = torch.load(os.path.join(log_dir, "params.pth"), weights_only=False)
test_dataset, dataset_info = load_dataset(
    params.dataset, params, device, split="test", return_info=True
)
print(f"✓ Loaded {len(test_dataset)} test samples")

# Create results directory
os.makedirs(results_dir, exist_ok=True)

print(f"Generating {args.n_samples} sample plots...")
# Plot results for this model
success = plot_model_results(
    model_name=model_name,
    log_dir=log_dir,
    results_dir=results_dir,
    test_dataset=test_dataset,
    dataset_info=dataset_info,
    n_samples=args.n_samples,
    seed=args.seed,
)

if success:
    print(f"✓ Generated {args.n_samples} plots → {results_dir}")
else:
    print(f"✗ Plot generation failed")
    exit(1)
