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

from inverse_neural_operator.models.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)

from data.load_dataset import load_dataset
from models.load_model import load_models, load_forward_model

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
        u_pred = evaluate_fn(
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
    """Plot results for a single model."""

    model_log_dir = os.path.join(log_dir, model_name, f"seed_{seed}")

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
    forward_model = load_forward_model(log_dir=model_log_dir, device=device)

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

    return True


# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot Burgers 1D results for all models.")
parser.add_argument(
    "--log_dir",
    type=str,
    default="/workspaces/b2b-operator-inverse/logs",
    help="Base log directory",
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
    "--seed", type=int, default=1, help="Random seed for reproducibility"
)
parser.add_argument(
    "--model",
    type=str,
    required=False,
    default=None,
    help="Model name to plot results for. If not specified, plots all available models.",
)

args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Hardcoded dataset
dataset = "burgers_1d"

# Construct paths
log_dir = os.path.join(args.log_dir, dataset)
results_dir = os.path.join(args.results_dir)

# Determine which models to plot
if args.model is not None:
    # Single model specified
    models_to_plot = [args.model]
else:
    # Auto-detect available models
    models_to_plot = []
    if os.path.exists(log_dir):
        for model_name in MODELS:
            model_path = os.path.join(
                log_dir, model_name, f"seed_{args.seed}", "params.pth"
            )
            if os.path.exists(model_path):
                models_to_plot.append(model_name)

    if not models_to_plot:
        print(f"ERROR: No trained models found in {log_dir}")
        exit(1)
    else:
        print(
            f"Found {len(models_to_plot)} models to plot: {', '.join(models_to_plot)}"
        )

# Load dataset using the first available model's parameters
first_model = models_to_plot[0]
temp_log_dir = os.path.join(log_dir, first_model, f"seed_{args.seed}")
temp_params = torch.load(os.path.join(temp_log_dir, "params.pth"), weights_only=False)

test_dataset, dataset_info = load_dataset(
    temp_params.dataset, temp_params, device, split="test", return_info=True
)

# Plot results for all selected models
total_success = 0
total_failed = []

for model_name in models_to_plot:
    # Check if the model exists
    model_path = os.path.join(log_dir, model_name, f"seed_{args.seed}", "params.pth")
    if not os.path.exists(model_path):
        print(f"WARNING: Skipping {model_name} - model not found at {model_path}")
        total_failed.append(model_name)
        continue

    # Use results_dir directly (already includes model path from plot_all.sh)
    os.makedirs(results_dir, exist_ok=True)

    # Plot results for this model
    success = plot_model_results(
        model_name=model_name,
        log_dir=log_dir,
        results_dir=results_dir,  # Use directory passed from plot_all.sh
        test_dataset=test_dataset,
        dataset_info=dataset_info,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    if success:
        print(
            f"✓ Plotted {model_name} ({args.n_samples} samples) → {results_dir}"
        )
        total_success += 1
    else:
        print(f"✗ Failed to plot {model_name}")
        total_failed.append(model_name)

# Print summary
print(f"\n{'='*50}")
if total_success > 0:
    print(f"SUCCESS: Plotted {total_success} model(s)")
if total_failed:
    print(
        f"FAILED: Could not plot {len(total_failed)} model(s): {', '.join(total_failed)}"
    )

if total_success == 0:
    exit(1)
