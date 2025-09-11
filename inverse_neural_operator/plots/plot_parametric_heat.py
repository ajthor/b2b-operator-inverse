"""
Plot the results of the Parametric Heat 2D dataset for all models.
To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_parametric_heat
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

    # Try to determine grid size (assuming square grid)
    n_points = len(u_true_np)
    grid_size = int(np.sqrt(n_points))

    if grid_size * grid_size == n_points:
        # 2D visualization
        u_true_2d = u_true_np.reshape(grid_size, grid_size)
        s_true_2d = s_true_np.reshape(grid_size, grid_size)
        s_pred_2d = s_pred_np.reshape(grid_size, grid_size)

        # Create plot with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Physical domain extent
        extent = [0, 1, 0, 1]

        # Plot 1: Input function (what we start with)
        im1 = axes[0].imshow(u_true_2d, cmap="viridis", extent=extent, origin="lower")
        axes[0].set_title("Input Function u(x,y)", fontsize=12)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Plot 2: Output function comparison
        im2 = axes[1].imshow(s_pred_2d, cmap="plasma", extent=extent, origin="lower")
        axes[1].set_title("Forward Model: Output Prediction s(x,y)", fontsize=12)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

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


def plot_heat_sample(
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
    Plot a single Parametric Heat sample with 5-panel figure:
    1. Observed output function
    2. Inverse model prediction (predicted input)
    3. True input
    4. Re-simulation prediction 
    5. Absolute error between observed output and re-simulation
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

    # Try to determine grid size (assuming square grid)
    n_points = len(u_true_np)
    grid_size = int(np.sqrt(n_points))

    if grid_size * grid_size == n_points:
        # 2D visualization - reshape to 2D grids
        u_true_2d = u_true_np.reshape(grid_size, grid_size)
        u_pred_2d = u_pred_np.reshape(grid_size, grid_size)
        s_observed_2d = s_observed_np.reshape(grid_size, grid_size)
        s_resim_2d = s_resim_np.reshape(grid_size, grid_size)

        # Calculate absolute error
        error_2d = np.abs(s_observed_2d - s_resim_2d)

        # Create the 5-panel figure
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        # Physical domain extent (assuming normalized coordinates)
        extent = [0, 1, 0, 1]

        # Panel 1: Observed output function
        im1 = axes[0].imshow(s_observed_2d, cmap="viridis", extent=extent, origin="lower")
        axes[0].set_title("Observed Output\nFunction s(x,y)", fontsize=12)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Panel 2: Inverse model prediction (predicted input)
        im2 = axes[1].imshow(u_pred_2d, cmap="plasma", extent=extent, origin="lower")
        axes[1].set_title("Inverse Model\nPrediction û(x,y)", fontsize=12)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        # Panel 3: True input
        im3 = axes[2].imshow(u_true_2d, cmap="plasma", extent=extent, origin="lower")
        axes[2].set_title("True Input\nFunction u(x,y)", fontsize=12)
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        plt.colorbar(im3, ax=axes[2], fraction=0.046)

        # Panel 4: Re-simulation prediction
        im4 = axes[3].imshow(s_resim_2d, cmap="viridis", extent=extent, origin="lower")
        axes[3].set_title("Re-simulation\nPrediction ŝ(x,y)", fontsize=12)
        axes[3].set_xlabel("x")
        axes[3].set_ylabel("y")
        plt.colorbar(im4, ax=axes[3], fraction=0.046)

        # Panel 5: Absolute error
        im5 = axes[4].imshow(error_2d, cmap="Reds", extent=extent, origin="lower")
        mse_resim = np.mean((s_observed_2d - s_resim_2d) ** 2)
        mae_resim = np.mean(error_2d)
        axes[4].set_title(f"Absolute Error\n|ŝ - s| (MAE: {mae_resim:.4f})", fontsize=12)
        axes[4].set_xlabel("x")
        axes[4].set_ylabel("y")
        plt.colorbar(im5, ax=axes[4], fraction=0.046)

        plt.tight_layout()

        # Save plot if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{model_name}_sample_{sample_idx}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved 5-panel plot → {save_path}")

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
        plot_heat_sample(
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
parser = argparse.ArgumentParser(description="Plot Parametric Heat 2D results for all models.")
parser.add_argument(
    "--log_dir",
    type=str,
    default="/workspaces/b2b-operator-inverse/logs",
    help="Base log directory",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="results/parametric_heat_plots",
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
dataset = "parametric_heat"

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