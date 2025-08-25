"""
Plot the results of the Darcy 1D dataset for all models.
To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_darcy
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

from inverse_neural_operator.plots.load_dataset import load_dataset
from inverse_neural_operator.plots.load_model import load_models

device = "cpu"

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Available models to plot
MODELS = [
    "b2b_linear",
    "b2b_nonlinear",
    "variational_autoencoder",
    "invertible_network",
    "realnvp",
    "deeponet",
]


def plot_darcy_sample(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    sample,
    sample_idx,
    model_name,
    save_dir=None,
):
    """
    Plot a single Darcy sample with input, prediction, ground truth, and error.
    """
    model.eval()

    X, u_true, Y, s = sample

    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s = s.to(device)

    # Get model prediction
    with torch.no_grad():
        point = (X.unsqueeze(0), u_true.unsqueeze(0), Y.unsqueeze(0), s.unsqueeze(0))
        u_pred = evaluate_fn(
            model, point, input_function_encoder, output_function_encoder
        )
        u_pred = u_pred.squeeze(0)

    # Convert to numpy for plotting
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    u_pred_np = u_pred.squeeze(-1).cpu().numpy()
    s_np = s.squeeze(-1).cpu().numpy()
    X_np = X.squeeze(-1).cpu().numpy()

    # Try to determine grid size (assuming square grid)
    n_points = len(X_np)
    grid_size = int(np.sqrt(n_points))

    # If not a perfect square, use the data as-is for 1D case
    if grid_size * grid_size != n_points:
        # Create a simple 1D plot with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Extract x coordinates for 1D plotting
        if X_np.ndim == 1:
            x_coords = X_np  # Already 1D coordinates
        else:
            x_coords = X_np[:, 0]  # Use the first coordinate (x)

        # Plot 1: Observed output function (what we can measure)
        axes[0].plot(x_coords, s_np, "g-", label="Observed Output Function")
        axes[0].set_title("Observed Output Function s(x)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("s(x)")
        axes[0].legend()
        axes[0].grid(True)

        # Plot 2: Input function comparison (what we want to predict)
        axes[1].plot(x_coords, u_true_np, "b-", label="True Input", alpha=0.7)
        axes[1].plot(x_coords, u_pred_np, "r--", label="Predicted Input", alpha=0.7)
        axes[1].set_title("Input Function: True vs Predicted u(x)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("u(x)")
        axes[1].legend()
        axes[1].grid(True)

    else:
        # 2D visualization
        # Reshape to 2D grids
        u_true_2d = u_true_np.reshape(grid_size, grid_size)
        u_pred_2d = u_pred_np.reshape(grid_size, grid_size)
        s_2d = s_np.reshape(grid_size, grid_size)

        # Calculate error
        error_2d = np.abs(u_pred_2d - u_true_2d)

        # Create the plot with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Physical domain extent (assuming normalized coordinates)
        extent = [0, 1, 0, 1]

        # Plot 1: Observed output function
        im1 = axes[0].imshow(s_2d, cmap="viridis", extent=extent, origin="lower")
        axes[0].set_title("Observed Output Function s(x,y)", fontsize=12)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Plot 2: Absolute error for input function prediction
        im2 = axes[1].imshow(error_2d, cmap="Reds", extent=extent, origin="lower")
        axes[1].set_title("Input Prediction Error |u_pred - u_true|", fontsize=12)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

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
        plot_darcy_sample(
            model,
            evaluate_fn,
            input_function_encoder,
            output_function_encoder,
            sample,
            idx,
            model_name,
            save_dir,
        )


def plot_model_results(
    model_name, log_dir, results_dir, test_dataset, dataset_info, n_samples=3
):
    """Plot results for a single model."""

    model_log_dir = os.path.join(log_dir, model_name, "seed_1")

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

    # Use results_dir directly (already includes dataset/model path from plot_all.sh)
    # model_results_dir = os.path.join(results_dir, model_name)

    # Plot results
    plot_multiple_samples(
        model=model,
        evaluate_fn=evaluate_fn,
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
        test_dataset=test_dataset,
        model_name=model_name,
        n_samples=n_samples,
        save_dir=results_dir,
    )

    return True


# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot Darcy 1D results for all models.")
parser.add_argument(
    "--log_dir",
    type=str,
    default="/workspaces/b2b-operator-inverse/logs",
    help="Base log directory",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="results/darcy_plots",
    help="Results directory for saving plots",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=3,
    help="Number of random samples to plot per model",
)
parser.add_argument(
    "--seed", type=int, default=1, help="Random seed for reproducibility"
)
parser.add_argument(
    "--model", type=str, required=True, help="Model name to plot results for"
)

args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Hardcoded dataset
dataset = "darcy_1d"

# Construct paths
log_dir = os.path.join(args.log_dir, dataset)
results_dir = args.results_dir

# Validate the specified model
model_name = args.model
if model_name not in MODELS:
    print(f"ERROR: Unknown model: {model_name}. Available models: {MODELS}")
    exit(1)

# Check if the specified model is available
model_path = os.path.join(log_dir, model_name, f"seed_{args.seed}", "params.pth")
if not os.path.exists(model_path):
    print(f"ERROR: Trained model not found: {model_path}")
    exit(1)

# Load dataset using the specified model's parameters
temp_log_dir = os.path.join(log_dir, model_name, f"seed_{args.seed}")
temp_params = torch.load(os.path.join(temp_log_dir, "params.pth"), weights_only=False)

test_dataset, dataset_info = load_dataset(temp_params, device)

# Plot results for the specified model
success = plot_model_results(
    model_name=model_name,
    log_dir=log_dir,
    results_dir=results_dir,
    test_dataset=test_dataset,
    dataset_info=dataset_info,
    n_samples=args.n_samples,
)

if success:
    print(f"SUCCESS: Plotted {model_name}, {args.n_samples} total plots")
else:
    print(f"ERROR: Failed to plot {model_name}")
    exit(1)
