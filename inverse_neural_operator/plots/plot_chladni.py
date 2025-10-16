"""
Plot the results of the Chladni 2D dataset for inverse problem.
To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_chladni
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

device = "cpu"

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def plot_chladni_sample(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    sample,
    sample_idx,
    model_name,
    save_dir=None,
    dataset_info=None,
):
    """
    Plot a single Chladni sample with 5-panel layout:
    1. Measured displacement field (output)
    2. Predicted force field (input)
    3. True force field (input)
    4. Re-simulated displacement field (output)
    5. Error between measured and re-simulated
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

    # Get spatial dimensions from dataset_info
    if dataset_info is not None:
        h, w = dataset_info.get("input_spatial_dims", (64, 64))
    else:
        # Default to square grid
        total_points = len(s_observed_np)
        h = w = int(np.sqrt(total_points))

    # Reshape fields to 2D
    s_observed_2d = s_observed_np.reshape(h, w)
    s_resim_2d = s_resim_np.reshape(h, w)
    u_true_2d = u_true_np.reshape(h, w)
    u_pred_2d = u_pred_np.reshape(h, w)

    # Calculate error
    error_2d = np.abs(s_observed_2d - s_resim_2d)
    mse_resim = np.mean((s_observed_2d - s_resim_2d) ** 2)
    mae_resim = np.mean(np.abs(s_observed_2d - s_resim_2d))

    # Create 5-panel plot
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Panel 1: Measured Output (Displacement Field)
    im1 = axes[0].imshow(s_observed_2d, cmap="viridis", origin="lower", aspect="equal")
    axes[0].set_title("Measured Displacement s(x,y)", fontsize=12)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Panel 2: Predicted Input (Force Field)
    im2 = axes[1].imshow(u_pred_2d, cmap="RdBu_r", origin="lower", aspect="equal")
    axes[1].set_title("Predicted Force û(x,y)", fontsize=12)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Panel 3: True Input (Force Field)
    im3 = axes[2].imshow(u_true_2d, cmap="RdBu_r", origin="lower", aspect="equal")
    axes[2].set_title("True Force u(x,y)", fontsize=12)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    # Panel 4: Re-simulated Output
    im4 = axes[3].imshow(s_resim_2d, cmap="viridis", origin="lower", aspect="equal")
    axes[3].set_title("Re-simulated Displacement ŝ(x,y)", fontsize=12)
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")
    plt.colorbar(im4, ax=axes[3], fraction=0.046)

    # Panel 5: Error Field
    im5 = axes[4].imshow(error_2d, cmap="Reds", origin="lower", aspect="equal")
    axes[4].set_title(
        f"Re-simulation Error |s - ŝ|\nMSE: {mse_resim:.6f}, MAE: {mae_resim:.6f}",
        fontsize=12,
    )
    axes[4].set_xlabel("x")
    axes[4].set_ylabel("y")
    plt.colorbar(im5, ax=axes[4], fraction=0.046)

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
    dataset_info=None,
):
    """Plot multiple random samples from the test set."""

    # Select random samples
    test_indices = random.sample(
        range(len(test_dataset)), min(n_samples, len(test_dataset))
    )

    for idx in test_indices:
        sample = test_dataset[idx]
        plot_chladni_sample(
            model,
            evaluate_fn,
            input_function_encoder,
            output_function_encoder,
            forward_model,
            sample,
            idx,
            model_name,
            save_dir,
            dataset_info,
        )


def plot_model_results(
    model_name, log_dir, results_dir, test_dataset, dataset_info, n_samples=3
):
    """Plot results for a single model.

    Args:
        model_name: Name of the model
        log_dir: Complete path to the model directory (e.g., /path/to/logs/dataset/model/seed_1)
        results_dir: Directory to save results
        test_dataset: Test dataset
        dataset_info: Dataset information
        n_samples: Number of samples to plot
    """

    # log_dir is now the complete path to the model directory
    model_log_dir = log_dir

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
    forward_model = load_forward_model(log_dir=model_log_dir, forward_model_name='b2b_nonlinear', device=device)

    # Plot results
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
        dataset_info=dataset_info,
    )


# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Plot Chladni 2D results for all models."
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="/workspaces/b2b-operator-inverse/logs",
    help="Complete path to model directory (e.g., /path/to/logs/dataset/model/seed_1)",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="results/chladni_2d_plots",
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
    "--model", type=str, required=True, help="Model name to plot results for"
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
# Load dataset using the model's parameters
params = torch.load(os.path.join(log_dir, "params.pth"), weights_only=False)
test_dataset, dataset_info = load_dataset(
    params.dataset, params, device, split="test", return_info=True
)

print(f"✓ Loaded {len(test_dataset)} test samples")

# Create results directory
os.makedirs(results_dir, exist_ok=True)

print(f"Generating {args.n_samples} sample plots...")
# Plot results for the specified model
plot_model_results(
    model_name=model_name,
    log_dir=log_dir,
    results_dir=results_dir,
    test_dataset=test_dataset,
    dataset_info=dataset_info,
    n_samples=args.n_samples,
)

print(f"✓ Generated {args.n_samples} plots → {results_dir}")
