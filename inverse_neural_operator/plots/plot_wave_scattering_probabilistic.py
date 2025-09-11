"""
Plot probabilistic results for the Wave Scattering dataset with multiple realizations.

For probabilistic models, samples 10 realizations from the posterior and takes the
average of the realizations for visualization, since individual realizations would
be too cluttered for 2D wave scattering data.

To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_wave_scattering_probabilistic
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

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def plot_wave_scattering_probabilistic_sample(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    sample,
    sample_idx,
    model_name,
    n_realizations=10,
    save_dir=None,
):
    """
    Plot a single wave scattering sample with probabilistic realizations.
    For wave scattering, takes the mean of realizations for cleaner visualization.
    """
    model.eval()
    forward_model.eval()

    X, u_true, Y, s_observed = sample

    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_observed = s_observed.to(device)

    # Check if model supports probabilistic sampling
    is_probabilistic = (
        hasattr(model, "sample_posterior")
        or hasattr(model, "sample_prior")
        or (
            hasattr(model, "inverse")
            and hasattr(model, "forward")
            and "mixture" in model.__class__.__name__.lower()
        )
    )

    if is_probabilistic:
        # Probabilistic model - sample multiple realizations
        with torch.no_grad():
            # Add batch dimension
            X_batch = X.unsqueeze(0)
            Y_batch = Y.unsqueeze(0)
            s_observed_batch = s_observed.unsqueeze(0)

            # Compute beta coefficients from observed output
            beta_observed, _ = output_function_encoder.compute_coefficients(
                Y_batch, s_observed_batch
            )

            # Sample based on model type
            alpha_samples_list = []

            if hasattr(model, "sample_posterior"):
                # INN models
                alpha_samples = model.sample_posterior(beta_observed, n_realizations)
                alpha_samples_list = [alpha_samples[i] for i in range(n_realizations)]
            elif hasattr(model, "sample_prior"):
                # VAE model
                batch_size = beta_observed.shape[0]
                for _ in range(n_realizations):
                    z = model.sample_prior(batch_size, device=beta_observed.device)
                    alpha_sample = model.inverse(beta_observed, z)
                    alpha_samples_list.append(alpha_sample)
            else:
                # MDN model - call inverse multiple times
                for _ in range(n_realizations):
                    alpha_sample = model.inverse(beta_observed)
                    alpha_samples_list.append(alpha_sample)

            # Generate input functions and outputs from alpha samples
            u_samples = []
            s_resim_samples = []

            for alpha_sample in alpha_samples_list:
                # Reconstruct input function from alpha
                u_sample = input_function_encoder(X_batch, alpha_sample)
                u_samples.append(u_sample.squeeze(0))

                # Forward simulate to get output
                beta_resim = forward_model.forward(alpha_sample)
                s_resim = output_function_encoder(Y_batch, beta_resim)
                s_resim_samples.append(s_resim.squeeze(0))

            # Compute mean and std of realizations
            u_samples_tensor = torch.stack(
                u_samples
            )  # [n_realizations, spatial_points, features]
            s_resim_samples_tensor = torch.stack(s_resim_samples)

            u_mean = torch.mean(u_samples_tensor, dim=0)
            u_std = torch.std(u_samples_tensor, dim=0)
            s_resim_mean = torch.mean(s_resim_samples_tensor, dim=0)
            s_resim_std = torch.std(s_resim_samples_tensor, dim=0)

    else:
        # Deterministic model - use single prediction
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

            # Re-simulate for deterministic model
            X_batch = X.unsqueeze(0)
            u_pred_batch = u_pred
            Y_batch = Y.unsqueeze(0)

            alpha, _ = input_function_encoder.compute_coefficients(
                X_batch, u_pred_batch
            )
            beta_pred = forward_model.forward(alpha)
            s_resim = output_function_encoder(Y_batch, beta_pred)

            u_mean = u_pred.squeeze(0)
            u_std = torch.zeros_like(u_mean)
            s_resim_mean = s_resim.squeeze(0)
            s_resim_std = torch.zeros_like(s_resim_mean)

    # Convert to numpy for plotting
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    u_mean_np = u_mean.squeeze(-1).cpu().numpy()
    u_std_np = u_std.squeeze(-1).cpu().numpy()
    s_observed_np = s_observed.squeeze(-1).cpu().numpy()
    s_resim_mean_np = s_resim_mean.squeeze(-1).cpu().numpy()
    s_resim_std_np = s_resim_std.squeeze(-1).cpu().numpy()
    X_np = X.squeeze(-1).cpu().numpy()

    # Reshape density fields from flattened (40000,) to 2D (200, 200)
    grid_size = 200
    s_observed_2d = s_observed_np.reshape(grid_size, grid_size)
    s_resim_mean_2d = s_resim_mean_np.reshape(grid_size, grid_size)
    s_resim_std_2d = s_resim_std_np.reshape(grid_size, grid_size)

    s_resim_mean_2d_binary = (s_resim_mean_2d > 0.5).astype(float)

    # Calculate error between mean re-simulation and observation
    error_2d = np.abs(s_observed_2d - s_resim_mean_2d)
    mse_resim = np.mean((s_observed_2d - s_resim_mean_2d) ** 2)
    mae_resim = np.mean(np.abs(s_observed_2d - s_resim_mean_2d))

    # Extract theta coordinates for polar plots (X_np contains [cos(theta), sin(theta)])
    theta = np.arctan2(X_np[:, 1], X_np[:, 0])  # Convert back from Cartesian to angles

    # Ensure theta matches the size of u arrays by trimming to minimum size
    min_size = min(len(theta), len(u_true_np), len(u_mean_np))
    theta = theta[:min_size]
    u_true_np = u_true_np[:min_size]
    u_mean_np = u_mean_np[:min_size]
    u_std_np = u_std_np[:min_size]

    # Create 6-panel plot (original 5 + uncertainty visualization)
    fig = plt.figure(figsize=(30, 5))

    # Physical domain extent
    extent = [0, 1, 0, 1]

    # Panel 1: Measured Output (Density Field)
    ax1 = plt.subplot(1, 6, 1)
    im1 = ax1.imshow(s_observed_2d, cmap="viridis", extent=extent, origin="lower")
    ax1.set_title("Measured Density Field s(x,y)", fontsize=12)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Panel 2: True Far Field (Polar)
    ax2 = plt.subplot(1, 6, 2, projection="polar")
    ax2.plot(theta, np.abs(u_true_np), "b-", linewidth=2, label="True")
    ax2.set_title("True Far Field u(θ)", fontsize=12, pad=20)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    # Panel 3: Predicted Far Field (Polar) with uncertainty
    ax3 = plt.subplot(1, 6, 3, projection="polar")
    u_mean_abs = np.abs(u_mean_np)
    ax3.plot(theta, u_mean_abs, "r-", linewidth=2, label="Mean Prediction")

    # Add uncertainty bands if probabilistic
    if hasattr(model, "sample_posterior"):
        # For complex numbers, uncertainty in magnitude - ensure 1D arrays
        u_std_abs = (
            np.abs(u_std_np).flatten() if u_std_np.ndim > 1 else np.abs(u_std_np)
        )
        u_mean_abs_flat = u_mean_abs.flatten() if u_mean_abs.ndim > 1 else u_mean_abs
        theta_flat = theta.flatten() if theta.ndim > 1 else theta

        u_upper = u_mean_abs_flat + u_std_abs
        u_lower = np.maximum(
            0, u_mean_abs_flat - u_std_abs
        )  # Ensure non-negative for magnitude
        ax3.fill_between(
            theta_flat, u_lower, u_upper, alpha=0.3, color="red", label="±1 std"
        )

    ax3.set_title("Predicted Far Field û(θ)", fontsize=12, pad=20)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    # Panel 4: Re-simulated Output (Mean)
    ax4 = plt.subplot(1, 6, 4)
    im4 = ax4.imshow(s_resim_mean_2d, cmap="viridis", extent=extent, origin="lower")
    title = "Re-simulated Density Field ŝ(x,y)"
    if hasattr(model, "sample_posterior"):
        title += f" (Mean of {n_realizations})"
    ax4.set_title(title, fontsize=12)
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # Panel 5: Error Field
    ax5 = plt.subplot(1, 6, 5)
    im5 = ax5.imshow(error_2d, cmap="Reds", extent=extent, origin="lower")
    ax5.set_title(
        f"Re-simulation Error |s - ŝ|\nMSE: {mse_resim:.6f}, MAE: {mae_resim:.6f}",
        fontsize=12,
    )
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    # Panel 6: Uncertainty in Re-simulation (only for probabilistic models)
    ax6 = plt.subplot(1, 6, 6)
    if hasattr(model, "sample_posterior") and np.any(s_resim_std_2d > 0):
        im6 = ax6.imshow(s_resim_std_2d, cmap="Oranges", extent=extent, origin="lower")
        ax6.set_title(f"Re-simulation Uncertainty σ(ŝ)", fontsize=12)
        ax6.set_xlabel("x")
        ax6.set_ylabel("y")
        plt.colorbar(im6, ax=ax6, fraction=0.046)

        # Add statistics
        mean_uncertainty = np.mean(s_resim_std_2d)
        max_uncertainty = np.max(s_resim_std_2d)
        ax6.text(
            0.02,
            0.98,
            f"Mean σ: {mean_uncertainty:.4f}\nMax σ: {max_uncertainty:.4f}",
            transform=ax6.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    else:
        ax6.text(
            0.5,
            0.5,
            "Deterministic Model\n(No Uncertainty)",
            ha="center",
            va="center",
            transform=ax6.transAxes,
            fontsize=14,
        )
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.set_xlabel("x")
        ax6.set_ylabel("y")
        ax6.set_title("Model Uncertainty", fontsize=12)

    # Add model type info to main title
    model_type = "Probabilistic" if is_probabilistic else "Deterministic"
    fig.suptitle(
        f"{model_name} ({model_type}) - Wave Scattering Sample {sample_idx}",
        fontsize=16,
    )

    plt.tight_layout()

    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{model_name}_probabilistic_sample_{sample_idx}.png"
        )
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
    n_realizations=10,
    save_dir=None,
):
    """Plot multiple random samples from the test set."""

    # Select random samples
    test_indices = random.sample(
        range(len(test_dataset)), min(n_samples, len(test_dataset))
    )

    for idx in test_indices:
        sample = test_dataset[idx]
        plot_wave_scattering_probabilistic_sample(
            model,
            evaluate_fn,
            input_function_encoder,
            output_function_encoder,
            forward_model,
            sample,
            idx,
            model_name,
            n_realizations,
            save_dir,
        )


def plot_model_results(
    model_name,
    log_dir,
    results_dir,
    test_dataset,
    dataset_info,
    n_samples=3,
    n_realizations=10,
):
    """Plot results for a single model."""

    model_log_dir = os.path.join(log_dir, model_name, "seed_1")

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
        n_realizations=n_realizations,
        save_dir=results_dir,
    )


# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Plot Wave Scattering probabilistic results."
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="/workspaces/b2b-operator-inverse/logs",
    help="Base log directory",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="results/wave_scattering_probabilistic_plots",
    help="Results directory for saving plots",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=5,
    help="Number of random samples to plot per model",
)
parser.add_argument(
    "--n_realizations",
    type=int,
    default=10,
    help="Number of posterior realizations to sample",
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
dataset = "wave_scattering"

# Construct paths
log_dir = os.path.join(args.log_dir, dataset)
results_dir = args.results_dir

model_name = args.model

# Load dataset using the specified model's parameters
temp_log_dir = os.path.join(log_dir, model_name, f"seed_{args.seed}")
temp_params = torch.load(os.path.join(temp_log_dir, "params.pth"), weights_only=False)

test_dataset, dataset_info = load_dataset(
    temp_params.dataset, temp_params, device, split="test", return_info=True
)

# Plot results for the specified model
plot_model_results(
    model_name=model_name,
    log_dir=log_dir,
    results_dir=results_dir,
    test_dataset=test_dataset,
    dataset_info=dataset_info,
    n_samples=args.n_samples,
    n_realizations=args.n_realizations,
)

print(
    f"SUCCESS: Plotted {model_name} probabilistic results, {args.n_samples} samples with {args.n_realizations} realizations each"
)
