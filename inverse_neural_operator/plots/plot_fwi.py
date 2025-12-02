import os
import argparse
import json
import random
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import torch

from b2b.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)

from data.load_dataset import load_dataset
from models.load_model import load_models
from plots.plot_utils import find_best_worst_samples

device = "cpu"


# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot results.")
parser.add_argument(
    "--model", type=str, required=True, help="Model name to plot results for"
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="/store/at46867/b2b_operator_inverse",
    help="Complete path to model directory (e.g., /path/to/logs/dataset/model/seed_1)",
)
parser.add_argument(
    "--results_dir", type=str, default="results/fwi/variational_autoencoder"
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)

args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# log_dir is now the complete path to the model directory
log_dir = args.log_dir
model_name = args.model
results_dir = args.results_dir

# Check if model directory exists
if not os.path.exists(os.path.join(log_dir, "params.pth")):
    print(f"✗ Model not found at {log_dir}")
    exit(1)

print(f"Loading model parameters and dataset...")
# Load params
params = torch.load(os.path.join(log_dir, "params.pth"), weights_only=False)

# Load dataset
test_dataset, dataset_info = load_dataset(
    params.dataset, params, device, split="test", return_info=True
)
print(f"✓ Loaded {len(test_dataset)} test samples")

# Load normalization statistics
stats_path = os.path.join(os.path.dirname(__file__), "../data/fwi_stats.json")
with open(stats_path, "r") as f:
    stats = json.load(f)
vmin = stats["models_min"]
vmax = stats["models_max"]
print(f"✓ Loaded normalization stats: velocity range [{vmin:.2f}, {vmax:.2f}]")

# # Load/create gradient for reconstruction
# from data.fwi_data import _create_linear_gradient
# gradient = _create_linear_gradient()
# gradient_flat = gradient.flatten()
# print(f"✓ Created gradient for reconstruction: [{gradient.min():.2f}, {gradient.max():.2f}]")

# Create results directory
os.makedirs(results_dir, exist_ok=True)

print(f"Loading models and generating plots...")

# Extract components from model directory path
path_parts = Path(log_dir).parts
seed = int(path_parts[-1].replace("seed_", ""))
model_name_from_path = path_parts[-2]
dataset_name = path_parts[-3]
models_idx = path_parts.index("models")
base_dir = str(Path(*path_parts[:models_idx]))

# Load models using new signature
input_function_encoder, output_function_encoder, model, evaluate_fn = load_models(
    base_dir=base_dir,
    dataset=dataset_name,
    model_name=model_name,
    seed=seed,
    device=device,
)


def plot_fwi_sample(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    sample,
    sample_idx,
    model_name,
    vmin,
    vmax,
    save_dir=None,
):
    """
    Plot a single FWI sample with 5-panel layout:
    1. True velocity model (input)
    2. Predicted velocity model
    3. Measured seismic transform (output)
    4. Re-simulated seismic transform
    5. Error between measured and re-simulated seismic transforms
    """
    model.eval()
    forward_model.eval()

    X, u_true, Y, s_observed = sample

    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_observed = s_observed.to(device)

    # Get model prediction for input velocity model
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

    # Re-simulate using forward model (required)
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

    # Denormalize velocity fields from [-1, 1] to physical units (m/s)
    # Residual normalization range used during training: [vmin - 900, vmax - 100]
    residual_min = vmin - 900.0
    residual_max = vmax - 100.0

    # Denormalize to velocity units (no gradient re-addition needed)
    u_true_velocity = ((u_true_np + 1) / 2) * (
        residual_max - residual_min
    ) + residual_min
    u_pred_velocity = ((u_pred_np + 1) / 2) * (
        residual_max - residual_min
    ) + residual_min

    # Reshape velocity models from flattened (1152,) to 2D (24, 48)
    u_true_2d = u_true_velocity.reshape(24, 48)
    u_pred_2d = u_pred_velocity.reshape(24, 48)

    # Reshape seismic transforms from flattened (30400,) to 2D (400, 76)
    s_observed_2d = s_observed_np.reshape(400, 76)

    # Re-simulation output
    s_resim_np = s_resim.squeeze(-1).cpu().numpy()
    s_resim_2d = s_resim_np.reshape(400, 76)

    # Create figure with proper spacing - use constrained_layout for automatic spacing
    fig = plt.figure(figsize=(28, 6), constrained_layout=True)

    # Create GridSpec with 3 groups: velocity pair, seismic pair, error
    # Each group has space for data plots + colorbar
    gs = fig.add_gridspec(
        1, 13, width_ratios=[4, 4, 0.3, 0.5, 4, 4, 0.3, 0.5, 4, 0.3, 0.5, 0.5, 0.5]
    )

    # Velocity model panels
    ax_vel_true = fig.add_subplot(gs[0, 0])
    ax_vel_pred = fig.add_subplot(gs[0, 1])
    ax_vel_cbar = fig.add_subplot(gs[0, 2])

    # Seismic transform panels
    ax_seismic_obs = fig.add_subplot(gs[0, 4])
    ax_seismic_resim = fig.add_subplot(gs[0, 5])
    ax_seismic_cbar = fig.add_subplot(gs[0, 6])

    # Error panel
    ax_error = fig.add_subplot(gs[0, 8])
    ax_error_cbar = fig.add_subplot(gs[0, 9])

    # Panel 1 & 2: Velocity Models with fixed absolute color scale
    norm_velocity = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    im_vel_true = ax_vel_true.imshow(
        u_true_2d,
        cmap="magma_r",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        aspect="equal",
    )
    ax_vel_true.set_title("True Velocity Model u(x,y)", fontsize=13, fontweight="bold")
    ax_vel_true.set_xlabel("x", fontsize=11)
    ax_vel_true.set_ylabel("y", fontsize=11)

    im_vel_pred = ax_vel_pred.imshow(
        u_pred_2d,
        cmap="magma_r",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        aspect="equal",
    )
    ax_vel_pred.set_title(
        "Predicted Velocity Model û(x,y)", fontsize=13, fontweight="bold"
    )
    ax_vel_pred.set_xlabel("x", fontsize=11)
    ax_vel_pred.set_ylabel("y", fontsize=11)

    # Add colorbar for velocity panels in dedicated axis
    scalar_mappable = mpl.cm.ScalarMappable(norm=norm_velocity, cmap="magma_r")
    scalar_mappable.set_array([])
    cbar1 = fig.colorbar(scalar_mappable, cax=ax_vel_cbar)
    cbar1.set_label("Velocity (m/s)", fontsize=10)
    cbar1.ax.invert_yaxis()

    # Panel 3 & 4: Seismic Transforms with shared scale
    seismic_vmin = min(s_observed_2d.min(), s_resim_2d.min())
    seismic_vmax = max(s_observed_2d.max(), s_resim_2d.max())

    # Panel 3: Measured Seismic Transform
    im3 = ax_seismic_obs.imshow(
        s_observed_2d,
        cmap="turbo",
        aspect="auto",
        origin="lower",
        vmin=seismic_vmin,
        vmax=seismic_vmax,
    )
    ax_seismic_obs.set_title(
        "Measured Seismic Transform s(f,t)", fontsize=13, fontweight="bold"
    )
    ax_seismic_obs.set_xlabel("Frequency", fontsize=11)
    ax_seismic_obs.set_ylabel("Time", fontsize=11)

    # Panel 4: Re-simulated Seismic Transform
    im4 = ax_seismic_resim.imshow(
        s_resim_2d,
        cmap="turbo",
        aspect="auto",
        origin="lower",
        vmin=seismic_vmin,
        vmax=seismic_vmax,
    )
    ax_seismic_resim.set_title(
        "Re-simulated Seismic Transform ŝ(f,t)", fontsize=13, fontweight="bold"
    )
    ax_seismic_resim.set_xlabel("Frequency", fontsize=11)
    ax_seismic_resim.set_ylabel("Time", fontsize=11)

    # Add colorbar for seismic transforms in dedicated axis
    cbar2 = fig.colorbar(im4, cax=ax_seismic_cbar)
    cbar2.set_label("Amplitude", fontsize=10)

    # Panel 5: Error Field
    error_2d = np.abs(s_observed_2d - s_resim_2d)
    mse_resim = np.mean((s_observed_2d - s_resim_2d) ** 2)
    mae_resim = np.mean(np.abs(s_observed_2d - s_resim_2d))

    im5 = ax_error.imshow(error_2d, cmap="Reds", aspect="auto", origin="lower")
    ax_error.set_title(
        f"Re-simulation Error |s - ŝ|\nMSE: {mse_resim:.6f}, MAE: {mae_resim:.6f}",
        fontsize=13,
        fontweight="bold",
    )
    ax_error.set_xlabel("Frequency", fontsize=11)
    ax_error.set_ylabel("Time", fontsize=11)

    cbar3 = fig.colorbar(im5, cax=ax_error_cbar)
    cbar3.set_label("Error", fontsize=10)

    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_name}_sample_{sample_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    plt.close()


def plot_multiple_samples(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    test_dataset,
    model_name,
    vmin,
    vmax,
    n_samples=3,
    save_dir=None,
):
    """Plot multiple random samples from the test set."""
    import random

    # Select random samples
    test_indices = random.sample(
        range(len(test_dataset)), min(n_samples, len(test_dataset))
    )

    for idx in test_indices:
        sample = test_dataset[idx]
        plot_fwi_sample(
            model,
            evaluate_fn,
            input_function_encoder,
            output_function_encoder,
            forward_model,
            sample,
            idx,
            model_name,
            vmin,
            vmax,
            save_dir,
        )


# Load forward model for re-simulation (required)
from b2b.load_model import load_forward_model

shared_dir = os.path.join(base_dir, "models", dataset_name, "shared", f"seed_{seed}")
forward_model = load_forward_model(
    model_dir=shared_dir, forward_model_name="b2b_nonlinear", device=device
)
print("✓ Loaded forward model for re-simulation")

# Plot results
plot_multiple_samples(
    model=model,
    evaluate_fn=evaluate_fn,
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
    forward_model=forward_model,
    test_dataset=test_dataset,
    model_name=model_name,
    vmin=vmin,
    vmax=vmax,
    n_samples=10,
    save_dir=results_dir,
)

# Find and plot best/worst case samples
print(f"Finding best and worst case samples for {model_name}...")
best_idx, worst_idx, best_ssim, worst_ssim = find_best_worst_samples(
    model=model,
    evaluate_fn=evaluate_fn,
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
    forward_model=forward_model,
    test_dataset=test_dataset,
    device=device,
    metric="ssim",
    output_shape=dataset_info.get("output_spatial_dims", None),
)

# Plot best case
print(f"Plotting best case (SSIM: {best_ssim:.4f})...")
best_sample = test_dataset[best_idx]
plot_fwi_sample(
    model=model,
    evaluate_fn=evaluate_fn,
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
    forward_model=forward_model,
    sample=best_sample,
    sample_idx=f"best_{best_idx}",
    model_name=model_name,
    vmin=vmin,
    vmax=vmax,
    save_dir=results_dir,
)

# Plot worst case
print(f"Plotting worst case (SSIM: {worst_ssim:.4f})...")
worst_sample = test_dataset[worst_idx]
plot_fwi_sample(
    model=model,
    evaluate_fn=evaluate_fn,
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
    forward_model=forward_model,
    sample=worst_sample,
    sample_idx=f"worst_{worst_idx}",
    model_name=model_name,
    vmin=vmin,
    vmax=vmax,
    save_dir=results_dir,
)

print(f"✓ Generated FWI plots → {results_dir}")
