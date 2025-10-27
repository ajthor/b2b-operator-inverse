#!/usr/bin/env python
"""
Fixed plotting script for Elastic Plate results.
Creates plots with 4 subplots: forcing function, prediction, ground truth, and absolute error.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from scipy.interpolate import griddata

# Add path for imports
sys.path.insert(0, 'inverse_neural_operator')

from b2b.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)
from data.load_dataset import load_dataset
from models.load_model import load_models

device = "cpu"
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def create_circular_mask(x, y, center_x=0.5, center_y=0.5, radius=0.25):
    """Create a circular mask for the hole in the plate."""
    return (x - center_x)**2 + (y - center_y)**2 <= radius**2


def plot_displacement_field(coords, displacement, title, ax, colorbar_label='x-displacement',
                           add_colorbar=True, vmin=None, vmax=None, cax=None, scaling_order=None):
    """Plot displacement field with circular hole."""
    # Extract coordinates
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    # Create regular grid for interpolation
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Create dense grid for smooth visualization
    xi = np.linspace(x_min, x_max, 200)
    yi = np.linspace(y_min, y_max, 200)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate displacement values onto regular grid
    Zi = griddata((x_coords, y_coords), displacement, (Xi, Yi), method='cubic')

    # Scale data if scaling_order provided
    if scaling_order is not None:
        scale = 10 ** -scaling_order
        Zi *= scale
        if vmin is not None:
            vmin *= scale
        if vmax is not None:
            vmax *= scale

    # Create circular mask for the hole
    hole_mask = create_circular_mask(Xi, Yi)

    # Set hole region to NaN so it appears empty
    Zi[hole_mask] = np.nan

    # Plot displacement field with optional vmin/vmax
    im = ax.contourf(Xi, Yi, Zi, levels=100, cmap='jet', vmin=vmin, vmax=vmax)

    # Set color limits if provided
    if vmin is not None and vmax is not None:
        im.set_clim(vmin, vmax)

    if cax is not None:
        cbar = plt.colorbar(im, cax=cax, format='%.1f')
        cbar.set_label(colorbar_label, rotation=270, labelpad=20, fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.get_offset_text().set_size(16)
        if scaling_order is not None:
            cbar.ax.set_title('10^{%d}' % scaling_order, fontsize=16, pad=10)
    elif add_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.1f')
        cbar.set_label(colorbar_label, rotation=270, labelpad=20, fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.get_offset_text().set_size(16)
        if scaling_order is not None:
            cbar.ax.set_title('10^{%d}' % scaling_order, fontsize=16, pad=10)

    # Set equal aspect ratio and add axes
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_title(title, fontsize=18, pad=15)
    ax.tick_params(axis='both', which='major', labelsize=14)


def plot_forcing_function(force_coords, force_values, title, ax):
    """Plot the forcing function as a 1D plot."""
    # Extract y-coordinates and force values
    force_y = force_coords[:, 1]  # y-coordinates of force points
    force_x = force_values  # force magnitudes

    # Plot force as horizontal line plot
    ax.plot(force_x, force_y, 'r-', linewidth=2)

    # Set up force subplot
    ax.set_ylim(force_y.min(), force_y.max())
    ax.set_xlabel('Force value', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Invert x-axis so zero is at the right (force applied from right side)
    ax.invert_xaxis()

    # Add vertical line at zero force
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)



def plot_elastic_sample(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    sample,
    sample_idx,
    model_name,
    save_dir=None,
):
    """Plot Elastic Plate inverse problem: observed displacement and predicted vs true forcing."""
    model.eval()

    X, u_true, Y, s_observed = sample

    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_observed = s_observed.to(device)

    # Get model prediction for input (inverse problem: predict force from displacement)
    # For stochastic models, draw multiple samples and plot overlays
    sampled_predictions_np = None
    with torch.no_grad():
        point = (
            X.unsqueeze(0),
            u_true.unsqueeze(0),
            Y.unsqueeze(0),
            s_observed.unsqueeze(0),
        )

        if model_name == "variational_autoencoder":
            num_samples = 10
            # Compute beta coefficients from observed displacement
            beta, _ = output_function_encoder.compute_coefficients(point[2], point[3])
            # Sample z ~ N(0, I)
            z = model.sample_prior(num_samples, device=X.device)
            # Repeat beta along batch to match z samples
            beta_rep = beta.expand(num_samples, -1)
            # Invert to alpha for each sample and decode to u on X grid
            alpha_samples = model.inverse(beta_rep, z)
            X_rep = X.unsqueeze(0).repeat(num_samples, 1, 1)
            u_pred_samples = input_function_encoder(X_rep, alpha_samples)  # [S, N, 1]
            sampled_predictions_np = u_pred_samples.squeeze(-1).cpu().numpy()      # [S, N]
            # Use the mean as the representative single prediction
            u_pred = u_pred_samples.mean(dim=0)  # [N, 1]
        elif model_name in [
            "mixture_density_network",
            "inn_affine",
            "cinn_affine",
            "cinn_additive",
        ]:
            num_samples = 10
            # Compute beta coefficients from observed displacement
            beta, _ = output_function_encoder.compute_coefficients(point[2], point[3])  # [1, B]

            if model_name == "mixture_density_network":
                # MDN samples by calling inverse(beta) repeatedly
                alpha_list = []
                for _ in range(num_samples):
                    alpha_i = model.inverse(beta)  # [1, A]
                    alpha_list.append(alpha_i.squeeze(0))  # [A]
                alpha_samples = torch.stack(alpha_list, dim=0)  # [S, A]
            elif model_name == "inn_affine":
                # Build batch of z and beta to sample in one call
                input_size = model.coupling_layers[0].input_size
                z_size = input_size - model.output_size
                z = torch.randn(num_samples, z_size, device=X.device, dtype=beta.dtype)
                beta_rep = beta.expand(num_samples, -1)
                alpha_samples = model.inverse(beta=beta_rep, z=z)  # [S, A]
            else:  # cinn_affine or cinn_additive
                # Sample z with alpha dimensionality
                alpha_dim = model.coupling_layers[0].input_size
                z = torch.randn(num_samples, alpha_dim, device=X.device, dtype=beta.dtype)
                beta_rep = beta.expand(num_samples, -1)
                alpha_samples = model.inverse(z, beta_rep)  # [S, A]

            # Decode all alpha samples on X grid
            X_rep = X.unsqueeze(0).repeat(num_samples, 1, 1)  # [S, N, d]
            u_pred_samples = input_function_encoder(X_rep, alpha_samples)  # [S, N, 1]
            sampled_predictions_np = u_pred_samples.squeeze(-1).cpu().numpy()  # [S, N]
            # Mean prediction across samples
            u_pred = u_pred_samples.mean(dim=0)  # [N, 1]
        else:
            u_pred, _ = evaluate_fn(
                model, point, input_function_encoder, output_function_encoder
            )
            u_pred = u_pred.squeeze(0)

    # Convert to numpy
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    u_pred_np = u_pred.squeeze(-1).cpu().numpy()
    s_observed_np = s_observed.squeeze(-1).cpu().numpy()
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    # Calculate force prediction error
    force_mse = np.mean((u_true_np - u_pred_np) ** 2)
    force_mae = np.mean(np.abs(u_true_np - u_pred_np))

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 7))

    # Use GridSpec for better control - increased spacing
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8], wspace=0.4)

    # Subplot 1: Observed Displacement Field (what we measure)
    ax1 = fig.add_subplot(gs[0])

    # Compute scaling for displacement
    disp_vabs = np.max(np.abs(s_observed_np))
    disp_order = np.floor(np.log10(disp_vabs)) if disp_vabs > 0 else 0

    # Plot displacement field with circular hole
    plot_displacement_field(Y_np, s_observed_np, 'Observed Displacement Field', ax1,
                           colorbar_label='Displacement', add_colorbar=True,
                           scaling_order=disp_order)

    # Subplot 2: Forcing Function Comparison (what we want to predict)
    ax2 = fig.add_subplot(gs[1])

    # Extract y-coordinates for force points
    force_y = X_np[:, 1]

    # Plot true and predicted forces
    ax2.plot(u_true_np, force_y, 'b-', linewidth=3, label='True Force', alpha=0.8)

    if sampled_predictions_np is not None:
        # Plot individual sampled predictions (thin, transparent)
        for i in range(sampled_predictions_np.shape[0]):
            ax2.plot(
                sampled_predictions_np[i],
                force_y,
                color='r',
                linewidth=1,
                alpha=0.25,
                label='Samples' if i == 0 else None,
            )
        # Plot the mean prediction prominently
        ax2.plot(u_pred_np, force_y, 'r--', linewidth=3, label='Mean', alpha=0.9)
    else:
        ax2.plot(u_pred_np, force_y, 'r--', linewidth=3, label='Predicted Force', alpha=0.8)

    # Formatting
    ax2.set_ylim(force_y.min(), force_y.max())
    ax2.set_xlabel('Force Magnitude', fontsize=16)
    ax2.set_ylabel('Position along Boundary (y)', fontsize=16)
    ax2.set_title('Forcing Function Comparison', fontsize=18, pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.legend(fontsize=14, loc='best')

    # Add vertical line at zero
    ax2.axvline(x=0, color='k', linestyle=':', alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    if save_dir:
        # Create subdirectory for this model
        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        save_path = os.path.join(model_save_dir, f"sample_{sample_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    → Saved: {save_path}")

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot Elastic Plate results.")
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="Model name to plot. If not specified, plots all available models. "
             "Available models: b2b_linear, b2b_nonlinear, variational_autoencoder, "
             "inn_additive, cinn_additive, inn_affine, cinn_affine, mixture_density_network"
    )
    parser.add_argument("--log_dir", type=str, default="logs", help="Base log directory")
    parser.add_argument("--results_dir", type=str, default="results/elastic_plots", help="Results directory")
    parser.add_argument("--n_samples", type=int, default=3, help="Number of samples to plot")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = "elastic_plate"

    # List of all possible models that can be trained
    # These are the models supported by the training pipeline
    ALL_MODELS = [
        "b2b_linear",
        "b2b_nonlinear",
        "variational_autoencoder",
        "inn_additive",
        "cinn_additive",
        "inn_affine",
        "cinn_affine",
        "mixture_density_network"
    ]

    # Construct paths
    log_dir = os.path.join(args.log_dir, dataset)
    shared_dir = os.path.join(log_dir, "shared", f"seed_{args.seed}")

    # Determine which models to plot
    if args.model:
        # Single model specified
        models_to_plot = [args.model]
    else:
        # No model specified - find all available trained models
        models_to_plot = []
        if os.path.exists(log_dir):
            for model_name in ALL_MODELS:
                model_path = os.path.join(log_dir, model_name, f"seed_{args.seed}", "params.pth")
                if os.path.exists(model_path):
                    models_to_plot.append(model_name)

        if not models_to_plot:
            print(f"ERROR: No trained models found in {log_dir}")
            print("\nTo train models, run: ./run_all.sh")
            exit(1)

        print(f"Found {len(models_to_plot)} trained models: {', '.join(models_to_plot)}")

    # Load dataset once (use first model's params)
    first_model_path = os.path.join(log_dir, models_to_plot[0], f"seed_{args.seed}", "params.pth")
    temp_params = torch.load(first_model_path, weights_only=False)

    print("Loading dataset...")
    test_dataset, dataset_info = load_dataset(
        temp_params.dataset, temp_params, device, split="test", return_info=True
    )
    print(f"  Test samples: {len(test_dataset)}")


    # Generate random sample indices once for consistency across models
    test_indices = random.sample(
        range(len(test_dataset)), min(args.n_samples, len(test_dataset))
    )

    # Process each model
    successful_models = []
    failed_models = []

    for model_name in models_to_plot:
        print(f"\n{'='*50}")
        print(f"Processing model: {model_name}")
        print('='*50)

        model_log_dir = os.path.join(log_dir, model_name, f"seed_{args.seed}")
        params_path = os.path.join(model_log_dir, "params.pth")

        # Check if model exists
        if not os.path.exists(params_path):
            print(f"ERROR: Model '{model_name}' not found at {params_path}")
            failed_models.append(model_name)
            continue

        try:
            # Load model parameters
            params = torch.load(params_path, weights_only=False)

            # Load the model
            print(f"Loading {model_name}...")
            input_function_encoder, output_function_encoder, model, evaluate_fn = load_models(
                log_dir=model_log_dir,
                dataset_info=dataset_info,
                params=params,
                device=device,
            )

            # Plot samples for this model
            print(f"Plotting {args.n_samples} samples...")
            for i, idx in enumerate(test_indices):
                print(f"  Processing sample {i+1}/{args.n_samples} (index {idx})...")
                sample = test_dataset[idx]
                plot_elastic_sample(
                    model,
                    evaluate_fn,
                    input_function_encoder,
                    output_function_encoder,
                    sample,
                    idx,
                    model_name,
                    args.results_dir,
                )

            successful_models.append(model_name)
            print(f"✓ Successfully plotted {model_name}")

        except Exception as e:
            print(f"✗ Failed to plot {model_name}: {e}")
            failed_models.append(model_name)

    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    if successful_models:
        print(f"✓ Successfully plotted {len(successful_models)} model(s): {', '.join(successful_models)}")
    if failed_models:
        print(f"✗ Failed to plot {len(failed_models)} model(s): {', '.join(failed_models)}")
    print(f"\n  Results saved to: {args.results_dir}/")
    print(f"  Directory structure:")
    for model in successful_models:
        print(f"    └── {model}/")
        print(f"        └── sample_*.png")


if __name__ == "__main__":
    main()