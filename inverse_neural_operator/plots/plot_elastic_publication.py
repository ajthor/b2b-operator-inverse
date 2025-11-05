"""
Publication-quality plotting script for Elastic Plate inverse problem results.

Creates publication-ready figures with:
- 6.5 inch width for journal publications
- 5-7pt sans serif fonts
- Professional styling and layout
- High-resolution output (300 DPI)
- 2x3 grid layout showing top 5 models + ground truth
- Left: Predicted boundary forcing functions (1D line plots)
- Right: Re-simulated displacement fields (2D contour plots with circular void)

To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_elastic_publication
"""

import argparse
import os
import random
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

# Add project root to path for module imports
sys.path.insert(0, "inverse_neural_operator")

from data.load_dataset import load_dataset
from plots.utils.model_utils import (
    load_all_models,
    select_models_and_sample,
)
from plots.utils.plot_utils import (
    setup_publication_style,
    display_name,
    find_params,
    load_forward_model,
)

DEVICE = "cpu"

INVERSE_MODELS = (
    "linear",
    "linear_inverse",
    "nonlinear",
    "inn_affine",
    "cinn_affine",
    "variational_autoencoder",
    "conditional_realnvp",
    "mixture_density_network",
)

SAMPLING_MODELS = {
    "variational_autoencoder",
    "mixture_density_network",
    "inn_affine",
    "cinn_affine",
    "cinn_additive",
}

MAX_MODELS = 6  # Number of models to plot on left (force curves)
MAX_DISPLACEMENT_MODELS = 5  # Number of model displacement fields (5 models + 1 ground truth)
N_SAMPLES = 8

MODEL_CMAP = mpl.cm.get_cmap("tab10")
GROUND_TRUTH_COLOR = "#CCCCCC"  # Gray - for ground truth lines


def create_circular_mask(x, y, center=(0.5, 0.5), radius=0.25):
    """Return True inside the plate's circular void."""
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2


def _create_unified_figure():
    """Create figure with unified gridspec for elastic plate visualization.

    Returns:
        fig: Figure object
        gs: GridSpec object (to be used for colorbar later)
        axes_left: List of 6 axes for left grid (force plots)
        axes_right: List of 6 axes for right grid (displacement fields)
    """
    # Calculate figure size to create square subplot cells
    # 2 rows × 6 columns (plus thin colorbar)
    # For square cells: width/6 = height/2, so width = 3*height
    # With 6.5 inch width standard, height = 6.5/3 ≈ 2.17
    fig = plt.figure(figsize=(6.5, 2.17), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=0.5 / 72.0, h_pad=0.5 / 72.0, hspace=0.0, wspace=0.0
    )

    # Single gridspec: 2 rows × 7 columns
    # Columns: [plot, plot, plot, plot, plot, plot, colorbar]
    # Equal ratios create square cells with correct figure aspect
    gs = fig.add_gridspec(
        2,
        7,
        width_ratios=[1, 1, 1, 1, 1, 1, 0.05],
        height_ratios=[1, 1],
        hspace=0.0,
        wspace=0.0,
        left=0,
        right=1,
        top=1,
        bottom=0,
    )

    # Create parent axes for shared labels
    ax_left_parent = fig.add_subplot(gs[:, 0:3], frameon=False)
    ax_left_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_left_parent.set_xlabel("Force Magnitude", labelpad=-8)
    ax_left_parent.set_ylabel("Boundary Position (y)", labelpad=-8)
    ax_left_parent.set_title("Predicted Boundary Forces")

    ax_right_parent = fig.add_subplot(gs[:, 3:6], frameon=False)
    ax_right_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_right_parent.set_xlabel(r"$x$", labelpad=-8)
    ax_right_parent.set_ylabel(r"$y$", labelpad=-8)
    ax_right_parent.set_title("Re-simulated Displacement Fields")

    # Create subplot axes for left grid (force plots)
    axes_left = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            axes_left.append(ax)

    # Create subplot axes for right grid (displacement fields)
    axes_right = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j + 3])
            axes_right.append(ax)

    return fig, gs, axes_left, axes_right


def _plot_force_curve(ax, force_y, force_mag_true, force_mag_pred=None, annotation=None, color="b", linewidth=1.5):
    """Plot a 1D forcing function along the boundary with ground truth.

    Args:
        ax: Axis to plot on
        force_y: Y coordinates along boundary
        force_mag_true: Ground truth force magnitude values
        force_mag_pred: Predicted force magnitude values (optional)
        annotation: Optional text annotation for upper-left corner
        color: Line color for prediction
        linewidth: Line width

    Returns:
        The prediction line object
    """
    # Plot ground truth first (gray dashed line)
    ax.plot(force_mag_true, force_y, color=GROUND_TRUTH_COLOR, linewidth=1.0, linestyle="dashed")

    # Plot prediction if provided (colored solid line)
    line = None
    if force_mag_pred is not None:
        line = ax.plot(force_mag_pred, force_y, color=color, linewidth=linewidth)[0]

    # Set y limits to match boundary coordinates [0, 1]
    ax.set_ylim(force_y.min(), force_y.max())

    # Clean minimal styling (no aspect - force plots fill their cells)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.axvline(x=0, color="k", linestyle=":", alpha=0.3, linewidth=0.5)

    # Add annotation if provided
    if annotation:
        ax.text(
            0.05,
            0.95,
            annotation,
            transform=ax.transAxes,
            fontsize=5,
            color="white",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="none"
            ),
        )

    return line


def _plot_displacement_field(
    ax, coords, displacement, cmap="jet", vmin=None, vmax=None, annotation=None
):
    """Plot a 2D displacement field with circular void masked.

    Args:
        ax: Axis to plot on
        coords: N×2 array of (x, y) coordinates
        displacement: N-length array of displacement values
        cmap: Colormap name
        vmin, vmax: Color scale limits
        annotation: Optional text annotation for upper-left corner

    Returns:
        The contour object
    """
    # Create interpolation grid
    xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 150)
    yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 150)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((coords[:, 0], coords[:, 1]), displacement, (Xi, Yi), method="cubic")

    # Mask circular void
    Zi[create_circular_mask(Xi, Yi)] = np.nan

    # Plot as contour
    im = ax.contourf(Xi, Yi, Zi, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)

    # Set data limits
    ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())

    # Don't set aspect - let gridspec sizing control the shape
    # Figure dimensions are calculated to create square cells
    ax.set_xticks([])
    ax.set_yticks([])

    # Add annotation if provided
    if annotation:
        ax.text(
            0.05,
            0.95,
            annotation,
            transform=ax.transAxes,
            fontsize=5,
            color="white",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="none"
            ),
        )

    return im


def sample_alpha(model_name, model, beta, device, dtype, num_samples):
    """Sample alpha coefficients from model-specific inverse mappings."""
    beta_rep = beta.expand(num_samples, -1)

    if model_name == "variational_autoencoder":
        z = model.sample_prior(num_samples, device=device)
        return model.inverse(beta_rep, z)

    if model_name == "mixture_density_network":
        return torch.stack([model.inverse(beta).squeeze(0) for _ in range(num_samples)])

    if model_name == "inn_affine":
        z_size = model.coupling_layers[0].input_size - model.output_size
        z = torch.randn(num_samples, z_size, device=device, dtype=dtype)
        return model.inverse(beta=beta_rep, z=z)

    if model_name in {"cinn_affine", "cinn_additive"}:
        alpha_dim = model.coupling_layers[0].input_size
        z = torch.randn(num_samples, alpha_dim, device=device, dtype=dtype)
        return model.inverse(z, beta_rep)

    return None


def collect_elastic_predictions(
    sample,
    models_to_plot,
    models_dict,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    n_samples_per_model=8,
    device="cpu",
):
    """Collect predictions specifically for elastic plate problem.

    Args:
        sample: Single test sample (X, u_true, Y, s_observed)
        models_to_plot: Model names to collect predictions from
        models_dict: Dictionary of loaded models
        input_function_encoder: Input encoder
        output_function_encoder: Output encoder
        forward_model: Forward model for re-simulation
        n_samples_per_model: Number of predictions per model
        device: Device for computation

    Returns:
        predictions: {model_name: {"inputs": array, "outputs": array}}
        meta: {"x": array, "y": array, "u_true": array, "s_true": array}
    """
    X, u_true, Y, s_observed = sample
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_observed = s_observed.to(device)

    batch = (X.unsqueeze(0), u_true.unsqueeze(0), Y.unsqueeze(0), s_observed.unsqueeze(0))

    predictions = {}
    meta = {
        "x": X.cpu().numpy(),  # Keep as [N, 2]
        "y": Y.cpu().numpy(),  # Keep as [M, 2]
        "u_true": u_true.squeeze(-1).cpu().numpy(),  # [N]
        "s_true": s_observed.squeeze(-1).cpu().numpy(),  # [M]
    }

    for model_name in models_to_plot:
        if model_name not in models_dict:
            print(f"  WARNING: {model_name} not in models_dict")
            continue

        model, evaluate_fn = models_dict[model_name]
        model.eval()

        input_samples = []
        output_samples = []

        with torch.no_grad():
            # Check if this is a sampling model
            if model_name in SAMPLING_MODELS:
                # For sampling models, use latent space sampling
                beta, _ = output_function_encoder.compute_coefficients(batch[2], batch[3])
                alpha_samples = sample_alpha(
                    model_name, model, beta, device, beta.dtype, n_samples_per_model
                )

                if alpha_samples is not None:
                    # Reconstruct forces from sampled alpha coefficients
                    X_rep = batch[0].repeat(alpha_samples.size(0), 1, 1)
                    u_samples = input_function_encoder(X_rep, alpha_samples)

                    # Store all samples
                    for i in range(n_samples_per_model):
                        u_pred_np = u_samples[i].squeeze(-1).cpu().numpy()
                        input_samples.append(u_pred_np)

                        # Re-simulate if forward model available
                        if forward_model is not None:
                            alpha_single = alpha_samples[i:i+1]
                            beta_resim = forward_model(alpha_single)
                            s_resim = output_function_encoder(batch[2], beta_resim)
                            s_resim_np = s_resim.squeeze(0).squeeze(-1).cpu().numpy()
                            output_samples.append(s_resim_np)
                else:
                    # Fallback to evaluate_fn if sampling fails
                    for i in range(n_samples_per_model):
                        u_pred, alpha_pred = evaluate_fn(model, batch, input_function_encoder, output_function_encoder)
                        u_pred_np = u_pred.squeeze(0).squeeze(-1).cpu().numpy()
                        input_samples.append(u_pred_np)

                        if forward_model is not None and alpha_pred is not None:
                            beta_resim = forward_model(alpha_pred)
                            s_resim = output_function_encoder(batch[2], beta_resim)
                            s_resim_np = s_resim.squeeze(0).squeeze(-1).cpu().numpy()
                            output_samples.append(s_resim_np)
            else:
                # For deterministic models, just call evaluate_fn once
                u_pred, alpha_pred = evaluate_fn(model, batch, input_function_encoder, output_function_encoder)
                u_pred_np = u_pred.squeeze(0).squeeze(-1).cpu().numpy()

                # Repeat the same prediction n_samples_per_model times for consistent interface
                for i in range(n_samples_per_model):
                    input_samples.append(u_pred_np.copy())

                # Re-simulate if forward model available (only once)
                if forward_model is not None and alpha_pred is not None:
                    beta_resim = forward_model(alpha_pred)
                    s_resim = output_function_encoder(batch[2], beta_resim)
                    s_resim_np = s_resim.squeeze(0).squeeze(-1).cpu().numpy()

                    # Repeat the same re-simulation n_samples_per_model times
                    for i in range(n_samples_per_model):
                        output_samples.append(s_resim_np.copy())

        predictions[model_name] = {
            "inputs": np.stack(input_samples) if input_samples else np.empty((0,)),
            "outputs": np.stack(output_samples) if output_samples else np.empty((0,)),
        }

    return predictions, meta


def plot_comparison(sample_idx, models_to_plot, predictions, meta, save_dir):
    """Create publication-quality comparison plot for elastic plate.

    Args:
        sample_idx: Index of the sample being plotted
        models_to_plot: List of model names to plot (up to 5)
        predictions: Dictionary of predictions for each model
        meta: Dictionary with ground truth data
        save_dir: Directory to save output files
    """
    # Extract data from meta (now properly structured)
    x_2d = meta["x"]  # Boundary coordinates [N, 2]
    u_true = meta["u_true"]  # Boundary forces [N]
    y_2d = meta["y"]  # Domain coordinates [M, 2]
    s_true = meta["s_true"]  # Displacement field [M]

    # Extract y-coordinates along boundary for force plot
    force_y = x_2d[:, 1]

    # Create figure
    fig, gs, axes_left, axes_right = _create_unified_figure()

    # Process predictions
    processed_preds = {}
    for model_name in models_to_plot[:MAX_MODELS]:
        preds = predictions.get(model_name, {})
        u_samples = preds.get("inputs", np.empty((0,)))
        s_samples = preds.get("outputs", np.empty((0,)))

        if u_samples.size > 0 and s_samples.size > 0:
            # Average over samples
            u_mean = u_samples.mean(axis=0)
            s_mean = s_samples.mean(axis=0)

            processed_preds[model_name] = (u_mean, s_mean)

    # Determine color limits based only on ground truth (like original script)
    # This shows predictions relative to the expected displacement range
    s_abs_max = np.max(np.abs(s_true))
    s_min = -s_abs_max
    s_max = s_abs_max

    # Plot models
    im_right = None

    # Left side: Plot force curves for top 6 models (each with ground truth overlay)
    for idx, model_name in enumerate(models_to_plot[:MAX_MODELS]):
        ax = axes_left[idx]
        if model_name in processed_preds:
            u_mean, s_mean = processed_preds[model_name]
            color = MODEL_CMAP(idx % MODEL_CMAP.N)

            # Force curve with ground truth overlay
            _plot_force_curve(
                ax, force_y, u_true, u_mean, annotation=display_name(model_name), color=color, linewidth=1.5
            )
        else:
            # Turn off empty axes
            ax.axis("off")

    # Right side: Plot displacement fields for top 5 models
    for idx, model_name in enumerate(models_to_plot[:MAX_DISPLACEMENT_MODELS]):
        ax = axes_right[idx]
        if model_name in processed_preds:
            u_mean, s_mean = processed_preds[model_name]

            # Displacement field
            im_right = _plot_displacement_field(
                ax,
                y_2d,
                s_mean,
                cmap="jet",
                vmin=s_min,
                vmax=s_max,
                annotation=display_name(model_name),
            )
        else:
            # Turn off empty axes
            ax.axis("off")

    # 6th position on right: ground truth displacement field
    ax = axes_right[5]
    im_right = _plot_displacement_field(
        ax,
        y_2d,
        s_true,
        cmap="jet",
        vmin=s_min,
        vmax=s_max,
        annotation="Ground Truth",
    )

    # Add colorbar for displacement fields
    if im_right is not None:
        cax_right = fig.add_subplot(gs[:, 6])
        cbar_right = fig.colorbar(im_right, cax=cax_right, use_gridspec=True)
        cbar_right.set_label("Displacement", rotation=270, labelpad=10)
        cbar_right.ax.tick_params(labelsize=5)

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, f"elastic_plate_sample_{sample_idx}_publication")
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {base}.pdf")
    print(f"✓ Saved: {base}.png")

    plt.close(fig)


def main():
    setup_publication_style(figsize=(6.5, 3.5))

    parser = argparse.ArgumentParser(
        description="Create publication-quality Elastic Plate plots."
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs"
    )
    parser.add_argument("--results_dir", type=str, default="results/elastic_plate")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample_index", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = "elastic_plate"
    log_dir = os.path.join(args.log_dir, dataset)

    if not os.path.exists(log_dir):
        print(f"✗ Log directory not found: {log_dir}")
        exit(1)

    # Load dataset
    print("Loading dataset...")
    params = find_params(log_dir, INVERSE_MODELS, args.seed)
    if params is None:
        print(f"✗ No trained models found in {log_dir}")
        exit(1)

    test_dataset, dataset_info = load_dataset(
        params.dataset, params, DEVICE, split="test", return_info=True
    )
    print(f"✓ Loaded {len(test_dataset)} test samples")

    # Load models
    print("\nLoading models...")
    models_dict, input_enc, output_enc = load_all_models(
        log_dir, dataset_info, INVERSE_MODELS, args.seed, DEVICE
    )

    if not models_dict:
        print("ERROR: no models loaded")
        raise SystemExit(1)

    forward_model = load_forward_model(log_dir, args.seed, device=DEVICE)

    # Evaluate models and select best performers
    models_to_plot, sample_idx = select_models_and_sample(
        test_dataset,
        models_dict,
        input_enc,
        output_enc,
        max_models=MAX_MODELS,
        sample_index=args.sample_index,
        device=DEVICE,
    )

    print("Collecting predictions...")
    predictions, meta = collect_elastic_predictions(
        test_dataset[sample_idx],
        models_to_plot,
        models_dict,
        input_enc,
        output_enc,
        forward_model,
        n_samples_per_model=N_SAMPLES,
        device=DEVICE,
    )

    print("Rendering figure...")
    plot_comparison(sample_idx, models_to_plot, predictions, meta, args.results_dir)
    print(f"SUCCESS: Created publication figure → {args.results_dir}")


if __name__ == "__main__":
    main()
