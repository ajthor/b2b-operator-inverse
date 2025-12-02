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

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

# Add project root to path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.load_dataset import load_dataset
from plots.utils.model_utils import (
    evaluate_models_on_subset,
    load_all_models,
    select_models_and_sample,
)
from plots.utils.plot_utils import (
    setup_publication_style,
    display_name,
    get_model_color,
    find_params,
    load_forward_model,
)
from models.ifno import create_model as create_ifno_model, load as load_ifno_weights

DEVICE = "cpu"

PUBLICATION_DISPLAY_OVERRIDES = {
    "cinn_additive_probabilistic": "cINN-Add-Prob",
}


def publication_display_name(model_name: str) -> str:
    """Return display label with local overrides for publication plots."""
    return PUBLICATION_DISPLAY_OVERRIDES.get(model_name, display_name(model_name))


INVERSE_MODELS = (
    #   "linear",
    "linear_inverse",
    "nonlinear",
    "inn_affine",
    #   "inn_additive",
    "cinn_affine",
    "variational_autoencoder",
    "conditional_realnvp",
    "mixture_density_network",
    "cinn_additive",
    "cinn_additive_probabilistic",
    "cinn_affine_probabilistic",
)

SAMPLING_MODELS = {
    "variational_autoencoder",
    "mixture_density_network",
    "cinn_additive_probabilistic",
    "cinn_affine_probabilistic",
    "conditional_realnvp",
}

MAX_MODELS = 8  # Total number of models evaluated (including IFNO)
MAX_FORCE_MODELS = 8  # Number of model force curves shown (plus separate GT panel)
MAX_DISPLACEMENT_MODELS = 8  # Number of model displacement fields shown (plus GT)
N_SAMPLES = 8

GROUND_TRUTH_COLOR = "#CCCCCC"  # Gray - for ground truth lines


def create_circular_mask(x, y, center=(0.5, 0.5), radius=0.25):
    """Return True inside the plate's circular void."""
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2


def load_ifno_model(
    dataset_info, device="cpu", ifno_path="logs_ifno/elastic_plate/ifno_model.pth"
):
    """Load IFNO model for elastic plate problem."""
    if not os.path.exists(ifno_path):
        print(f"IFNO model not found at {ifno_path}")
        return None

    # Create IFNO model with dataset-specific configuration
    ifno_model = create_ifno_model(
        input_size=None,
        modes1=16,
        modes2=16,
        width=64,
        beta=2.0,
        n_layers=3,
        padding=20,
        vae_latent_dim=24,
        intermediate_dim=32,
        input_spatial_dims=dataset_info["input_spatial_dims"],
        output_spatial_dims=dataset_info["output_spatial_dims"],
        input_function_channels=dataset_info["input_function_channels"],
        output_function_channels=dataset_info["output_function_channels"],
        coordinate_dim=dataset_info["coordinate_dim"],
    ).to(device)

    # Load weights
    load_ifno_weights(ifno_model, ifno_path, device=device)
    ifno_model.eval()

    print(f"✓ Loaded IFNO model from {ifno_path}")
    return ifno_model


def _create_unified_figure():
    """Create figure with unified gridspec for elastic plate visualization.

    Returns:
        fig: Figure object
        gs: GridSpec object (to be used for colorbar later)
        axes_left: List of 9 axes for left grid (force plots)
        axes_right: List of 9 axes for right grid (displacement fields)
    """
    # Calculate figure size so subplot cells stay square despite colorbar column
    width_ratios = [1, 1, 1, 1, 1, 1, 0.05]
    height_ratios = [1, 1, 1]
    fig_width = 6.5
    plot_width_units = sum(width_ratios)  # include colorbar column for accurate scaling
    # Slightly reduce height to compensate for constrained layout padding squeezing width
    square_adjust = 0.95
    fig_height = square_adjust * fig_width * sum(height_ratios) / plot_width_units
    fig = plt.figure(figsize=(fig_width, fig_height), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=0.5 / 72.0, h_pad=0.5 / 72.0, hspace=0.0, wspace=0.0
    )

    # Single gridspec: 2 rows × 7 columns
    # Columns: [plot, plot, plot, plot, plot, plot, colorbar]
    # Equal ratios create square cells with correct figure aspect
    gs = fig.add_gridspec(
        3,
        7,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
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
    ax_right_parent.set_title("Elastic Plate Re-simulated Displacement Fields")

    # Create subplot axes for left grid (force plots)
    axes_left = []
    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            axes_left.append(ax)

    # Create subplot axes for right grid (displacement fields)
    axes_right = []
    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j + 3])
            axes_right.append(ax)

    return fig, gs, axes_left, axes_right


def _plot_force_curve(
    ax,
    force_y,
    force_mag_true,
    force_mag_samples=None,
    annotation=None,
    color="b",
    show_gt_overlay=True,
):
    """Plot a 1D forcing function along the boundary with optional ground truth overlay.

    Args:
        ax: Axis to plot on
        force_y: Y coordinates along boundary
        force_mag_true: Ground truth force magnitude values
        force_mag_samples: Array of predicted force samples or None for GT-only plot
        annotation: Optional text annotation
        color: Line color for prediction
        show_gt_overlay: If True, show gray dashed GT line (for model plots), if False use solid line (for GT-only plot)
    """
    # Plot ground truth
    if show_gt_overlay:
        ax.plot(
            force_mag_true,
            force_y,
            color=GROUND_TRUTH_COLOR,
            linewidth=0.5,
            linestyle="dashed",
        )
    else:
        ax.plot(force_mag_true, force_y, color="black", linewidth=0.5)

    # Plot prediction samples if provided
    if force_mag_samples is not None:
        if force_mag_samples.ndim == 1:
            ax.plot(force_mag_samples, force_y, color=color, linewidth=0.5)
        else:
            for sample in force_mag_samples:
                ax.plot(sample, force_y, color=color, alpha=0.6, linewidth=0.5)

    # Styling
    ax.set_ylim(force_y.min(), force_y.max())
    xlim = ax.get_xlim()
    ax.set_xticks(np.linspace(xlim[0], xlim[1], 4))
    ax.set_yticks(np.linspace(force_y.min(), force_y.max(), 4))
    ax.tick_params(labelbottom=False, labelleft=False, length=0, width=0.5)
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)

    if annotation:
        ax.text(
            0.05,
            0.95,
            annotation,
            transform=ax.transAxes,
            fontsize=5,
            color="white",
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="none"
            ),
        )


def _plot_displacement_field(
    ax, coords, displacement, cmap="jet", vmin=None, vmax=None, annotation=None
):
    """Plot a 2D displacement field with circular void masked."""
    # Create interpolation grid
    xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 150)
    yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 150)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((coords[:, 0], coords[:, 1]), displacement, (Xi, Yi), method="cubic")
    Zi[create_circular_mask(Xi, Yi)] = np.nan

    # Plot
    im = ax.contourf(Xi, Yi, Zi, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    ax.set_xticks([])
    ax.set_yticks([])

    if annotation:
        ax.text(
            0.05,
            0.95,
            annotation,
            transform=ax.transAxes,
            fontsize=5,
            color="white",
            va="top",
            ha="left",
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

    if model_name == "cinn_affine":
        alpha_dim = model.coupling_layers[0].input_size
        # Deterministic affine cINN: use zero latent for inverse mapping
        z_zero = torch.zeros(num_samples, alpha_dim, device=device, dtype=dtype)
        return model.inverse(z_zero, beta_rep)

    if model_name == "cinn_additive_probabilistic":
        # Use the model's posterior sampler for better-calibrated stochastic samples
        # beta has shape [1, beta_dim] here; sample_posterior returns [num_samples, 1, alpha_dim]
        samples = model.sample_posterior(beta, num_samples)
        return samples.squeeze(1)

    if model_name == "cinn_affine_probabilistic":
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
    ifno_model=None,
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
        ifno_model: Optional IFNO model
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

    batch = (
        X.unsqueeze(0),
        u_true.unsqueeze(0),
        Y.unsqueeze(0),
        s_observed.unsqueeze(0),
    )

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
                beta, _ = output_function_encoder.compute_coefficients(
                    batch[2], batch[3]
                )
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
                            alpha_single = alpha_samples[i : i + 1]
                            beta_resim = forward_model(alpha_single)
                            s_resim = output_function_encoder(batch[2], beta_resim)
                            s_resim_np = s_resim.squeeze(0).squeeze(-1).cpu().numpy()
                            output_samples.append(s_resim_np)
                else:
                    # Fallback to evaluate_fn if sampling fails
                    for i in range(n_samples_per_model):
                        u_pred, alpha_pred = evaluate_fn(
                            model,
                            batch,
                            input_function_encoder,
                            output_function_encoder,
                        )
                        u_pred_np = u_pred.squeeze(0).squeeze(-1).cpu().numpy()
                        input_samples.append(u_pred_np)

                        if forward_model is not None and alpha_pred is not None:
                            beta_resim = forward_model(alpha_pred)
                            s_resim = output_function_encoder(batch[2], beta_resim)
                            s_resim_np = s_resim.squeeze(0).squeeze(-1).cpu().numpy()
                            output_samples.append(s_resim_np)
            else:
                # For deterministic models, just call evaluate_fn once
                u_pred, alpha_pred = evaluate_fn(
                    model, batch, input_function_encoder, output_function_encoder
                )
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

    # Add IFNO predictions if model is provided
    if ifno_model is not None:
        with torch.no_grad():
            # IFNO inverse: s_observed -> u_pred
            s_input = torch.cat([batch[2], batch[3]], dim=-1)  # [Y, s_observed]
            result = ifno_model.inverse(s_input)

            # Handle tuple return for symmetric models
            if isinstance(result, tuple):
                pred_u_full, _ = result
            else:
                pred_u_full = result

            # Extract function values only (last channel)
            if pred_u_full.shape[-1] > batch[1].shape[-1]:
                pred_u = pred_u_full[..., -batch[1].shape[-1] :]
            else:
                pred_u = pred_u_full

            u_pred_np = pred_u.squeeze(0).squeeze(-1).cpu().numpy()

            # Repeat for consistent interface
            input_samples = [u_pred_np.copy() for _ in range(n_samples_per_model)]

            # For displacement: use forward model if available, otherwise use IFNO forward
            output_samples = []
            if forward_model is not None:
                # Get alpha coefficients from predicted u
                alpha_pred, _ = input_function_encoder.compute_coefficients(
                    batch[0], pred_u
                )
                beta_resim = forward_model(alpha_pred)
                s_resim = output_function_encoder(batch[2], beta_resim)
                s_resim_np = s_resim.squeeze(0).squeeze(-1).cpu().numpy()
                output_samples = [s_resim_np.copy() for _ in range(n_samples_per_model)]
            else:
                # Use IFNO forward pass
                u_input = torch.cat([batch[0], pred_u], dim=-1)
                result_fwd = ifno_model(u_input)
                if isinstance(result_fwd, tuple):
                    pred_s, _ = result_fwd
                else:
                    pred_s = result_fwd

                if pred_s.shape[-1] > batch[3].shape[-1]:
                    pred_s = pred_s[..., -batch[3].shape[-1] :]

                s_pred_np = pred_s.squeeze(0).squeeze(-1).cpu().numpy()
                output_samples = [s_pred_np.copy() for _ in range(n_samples_per_model)]

            predictions["ifno"] = {
                "inputs": np.stack(input_samples),
                "outputs": np.stack(output_samples),
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

    # Process predictions (IFNO is already in models_to_plot if it was loaded)
    processed_preds = {}

    for model_name in models_to_plot[:MAX_MODELS]:
        preds = predictions.get(model_name, {})
        u_samples = preds.get("inputs", np.empty((0,)))
        s_samples = preds.get("outputs", np.empty((0,)))

        if u_samples.size > 0 and s_samples.size > 0:
            # For force curves: keep all samples for probabilistic models, mean for deterministic
            if model_name in SAMPLING_MODELS and u_samples.shape[0] > 1:
                u_data = u_samples  # Keep all samples [n_samples, n_points]
            else:
                u_data = u_samples.mean(axis=0)  # Single prediction [n_points]

            # For displacement fields: always use mean
            s_mean = s_samples.mean(axis=0)

            processed_preds[model_name] = (u_data, s_mean)

    # Determine color limits based only on ground truth (like original script)
    # This shows predictions relative to the expected displacement range
    s_abs_max = np.max(np.abs(s_true))
    s_min = -s_abs_max
    s_max = s_abs_max

    # Plot models
    im_right = None

    # Left side: Force curves (top 5 models + ground truth)
    for idx, model_name in enumerate(models_to_plot[:MAX_FORCE_MODELS]):
        if model_name in processed_preds:
            u_data, s_mean = processed_preds[model_name]
            _plot_force_curve(
                axes_left[idx],
                force_y,
                u_true,
                u_data,
                annotation=publication_display_name(model_name),
                color=get_model_color(model_name, idx),
            )
        else:
            axes_left[idx].axis("off")

    _plot_force_curve(
        axes_left[-1],
        force_y,
        u_true,
        annotation="Ground Truth",
        show_gt_overlay=False,
    )

    # Right side: Displacement fields (top 5 models + ground truth)
    for idx, model_name in enumerate(models_to_plot[:MAX_DISPLACEMENT_MODELS]):
        if model_name in processed_preds:
            u_data, s_mean = processed_preds[model_name]
            im_right = _plot_displacement_field(
                axes_right[idx],
                y_2d,
                s_mean,
                vmin=s_min,
                vmax=s_max,
                annotation=publication_display_name(model_name),
            )
        else:
            axes_right[idx].axis("off")

    im_right = _plot_displacement_field(
        axes_right[-1],
        y_2d,
        s_true,
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
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--results_dir", type=str, default="results/elastic_plate")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument(
        "--num_random_plots",
        type=int,
        default=5,
        help=(
            "Number of random samples to plot when --sample_index is not provided "
            "(default: 5; each sample yields both PDF and PNG outputs)."
        ),
    )
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

    print("\nComputing inverse reconstruction error for all models...")
    per_model_mses, _ = evaluate_models_on_subset(
        test_dataset,
        models_dict,
        input_enc,
        output_enc,
        max_samples=len(test_dataset),
        device=DEVICE,
    )
    model_inverse_mse = {
        name: float(np.mean(mse_list)) if mse_list else float("inf")
        for name, mse_list in per_model_mses.items()
    }
    for name in sorted(model_inverse_mse, key=model_inverse_mse.get):
        print(f"  {publication_display_name(name)}: {model_inverse_mse[name]:.6e}")

    # Load IFNO model
    print("\nLoading IFNO model...")
    ifno_model = load_ifno_model(dataset_info, device=DEVICE)
    ifno_mse = None

    # Evaluate models and select best performers (get top MAX_MODELS to have room for IFNO)
    models_to_plot, sample_idx, per_model_losses = select_models_and_sample(
        test_dataset,
        models_dict,
        input_enc,
        output_enc,
        forward_model,
        max_models=MAX_MODELS,  # Get top MAX_MODELS models
        sample_index=args.sample_index,
        device=DEVICE,
        return_metrics=True,
    )
    resim_mse = {
        name: (
            float(np.mean(losses["pred_loss"])) if losses["pred_loss"] else float("inf")
        )
        for name, losses in per_model_losses.items()
    }
    print("\nRe-simulation MSE summary:")
    for name in sorted(resim_mse, key=resim_mse.get):
        val = resim_mse[name]
        display = "inf" if not np.isfinite(val) else f"{val:.6e}"
        print(f"  {publication_display_name(name)}: {display}")

    # Evaluate IFNO and insert it into the sorted list based on accuracy
    if ifno_model is not None:
        print("\nEvaluating IFNO performance...")

        # Evaluate IFNO on test dataset
        ifno_errors = []
        ifno_model.eval()
        with torch.no_grad():
            for sample in test_dataset:
                X, u_true, Y, s_observed = sample
                X = X.to(DEVICE).unsqueeze(0)
                u_true = u_true.to(DEVICE).unsqueeze(0)
                Y = Y.to(DEVICE).unsqueeze(0)
                s_observed = s_observed.to(DEVICE).unsqueeze(0)

                # IFNO inverse: s_observed -> u_pred
                s_input = torch.cat([Y, s_observed], dim=-1)
                result = ifno_model.inverse(s_input)

                if isinstance(result, tuple):
                    pred_u, _ = result
                else:
                    pred_u = result

                # Extract function values only
                if pred_u.shape[-1] > u_true.shape[-1]:
                    pred_u = pred_u[..., -u_true.shape[-1] :]

                # Compute MSE (to match other models' evaluation)
                error = ((pred_u - u_true) ** 2).mean()
                ifno_errors.append(error.item())

        ifno_mse = np.mean(ifno_errors)
        print(f"  IFNO MSE: {ifno_mse:.6e}")
    else:
        print("  IFNO model not available; skipping IFNO evaluation.")

    models_with_errors = [
        (name, resim_mse.get(name, float("inf"))) for name in resim_mse.keys()
    ]
    if ifno_mse is not None:
        models_with_errors.append(("ifno", ifno_mse))
    models_with_errors.sort(key=lambda x: x[1])
    ranking_title = (
        f"\nTop {MAX_MODELS} models by re-simulation MSE (including IFNO):"
        if ifno_mse is not None
        else f"\nTop {MAX_MODELS} models by re-simulation MSE:"
    )
    print(ranking_title)
    for name, error in models_with_errors[:MAX_MODELS]:
        print(f"  {publication_display_name(name)}: {error:.6e}")
    models_to_plot = [name for name, _ in models_with_errors[:MAX_MODELS]]

    num_plots = max(1, args.num_random_plots)
    if args.sample_index is not None:
        sample_indices = [args.sample_index]
    elif num_plots > 1:
        total_samples = len(test_dataset)
        if num_plots > total_samples:
            print(
                f"  Warning: Requested {num_plots} plots but only {total_samples} samples available. "
                f"Using {total_samples} unique samples instead."
            )
        k = min(num_plots, total_samples)
        sample_indices = random.sample(range(total_samples), k=k)
        sample_indices.sort()
        print(f"Randomly selected sample indices: {sample_indices}")
    else:
        sample_indices = [sample_idx]

    for idx in sample_indices:
        print(f"\nCollecting predictions for sample {idx}...")
        predictions, meta = collect_elastic_predictions(
            test_dataset[idx],
            models_to_plot,
            models_dict,
            input_enc,
            output_enc,
            forward_model,
            ifno_model=ifno_model,
            n_samples_per_model=N_SAMPLES,
            device=DEVICE,
        )

        print("Rendering figure...")
        plot_comparison(idx, models_to_plot, predictions, meta, args.results_dir)

    print(f"SUCCESS: Created publication figure(s) → {args.results_dir}")


if __name__ == "__main__":
    main()
