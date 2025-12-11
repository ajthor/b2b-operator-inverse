"""
 Publication-quality plotting script for Burgers 1D inverse problem results.

Creates publication-ready figures with:
- 6.5 inch width for journal publications
- 5-7pt sans serif fonts
- Professional styling and layout
- High-resolution output (300 DPI)
- 2x3 grid layout showing top 5 models + ground truth
- Left: Predicted initial conditions u_0(x) (1D line plots)
- Right: Re-simulated final solutions u(x, t=1) (1D line plots)

To run: cd /workspaces/b2b-operator-inverse && python inverse_neural_operator/plots/plot_burgers_nine_publication.py
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path for module imports
sys.path.insert(0, "inverse_neural_operator")

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

DEVICE = "cpu"

PUBLICATION_DISPLAY_OVERRIDES = {
    "cinn_additive_probabilistic": "cINN-Add-Prob",
}


def publication_display_name(model_name: str) -> str:
    """Return display label with local overrides for publication plots."""
    return PUBLICATION_DISPLAY_OVERRIDES.get(model_name, display_name(model_name))


INVERSE_MODELS = (
    "linear",
    "linear_inverse",
    "nonlinear",
    "inn_affine",
    "inn_additive",
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

MAX_MODELS = 8  # Total number of models evaluated
MAX_INPUT_MODELS = 8  # Number of model input curves shown
MAX_OUTPUT_MODELS = 8  # Number of model output curves shown
N_SAMPLES = 8

GROUND_TRUTH_COLOR = "#CCCCCC"  # Gray - for ground truth lines


def _create_unified_figure():
    """Create figure with unified gridspec for Burgers 1D visualization.

    Returns:
        fig: Figure object
        gs: GridSpec object
        axes_left: List of axes for left grid (initial condition plots)
        axes_right: List of axes for right grid (final solution plots)
    """
    # Calculate figure size
    width_ratios = [1, 1, 1, 1, 1, 1]
    height_ratios = [1, 1, 1]
    fig_width = 6.5

    # Adjust aspect ratio
    square_adjust = 0.8
    fig_height = square_adjust * fig_width * sum(height_ratios) / sum(width_ratios)

    fig = plt.figure(figsize=(fig_width, fig_height), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=0.5 / 72.0, h_pad=0.5 / 72.0, hspace=0.0, wspace=0.0
    )

    # Single gridspec: 3 rows × 6 columns
    # Columns: [plot, plot, plot, plot, plot, plot] (left 3 for input, right 3 for output)
    gs = fig.add_gridspec(
        3,
        6,
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
    ax_left_parent.set_xlabel(r"$x$", labelpad=-8)
    ax_left_parent.set_ylabel("Initial Condition $u_0(x)$", labelpad=-8)
    ax_left_parent.set_title("Predicted Initial Condition (Normalized)")

    ax_right_parent = fig.add_subplot(gs[:, 3:6], frameon=False)
    ax_right_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_right_parent.set_xlabel(r"$x$", labelpad=-8)
    ax_right_parent.set_ylabel("Final Solution $u(x, t=1)$", labelpad=-8)
    ax_right_parent.set_title("Re-simulated Final Solution (Normalized)")

    # Create subplot axes for left grid (input plots)
    axes_left = []
    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            axes_left.append(ax)

    # Create subplot axes for right grid (output plots)
    axes_right = []
    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j + 3])
            axes_right.append(ax)

    return fig, gs, axes_left, axes_right


def _plot_1d_curve(
    ax, x_grid, y_true, y_samples=None, annotation=None, color="b", show_gt_overlay=True
):
    """Plot a 1D curve with optional ground truth overlay.

    Args:
        ax: Axis to plot on
        x_grid: X coordinates
        y_true: Ground truth values
        y_samples: Array of predicted samples or None for GT-only plot
        annotation: Optional text annotation
        color: Line color for prediction
        show_gt_overlay: If True, show gray dashed GT line
    """
    # Ensure x_grid and y_true are 1D arrays
    x_grid = np.squeeze(x_grid)
    y_true = np.squeeze(y_true)

    # Plot ground truth
    if show_gt_overlay:
        ax.plot(
            x_grid, y_true, color=GROUND_TRUTH_COLOR, linewidth=0.5, linestyle="dashed"
        )
    else:
        ax.plot(x_grid, y_true, color="black", linewidth=0.5)

    # Plot prediction samples if provided
    if y_samples is not None:
        if y_samples.ndim == 1:
            ax.plot(x_grid, y_samples, color=color, linewidth=0.5)
        else:
            for sample in y_samples:
                ax.plot(
                    x_grid, np.squeeze(sample), color=color, alpha=0.6, linewidth=0.5
                )

    # Styling
    ax.set_xlim(x_grid.min(), x_grid.max())

    # Set reasonable y-limits based on data range
    y_min, y_max = y_true.min(), y_true.max()
    if y_samples is not None:
        y_min = min(y_min, y_samples.min())
        y_max = max(y_max, y_samples.max())

    # Add some padding
    padding = (y_max - y_min) * 0.1
    if padding == 0:
        padding = 1.0
    ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_xticks([])
    ax.set_yticks([])
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

    if model_name == "cinn_additive":
        alpha_dim = model.coupling_layers[0].input_size
        # Deterministic additive cINN: use zero latent for inverse mapping
        z_zero = torch.zeros(num_samples, alpha_dim, device=device, dtype=dtype)
        return model.inverse(z_zero, beta_rep)

    if model_name == "cinn_additive_probabilistic":
        # Use the model's posterior sampler for better-calibrated stochastic samples
        samples = model.sample_posterior(beta, num_samples)
        return samples.squeeze(1)

    if model_name == "cinn_affine_probabilistic":
        alpha_dim = model.coupling_layers[0].input_size
        z = torch.randn(num_samples, alpha_dim, device=device, dtype=dtype)
        return model.inverse(z, beta_rep)

    return None


def collect_burgers_predictions(
    sample,
    models_to_plot,
    models_dict,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    n_samples_per_model=8,
    device="cpu",
):
    """Collect predictions specifically for Burgers 1D problem.

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
    # Note: In Burgers, u is input (initial condition), s is output (final solution)
    # X is spatial coord for u, Y is spatial coord for s

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
        "x": X.cpu().numpy(),  # [N, 1]
        "y": Y.cpu().numpy(),  # [M, 1]
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
                    # Reconstruct initial condition from sampled alpha coefficients
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

    return predictions, meta


def plot_comparison(sample_idx, models_to_plot, predictions, meta, save_dir):
    """Create publication-quality comparison plot for Burgers 1D.

    Args:
        sample_idx: Index of the sample being plotted
        models_to_plot: List of model names to plot (up to 5)
        predictions: Dictionary of predictions for each model
        meta: Dictionary with ground truth data
        save_dir: Directory to save output files
    """
    # Extract data from meta
    x_coords = meta["x"]  # [N, 1]
    u_true = meta["u_true"]  # Initial Condition [N]
    y_coords = meta["y"]  # [M, 1]
    s_true = meta["s_true"]  # Final Solution [M]

    # Create figure
    fig, gs, axes_left, axes_right = _create_unified_figure()

    # Process predictions
    processed_preds = {}

    for model_name in models_to_plot[:MAX_MODELS]:
        preds = predictions.get(model_name, {})
        u_samples = preds.get("inputs", np.empty((0,)))
        s_samples = preds.get("outputs", np.empty((0,)))

        if u_samples.size > 0 and s_samples.size > 0:
            # For probabilistic models keep all samples, for deterministic keep mean
            if model_name in SAMPLING_MODELS and u_samples.shape[0] > 1:
                u_data = u_samples
                s_data = s_samples
            else:
                u_data = u_samples.mean(axis=0)
                s_data = s_samples.mean(axis=0)

            processed_preds[model_name] = (u_data, s_data)

    # Left side: Initial Condition curves (top 5 models + ground truth)
    for idx, model_name in enumerate(models_to_plot[:MAX_INPUT_MODELS]):
        if model_name in processed_preds:
            u_data, _ = processed_preds[model_name]
            _plot_1d_curve(
                axes_left[idx],
                x_coords,
                u_true,
                u_data,
                annotation=publication_display_name(model_name),
                color=get_model_color(model_name, idx),
            )
        else:
            axes_left[idx].axis("off")

    _plot_1d_curve(
        axes_left[-1],
        x_coords,
        u_true,
        annotation="Ground Truth",
        show_gt_overlay=False,
    )

    # Right side: Final Solution curves (top 5 models + ground truth)
    for idx, model_name in enumerate(models_to_plot[:MAX_OUTPUT_MODELS]):
        if model_name in processed_preds:
            _, s_data = processed_preds[model_name]
            _plot_1d_curve(
                axes_right[idx],
                y_coords,
                s_true,
                s_data,
                annotation=publication_display_name(model_name),
                color=get_model_color(model_name, idx),
            )
        else:
            axes_right[idx].axis("off")

    _plot_1d_curve(
        axes_right[-1],
        y_coords,
        s_true,
        annotation="Ground Truth",
        show_gt_overlay=False,
    )

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, f"burgers_norm_sample_{sample_idx}_publication")
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {base}.pdf")
    print(f"✓ Saved: {base}.png")

    plt.close(fig)


def main():
    setup_publication_style(figsize=(6.5, 3.5))

    parser = argparse.ArgumentParser(
        description="Create publication-quality Normalized Burgers 1D plots."
    )
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--results_dir", type=str, default="results/burgers_1d")
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

    # Force dataset to be burgers_1d
    dataset = "burgers_1d"
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

    # Ensure params uses correct dataset name
    params.dataset = dataset

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
        max_samples=min(len(test_dataset), 200),
        device=DEVICE,
    )
    model_inverse_mse = {
        name: float(np.mean(mse_list)) if mse_list else float("inf")
        for name, mse_list in per_model_mses.items()
    }
    for name in sorted(model_inverse_mse, key=model_inverse_mse.get):
        print(f"  {publication_display_name(name)}: {model_inverse_mse[name]:.6e}")

    # Evaluate models and select best performers (get top MAX_MODELS)
    models_to_plot, sample_idx, per_model_losses = select_models_and_sample(
        test_dataset,
        models_dict,
        input_enc,
        output_enc,
        forward_model,
        max_models=MAX_MODELS,
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

    models_with_errors = [
        (name, resim_mse.get(name, float("inf"))) for name in resim_mse.keys()
    ]
    models_with_errors.sort(key=lambda x: x[1])

    ranking_title = f"\nTop {MAX_MODELS} models by re-simulation MSE:"
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
        predictions, meta = collect_burgers_predictions(
            test_dataset[idx],
            models_to_plot,
            models_dict,
            input_enc,
            output_enc,
            forward_model,
            n_samples_per_model=N_SAMPLES,
            device=DEVICE,
        )

        print("Rendering figure...")
        plot_comparison(idx, models_to_plot, predictions, meta, args.results_dir)

    print(f"SUCCESS: Created publication figure(s) → {args.results_dir}")


if __name__ == "__main__":
    main()
