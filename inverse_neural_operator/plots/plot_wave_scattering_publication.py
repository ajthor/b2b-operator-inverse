"""
Publication-quality plotting script for Wave Scattering inverse problem results.

Creates publication-ready figures with:
- 6.5 inch width for journal publications
- 5-7pt sans serif fonts
- Professional styling and layout
- High-resolution output (300 DPI)
- Unified gridspec with polar plots (input) and 2D density fields (output)
- Top 5 performing models + ground truth/observed

To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_wave_scattering_publication
"""

import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random

import torch

from data.load_dataset import load_dataset
from plots.utils.model_utils import (
    load_all_models,
    select_models_and_sample,
    collect_predictions,
)
from plots.utils.plot_utils import (
    setup_publication_style,
    display_name,
    find_params,
    load_forward_model,
    make_output_transform,
)
from plots.utils.ifno_utils import load_ifno_model, collect_ifno_predictions

device = "cpu"

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

MAX_MODELS = 5
N_SAMPLES = 8
MODEL_CMAP = mpl.cm.get_cmap("tab10")


def _create_unified_figure():
    """Create figure with unified gridspec matching test_gridspec.py exactly.

    Returns:
        fig: Figure object
        axes_left: List of 6 axes for left grid (polar plots)
        axes_right: List of 6 axes for right grid (2D density fields)
        cax_left: Colorbar axis for left grid
        cax_right: Colorbar axis for right grid
    """
    fig = plt.figure(figsize=(6.5, 2.0), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=0.5 / 72.0, h_pad=0.5 / 72.0, hspace=0.0, wspace=0.0
    )

    # Single gridspec: 2 rows × 8 columns
    # Columns: [plot, plot, plot, colorbar, plot, plot, plot, colorbar]
    gs = fig.add_gridspec(
        2,
        7,
        width_ratios=[1, 1, 1, 1, 1, 1, 0.05],
        hspace=0.0,
        wspace=0.0,
        left=0,
        right=1,
        top=1,
        bottom=0,
    )

    # Create parent axes for shared labels (invisible, just for labels)
    ax_left_parent = fig.add_subplot(gs[:, 0:3], frameon=False)
    ax_left_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    # ax_left_parent.set_xlabel(r"$\theta$ (radians)", labelpad=-8)
    # ax_left_parent.set_ylabel(r"$|u(\theta)|$", labelpad=-8)
    ax_left_parent.set_title("Predicted Far-Field Patterns")

    ax_right_parent = fig.add_subplot(gs[:, 3:6], frameon=False)
    ax_right_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_right_parent.set_xlabel(r"$x$", labelpad=2)
    ax_right_parent.set_ylabel(r"$y$", labelpad=-8)
    ax_right_parent.set_title("Re-simulated Density Fields")

    # Create subplot axes for left grid (polar plots)
    axes_left = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j], projection="polar")
            axes_left.append(ax)

    # Sharex and sharey for left grid
    for ax in axes_left[1:]:
        ax.sharex(axes_left[0])
        ax.sharey(axes_left[0])

    # Create subplot axes for right grid (2D density fields)
    axes_right = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j + 3])
            axes_right.append(ax)

    # Colorbar axes
    # cax_left = fig.add_subplot(gs[:, 3])
    cax_right = fig.add_subplot(gs[:, 6])

    return fig, axes_left, axes_right, None, cax_right


def _plot_polar_field(ax, theta, magnitude, annotation=None):
    """Plot a polar field (far-field pattern).

    Args:
        ax: Polar axis to plot on
        theta: Angle values (radians)
        magnitude: Magnitude values
        annotation: Optional text annotation for upper-left corner

    Returns:
        The plot object
    """
    # Sort by theta for smooth plotting
    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]
    magnitude_sorted = magnitude[sort_idx]

    # Plot the polar pattern
    line = ax.plot(theta_sorted, magnitude_sorted, "b-", linewidth=0.5)[0]

    # Remove tick labels for cleaner look
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])

    # Make tick labels tiny
    ax.tick_params(axis="both", which="major", labelsize=3)
    # Move radial labels to be closer to the plot
    ax.xaxis.set_tick_params(pad=-5)

    # Add annotation if provided
    if annotation:
        ax.text(
            0.05,
            0.95,
            annotation,
            transform=ax.transAxes,
            fontsize=6,
            color="white",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="none"
            ),
        )

    return line


def _plot_2d_field(
    ax, field_2d, cmap="viridis", vmin=None, vmax=None, annotation=None, show_labels=False
):
    """Plot a 2D density field.

    Args:
        ax: Axis to plot on
        field_2d: 2D numpy array to visualize
        cmap: Colormap name
        vmin, vmax: Color scale limits
        annotation: Optional text annotation for upper-left corner

    Returns:
        The image object
    """
    extent = [0, 1, 0, 1]
    im = ax.imshow(
        field_2d, cmap=cmap, extent=extent, origin="lower", vmin=vmin, vmax=vmax
    )
    ticks = [0.0, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(
        bottom=True,
        left=True,
        top=False,
        right=False,
        labelbottom=show_labels,
        labelleft=show_labels,
    )
    ax.set_aspect("equal")

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


def _plot_2d_field_cutaway(
    ax,
    field_binary,
    field_continuous,
    cmap="viridis",
    vmin=None,
    vmax=None,
    annotation=None,
    show_labels=False,
):
    """Plot a 2D density field with small inset showing continuous values.

    Args:
        ax: Axis to plot on
        field_binary: 2D numpy array for binary/thresholded field (background)
        field_continuous: 2D numpy array for continuous values (inset overlay)
        cmap: Colormap name
        vmin, vmax: Color scale limits
        annotation: Optional text annotation for upper-left corner

    Returns:
        The image object from the main binary field
    """
    extent = [0, 1, 0, 1]

    # Plot binary field as main image
    im = ax.imshow(
        field_binary, cmap=cmap, extent=extent, origin="lower", vmin=vmin, vmax=vmax
    )

    ticks = [0.0, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(
        bottom=True,
        left=True,
        top=False,
        right=False,
        labelbottom=show_labels,
        labelleft=show_labels,
    )
    ax.set_aspect("equal")

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

    # Create inset axes in lower-right corner (about 1/4 area)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax_inset = inset_axes(
        ax, width="50%", height="50%", loc="lower right", borderpad=0.3
    )

    # Plot continuous field in inset
    ax_inset.imshow(
        field_continuous, cmap=cmap, extent=extent, origin="lower", vmin=vmin, vmax=vmax
    )
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    # Add white border to make inset stand out
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.5)

    return im


# def plot_unified_comparison(
#     models_dict,
#     ranked_models,
#     input_function_encoder,
#     output_function_encoder,
#     forward_model,
#     test_dataset,
#     sample_idx,
#     save_path=None,
# ):
#     """Plot unified comparison showing top 5 models + ground truth/observed.

#     Args:
#         models_dict: Dictionary of {model_name: (model, evaluate_fn)}
#         ranked_models: List of model names ranked by performance
#         input_function_encoder: Input function encoder
#         output_function_encoder: Output function encoder
#         forward_model: Forward model for re-simulation
#         test_dataset: Test dataset
#         sample_idx: Index of sample to visualize
#         save_path: Optional path to save figure
#     """
#     # Get sample data
#     X, u_true, Y, s_observed = test_dataset[sample_idx]

#     # Ensure tensors are on correct device
#     X = X.to(device)
#     u_true = u_true.to(device)
#     Y = Y.to(device)
#     s_observed = s_observed.to(device)

#     # Extract theta coordinates from X (X is [cos(theta), sin(theta)])
#     X_np = X.squeeze(-1).cpu().numpy()
#     theta = np.arctan2(X_np[:, 1], X_np[:, 0])

#     # Convert ground truth to numpy
#     u_true_np = u_true.squeeze(-1).cpu().numpy()
#     s_observed_np = s_observed.squeeze(-1).cpu().numpy()

#     # Reshape density fields
#     grid_size = 200
#     s_observed_2d = s_observed_np.reshape(grid_size, grid_size)

#     # Create thresholded version (binary density field)
#     s_observed_thresholded = (s_observed_2d > 0.5).astype(float)

#     # Create figure
#     fig, axes_left, axes_right, cax_left, cax_right = _create_unified_figure()

#     # Select top 5 models
#     top_5_models = ranked_models[:5]

#     # Store image object for colorbar
#     im_right = None

#     # Number of samples for averaging (probabilistic models)
#     n_samples = 10

#     # Determine value limits across all models for consistent colormaps
#     all_s_values = [s_observed_thresholded]

#     # Store predictions for each model to avoid recomputing
#     predictions = {}  # model_name -> (u_pred_np, s_resim_2d)

#     # Compute predictions for all top 5 models
#     for idx, model_name in enumerate(top_5_models):
#         model, evaluate_fn = models_dict[model_name]
#         model.eval()

#         # Sample multiple evaluations and average
#         u_pred_samples = []
#         s_resim_samples = []

#         with torch.no_grad():
#             point = (
#                 X.unsqueeze(0),
#                 u_true.unsqueeze(0),
#                 Y.unsqueeze(0),
#                 s_observed.unsqueeze(0),
#             )

#             for _ in range(n_samples):
#                 # Predict input (far-field pattern)
#                 u_pred, _ = evaluate_fn(
#                     model, point, input_function_encoder, output_function_encoder
#                 )
#                 u_pred = u_pred.squeeze(0)
#                 u_pred_samples.append(u_pred.cpu().numpy())

#                 # Re-simulate using forward model
#                 X_batch = X.unsqueeze(0)
#                 u_pred_batch = u_pred.unsqueeze(0)
#                 Y_batch = Y.unsqueeze(0)

#                 # Compute alpha coefficients from predicted input
#                 alpha, _ = input_function_encoder.compute_coefficients(
#                     X_batch, u_pred_batch
#                 )

#                 # Forward pass through model to get beta coefficients
#                 beta_pred = forward_model.forward(alpha)

#                 # Reconstruct re-simulation output
#                 s_resim = output_function_encoder(Y_batch, beta_pred)
#                 s_resim = s_resim.squeeze(0)
#                 s_resim_samples.append(s_resim.cpu().numpy())

#             # Average the samples
#             u_pred_avg = np.mean(u_pred_samples, axis=0)
#             s_resim_avg = np.mean(s_resim_samples, axis=0)

#             # Convert to numpy
#             u_pred_np = u_pred_avg.squeeze(-1)
#             s_resim_np = s_resim_avg.squeeze(-1)

#             # Reshape to 2D (continuous, not thresholded)
#             s_resim_2d = s_resim_np.reshape(grid_size, grid_size)

#             # Create thresholded version (binary density field)
#             s_resim_thresholded = (s_resim_2d > 0.5).astype(float)

#             # Store for later plotting
#             predictions[model_name] = (u_pred_np, s_resim_thresholded)
#             all_s_values.append(s_resim_thresholded)

#     # Determine consistent value limits for density field plots
#     s_min = min(np.min(v) for v in all_s_values)
#     s_max = max(np.max(v) for v in all_s_values)

#     # Plot using stored predictions
#     for idx, model_name in enumerate(top_5_models):
#         u_pred_np, s_resim_thresholded = predictions[model_name]

#         # Plot predicted far-field pattern (left side - polar)
#         ax = axes_left[idx]
#         _plot_polar_field(
#             ax, theta, np.abs(u_pred_np), annotation=display_name(model_name)
#         )

#         # Plot re-simulated density field (right side - 2D, continuous)
#         ax = axes_right[idx]
#         im_right = _plot_2d_field(
#             ax,
#             s_resim_thresholded,
#             cmap="viridis",
#             vmin=s_min,
#             vmax=s_max,
#             annotation=display_name(model_name),
#         )

#     # 6th position on left: ground truth far-field pattern
#     ax = axes_left[5]
#     _plot_polar_field(ax, theta, np.abs(u_true_np), annotation="Ground Truth")

#     # 6th position on right: observed density field
#     ax = axes_right[5]
#     im_right = _plot_2d_field(
#         ax,
#         s_observed_thresholded,
#         cmap="viridis",
#         vmin=s_min,
#         vmax=s_max,
#         annotation="Observed",
#     )

#     # Add colorbars (for the 2D density fields only)
#     # Left colorbar: we don't need one for polar plots, so remove tick labels
#     cax_left.set_visible(False)

#     # Right colorbar: for density fields
#     cbar_right = fig.colorbar(im_right, cax=cax_right, use_gridspec=True)
#     cbar_right.set_label("Density")

#     # Save figure if path provided
#     if save_path:
#         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#         # Determine format from extension
#         if save_path.endswith(".pdf"):
#             plt.savefig(
#                 save_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.05
#             )
#         else:
#             plt.savefig(
#                 save_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05
#             )
#         print(f"✓ Saved publication figure → {save_path}")

#     plt.close()


def plot_comparison(sample_idx, models_to_plot, predictions, meta, save_dir):
    """Create publication-quality comparison plot."""
    # Extract data from meta
    x = meta["x"]
    u_true = meta["u_true"]
    s_true = meta["s_true"]

    # Extract theta from x coordinates (x is flattened [cos, sin, cos, sin, ...])
    x_2d = x.reshape(-1, 2)
    theta = np.arctan2(x_2d[:, 1], x_2d[:, 0])

    # u_true is flattened [real, imag, real, imag, ...]
    u_true_2d = u_true.reshape(-1, 2)
    u_true_mag = np.sqrt(u_true_2d[:, 0] ** 2 + u_true_2d[:, 1] ** 2)

    # Reshape ground truth density field
    grid_size = 200
    s_true_2d = s_true.reshape(grid_size, grid_size)

    # Create thresholded version (binary density field)
    s_true_thresholded = (s_true_2d > 0.5).astype(float)

    # Create figure
    fig, axes_left, axes_right, cax_left, cax_right = _create_unified_figure()

    # Collect all density values for consistent colormap
    all_s_values = [s_true_thresholded]

    # Process predictions
    processed_preds = {}
    for model_name in models_to_plot[:5]:
        preds = predictions.get(model_name, {})
        u_samples = preds.get("inputs", np.empty((0,)))
        s_samples = preds.get("outputs", np.empty((0,)))

        if u_samples.size > 0 and s_samples.size > 0:
            # u_samples is (n_samples, N*2) with [real, imag] pairs
            u_samples_2d = u_samples.reshape(u_samples.shape[0], -1, 2)
            u_magnitudes = np.sqrt(
                u_samples_2d[..., 0] ** 2 + u_samples_2d[..., 1] ** 2
            )
            u_mean_mag = u_magnitudes.mean(axis=0)

            # s_samples is (n_samples, N*M) flattened
            s_mean = s_samples.mean(axis=0)
            s_mean_2d = s_mean.reshape(grid_size, grid_size)

            # Create thresholded version (binary density field)
            s_mean_thresholded = (s_mean_2d > 0.5).astype(float)

            # For iFNO, store both continuous and thresholded versions
            if model_name == "ifno":
                processed_preds[model_name] = (
                    u_mean_mag,
                    s_mean_thresholded,
                    s_mean_2d,
                )
            else:
                processed_preds[model_name] = (u_mean_mag, s_mean_thresholded)
            all_s_values.append(s_mean_thresholded)

    # Determine consistent value limits
    s_min = min(np.min(v) for v in all_s_values)
    s_max = max(np.max(v) for v in all_s_values)

    # Plot models
    im_right = None
    for idx, model_name in enumerate(models_to_plot[:5]):
        if model_name in processed_preds:
            pred_data = processed_preds[model_name]

            # Check if this is iFNO (has 3 elements: u_mag, binary, continuous)
            if len(pred_data) == 3:
                u_mag, s_binary, s_continuous = pred_data
                is_ifno = True
            else:
                u_mag, s_2d = pred_data
                is_ifno = False

            # Far-field pattern (left - polar)
            ax = axes_left[idx]
            show_labels = idx == 3
            _plot_polar_field(
                ax,
                theta,
                u_mag,
                annotation=display_name(model_name),
            )

            # Density field (right - 2D)
            ax = axes_right[idx]
            if is_ifno:
                # Use cutaway plot for iFNO to show continuous values
                im_right = _plot_2d_field_cutaway(
                    ax,
                    s_binary,
                    s_continuous,
                    cmap="viridis",
                    vmin=s_min,
                    vmax=s_max,
                    annotation=display_name(model_name),
                    show_labels=show_labels,
                )
            else:
                # Regular binary plot for other models
                im_right = _plot_2d_field(
                    ax,
                    s_2d,
                    cmap="viridis",
                    vmin=s_min,
                    vmax=s_max,
                    annotation=display_name(model_name),
                    show_labels=show_labels,
                )

    # 6th position: ground truth
    ax = axes_left[5]
    _plot_polar_field(ax, theta, u_true_mag, annotation="Ground Truth")

    ax = axes_right[5]
    im_right = _plot_2d_field(
        ax,
        s_true_thresholded,
        cmap="viridis",
        vmin=s_min,
        vmax=s_max,
        annotation="Ground Truth",
    )

    # Colorbars
    # cax_left.set_visible(False)
    if im_right is not None:
        cbar_right = fig.colorbar(im_right, cax=cax_right, use_gridspec=True)
        cbar_right.set_label("Density")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(
        save_dir, f"wave_scattering_sample_{sample_idx}_publication.pdf"
    )
    png_path = os.path.join(
        save_dir, f"wave_scattering_sample_{sample_idx}_publication.png"
    )

    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()


def main():
    setup_publication_style()
    mpl.rcParams.update(
        {
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "xtick.major.pad": 1.0,
            "ytick.major.pad": 1.0,
        }
    )

    parser = argparse.ArgumentParser(
        description="Create publication-quality Wave Scattering plots."
    )
    parser.add_argument(
        "--log_dir", type=str, default="/store/at46867/b2b_operator_inverse"
    )
    parser.add_argument("--results_dir", type=str, default="results/wave_scattering")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument(
        "--ifno_checkpoint",
        type=str,
        default="",
        help="Optional override for IFNO checkpoint path (otherwise auto-detected).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    results_dir = os.path.abspath(args.results_dir)
    dataset = "wave_scattering"
    models_root = args.log_dir
    log_dir = os.path.join(models_root, dataset)
    if not os.path.exists(log_dir):
        alt_root = os.path.join(models_root, "models")
        alt_log_dir = os.path.join(alt_root, dataset)
        if os.path.exists(alt_log_dir):
            models_root = alt_root
            log_dir = alt_log_dir

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
        params.dataset, params, device, split="test", return_info=True
    )
    print(f"✓ Loaded {len(test_dataset)} test samples")
    cache_path = os.path.join(results_dir, "plot_cache.json")
    cache_key = f"seed_{args.seed}"
    cache_metadata = {"dataset": params.dataset, "seed": int(args.seed)}

    # Load models
    print("\nLoading models...")
    normalized_root = os.path.normpath(models_root)
    if os.path.basename(normalized_root) == "models":
        models_base_dir = os.path.dirname(normalized_root)
    else:
        models_base_dir = models_root
    models_dict, input_enc, output_enc = load_all_models(
        models_base_dir, params.dataset, INVERSE_MODELS, args.seed, device
    )

    if not models_dict:
        print("ERROR: no models loaded")
        raise SystemExit(1)

    forward_model = load_forward_model(log_dir, args.seed, device=device)
    output_transform = make_output_transform(forward_model, output_enc)
    repo_root = Path(__file__).resolve().parents[2]

    # Evaluate models and select best performers
    models_to_plot, sample_idx = select_models_and_sample(
        test_dataset,
        models_dict,
        input_enc,
        output_enc,
        forward_model,
        max_models=MAX_MODELS,
        sample_index=args.sample_index,
        device=device,
        cache_path=cache_path,
        cache_metadata=cache_metadata,
        cache_key=cache_key,
    )

    default_ifno_checkpoint = os.path.join(
        models_base_dir,
        "models",
        params.dataset,
        "ifno",
        f"seed_{args.seed}",
        "ifno_model.safetensors",
    )
    manual_ifno = args.ifno_checkpoint.strip() if args.ifno_checkpoint else ""
    if manual_ifno:
        manual_ifno = os.path.abspath(manual_ifno)
        if not os.path.exists(manual_ifno):
            raise SystemExit(
                f"ERROR: IFNO checkpoint override not found: {manual_ifno}"
            )
        ifno_checkpoint = manual_ifno
    else:
        if os.path.exists(default_ifno_checkpoint):
            ifno_checkpoint = default_ifno_checkpoint
        else:
            raise SystemExit(
                f"ERROR: IFNO checkpoint not found: {default_ifno_checkpoint}\n"
                "       Please train IFNO or provide --ifno_checkpoint."
            )
    include_ifno = bool(ifno_checkpoint)
    if include_ifno and not os.path.exists(ifno_checkpoint):
        print(
            f"  Warning: IFNO checkpoint not found at {ifno_checkpoint}. Skipping IFNO panel."
        )
        include_ifno = False

    b2b_models_to_plot = list(models_to_plot)
    if include_ifno and len(b2b_models_to_plot) >= MAX_MODELS:
        b2b_models_to_plot = b2b_models_to_plot[:-1]

    print("Collecting predictions...")
    predictions, meta = collect_predictions(
        test_dataset[sample_idx],
        b2b_models_to_plot,
        models_dict,
        input_enc,
        output_enc,
        output_transform=output_transform,
        n_samples_per_model=N_SAMPLES,
        device=device,
    )

    final_model_order = list(b2b_models_to_plot)
    if include_ifno:
        try:
            print("Evaluating IFNO model for visualization...")
            ifno_model = load_ifno_model(dataset_info, ifno_checkpoint, device=device)
            predictions["ifno"] = collect_ifno_predictions(
                ifno_model,
                test_dataset[sample_idx],
                device=device,
                n_samples=N_SAMPLES,
            )
            final_model_order.append("ifno")
        except Exception as exc:
            print(
                f"  Warning: Unable to evaluate IFNO checkpoint ({exc}). Skipping IFNO panel."
            )

    print("Rendering figure...")
    plot_comparison(sample_idx, final_model_order, predictions, meta, results_dir)
    print(f"SUCCESS: Created publication figure → {results_dir}")


if __name__ == "__main__":
    main()
