"""
Publication-quality plotting script for FWI inverse problem results.

Creates publication-ready figures with:
- 6.5 inch width for journal publications
- 5-7pt sans serif fonts
- Professional styling and layout
- High-resolution output (300 DPI)
- 2x2 grid layout with paired velocity/seismic comparisons

To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_fwi_publication
"""

import argparse
import json
import os
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as compute_ssim

from data.load_dataset import load_dataset
from plots.utils.model_utils import load_all_models
from plots.utils.plot_utils import (
    setup_publication_style,
    display_name,
    find_params,
    load_forward_model,
)

device = "cpu"

# Models to plot (commenting out selection for now)
MODELS_TO_PLOT = [
    "linear",
    "nonlinear",
    "conditional_realnvp",
]

MODEL_CMAP = mpl.cm.get_cmap("tab10")


def compute_sample_ssim_scores(
    test_dataset,
    models_dict,
    input_enc,
    output_enc,
    forward_model,
    output_shape,
    device="cpu",
):
    """Compute median SSIM across models for every test sample."""
    scores = []
    if forward_model is not None:
        forward_model.eval()

    for idx in range(len(test_dataset)):
        X, u_true, Y, s_observed = test_dataset[idx]
        X = X.to(device)
        u_true = u_true.to(device)
        Y = Y.to(device)
        s_observed = s_observed.to(device)

        point = (
            X.unsqueeze(0),
            u_true.unsqueeze(0),
            Y.unsqueeze(0),
            s_observed.unsqueeze(0),
        )

        sample_ssims = []
        for model_name, (model, evaluate_fn) in models_dict.items():
            model.eval()

            with torch.no_grad():
                u_pred, _ = evaluate_fn(model, point, input_enc, output_enc)
                u_pred = u_pred.squeeze(0)

                if forward_model is not None:
                    alpha, _ = input_enc.compute_coefficients(
                        X.unsqueeze(0), u_pred.unsqueeze(0)
                    )
                    beta_pred = forward_model.forward(alpha)
                    s_resim = output_enc(Y.unsqueeze(0), beta_pred)
                    s_resim = s_resim.squeeze(0)
                else:
                    # Fallback to model's predicted outputs if forward model unavailable
                    s_resim = output_enc(Y.unsqueeze(0), u_pred.unsqueeze(0)).squeeze(0)

            s_obs_np = s_observed.squeeze(-1).cpu().numpy().reshape(output_shape)
            s_resim_np = s_resim.squeeze(-1).cpu().numpy().reshape(output_shape)

            data_range = max(s_obs_np.max(), s_resim_np.max()) - min(
                s_obs_np.min(), s_resim_np.min()
            )
            if data_range == 0:
                data_range = 1.0

            ssim_val = compute_ssim(
                s_obs_np,
                s_resim_np,
                data_range=data_range,
                channel_axis=None,
            )
            sample_ssims.append(ssim_val)

        if sample_ssims:
            scores.append((idx, float(np.median(sample_ssims))))

        if (idx + 1) % 100 == 0:
            print(f"    Processed {idx + 1}/{len(test_dataset)} samples...")

    return scores


def plot_fwi_publication(
    sample_idx,
    models_to_plot,
    models_dict,
    test_dataset,
    input_enc,
    output_enc,
    forward_model,
    stats,
    save_dir,
):
    """Create the FWI publication figure with 2x2 velocity and 2x2 seismic grids."""

    # Set publication-quality rcParams
    plt.rcParams.update(
        {
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "lines.linewidth": 1.0,
        }
    )

    fig = plt.figure(figsize=(6.5, 1.8), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=0.5 / 72.0, h_pad=0.5 / 72.0, hspace=0.0, wspace=0.0
    )

    # Single gridspec: 2 rows x 6 columns
    # Columns: [plot, plot, colorbar, plot, plot, colorbar]
    # Velocity models are 24x48, so use larger width ratio to make them more square
    gs = fig.add_gridspec(
        2,
        6,
        width_ratios=[1.2, 1.2, 0.05, 0.8, 0.8, 0.05],
        hspace=0.0,
        wspace=0.0,
        left=0,
        right=1,
        top=1,
        bottom=0,
    )

    # Create parent axes for shared labels (invisible, just for labels)
    ax_left_parent = fig.add_subplot(gs[:, 0:2], frameon=False)
    ax_left_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_left_parent.set_xlabel("x", labelpad=-8)
    ax_left_parent.set_ylabel("y", labelpad=-8)
    ax_left_parent.set_title("Velocity Model Reconstructions")

    ax_right_parent = fig.add_subplot(gs[:, 3:5], frameon=False)
    ax_right_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_right_parent.set_xlabel("Frequency", labelpad=-8)
    ax_right_parent.set_ylabel("Time", labelpad=-8)
    ax_right_parent.set_title("Seismic Transform Predictions")

    # Get ground truth sample
    X, u_true, Y, s_observed = test_dataset[sample_idx]
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_observed = s_observed.to(device)

    # Denormalize velocity fields (dataset is normalized without gradient subtraction)
    vmin = stats["models_min"]
    vmax = stats["models_max"]
    residual_min = vmin - 900.0
    residual_max = vmax - 100.0

    def denormalize_velocity(u_normalized):
        """Denormalize velocity prediction to physical units (m/s)."""
        u_np = u_normalized.squeeze(-1).cpu().numpy()
        u_velocity = ((u_np + 1) / 2) * (residual_max - residual_min) + residual_min
        return u_velocity.reshape(24, 48)

    def reshape_seismic(s):
        """Reshape seismic transform from flat to 2D."""
        s_np = s.squeeze(-1).cpu().numpy()
        return s_np.reshape(400, 76)

    # Ground truth data
    u_true_2d = denormalize_velocity(u_true)
    s_observed_2d = reshape_seismic(s_observed)

    # Collect predictions from models
    predictions_velocity = []
    predictions_seismic = []

    point = (
        X.unsqueeze(0),
        u_true.unsqueeze(0),
        Y.unsqueeze(0),
        s_observed.unsqueeze(0),
    )

    for model_name in models_to_plot:
        if model_name not in models_dict:
            print(f"Warning: {model_name} not found in loaded models")
            continue

        model, evaluate_fn = models_dict[model_name]
        model.eval()

        with torch.no_grad():
            u_pred, _ = evaluate_fn(model, point, input_enc, output_enc)
            u_pred = u_pred.squeeze(0)

            # Get velocity prediction
            u_pred_2d = denormalize_velocity(u_pred)
            predictions_velocity.append((model_name, u_pred_2d))

            # Re-simulate seismic transform
            if forward_model is not None:
                forward_model.eval()
                X_batch = X.unsqueeze(0)
                u_pred_batch = u_pred.unsqueeze(0)
                Y_batch = Y.unsqueeze(0)

                alpha, _ = input_enc.compute_coefficients(X_batch, u_pred_batch)
                beta_pred = forward_model.forward(alpha)
                s_resim = output_enc(Y_batch, beta_pred)
                s_resim_2d = reshape_seismic(s_resim.squeeze(0))
                predictions_seismic.append((model_name, s_resim_2d))

    # Add ground truth as 4th item
    predictions_velocity.append(("Ground Truth", u_true_2d))
    predictions_seismic.append(("Ground Truth", s_observed_2d))

    # Use fixed velocity range from FWI data specification
    # Velocity range: 100 m/s (lighter) to 900 m/s (darker with magma_r)
    vel_min = 100.0
    vel_max = 900.0

    # Compute shared color limits for seismic transforms
    seismic_min = s_observed_2d.min()
    seismic_max = s_observed_2d.max()
    for _, s_resim_2d in predictions_seismic:
        seismic_min = min(seismic_min, s_resim_2d.min())
        seismic_max = max(seismic_max, s_resim_2d.max())

    # Create coordinate grids for velocity contour plots
    x_coords = np.arange(48)
    y_coords = np.arange(24)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

    norm_velocity = mpl.colors.Normalize(vmin=vel_min, vmax=vel_max)

    # Plot velocity models (left grid - columns 0-1)
    for idx, (model_name, u_2d) in enumerate(predictions_velocity):
        i = idx // 2  # row
        j = idx % 2  # column
        ax = fig.add_subplot(gs[i, j])

        im = ax.imshow(
            u_2d,
            cmap="magma_r",
            vmin=vel_min,
            vmax=vel_max,
            origin="upper",
            aspect="equal",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # Add annotation
        ax.text(
            0.05,
            0.95,
            display_name(model_name),
            transform=ax.transAxes,
            fontsize=6,
            color="white",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="none"
            ),
        )

    # Shared colorbar for velocity models (column 2) with fixed absolute range
    cax_left = fig.add_subplot(gs[:, 2])
    scalar_mappable = mpl.cm.ScalarMappable(norm=norm_velocity, cmap="magma_r")
    scalar_mappable.set_array([])
    cbar_left = fig.colorbar(scalar_mappable, cax=cax_left, use_gridspec=True)
    cbar_left.set_label("Velocity (m/s)")
    cbar_left.ax.invert_yaxis()

    # Plot seismic transforms (right grid - columns 3-4)
    for idx, (model_name, s_2d) in enumerate(predictions_seismic):
        i = idx // 2  # row
        j = idx % 2  # column
        ax = fig.add_subplot(gs[i, j + 3])

        im = ax.imshow(
            s_2d,
            cmap="turbo",
            aspect="auto",
            origin="lower",
            vmin=seismic_min,
            vmax=seismic_max,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")

        # Add annotation
        ax.text(
            0.05,
            0.95,
            display_name(model_name),
            transform=ax.transAxes,
            fontsize=6,
            color="white",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="none"
            ),
        )

    # Shared colorbar for seismic transforms (column 5)
    cax_right = fig.add_subplot(gs[:, 5])
    cbar_right = fig.colorbar(im, cax=cax_right, use_gridspec=True)
    cbar_right.set_label("Amplitude")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, f"fwi_comparison_sample_{sample_idx}")
        fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {base}.pdf and {base}.png")

    plt.close(fig)


def main():
    setup_publication_style()

    parser = argparse.ArgumentParser(
        description="Create publication-quality FWI plots."
    )
    parser.add_argument(
        "--base_dir", type=str, default="/store/at46867/b2b_operator_inverse",
        help="Base directory for models and results"
    )
    parser.add_argument("--results_dir", type=str, default="results/fwi")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="fwi")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    base_dir = args.base_dir
    model_dir = os.path.join(base_dir, "models", args.dataset)
    log_dir = os.path.join(base_dir, "runs", args.dataset)
    if not os.path.exists(model_dir):
        print(f"ERROR: model directory not found: {model_dir}")
        raise SystemExit(1)

    # Load normalization statistics
    stats_path = os.path.join(os.path.dirname(__file__), "../data/fwi_stats.json")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    print(
        f"✓ Loaded normalization stats: velocity range [{stats['models_min']:.2f}, {stats['models_max']:.2f}]"
    )

    params = find_params(model_dir, MODELS_TO_PLOT, args.seed)
    test_dataset, dataset_info = load_dataset(
        params.dataset, params, device, split="test", return_info=True
    )
    print(f"✓ Loaded {len(test_dataset)} test samples")

    print("Loading models...")
    models_dict, input_enc, output_enc = load_all_models(
        base_dir, args.dataset, MODELS_TO_PLOT, seed=args.seed, device=device
    )

    if not models_dict:
        print("ERROR: no models loaded")
        raise SystemExit(1)

    # Load forward model for re-simulation
    forward_model = load_forward_model(
        base_dir, args.dataset, args.seed, forward_model_name="b2b_nonlinear", device=device
    )

    # Use the specified models (don't rank them)
    models_to_plot = MODELS_TO_PLOT

    # Select the best sample based on performance across all models
    if args.sample_index is None:
        print("Evaluating models on full test set to find best sample (SSIM)...")
        output_shape = dataset_info.get("output_spatial_dims", (400, 76))
        sample_scores = compute_sample_ssim_scores(
            test_dataset,
            models_dict,
            input_enc,
            output_enc,
            forward_model,
            output_shape,
            device=device,
        )

        if not sample_scores:
            raise ValueError("Unable to compute SSIM scores for samples.")

        # Select best sample (highest median SSIM across models)
        sample_scores_sorted = sorted(sample_scores, key=lambda x: x[1], reverse=True)
        sample_idx = sample_scores_sorted[0][0]
        best_ssim = sample_scores_sorted[0][1]
        print(f"Selected best sample: {sample_idx} (median SSIM: {best_ssim:.4f})")
    else:
        sample_idx = args.sample_index
        print(f"Using provided sample: {sample_idx}")

    print("Rendering figure...")
    plot_fwi_publication(
        sample_idx=sample_idx,
        models_to_plot=models_to_plot,
        models_dict=models_dict,
        test_dataset=test_dataset,
        input_enc=input_enc,
        output_enc=output_enc,
        forward_model=forward_model,
        stats=stats,
        save_dir=args.results_dir,
    )
    print(f"SUCCESS: Created publication figure → {args.results_dir}")


if __name__ == "__main__":
    main()
