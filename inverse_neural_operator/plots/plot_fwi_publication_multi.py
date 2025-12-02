"""
Multi-sample publication-quality plotting script for the FWI inverse problem.

Creates a single figure with a unified gridspec showing multiple test samples.
Each row displays all model predictions for one sample, with horizontal colorbars at the bottom.
"""

import argparse
import json
import os
import random
from typing import Iterable, Tuple

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

# Default models used in the single-sample publication plot
MODELS_TO_PLOT = [
    "linear",
    "nonlinear",
    "conditional_realnvp",
]

# Temporary static samples used while SSIM-based selection is disabled.
STATIC_SAMPLE_INDICES = [9012, 42, 2, 666, 890]


def denormalize_velocity(
    u_normalized: torch.Tensor, residual_min: float, residual_max: float
) -> np.ndarray:
    """Map normalized velocity residuals back to physical units (m/s)."""
    u_np = u_normalized.squeeze(-1).cpu().numpy()
    u_velocity = ((u_np + 1) / 2) * (residual_max - residual_min) + residual_min
    return u_velocity.reshape(24, 48)


def reshape_seismic(s: torch.Tensor, output_shape: Tuple[int, int]) -> np.ndarray:
    """Reshape flattened seismic transform into 2D grid."""
    s_np = s.squeeze(-1).cpu().numpy()
    return s_np.reshape(*output_shape)


def compute_predictions_for_sample(
    sample_idx: int,
    models_to_plot: Iterable[str],
    models_dict,
    test_dataset,
    input_enc,
    output_enc,
    forward_model,
    stats,
    output_shape: Tuple[int, int],
):
    """Compute predictions for a single sample and return denormalized results with SSIM scores."""
    # Retrieve sample tensors
    X, u_true, Y, s_observed = test_dataset[sample_idx]
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_observed = s_observed.to(device)

    # Denormalization helpers
    vmin = stats["models_min"]
    vmax = stats["models_max"]
    residual_min = vmin - 900.0
    residual_max = vmax - 100.0

    u_true_2d = denormalize_velocity(u_true, residual_min, residual_max)
    s_observed_2d = reshape_seismic(s_observed, output_shape)

    predictions_velocity = []
    predictions_seismic = []
    ssim_scores_velocity = []
    ssim_scores_seismic = []

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

            u_pred_2d = denormalize_velocity(u_pred, residual_min, residual_max)
            predictions_velocity.append((model_name, u_pred_2d))

            # Compute SSIM for velocity models
            vel_data_range = max(u_true_2d.max(), u_pred_2d.max()) - min(
                u_true_2d.min(), u_pred_2d.min()
            )
            if vel_data_range == 0:
                vel_data_range = 1.0
            vel_ssim = compute_ssim(
                u_true_2d,
                u_pred_2d,
                data_range=vel_data_range,
                channel_axis=None,
            )
            ssim_scores_velocity.append((model_name, vel_ssim))

            if forward_model is not None:
                forward_model.eval()
                alpha, _ = input_enc.compute_coefficients(
                    X.unsqueeze(0), u_pred.unsqueeze(0)
                )
                beta_pred = forward_model.forward(alpha)
                s_resim = output_enc(Y.unsqueeze(0), beta_pred)
                s_resim_2d = reshape_seismic(s_resim.squeeze(0), output_shape)
                predictions_seismic.append((model_name, s_resim_2d))

                # Compute SSIM for seismic transforms
                seis_data_range = max(s_observed_2d.max(), s_resim_2d.max()) - min(
                    s_observed_2d.min(), s_resim_2d.min()
                )
                if seis_data_range == 0:
                    seis_data_range = 1.0
                seis_ssim = compute_ssim(
                    s_observed_2d,
                    s_resim_2d,
                    data_range=seis_data_range,
                    channel_axis=None,
                )
                ssim_scores_seismic.append((model_name, seis_ssim))

    predictions_velocity.append(("Ground Truth", u_true_2d))
    predictions_seismic.append(("Ground Truth", s_observed_2d))

    return (
        predictions_velocity,
        predictions_seismic,
        ssim_scores_velocity,
        ssim_scores_seismic,
    )


def main():
    setup_publication_style()

    parser = argparse.ArgumentParser(
        description="Create multi-sample publication-quality FWI plots."
    )
    parser.add_argument(
        "--base_dir", type=str, default="/store/at46867/b2b_operator_inverse",
        help="Base directory for models and results"
    )
    parser.add_argument("--results_dir", type=str, default="results/fwi")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="fwi")
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="*",
        help="Specific sample indices to plot (expects three values).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to plot when sample_indices not provided.",
    )
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

    forward_model = load_forward_model(
        base_dir, args.dataset, args.seed, forward_model_name="b2b_nonlinear", device=device
    )

    output_shape = dataset_info.get("output_spatial_dims", (400, 76))
    score_dict = {}

    if args.sample_indices:
        sample_indices = args.sample_indices
        if len(sample_indices) != args.num_samples:
            print(
                f"⚠ Provided {len(sample_indices)} sample indices, but num_samples={args.num_samples}. "
                f"Using provided indices and ignoring num_samples."
            )
    else:
        sample_indices = STATIC_SAMPLE_INDICES[: args.num_samples]
        if len(sample_indices) < args.num_samples:
            raise ValueError(
                f"Requested num_samples={args.num_samples}, "
                f"but only {len(STATIC_SAMPLE_INDICES)} static indices are configured."
            )

    # Ensure indices are within dataset bounds to avoid indexing errors.
    invalid_indices = [idx for idx in sample_indices if idx >= len(test_dataset)]
    if invalid_indices:
        raise ValueError(
            f"Static sample indices {invalid_indices} exceed dataset size ({len(test_dataset)}). "
            "Update STATIC_SAMPLE_INDICES to valid values."
        )

    print(f"✓ Selected samples: {sample_indices}")

    # Compute all predictions
    all_predictions = []
    global_seismic_min = float("inf")
    global_seismic_max = float("-inf")

    for sample_idx in sample_indices:
        pred_vel, pred_seis, ssim_vel, ssim_seis = compute_predictions_for_sample(
            sample_idx,
            MODELS_TO_PLOT,
            models_dict,
            test_dataset,
            input_enc,
            output_enc,
            forward_model,
            stats,
            output_shape,
        )
        all_predictions.append((pred_vel, pred_seis, ssim_vel, ssim_seis))

        # Track global seismic limits
        for _, s_2d in pred_seis:
            global_seismic_min = min(global_seismic_min, s_2d.min())
            global_seismic_max = max(global_seismic_max, s_2d.max())

    # Fixed velocity scale
    vel_min = 100.0
    vel_max = 900.0

    # Create figure with unified gridspec
    n_samples = len(sample_indices)
    fig_height = 3.6

    fig = plt.figure(figsize=(6.5, fig_height), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=0.5 / 72.0, h_pad=0.5 / 72.0, hspace=0.0, wspace=0.0
    )

    # Gridspec: n_samples rows + 1 colorbar row, 8 columns (4 velocity + 4 seismic)
    gs = fig.add_gridspec(
        n_samples + 1,
        8,
        width_ratios=[1.2, 1.2, 1.2, 1.2, 0.8, 0.8, 0.8, 0.8],
        height_ratios=[1.0] * n_samples + [0.1],
        hspace=0.04,
        wspace=0.02,
        left=0.02,
        right=0.98,
        top=0.96,
        bottom=0.02,
    )

    # Create parent axes for shared labels (velocity models)
    ax_vel_parent = fig.add_subplot(gs[:-1, 0:4], frameon=False)
    ax_vel_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_vel_parent.set_xlabel("Width (m)", labelpad=-8)
    ax_vel_parent.set_ylabel("Depth (m)", labelpad=-8)
    ax_vel_parent.set_title("FWI Velocity Model Reconstructions", pad=12)

    # Create parent axes for shared labels (seismic transforms)
    ax_seis_parent = fig.add_subplot(gs[:-1, 4:8], frameon=False)
    ax_seis_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_seis_parent.set_xlabel("Frequency", labelpad=-8)
    ax_seis_parent.set_ylabel("Time", labelpad=-8)
    ax_seis_parent.set_title("FWI Seismic Transform Re-simulations", pad=12)

    # Plot all samples
    for row_idx, (pred_vel, pred_seis, ssim_vel, ssim_seis) in enumerate(
        all_predictions
    ):
        # Create SSIM lookup dicts for this sample
        ssim_vel_dict = {name: score for name, score in ssim_vel}
        ssim_seis_dict = {name: score for name, score in ssim_seis}

        # Plot velocity models
        for col_idx, (model_name, u_2d) in enumerate(pred_vel):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(
                u_2d,
                cmap="magma_r",
                vmin=vel_min,
                vmax=vel_max,
                origin="upper",
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])

            # Add model label only on first row
            if row_idx == 0:
                ax.text(
                    0.5,
                    1.05,
                    display_name(model_name),
                    transform=ax.transAxes,
                    fontsize=6,
                    ha="center",
                    va="bottom",
                )

            # Add SSIM annotation (skip ground truth)
            # if model_name != "Ground Truth":
            #     ssim_val = ssim_vel_dict.get(model_name, None)
            #     if ssim_val is not None:
            #         ax.text(
            #             0.95,
            #             0.95,
            #             f"SSIM: {ssim_val:.3f}",
            #             transform=ax.transAxes,
            #             fontsize=5,
            #             color="white",
            #             verticalalignment="top",
            #             horizontalalignment="right",
            #             bbox=dict(
            #                 boxstyle="round,pad=0.3",
            #                 facecolor="black",
            #                 alpha=0.7,
            #                 edgecolor="none",
            #             ),
            #         )

        # Plot seismic transforms
        for col_idx, (model_name, s_2d) in enumerate(pred_seis):
            ax = fig.add_subplot(gs[row_idx, col_idx + 4])
            ax.imshow(
                s_2d,
                cmap="turbo",
                aspect="auto",
                origin="lower",
                vmin=global_seismic_min,
                vmax=global_seismic_max,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("auto")

            # Add model label only on first row
            if row_idx == 0:
                ax.text(
                    0.5,
                    1.05,
                    display_name(model_name),
                    transform=ax.transAxes,
                    fontsize=6,
                    ha="center",
                    va="bottom",
                )

            # Add SSIM annotation (skip ground truth)
            # if model_name != "Ground Truth":
            #     ssim_val = ssim_seis_dict.get(model_name, None)
            #     if ssim_val is not None:
            #         ax.text(
            #             0.95,
            #             0.95,
            #             f"SSIM: {ssim_val:.3f}",
            #             transform=ax.transAxes,
            #             fontsize=5,
            #             color="white",
            #             verticalalignment="top",
            #             horizontalalignment="right",
            #             bbox=dict(
            #                 boxstyle="round,pad=0.3",
            #                 facecolor="black",
            #                 alpha=0.7,
            #                 edgecolor="none",
            #             ),
            #         )

    # Add horizontal colorbars at the bottom
    # Velocity colorbar
    cax_vel = fig.add_subplot(gs[-1, 0:4])
    norm_velocity = mpl.colors.Normalize(vmin=vel_min, vmax=vel_max)
    scalar_mappable_vel = mpl.cm.ScalarMappable(norm=norm_velocity, cmap="magma_r")
    scalar_mappable_vel.set_array([])
    cbar_vel = fig.colorbar(
        scalar_mappable_vel, cax=cax_vel, orientation="horizontal", use_gridspec=True
    )
    cbar_vel.set_label("Velocity (m/s)")
    # cbar_vel.ax.invert_xaxis()

    # Seismic colorbar
    cax_seis = fig.add_subplot(gs[-1, 4:8])
    norm_seismic = mpl.colors.Normalize(
        vmin=global_seismic_min, vmax=global_seismic_max
    )
    scalar_mappable_seis = mpl.cm.ScalarMappable(norm=norm_seismic, cmap="turbo")
    scalar_mappable_seis.set_array([])
    cbar_seis = fig.colorbar(
        scalar_mappable_seis, cax=cax_seis, orientation="horizontal", use_gridspec=True
    )
    cbar_seis.set_label("Amplitude")
    cbar_seis.ax.invert_xaxis()

    # Save outputs
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        base_path = os.path.join(args.results_dir, "fwi_publication_multi")
        fig.savefig(f"{base_path}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
        print(f"SUCCESS: Saved multi-sample figure → {base_path}.pdf/.png")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
