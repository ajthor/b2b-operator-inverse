"""
Multi-sample publication-quality plotting script for the FWI inverse problem.

Creates a single figure that stacks three instances of the standard FWI
publication layout vertically, each showcasing a different test sample.
"""

import argparse
import json
import os
import random
from typing import Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.load_dataset import load_dataset
from plots.plot_fwi_publication import compute_sample_ssim_scores
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


def denormalize_velocity(u_normalized: torch.Tensor, residual_min: float, residual_max: float) -> np.ndarray:
    """Map normalized velocity residuals back to physical units (m/s)."""
    u_np = u_normalized.squeeze(-1).cpu().numpy()
    u_velocity = ((u_np + 1) / 2) * (residual_max - residual_min) + residual_min
    return u_velocity.reshape(24, 48)


def reshape_seismic(s: torch.Tensor, output_shape: Tuple[int, int]) -> np.ndarray:
    """Reshape flattened seismic transform into 2D grid."""
    s_np = s.squeeze(-1).cpu().numpy()
    return s_np.reshape(*output_shape)


def render_sample_panel(
    subfig: mpl.figure.SubFigure,
    sample_idx: int,
    sample_label: str,
    models_to_plot: Iterable[str],
    models_dict,
    test_dataset,
    input_enc,
    output_enc,
    forward_model,
    stats,
    output_shape: Tuple[int, int],
):
    """Render a single FWI sample using the publication layout into the provided subfigure."""
    subfig.set_constrained_layout(True)
    subfig.set_constrained_layout_pads(w_pad=0.5 / 72.0, h_pad=0.5 / 72.0, hspace=0.0, wspace=0.0)

    gs = subfig.add_gridspec(
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

    # Parent axes for shared labels
    ax_left_parent = subfig.add_subplot(gs[:, 0:2], frameon=False)
    ax_left_parent.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    ax_left_parent.set_xlabel("x", labelpad=-8)
    ax_left_parent.set_ylabel("y", labelpad=-8)
    ax_left_parent.set_title("Velocity Model Reconstructions")

    ax_right_parent = subfig.add_subplot(gs[:, 3:5], frameon=False)
    ax_right_parent.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    ax_right_parent.set_xlabel("Frequency", labelpad=-8)
    ax_right_parent.set_ylabel("Time", labelpad=-8)
    ax_right_parent.set_title("Seismic Transform Predictions")

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

            if forward_model is not None:
                forward_model.eval()
                alpha, _ = input_enc.compute_coefficients(X.unsqueeze(0), u_pred.unsqueeze(0))
                beta_pred = forward_model.forward(alpha)
                s_resim = output_enc(Y.unsqueeze(0), beta_pred)
                s_resim_2d = reshape_seismic(s_resim.squeeze(0), output_shape)
                predictions_seismic.append((model_name, s_resim_2d))

    predictions_velocity.append(("Ground Truth", u_true_2d))
    predictions_seismic.append(("Ground Truth", s_observed_2d))

    # Fixed velocity scale and shared seismic limits
    vel_min = 100.0
    vel_max = 900.0

    seismic_min = s_observed_2d.min()
    seismic_max = s_observed_2d.max()
    for _, s_resim_2d in predictions_seismic:
        seismic_min = min(seismic_min, s_resim_2d.min())
        seismic_max = max(seismic_max, s_resim_2d.max())

    # Velocity panels
    norm_velocity = mpl.colors.Normalize(vmin=vel_min, vmax=vel_max)

    for idx, (model_name, u_2d) in enumerate(predictions_velocity):
        i = idx // 2
        j = idx % 2
        ax = subfig.add_subplot(gs[i, j])
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

    cax_left = subfig.add_subplot(gs[:, 2])
    scalar_mappable = mpl.cm.ScalarMappable(norm=norm_velocity, cmap="magma_r")
    scalar_mappable.set_array([])
    cbar_left = subfig.colorbar(scalar_mappable, cax=cax_left, use_gridspec=True)
    cbar_left.set_label("Velocity (m/s)")
    cbar_left.ax.invert_yaxis()

    # Seismic panels
    for idx, (model_name, s_2d) in enumerate(predictions_seismic):
        i = idx // 2
        j = idx % 2
        ax = subfig.add_subplot(gs[i, j + 3])
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

    cax_right = subfig.add_subplot(gs[:, 5])
    cbar_right = subfig.colorbar(im, cax=cax_right, use_gridspec=True)
    cbar_right.set_label("Amplitude")

    subfig.suptitle(sample_label, fontsize=7, y=1.02)


def main():
    setup_publication_style()

    parser = argparse.ArgumentParser(
        description="Create multi-sample publication-quality FWI plots."
    )
    parser.add_argument(
        "--log_dir", type=str, default="/store/at46867/b2b_operator_inverse"
    )
    parser.add_argument("--results_dir", type=str, default="results/fwi")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="*",
        help="Specific sample indices to plot (expects three values).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to plot when sample_indices not provided.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    log_dir = os.path.join(args.log_dir, "fwi")
    if not os.path.exists(log_dir):
        print(f"ERROR: log directory not found: {log_dir}")
        raise SystemExit(1)

    # Load normalization statistics
    stats_path = os.path.join(os.path.dirname(__file__), "../data/fwi_stats.json")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    print(
        f"✓ Loaded normalization stats: velocity range [{stats['models_min']:.2f}, {stats['models_max']:.2f}]"
    )

    params = find_params(log_dir, MODELS_TO_PLOT, args.seed)
    test_dataset, dataset_info = load_dataset(
        params.dataset, params, device, split="test", return_info=True
    )
    print(f"✓ Loaded {len(test_dataset)} test samples")

    print("Loading models...")
    models_dict, input_enc, output_enc = load_all_models(
        log_dir, dataset_info, MODELS_TO_PLOT, seed=args.seed, device=device
    )

    if not models_dict:
        print("ERROR: no models loaded")
        raise SystemExit(1)

    forward_model = load_forward_model(
        log_dir, args.seed, forward_model_name="b2b_nonlinear", device=device
    )

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
    score_dict = {idx: score for idx, score in sample_scores}

    if args.sample_indices:
        sample_indices = args.sample_indices
        if len(sample_indices) != args.num_samples:
            print(
                f"⚠ Provided {len(sample_indices)} sample indices, but num_samples={args.num_samples}. "
                f"Using provided indices and ignoring num_samples."
            )
    else:
        if not sample_scores:
            raise ValueError("Unable to compute SSIM-based sample selection.")
        sorted_scores = sorted(sample_scores, key=lambda x: x[1], reverse=True)
        num = min(args.num_samples, len(sorted_scores))
        if num == 1:
            sample_indices = [sorted_scores[0][0]]
        elif num == 2:
            sample_indices = [sorted_scores[0][0], sorted_scores[-1][0]]
        else:
            best = sorted_scores[0][0]
            worst = sorted_scores[-1][0]
            median = sorted_scores[len(sorted_scores) // 2][0]
            sample_indices = [best, median, worst][:num]

    print(f"✓ Selected samples: {sample_indices}")

    # Prepare labels for each sample
    sample_labels = []
    for idx in sample_indices:
        ssim_val = score_dict.get(idx, None)
        if ssim_val is not None:
            label = f"Sample {idx} • median SSIM={ssim_val:.4f}"
        else:
            label = f"Sample {idx}"
        sample_labels.append(label)

    # Create figure with stacked subfigures
    n_samples = len(sample_indices)
    height_per_sample = 1.9
    fig_height = height_per_sample * n_samples + 0.3 * (n_samples - 1)
    fig = plt.figure(figsize=(6.5, fig_height), layout="constrained")
    outer_gs = fig.add_gridspec(n_samples, 1, hspace=0.4)

    for row, (sample_idx, label) in enumerate(zip(sample_indices, sample_labels)):
        subfig = fig.add_subfigure(outer_gs[row, 0])
        render_sample_panel(
            subfig=subfig,
            sample_idx=sample_idx,
            sample_label=label,
            models_to_plot=MODELS_TO_PLOT,
            models_dict=models_dict,
            test_dataset=test_dataset,
            input_enc=input_enc,
            output_enc=output_enc,
            forward_model=forward_model,
            stats=stats,
            output_shape=output_shape,
        )

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
