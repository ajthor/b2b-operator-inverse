#!/usr/bin/env python
"""Publication-ready visualization for the Elastic Plate inverse problem.

Generates a compact figure with predicted forcing functions on the left
and re-simulated displacement fields on the right, mimicking the layout used
for the wave scattering publication figures.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plots.utils.model_utils import (  # noqa: E402
    collect_predictions,
    load_all_models,
    select_models_and_sample,
)
from plots.utils.plot_utils import (  # noqa: E402
    display_name,
    find_params,
    load_forward_model,
    make_output_transform,
    setup_publication_style,
)

from data.load_dataset import load_dataset  # noqa: E402

device = "cpu"

INVERSE_MODELS: Tuple[str, ...] = (
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


def _create_publication_figure():
    """Create the shared figure layout (2×3 model grids + colorbar)."""
    fig = plt.figure(figsize=(6.5, 2.4), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=0.6 / 72.0,
        h_pad=0.6 / 72.0,
        hspace=0.0,
        wspace=0.0,
    )

    gs = fig.add_gridspec(
        2,
        7,
        width_ratios=[1, 1, 1, 0.15, 1, 1, 1],
        left=0.02,
        right=0.98,
        top=0.95,
        bottom=0.12,
    )

    # Parent axes just for shared labels / titles
    ax_force_parent = fig.add_subplot(gs[:, 0:3], frameon=False)
    ax_force_parent.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    ax_force_parent.set_xlabel("Force Magnitude", labelpad=2)
    ax_force_parent.set_ylabel("Boundary Position (y)", labelpad=2)
    ax_force_parent.set_title("Predicted Forcing Functions")

    ax_disp_parent = fig.add_subplot(gs[:, 4:7], frameon=False)
    ax_disp_parent.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    ax_disp_parent.set_xlabel("x", labelpad=2)
    ax_disp_parent.set_ylabel("y", labelpad=2)
    ax_disp_parent.set_title("Re-simulated Displacement Fields")

    force_axes: List[plt.Axes] = []
    disp_axes: List[plt.Axes] = []

    for row in range(2):
        for col in range(3):
            force_ax = fig.add_subplot(gs[row, col])
            force_axes.append(force_ax)

            disp_ax = fig.add_subplot(gs[row, col + 4])
            disp_axes.append(disp_ax)

    cax = fig.add_subplot(gs[:, 3])
    return fig, force_axes, disp_axes, cax


def _prepare_force_data(meta: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and sort boundary coordinates / forcing values."""
    coords = meta["x"].reshape(-1, 2)
    values = meta["u_true"].reshape(-1)

    order = np.argsort(coords[:, 1])
    y = coords[order, 1]
    true_force = values[order]

    return y, true_force, order


def _make_grid(coords: np.ndarray, resolution: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a regular interpolation grid covering the supplied coordinates."""
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    return np.meshgrid(xi, yi)


def _hole_mask(X: np.ndarray, Y: np.ndarray, center: Tuple[float, float] = (0.5, 0.5), radius: float = 0.25):
    """Mask for the central hole in the elastic plate."""
    return (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius**2


def _interpolate_field(
    coords: np.ndarray,
    values: np.ndarray,
    grid: Tuple[np.ndarray, np.ndarray],
    mask_center: Tuple[float, float] = (0.5, 0.5),
    mask_radius: float = 0.25,
) -> np.ndarray:
    """Interpolate scattered displacement data onto a regular grid."""
    Xi, Yi = grid
    Zi = griddata(coords, values, (Xi, Yi), method="cubic")
    hole = _hole_mask(Xi, Yi, center=mask_center, radius=mask_radius)
    Zi[hole] = np.nan
    return Zi


def _plot_force_panel(
    ax: plt.Axes,
    y: np.ndarray,
    true_force: np.ndarray,
    predictions: np.ndarray | None,
    label: str,
    x_limits: Tuple[float, float],
):
    """Render a single forcing subplot."""
    ax.plot(true_force, y, color="0.7", linewidth=0.8, label="Ground Truth")

    if predictions is not None and predictions.size:
        finite = np.where(np.isfinite(predictions), predictions, np.nan)
        mean_pred = np.nanmean(finite, axis=0)
        ax.plot(mean_pred, y, color="#1f77b4", linewidth=1.2, label="Prediction")

        if predictions.shape[0] > 1:
            lower = np.nanpercentile(finite, 10, axis=0)
            upper = np.nanpercentile(finite, 90, axis=0)
            ax.fill_betweenx(y, lower, upper, color="#1f77b4", alpha=0.2)

    ax.tick_params(direction="out", length=2)
    xmin, xmax = x_limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(y.min(), y.max())
    ax.grid(True, alpha=0.2, linewidth=0.3)

    ax.text(
        0.03,
        0.95,
        label,
        transform=ax.transAxes,
        fontsize=6,
        color="white",
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.65, edgecolor="none"),
    )


def _plot_displacement_panel(
    ax: plt.Axes,
    Xi: np.ndarray,
    Yi: np.ndarray,
    field: np.ndarray,
    norm: Normalize,
    label: str,
):
    """Render a single displacement subplot."""
    im = ax.imshow(
        field,
        extent=(Xi.min(), Xi.max(), Yi.min(), Yi.max()),
        origin="lower",
        cmap="RdBu_r",
        norm=norm,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.text(
        0.03,
        0.95,
        label,
        transform=ax.transAxes,
        fontsize=6,
        color="white",
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.65, edgecolor="none"),
    )
    return im


def plot_comparison(
    sample_idx: int,
    models_to_plot: Iterable[str],
    predictions: Dict[str, Dict[str, np.ndarray]],
    meta: Dict[str, np.ndarray],
    save_dir: str,
):
    """Render the publication figure."""
    y_coords, true_force, order = _prepare_force_data(meta)

    disp_coords = meta["y"].reshape(-1, 2)
    Xi, Yi = _make_grid(disp_coords, resolution=200)

    true_disp = meta["s_true"].reshape(-1)
    true_field = _interpolate_field(disp_coords, true_disp, (Xi, Yi))

    # Gather fields for consistent color scaling
    displacement_fields: Dict[str, np.ndarray] = {}
    displacement_fields["Ground Truth"] = true_field

    # Precompute predicted forces for annotation
    force_predictions: Dict[str, np.ndarray] = {}

    for model_name in models_to_plot[:MAX_MODELS]:
        preds = predictions.get(model_name, {})
        input_samples = preds.get("inputs")
        output_samples = preds.get("outputs")

        if input_samples is not None and input_samples.size:
            force_predictions[model_name] = input_samples[:, order]

        if output_samples is not None and output_samples.size:
            mean_output = output_samples.mean(axis=0)
            field = _interpolate_field(disp_coords, mean_output, (Xi, Yi))
            displacement_fields[model_name] = field

    # Determine color limits (ignore NaNs)
    finite_list = [field[~np.isnan(field)] for field in displacement_fields.values() if np.any(~np.isnan(field))]
    finite_vals = np.concatenate(finite_list) if finite_list else np.array([1.0])
    vmax = float(np.percentile(np.abs(finite_vals), 99)) if finite_vals.size else 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)

    force_min = float(true_force.min())
    force_max = float(true_force.max())
    for arr in force_predictions.values():
        if arr.size:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                force_min = min(force_min, float(finite.min()))
                force_max = max(force_max, float(finite.max()))
    if np.isclose(force_min, force_max):
        delta = max(1e-3, abs(force_min) * 0.1 or 1e-2)
        force_min -= delta
        force_max += delta
    force_limits = (force_min, force_max)

    fig, force_axes, disp_axes, cax = _create_publication_figure()

    # Plot models (up to 5)
    im = None

    for idx, model_name in enumerate(models_to_plot[:MAX_MODELS]):
        label = display_name(model_name)
        force_ax = force_axes[idx]
        disp_ax = disp_axes[idx]

        pred_forces = force_predictions.get(model_name)
        if pred_forces is not None:
            samples_ordered = pred_forces
        else:
            samples_ordered = None

        _plot_force_panel(force_ax, y_coords, true_force, samples_ordered, label, force_limits)

        disp_field = displacement_fields.get(model_name)
        if disp_field is not None:
            im = _plot_displacement_panel(disp_ax, Xi, Yi, disp_field, norm, label)
        else:
            disp_ax.set_axis_off()
            disp_ax.text(
                0.5,
                0.5,
                "N/A",
                transform=disp_ax.transAxes,
                ha="center",
                va="center",
                fontsize=6,
                color="0.3",
            )

    # Sixth slot: ground truth
    ax_force_gt = force_axes[5]
    ax_disp_gt = disp_axes[5]

    _plot_force_panel(ax_force_gt, y_coords, true_force, None, "Ground Truth", force_limits)
    im = _plot_displacement_panel(ax_disp_gt, Xi, Yi, true_field, norm, "Ground Truth")

    if im is not None:
        cbar = fig.colorbar(im, cax=cax, use_gridspec=True)
        cbar.set_label("Displacement")

    for idx, ax in enumerate(force_axes):
        if idx < 3:
            ax.set_xticklabels([])
        if idx % 3 != 0:
            ax.set_yticklabels([])

    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f"elastic_plate_sample_{sample_idx}_publication.pdf")
    png_path = os.path.join(save_dir, f"elastic_plate_sample_{sample_idx}_publication.png")
    plt.savefig(pdf_path, format="pdf", dpi=300)
    plt.savefig(png_path, format="png", dpi=300)
    print(f"✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")
    plt.close(fig)


def main():
    setup_publication_style(figsize=(6.5, 2.4))

    parser = argparse.ArgumentParser(description="Elastic Plate publication figure generator.")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--results_dir", type=str, default="results/elastic_publication")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--forward_model", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = "elastic_plate"
    log_dir = os.path.join(args.log_dir, dataset)
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    print("Loading dataset...")
    params = find_params(log_dir, INVERSE_MODELS, args.seed)
    test_dataset, dataset_info = load_dataset(
        params.dataset,
        params,
        device,
        split="test",
        return_info=True,
    )
    print(f"✓ Loaded {len(test_dataset)} test samples")

    print("\nLoading models...")
    models_dict, input_enc, output_enc = load_all_models(
        log_dir,
        dataset_info,
        INVERSE_MODELS,
        seed=args.seed,
        device=device,
    )
    if not models_dict:
        raise RuntimeError("No trained models found for publication plotting.")

    forward_model_name = args.forward_model or getattr(params, "forward_model", "b2b_nonlinear")
    forward_model = load_forward_model(
        log_dir,
        args.seed,
        forward_model_name=forward_model_name,
        device=device,
    )
    output_transform = make_output_transform(forward_model, output_enc)

    models_to_plot, sample_idx = select_models_and_sample(
        test_dataset,
        models_dict,
        input_enc,
        output_enc,
        max_models=MAX_MODELS,
        sample_index=args.sample_index,
        device=device,
    )

    print("\nCollecting predictions...")
    predictions, meta = collect_predictions(
        test_dataset[sample_idx],
        models_to_plot,
        models_dict,
        input_enc,
        output_enc,
        output_transform=output_transform,
        n_samples_per_model=N_SAMPLES,
        device=device,
    )

    print("Rendering publication figure...")
    plot_comparison(sample_idx, models_to_plot, predictions, meta, args.results_dir)
    print(f"SUCCESS: Created publication figure at {args.results_dir}")


if __name__ == "__main__":
    main()
