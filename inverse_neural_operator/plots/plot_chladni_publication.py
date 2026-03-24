"""
Publication-quality plotting script for Chladni 2D inverse problem results.

Creates publication-ready figures with:
- 6.5 inch width for journal publications
- 5-7pt sans serif fonts
- Professional styling and layout
- High-resolution output (300 DPI)
- 2x3 + 2x3 grid layout (6 models)
- 2D heatmaps for both inputs and outputs

To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_chladni_publication
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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


def plot_comparison(sample_idx, models_to_plot, predictions, meta, save_dir):
    """Create the publication figure."""
    fig = plt.figure(figsize=(6.5, 2.0), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=0.5 / 72.0, h_pad=0.5 / 72.0, hspace=0.0, wspace=0.0
    )

    gs = fig.add_gridspec(
        2,
        8,
        width_ratios=[1, 1, 1, 0.05, 1, 1, 1, 0.05],
        hspace=0.0,
        wspace=0.0,
        left=0,
        right=1,
        top=1,
        bottom=0,
    )

    # Parent axes for labels
    ax_left = fig.add_subplot(gs[:, 0:3], frameon=False)
    ax_left.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_left.set_xlabel(r"$x$", labelpad=2)
    ax_left.set_ylabel(r"$y$", labelpad=-2)
    ax_left.set_title("Predicted Force Fields")

    ax_right = fig.add_subplot(gs[:, 4:7], frameon=False)
    ax_right.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_right.set_xlabel(r"$x$", labelpad=2)
    ax_right.set_ylabel(r"$y$", labelpad=-2)
    ax_right.set_title("Re-simulated Chladni Patterns")

    # Spacers
    fig.add_subplot(gs[:, 3]).axis("off")
    fig.add_subplot(gs[:, 7]).axis("off")

    u_true = meta["u_true"]
    s_true = meta["s_true"]

    # Compute colorbar limits (excluding iFNO to avoid extreme values)
    u_min = u_true.min()
    u_max = u_true.max()
    s_min = s_true.min()
    s_max = s_true.max()

    for model_name in models_to_plot:
        # Skip iFNO when computing color limits
        if model_name == "ifno":
            continue

        preds = predictions.get(model_name, {})
        if "inputs" in preds and preds["inputs"].size > 0:
            u_min = min(u_min, preds["inputs"].min())
            u_max = max(u_max, preds["inputs"].max())
        if "outputs" in preds and preds["outputs"].size > 0:
            s_min = min(s_min, preds["outputs"].min())
            s_max = max(s_max, preds["outputs"].max())

    # Reshape to 2D for imshow
    grid_size = int(np.sqrt(len(u_true)))
    tick_positions = [0, grid_size // 2, grid_size - 1]
    u_true_2d = u_true.reshape(grid_size, grid_size)
    s_true_2d = s_true.reshape(grid_size, grid_size)

    im_left = None
    im_right = None

    # Plot inputs (left grid - force fields)
    panel_models = list(models_to_plot[:5])

    for idx, model_name in enumerate(panel_models):
        i = idx // 3
        j = idx % 3
        ax = fig.add_subplot(gs[i, j])

        preds = predictions.get(model_name, {})
        u_samples = preds.get("inputs", np.empty((0,)))

        if u_samples.size > 0:
            u_mean = u_samples.mean(axis=0)
            u_mean_2d = u_mean.reshape(grid_size, grid_size)

            # Clip iFNO values to computed range
            if model_name == "ifno":
                u_mean_2d = np.clip(u_mean_2d, u_min, u_max)

            im_left = ax.imshow(
                u_mean_2d,
                cmap="RdBu_r",
                vmin=u_min,
                vmax=u_max,
                origin="lower",
                aspect="equal",
            )
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            show_labels = i == 1 and j == 0
            ax.tick_params(
                bottom=True,
                left=True,
                top=False,
                right=False,
                labelbottom=show_labels,
                labelleft=show_labels,
            )
            ax.text(
                0.05,
                0.95,
                display_name(model_name),
                transform=ax.transAxes,
                fontsize=5,
                color="white",
                va="top",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="black",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )
        else:
            ax.axis("off")

    # Plot outputs (right grid - Chladni patterns)
    for idx, model_name in enumerate(panel_models):
        i = idx // 3
        j = idx % 3
        ax = fig.add_subplot(gs[i, j + 4])

        preds = predictions.get(model_name, {})
        s_samples = preds.get("outputs", np.empty((0,)))

        if s_samples.size > 0:
            s_mean = s_samples.mean(axis=0)
            s_mean_2d = s_mean.reshape(grid_size, grid_size)

            # Clip iFNO values to computed range
            if model_name == "ifno":
                s_mean_2d = np.clip(s_mean_2d, s_min, s_max)

            im_right = ax.imshow(
                s_mean_2d,
                cmap="viridis",
                vmin=s_min,
                vmax=s_max,
                origin="lower",
                aspect="equal",
            )
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            show_labels = i == 1 and j == 0
            ax.tick_params(
                bottom=True,
                left=True,
                top=False,
                right=False,
                labelbottom=show_labels,
                labelleft=show_labels,
            )
            ax.text(
                0.05,
                0.95,
                display_name(model_name),
                transform=ax.transAxes,
                fontsize=5,
                color="white",
                va="top",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="black",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )
        else:
            ax.axis("off")

    # Ground truth occupies last slot (index 5)
    gt_idx = len(panel_models)
    gt_row = gt_idx // 3
    gt_col = gt_idx % 3

    ax_gt_input = fig.add_subplot(gs[gt_row, gt_col])
    im_left = ax_gt_input.imshow(
        u_true_2d,
        cmap="RdBu_r",
        vmin=u_min,
        vmax=u_max,
        origin="lower",
        aspect="equal",
    )
    ax_gt_input.set_xticks([])
    ax_gt_input.set_yticks([])
    ax_gt_input.text(
        0.05,
        0.95,
        "Ground Truth",
        transform=ax_gt_input.transAxes,
        fontsize=5,
        color="white",
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="none"
        ),
    )

    ax_gt_output = fig.add_subplot(gs[gt_row, gt_col + 4])
    im_right = ax_gt_output.imshow(
        s_true_2d,
        cmap="viridis",
        vmin=s_min,
        vmax=s_max,
        origin="lower",
        aspect="equal",
    )
    ax_gt_output.set_xticks([])
    ax_gt_output.set_yticks([])
    ax_gt_output.text(
        0.05,
        0.95,
        "Ground Truth",
        transform=ax_gt_output.transAxes,
        fontsize=5,
        color="white",
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="none"
        ),
    )

    # Add colorbars
    if im_left is not None:
        cax_left = fig.add_subplot(gs[:, 3])
        fig.colorbar(im_left, cax=cax_left, use_gridspec=True)

    if im_right is not None:
        cax_right = fig.add_subplot(gs[:, 7])
        fig.colorbar(im_right, cax=cax_right, use_gridspec=True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, f"chladni_comparison_sample_{sample_idx}")
        fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    setup_publication_style(figsize=(6.5, 2.0))
    mpl.rcParams.update(
        {
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "xtick.major.pad": 1.0,
            "ytick.major.pad": 1.0,
        }
    )

    parser = argparse.ArgumentParser(
        description="Create publication-quality Chladni plots."
    )
    parser.add_argument(
        "--log_dir", type=str, default="/store/at46867/b2b_operator_inverse"
    )
    parser.add_argument("--results_dir", type=str, default="results/chladni_2d")
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
    dataset_name = "chladni_2d"
    models_root = args.log_dir
    log_dir = os.path.join(models_root, dataset_name)
    if not os.path.exists(log_dir):
        alt_root = os.path.join(models_root, "models")
        alt_log_dir = os.path.join(alt_root, dataset_name)
        if os.path.exists(alt_log_dir):
            models_root = alt_root
            log_dir = alt_log_dir
    if not os.path.exists(log_dir):
        print(f"ERROR: log directory not found: {log_dir}")
        raise SystemExit(1)

    params = find_params(log_dir, list(INVERSE_MODELS), args.seed)
    cache_path = os.path.join(results_dir, "plot_cache.json")
    cache_key = f"seed_{args.seed}"
    cache_metadata = {"dataset": params.dataset, "seed": int(args.seed)}
    normalized_root = os.path.normpath(models_root)
    if os.path.basename(normalized_root) == "models":
        models_base_dir = os.path.dirname(normalized_root)
    else:
        models_base_dir = models_root
    test_dataset, dataset_info = load_dataset(
        params.dataset, params, device, split="test", return_info=True
    )
    repo_root = Path(__file__).resolve().parents[2]

    print("Loading models...")
    models_dict, input_enc, output_enc = load_all_models(
        models_base_dir, params.dataset, INVERSE_MODELS, seed=args.seed, device=device
    )

    if not models_dict:
        print("ERROR: no models loaded")
        raise SystemExit(1)

    forward_model = load_forward_model(log_dir, args.seed, device=device)
    output_transform = make_output_transform(forward_model, output_enc)

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
                n_samples=1,
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
