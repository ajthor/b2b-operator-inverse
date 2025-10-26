"""
Publication-quality plotting script for Chladni 2D inverse problem results.

Creates publication-ready figures with:
- 6.5 inch width for journal publications
- 5-7pt sans serif fonts
- Professional styling and layout
- High-resolution output (300 DPI)
- Optimized 2D heatmap visualization with shared colorbars

To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_chladni_publication
"""

import os
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random

import torch

from data.load_dataset import load_dataset
from models.load_model import load_models
from b2b.load_model import load_forward_model

device = "cpu"

# Available inverse models - all models for ranking
INVERSE_MODELS = [
    "linear",
    "linear_inverse",
    "nonlinear",
    "variational_autoencoder",
    "inn_affine",
    "inn_additive",
    "cinn_affine",
    "cinn_additive",
    "conditional_realnvp",
    "mixture_density_network",
]


def setup_publication_style():
    """Configure matplotlib for publication-quality output."""
    # Publication style configuration
    plt.style.use("default")  # Start with clean default

    # Font configuration
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    mpl.rcParams["font.size"] = 6
    mpl.rcParams["axes.labelsize"] = 6
    mpl.rcParams["axes.titlesize"] = 7
    mpl.rcParams["xtick.labelsize"] = 5
    mpl.rcParams["ytick.labelsize"] = 5
    mpl.rcParams["legend.fontsize"] = 5

    # Line and marker configuration
    mpl.rcParams["lines.linewidth"] = 1.0
    mpl.rcParams["lines.markersize"] = 3

    # Figure and subplot configuration
    mpl.rcParams["figure.figsize"] = [6.5, 3.0]  # Adjusted height for 2D images
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.05

    # Grid and axes
    mpl.rcParams["axes.grid"] = False  # Disable grid for 2D heatmaps
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["xtick.major.width"] = 0.8
    mpl.rcParams["ytick.major.width"] = 0.8

    # Colorbar
    mpl.rcParams["axes.axisbelow"] = True


def _model_colors():
    """Return consistent color map for models."""
    return {
        "color_true": "#1f77b4",
        "color_observed": "#2E8B57",
        "models": {
            "linear": "#8c564b",
            "linear_inverse": "#e377c2",
            "nonlinear": "#d62728",
            "variational_autoencoder": "#ff7f0e",
            "inn_affine": "#2ca02c",
            "cinn_affine": "#17becf",
            "conditional_realnvp": "#9467bd",
            "mixture_density_network": "#bcbd22",
        },
    }


def _display_name(model_name: str) -> str:
    """Map internal model names to display names for annotations."""
    mapping = {
        "linear": "Linear",
        "linear_inverse": "Linear-Inv",
        "nonlinear": "Nonlinear",
        "inn_affine": "INN-Affine",
        "inn_additive": "INN-Add",
        "cinn_affine": "cINN-Affine",
        "cinn_additive": "cINN-Add",
        "variational_autoencoder": "cVAE",
        "conditional_realnvp": "RealNVP",
        "mixture_density_network": "MDN",
    }
    if model_name in mapping:
        return mapping[model_name]
    # Fallback: prettify by replacing underscores and capitalizing words
    return " ".join([p.capitalize() for p in model_name.split("_")])


def _plot_2d_field(
    ax,
    field_2d,
    cmap,
    vmin,
    vmax,
    annotation=None,
    annotation_color="white",
):
    """Plot a 2D field as a heatmap on the given axis.

    Args:
        ax: Matplotlib axis
        field_2d: 2D numpy array to plot (h, w)
        cmap: Colormap to use
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        annotation: Text to show in corner (e.g., model name)
        annotation_color: Color for annotation text

    Returns:
        Image object for colorbar
    """
    # Plot the field
    im = ax.imshow(
        field_2d,
        cmap=cmap,
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )

    # Remove ticks and labels for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    # Add model annotation in upper-left corner
    if annotation is not None:
        ax.text(
            0.05,
            0.95,
            str(annotation),
            transform=ax.transAxes,
            fontsize=6,
            color=annotation_color,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="black",
                alpha=0.7,
                edgecolor="none",
            ),
        )

    return im


def _create_unified_figure():
    """Create the publication-style figure and return fig, left_axes, right_axes, cbar_axes.

    Uses single unified gridspec with 2 rows x 8 columns:
    - Columns 0-2: Left plots (predicted force fields)
    - Column 3: Left colorbar
    - Columns 4-6: Right plots (re-simulated displacement fields)
    - Column 7: Right colorbar

    Matches spacing exactly from test_gridspec.py
    """
    # Create figure with constrained layout for optimal spacing
    fig = plt.figure(figsize=(6.5, 2.0), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=0.5 / 72.0, h_pad=0.5 / 72.0, hspace=0.0, wspace=0.0
    )

    # Single gridspec: 2 rows x 8 columns
    # Columns: [plot, plot, plot, colorbar, plot, plot, plot, colorbar]
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

    # Create parent axes for shared labels (invisible, just for labels)
    ax_left_parent = fig.add_subplot(gs[:, 0:3], frameon=False)
    ax_left_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_left_parent.set_xlabel("x", labelpad=-8)
    ax_left_parent.set_ylabel("y", labelpad=-8)
    ax_left_parent.set_title("Predicted Force Fields")

    ax_right_parent = fig.add_subplot(gs[:, 4:7], frameon=False)
    ax_right_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_right_parent.set_xlabel("x", labelpad=-8)
    ax_right_parent.set_ylabel("y", labelpad=-8)
    ax_right_parent.set_title("Re-simulated Displacement Fields")

    # Create axes for left grid (columns 0-2)
    axes_left = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            axes_left.append(ax)

    # Create axes for right grid (columns 4-6)
    axes_right = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j + 4])
            axes_right.append(ax)

    # Create colorbar axes spanning both rows
    cbar_ax_left = fig.add_subplot(gs[:, 3])
    cbar_ax_right = fig.add_subplot(gs[:, 7])

    return fig, axes_left, axes_right, cbar_ax_left, cbar_ax_right


def rank_models_by_performance(
    test_dataset, models_dict, input_function_encoder, output_function_encoder
):
    """
    Rank models by their average MSE performance across test samples.

    Args:
        test_dataset: Test dataset
        models_dict: Dictionary mapping model_name -> (model, evaluate_fn)
        input_function_encoder: Input function encoder
        output_function_encoder: Output function encoder

    Returns:
        list: List of model names sorted by performance (best first)
    """
    model_mses = {model_name: [] for model_name in models_dict.keys()}

    # Evaluate on first 50 samples
    for idx in range(min(50, len(test_dataset))):
        sample = test_dataset[idx]
        X, u_true, Y, s_observed = sample

        X = X.to(device)
        u_true = u_true.to(device)
        Y = Y.to(device)
        s_observed = s_observed.to(device)

        for model_name, (model, evaluate_fn) in models_dict.items():
            try:
                with torch.no_grad():
                    point = (
                        X.unsqueeze(0),
                        u_true.unsqueeze(0),
                        Y.unsqueeze(0),
                        s_observed.unsqueeze(0),
                    )
                    u_pred, _ = evaluate_fn(
                        model, point, input_function_encoder, output_function_encoder
                    )
                    u_pred = u_pred.squeeze(0)

                    # Compute MSE for input reconstruction
                    mse = torch.mean((u_pred - u_true) ** 2).item()
                    model_mses[model_name].append(mse)
            except:
                pass

    # Compute average MSE for each model and rank
    model_rankings = []
    for model_name, mses in model_mses.items():
        if mses:
            avg_mse = np.mean(mses)
            model_rankings.append((model_name, avg_mse))

    # Sort by MSE (lower is better)
    model_rankings.sort(key=lambda x: x[1])

    return [model_name for model_name, _ in model_rankings]


def select_best_sample(
    test_dataset, models_dict, input_function_encoder, output_function_encoder
):
    """
    Select the best sample based on median MSE across all models.

    Args:
        test_dataset: Test dataset
        models_dict: Dictionary mapping model_name -> (model, evaluate_fn)
        input_function_encoder: Input function encoder
        output_function_encoder: Output function encoder

    Returns:
        int: Index of selected sample
    """
    sample_mses = []

    for idx in range(min(50, len(test_dataset))):  # Check first 50 samples
        sample = test_dataset[idx]
        X, u_true, Y, s_observed = sample

        X = X.to(device)
        u_true = u_true.to(device)
        Y = Y.to(device)
        s_observed = s_observed.to(device)

        mses = []
        for _, (model, evaluate_fn) in models_dict.items():
            try:
                with torch.no_grad():
                    point = (
                        X.unsqueeze(0),
                        u_true.unsqueeze(0),
                        Y.unsqueeze(0),
                        s_observed.unsqueeze(0),
                    )
                    u_pred, _ = evaluate_fn(
                        model, point, input_function_encoder, output_function_encoder
                    )
                    u_pred = u_pred.squeeze(0)

                    # Compute MSE for input reconstruction
                    mse = torch.mean((u_pred - u_true) ** 2).item()
                    mses.append(mse)
            except:
                pass

        if mses:
            sample_mses.append((idx, np.median(mses)))

    # Sort by median MSE and select the middle one (representative difficulty)
    sample_mses.sort(key=lambda x: x[1])
    median_idx = len(sample_mses) // 2

    return sample_mses[median_idx][0]


def load_all_models(log_dir, dataset_info, seed=1):
    """
    Load all available inverse models.

    Returns:
        tuple: (models_dict, input_function_encoder, output_function_encoder)
    """
    models_dict = {}
    input_function_encoder = None
    output_function_encoder = None

    for model_name in INVERSE_MODELS:
        model_log_dir = os.path.join(log_dir, model_name, f"seed_{seed}")

        # Check if model exists
        if not os.path.exists(os.path.join(model_log_dir, "params.pth")):
            print(f"  Skipping {model_name} - not found")
            continue

        try:
            # Load model parameters
            params = torch.load(
                os.path.join(model_log_dir, "params.pth"), weights_only=False
            )

            # Load models
            inp_enc, out_enc, model, evaluate_fn = load_models(
                log_dir=model_log_dir,
                dataset_info=dataset_info,
                params=params,
                device=device,
            )

            # Store encoders from first model (they should all be the same)
            if input_function_encoder is None:
                input_function_encoder = inp_enc
                output_function_encoder = out_enc

            models_dict[model_name] = (model, evaluate_fn)

            print(f"  Loaded {model_name}")

        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
            continue

    return models_dict, input_function_encoder, output_function_encoder


def plot_unified_comparison(
    models_dict,
    ranked_models,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    sample,
    sample_idx,
    dataset_info,
    save_dir=None,
):
    """
    Create unified multi-model comparison with 2x3 input grid + 2x3 re-simulation grid.
    Shows top 5 models + ground truth in 6th position.

    Args:
        models_dict: Dictionary mapping model_name -> (model, evaluate_fn)
        ranked_models: List of model names ranked by performance (best first)
        input_function_encoder: Input function encoder
        output_function_encoder: Output function encoder
        forward_model: Forward model for computing re-simulations
        sample: Test sample (X, u_true, Y, s_observed)
        sample_idx: Sample index for filename
        dataset_info: Dataset information (for spatial dimensions)
        save_dir: Directory to save plots
    """
    X, u_true, Y, s_observed = sample

    # Ensure models are in eval mode
    forward_model.eval()

    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_observed = s_observed.to(device)

    # Get spatial dimensions
    if dataset_info is not None:
        h, w = dataset_info.get("input_spatial_dims", (25, 25))
    else:
        # Default to square grid
        total_points = len(s_observed.squeeze())
        h = w = int(np.sqrt(total_points))

    # Select top 5 models
    top_5_models = ranked_models[:5]

    # Compute predictions for top 5 models with multiple samples
    # For 2D visualization, we'll average the samples
    n_samples = 10
    predictions = {}  # model_name -> averaged 2D prediction
    resimulations = {}  # model_name -> averaged 2D resimulation

    for model_name in top_5_models:
        if model_name not in models_dict:
            continue

        model, evaluate_fn = models_dict[model_name]
        model.eval()

        u_pred_samples = []
        s_resim_samples = []

        with torch.no_grad():
            point = (
                X.unsqueeze(0),
                u_true.unsqueeze(0),
                Y.unsqueeze(0),
                s_observed.unsqueeze(0),
            )

            # Generate multiple samples
            for _ in range(n_samples):
                # evaluate_fn returns (u_pred, alpha_pred)
                u_pred, alpha_pred = evaluate_fn(
                    model, point, input_function_encoder, output_function_encoder
                )
                u_pred = u_pred.squeeze(0)
                u_pred_samples.append(u_pred.cpu().numpy().flatten())

                # Compute re-simulation for all models using forward model
                # Forward pass through model to get beta coefficients
                beta_pred = forward_model.forward(alpha_pred)

                # Reconstruct re-simulation output
                s_resim = output_function_encoder(Y.unsqueeze(0), beta_pred)
                s_resim = s_resim.squeeze(0)  # Remove batch dimension

                s_resim_samples.append(s_resim.cpu().numpy().flatten())

        # Average samples and reshape to 2D
        u_pred_avg = np.mean(u_pred_samples, axis=0).reshape(h, w)
        predictions[model_name] = u_pred_avg

        s_resim_avg = np.mean(s_resim_samples, axis=0).reshape(h, w)
        resimulations[model_name] = s_resim_avg

    # Convert ground truth to numpy and reshape to 2D
    u_true_2d = u_true.squeeze(-1).cpu().numpy().reshape(h, w)
    s_observed_2d = s_observed.squeeze(-1).cpu().numpy().reshape(h, w)

    # Determine consistent value limits for input plots (force fields)
    all_u_values = [u_true_2d]
    for pred_2d in predictions.values():
        all_u_values.append(pred_2d)
    u_min = min(np.min(v) for v in all_u_values)
    u_max = max(np.max(v) for v in all_u_values)

    # Determine consistent value limits for output plots (displacements)
    all_s_values = [s_observed_2d]
    for resim_2d in resimulations.values():
        all_s_values.append(resim_2d)
    s_min = min(np.min(v) for v in all_s_values)
    s_max = max(np.max(v) for v in all_s_values)

    # Create figure
    fig, axes_left, axes_right, cbar_ax_left, cbar_ax_right = _create_unified_figure()

    # Plot input predictions (left 2x3 grid - top 5 models + ground truth)
    images_left = []

    # First 5 positions: top 5 models
    for idx, model_name in enumerate(top_5_models):
        if model_name not in predictions:
            continue
        ax = axes_left[idx]
        u_pred_2d = predictions[model_name]

        im = _plot_2d_field(
            ax,
            u_pred_2d,
            cmap="RdBu_r",
            vmin=u_min,
            vmax=u_max,
            annotation=_display_name(model_name),
            annotation_color="white",
        )
        images_left.append(im)

    # 6th position: ground truth
    ax = axes_left[5]
    im = _plot_2d_field(
        ax,
        u_true_2d,
        cmap="RdBu_r",
        vmin=u_min,
        vmax=u_max,
        annotation="Ground Truth",
        annotation_color="white",
    )
    images_left.append(im)

    # Add shared colorbar for left grid
    if images_left:
        fig.colorbar(images_left[0], cax=cbar_ax_left, label="Force")

    # Plot re-simulations (right 2x3 grid - top 5 models + observed)
    images_right = []

    # First 5 positions: top 5 models
    for idx, model_name in enumerate(top_5_models):
        if model_name not in resimulations:
            continue
        ax = axes_right[idx]
        s_resim_2d = resimulations[model_name]

        im = _plot_2d_field(
            ax,
            s_resim_2d,
            cmap="viridis",
            vmin=s_min,
            vmax=s_max,
            annotation=_display_name(model_name),
            annotation_color="white",
        )
        images_right.append(im)

    # 6th position: observed displacement
    ax = axes_right[5]
    im = _plot_2d_field(
        ax,
        s_observed_2d,
        cmap="viridis",
        vmin=s_min,
        vmax=s_max,
        annotation="Observed",
        annotation_color="white",
    )
    images_right.append(im)

    # Add shared colorbar for right grid
    if images_right:
        fig.colorbar(images_right[0], cax=cbar_ax_right, label="Displacement")

    # Save plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"chladni_comparison_sample_{sample_idx}_publication.pdf"
        )
        plt.savefig(
            save_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.05
        )

        png_path = os.path.join(
            save_dir, f"chladni_comparison_sample_{sample_idx}_publication.png"
        )
        plt.savefig(
            png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05
        )

    plt.close()


def plot_all_models_comparison(
    log_dir, results_dir, test_dataset, dataset_info, seed=1, sample_index=None
):
    """Plot unified comparison of all models on a single best sample."""

    print("Loading all models...")
    models_dict, input_function_encoder, output_function_encoder = load_all_models(
        log_dir, dataset_info, seed
    )

    if not models_dict:
        print("ERROR: No models found")
        return False

    # Load forward model for re-simulations (using b2b_nonlinear)
    print("Loading forward model for re-simulations...")
    shared_log_dir = os.path.join(log_dir, "shared", f"seed_{seed}")
    forward_model_name = "b2b_nonlinear"
    forward_model = load_forward_model(
        log_dir=shared_log_dir,
        forward_model_name=forward_model_name,
        device=device,
    )
    forward_model.eval()
    print(f"  Loaded {forward_model_name}")

    print(f"Loaded {len(models_dict)} models: {', '.join(models_dict.keys())}")

    # Rank models by performance
    print("Ranking models by performance...")
    ranked_models = rank_models_by_performance(
        test_dataset, models_dict, input_function_encoder, output_function_encoder
    )
    print(f"Model rankings (best to worst): {ranked_models}")

    # Select best sample if not specified
    if sample_index is None:
        print("Selecting best sample...")
        sample_index = select_best_sample(
            test_dataset, models_dict, input_function_encoder, output_function_encoder
        )
        print(f"Selected sample index: {sample_index}")
    else:
        print(f"Using specified sample index: {sample_index}")

    # Get the sample
    sample = test_dataset[sample_index]

    # Create unified comparison plot
    print("Creating unified comparison plot...")
    plot_unified_comparison(
        models_dict=models_dict,
        ranked_models=ranked_models,
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
        forward_model=forward_model,
        sample=sample,
        sample_idx=sample_index,
        dataset_info=dataset_info,
        save_dir=results_dir,
    )

    return True


if __name__ == "__main__":
    # Setup publication style
    setup_publication_style()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create publication-quality Chladni plots."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/store/at46867/b2b_operator_inverse",
        help="Base log directory",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/chladni_2d",
        help="Results directory for saving publication plots",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Specific sample index to plot (overrides automatic selection)",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Hardcoded dataset
    dataset = "chladni_2d"

    # Construct paths
    log_dir = os.path.join(args.log_dir, dataset)
    results_dir = os.path.join(args.results_dir)

    # Check if log directory exists
    if not os.path.exists(log_dir):
        print(f"ERROR: Log directory not found: {log_dir}")
        exit(1)

    # Load dataset - try to find any available model's params
    print("Loading dataset...")
    temp_params = None
    for model_name in INVERSE_MODELS:
        temp_log_dir = os.path.join(log_dir, model_name, f"seed_{args.seed}")
        params_path = os.path.join(temp_log_dir, "params.pth")
        if os.path.exists(params_path):
            temp_params = torch.load(params_path, weights_only=False)
            break

    if temp_params is None:
        print(f"ERROR: No trained models found in {log_dir}")
        exit(1)

    test_dataset, dataset_info = load_dataset(
        temp_params.dataset, temp_params, device, split="test", return_info=True
    )

    os.makedirs(results_dir, exist_ok=True)

    # Create unified comparison plot
    success = plot_all_models_comparison(
        log_dir=log_dir,
        results_dir=results_dir,
        test_dataset=test_dataset,
        dataset_info=dataset_info,
        seed=args.seed,
        sample_index=args.sample_index,
    )

    # Print summary
    print(f"\n{'='*50}")
    if success:
        print(f"SUCCESS: Created unified comparison plot → {results_dir}")
    else:
        print("FAILED: Could not create comparison plot")
        exit(1)
