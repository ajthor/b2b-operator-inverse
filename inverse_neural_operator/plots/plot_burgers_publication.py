"""
Publication-quality plotting script for Burgers 1D inverse problem results.

Creates publication-ready figures with:
- 6.5 inch width for journal publications
- 5-7pt sans serif fonts
- Professional styling and layout
- High-resolution output (300 DPI)
- Optimized 1D time-series visualization

To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_burgers_publication
"""

import os
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random

import torch

# Function encoder imports handled by load_models

from data.load_dataset import load_dataset
from models.load_model import load_models
from b2b.load_model import load_forward_model

device = "cpu"

# Available inverse models to plot (b2b_linear and b2b_nonlinear are forward models in shared/)
INVERSE_MODELS = [
    "linear",
    "linear_inverse",
    "nonlinear",
    "inn_affine",
    "cinn_affine",
    "variational_autoencoder",
    "conditional_realnvp",
    "mixture_density_network",
]

# Best 4 models for re-simulation comparison (right 2x2 grid)
BEST_MODELS = [
    "nonlinear",
    "inn_affine",
    "variational_autoencoder",
    "conditional_realnvp",
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
    mpl.rcParams["figure.figsize"] = [6.5, 2.2]  # 6.5" width for publication
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.05

    # Grid and axes
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.alpha"] = 0.3
    mpl.rcParams["grid.linewidth"] = 0.5
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["xtick.major.width"] = 0.8
    mpl.rcParams["ytick.major.width"] = 0.8

    # Colorbar
    mpl.rcParams["axes.axisbelow"] = True


def _model_colors():
    """Return consistent color map for models and shared colors."""
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
    """Map internal model names to nicer display/abbreviated names for annotations.

    Falls back to a cleaned, capitalized form when a mapping is not present.
    """
    mapping = {
        "linear": "Linear",
        "linear_inverse": "Linear-Inv",
        "nonlinear": "Nonlinear",
        "inn_affine": "INN",
        "cinn_affine": "cINN",
        "variational_autoencoder": "cVAE",
        "conditional_realnvp": "RealNVP",
        "mixture_density_network": "MDN",
    }
    if model_name in mapping:
        return mapping[model_name]
    # Fallback: prettify by replacing underscores and capitalizing words
    return " ".join([p.capitalize() for p in model_name.split("_")])


def _plot_1d_samples(
    ax,
    coords,
    true_signal,
    sample_signals,
    color_true,
    color_pred,
    x_lim,
    y_lim,
    true_label=None,
    pred_label=None,
    annotation=None,
    annotation_color=None,
):
    """Plot a true signal and multiple predicted samples on ax.

    Keeps ticks minimal and applies provided limits.
    """
    # True signal
    if true_signal is not None:
        ax.plot(coords, true_signal, color=color_true, linewidth=0.8, label=true_label)

    # Predicted samples (transparent)
    for i, sig in enumerate(sample_signals):
        lbl = pred_label if i == 0 else None
        ax.plot(coords, sig, color=color_pred, linewidth=0.6, label=lbl, alpha=0.6)

    ax.set_xlim([coords.min(), coords.max()])
    ax.set_ylim(y_lim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(labelsize=5, length=0)

    # Add model annotation in bottom-left corner of the axis (axes fraction coords)
    if annotation is not None:
        ann_color = annotation_color if annotation_color is not None else color_pred
        # small, unobtrusive label using axes-relative coordinates
        ax.text(
            0.02,
            0.02,
            str(annotation),
            transform=ax.transAxes,
            fontsize=5,
            color=ann_color,
            va="bottom",
            ha="left",
            alpha=0.9,
        )


def _create_unified_figure():
    """Create the publication-style figure and return fig, left_axes, right_axes."""
    fig = plt.figure(figsize=(6.5, 2.2))
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(
        2,
        7,
        figure=fig,
        hspace=0.08,
        wspace=0.08,
        width_ratios=[1, 1, 1, 1, 0.3, 1, 1],
        left=0.06,
        right=0.98,
        top=0.88,
        bottom=0.06,
    )

    axes_left = [fig.add_subplot(gs[row, col]) for row in range(2) for col in range(4)]
    axes_right = [fig.add_subplot(gs[row, 5 + col]) for row in range(2) for col in range(2)]
    return fig, axes_left, axes_right


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


def plot_unified_comparison(
    models_dict,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    sample,
    sample_idx,
    save_dir=None,
):
    """
    Create unified multi-model comparison with 2x4 input grid + 2x2 re-simulation grid.

    Args:
        models_dict: Dictionary mapping model_name -> (model, evaluate_fn)
        input_function_encoder: Input function encoder
        output_function_encoder: Output function encoder
        forward_model: Forward model for computing re-simulations
        sample: Test sample (X, u_true, Y, s_observed)
        sample_idx: Sample index for filename
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

    # Extract coordinates for 1D plotting - flatten to ensure 1D arrays
    x_coords = X.cpu().numpy().flatten()
    y_coords = Y.cpu().numpy().flatten()

    colors = _model_colors()
    color_true = colors["color_true"]
    color_observed = colors["color_observed"]
    model_colors = colors["models"]

    # Compute predictions for all models with multiple samples
    # For probabilistic models, each call to evaluate gives different samples
    # For deterministic models, all samples will be identical
    n_samples = 10
    predictions = {}  # model_name -> list of prediction arrays
    resimulations = {}  # model_name -> list of resimulation arrays

    for model_name in INVERSE_MODELS:
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
                # evaluate_fn returns (u_pred, alpha_pred) not (u_pred, s_pred)
                u_pred, alpha_pred = evaluate_fn(
                    model, point, input_function_encoder, output_function_encoder
                )
                u_pred = u_pred.squeeze(0)
                pred_np = u_pred.cpu().numpy().flatten()
                u_pred_samples.append(pred_np)

                # Compute re-simulation for best models using forward model
                # Use alpha_pred directly from evaluate (don't recompute coefficients)
                if model_name in BEST_MODELS:
                    # Forward pass through model to get beta coefficients
                    beta_pred = forward_model.forward(alpha_pred)

                    # Reconstruct re-simulation output
                    s_resim = output_function_encoder(Y.unsqueeze(0), beta_pred)
                    s_resim = s_resim.squeeze(0)  # Remove batch dimension

                    resim_np = s_resim.cpu().numpy().flatten()
                    s_resim_samples.append(resim_np)

        predictions[model_name] = u_pred_samples
        if model_name in BEST_MODELS:
            resimulations[model_name] = s_resim_samples

    # Convert ground truth to numpy
    u_true_np = u_true.cpu().numpy().flatten()
    s_observed_np = s_observed.cpu().numpy().flatten()

    # Determine consistent axis limits for input plots
    all_u_values = [u_true_np]
    for pred_samples in predictions.values():
        for pred in pred_samples:
            all_u_values.append(pred)
    u_min = min(np.min(v) for v in all_u_values)
    u_max = max(np.max(v) for v in all_u_values)
    u_margin = (u_max - u_min) * 0.05
    u_lim = [u_min - u_margin, u_max + u_margin]

    # Determine consistent axis limits for output plots
    all_s_values = [s_observed_np]
    for resim_samples in resimulations.values():
        for resim in resim_samples:
            all_s_values.append(resim)
    s_min = min(np.min(v) for v in all_s_values)
    s_max = max(np.max(v) for v in all_s_values)
    s_margin = (s_max - s_min) * 0.05
    s_lim = [s_min - s_margin, s_max + s_margin]

    fig, axes_left, axes_right = _create_unified_figure()

    # Plot input predictions (left 2x4 grid) using helper
    for idx, model_name in enumerate(INVERSE_MODELS):
        if idx >= 8 or model_name not in predictions:
            continue
        ax = axes_left[idx]
        u_pred_samples = predictions[model_name]
        color_pred = model_colors.get(model_name, "#d62728")

        _plot_1d_samples(
            ax,
            x_coords,
            u_true_np,
            u_pred_samples,
            color_true,
            color_pred,
            x_lim=[x_coords.min(), x_coords.max()],
            y_lim=u_lim,
            true_label="True Input",
            pred_label="Predicted Input",
            annotation=_display_name(model_name),
            annotation_color=color_pred,
        )

    # Add shared legend for input predictions (above left grid)
    from matplotlib.lines import Line2D

    legend_handles_left = [
        Line2D([0], [0], color=color_true, linewidth=0.8, label="True Input"),
        # Predicted Input legend intentionally commented out to reduce clutter
        # Line2D([0], [0], color="gray", linewidth=0.6, alpha=0.6, label="Predicted Input"),
    ]
    fig.legend(
        handles=legend_handles_left,
        loc="upper center",
        bbox_to_anchor=(0.30, 1.00),
        ncol=2,
        fontsize=5,
        frameon=False,
    )

    # Plot re-simulations (right 2x2 grid) using helper
    for idx, model_name in enumerate(BEST_MODELS):
        if idx >= 4 or model_name not in resimulations:
            continue
        ax = axes_right[idx]
        s_resim_samples = resimulations[model_name]
        color_resim = model_colors.get(model_name, "#9467bd")

        _plot_1d_samples(
            ax,
            y_coords,
            s_observed_np,
            s_resim_samples,
            color_observed,
            color_resim,
            x_lim=[y_coords.min(), y_coords.max()],
            y_lim=s_lim,
            true_label="Observed Output",
            pred_label="Re-simulated Output",
            annotation=_display_name(model_name),
            annotation_color=color_resim,
        )

    # Add shared legend for re-simulations (above right grid)
    legend_handles_right = [
        Line2D([0], [0], color=color_observed, linewidth=0.8, label="Observed Output"),
        # Re-simulated Output legend intentionally commented out to reduce clutter
        # Line2D([0], [0], color="gray", linewidth=0.6, alpha=0.6, label="Re-simulated Output"),
    ]
    fig.legend(
        handles=legend_handles_right,
        loc="upper center",
        bbox_to_anchor=(0.82, 1.00),
        ncol=2,
        fontsize=5,
        frameon=False,
    )

    # Add shared axis labels with LaTeX
    # Left column label (for input predictions)
    fig.text(0.025, 0.47, r"$f(x)$", va="center", rotation="vertical", fontsize=6)
    # Bottom label for left grid
    fig.text(0.30, 0.025, r"$x$", ha="center", fontsize=6)

    # Right grid labels (for re-simulations)
    fig.text(0.67, 0.47, r"$h(y)$", va="center", rotation="vertical", fontsize=6)
    # Bottom label for right grid
    fig.text(0.82, 0.025, r"$y$", ha="center", fontsize=6)

    # Save plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"unified_comparison_sample_{sample_idx}_publication.pdf"
        )
        plt.savefig(
            save_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.05
        )

        png_path = os.path.join(
            save_dir, f"unified_comparison_sample_{sample_idx}_publication.png"
        )
        plt.savefig(
            png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05
        )

    plt.close()


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
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
        forward_model=forward_model,
        sample=sample,
        sample_idx=sample_index,
        save_dir=results_dir,
    )

    return True


if __name__ == "__main__":
    # Setup publication style
    setup_publication_style()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create publication-quality Burgers plots."
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
        default="results/burgers_1d",
        help="Results directory for saving publication plots",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=31,
        help="Specific sample index to plot (overrides automatic selection)",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Hardcoded dataset
    dataset = "burgers_1d"

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
