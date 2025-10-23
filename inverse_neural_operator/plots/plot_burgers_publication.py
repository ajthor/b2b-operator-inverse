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
    "variational_autoencoder",
    "inn_affine",
    "cinn_affine",
    "conditional_realnvp",
    "mixture_density_network",
]

# Best 4 models for re-simulation comparison (right 2x2 grid)
BEST_MODELS = [
    "nonlinear",
    "variational_autoencoder",
    "inn_affine",
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

    # Define professional color scheme
    color_true = "#1f77b4"  # Blue
    color_pred = "#d62728"  # Red
    color_observed = "#2E8B57"  # Sea green
    color_resim = "#9467bd"  # Purple

    # Compute predictions for all models
    predictions = {}
    resimulations = {}

    for model_name in INVERSE_MODELS:
        if model_name not in models_dict:
            continue

        model, evaluate_fn = models_dict[model_name]
        model.eval()

        with torch.no_grad():
            point = (
                X.unsqueeze(0),
                u_true.unsqueeze(0),
                Y.unsqueeze(0),
                s_observed.unsqueeze(0),
            )
            # evaluate_fn returns (u_pred, alpha_pred) not (u_pred, s_pred)
            u_pred, _ = evaluate_fn(
                model, point, input_function_encoder, output_function_encoder
            )
            u_pred = u_pred.squeeze(0)
            pred_np = u_pred.cpu().numpy().flatten()
            predictions[model_name] = pred_np

            # Compute re-simulation for best models using forward model
            if model_name in BEST_MODELS:
                # Add batch dimensions
                X_batch = X.unsqueeze(0)
                u_pred_batch = u_pred.unsqueeze(0)
                Y_batch = Y.unsqueeze(0)

                # Compute alpha coefficients from predicted input
                alpha, _ = input_function_encoder.compute_coefficients(
                    X_batch, u_pred_batch
                )

                # Forward pass through model to get beta coefficients
                beta_pred = forward_model.forward(alpha)

                # Reconstruct re-simulation output
                s_resim = output_function_encoder(Y_batch, beta_pred)
                s_resim = s_resim.squeeze(0)  # Remove batch dimension

                resim_np = s_resim.cpu().numpy().flatten()
                resimulations[model_name] = resim_np

    # Convert ground truth to numpy
    u_true_np = u_true.cpu().numpy().flatten()
    s_observed_np = s_observed.cpu().numpy().flatten()

    # Determine consistent axis limits for input plots
    all_u_values = [u_true_np]
    for pred in predictions.values():
        all_u_values.append(pred)
    u_min = min(np.min(v) for v in all_u_values)
    u_max = max(np.max(v) for v in all_u_values)
    u_margin = (u_max - u_min) * 0.05
    u_lim = [u_min - u_margin, u_max + u_margin]

    # Determine consistent axis limits for output plots
    all_s_values = [s_observed_np]
    for resim in resimulations.values():
        all_s_values.append(resim)
    s_min = min(np.min(v) for v in all_s_values)
    s_max = max(np.max(v) for v in all_s_values)
    s_margin = (s_max - s_min) * 0.05
    s_lim = [s_min - s_margin, s_max + s_margin]

    # Create figure with GridSpec layout
    # Shorter figure for more square-looking axes
    fig = plt.figure(figsize=(6.5, 2.2))
    import matplotlib.gridspec as gridspec

    # Use smaller spacing between axes, with gap between left and right grids
    # width_ratios creates space between the two grid sections
    gs = gridspec.GridSpec(
        2, 7, figure=fig,
        hspace=0.08, wspace=0.08,
        width_ratios=[1, 1, 1, 1, 0.3, 1, 1],  # 0.3 creates gap between grids
        left=0.02, right=0.98, top=0.88, bottom=0.02
    )

    # Left grid: 2 rows x 4 columns for input predictions
    axes_left = []
    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(gs[row, col])
            axes_left.append(ax)

    # Right grid: 2 rows x 2 columns for re-simulations
    # Skip column 4 (the gap), use columns 5 and 6
    axes_right = []
    for row in range(2):
        for col in range(2):
            ax = fig.add_subplot(gs[row, 5 + col])
            axes_right.append(ax)

    # Plot input predictions (left 2x4 grid)
    for idx, model_name in enumerate(INVERSE_MODELS):
        if idx >= 8 or model_name not in predictions:
            continue

        ax = axes_left[idx]
        u_pred_np = predictions[model_name]

        ax.plot(x_coords, u_true_np, color=color_true, linewidth=1.0, label="True")
        ax.plot(
            x_coords,
            u_pred_np,
            color=color_pred,
            linewidth=1.0,
            linestyle="--",
            label="Pred",
            alpha=0.8,
        )

        # Set consistent axis limits
        ax.set_xlim([x_coords.min(), x_coords.max()])
        ax.set_ylim(u_lim)

        # Remove tick labels and tick marks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(labelsize=5, length=0)

    # Add shared legend for input predictions (above left grid)
    from matplotlib.lines import Line2D
    legend_handles_left = [
        Line2D([0], [0], color=color_true, linewidth=1.0, label="True"),
        Line2D([0], [0], color=color_pred, linewidth=1.0, linestyle="--", label="Pred"),
    ]
    fig.legend(
        handles=legend_handles_left,
        loc="upper center",
        bbox_to_anchor=(0.30, 1.00),
        ncol=2,
        fontsize=5,
        frameon=False,
    )

    # Plot re-simulations (right 2x2 grid)
    for idx, model_name in enumerate(BEST_MODELS):
        if idx >= 4 or model_name not in resimulations:
            continue

        ax = axes_right[idx]
        s_resim_np = resimulations[model_name]

        ax.plot(
            y_coords, s_observed_np, color=color_observed, linewidth=1.0, label="Obs"
        )
        ax.plot(
            y_coords,
            s_resim_np,
            color=color_resim,
            linewidth=1.0,
            linestyle="--",
            label="Resim",
            alpha=0.8,
        )

        # Set consistent axis limits
        ax.set_xlim([y_coords.min(), y_coords.max()])
        ax.set_ylim(s_lim)

        # Remove tick labels and tick marks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(labelsize=5, length=0)

    # Add shared legend for re-simulations (above right grid)
    legend_handles_right = [
        Line2D([0], [0], color=color_observed, linewidth=1.0, label="Obs"),
        Line2D([0], [0], color=color_resim, linewidth=1.0, linestyle="--", label="Resim"),
    ]
    fig.legend(
        handles=legend_handles_right,
        loc="upper center",
        bbox_to_anchor=(0.82, 1.00),
        ncol=2,
        fontsize=5,
        frameon=False,
    )

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
