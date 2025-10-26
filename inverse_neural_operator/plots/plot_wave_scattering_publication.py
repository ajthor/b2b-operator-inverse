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
    mpl.rcParams["figure.figsize"] = [6.5, 2.0]
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.05

    # Grid and axes
    mpl.rcParams["axes.grid"] = False
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["xtick.major.width"] = 0.8
    mpl.rcParams["ytick.major.width"] = 0.8

    # Colorbar
    mpl.rcParams["axes.axisbelow"] = True

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


def _display_name(model_name: str) -> str:
    """Convert internal model names to publication-ready display names."""
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
    return mapping.get(model_name, model_name)


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


def rank_models_by_performance(
    test_dataset, models_dict, input_function_encoder, output_function_encoder, forward_model
):
    """Rank models by their average MSE performance across test samples.

    For wave scattering, we evaluate on the re-simulated density field (output).
    For probabilistic models, we sample multiple evaluations and average.

    Args:
        test_dataset: Test dataset
        models_dict: Dictionary of {model_name: (model, evaluate_fn)}
        input_function_encoder: Input function encoder
        output_function_encoder: Output function encoder
        forward_model: Forward model for re-simulation

    Returns:
        List of model names sorted by performance (best first)
    """
    print("\n📊 Ranking models by performance...")
    model_mses = {model_name: [] for model_name in models_dict.keys()}

    n_eval_samples = min(50, len(test_dataset))
    n_samples_per_eval = 10  # Number of samples for probabilistic models

    # Evaluate on first 50 samples
    for idx in range(n_eval_samples):
        X, u_true, Y, s_observed = test_dataset[idx]

        # Ensure tensors are on correct device
        X = X.to(device)
        u_true = u_true.to(device)
        Y = Y.to(device)
        s_observed = s_observed.to(device)

        for model_name, (model, evaluate_fn) in models_dict.items():
            model.eval()

            # Sample multiple evaluations and average for probabilistic models
            s_resim_samples = []

            with torch.no_grad():
                point = (
                    X.unsqueeze(0),
                    u_true.unsqueeze(0),
                    Y.unsqueeze(0),
                    s_observed.unsqueeze(0),
                )

                for _ in range(n_samples_per_eval):
                    # Predict input (far-field pattern)
                    u_pred, _ = evaluate_fn(
                        model, point, input_function_encoder, output_function_encoder
                    )
                    u_pred = u_pred.squeeze(0)

                    # Re-simulate using forward model
                    X_batch = X.unsqueeze(0)
                    u_pred_batch = u_pred.unsqueeze(0)
                    Y_batch = Y.unsqueeze(0)

                    # Compute alpha coefficients from predicted input
                    alpha, _ = input_function_encoder.compute_coefficients(X_batch, u_pred_batch)

                    # Forward pass through model to get beta coefficients
                    beta_pred = forward_model.forward(alpha)

                    # Reconstruct re-simulation output
                    s_resim = output_function_encoder(Y_batch, beta_pred)
                    s_resim = s_resim.squeeze(0)

                    s_resim_samples.append(s_resim.cpu().numpy())

                # Average the samples (continuous values)
                s_resim_avg = np.mean(s_resim_samples, axis=0)
                s_resim_avg_tensor = torch.tensor(s_resim_avg, device=device)

                # Calculate MSE on continuous output
                mse = torch.mean((s_observed - s_resim_avg_tensor) ** 2).item()
                model_mses[model_name].append(mse)

    # Calculate average MSE for each model
    model_rankings = []
    for model_name, mse_list in model_mses.items():
        avg_mse = np.mean(mse_list)
        model_rankings.append((model_name, avg_mse))
        print(f"  {_display_name(model_name):15s}: MSE = {avg_mse:.6f}")

    # Sort by MSE (lower is better)
    model_rankings.sort(key=lambda x: x[1])

    print(f"\n🏆 Top 5 models:")
    for i, (model_name, mse) in enumerate(model_rankings[:5], 1):
        print(f"  {i}. {_display_name(model_name):15s}: MSE = {mse:.6f}")

    return [model_name for model_name, _ in model_rankings]


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
    ax_left_parent.set_xlabel(r"$\theta$ (radians)", labelpad=-8)
    ax_left_parent.set_ylabel(r"$|u(\theta)|$", labelpad=-8)
    ax_left_parent.set_title("Predicted Far-Field Patterns")

    ax_right_parent = fig.add_subplot(gs[:, 4:7], frameon=False)
    ax_right_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_right_parent.set_xlabel(r"$x$", labelpad=-8)
    ax_right_parent.set_ylabel(r"$y$", labelpad=-8)
    ax_right_parent.set_title("Re-simulated Density Fields")

    # Create subplot axes for left grid (polar plots)
    axes_left = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j], projection='polar')
            axes_left.append(ax)

    # Create subplot axes for right grid (2D density fields)
    axes_right = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j + 4])
            axes_right.append(ax)

    # Colorbar axes
    cax_left = fig.add_subplot(gs[:, 3])
    cax_right = fig.add_subplot(gs[:, 7])

    return fig, axes_left, axes_right, cax_left, cax_right


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
    line = ax.plot(theta_sorted, magnitude_sorted, 'b-', linewidth=0.5)[0]

    # Remove tick labels for cleaner look
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add annotation if provided
    if annotation:
        ax.text(
            0.05, 0.95, annotation,
            transform=ax.transAxes,
            fontsize=6,
            color='white',
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none')
        )

    return line


def _plot_2d_field(ax, field_2d, cmap="viridis", vmin=None, vmax=None, annotation=None):
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
    im = ax.imshow(field_2d, cmap=cmap, extent=extent, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    # Add annotation if provided
    if annotation:
        ax.text(
            0.05, 0.95, annotation,
            transform=ax.transAxes,
            fontsize=6,
            color='white',
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none')
        )

    return im


def plot_unified_comparison(
    models_dict,
    ranked_models,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    test_dataset,
    sample_idx,
    save_path=None,
):
    """Plot unified comparison showing top 5 models + ground truth/observed.

    Args:
        models_dict: Dictionary of {model_name: (model, evaluate_fn)}
        ranked_models: List of model names ranked by performance
        input_function_encoder: Input function encoder
        output_function_encoder: Output function encoder
        forward_model: Forward model for re-simulation
        test_dataset: Test dataset
        sample_idx: Index of sample to visualize
        save_path: Optional path to save figure
    """
    # Get sample data
    X, u_true, Y, s_observed = test_dataset[sample_idx]

    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_observed = s_observed.to(device)

    # Extract theta coordinates from X (X is [cos(theta), sin(theta)])
    X_np = X.squeeze(-1).cpu().numpy()
    theta = np.arctan2(X_np[:, 1], X_np[:, 0])

    # Convert ground truth to numpy
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    s_observed_np = s_observed.squeeze(-1).cpu().numpy()

    # Reshape density fields
    grid_size = 200
    s_observed_2d = s_observed_np.reshape(grid_size, grid_size)

    # Create figure
    fig, axes_left, axes_right, cax_left, cax_right = _create_unified_figure()

    # Select top 5 models
    top_5_models = ranked_models[:5]

    # Store image object for colorbar
    im_right = None

    # Number of samples for averaging (probabilistic models)
    n_samples = 10

    # Determine value limits across all models for consistent colormaps
    all_s_values = [s_observed_2d]

    # Store predictions for each model to avoid recomputing
    predictions = {}  # model_name -> (u_pred_np, s_resim_2d)

    # Compute predictions for all top 5 models
    for idx, model_name in enumerate(top_5_models):
        model, evaluate_fn = models_dict[model_name]
        model.eval()

        # Sample multiple evaluations and average
        u_pred_samples = []
        s_resim_samples = []

        with torch.no_grad():
            point = (
                X.unsqueeze(0),
                u_true.unsqueeze(0),
                Y.unsqueeze(0),
                s_observed.unsqueeze(0),
            )

            for _ in range(n_samples):
                # Predict input (far-field pattern)
                u_pred, _ = evaluate_fn(
                    model, point, input_function_encoder, output_function_encoder
                )
                u_pred = u_pred.squeeze(0)
                u_pred_samples.append(u_pred.cpu().numpy())

                # Re-simulate using forward model
                X_batch = X.unsqueeze(0)
                u_pred_batch = u_pred.unsqueeze(0)
                Y_batch = Y.unsqueeze(0)

                # Compute alpha coefficients from predicted input
                alpha, _ = input_function_encoder.compute_coefficients(X_batch, u_pred_batch)

                # Forward pass through model to get beta coefficients
                beta_pred = forward_model.forward(alpha)

                # Reconstruct re-simulation output
                s_resim = output_function_encoder(Y_batch, beta_pred)
                s_resim = s_resim.squeeze(0)
                s_resim_samples.append(s_resim.cpu().numpy())

            # Average the samples
            u_pred_avg = np.mean(u_pred_samples, axis=0)
            s_resim_avg = np.mean(s_resim_samples, axis=0)

            # Convert to numpy
            u_pred_np = u_pred_avg.squeeze(-1)
            s_resim_np = s_resim_avg.squeeze(-1)

            # Reshape to 2D (continuous, not thresholded)
            s_resim_2d = s_resim_np.reshape(grid_size, grid_size)

            # Store for later plotting
            predictions[model_name] = (u_pred_np, s_resim_2d)
            all_s_values.append(s_resim_2d)

    # Determine consistent value limits for density field plots
    s_min = min(np.min(v) for v in all_s_values)
    s_max = max(np.max(v) for v in all_s_values)

    # Plot using stored predictions
    for idx, model_name in enumerate(top_5_models):
        u_pred_np, s_resim_2d = predictions[model_name]

        # Plot predicted far-field pattern (left side - polar)
        ax = axes_left[idx]
        _plot_polar_field(
            ax, theta, np.abs(u_pred_np), annotation=_display_name(model_name)
        )

        # Plot re-simulated density field (right side - 2D, continuous)
        ax = axes_right[idx]
        im_right = _plot_2d_field(
            ax, s_resim_2d, cmap="viridis", vmin=s_min, vmax=s_max,
            annotation=_display_name(model_name)
        )

    # 6th position on left: ground truth far-field pattern
    ax = axes_left[5]
    _plot_polar_field(
        ax, theta, np.abs(u_true_np), annotation="Ground Truth"
    )

    # 6th position on right: observed density field
    ax = axes_right[5]
    im_right = _plot_2d_field(
        ax, s_observed_2d, cmap="viridis", vmin=s_min, vmax=s_max,
        annotation="Observed"
    )

    # Add colorbars (for the 2D density fields only)
    # Left colorbar: we don't need one for polar plots, so remove tick labels
    cax_left.set_visible(False)

    # Right colorbar: for density fields
    cbar_right = fig.colorbar(im_right, cax=cax_right, use_gridspec=True)
    cbar_right.set_label("Density")

    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        # Determine format from extension
        if save_path.endswith('.pdf'):
            plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
        else:
            plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05)
        print(f"✓ Saved publication figure → {save_path}")

    plt.close()


def plot_all_models_comparison(
    log_dir, results_dir, test_dataset, dataset_info, seed=1, sample_index=None
):
    """Plot unified comparison of all models on a single sample."""

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
        test_dataset, models_dict, input_function_encoder, output_function_encoder, forward_model
    )
    print(f"Model rankings (best to worst): {ranked_models}")

    # Use specified sample or default to 0
    if sample_index is None:
        sample_index = 0
        print(f"Using default sample index: {sample_index}")
    else:
        print(f"Using specified sample index: {sample_index}")

    # Create unified comparison plot
    print("Creating unified comparison plot...")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(
        results_dir, f"wave_scattering_sample_{sample_index}_publication.pdf"
    )

    plot_unified_comparison(
        models_dict=models_dict,
        ranked_models=ranked_models,
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
        forward_model=forward_model,
        test_dataset=test_dataset,
        sample_idx=sample_index,
        save_path=save_path,
    )

    # Also save PNG version
    png_path = os.path.join(
        results_dir, f"wave_scattering_sample_{sample_index}_publication.png"
    )
    plot_unified_comparison(
        models_dict=models_dict,
        ranked_models=ranked_models,
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
        forward_model=forward_model,
        test_dataset=test_dataset,
        sample_idx=sample_index,
        save_path=png_path,
    )

    return True


def main():
    # Setup publication style
    setup_publication_style()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create publication-quality Wave Scattering plots."
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
        default="results/wave_scattering",
        help="Results directory for saving publication plots",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Specific sample index to plot (default: 0)",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Hardcoded dataset
    dataset = "wave_scattering"

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


if __name__ == "__main__":
    main()
