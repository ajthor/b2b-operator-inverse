"""
Plot the results of the Darcy 1D dataset for all models.
To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_darcy
"""

import os
import sys
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Optional, Dict
from collections import defaultdict

import torch

# Ensure project root and package root are on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "inverse_neural_operator"
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

DEFAULT_DATASET = "darcy_1d"
DEFAULT_MODEL_NAME = "nonlinear"
DEFAULT_SEED = 1

from inverse_neural_operator.data.load_dataset import load_dataset
from inverse_neural_operator.models.load_model import load_models
from inverse_neural_operator.b2b.load_model import load_forward_model
from inverse_neural_operator.plots.plot_utils import find_best_worst_samples

device = "cpu"

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def to_device(*tensors):
    return tuple(t.to(device) for t in tensors)


def add_batch_dim(*tensors):
    return tuple(t.unsqueeze(0) for t in tensors)


def squeeze_to_numpy(tensor: torch.Tensor):
    return tensor.squeeze(-1).cpu().numpy()


def apply_inverse_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    """Add zero-mean Gaussian noise for inverse inference without mutating input."""
    if std <= 0:
        return tensor
    return tensor + torch.randn_like(tensor) * std


def format_noise_value(std: float) -> str:
    trimmed = f"{std:.6f}".rstrip("0").rstrip(".")
    return trimmed or format(std, "g")


def infer_dataset_name(log_path: Path) -> str:
    if len(log_path.parents) > 1 and log_path.parents[1].name:
        return log_path.parents[1].name
    if log_path.parent.name:
        return log_path.parent.name
    return DEFAULT_DATASET


def collect_model_log_dirs(base_path: Path, seed: int, model_filter: Optional[set[str]]) -> Dict[str, Path]:
    seed_dir_name = f"seed_{seed}"
    model_dirs = {}

    if (base_path / "params.pth").exists():
        fallback_model = next(iter(model_filter)) if model_filter else DEFAULT_MODEL_NAME
        model_name = base_path.parent.name or fallback_model
        if not model_filter or model_name in model_filter:
            model_dirs[model_name] = base_path
        return model_dirs

    seed_candidate = base_path / seed_dir_name
    if (seed_candidate / "params.pth").exists():
        model_name = base_path.name
        if not model_filter or model_name in model_filter:
            model_dirs[model_name] = seed_candidate
        return model_dirs

    if not base_path.is_dir():
        return model_dirs

    for entry in base_path.iterdir():
        if not entry.is_dir() or entry.name == "shared":
            continue
        seed_dir = entry / seed_dir_name
        if not (seed_dir / "params.pth").exists():
            continue
        if model_filter and entry.name not in model_filter:
            continue
        model_dirs[entry.name] = seed_dir

    return model_dirs


def compute_mse_errors(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    dataset,
    inverse_noise_std: float = 0.0,
):
    """Compute dataset-wide MSE for inverse prediction and forward re-simulation."""
    model.eval()
    forward_model.eval()

    inverse_sq_error = 0.0
    forward_sq_error = 0.0
    inverse_count = 0
    forward_count = 0

    with torch.no_grad():
        for sample in dataset:
            X, u_true, Y, s_observed = to_device(*sample)
            s_input = apply_inverse_noise(s_observed, inverse_noise_std)

            X_b, u_b, Y_b, s_b = add_batch_dim(X, u_true, Y, s_input)
            u_pred, _ = evaluate_fn(
                model, (X_b, u_b, Y_b, s_b), input_function_encoder, output_function_encoder
            )
            u_pred = u_pred.squeeze(0)

            alpha, _ = input_function_encoder.compute_coefficients(X_b, u_pred.unsqueeze(0))
            beta_pred = forward_model.forward(alpha)
            s_resim = output_function_encoder(Y_b, beta_pred).squeeze(0)

            inverse_sq_error += torch.sum((u_pred - u_true) ** 2).item()
            forward_sq_error += torch.sum((s_resim - s_observed) ** 2).item()
            inverse_count += u_true.numel()
            forward_count += s_observed.numel()

    inverse_mse = inverse_sq_error / inverse_count if inverse_count else 0.0
    forward_mse = forward_sq_error / forward_count if forward_count else 0.0

    return inverse_mse, forward_mse


def plot_darcy_sample(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    sample,
    sample_idx,
    model_name,
    save_dir=None,
    inverse_noise_std: float = 0.0,
):
    """
    Plot a single Darcy sample with observed output, true vs predicted input, and re-simulation.
    """
    model.eval()
    forward_model.eval()

    X, u_true, Y, s_observed = to_device(*sample)
    s_input = apply_inverse_noise(s_observed, inverse_noise_std)

    with torch.no_grad():
        X_b, u_b, Y_b, s_b = add_batch_dim(X, u_true, Y, s_input)
        u_pred, _ = evaluate_fn(
            model, (X_b, u_b, Y_b, s_b), input_function_encoder, output_function_encoder
        )
        u_pred = u_pred.squeeze(0)

        alpha, _ = input_function_encoder.compute_coefficients(X_b, u_pred.unsqueeze(0))
        beta_pred = forward_model.forward(alpha)
        s_resim = output_function_encoder(Y_b, beta_pred).squeeze(0)

    # Convert to numpy for plotting
    u_true_np = squeeze_to_numpy(u_true)
    u_pred_np = squeeze_to_numpy(u_pred)
    s_observed_np = squeeze_to_numpy(s_observed)
    s_input_np = squeeze_to_numpy(s_input)
    s_resim_np = squeeze_to_numpy(s_resim)
    X_np = X.squeeze(-1).cpu().numpy()
    Y_np = Y.squeeze(-1).cpu().numpy()

    # Try to determine grid size (assuming square grid)
    n_points = len(X_np)
    grid_size = int(np.sqrt(n_points))

    # If not a perfect square, use the data as-is for 1D case
    if grid_size * grid_size != n_points:
        # Create 1D plot with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Extract x coordinates for 1D plotting
        if X_np.ndim == 1:
            x_coords = X_np  # Already 1D coordinates
            y_coords = Y_np
        else:
            x_coords = X_np[:, 0]  # Use the first coordinate (x)
            y_coords = Y_np[:, 0]  # Use the first coordinate (y)

        # Plot 1: Observed output function (what we can measure)
        axes[0].plot(
            y_coords, s_observed_np, "g-", label="Observed Output s(y)", linewidth=2
        )
        if inverse_noise_std > 0:
            axes[0].plot(
                y_coords,
                s_input_np,
                "k--",
                label="Noisy Measurement",
                linewidth=1.5,
                alpha=0.9,
            )
        axes[0].set_title("Observed Output Function s(y)", fontsize=12)
        axes[0].set_xlabel("y")
        axes[0].set_ylabel("s(y)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Input function comparison (what we want to predict)
        axes[1].plot(
            x_coords, u_true_np, "b-", label="True Input u(x)", linewidth=2, alpha=0.8
        )
        axes[1].plot(
            x_coords,
            u_pred_np,
            "r--",
            label="Predicted Input û(x)",
            linewidth=2,
            alpha=0.8,
        )
        axes[1].set_title("Input Function: True vs Predicted", fontsize=12)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("u(x)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Re-simulation comparison
        axes[2].plot(
            y_coords,
            s_observed_np,
            "g-",
            label="True Observed s(y)",
            linewidth=2,
            alpha=0.8,
        )
        axes[2].plot(
            y_coords,
            s_resim_np,
            "m--",
            label="Re-simulated ŝ(y)",
            linewidth=2,
            alpha=0.8,
        )

        # Calculate and display error metrics
        mse_resim = np.mean((s_observed_np - s_resim_np) ** 2)
        mae_resim = np.mean(np.abs(s_observed_np - s_resim_np))

        axes[2].set_title(
            f"Re-simulation vs Observed\nMSE: {mse_resim:.6f}, MAE: {mae_resim:.6f}",
            fontsize=12,
        )
        axes[2].set_xlabel("y")
        axes[2].set_ylabel("s(y)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    else:
        # 2D visualization
        # Reshape to 2D grids
        u_true_2d = u_true_np.reshape(grid_size, grid_size)
        u_pred_2d = u_pred_np.reshape(grid_size, grid_size)
        s_observed_2d = s_observed_np.reshape(grid_size, grid_size)
        s_resim_2d = s_resim_np.reshape(grid_size, grid_size)

        # Calculate errors
        input_error_2d = np.abs(u_pred_2d - u_true_2d)
        resim_error_2d = np.abs(s_resim_2d - s_observed_2d)

        # Create the plot with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Physical domain extent (assuming normalized coordinates)
        extent = [0, 1, 0, 1]

        # Plot 1: Observed output function
        im1 = axes[0].imshow(
            s_observed_2d, cmap="viridis", extent=extent, origin="lower"
        )
        axes[0].set_title("Observed Output Function s(x,y)", fontsize=12)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Plot 2: Input prediction error
        im2 = axes[1].imshow(input_error_2d, cmap="Reds", extent=extent, origin="lower")
        axes[1].set_title("Input Prediction Error |û - u|", fontsize=12)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        # Plot 3: Re-simulation error
        im3 = axes[2].imshow(
            resim_error_2d, cmap="Blues", extent=extent, origin="lower"
        )
        mse_resim = np.mean((s_observed_2d - s_resim_2d) ** 2)
        axes[2].set_title(
            f"Re-simulation Error |ŝ - s|\nMSE: {mse_resim:.6f}", fontsize=12
        )
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        plt.colorbar(im3, ax=axes[2], fraction=0.046)

    plt.tight_layout()

    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_name}_sample_{sample_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_multiple_samples(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    test_dataset,
    model_name,
    n_samples=3,
    save_dir=None,
    inverse_noise_std: float = 0.0,
):
    """Plot multiple random samples from the test set."""

    # Select random samples
    test_indices = random.sample(
        range(len(test_dataset)), min(n_samples, len(test_dataset))
    )

    for idx in test_indices:
        plot_darcy_sample(
            model,
            evaluate_fn,
            input_function_encoder,
            output_function_encoder,
            forward_model,
            test_dataset[idx],
            idx,
            model_name,
            save_dir,
            inverse_noise_std=inverse_noise_std,
        )


def plot_model_results(
    model_name,
    log_dir,
    results_dir,
    test_dataset,
    dataset_info,
    n_samples=3,
    inverse_noise_std: float = 0.0,
):
    """Plot results for a single model.

    Args:
        model_name: Name of the model
        log_dir: Complete path to the model directory (e.g., /path/to/logs/dataset/model/seed_1)
        results_dir: Directory to save results
        test_dataset: Test dataset
        dataset_info: Dataset information
        n_samples: Number of samples to plot
    """

    # Load model parameters
    params = torch.load(os.path.join(log_dir, "params.pth"), weights_only=False)

    # Load models
    input_function_encoder, output_function_encoder, model, evaluate_fn = load_models(
        log_dir=log_dir,
        dataset_info=dataset_info,
        params=params,
        device=device,
    )

    # Load forward model for re-simulation
    forward_model = load_forward_model(log_dir=log_dir, forward_model_name='b2b_nonlinear', device=device)

    inverse_mse, forward_mse = compute_mse_errors(
        model=model,
        evaluate_fn=evaluate_fn,
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
        forward_model=forward_model,
        dataset=test_dataset,
        inverse_noise_std=inverse_noise_std,
    )

    # Plot results
    plot_multiple_samples(
        model=model,
        evaluate_fn=evaluate_fn,
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
        forward_model=forward_model,
        test_dataset=test_dataset,
        model_name=model_name,
        n_samples=n_samples,
        save_dir=results_dir,
        inverse_noise_std=inverse_noise_std,
    )

    # Find and plot best/worst case samples
    print(f"Finding best and worst case samples for {model_name}...")
    best_idx, worst_idx, best_mse, worst_mse = find_best_worst_samples(
        model=model,
        evaluate_fn=evaluate_fn,
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
        forward_model=forward_model,
        test_dataset=test_dataset,
        device=device,
    )

    # Plot best case
    print(f"Plotting best case (MSE: {best_mse:.6e})...")
    best_sample = test_dataset[best_idx]
    plot_darcy_sample(
        model=model,
        evaluate_fn=evaluate_fn,
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
        forward_model=forward_model,
        sample=best_sample,
        sample_idx=f"best_{best_idx}",
        model_name=model_name,
        save_dir=results_dir,
        inverse_noise_std=inverse_noise_std,
    )

    # Plot worst case
    print(f"Plotting worst case (MSE: {worst_mse:.6e})...")
    worst_sample = test_dataset[worst_idx]
    plot_darcy_sample(
        model=model,
        evaluate_fn=evaluate_fn,
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
        forward_model=forward_model,
        sample=worst_sample,
        sample_idx=f"worst_{worst_idx}",
        model_name=model_name,
        save_dir=results_dir,
        inverse_noise_std=inverse_noise_std,
    )

    return {
        "model": model_name,
        "inverse_mse": inverse_mse,
        "forward_mse": forward_mse,
        "inverse_noise_std": inverse_noise_std,
        "log_dir": log_dir,
    }


# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot Darcy 1D results for all models.")
parser.add_argument("--log_dir", type=str, default=None, help="Path to logs root (defaults to logs/darcy_1d; accepts model/seed paths too)")
parser.add_argument("--results_dir", type=str, default=None, help="Results directory for saving plots")
parser.add_argument("--n_samples", type=int, default=3, help="Number of random samples to plot per model")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--model", type=str, default=None, help="Model name to plot results for (defaults to all models)")
parser.add_argument("--inverse_noise_std", type=float, default=0.0, help="Stddev of Gaussian noise added to inverse-model inputs")

args = parser.parse_args()

if args.log_dir is None:
    args.log_dir = str(PROJECT_ROOT / "logs" / DEFAULT_DATASET)

inverse_noise_std = max(args.inverse_noise_std, 0.0)

# Set random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

base_log_path = Path(args.log_dir)
if not base_log_path.exists():
    print(f"✗ Log directory not found: {base_log_path}")
    exit(1)
model_filter = {args.model} if args.model else None
model_log_dirs = collect_model_log_dirs(base_log_path, args.seed, model_filter)

if not model_log_dirs:
    target = args.model or "any model"
    print(f"✗ No models found at {base_log_path} for {target}")
    exit(1)

multiple_models = len(model_log_dirs) > 1
custom_results_dir = Path(args.results_dir).resolve() if args.results_dir else None
metrics_aggregate: Dict[Path, Dict[str, Dict]] = defaultdict(dict)

for model_name, model_log_dir in sorted(model_log_dirs.items()):
    dataset_name = infer_dataset_name(model_log_dir)
    aggregator_base = custom_results_dir if custom_results_dir else PROJECT_ROOT / "results" / dataset_name
    noise_suffix = format_noise_value(inverse_noise_std) if inverse_noise_std > 0 else None
    aggregator_dir = aggregator_base if not noise_suffix else aggregator_base / noise_suffix

    if custom_results_dir:
        results_dir_path = aggregator_dir / model_name if multiple_models else aggregator_dir
    else:
        results_dir_path = aggregator_dir / model_name

    results_dir = str(results_dir_path)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading model parameters and dataset for {model_name}...")
    params = torch.load(os.path.join(model_log_dir, "params.pth"), weights_only=False)
    test_dataset, dataset_info = load_dataset(params.dataset, params, device, split="test", return_info=True)
    print(f"✓ {model_name}: loaded {len(test_dataset)} test samples")

    print(f"Generating {args.n_samples} sample plots for {model_name}...")
    metrics = plot_model_results(
        model_name=model_name,
        log_dir=str(model_log_dir),
        results_dir=results_dir,
        test_dataset=test_dataset,
        dataset_info=dataset_info,
        n_samples=args.n_samples,
        inverse_noise_std=inverse_noise_std,
    )
    metrics["results_dir"] = results_dir
    metrics_aggregate[aggregator_dir][model_name] = metrics
    print(f"✓ {model_name}: saved plots → {results_dir}")

for metrics_dir, data in metrics_aggregate.items():
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(data, metrics_file, indent=2)
    print(f"✓ Saved metrics for {len(data)} model(s) → {metrics_path}")
