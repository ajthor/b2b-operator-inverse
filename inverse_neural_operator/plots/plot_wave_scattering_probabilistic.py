"""
Plot probabilistic results for the Wave Scattering dataset with multiple realizations.

For probabilistic models, samples 10 realizations from the posterior and plots all
forward resimulations. For 1D datasets, all realizations are plotted as lines.

To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_burgers_probabilistic
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random

import torch

from inverse_neural_operator.b2b.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)

from data.load_dataset import load_dataset
from models.load_model import load_models
from b2b.load_model import load_forward_model

device = "cpu"

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def plot_burgers_probabilistic_sample(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    sample,
    sample_idx,
    model_name,
    n_realizations=10,
    save_dir=None,
):
    """
    Plot a single Wave Scattering sample with probabilistic realizations.
    """
    model.eval()
    forward_model.eval()

    X, u_true, Y, s_observed = sample

    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_observed = s_observed.to(device)

    # Generate multiple realizations by calling evaluate multiple times
    # For probabilistic models (VAE, MDN, cINN_probabilistic), each call gives different samples
    # For deterministic models (b2b, cINN deterministic, INN), each call gives the same result
    u_samples = []
    s_resim_samples = []

    with torch.no_grad():
        point = (
            X.unsqueeze(0),
            u_true.unsqueeze(0),
            Y.unsqueeze(0),
            s_observed.unsqueeze(0),
        )

        for _ in range(n_realizations):
            # Call evaluate to get one realization
            u_pred, alpha_pred = evaluate_fn(
                model, point, input_function_encoder, output_function_encoder
            )
            u_samples.append(u_pred.squeeze(0))

            # Forward simulate this realization using alpha_pred
            beta_resim = forward_model.forward(alpha_pred)
            s_resim = output_function_encoder(Y.unsqueeze(0), beta_resim)
            s_resim_samples.append(s_resim.squeeze(0))

    # Convert to numpy for plotting
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    s_observed_np = s_observed.squeeze(-1).cpu().numpy()
    X_np = X.squeeze(-1).cpu().numpy()
    Y_np = Y.squeeze(-1).cpu().numpy()

    u_samples_np = [u.squeeze(-1).cpu().numpy() for u in u_samples]
    s_resim_samples_np = [s.squeeze(-1).cpu().numpy() for s in s_resim_samples]

    # Extract coordinates for 1D plotting
    if X_np.ndim == 1:
        x_coords = X_np
        y_coords = Y_np
    else:
        x_coords = X_np[:, 0]
        y_coords = Y_np[:, 0]

    # Create 1D plot with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Observed output function
    axes[0].plot(
        y_coords, s_observed_np, "g-", label="Observed Output s(y)", linewidth=3
    )
    axes[0].set_title("Observed Output Function s(y)", fontsize=12)
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("s(y)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Input function realizations
    axes[1].plot(
        x_coords, u_true_np, "b-", label="True Input u(x)", linewidth=3, alpha=0.8
    )

    # Plot all realizations
    for i, u_sample_np in enumerate(u_samples_np):
        alpha = 0.6 if len(u_samples_np) > 1 else 0.8
        color = (
            "r"
            if len(u_samples_np) == 1
            else plt.cm.Reds(0.4 + 0.6 * i / max(1, len(u_samples_np) - 1))
        )
        label = f"Realization {i+1}" if len(u_samples_np) > 1 and i < 5 else None
        if len(u_samples_np) == 1:
            label = "Predicted Input û(x)"
        axes[1].plot(
            x_coords,
            u_sample_np,
            "--",
            color=color,
            linewidth=2,
            alpha=alpha,
            label=label,
        )

    axes[1].set_title(
        f"Input Function: True vs {len(u_samples_np)} Realizations", fontsize=12
    )
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
        linewidth=3,
        alpha=0.8,
    )

    # Plot all re-simulation realizations
    mse_values = []
    for i, s_resim_np in enumerate(s_resim_samples_np):
        alpha = 0.6 if len(s_resim_samples_np) > 1 else 0.8
        color = (
            "m"
            if len(s_resim_samples_np) == 1
            else plt.cm.Purples(0.4 + 0.6 * i / max(1, len(s_resim_samples_np) - 1))
        )
        label = f"Re-sim {i+1}" if len(s_resim_samples_np) > 1 and i < 5 else None
        if len(s_resim_samples_np) == 1:
            label = "Re-simulated ŝ(y)"
        axes[2].plot(
            y_coords,
            s_resim_np,
            "--",
            color=color,
            linewidth=2,
            alpha=alpha,
            label=label,
        )

        # Calculate error metrics
        mse = np.mean((s_observed_np - s_resim_np) ** 2)
        mse_values.append(mse)

    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values) if len(mse_values) > 1 else 0

    title = f"Re-simulation vs Observed\nMean MSE: {mean_mse:.6f}"
    if len(mse_values) > 1:
        title += f" ± {std_mse:.6f}"
    axes[2].set_title(title, fontsize=12)
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("s(y)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Detect if model is probabilistic by checking for variation in samples
    is_probabilistic = (
        len(set([tuple(u.flatten().tolist()) for u in u_samples_np[:2]])) > 1
        if len(u_samples_np) > 1
        else False
    )
    model_type = "Probabilistic" if is_probabilistic else "Deterministic"
    fig.suptitle(f"{model_name} ({model_type}) - Sample {sample_idx}", fontsize=14)

    plt.tight_layout()

    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{model_name}_probabilistic_sample_{sample_idx}.png"
        )
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
    n_realizations=10,
    save_dir=None,
):
    """Plot multiple random samples from the test set."""

    # Select random samples
    test_indices = random.sample(
        range(len(test_dataset)), min(n_samples, len(test_dataset))
    )

    for i, idx in enumerate(test_indices):
        sample = test_dataset[idx]
        plot_burgers_probabilistic_sample(
            model,
            evaluate_fn,
            input_function_encoder,
            output_function_encoder,
            forward_model,
            sample,
            idx,
            model_name,
            n_realizations,
            save_dir,
        )


def plot_model_results(
    model_name,
    log_dir,
    results_dir,
    test_dataset,
    dataset_info,
    n_samples=3,
    n_realizations=10,
):
    """Plot results for a single model.

    Args:
        model_name: Name of the model
        log_dir: Complete path to the model directory (e.g., /path/to/logs/dataset/model/seed_1)
        results_dir: Directory to save results
        test_dataset: Test dataset
        dataset_info: Dataset information
        n_samples: Number of samples to plot
        n_realizations: Number of posterior realizations to sample
    """

    # log_dir is now the complete path to the model directory
    model_log_dir = log_dir

    # Load model parameters
    params = torch.load(os.path.join(model_log_dir, "params.pth"), weights_only=False)

    # Load models
    input_function_encoder, output_function_encoder, model, evaluate_fn = load_models(
        log_dir=model_log_dir,
        dataset_info=dataset_info,
        params=params,
        device=device,
    )

    # Load forward model for re-simulation
    forward_model = load_forward_model(
        log_dir=model_log_dir, forward_model_name="b2b_nonlinear", device=device
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
        n_realizations=n_realizations,
        save_dir=results_dir,
    )


# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Plot Wave Scattering probabilistic results."
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="/store/at46867/b2b_operator_inverse",
    help="Complete path to model directory (e.g., /path/to/logs/dataset/model/seed_1)",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="results/burgers_probabilistic_plots",
    help="Results directory for saving plots",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=3,
    help="Number of random samples to plot per model",
)
parser.add_argument(
    "--n_realizations",
    type=int,
    default=10,
    help="Number of posterior realizations to sample",
)
parser.add_argument(
    "--seed", type=int, default=1, help="Random seed for reproducibility"
)
parser.add_argument(
    "--model", type=str, required=True, help="Model name to plot results for"
)

args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# log_dir is now the complete path to the model directory
log_dir = args.log_dir
results_dir = args.results_dir
model_name = args.model

# Check if model directory exists
if not os.path.exists(os.path.join(log_dir, "params.pth")):
    print(f"✗ Model not found at {log_dir}")
    exit(1)

print(f"Loading model parameters and dataset...")
# Load dataset using the model's parameters
params = torch.load(os.path.join(log_dir, "params.pth"), weights_only=False)
test_dataset, dataset_info = load_dataset(
    params.dataset, params, device, split="test", return_info=True
)
print(f"✓ Loaded {len(test_dataset)} test samples")

# Create results directory
os.makedirs(results_dir, exist_ok=True)

print(
    f"Generating {args.n_samples} probabilistic plots ({args.n_realizations} realizations each)..."
)
# Plot results for the specified model
plot_model_results(
    model_name=model_name,
    log_dir=log_dir,
    results_dir=results_dir,
    test_dataset=test_dataset,
    dataset_info=dataset_info,
    n_samples=args.n_samples,
    n_realizations=args.n_realizations,
)

print(f"✓ Generated {args.n_samples} probabilistic plots → {results_dir}")
