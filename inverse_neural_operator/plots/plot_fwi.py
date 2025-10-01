import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch

from inverse_neural_operator.models.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)

from data.load_dataset import load_dataset
from models.load_model import load_models

from inverse_neural_operator.models.model_evaluation import (
    evaluate_random,
    find_best_case,
    find_worst_case,
)

device = "cpu"

torch.manual_seed(1)


# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot results.")
parser.add_argument("--model", type=str, default="variational_autoencoder")
parser.add_argument(
    "--log_dir", type=str, default="/store/at46867/b2b_operator_inverse"
)
parser.add_argument(
    "--results_dir", type=str, default="results/burgers_1d/variational_autoencoder"
)

args = parser.parse_args()

log_dir = args.log_dir
model_name = args.model
dataset = "fwi"
results_dir = args.results_dir

# Construct path to dataset directory
log_dir = os.path.join(log_dir, dataset)

# Load params from specific model/seed
model_log_dir = os.path.join(log_dir, model_name, "seed_1")
params = torch.load(os.path.join(model_log_dir, "params.pth"), weights_only=False)


# Load dataset
test_dataset, dataset_info = load_dataset(
    params.dataset, params, device, split="test", return_info=True
)

# Load models
input_function_encoder, output_function_encoder, model, evaluate_fn = load_models(
    model_log_dir,
    dataset_info,
    params,
    device=device,
)


def plot_fwi_sample(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    sample,
    sample_idx,
    model_name,
    save_dir=None,
):
    """
    Plot a single FWI sample with 5-panel layout:
    1. True velocity model (input)
    2. Predicted velocity model
    3. Measured seismic transform (output)
    4. Re-simulated seismic transform
    5. Error between measured and re-simulated seismic transforms
    """
    model.eval()
    if forward_model is not None:
        forward_model.eval()

    X, u_true, Y, s_observed = sample

    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_observed = s_observed.to(device)

    # Get model prediction for input velocity model
    with torch.no_grad():
        point = (
            X.unsqueeze(0),
            u_true.unsqueeze(0),
            Y.unsqueeze(0),
            s_observed.unsqueeze(0),
        )
        u_pred = evaluate_fn(
            model, point, input_function_encoder, output_function_encoder
        )
        u_pred = u_pred.squeeze(0)

    # Re-simulate using forward model if available
    s_resim = None
    if forward_model is not None:
        with torch.no_grad():
            # Add batch dimension for forward model
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

    # Convert to numpy for plotting
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    u_pred_np = u_pred.squeeze(-1).cpu().numpy()
    s_observed_np = s_observed.squeeze(-1).cpu().numpy()

    # Reshape velocity models from flattened (1152,) to 2D (24, 48)
    u_true_2d = u_true_np.reshape(24, 48)
    u_pred_2d = u_pred_np.reshape(24, 48)

    # Reshape seismic transforms from flattened (30400,) to 2D (400, 76)
    s_observed_2d = s_observed_np.reshape(400, 76)

    # Create figure with subplots
    n_panels = 4 if s_resim is None else 5
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: True Velocity Model (Waterfall Plot)
    x_coords = np.arange(48)  # x-axis coordinates
    offset_scale = np.max(u_true_2d) - np.min(u_true_2d)
    for i in range(24):  # Each row (y-coordinate)
        y_offset = i * offset_scale * 0.3  # Vertical offset for waterfall effect
        axes[0].plot(x_coords, u_true_2d[i, :] + y_offset, linewidth=1, alpha=0.8)
    axes[0].set_title("True Velocity Model u(x,y)", fontsize=12)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y (with offset)")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Predicted Velocity Model (Waterfall Plot)
    offset_scale = np.max(u_pred_2d) - np.min(u_pred_2d)
    for i in range(24):  # Each row (y-coordinate)
        y_offset = i * offset_scale * 0.3  # Vertical offset for waterfall effect
        axes[1].plot(x_coords, u_pred_2d[i, :] + y_offset, linewidth=1, alpha=0.8)
    axes[1].set_title("Predicted Velocity Model û(x,y)", fontsize=12)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y (with offset)")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Measured Seismic Transform
    im3 = axes[2].imshow(s_observed_2d, cmap="magma", aspect="auto", origin="lower")
    axes[2].set_title("Measured Seismic Transform s(f,t)", fontsize=12)
    axes[2].set_xlabel("Frequency")
    axes[2].set_ylabel("Time")
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    if s_resim is not None:
        s_resim_np = s_resim.squeeze(-1).cpu().numpy()
        s_resim_2d = s_resim_np.reshape(400, 76)

        # Panel 4: Re-simulated Seismic Transform
        im4 = axes[3].imshow(s_resim_2d, cmap="magma", aspect="auto", origin="lower")
        axes[3].set_title("Re-simulated Seismic Transform ŝ(f,t)", fontsize=12)
        axes[3].set_xlabel("Frequency")
        axes[3].set_ylabel("Time")
        plt.colorbar(im4, ax=axes[3], fraction=0.046)

        # Panel 5: Error Field
        error_2d = np.abs(s_observed_2d - s_resim_2d)
        mse_resim = np.mean((s_observed_2d - s_resim_2d) ** 2)
        mae_resim = np.mean(np.abs(s_observed_2d - s_resim_2d))

        im5 = axes[4].imshow(error_2d, cmap="Reds", aspect="auto", origin="lower")
        axes[4].set_title(
            f"Re-simulation Error |s - ŝ|\nMSE: {mse_resim:.6f}, MAE: {mae_resim:.6f}",
            fontsize=12,
        )
        axes[4].set_xlabel("Frequency")
        axes[4].set_ylabel("Time")
        plt.colorbar(im5, ax=axes[4], fraction=0.046)

    plt.tight_layout()

    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_name}_sample_{sample_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    plt.show()
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
):
    """Plot multiple random samples from the test set."""
    import random

    # Select random samples
    test_indices = random.sample(
        range(len(test_dataset)), min(n_samples, len(test_dataset))
    )

    for idx in test_indices:
        sample = test_dataset[idx]
        plot_fwi_sample(
            model,
            evaluate_fn,
            input_function_encoder,
            output_function_encoder,
            forward_model,
            sample,
            idx,
            model_name,
            save_dir,
        )


# Try to load forward model for re-simulation (may not exist)
forward_model = None
try:
    from models.load_model import load_forward_model

    forward_model = load_forward_model(log_dir=model_log_dir, device=device)
    print("Loaded forward model for re-simulation")
except (FileNotFoundError, ImportError) as e:
    print(f"Forward model not available: {e}")

# Plot results
plot_multiple_samples(
    model=model,
    evaluate_fn=evaluate_fn,
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
    forward_model=forward_model,
    test_dataset=test_dataset,
    model_name=model_name,
    n_samples=3,
    save_dir=results_dir,
)

print(f"SUCCESS: Plotted {model_name} FWI results")
