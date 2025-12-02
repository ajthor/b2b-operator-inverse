"""
Publication-quality plotting script for Burgers B2B model performance.

Creates a 4x3 gridspec layout showing:
- Row 1: Input function encoder reconstructions (3 random realizations)
- Row 2: Output function encoder reconstructions (3 random realizations)
- Row 3: Linear B2B forward model predictions (3 random realizations)
- Row 4: Nonlinear B2B forward model predictions (3 random realizations)

Each row has a common parent axis with shared labels and title.

To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_burgers_b2b
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random

import torch

from b2b.load_model import (
    load_function_encoders,
    load_forward_model,
)
from data.load_dataset import load_dataset
from data.process_data import InputFunctionEncoderDataset, OutputFunctionEncoderDataset
from plots.utils.plot_utils import setup_publication_style

device = "cpu"


def plot_b2b_publication_grid(
    input_function_encoder,
    output_function_encoder,
    linear_forward_model,
    nonlinear_forward_model,
    input_encoder_dataset,
    output_encoder_dataset,
    test_dataset,
    n_realizations=3,
    seed=42,
    save_path=None,
):
    """
    Create publication-quality 4x3 gridspec plot for B2B performance.
    Each column shows a different random realization.
    Each row shows: input function encoder, output function encoder, linear B2B, nonlinear B2B.

    Args:
        input_function_encoder: Input function encoder model
        output_function_encoder: Output function encoder model
        linear_forward_model: Linear B2B forward model
        nonlinear_forward_model: Nonlinear B2B forward model
        input_encoder_dataset: Input encoder dataset
        output_encoder_dataset: Output encoder dataset
        test_dataset: Test dataset
        n_realizations: Number of random realizations to show (columns)
        seed: Random seed
        save_path: Path to save the figure
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set models to eval mode
    input_function_encoder.eval()
    output_function_encoder.eval()
    linear_forward_model.eval()
    nonlinear_forward_model.eval()

    # Randomly select samples
    indices = random.sample(
        range(len(test_dataset)), min(n_realizations, len(test_dataset))
    )

    # Set up publication style
    setup_publication_style()

    # Create figure with constrained layout
    fig = plt.figure(figsize=(6.5, 4.5), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=0.5 / 72.0, h_pad=0.5 / 72.0, hspace=0.0, wspace=0.0
    )

    # Create gridspec: 4 rows x n_realizations columns with minimal spacing
    gs = fig.add_gridspec(
        4,
        n_realizations,
        hspace=0.1,
        wspace=0.0,
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
    )

    # Row titles and labels
    row_configs = [
        {
            "title": "Burgers Dataset Input Function Encoder Reconstructions",
            "xlabel": r"$x$",
            "ylabel": r"$u(x)$",
        },
        {
            "title": "Burgers Dataset Output Function Encoder Reconstructions",
            "xlabel": r"$y$",
            "ylabel": r"$s(y)$",
        },
        {
            "title": "Burgers Dataset Linear B2B Forward Model",
            "xlabel": r"$y$",
            "ylabel": r"$s(y)$",
        },
        {
            "title": "Burgers Dataset Nonlinear B2B Forward Model",
            "xlabel": r"$y$",
            "ylabel": r"$s(y)$",
        },
    ]

    # Create parent axes for each row
    parent_axes = []
    for row_idx, config in enumerate(row_configs):
        ax_parent = fig.add_subplot(gs[row_idx, :], frameon=False)
        ax_parent.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )
        ax_parent.set_xlabel(config["xlabel"], labelpad=0 if row_idx == 3 else -8)
        ax_parent.set_ylabel(config["ylabel"], labelpad=4)
        ax_parent.set_title(config["title"], pad=4)
        parent_axes.append(ax_parent)

    # Collect all data for consistent y-limits within each row
    row_data = [[] for _ in range(4)]

    with torch.no_grad():
        for col_idx, sample_idx in enumerate(indices):
            # Get sample
            X, u, Y, s_true = test_dataset[sample_idx]
            X = X.to(device)
            u = u.to(device)
            Y = Y.to(device)
            s_true = s_true.to(device)

            # Add batch dimension
            X_batch = X.unsqueeze(0)
            u_batch = u.unsqueeze(0)
            Y_batch = Y.unsqueeze(0)

            # Get input encoder reconstruction
            example_xs_input, example_ys_input, xs_input, ys_input = (
                input_encoder_dataset[sample_idx]
            )
            example_xs_input = example_xs_input.unsqueeze(0).to(device)
            example_ys_input = example_ys_input.unsqueeze(0).to(device)
            alpha, _ = input_function_encoder.compute_coefficients(
                example_xs_input, example_ys_input
            )
            u_recon = input_function_encoder(X_batch, alpha).squeeze(0)

            # Get output encoder reconstruction
            example_xs_output, example_ys_output, xs_output, ys_output = (
                output_encoder_dataset[sample_idx]
            )
            example_xs_output = example_xs_output.unsqueeze(0).to(device)
            example_ys_output = example_ys_output.unsqueeze(0).to(device)
            beta, _ = output_function_encoder.compute_coefficients(
                example_xs_output, example_ys_output
            )
            s_recon = output_function_encoder(Y_batch, beta).squeeze(0)

            # Get linear B2B prediction
            beta_linear = linear_forward_model.forward(alpha)
            s_linear = output_function_encoder(Y_batch, beta_linear).squeeze(0)

            # Get nonlinear B2B prediction
            beta_nonlinear = nonlinear_forward_model.forward(alpha)
            s_nonlinear = output_function_encoder(Y_batch, beta_nonlinear).squeeze(0)

            # Store data for plotting
            x_coords = X.cpu().numpy().squeeze()
            y_coords = Y.cpu().numpy().squeeze()
            u_true_np = u.cpu().numpy().squeeze()
            s_true_np = s_true.cpu().numpy().squeeze()
            u_recon_np = u_recon.cpu().numpy().squeeze()
            s_recon_np = s_recon.cpu().numpy().squeeze()
            s_linear_np = s_linear.cpu().numpy().squeeze()
            s_nonlinear_np = s_nonlinear.cpu().numpy().squeeze()

            row_data[0].append((x_coords, u_true_np, u_recon_np))
            row_data[1].append((y_coords, s_true_np, s_recon_np))
            row_data[2].append((y_coords, s_true_np, s_linear_np))
            row_data[3].append((y_coords, s_true_np, s_nonlinear_np))

    # Compute y-limits for each row
    row_ylims = []
    for row_idx in range(4):
        y_min = float("inf")
        y_max = float("-inf")
        for data in row_data[row_idx]:
            coords, true_vals, pred_vals = data
            y_min = min(y_min, true_vals.min())
            y_max = max(y_max, true_vals.max())
            y_min = min(y_min, pred_vals.min())
            y_max = max(y_max, pred_vals.max())
        y_range = y_max - y_min
        row_ylims.append((y_min - 0.05 * y_range, y_max + 0.05 * y_range))

    # Plot all subplots
    for row_idx in range(4):
        for col_idx in range(n_realizations):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            coords, true_vals, pred_vals = row_data[row_idx][col_idx]

            # Plot ground truth as dashed gray line and prediction/reconstruction
            # Use different color for each row: C0 (input), C1 (output), C2 (linear), C3 (nonlinear)
            colors = ["C0", "C1", "C2", "C3"]
            color = colors[row_idx]
            ax.plot(coords, true_vals, "--", color="gray", linewidth=1.0, alpha=0.8)
            ax.plot(coords, pred_vals, "-", color=color, linewidth=1.0, alpha=0.8)

            # Compute relative L2 error
            l2_error = np.linalg.norm(true_vals - pred_vals) / np.linalg.norm(true_vals)

            # Add L2 error text annotation with dark gray background
            ax.text(
                0.95,
                0.95,
                f"L2: {l2_error:.2e}",
                transform=ax.transAxes,
                fontsize=5,
                color="white",
                va="top",
                ha="right",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#333333",
                    alpha=0.8,
                    edgecolor="none",
                ),
            )

            # Set limits
            ax.set_ylim(row_ylims[row_idx])
            ax.set_xlim(coords.min(), coords.max())

            # Add tick labels only on left column and bottom row
            show_left = col_idx == 0
            show_bottom = row_idx == 3
            ax.tick_params(
                labelbottom=show_bottom,
                labelleft=show_left,
                length=2,
                width=0.5,
                labelsize=5,
            )

            # Add grid
            ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.6)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
        # Also save as PDF
        pdf_path = save_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {pdf_path}")

    plt.close()


# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Create publication-quality B2B performance plot for Burgers 1D dataset."
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="/store/at46867/b2b_operator_inverse",
    help="Base log directory containing dataset subdirectories",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="runs",
    help="Base results directory for saving plots",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
parser.add_argument(
    "--n_realizations",
    type=int,
    default=3,
    help="Number of random realizations to show (columns, default: 3)",
)

args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Dataset name
dataset_name = "burgers_1d"

# Construct paths
shared_log_dir = os.path.join(args.log_dir, dataset_name, "shared", "seed_1")
shared_results_dir = os.path.join(args.results_dir, dataset_name, "shared", "seed_1")

# Check if shared directory exists
if not os.path.exists(os.path.join(shared_log_dir, "input_function_encoder.pth")):
    print(f"✗ Function encoders not found at {shared_log_dir}")
    exit(1)

print(f"Loading function encoders and dataset...")
# Load params
params = torch.load(os.path.join(shared_log_dir, "params.pth"), weights_only=False)

# Load dataset
test_dataset, dataset_info = load_dataset(
    dataset_name, params, device, split="test", return_info=True
)
print(f"✓ Loaded {len(test_dataset)} test samples")

# Create function encoder datasets
input_encoder_dataset = InputFunctionEncoderDataset(test_dataset, device=device)
output_encoder_dataset = OutputFunctionEncoderDataset(test_dataset, device=device)
print(f"✓ Created function encoder datasets")

# Load function encoders
input_function_encoder, output_function_encoder = load_function_encoders(
    log_dir=shared_log_dir,
    dataset_info=dataset_info,
    params=params,
    device=device,
)
print(f"✓ Loaded function encoders")

# Load forward models
print(f"Loading forward models...")
linear_forward_model = load_forward_model(
    log_dir=shared_log_dir,
    forward_model_name="b2b_linear",
    device=device,
)
print(f"✓ Loaded linear forward model")

nonlinear_forward_model = load_forward_model(
    log_dir=shared_log_dir,
    forward_model_name="b2b_nonlinear",
    device=device,
)
print(f"✓ Loaded nonlinear forward model")

# Create results directory
os.makedirs(shared_results_dir, exist_ok=True)

# Generate publication-quality B2B plot
print(f"Generating publication-quality B2B plot...")
plot_b2b_publication_grid(
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
    linear_forward_model=linear_forward_model,
    nonlinear_forward_model=nonlinear_forward_model,
    input_encoder_dataset=input_encoder_dataset,
    output_encoder_dataset=output_encoder_dataset,
    test_dataset=test_dataset,
    n_realizations=args.n_realizations,
    seed=args.seed,
    save_path=os.path.join(shared_results_dir, "burgers_b2b_publication.png"),
)

print(f"✓ Publication-quality B2B plot generated → {shared_results_dir}")
