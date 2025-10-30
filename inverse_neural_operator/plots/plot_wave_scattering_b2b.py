"""
Publication-quality plotting script for Wave Scattering B2B model performance.

Creates a 4x4 gridspec layout showing:
- Row 1: Input function encoder reconstructions (3 polar plots + colorbar)
- Row 2: Output function encoder reconstructions (3 pairs: prediction|error + colorbar)
- Row 3: Linear B2B forward model (3 pairs: prediction|error + colorbar)
- Row 4: Nonlinear B2B forward model (3 pairs: prediction|error + colorbar)

Each row has a common parent axis with shared labels and title.

To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_wave_scattering_b2b
"""

import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import random

import torch

from skimage.metrics import structural_similarity as ssim

from b2b.load_model import (
    load_function_encoders,
    load_forward_model,
)
from data.load_dataset import load_dataset
from data.process_data import InputFunctionEncoderDataset, OutputFunctionEncoderDataset
from plots.utils.plot_utils import setup_publication_style

device = "cpu"
GRID_SIZE = 200  # Wave scattering output grid size


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
    Create publication-quality 4x4 gridspec plot for B2B performance.
    Each column shows a different random realization (plus colorbar column).
    Row 0: input function encoder (polar plots)
    Rows 1-3: output function encoder, linear B2B, nonlinear B2B (2D imshow: prediction|error pairs)

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
    fig = plt.figure(figsize=(6.5, 5.5))
    # fig.set_constrained_layout_pads(
    #     w_pad=5.0 / 72.0, h_pad=5.0 / 72.0, hspace=0.0, wspace=0.0
    # )

    # Create gridspec: 4 rows × 4 columns (3 data + 1 colorbar)
    gs = fig.add_gridspec(
        4,
        4,
        hspace=0.35,
        wspace=0.05,
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
        width_ratios=[1, 1, 1, 0.05],
        height_ratios=[1.5, 1, 1, 1],
    )

    # Row titles and labels
    row_configs = [
        {
            "title": "Wave Scattering Dataset Input Function Encoder Reconstructions",
            "xlabel": "",
            "ylabel": r"$|u(\theta)|$",
        },
        {
            "title": "Wave Scattering Dataset Output Function Encoder Reconstructions",
            "xlabel": r"$x$",
            "ylabel": r"$y$",
        },
        {
            "title": "Wave Scattering Dataset Linear B2B Forward Model",
            "xlabel": r"$x$",
            "ylabel": r"$y$",
        },
        {
            "title": "Wave Scattering Dataset Nonlinear B2B Forward Model",
            "xlabel": r"$x$",
            "ylabel": r"$y$",
        },
    ]

    # Create parent axes for each row
    parent_axes = []
    for row_idx, config in enumerate(row_configs):
        ax_parent = fig.add_subplot(gs[row_idx, :3], frameon=False)
        ax_parent.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )
        if config["xlabel"]:
            ax_parent.set_xlabel(config["xlabel"], labelpad=0 if row_idx == 3 else -8)
        if config["ylabel"]:
            ax_parent.set_ylabel(config["ylabel"], labelpad=8)
        ax_parent.set_title(config["title"], pad=12 if row_idx == 0 else 4)
        parent_axes.append(ax_parent)

    # Collect all data for consistent limits within each row
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

            # Store data for plotting (convert to numpy)
            X_np = X.cpu().numpy().squeeze()
            u_true_np = u.cpu().numpy().squeeze()
            u_recon_np = u_recon.cpu().numpy().squeeze()
            s_true_np = s_true.cpu().numpy().squeeze()
            s_recon_np = s_recon.cpu().numpy().squeeze()
            s_linear_np = s_linear.cpu().numpy().squeeze()
            s_nonlinear_np = s_nonlinear.cpu().numpy().squeeze()

            # Extract theta for polar plot
            theta = np.arctan2(X_np[:, 1], X_np[:, 0])

            row_data[0].append((theta, u_true_np, u_recon_np))
            row_data[1].append((s_true_np, s_recon_np))
            row_data[2].append((s_true_np, s_linear_np))
            row_data[3].append((s_true_np, s_nonlinear_np))

    # Compute vmin/vmax for each 2D row (rows 1-3) for shared colorbars
    row_vlims_pred = []
    row_vlims_error = []
    for row_idx in range(1, 4):
        v_min_pred, v_max_pred = float("inf"), float("-inf")
        v_min_err, v_max_err = float("inf"), float("-inf")
        for s_true, s_pred in row_data[row_idx]:
            # Reshape and threshold to binary density fields
            s_true_2d = s_true.reshape(GRID_SIZE, GRID_SIZE)
            s_pred_2d = s_pred.reshape(GRID_SIZE, GRID_SIZE)
            s_true_binary = (s_true_2d > 0.5).astype(float)
            s_pred_binary = (s_pred_2d > 0.5).astype(float)

            v_min_pred = min(v_min_pred, s_pred_binary.min())
            v_max_pred = max(v_max_pred, s_pred_binary.max())
            error = np.abs(s_true_binary - s_pred_binary)
            v_min_err = min(v_min_err, error.min())
            v_max_err = max(v_max_err, error.max())
        row_vlims_pred.append((v_min_pred, v_max_pred))
        row_vlims_error.append((v_min_err, v_max_err))

    # Plot row 0 (input function encoder - polar plots)
    for col_idx in range(n_realizations):
        ax = fig.add_subplot(gs[0, col_idx], projection="polar")
        theta, u_true, u_recon = row_data[0][col_idx]

        # Plot ground truth as gray dashed, reconstruction as C0 solid
        ax.plot(theta, np.abs(u_true), "--", color="gray", linewidth=1.0, alpha=0.8)
        ax.plot(theta, np.abs(u_recon), "-", color="C0", linewidth=1.0, alpha=0.8)

        # # Compute MSE
        # mse = np.mean((u_true - u_recon) ** 2)

        # Compute L2 error
        l2_error = np.linalg.norm(u_true - u_recon) / np.linalg.norm(u_true)

        # Add MSE annotation
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

        ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.6)

        # Add tick labels to all polar plots
        ax.tick_params(
            labelsize=5,
            pad=-3,
        )

    # No colorbar for row 0 (polar plots)
    fig.add_subplot(gs[0, 3]).axis("off")

    # Plot rows 1-3 (output encoder, linear/nonlinear B2B - 2D imshow)
    for row_idx in range(1, 4):
        for col_idx in range(n_realizations):
            # Create nested gridspec for prediction|error side-by-side
            cell_gs = GridSpecFromSubplotSpec(
                1, 2, subplot_spec=gs[row_idx, col_idx], wspace=0.0, hspace=0.0
            )

            s_true, s_pred = row_data[row_idx][col_idx]

            # Reshape to 2D
            s_true_2d = s_true.reshape(GRID_SIZE, GRID_SIZE)
            s_pred_2d = s_pred.reshape(GRID_SIZE, GRID_SIZE)

            # Create thresholded versions (binary density fields)
            s_true_binary = (s_true_2d > 0.5).astype(float)
            s_pred_binary = (s_pred_2d > 0.5).astype(float)

            # Compute error between binary density fields
            error_2d = np.abs(s_true_binary - s_pred_binary)

            # Left: predicted binary density field
            ax_pred = fig.add_subplot(cell_gs[0])
            ax_pred.imshow(
                s_pred_binary,
                cmap="viridis",
                origin="lower",
                extent=[0, 1, 0, 1],
                vmin=row_vlims_pred[row_idx - 1][0],
                vmax=row_vlims_pred[row_idx - 1][1],
            )

            # Show y-axis labels on left column, x-axis labels on bottom row
            show_left = col_idx == 0
            show_bottom = row_idx == 3
            ax_pred.tick_params(
                labelbottom=show_bottom,
                labelleft=show_left,
                length=2,
                width=0.5,
                labelsize=5,
            )
            ax_pred.set_aspect("equal")

            # Right: error in binary density field
            ax_err = fig.add_subplot(cell_gs[1])
            im_err = ax_err.imshow(
                error_2d,
                cmap="Reds",
                origin="lower",
                extent=[0, 1, 0, 1],
                vmin=row_vlims_error[row_idx - 1][0],
                vmax=row_vlims_error[row_idx - 1][1],
            )
            ax_err.set_xticks([])
            ax_err.set_yticks([])
            ax_err.set_aspect("equal")

            # Compute SSIM between binary density fields
            ssim_value = ssim(
                s_true_binary,
                s_pred_binary,
                data_range=s_true_binary.max() - s_true_binary.min(),
            )

            # Add SSIM annotation to error plot
            ax_err.text(
                0.95,
                0.95,
                f"SSIM: {ssim_value:.3f}",
                transform=ax_err.transAxes,
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

        # Add shared colorbar for this row (column 3)
        # Use the last im_err for colorbar (they all share same vmin/vmax)
        cax = fig.add_subplot(gs[row_idx, 3])
        fig.colorbar(im_err, cax=cax, use_gridspec=True)

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
    description="Create publication-quality B2B performance plot for Wave Scattering dataset."
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
    default="results",
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
dataset_name = "wave_scattering"

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
    save_path=os.path.join(shared_results_dir, "wave_scattering_b2b_publication.png"),
)

print(f"✓ Publication-quality B2B plot generated → {shared_results_dir}")
