#!/usr/bin/env python3
"""
Standalone script to visualize IFNO results on the Burgers 1D dataset.
This mirrors the Darcy visualization script but adapts labels to Burgers physics.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from models.ifno import create_model
from data.burgers_1d import load_data

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = "cuda:1" if torch.cuda.is_available() else "cpu"


def _infer_ifno_config_from_state_dict_path(state_dict_path: str):
    """Infer IFNO hyperparameters from a saved state_dict path."""
    try:
        sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    cfg = {}
    # width from p1 weight out_features (symmetric problems)
    if "p1.weight" in sd:
        cfg["width"] = int(sd["p1.weight"].shape[0])
    # n_layers from number of convs entries (two convs per layer)
    conv_keys = [
        k for k in sd.keys() if k.startswith("convs.") and k.endswith(".weights")
    ]
    if conv_keys:
        cfg["n_layers"] = len(conv_keys) // 2
        # modes from conv weight last dim
        sample_conv = sd[conv_keys[0]]
        cfg["modes1"] = int(sample_conv.shape[-1])
        cfg["modes2"] = cfg["modes1"]
    # vae_latent_dim from fc_mu weight out_features
    if "vae_net.fc_mu.weight" in sd:
        cfg["vae_latent_dim"] = int(sd["vae_net.fc_mu.weight"].shape[0])
    return cfg if cfg else None


def visualize_ifno_results(
    model_path, n_samples=3, save_dir="results/ifno_plots_burgers/"
):
    """Visualize IFNO results on Burgers 1D dataset."""

    print("Loading Burgers 1D test dataset...")
    test_dataset = load_data(None, device=device, split="test")
    dataset_info = test_dataset.get_info()

    print(f"Dataset info: {dataset_info}")

    print("Creating IFNO model...")
    inferred_cfg = _infer_ifno_config_from_state_dict_path(model_path)
    if inferred_cfg:
        print(f"Inferred model config from checkpoint: {inferred_cfg}")
    model = create_model(
        input_size=None,  # Not used by IFNO
        hidden_sizes=[256, 256, 256],  # Not used by IFNO
        n_coupling_layers=2,
        modes1=(inferred_cfg.get("modes1") if inferred_cfg else 8),
        modes2=(inferred_cfg.get("modes2") if inferred_cfg else 8),
        width=(inferred_cfg.get("width") if inferred_cfg else 24),
        beta=2.0,
        n_layers=(inferred_cfg.get("n_layers") if inferred_cfg else 3),
        padding=20,
        vae_latent_dim=(inferred_cfg.get("vae_latent_dim") if inferred_cfg else 8),
        intermediate_dim=32,
        # IFNO-specific parameters from dataset info
        input_spatial_dims=dataset_info["input_spatial_dims"],
        output_spatial_dims=dataset_info["output_spatial_dims"],
        input_function_channels=dataset_info["input_function_channels"],
        output_function_channels=dataset_info["output_function_channels"],
        coordinate_dim=dataset_info["coordinate_dim"],
    ).to(device)

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Select random samples
    test_indices = random.sample(
        range(len(test_dataset)), min(n_samples, len(test_dataset))
    )

    print(f"Generating {n_samples} visualization plots...")

    for i, idx in enumerate(test_indices):
        sample = test_dataset[idx]
        plot_sample(model, sample, idx, save_dir)

    print(f"Visualization completed! Plots saved to: {save_dir}")


def plot_sample(model, sample, sample_idx, save_dir):
    """Plot a single Burgers 1D sample with IFNO results for both forward and inverse problems."""

    X, u_initial, Y, u_final = sample

    with torch.no_grad():
        # === FORWARD PREDICTION: initial -> final ===
        # For forward problem: given initial condition, predict final solution
        X_batch = X.unsqueeze(0)  # Add batch dimension
        u_initial_batch = u_initial.unsqueeze(0)
        forward_input = torch.cat([X_batch, u_initial_batch], dim=-1)

        forward_result = model.inverse(forward_input)

        if isinstance(forward_result, tuple):
            u_final_pred, _ = forward_result
        else:
            u_final_pred = forward_result

        if u_final_pred.shape[-1] > u_final.unsqueeze(0).shape[-1]:
            u_final_pred = u_final_pred[..., -u_final.unsqueeze(0).shape[-1] :]

        u_final_pred = u_final_pred.squeeze(0)

        # === INVERSE PREDICTION: final -> initial ===
        # For inverse problem: given final solution, predict initial condition
        Y_batch = Y.unsqueeze(0)
        u_final_batch = u_final.unsqueeze(0)
        inverse_input = torch.cat([Y_batch, u_final_batch], dim=-1)

        inverse_result = model(inverse_input)

        if isinstance(inverse_result, tuple):
            u_initial_pred, _ = inverse_result
        else:
            u_initial_pred = inverse_result

        if u_initial_pred.shape[-1] > u_initial.unsqueeze(0).shape[-1]:
            u_initial_pred = u_initial_pred[..., -u_initial.unsqueeze(0).shape[-1] :]

        u_initial_pred = u_initial_pred.squeeze(0)

    # Convert to numpy for plotting
    u_initial_np = u_initial.squeeze(-1).cpu().numpy()
    u_initial_pred_np = u_initial_pred.squeeze(-1).cpu().numpy()
    u_final_np = u_final.squeeze(-1).cpu().numpy()
    u_final_pred_np = u_final_pred.squeeze(-1).cpu().numpy()
    X_np = X.squeeze(-1).cpu().numpy()
    Y_np = Y.squeeze(-1).cpu().numpy()

    # Extract x coordinates for 1D plotting
    x_coords = X_np if X_np.ndim == 1 else X_np[:, 0]
    y_coords = Y_np if Y_np.ndim == 1 else Y_np[:, 0]

    # === FORWARD PLOT (initial -> final) ===
    fig_fwd, axes_fwd = plt.subplots(1, 2, figsize=(12, 5))

    axes_fwd[0].plot(x_coords, u_initial_np, "b-", label="Initial Condition $u_0(x)$")
    axes_fwd[0].set_title("Initial Condition $u_0(x)$")
    axes_fwd[0].set_xlabel("x")
    axes_fwd[0].set_ylabel("$u_0(x)$")
    axes_fwd[0].legend()
    axes_fwd[0].grid(True)

    axes_fwd[1].plot(y_coords, u_final_np, "g-", label="True Final $u_T(x)$", alpha=0.7)
    axes_fwd[1].plot(
        y_coords,
        u_final_pred_np,
        "orange",
        linestyle="--",
        label="Predicted Final $\\hat{u}_T(x)$",
        alpha=0.7,
    )
    axes_fwd[1].set_title("Final State: True vs Predicted")
    axes_fwd[1].set_xlabel("x")
    axes_fwd[1].set_ylabel("$u_T(x)$")
    axes_fwd[1].legend()
    axes_fwd[1].grid(True)

    plt.tight_layout()

    save_path_fwd = os.path.join(save_dir, f"ifno_forward_sample_{sample_idx}.png")
    plt.savefig(save_path_fwd, dpi=300, bbox_inches="tight")
    print(f"Saved forward plot: {save_path_fwd}")
    plt.close()

    # === INVERSE PLOT (final -> initial) ===
    fig_inv, axes_inv = plt.subplots(1, 2, figsize=(12, 5))

    axes_inv[0].plot(y_coords, u_final_np, "g-", label="Observed Final $u_T(x)$")
    axes_inv[0].set_title("Observed Final State $u_T(x)$")
    axes_inv[0].set_xlabel("x")
    axes_inv[0].set_ylabel("$u_T(x)$")
    axes_inv[0].legend()
    axes_inv[0].grid(True)

    axes_inv[1].plot(
        x_coords, u_initial_np, "b-", label="True Initial $u_0(x)$", alpha=0.7
    )
    axes_inv[1].plot(
        x_coords,
        u_initial_pred_np,
        "r--",
        label="Predicted Initial $\\hat{u}_0(x)$",
        alpha=0.7,
    )
    axes_inv[1].set_title("Initial State: True vs Predicted")
    axes_inv[1].set_xlabel("x")
    axes_inv[1].set_ylabel("$u_0(x)$")
    axes_inv[1].legend()
    axes_inv[1].grid(True)

    plt.tight_layout()

    save_path_inv = os.path.join(save_dir, f"ifno_inverse_sample_{sample_idx}.png")
    plt.savefig(save_path_inv, dpi=300, bbox_inches="tight")
    print(f"Saved inverse plot: {save_path_inv}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize IFNO results on Burgers 1D dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="logs_ifno/burgers_1d/seed_0/ifno_model.pth",
        help="Path to trained IFNO model",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/burgers_1d/ifno/",
        help="Directory to save visualization plots",
    )

    args = parser.parse_args()

    print("IFNO Burgers 1D Visualization")
    print("=" * 50)

    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print(
            "Please train the model first using: python train_ifno_standalone.py --dataset burgers_1d"
        )
        raise SystemExit(1)

    visualize_ifno_results(
        model_path=args.model_path,
        n_samples=args.n_samples,
        save_dir=args.save_dir,
    )
