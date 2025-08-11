#!/usr/bin/env python3
"""
Standalone script to visualize IFNO results on Burgers 1D dataset.
This doesn't interfere with the existing plotting infrastructure.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from inverse_neural_operator.models.ifno import create_model
from inverse_neural_operator.data.burgers_1d import load_data

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = "cuda:1" if torch.cuda.is_available() else "cpu"


def _infer_ifno_config_from_state_dict_path(state_dict_path: str):
    """Infer IFNO hyperparameters from a saved state_dict path."""
    try:
        sd = torch.load(state_dict_path, map_location="cpu")
    except Exception:
        return None
    cfg = {}
    # width from p1 weight out_features (symmetric problems)
    if "p1.weight" in sd:
        cfg["width"] = int(sd["p1.weight"].shape[0])
    # n_layers from number of convs entries (two convs per layer)
    conv_keys = [k for k in sd.keys() if k.startswith("convs.") and k.endswith(".weights")]
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


def visualize_ifno_results(model_path, n_samples=3, save_dir="results/ifno_plots_burger/"):
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Select random samples
    test_indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))

    print(f"Generating {n_samples} visualization plots...")

    for i, idx in enumerate(test_indices):
        sample = test_dataset[idx]
        plot_sample(model, sample, idx, save_dir)

    print(f"Visualization completed! Plots saved to: {save_dir}")


def plot_sample(model, sample, sample_idx, save_dir):
    """Plot a single Burgers 1D sample with IFNO results for both forward and inverse problems."""

    X, u_true, Y, s = sample

    with torch.no_grad():
        # === FORWARD PREDICTION: s -> u ===
        Y_batch = Y.unsqueeze(0)  # Add batch dimension
        s_batch = s.unsqueeze(0)  # Add batch dimension
        s_input = torch.cat([Y_batch, s_batch], dim=-1)

        # Use IFNO inverse function (which actually takes s and produces u)
        forward_result = model.inverse(s_input)

        # Handle tuple return (IFNO returns tuple for symmetric models)
        if isinstance(forward_result, tuple):
            u_pred, _ = forward_result  # Extract prediction, ignore reconstruction loss
        else:
            u_pred = forward_result

        # Extract function values only (remove coordinates if present)
        if u_pred.shape[-1] > u_true.unsqueeze(0).shape[-1]:
            u_pred = u_pred[..., -u_true.unsqueeze(0).shape[-1]:]  # Take last channels (function values)

        u_pred = u_pred.squeeze(0)

        # === INVERSE PREDICTION: u -> s ===
        X_batch = X.unsqueeze(0)  # Add batch dimension
        u_batch = u_true.unsqueeze(0)  # Add batch dimension
        u_input = torch.cat([X_batch, u_batch], dim=-1)

        # Use IFNO forward function (which actually takes u and produces s)
        inverse_result = model(u_input)

        # Handle tuple return
        if isinstance(inverse_result, tuple):
            s_pred, _ = inverse_result  # Extract prediction, ignore reconstruction loss
        else:
            s_pred = inverse_result

        # Extract function values only (remove coordinates if present)
        if s_pred.shape[-1] > s.unsqueeze(0).shape[-1]:
            s_pred = s_pred[..., -s.unsqueeze(0).shape[-1]:]  # Take last channels (function values)

        s_pred = s_pred.squeeze(0)

    # Convert to numpy for plotting
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    u_pred_np = u_pred.squeeze(-1).cpu().numpy()
    s_true_np = s.squeeze(-1).cpu().numpy()
    s_pred_np = s_pred.squeeze(-1).cpu().numpy()
    X_np = X.squeeze(-1).cpu().numpy()

    # Extract x coordinates for 1D plotting
    if X_np.ndim == 1:
        x_coords = X_np  # Already 1D coordinates
    else:
        x_coords = X_np[:, 0]  # Use the first coordinate (x)

    # === INVERSE PLOT (s -> u) ===
    fig_inv, axes_inv = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Observed output function u(x)
    axes_inv[0].plot(x_coords, s_true_np, 'g-', label='Observed Output Function')
    axes_inv[0].set_title('Observed Output Function u(x)')
    axes_inv[0].set_xlabel('x')
    axes_inv[0].set_ylabel('u(x)')
    axes_inv[0].legend()
    axes_inv[0].grid(True)

    # Plot 2: Input function comparison (what we want to find)
    axes_inv[1].plot(x_coords, u_true_np, 'b-', label='True Input', alpha=0.7)
    axes_inv[1].plot(x_coords, u_pred_np, 'r--', label='Predicted Input', alpha=0.7)
    axes_inv[1].set_title('Input Function: True vs Predicted s(x)')
    axes_inv[1].set_xlabel('x')
    axes_inv[1].set_ylabel('s(x)')
    axes_inv[1].legend()
    axes_inv[1].grid(True)

    plt.tight_layout()

    # Save inverse plot
    save_path_inv = os.path.join(save_dir, f'ifno_inverse_sample_{sample_idx}.png')
    plt.savefig(save_path_inv, dpi=300, bbox_inches='tight')
    print(f"Saved inverse plot: {save_path_inv}")
    plt.close()

    # === FORWARD PLOT (u -> s) ===
    fig_fwd, axes_fwd = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Input function s(x)
    axes_fwd[0].plot(x_coords, u_true_np, 'b-', label='Input Function s(x)')
    axes_fwd[0].set_title('Input Function s(x)')
    axes_fwd[0].set_xlabel('x')
    axes_fwd[0].set_ylabel('s(x)')
    axes_fwd[0].legend()
    axes_fwd[0].grid(True)

    # Plot 2: Output function comparison (what we predict)
    axes_fwd[1].plot(x_coords, s_true_np, 'g-', label='True Output', alpha=0.7)
    axes_fwd[1].plot(x_coords, s_pred_np, 'orange', linestyle='--', label='Predicted Output', alpha=0.7)
    axes_fwd[1].set_title('Output Function: True vs Predicted u(x)')
    axes_fwd[1].set_xlabel('x')
    axes_fwd[1].set_ylabel('u(x)')
    axes_fwd[1].legend()
    axes_fwd[1].grid(True)

    plt.tight_layout()

    # Save forward plot
    save_path_fwd = os.path.join(save_dir, f'ifno_forward_sample_{sample_idx}.png')
    plt.savefig(save_path_fwd, dpi=300, bbox_inches='tight')
    print(f"Saved forward plot: {save_path_fwd}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize IFNO results on Burgers 1D dataset')
    parser.add_argument('--model_path', type=str,
                       default='./logs/burgers_1d_ifno_standalone/ifno_model.pth',
                       help='Path to trained IFNO model')
    parser.add_argument('--n_samples', type=int, default=3,
                       help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='results/ifno_plots_burger/',
                       help='Directory to save visualization plots')

    args = parser.parse_args()

    print("IFNO Burgers 1D Visualization")
    print("=" * 50)

    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("Please train the model first using: python train_ifno_standalone.py --dataset burgers_1d")
        exit(1)

    visualize_ifno_results(
        model_path=args.model_path,
        n_samples=args.n_samples,
        save_dir=args.save_dir
    )


