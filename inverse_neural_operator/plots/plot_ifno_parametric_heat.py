#!/usr/bin/env python3
"""
Standalone script to visualize IFNO results on Parametric Heat 2D dataset.
This doesn't interfere with the existing plotting infrastructure.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from inverse_neural_operator.models.ifno import create_model
from inverse_neural_operator.data.parametric_heat import load_data

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
    # Robustly detect any convs.<idx>. parameter groups
    conv_indices = []
    for k in sd.keys():
        if k.startswith("convs."):
            try:
                idx_str = k.split(".")[1]
                conv_indices.append(int(idx_str))
            except Exception:
                continue
    if conv_indices:
        num_convs = max(conv_indices) + 1
        cfg["n_layers"] = num_convs // 2
    # Try to infer modes if 1D simple Fourier layers exist
    conv_weight_keys = [k for k in sd.keys() if k.startswith("convs.") and k.endswith(".weights")]
    if conv_weight_keys:
        sample_conv = sd[conv_weight_keys[0]]
        cfg["modes1"] = int(sample_conv.shape[-1])
        cfg["modes2"] = cfg["modes1"]
    # vae_latent_dim from fc_mu weight out_features
    if "vae_net.fc_mu.weight" in sd:
        cfg["vae_latent_dim"] = int(sd["vae_net.fc_mu.weight"].shape[0])
    return cfg if cfg else None


def visualize_ifno_results(model_path, n_samples=3, save_dir="results/ifno_plots_parametric_heat/"):
    """Visualize IFNO results on Parametric Heat 2D dataset."""

    print("Loading Parametric Heat 2D test dataset...")
    test_dataset = load_data(None, device=device, split="test")
    dataset_info = test_dataset.get_info()

    print(f"Dataset info: {dataset_info}")

    print("Creating IFNO model...")
    inferred_cfg = _infer_ifno_config_from_state_dict_path(model_path)
    if inferred_cfg:
        print(f"Inferred model config from checkpoint: {inferred_cfg}")
    # Safe defaults if inference failed for some fields
    modes1 = (inferred_cfg.get("modes1") if inferred_cfg and "modes1" in inferred_cfg else 8)
    modes2 = (inferred_cfg.get("modes2") if inferred_cfg and "modes2" in inferred_cfg else 8)
    width = (inferred_cfg.get("width") if inferred_cfg and "width" in inferred_cfg else 24)
    n_layers = (inferred_cfg.get("n_layers") if inferred_cfg and "n_layers" in inferred_cfg else 3)
    vae_latent_dim = (inferred_cfg.get("vae_latent_dim") if inferred_cfg and "vae_latent_dim" in inferred_cfg else 8)

    model = create_model(
        input_size=None,  # Not used by IFNO
        hidden_sizes=[256, 256, 256],  # Not used by IFNO
        n_coupling_layers=2,
        modes1=modes1,
        modes2=modes2,
        width=width,
        beta=2.0,
        n_layers=n_layers,
        padding=20,
        vae_latent_dim=vae_latent_dim,
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
        plot_sample(model, sample, idx, save_dir, dataset_info)

    print(f"Visualization completed! Plots saved to: {save_dir}")


def _reshape_field_2d(field_tensor, spatial_dims):
    """Reshape a flattened (H*W, 1) tensor to (H, W)."""
    H, W = spatial_dims
    return field_tensor.squeeze(-1).view(H, W)


def plot_sample(model, sample, sample_idx, save_dir, dataset_info):
    """Plot a single 2D sample with IFNO results for both forward and inverse problems."""

    H, W = dataset_info["input_spatial_dims"]
    X, u_true, Y, s = sample

    with torch.no_grad():
        # === FORWARD PREDICTION: s -> u ===
        Y_batch = Y.unsqueeze(0)  # (1, H*W, 2)
        s_batch = s.unsqueeze(0)  # (1, H*W, 1)
        s_input = torch.cat([Y_batch, s_batch], dim=-1)

        forward_result = model.inverse(s_input)

        if isinstance(forward_result, tuple):
            u_pred, _ = forward_result
        else:
            u_pred = forward_result

        # Extract function values only (remove coordinates if present)
        if u_pred.shape[-1] > u_true.unsqueeze(0).shape[-1]:
            u_pred = u_pred[..., -u_true.unsqueeze(0).shape[-1]:]

        u_pred = u_pred.squeeze(0)

        # === INVERSE PREDICTION: u -> s ===
        X_batch = X.unsqueeze(0)  # (1, H*W, 2)
        u_batch = u_true.unsqueeze(0)  # (1, H*W, 1)
        u_input = torch.cat([X_batch, u_batch], dim=-1)

        inverse_result = model(u_input)

        if isinstance(inverse_result, tuple):
            s_pred, _ = inverse_result
        else:
            s_pred = inverse_result

        # Extract function values only (remove coordinates if present)
        if s_pred.shape[-1] > s.unsqueeze(0).shape[-1]:
            s_pred = s_pred[..., -s.unsqueeze(0).shape[-1]:]

        s_pred = s_pred.squeeze(0)

    # Reshape to 2D grids
    u_true_2d = _reshape_field_2d(u_true, (H, W)).cpu().numpy()
    u_pred_2d = _reshape_field_2d(u_pred, (H, W)).cpu().numpy()
    s_true_2d = _reshape_field_2d(s, (H, W)).cpu().numpy()
    s_pred_2d = _reshape_field_2d(s_pred, (H, W)).cpu().numpy()

    # === INVERSE PLOT (s -> u) ===
    fig_inv, axes_inv = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axes_inv[0].imshow(s_true_2d, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes_inv[0].set_title('Observed Output s(x, y)')
    axes_inv[0].set_xlabel('x')
    axes_inv[0].set_ylabel('y')
    fig_inv.colorbar(im0, ax=axes_inv[0], fraction=0.046, pad=0.04)

    im1 = axes_inv[1].imshow(u_true_2d, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes_inv[1].set_title('True Input u(x, y)')
    axes_inv[1].set_xlabel('x')
    axes_inv[1].set_ylabel('y')
    fig_inv.colorbar(im1, ax=axes_inv[1], fraction=0.046, pad=0.04)

    im2 = axes_inv[2].imshow(u_pred_2d, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes_inv[2].set_title('Predicted Input u(x, y)')
    axes_inv[2].set_xlabel('x')
    axes_inv[2].set_ylabel('y')
    fig_inv.colorbar(im2, ax=axes_inv[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path_inv = os.path.join(save_dir, f'ifno_inverse_sample_{sample_idx}.png')
    plt.savefig(save_path_inv, dpi=300, bbox_inches='tight')
    print(f"Saved inverse plot: {save_path_inv}")
    plt.close(fig_inv)

    # === FORWARD PLOT (u -> s) ===
    fig_fwd, axes_fwd = plt.subplots(1, 3, figsize=(15, 4.5))
    im3 = axes_fwd[0].imshow(u_true_2d, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes_fwd[0].set_title('Input u(x, y)')
    axes_fwd[0].set_xlabel('x')
    axes_fwd[0].set_ylabel('y')
    fig_fwd.colorbar(im3, ax=axes_fwd[0], fraction=0.046, pad=0.04)

    im4 = axes_fwd[1].imshow(s_true_2d, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes_fwd[1].set_title('True Output s(x, y)')
    axes_fwd[1].set_xlabel('x')
    axes_fwd[1].set_ylabel('y')
    fig_fwd.colorbar(im4, ax=axes_fwd[1], fraction=0.046, pad=0.04)

    im5 = axes_fwd[2].imshow(s_pred_2d, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes_fwd[2].set_title('Predicted Output s(x, y)')
    axes_fwd[2].set_xlabel('x')
    axes_fwd[2].set_ylabel('y')
    fig_fwd.colorbar(im5, ax=axes_fwd[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path_fwd = os.path.join(save_dir, f'ifno_forward_sample_{sample_idx}.png')
    plt.savefig(save_path_fwd, dpi=300, bbox_inches='tight')
    print(f"Saved forward plot: {save_path_fwd}")
    plt.close(fig_fwd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize IFNO results on Parametric Heat 2D dataset')
    parser.add_argument('--model_path', type=str,
                       default='./logs/parametric_heat_ifno_standalone/ifno_model.pth',
                       help='Path to trained IFNO model')
    parser.add_argument('--n_samples', type=int, default=3,
                       help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='results/ifno_plots_parametric_heat/',
                       help='Directory to save visualization plots')

    args = parser.parse_args()

    print("IFNO Parametric Heat 2D Visualization")
    print("=" * 50)

    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("Please train the model first using: python train_ifno_standalone.py --dataset parametric_heat")
        exit(1)

    visualize_ifno_results(
        model_path=args.model_path,
        n_samples=args.n_samples,
        save_dir=args.save_dir
    )


