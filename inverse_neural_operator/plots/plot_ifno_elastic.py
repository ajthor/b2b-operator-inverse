#!/usr/bin/env python3
"""
Standalone script to visualize IFNO results on Elastic Plate dataset.
Produces two figures per sample (Inverse and Forward), each with two subplots.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from inverse_neural_operator.models.ifno import create_model
from inverse_neural_operator.data.elastic_plate import load_data


device = "cuda:1" if torch.cuda.is_available() else "cpu"


def _infer_ifno_config_from_state_dict_path(state_dict_path: str):
    """Infer IFNO hyperparameters from a saved state_dict path (best-effort)."""
    try:
        sd = torch.load(state_dict_path, map_location="cpu")
    except Exception:
        return None
    cfg = {}
    # width from p1 weight out_features (symmetric problems)
    if "p1.weight" in sd:
        cfg["width"] = int(sd["p1.weight"].shape[0])
    # n_layers and modes from conv keys
    conv_keys = [k for k in sd.keys() if k.startswith("convs.") and k.endswith(".weights")]
    if conv_keys:
        cfg["n_layers"] = len(conv_keys) // 2
        sample_conv = sd[conv_keys[0]]
        cfg["modes1"] = int(sample_conv.shape[-1])
        cfg["modes2"] = cfg["modes1"]
    # vae_latent_dim from fc_mu weight out_features
    if "vae_net.fc_mu.weight" in sd:
        cfg["vae_latent_dim"] = int(sd["vae_net.fc_mu.weight"].shape[0])
    return cfg if cfg else None


def create_circular_mask(x, y, center_x=0.5, center_y=0.5, radius=0.25):
    """Create a circular boolean mask for the hole in the plate."""
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2


def plot_displacement_field(coords, displacement, title, ax, colorbar_label='x-displacement',
                            add_colorbar=True, vmin=None, vmax=None, cax=None, scaling_order=None):
    """Plot displacement field with circular hole using grid interpolation and contourf."""
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Dense grid for smooth visualization
    xi = np.linspace(x_min, x_max, 200)
    yi = np.linspace(y_min, y_max, 200)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate values onto grid
    Zi = griddata((x_coords, y_coords), displacement, (Xi, Yi), method='cubic')

    # Optional scaling annotation
    if scaling_order is not None:
        scale = 10 ** -scaling_order
        Zi = Zi * scale if Zi is not None else Zi
        if vmin is not None:
            vmin *= scale
        if vmax is not None:
            vmax *= scale

    # Mask circular hole
    hole_mask = create_circular_mask(Xi, Yi)
    if Zi is not None:
        Zi[hole_mask] = np.nan

    im = ax.contourf(Xi, Yi, Zi, levels=100, cmap='jet', vmin=vmin, vmax=vmax)

    if vmin is not None and vmax is not None:
        im.set_clim(vmin, vmax)

    if cax is not None:
        cbar = plt.colorbar(im, cax=cax, format='%.1f')
        cbar.set_label(colorbar_label, rotation=270, labelpad=16, fontsize=12)
        cbar.ax.tick_params(labelsize=11)
        cbar.ax.yaxis.get_offset_text().set_size(11)
        if scaling_order is not None:
            cbar.ax.set_title('10^{%d}' % scaling_order, fontsize=11, pad=8)
    elif add_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.1f')
        cbar.set_label(colorbar_label, rotation=270, labelpad=16, fontsize=12)
        cbar.ax.tick_params(labelsize=11)
        cbar.ax.yaxis.get_offset_text().set_size(11)
        if scaling_order is not None:
            cbar.ax.set_title('10^{%d}' % scaling_order, fontsize=11, pad=8)

    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)


def plot_force_curve(force_coords, force_values, title, ax, label=None):
    """Plot the forcing function as a 1D curve: force magnitude vs boundary y-position."""
    force_y = force_coords[:, 1]
    ax.plot(force_values, force_y, linewidth=2, label=label)
    ax.set_ylim(force_y.min(), force_y.max())
    ax.set_xlabel('Force value')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major')
    ax.invert_xaxis()
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    if label is not None:
        ax.legend(loc='best')


def _slice_to_function_channels(pred_tensor, target_tensor):
    """Slice predicted tensor's last dim to match function channels of the target if coordinates are concatenated."""
    if pred_tensor.shape[-1] > target_tensor.shape[-1]:
        return pred_tensor[..., -target_tensor.shape[-1]:]
    return pred_tensor


def plot_sample(model, sample, sample_idx, save_dir):
    """Plot a single Elastic Plate sample with IFNO results for inverse and forward problems."""
    X, u_true, Y, s_true = sample

    with torch.no_grad():
        # Inverse: s -> u
        Y_batch = Y.unsqueeze(0)
        s_batch = s_true.unsqueeze(0)
        s_input = torch.cat([Y_batch, s_batch], dim=-1)
        inv_out = model.inverse(s_input)
        u_pred = inv_out[0] if isinstance(inv_out, tuple) else inv_out
        u_pred = _slice_to_function_channels(u_pred, u_true.unsqueeze(0)).squeeze(0)

        # Forward: u -> s
        X_batch = X.unsqueeze(0)
        u_batch = u_true.unsqueeze(0)
        u_input = torch.cat([X_batch, u_batch], dim=-1)
        fwd_out = model(u_input)
        s_pred = fwd_out[0] if isinstance(fwd_out, tuple) else fwd_out
        s_pred = _slice_to_function_channels(s_pred, s_true.unsqueeze(0)).squeeze(0)

    # Convert to numpy
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    u_pred_np = u_pred.squeeze(-1).cpu().numpy()
    s_true_np = s_true.squeeze(-1).cpu().numpy()
    s_pred_np = s_pred.squeeze(-1).cpu().numpy()

    # Inverse figure: Left displacement (observed), Right force comparison
    fig_inv, axes_inv = plt.subplots(1, 2, figsize=(14, 6))

    # Displacement scaling order for readable colorbar
    disp_vabs = float(np.max(np.abs(s_true_np))) if s_true_np.size > 0 else 0.0
    disp_order = int(np.floor(np.log10(disp_vabs))) if disp_vabs > 0 else 0

    plot_displacement_field(
        Y_np,
        s_true_np,
        'Observed Displacement Field',
        axes_inv[0],
        colorbar_label='Displacement',
        add_colorbar=True,
        scaling_order=disp_order,
    )

    axes_inv[1].plot(u_true_np, X_np[:, 1], 'b-', linewidth=3, label='True Force', alpha=0.85)
    axes_inv[1].plot(u_pred_np, X_np[:, 1], 'r--', linewidth=3, label='Predicted Force', alpha=0.85)
    axes_inv[1].set_ylim(X_np[:, 1].min(), X_np[:, 1].max())
    axes_inv[1].set_xlabel('Force Magnitude')
    axes_inv[1].set_ylabel('Position along Boundary (y)')
    axes_inv[1].set_title('Forcing Function: True vs Predicted')
    axes_inv[1].grid(True, alpha=0.3)
    axes_inv[1].legend(loc='best')
    axes_inv[1].axvline(x=0, color='k', linestyle=':', alpha=0.3)
    axes_inv[1].invert_xaxis()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_inv = os.path.join(save_dir, f'ifno_inverse_sample_{sample_idx}.png')
    plt.savefig(save_inv, dpi=300, bbox_inches='tight')
    plt.close(fig_inv)

    # Forward figure: Left input force, Right displacement field with GT and Pred overlay
    fig_fwd, axes_fwd = plt.subplots(1, 2, figsize=(14, 6))

    # Left: input force
    axes_fwd[0].plot(u_true_np, X_np[:, 1], 'b-', linewidth=3, label='Input Force')
    axes_fwd[0].set_ylim(X_np[:, 1].min(), X_np[:, 1].max())
    axes_fwd[0].set_xlabel('Force Magnitude')
    axes_fwd[0].set_ylabel('Position along Boundary (y)')
    axes_fwd[0].set_title('Input Force s(y)')
    axes_fwd[0].grid(True, alpha=0.3)
    axes_fwd[0].legend(loc='best')
    axes_fwd[0].axvline(x=0, color='k', linestyle=':', alpha=0.3)
    axes_fwd[0].invert_xaxis()

    # Right: displacement predicted vs ground truth overlay
    # Compute common vmin/vmax based on both fields for consistent color scale
    vmax = float(np.max(np.abs([s_true_np.max(), s_true_np.min(), s_pred_np.max(), s_pred_np.min()])))
    vmin = -vmax
    # Plot predicted as filled contours
    plot_displacement_field(
        Y_np,
        s_pred_np,
        'Predicted Displacement Field',
        axes_fwd[1],
        colorbar_label='Displacement',
        add_colorbar=True,
        vmin=vmin,
        vmax=vmax,
        scaling_order=disp_order,
    )
    # Overlay ground truth as contour lines for comparison
    x_coords = Y_np[:, 0]
    y_coords = Y_np[:, 1]
    xi = np.linspace(x_coords.min(), x_coords.max(), 200)
    yi = np.linspace(y_coords.min(), y_coords.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi_true = griddata((x_coords, y_coords), s_true_np, (Xi, Yi), method='cubic')
    if disp_order is not None and Zi_true is not None:
        Zi_true = Zi_true * (10 ** -disp_order)
    # Mask circular hole on overlay
    hole_mask = create_circular_mask(Xi, Yi)
    if Zi_true is not None:
        Zi_true[hole_mask] = np.nan
    cs = axes_fwd[1].contour(Xi, Yi, Zi_true, levels=12, colors='k', linewidths=0.7, alpha=0.7)
    axes_fwd[1].clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    axes_fwd[1].set_title('Predicted (filled) with Ground Truth (lines)')

    plt.tight_layout()
    save_fwd = os.path.join(save_dir, f'ifno_forward_sample_{sample_idx}.png')
    plt.savefig(save_fwd, dpi=300, bbox_inches='tight')
    plt.close(fig_fwd)


def visualize_ifno_results(model_path, n_samples=3, save_dir="results/ifno_plots_elastic/"):
    print("Loading Elastic Plate test dataset...")
    test_dataset = load_data(None, device=device, split="test")
    dataset_info = test_dataset.get_info()
    print(f"Dataset info: {dataset_info}")

    print("Creating IFNO model...")
    inferred_cfg = _infer_ifno_config_from_state_dict_path(model_path)
    if inferred_cfg:
        print(f"Inferred model config from checkpoint: {inferred_cfg}")
    model = create_model(
        input_size=None,
        hidden_sizes=[256, 256, 256],
        n_coupling_layers=2,
        modes1=(inferred_cfg.get("modes1") if inferred_cfg else 16),
        modes2=(inferred_cfg.get("modes2") if inferred_cfg else 16),
        width=(inferred_cfg.get("width") if inferred_cfg else 64),
        beta=2.0,
        n_layers=(inferred_cfg.get("n_layers") if inferred_cfg else 3),
        padding=20,
        vae_latent_dim=(inferred_cfg.get("vae_latent_dim") if inferred_cfg else 24),
        intermediate_dim=32,
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

    os.makedirs(save_dir, exist_ok=True)

    # Select random samples
    test_indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))
    print(f"Generating {len(test_indices)} visualization plots...")
    for idx in test_indices:
        sample = test_dataset[idx]
        plot_sample(model, sample, idx, save_dir)

    print(f"Visualization completed! Plots saved to: {save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize IFNO results on Elastic Plate dataset')
    parser.add_argument('--model_path', type=str,
                        default='logs_ifno/elastic_plate/ifno_model.pth',
                        help='Path to trained IFNO model')
    parser.add_argument('--n_samples', type=int, default=3,
                        help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='results/ifno_plots_elastic/',
                        help='Directory to save visualization plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("Please train the model first using: python train_ifno_standalone.py --dataset elastic_plate")
        raise SystemExit(1)

    visualize_ifno_results(
        model_path=args.model_path,
        n_samples=args.n_samples,
        save_dir=args.save_dir,
    )




