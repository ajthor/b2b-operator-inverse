#!/usr/bin/env python3
"""
Standalone script to visualize IFNO results on Chladni 2D dataset, aligned with train_ifno_standalone.py.
Generates forward (u->s) and backward (s->u) plots for a few random samples.
"""

import os
import sys

# Ensure project root is on sys.path when running as a script
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import matplotlib.pyplot as plt
import numpy as np
import random

from inverse_neural_operator.models.ifno import create_model
from inverse_neural_operator.data.chladni_2d import load_data


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = "cuda:5" if torch.cuda.is_available() else "cpu"


def _infer_ifno_config_from_state_dict_path(state_dict_path: str):
    """Infer minimal IFNO hyperparameters from a saved state_dict path if possible."""
    try:
        sd = torch.load(state_dict_path, map_location="cpu")
    except Exception:
        return None
    cfg = {}
    if "p1.weight" in sd:
        cfg["width"] = int(sd["p1.weight"].shape[0])
    # Infer n_layers by scanning for highest convs.<idx>. keys
    conv_indices = []
    for k in sd.keys():
        if k.startswith("convs."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                conv_indices.append(int(parts[1]))
    if conv_indices:
        max_idx = max(conv_indices)
        cfg["n_layers"] = max(1, (max_idx + 1) // 2)
    if "vae_net.fc_mu.weight" in sd:
        cfg["vae_latent_dim"] = int(sd["vae_net.fc_mu.weight"].shape[0])
    return cfg if cfg else None


def visualize_ifno_results(
    model_path, n_samples=3, save_dir="results/ifno_plots_chladni_2d/"
):
    """Visualize IFNO results on Chladni 2D dataset with forward and backward plots."""

    print("Loading Chladni 2D test dataset...")
    test_dataset = load_data(None, device=device, split="test")
    dataset_info = test_dataset.get_info()

    print(f"Dataset info: {dataset_info}")

    print("Creating IFNO model...")
    inferred_cfg = _infer_ifno_config_from_state_dict_path(model_path)
    if inferred_cfg:
        print(f"Inferred model config from checkpoint: {inferred_cfg}")

    # Resolve final hyperparameters with safe defaults matching training script
    resolved_modes1 = inferred_cfg.get("modes1", 8) if inferred_cfg else 8
    resolved_modes2 = inferred_cfg.get("modes2", 8) if inferred_cfg else 8
    resolved_width = inferred_cfg.get("width", 8) if inferred_cfg else 8
    resolved_n_layers = inferred_cfg.get("n_layers", 2) if inferred_cfg else 2
    resolved_vae_latent = inferred_cfg.get("vae_latent_dim", 8) if inferred_cfg else 8

    print(
        f"Using config → modes1={resolved_modes1}, modes2={resolved_modes2}, width={resolved_width}, "
        f"n_layers={resolved_n_layers}, vae_latent_dim={resolved_vae_latent}"
    )

    model = create_model(
        input_size=None,
        hidden_sizes=[256, 256, 256],
        n_coupling_layers=2,
        modes1=resolved_modes1,
        modes2=resolved_modes2,
        width=resolved_width,
        beta=2.0,
        n_layers=resolved_n_layers,
        padding=20,
        vae_latent_dim=resolved_vae_latent,
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

    # Choose random test indices
    test_indices = random.sample(
        range(len(test_dataset)), min(n_samples, len(test_dataset))
    )
    print(f"Generating {len(test_indices)} visualization pairs (forward/backward)...")

    for idx in test_indices:
        sample = test_dataset[idx]
        plot_sample(model, sample, idx, save_dir, dataset_info)

    print(f"Visualization completed! Plots saved to: {save_dir}")


def _to_2d(field_1c_flat: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Convert flattened single-channel field tensor to 2D numpy array."""
    return field_1c_flat.squeeze(-1).detach().cpu().numpy().reshape(h, w)


def plot_sample(model, sample, sample_idx, save_dir, dataset_info):
    """Plot a single Chladni 2D sample with IFNO results for both forward and inverse problems."""

    X, u_true, Y, s_true = sample
    h_in, w_in = dataset_info["input_spatial_dims"]
    h_out, w_out = dataset_info["output_spatial_dims"]

    with torch.no_grad():
        # Backward: given s, predict u (model.inverse)
        Y_batch = Y.unsqueeze(0)
        s_batch = s_true.unsqueeze(0)
        s_input = torch.cat([Y_batch, s_batch], dim=-1)
        backward_result = model.inverse(s_input)
        if isinstance(backward_result, tuple):
            u_pred, _ = backward_result
        else:
            u_pred = backward_result
        if u_pred.shape[-1] > u_true.unsqueeze(0).shape[-1]:
            u_pred = u_pred[..., -u_true.unsqueeze(0).shape[-1] :]
        u_pred = u_pred.squeeze(0)

        # Forward: given u, predict s (model.forward)
        X_batch = X.unsqueeze(0)
        u_batch = u_true.unsqueeze(0)
        u_input = torch.cat([X_batch, u_batch], dim=-1)
        forward_result = model(u_input)
        if isinstance(forward_result, tuple):
            s_pred, _ = forward_result
        else:
            s_pred = forward_result
        if s_pred.shape[-1] > s_true.unsqueeze(0).shape[-1]:
            s_pred = s_pred[..., -s_true.unsqueeze(0).shape[-1] :]
        s_pred = s_pred.squeeze(0)

    # Convert to 2D arrays
    u_true_2d = _to_2d(u_true, h_in, w_in)
    u_pred_2d = _to_2d(u_pred, h_in, w_in)
    s_true_2d = _to_2d(s_true, h_out, w_out)
    s_pred_2d = _to_2d(s_pred, h_out, w_out)

    # === BACKWARD PLOT (s -> u) ===
    fig_bwd, axes_bwd = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes_bwd[0].contourf(u_true_2d, levels=20, cmap="Spectral_r")
    axes_bwd[0].set_title("True Input Force u(x,y)")
    axes_bwd[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes_bwd[0])
    im1 = axes_bwd[1].contourf(u_pred_2d, levels=20, cmap="RdBu_r")
    axes_bwd[1].set_title("Predicted Input Force û(x,y)")
    axes_bwd[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes_bwd[1])
    plt.tight_layout()
    save_path_bwd = os.path.join(save_dir, f"ifno_inverse_sample_{sample_idx}.png")
    plt.savefig(save_path_bwd, dpi=300, bbox_inches="tight")
    print(f"Saved inverse plot: {save_path_bwd}")
    plt.close(fig_bwd)

    # === FORWARD PLOT (u -> s) ===
    fig_fwd, axes_fwd = plt.subplots(1, 2, figsize=(12, 5))
    im2 = axes_fwd[0].contourf(s_true_2d, levels=20, cmap="Spectral_r")
    axes_fwd[0].set_title("True Output Displacement s(x,y)")
    axes_fwd[0].set_aspect("equal")
    plt.colorbar(im2, ax=axes_fwd[0])
    im3 = axes_fwd[1].contourf(s_pred_2d, levels=20, cmap="RdBu_r")
    axes_fwd[1].set_title("Predicted Output Displacement ŝ(x,y)")
    axes_fwd[1].set_aspect("equal")
    plt.colorbar(im3, ax=axes_fwd[1])
    plt.tight_layout()
    save_path_fwd = os.path.join(save_dir, f"ifno_forward_sample_{sample_idx}.png")
    plt.savefig(save_path_fwd, dpi=300, bbox_inches="tight")
    print(f"Saved forward plot: {save_path_fwd}")
    plt.close(fig_fwd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize IFNO results on Chladni 2D dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./logs_ifno/chladni_2d/seed_0/ifno_model.pth",
        help="Path to trained IFNO model",
    )
    parser.add_argument(
        "--n_samples", type=int, default=3, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/chladni_2d/ifno/",
        help="Directory to save visualization plots",
    )

    args = parser.parse_args()

    print("IFNO Chladni 2D Visualization")
    print("=" * 50)

    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print(
            "Please train the model first using: python train_ifno_standalone.py --dataset chladni_2d"
        )
        raise SystemExit(1)

    visualize_ifno_results(
        model_path=args.model_path, n_samples=args.n_samples, save_dir=args.save_dir
    )
