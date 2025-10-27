#!/usr/bin/env python3
"""
Standalone visualization for IFNO on the Wave Scattering dataset.
Generates polar far-field comparisons and spatial field reconstructions.
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from inverse_neural_operator.data.wave_scattering import load_data
from inverse_neural_operator.models.ifno import create_model

# Global device handle (can be overridden via CLI)
device = "cuda:0" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Reproducibility defaults
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def _infer_ifno_config_from_state_dict_path(state_dict_path: str):
    """Infer IFNO hyperparameters from a saved checkpoint."""
    try:
        state_dict = torch.load(state_dict_path, map_location="cpu")
    except Exception:
        return None

    cfg = {}

    # Width and intermediate dimension (asymmetric problems use linear layers p1/p2)
    if "p1.weight" in state_dict:
        cfg["width"] = int(state_dict["p1.weight"].shape[0])
        cfg["intermediate_dim"] = int(state_dict["p1.weight"].shape[1])

    # Number of layers and modes from Fourier blocks
    conv_keys = [k for k in state_dict.keys() if k.startswith("convs.") and k.endswith(".weights")]
    if conv_keys:
        cfg["n_layers"] = len(conv_keys) // 2
        sample_conv = state_dict[conv_keys[0]]
        cfg["modes1"] = int(sample_conv.shape[-1])
        cfg["modes2"] = cfg["modes1"]

    # Latent dimension from VAE
    if "vae_net.fc_mu.weight" in state_dict:
        cfg["vae_latent_dim"] = int(state_dict["vae_net.fc_mu.weight"].shape[0])

    return cfg if cfg else None


def _prepare_model(model_path: str, dataset_info: dict):
    inferred_cfg = _infer_ifno_config_from_state_dict_path(model_path)

    default_cfg = {
        "modes1": 16,
        "modes2": 16,
        "width": 64,
        "n_layers": 3,
        "vae_latent_dim": 24,
        "intermediate_dim": 32,
    }

    if inferred_cfg:
        default_cfg.update({k: v for k, v in inferred_cfg.items() if v is not None})

    model = create_model(
        input_size=None,
        hidden_sizes=[256, 256, 256],
        n_coupling_layers=2,
        modes1=default_cfg["modes1"],
        modes2=default_cfg["modes2"],
        width=default_cfg["width"],
        beta=2.0,
        n_layers=default_cfg["n_layers"],
        padding=20,
        vae_latent_dim=default_cfg["vae_latent_dim"],
        intermediate_dim=default_cfg["intermediate_dim"],
        input_spatial_dims=dataset_info["input_spatial_dims"],
        output_spatial_dims=dataset_info["output_spatial_dims"],
        input_function_channels=dataset_info["input_function_channels"],
        output_function_channels=dataset_info["output_function_channels"],
        coordinate_dim=dataset_info["coordinate_dim"],
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _format_theta(X: torch.Tensor):
    theta = X.squeeze(-1).detach().cpu().numpy()
    return theta * 2.0 * np.pi


def _slice_channels(tensor: torch.Tensor, target_channels: int):
    if tensor.shape[-1] > target_channels:
        return tensor[..., -target_channels:]
    return tensor


def plot_sample(model, sample, sample_idx: int, save_dir: str):
    X, u_true, Y, s_true = sample

    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_true = s_true.to(device)

    with torch.no_grad():
        # Forward prediction: u -> s
        u_input = torch.cat([X.unsqueeze(0), u_true.unsqueeze(0)], dim=-1)
        s_pred = model(u_input)
        s_pred = _slice_channels(s_pred, s_true.shape[-1]).squeeze(0)

        # Inverse prediction: s -> u
        s_input = torch.cat([Y.unsqueeze(0), s_true.unsqueeze(0)], dim=-1)
        u_pred = model.inverse(s_input)
        if isinstance(u_pred, tuple):
            u_pred = u_pred[0]
        u_pred = _slice_channels(u_pred, u_true.shape[-1]).squeeze(0)

    # Convert tensors for plotting
    theta = _format_theta(X)
    u_true_np = u_true.detach().cpu().numpy()
    u_pred_np = u_pred.detach().cpu().numpy()
    s_true_np = s_true.detach().cpu().numpy().squeeze(-1)
    s_pred_np = s_pred.detach().cpu().numpy().squeeze(-1)

    u_true_mag = np.linalg.norm(u_true_np, axis=-1)
    u_pred_mag = np.linalg.norm(u_pred_np, axis=-1)
    u_mag_err = np.abs(u_pred_mag - u_true_mag)

    grid_size = int(np.sqrt(s_true_np.shape[0]))
    s_true_2d = s_true_np.reshape(grid_size, grid_size)
    s_pred_2d = s_pred_np.reshape(grid_size, grid_size)
    s_err_2d = s_pred_2d - s_true_2d

    u_mse = float(np.mean((u_pred_np - u_true_np) ** 2))
    s_mse = float(np.mean((s_pred_np - s_true_np) ** 2))
    u_mae = float(np.mean(np.abs(u_pred_np - u_true_np)))
    s_mae = float(np.mean(np.abs(s_pred_np - s_true_np)))

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ifno_wave_scattering_sample_{sample_idx}.png")

    fig = plt.figure(figsize=(18, 10))

    ax_true_polar = plt.subplot(2, 3, 1, projection="polar")
    ax_true_polar.plot(theta, u_true_mag, color="tab:blue", linewidth=2)
    ax_true_polar.set_title("True Far Field |u(θ)|")

    ax_pred_polar = plt.subplot(2, 3, 2, projection="polar")
    ax_pred_polar.plot(theta, u_pred_mag, color="tab:orange", linewidth=2)
    ax_pred_polar.set_title("Predicted Far Field |û(θ)|")

    ax_error_line = plt.subplot(2, 3, 3)
    ax_error_line.plot(theta, u_mag_err, color="tab:red", linewidth=1.5)
    ax_error_line.set_xlabel("θ (radians)")
    ax_error_line.set_ylabel("|û(θ)| - |u(θ)|")
    ax_error_line.set_title("Far Field Magnitude Error")
    ax_error_line.grid(True, alpha=0.3)

    extent = [0.0, 1.0, 0.0, 1.0]

    ax_true_field = plt.subplot(2, 3, 4)
    im_true = ax_true_field.imshow(s_true_2d, cmap="viridis", origin="lower", extent=extent)
    ax_true_field.set_title("Observed Density Field s(x, y)")
    ax_true_field.set_xlabel("x")
    ax_true_field.set_ylabel("y")
    plt.colorbar(im_true, ax=ax_true_field, fraction=0.046, pad=0.04)

    ax_pred_field = plt.subplot(2, 3, 5)
    im_pred = ax_pred_field.imshow(s_pred_2d, cmap="viridis", origin="lower", extent=extent)
    ax_pred_field.set_title("Predicted Density Field ŝ(x, y)")
    ax_pred_field.set_xlabel("x")
    ax_pred_field.set_ylabel("y")
    plt.colorbar(im_pred, ax=ax_pred_field, fraction=0.046, pad=0.04)

    ax_err_field = plt.subplot(2, 3, 6)
    vmax = np.max(np.abs(s_err_2d))
    im_err = ax_err_field.imshow(
        s_err_2d, cmap="seismic", origin="lower", extent=extent, vmin=-vmax, vmax=vmax
    )
    ax_err_field.set_title("Density Field Error ŝ - s")
    ax_err_field.set_xlabel("x")
    ax_err_field.set_ylabel("y")
    plt.colorbar(im_err, ax=ax_err_field, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Sample {sample_idx} • Far-field MSE {u_mse:.4e} (MAE {u_mae:.4e}) • "
        f"Density MSE {s_mse:.4e} (MAE {s_mae:.4e})",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization: {save_path}")


def visualize_ifno_results(model_path: str, n_samples: int, save_dir: str, split: str = "test"):
    print("Loading Wave Scattering dataset...")
    dataset = load_data(None, device=device, split=split)
    dataset_info = dataset.get_info()
    print(f"Dataset info: {dataset_info}")

    print("Preparing IFNO model...")
    model = _prepare_model(model_path, dataset_info)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {total_params} parameters")

    os.makedirs(save_dir, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    print(f"Generating {len(indices)} sample visualizations...")
    for sample_idx in indices:
        plot_sample(model, dataset[sample_idx], sample_idx, save_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize IFNO predictions on the Wave Scattering dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="logs_ifno_back/wave_scattering/ifno_model.pth",
        help="Path to trained IFNO model checkpoint",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="Number of random test samples to visualize",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/ifno_plots_wave_scattering/",
        help="Directory to store generated plots",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to visualize",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device (e.g., 'cuda:0', 'cpu')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global device
    if args.device is not None:
        device = args.device

    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("Please train the model first using train_ifno_standalone.py")
        return

    print("IFNO Wave Scattering Visualization")
    print("=" * 60)
    print(f"Using device: {device}")

    visualize_ifno_results(
        model_path=args.model_path,
        n_samples=args.n_samples,
        save_dir=args.save_dir,
        split=args.split,
    )


if __name__ == "__main__":
    main()

