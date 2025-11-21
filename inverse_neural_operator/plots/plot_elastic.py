#!/usr/bin/env python
"""Plot Elastic Plate displacement fields and inferred boundary forces."""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

sys.path.insert(0, "inverse_neural_operator")

from inverse_neural_operator.data.load_dataset import load_dataset
from inverse_neural_operator.models.load_model import load_models
from inverse_neural_operator.b2b.load_model import load_forward_model

DEVICE = "cpu"
DEFAULT_SAMPLE_COUNT = 10
SAMPLING_MODELS = {
    "variational_autoencoder",
    "mixture_density_network",
    "inn_affine",
    "cinn_affine",
}
ALL_MODELS = [
    "linear", "linear_inverse", "nonlinear", "variational_autoencoder",
    "inn_additive", "cinn_additive", "inn_affine", "cinn_affine",
    "cinn_additive_probabilistic", "cinn_affine_probabilistic",
    "mixture_density_network"
]

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seeds(42)


def create_circular_mask(x, y, center=(0.5, 0.5), radius=0.25):
    """Return True inside the plate's circular void."""
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2


def add_colorbar(im, ax, label, scaling_order=None, cax=None):
    """Add formatted colorbar to plot."""
    kwargs = {"format": "%.1f", "ax": ax, "fraction": 0.046, "pad": 0.04} if cax is None else {"cax": cax, "format": "%.1f"}
    cbar = plt.colorbar(im, **kwargs)
    cbar.set_label(label, rotation=270, labelpad=20, fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.get_offset_text().set_size(16)
    if scaling_order is not None:
        cbar.ax.set_title(f"10^{int(scaling_order)}", fontsize=16, pad=10)
    return cbar


def plot_displacement_field(coords, displacement, title, ax, colorbar_label="x-displacement",
                           add_cbar=True, vmin=None, vmax=None, cax=None, scaling_order=None):
    """Smoothly interpolate and plot a displacement field with the central hole masked."""
    # Create interpolation grid
    xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 200)
    yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((coords[:, 0], coords[:, 1]), displacement, (Xi, Yi), method="cubic")

    # Apply scaling
    if scaling_order is not None:
        scale = 10 ** -scaling_order
        Zi, vmin, vmax = Zi * scale, vmin * scale if vmin else None, vmax * scale if vmax else None

    # Mask circular void and plot
    Zi[create_circular_mask(Xi, Yi)] = np.nan
    im = ax.contourf(Xi, Yi, Zi, levels=100, cmap="jet", vmin=vmin, vmax=vmax)
    if add_cbar:
        add_colorbar(im, ax, colorbar_label, scaling_order, cax)

    # Format axes
    ax.set_aspect("equal")
    ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.set_title(title, fontsize=18, pad=15)
    ax.tick_params(axis="both", which="major", labelsize=14)


def sample_alpha(model_name, model, beta, device, dtype, num_samples):
    """Sample alpha coefficients from model-specific inverse mappings."""
    beta_rep = beta.expand(num_samples, -1)

    if model_name == "variational_autoencoder":
        z = model.sample_prior(num_samples, device=device)
        return model.inverse(beta_rep, z)

    if model_name == "mixture_density_network":
        return torch.stack([model.inverse(beta).squeeze(0) for _ in range(num_samples)])

    if model_name == "inn_affine":
        z_size = model.coupling_layers[0].input_size - model.output_size
        z = torch.randn(num_samples, z_size, device=device, dtype=dtype)
        return model.inverse(beta=beta_rep, z=z)

    if model_name == "cinn_affine":
        alpha_dim = model.coupling_layers[0].input_size
        z = torch.randn(num_samples, alpha_dim, device=device, dtype=dtype)
        return model.inverse(z, beta_rep)

    return None


def predict_forces(model_name, model, evaluate_fn, input_function_encoder,
                   output_function_encoder, batch, num_samples=DEFAULT_SAMPLE_COUNT):
    """Predict boundary forces from observed displacement field."""
    X, _, Y, s_observed = batch

    # Deterministic models - single prediction
    if model_name not in SAMPLING_MODELS:
        u_pred, alpha_pred = evaluate_fn(model, batch, input_function_encoder, output_function_encoder)
        return u_pred.squeeze(0), alpha_pred, None

    # Sampling models - generate multiple predictions
    beta, _ = output_function_encoder.compute_coefficients(Y, s_observed)
    alpha_samples = sample_alpha(model_name, model, beta, X.device, beta.dtype, num_samples)

    if alpha_samples is None:
        u_pred, alpha_pred = evaluate_fn(model, batch, input_function_encoder, output_function_encoder)
        return u_pred.squeeze(0), alpha_pred, None

    # Reconstruct forces from sampled alpha coefficients
    X_rep = X.repeat(alpha_samples.size(0), 1, 1)
    u_samples = input_function_encoder(X_rep, alpha_samples)
    return u_samples.mean(dim=0), alpha_samples.mean(dim=0, keepdim=True), u_samples.squeeze(-1).cpu().numpy()


def plot_force_comparison(ax, u_true_np, u_pred_np, force_y, sampled_predictions_np=None):
    """Plot true vs predicted boundary forces."""
    ax.plot(u_true_np, force_y, "b-", linewidth=3, label="True Force", alpha=0.8)

    if sampled_predictions_np is not None:
        for i, sample_vals in enumerate(sampled_predictions_np):
            label = "Samples" if i == 0 else None
            ax.plot(sample_vals, force_y, color="r", linewidth=1, alpha=0.25, label=label)
        ax.plot(u_pred_np, force_y, "r--", linewidth=3, label="Mean", alpha=0.9)
    else:
        ax.plot(u_pred_np, force_y, "r--", linewidth=3, label="Predicted Force", alpha=0.8)

    ax.set_ylim(force_y.min(), force_y.max())
    ax.set_xlabel("Force Magnitude", fontsize=16)
    ax.set_ylabel("Position along Boundary (y)", fontsize=16)
    ax.set_title("Forcing Function Comparison", fontsize=18, pad=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=14, loc="best")
    ax.axvline(x=0, color="k", linestyle=":", alpha=0.3)


def compute_resimulation(forward_model, alpha_pred, output_function_encoder, Y_batch):
    """Compute re-simulated displacement field using forward model."""
    if forward_model is None or alpha_pred is None:
        return None
    beta_resim = forward_model(alpha_pred)
    resim_disp = output_function_encoder(Y_batch, beta_resim)
    return resim_disp.squeeze(0).squeeze(-1).cpu().numpy()


def plot_elastic_sample(model, evaluate_fn, input_function_encoder, output_function_encoder,
                       sample, model_name, forward_model=None, save_path=None):
    """Plot a single test sample with force prediction and displacement fields."""
    model.eval()
    batch = tuple(t.to(DEVICE).unsqueeze(0) for t in sample)
    X, u_true, Y, s_observed = (t.squeeze(0) for t in batch)

    # Predict forces and compute re-simulation
    with torch.no_grad():
        u_pred, alpha_pred, sampled_predictions_np = predict_forces(
            model_name, model, evaluate_fn, input_function_encoder, output_function_encoder, batch)
        resim_disp_np = compute_resimulation(forward_model, alpha_pred, output_function_encoder, batch[2])

    # Convert to numpy
    u_true_np, u_pred_np = u_true.squeeze(-1).cpu().numpy(), u_pred.squeeze(-1).cpu().numpy()
    s_observed_np, Y_np = s_observed.squeeze(-1).cpu().numpy(), Y.cpu().numpy()
    force_y = X.cpu().numpy()[:, 1]

    # Compute displacement scaling
    disp_vabs = max(np.max(np.abs(s_observed_np)),
                    np.max(np.abs(resim_disp_np)) if resim_disp_np is not None else 0)
    disp_order = int(np.floor(np.log10(disp_vabs))) if disp_vabs > 0 else 0
    vmin, vmax = (-disp_vabs, disp_vabs) if disp_vabs > 0 else (None, None)

    # Create figure
    fig = plt.figure(figsize=(22, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.1], wspace=0.4)

    # Plot force comparison
    plot_force_comparison(fig.add_subplot(gs[0, 0]), u_true_np, u_pred_np, force_y, sampled_predictions_np)

    # Plot observed displacement
    plot_displacement_field(Y_np, s_observed_np, "Observed Displacement Field",
                           fig.add_subplot(gs[0, 1]), "Displacement", scaling_order=disp_order, vmin=vmin, vmax=vmax)

    # Plot re-simulated displacement
    ax_resim = fig.add_subplot(gs[0, 2])
    if resim_disp_np is not None:
        plot_displacement_field(Y_np, resim_disp_np, "Re-simulated Displacement Field",
                               ax_resim, "Displacement", scaling_order=disp_order, vmin=vmin, vmax=vmax)
    else:
        ax_resim.axis("off")
        ax_resim.set_title("Re-simulated Displacement Unavailable", fontsize=18, pad=15)

    # Align subplots - match force plot height to displacement plots
    plt.tight_layout()
    fig.canvas.draw()
    force_ax = fig.axes[0]
    disp_ax = fig.axes[1]
    disp_bbox = disp_ax.get_position()
    force_bbox = force_ax.get_position()
    force_ax.set_position([force_bbox.x0, disp_bbox.y0, force_bbox.width, disp_bbox.height])
    fig.canvas.draw()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    → Saved: {save_path}")
    plt.close()


def list_models_to_plot(requested_model, log_dir, seed):
    """Return list of models to plot."""
    if requested_model:
        return [requested_model]

    available = [m for m in ALL_MODELS if (log_dir / m / f"seed_{seed}" / "params.pth").exists()]
    if not available:
        print(f"ERROR: No trained models found in {log_dir}\nTo train models, run: ./run_all.sh")
        sys.exit(1)

    print(f"Found {len(available)} trained models: {', '.join(available)}")
    return available


def load_test_dataset(models_to_plot, log_dir, seed):
    """Load test dataset from first available model."""
    params_path = log_dir / models_to_plot[0] / f"seed_{seed}" / "params.pth"
    params = torch.load(params_path, weights_only=False)
    print("Loading dataset...")
    test_dataset, dataset_info = load_dataset(params.dataset, params, DEVICE, split="test", return_info=True)
    print(f"  Test samples: {len(test_dataset)}")
    return test_dataset, dataset_info


def process_model(model_name, log_dir, seed, test_dataset, test_indices, dataset_info, results_root):
    """Process and plot a single model."""
    model_log_dir = log_dir / model_name / f"seed_{seed}"
    params_path = model_log_dir / "params.pth"

    if not params_path.exists():
        print(f"ERROR: Model '{model_name}' not found at {params_path}")
        return False

    try:
        params = torch.load(params_path, weights_only=False)
        print(f"Loading {model_name}...")
        input_function_encoder, output_function_encoder, model, evaluate_fn = load_models(
            log_dir=str(model_log_dir), dataset_info=dataset_info, params=params, device=DEVICE)

        # Load forward model if available
        forward_model = None
        if forward_model_name := getattr(params, "forward_model", None):
            try:
                forward_model = load_forward_model(str(model_log_dir), forward_model_name=forward_model_name, device=DEVICE)
            except Exception as exc:
                print(f"  Warning: forward model unavailable ({exc})")

        # Plot samples
        model_results_dir = results_root / model_name
        print(f"Plotting {len(test_indices)} samples...")
        for idx_num, sample_idx in enumerate(test_indices, start=1):
            print(f"  Processing sample {idx_num}/{len(test_indices)} (index {sample_idx})...")
            plot_elastic_sample(model, evaluate_fn, input_function_encoder, output_function_encoder,
                              test_dataset[sample_idx], model_name, forward_model,
                              save_path=model_results_dir / f"sample_{sample_idx}.png")

        print(f"✓ Successfully plotted {model_name}")
        return True

    except Exception as exc:
        print(f"✗ Failed to plot {model_name}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Plot Elastic Plate results.")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name to plot. If omitted, plot all trained models.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Base log directory.")
    parser.add_argument("--results_dir", type=str, default="results/elastic_plots", help="Output directory.")
    parser.add_argument("--n_samples", type=int, default=7, help="Number of samples to plot.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    args = parser.parse_args()

    set_seeds(args.seed)

    # Setup paths and load data
    log_dir = Path(args.log_dir) / "elastic_plate"
    models_to_plot = list_models_to_plot(args.model, log_dir, args.seed)
    test_dataset, dataset_info = load_test_dataset(models_to_plot, log_dir, args.seed)

    # Select random test samples
    sample_count = min(args.n_samples, len(test_dataset))
    test_indices = random.sample(range(len(test_dataset)), sample_count)
    results_root = Path(args.results_dir)

    # Process each model
    successful_models, failed_models = [], []
    for model_name in models_to_plot:
        print(f"\n{'=' * 50}\nProcessing model: {model_name}\n{'=' * 50}")
        success = process_model(model_name, log_dir, args.seed, test_dataset, test_indices, dataset_info, results_root)
        (successful_models if success else failed_models).append(model_name)

    # Print summary
    print(f"\n{'=' * 50}\nSUMMARY\n{'=' * 50}")
    if successful_models:
        print(f"✓ Successfully plotted {len(successful_models)} model(s): {', '.join(successful_models)}")
        print(f"\n  Results saved to: {results_root}/")
        for model in successful_models:
            print(f"    └── {model}/sample_*.png")
    if failed_models:
        print(f"✗ Failed to plot {len(failed_models)} model(s): {', '.join(failed_models)}")


if __name__ == "__main__":
    main()
