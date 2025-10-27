"""
Shared plotting utilities for publication-quality figures.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from b2b.load_model import load_forward_model as load_b2b_forward_model


# Display names for models
DISPLAY_NAMES = {
    "linear": "Linear",
    "linear_inverse": "Linear-Inv",
    "nonlinear": "Nonlinear",
    "inn_affine": "INN-Affine",
    "inn_additive": "INN-Add",
    "cinn_affine": "cINN-Affine",
    "cinn_additive": "cINN-Add",
    "variational_autoencoder": "cVAE",
    "conditional_realnvp": "RealNVP",
    "mixture_density_network": "MDN",
}


def setup_publication_style(figsize=(6.5, 1.5)):
    """Configure matplotlib for publication-quality output."""
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "figure.figsize": list(figsize),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": False,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
        }
    )


def display_name(model_name: str) -> str:
    """Convert internal model name to display name."""
    return DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title())


def find_params(log_dir: str, model_names, seed: int = 1):
    """Find any available params file to recover dataset settings."""
    for model_name in model_names:
        params_path = os.path.join(log_dir, model_name, f"seed_{seed}", "params.pth")
        if os.path.exists(params_path):
            return torch.load(params_path, weights_only=False)
    raise FileNotFoundError(f"No trained models found under {log_dir}")


def load_forward_model(log_dir: str, seed: int, forward_model_name: str = "b2b_nonlinear", device: str = "cpu"):
    """Load a forward model for re-simulations."""
    shared_log_dir = os.path.join(log_dir, "shared", f"seed_{seed}")
    try:
        forward_model = load_b2b_forward_model(
            log_dir=shared_log_dir, forward_model_name=forward_model_name, device=device
        )
        forward_model.eval()
        print(f"  Loaded forward model: {forward_model_name}")
        return forward_model
    except Exception as exc:
        print(f"  Warning: could not load forward model {forward_model_name}: {exc}")
        return None


def make_output_transform(forward_model, output_function_encoder):
    """Create a callable that maps latent codes to re-simulated outputs."""
    if forward_model is None:
        return None

    def transform(latent, Y):
        beta_pred = forward_model.forward(latent)
        return output_function_encoder(Y.unsqueeze(0), beta_pred)

    return transform
