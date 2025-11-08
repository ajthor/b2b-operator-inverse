"""
Helpers for loading and evaluating IFNO checkpoints inside publication plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from inverse_neural_operator.models.ifno import create_model


@dataclass
class IFNOConfig:
    modes1: int = 16
    modes2: int = 16
    width: int = 64
    beta: float = 2.0
    n_layers: int = 4
    padding: int = 20
    vae_latent_dim: int = 24
    intermediate_dim: int = 32


def _infer_ifno_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Infer missing architectural hyperparameters directly from saved weights."""
    inferred = {}
    if "p1.weight" in state_dict:
        inferred["width"] = int(state_dict["p1.weight"].shape[0])

    conv_keys = [k for k in state_dict.keys() if k.startswith("convs.") and k.endswith(".weights")]
    if conv_keys:
        inferred["n_layers"] = len(conv_keys) // 2  # two conv tensors per layer
        sample_conv = state_dict[conv_keys[0]]
        inferred["modes1"] = int(sample_conv.shape[-1])
        inferred["modes2"] = int(sample_conv.shape[-1])

    if "vae_net.fc_mu.weight" in state_dict:
        inferred["vae_latent_dim"] = int(state_dict["vae_net.fc_mu.weight"].shape[0])

    return inferred


def load_ifno_model(dataset_info: Dict, checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """Create an IFNO model with dataset-aware shapes and load weights from checkpoint."""
    if not checkpoint_path:
        raise ValueError("Missing IFNO checkpoint path.")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    cfg = IFNOConfig().__dict__.copy()
    cfg.update(_infer_ifno_config_from_state_dict(state_dict))

    model = create_model(
        input_size=None,
        hidden_sizes=[256, 256, 256],
        n_coupling_layers=2,
        modes1=cfg["modes1"],
        modes2=cfg["modes2"],
        width=cfg["width"],
        beta=cfg["beta"],
        n_layers=cfg["n_layers"],
        padding=cfg["padding"],
        vae_latent_dim=cfg["vae_latent_dim"],
        intermediate_dim=cfg["intermediate_dim"],
        input_spatial_dims=dataset_info["input_spatial_dims"],
        output_spatial_dims=dataset_info["output_spatial_dims"],
        input_function_channels=dataset_info["input_function_channels"],
        output_function_channels=dataset_info["output_function_channels"],
        coordinate_dim=dataset_info["coordinate_dim"],
    ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _trim_function_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Remove coordinate channels so shapes match the target function tensor."""
    if pred.shape[-1] > target.shape[-1]:
        return pred[..., -target.shape[-1] :]
    return pred


def collect_ifno_predictions(
    model: torch.nn.Module, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], device: str = "cpu"
) -> Dict[str, np.ndarray]:
    """Run IFNO inverse+forward passes for a single dataset sample."""
    model.eval()

    X, u_true, Y, s_true = sample
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_true = s_true.to(device)

    with torch.no_grad():
        # Reconstruct input function from observed output (inverse pass)
        s_input = torch.cat([Y.unsqueeze(0), s_true.unsqueeze(0)], dim=-1)
        inverse_result = model.inverse(s_input)
        if isinstance(inverse_result, (tuple, list)):
            u_pred = inverse_result[0]
        else:
            u_pred = inverse_result
        u_pred = _trim_function_channels(u_pred, u_true.unsqueeze(0))

        # Forward simulate from predicted input to obtain comparable output
        u_input = torch.cat([X.unsqueeze(0), u_pred], dim=-1)
        forward_result = model(u_input)
        if isinstance(forward_result, (tuple, list)):
            s_pred = forward_result[0]
        else:
            s_pred = forward_result
        s_pred = _trim_function_channels(s_pred, s_true.unsqueeze(0))

    u_np = u_pred.squeeze(0).detach().cpu().numpy().flatten()
    s_np = s_pred.squeeze(0).detach().cpu().numpy().flatten()

    return {
        "inputs": np.expand_dims(u_np, axis=0),
        "outputs": np.expand_dims(s_np, axis=0),
    }

