"""Load saved function encoder artifacts from the overhaul model layout."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml

from inverse_neural_operator.function_encoders.build import create_function_encoder


def _encoder_sizes(encoder_type: str, dataset_info: Dict[str, Any]) -> tuple[int, int]:
    if encoder_type == "input":
        return dataset_info["X_size"], dataset_info["u_size"]
    if encoder_type == "output":
        return dataset_info["Y_size"], dataset_info["s_size"]
    raise ValueError(f"Unknown function encoder type: {encoder_type}")


def load_function_encoder(
    artifact_dir: Path,
    *,
    encoder_type: str,
    dataset_info: Dict[str, Any],
    device,
):
    """Rebuild and load one saved function encoder."""
    import torch
    from safetensors.torch import load_file

    config_path = artifact_dir / "config.yaml"
    weights_path = artifact_dir / f"{encoder_type}_encoder.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing function encoder config: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing function encoder weights: {weights_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    basis_raw = raw_config.get("function_encoders", {}).get("basis", {})
    fe_raw = raw_config.get("function_encoders", {})
    basis = SimpleNamespace(
        kind=basis_raw.get("kind", "mlp"),
        hidden_sizes=basis_raw.get("hidden_sizes", [512, 512, 512]),
        n_basis=basis_raw.get("n_basis", 100),
        activation=basis_raw.get("activation", "relu"),
        omega_0=basis_raw.get("omega_0", 30.0),
    )
    fe_config = SimpleNamespace(
        basis=basis,
        regularization=fe_raw.get("regularization", 1e-3),
    )

    input_size, output_size = _encoder_sizes(encoder_type, dataset_info)
    model = create_function_encoder(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=fe_config.basis.hidden_sizes,
        n_basis=fe_config.basis.n_basis,
        basis_kind=fe_config.basis.kind,
        activation=getattr(fe_config.basis, "activation", "relu"),
        omega_0=getattr(fe_config.basis, "omega_0", 30.0),
        regularization=fe_config.regularization,
    ).to(device)
    model.load_state_dict(load_file(str(weights_path), device=str(device)))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
