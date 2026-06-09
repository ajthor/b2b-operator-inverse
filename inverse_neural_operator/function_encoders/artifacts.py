"""Load saved function encoder artifacts from the overhaul model layout."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import yaml


def _encoder_sizes(encoder_type: str, dataset_info: Dict[str, Any]) -> tuple[int, int]:
    if encoder_type == "input":
        return dataset_info["X_size"], dataset_info["u_size"]
    if encoder_type == "output":
        return dataset_info["Y_size"], dataset_info["s_size"]
    raise ValueError(f"Unknown function encoder type: {encoder_type}")


def function_encoder_files(encoder_types: Iterable[str]) -> Dict[str, str]:
    files = {
        "config": "config.yaml",
        "manifest": "manifest.json",
        "metrics": "metrics.json",
    }
    for encoder_type in encoder_types:
        files[encoder_type] = f"{encoder_type}_encoder.safetensors"
    return files


def missing_function_encoder_files(
    artifact_dir: Path,
    *,
    encoder_types: Iterable[str] = ("input", "output"),
) -> List[str]:
    return [
        filename
        for filename in function_encoder_files(encoder_types).values()
        if not (artifact_dir / filename).exists()
    ]


def require_function_encoder_artifact(
    artifact_dir: Path,
    *,
    encoder_types: Iterable[str] = ("input", "output"),
) -> None:
    missing = missing_function_encoder_files(
        artifact_dir,
        encoder_types=encoder_types,
    )
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Function encoder artifact is incomplete at {artifact_dir}: "
            f"missing {missing_list}"
        )


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
    from inverse_neural_operator.function_encoders.build import (
        create_function_encoder,
        memory_efficient_inner_product,
    )

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
        basis_chunk_size=fe_raw.get("basis_chunk_size"),
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
        basis_chunk_size=fe_config.basis_chunk_size,
        inner_product=(
            memory_efficient_inner_product
            if raw_config.get("dataset", {}).get("name") == "fwi"
            else None
        ),
    ).to(device)
    model.load_state_dict(load_file(str(weights_path), device=str(device)))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
