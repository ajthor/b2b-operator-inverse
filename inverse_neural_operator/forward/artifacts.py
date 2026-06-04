"""Load saved forward model artifacts from the overhaul model layout."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from safetensors.torch import load_file

from inverse_neural_operator.forward.build import create_forward_model


def load_forward_model(
    artifact_dir: Path,
    *,
    model_name: str,
    input_size: int,
    output_size: int,
    hidden_sizes: Optional[list[int]],
    device,
):
    weights_path = artifact_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing forward model weights: {weights_path}")
    model = create_forward_model(
        model_name,
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes or [128, 128],
    ).to(device)
    model.load_state_dict(load_file(str(weights_path), device=str(device)))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model

