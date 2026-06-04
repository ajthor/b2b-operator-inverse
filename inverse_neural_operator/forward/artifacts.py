"""Load saved forward model artifacts from the overhaul model layout."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from inverse_neural_operator.forward.build import create_forward_model


def forward_model_files() -> dict[str, str]:
    return {
        "model": "model.safetensors",
        "config": "config.yaml",
        "manifest": "manifest.json",
        "metrics": "metrics.json",
    }


def missing_forward_model_files(artifact_dir: Path) -> List[str]:
    return [
        filename
        for filename in forward_model_files().values()
        if not (artifact_dir / filename).exists()
    ]


def require_forward_model_artifact(artifact_dir: Path) -> None:
    missing = missing_forward_model_files(artifact_dir)
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Forward model artifact is incomplete at {artifact_dir}: "
            f"missing {missing_list}"
        )


def load_forward_model(
    artifact_dir: Path,
    *,
    model_name: str,
    input_size: int,
    output_size: int,
    hidden_sizes: Optional[list[int]],
    device,
):
    from safetensors.torch import load_file

    require_forward_model_artifact(artifact_dir)
    weights_path = artifact_dir / "model.safetensors"
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
