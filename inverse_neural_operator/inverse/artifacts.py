"""Check saved inverse model artifacts from the overhaul model layout."""

from __future__ import annotations

from pathlib import Path
from typing import List


def inverse_model_files() -> dict[str, str]:
    return {
        "model": "model.safetensors",
        "config": "config.yaml",
        "manifest": "manifest.json",
        "metrics": "metrics.json",
    }


def missing_inverse_model_files(artifact_dir: Path) -> List[str]:
    return [
        filename
        for filename in inverse_model_files().values()
        if not (artifact_dir / filename).exists()
    ]


def require_inverse_model_artifact(artifact_dir: Path) -> None:
    missing = missing_inverse_model_files(artifact_dir)
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Inverse model artifact is incomplete at {artifact_dir}: "
            f"missing {missing_list}"
        )
