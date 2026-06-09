"""Artifact and result path helpers for the overhaul layout."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_results_dir() -> Path:
    env_value = os.environ.get("B2B_RESULTS_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (repo_root() / "results").resolve()


def models_root(required: bool = False, override: Optional[str] = None) -> Optional[Path]:
    value = override or os.environ.get("B2B_MODELS_DIR")
    if value:
        return Path(value).expanduser().resolve()
    if required:
        raise RuntimeError(
            "B2B_MODELS_DIR must be set for commands that write uploadable model "
            "artifacts. Use --models-dir or export B2B_MODELS_DIR."
        )
    return None


def results_root(override: Optional[str] = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    return default_results_dir()


def model_artifact_dir(
    root: Path,
    dataset: str,
    stage: str,
    name: str,
    seed: int,
) -> Path:
    return root / "models" / dataset / stage / name / f"seed_{seed}"


def run_artifact_dir(
    root: Path,
    dataset: str,
    stage: str,
    name: str,
    seed: int,
) -> Path:
    return root / dataset / stage / name / f"seed_{seed}"


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return to_plain_data(asdict(value))
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain_data(payload), handle, sort_keys=False)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_plain_data(payload), handle, indent=2)


def write_manifest(
    artifact_dir: Path,
    *,
    artifact_type: str,
    dataset: str,
    name: str,
    seed: int,
    files: Dict[str, str],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "artifact_type": artifact_type,
        "dataset": dataset,
        "name": name,
        "seed": seed,
        "files": files,
    }
    if extra:
        payload.update(extra)
    write_json(artifact_dir / "manifest.json", payload)


def refuse_existing_artifact(path: Path, *, overwrite: bool) -> None:
    """Protect final uploadable artifacts from accidental overwrite."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing final artifact: {path}. "
            "Pass --overwrite only for an intentional rerun. Recovery checkpoints "
            "live under the results directory and are separate from final artifacts."
        )
