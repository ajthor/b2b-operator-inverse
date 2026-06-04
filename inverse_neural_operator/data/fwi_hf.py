"""CPU-loaded Hugging Face FWI dataset for DDP training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from torch.utils.data import Dataset


class FWICpuDataset(Dataset):
    """FWI dataset that keeps samples on host memory until the training loop."""

    def __init__(self, dataset, stats: Dict[str, Any]):
        self.dataset = dataset
        self.stats = stats
        x_input = torch.linspace(0, 1, 24)
        y_input = torch.linspace(0, 1, 48)
        X_input, Y_input = torch.meshgrid(x_input, y_input, indexing="ij")
        self.X_template = torch.stack([X_input.flatten(), Y_input.flatten()], dim=1)

        x_output = torch.linspace(0, 1, 400)
        y_output = torch.linspace(0, 1, 76)
        X_output, Y_output = torch.meshgrid(x_output, y_output, indexing="ij")
        self.Y_template = torch.stack([X_output.flatten(), Y_output.flatten()], dim=1)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        u = torch.as_tensor(sample["models"], dtype=torch.float32).view(-1, 1)
        s = torch.as_tensor(sample["transforms"], dtype=torch.float32).view(-1, 1)
        return self.X_template, u, self.Y_template, s

    def get_info(self) -> Dict[str, Any]:
        return {
            "X_size": self.X_template.shape[-1],
            "u_size": 1,
            "Y_size": self.Y_template.shape[-1],
            "s_size": 1,
            "X_len": self.X_template.shape[0],
            "u_len": 24 * 48,
            "Y_len": self.Y_template.shape[0],
            "s_len": 400 * 76,
            "input_spatial_dims": (24, 48),
            "output_spatial_dims": (400, 76),
            "input_function_channels": 1,
            "output_function_channels": 1,
            "coordinate_dim": 2,
        }


def _stats_path() -> Path:
    return Path(__file__).resolve().parent / "fwi_stats.json"


def _load_stats() -> Dict[str, Any]:
    path = _stats_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FWI normalization stats at {path}. Use the reference loader or "
            "precompute stats in the devcontainer before running the new DDP path."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_batch(batch, stats):
    models = np.array(batch["models"]).squeeze(-1)
    residual_min = stats["models_min"] - 900.0
    residual_max = stats["models_max"] - 100.0
    if residual_max > residual_min:
        models = 2 * (models - residual_min) / (residual_max - residual_min) - 1
    batch["models"] = models

    transforms = np.array(batch["transforms"])
    if stats["transforms_max"] > stats["transforms_min"]:
        transforms = (
            2
            * (transforms - stats["transforms_min"])
            / (stats["transforms_max"] - stats["transforms_min"])
            - 1
        )
    batch["transforms"] = transforms
    return batch


def _load_limited_streaming_dataset(source: str, split: str, sample_limit: int, stats):
    ds = load_dataset(source, split=split, streaming=True)
    samples = []
    for idx, sample in enumerate(ds):
        if idx >= sample_limit:
            break
        samples.append(_normalize_batch(dict(sample), stats))
    return samples


def load_fwi_dataset(
    *,
    split: str,
    source: str = "ajthor/fwi",
    sample_limit: Optional[int] = None,
) -> FWICpuDataset:
    """Load FWI from the Hugging Face cache/source without moving data to GPU."""
    stats = _load_stats()
    try:
        if sample_limit is not None:
            ds = _load_limited_streaming_dataset(source, split, sample_limit, stats)
        else:
            ds = load_dataset(source, split=split, streaming=False)
            ds.set_transform(lambda batch: _normalize_batch(batch, stats))
    except DatasetNotFoundError as exc:
        raise RuntimeError(
            f"Could not load Hugging Face FWI dataset '{source}'. Confirm the "
            "dataset name, network access, and Hugging Face authentication inside "
            "the devcontainer. If this dataset is private, run `huggingface-cli "
            "login` or provide an HF token before executing training."
        ) from exc
    wrapped = FWICpuDataset(ds, stats)
    return wrapped
