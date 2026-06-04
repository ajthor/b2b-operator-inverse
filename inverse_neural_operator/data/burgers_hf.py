"""CPU-loaded Hugging Face Burgers dataset for baseline smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class BurgersCpuDataset(Dataset):
    """Burgers dataset that keeps tensors on host memory until the training loop."""

    def __init__(self, samples, stats: Dict[str, Any]):
        self.samples = samples
        self.stats = stats

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        x = torch.as_tensor(sample["spatial_coordinates"], dtype=torch.float32)
        u = torch.as_tensor(sample["u_initial"], dtype=torch.float32)
        s = torch.as_tensor(sample["u_trajectory"], dtype=torch.float32)[-1]
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if u.dim() == 1:
            u = u.unsqueeze(-1)
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        u = (u - self.stats["u_mean"]) / self.stats["u_std"]
        s = (s - self.stats["s_mean"]) / self.stats["s_std"]
        return x, u, x, s

    def get_info(self) -> Dict[str, Any]:
        x, u, y, s = self[0]
        return {
            "X_size": x.shape[-1],
            "u_size": u.shape[-1],
            "Y_size": y.shape[-1],
            "s_size": s.shape[-1],
            "X_len": x.shape[0],
            "u_len": u.shape[0],
            "Y_len": y.shape[0],
            "s_len": s.shape[0],
            "input_spatial_dims": (x.shape[0],),
            "output_spatial_dims": (y.shape[0],),
            "input_function_channels": u.shape[-1],
            "output_function_channels": s.shape[-1],
            "coordinate_dim": x.shape[-1],
            "normalization_stats": self.stats,
        }


def _stats_path() -> Path:
    return Path(__file__).resolve().parent / "burgers_1d_normalization_stats.json"


def _load_stats() -> Dict[str, Any]:
    with _stats_path().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compute_stats(samples) -> Dict[str, Any]:
    u = np.array([sample["u_initial"] for sample in samples])
    s = np.array([sample["u_trajectory"][-1] for sample in samples])
    return {
        "u_mean": float(np.mean(u)),
        "u_std": float(np.std(u)),
        "s_mean": float(np.mean(s)),
        "s_std": float(np.std(s)),
    }


def _materialize(source: str, split: str, sample_limit: Optional[int]):
    if sample_limit is None:
        return list(load_dataset(source, split=split))
    stream = load_dataset(source, split=split, streaming=True)
    samples = []
    for idx, sample in enumerate(stream):
        if idx >= sample_limit:
            break
        samples.append(dict(sample))
    return samples


def load_burgers_dataset(
    *,
    split: str,
    source: str = "ajthor/burgers-fenics",
    sample_limit: Optional[int] = None,
) -> BurgersCpuDataset:
    samples = _materialize(source, split, sample_limit)
    if not samples:
        raise ValueError(f"No Burgers samples loaded from {source} split={split}.")
    stats = _load_stats() if _stats_path().exists() else _compute_stats(samples)
    return BurgersCpuDataset(samples, stats)

