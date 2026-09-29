"""Regression checks for SIREN² spectral statistics."""

import torch

from inverse_neural_operator.function_encoders.train import _estimate_spectral_stats


class _OneDimensionalDataset:
    def __init__(self, count=8, length=64):
        self.count = count
        self.length = length

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        x = torch.linspace(0, 1, self.length).unsqueeze(-1)
        frequency = index % 4 + 1
        u = torch.sin(2 * torch.pi * frequency * x)
        s = torch.cos(2 * torch.pi * frequency * x)
        return x, u, x, s


def test_spectral_stats_support_one_dimensional_fields():
    stats = _estimate_spectral_stats(
        _OneDimensionalDataset(),
        encoder_type="input",
        spatial_dims=(64,),
        sample_count=8,
        seed=1,
    )
    assert stats["enabled"] is True
    assert stats["sample_count"] == 8
    assert 0 < stats["spectral_centroid"] < 1
    assert 0 <= stats["high_frequency_ratio"] <= 1
