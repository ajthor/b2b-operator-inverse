"""Shared metric helpers for physical-space evaluations."""

from __future__ import annotations

from typing import Optional, Sequence


def relative_l2(prediction, target):
    import torch

    batch_size = prediction.shape[0]
    numerator = torch.norm(
        prediction.reshape(batch_size, -1) - target.reshape(batch_size, -1),
        p=2,
        dim=1,
    )
    denominator = torch.clamp(
        torch.norm(target.reshape(batch_size, -1), p=2, dim=1),
        min=1e-12,
    )
    return torch.mean(numerator / denominator)


def mean_ssim(prediction, target, spatial_dims: Sequence[int]) -> Optional[float]:
    if len(spatial_dims) != 2:
        return None
    try:
        import numpy as np
        from skimage.metrics import structural_similarity
    except Exception:
        return None

    pred = prediction.detach().float().cpu().reshape(
        prediction.shape[0],
        *spatial_dims,
        -1,
    )
    true = target.detach().float().cpu().reshape(target.shape[0], *spatial_dims, -1)
    values = []
    for pred_sample, true_sample in zip(pred, true):
        channel_values = []
        for channel in range(pred_sample.shape[-1]):
            pred_channel = pred_sample[..., channel].numpy()
            true_channel = true_sample[..., channel].numpy()
            data_range = float(np.max(true_channel) - np.min(true_channel))
            if data_range <= 0:
                data_range = 1.0
            min_dim = min(pred_channel.shape)
            if min_dim < 3:
                continue
            win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
            channel_values.append(
                structural_similarity(
                    true_channel,
                    pred_channel,
                    data_range=data_range,
                    win_size=win_size,
                )
            )
        if channel_values:
            values.append(float(np.mean(channel_values)))
    if not values:
        return None
    return float(np.mean(values))
