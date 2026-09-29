"""Deterministic supervised neural-operator baselines for inverse maps."""

from __future__ import annotations

import math
from typing import Iterable

import torch

from inverse_neural_operator.models.reviewer_baselines import FNOBlock1d, FNOBlock2d


def _mlp(input_size: int, hidden_sizes: Iterable[int], output_size: int):
    sizes = [input_size, *hidden_sizes, output_size]
    layers: list[torch.nn.Module] = []
    for index, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(torch.nn.Linear(in_size, out_size))
        if index < len(sizes) - 2:
            layers.append(torch.nn.GELU())
    return torch.nn.Sequential(*layers)


class DeterministicDeepONet(torch.nn.Module):
    """DeepONet trained directly on the inverse operator ``s -> u``.

    The branch receives the observed output function at a fixed sensor grid and
    the trunk receives each requested input-domain coordinate.  Separate basis
    coefficients are learned for every input-function channel.
    """

    def __init__(
        self,
        *,
        output_points: int,
        output_channels: int,
        input_coordinate_dim: int,
        input_channels: int,
        branch_hidden_sizes: Iterable[int],
        trunk_hidden_sizes: Iterable[int],
        n_basis: int,
    ):
        super().__init__()
        self.output_points = output_points
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.n_basis = n_basis
        basis_size = input_channels * n_basis
        self.branch = _mlp(
            output_points * output_channels,
            branch_hidden_sizes,
            basis_size,
        )
        self.trunk = _mlp(input_coordinate_dim, trunk_hidden_sizes, basis_size)
        self.bias = torch.nn.Parameter(torch.zeros(input_channels))

    def predict_inverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        del y  # Sensor locations are fixed by the dataset, as in standard DeepONet.
        batch, n_points, _ = x.shape
        if s.shape[1:] != (self.output_points, self.output_channels):
            raise ValueError(
                "DeepONet expected output observations with shape "
                f"(*, {self.output_points}, {self.output_channels}), got {tuple(s.shape)}."
            )
        branch = self.branch(s.reshape(batch, -1)).view(
            batch, self.input_channels, self.n_basis
        )
        trunk = self.trunk(x).view(
            batch, n_points, self.input_channels, self.n_basis
        )
        prediction = torch.einsum("bcp,bncp->bnc", branch, trunk)
        return prediction + self.bias.view(1, 1, -1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
        return self.predict_inverse(x, y, s)


class DeterministicFNO(torch.nn.Module):
    """Standard Fourier neural operator trained directly on ``s -> u``."""

    def __init__(
        self,
        *,
        spatial_dims: Iterable[int],
        coordinate_dim: int,
        output_channels: int,
        input_channels: int,
        lifting_channels: int,
        modes: int,
        n_fourier_layers: int,
        projection_channels: int,
    ):
        super().__init__()
        self.spatial_dims = tuple(int(dim) for dim in spatial_dims)
        self.spatial_ndim = len(self.spatial_dims)
        self.input_channels = input_channels
        self.lifting_channels = lifting_channels
        if self.spatial_ndim == 1:
            block_cls = FNOBlock1d
            conv_cls = torch.nn.Conv1d
        elif self.spatial_ndim == 2:
            block_cls = FNOBlock2d
            conv_cls = torch.nn.Conv2d
        else:
            raise ValueError(
                "DeterministicFNO supports 1D or 2D grids; got "
                f"spatial_dims={self.spatial_dims!r}."
            )
        self.lift = torch.nn.Linear(output_channels + coordinate_dim, lifting_channels)
        self.blocks = torch.nn.ModuleList(
            [block_cls(lifting_channels, modes) for _ in range(n_fourier_layers)]
        )
        self.project = torch.nn.Sequential(
            conv_cls(lifting_channels, projection_channels, kernel_size=1),
            torch.nn.GELU(),
            conv_cls(projection_channels, input_channels, kernel_size=1),
        )

    def predict_inverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        del x  # On same-grid problems y is also the prediction grid.
        batch, n_points, _ = s.shape
        expected_points = math.prod(self.spatial_dims)
        if n_points != expected_points:
            raise ValueError(
                f"FNO expected {expected_points} points for grid {self.spatial_dims}, "
                f"got {n_points}."
            )
        features = self.lift(torch.cat([s, y], dim=-1)).permute(0, 2, 1)
        if self.spatial_ndim == 2:
            features = features.reshape(batch, self.lifting_channels, *self.spatial_dims)
        for block in self.blocks:
            features = block(features)
        prediction = self.project(features)
        if self.spatial_ndim == 2:
            prediction = prediction.reshape(batch, self.input_channels, n_points)
        return prediction.permute(0, 2, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
        return self.predict_inverse(x, y, s)


def create_deeponet(dataset_info, config):
    cfg = config.baselines.deeponet
    return DeterministicDeepONet(
        output_points=math.prod(dataset_info["output_spatial_dims"]),
        output_channels=dataset_info["output_function_channels"],
        input_coordinate_dim=dataset_info["X_size"],
        input_channels=dataset_info["input_function_channels"],
        branch_hidden_sizes=cfg.branch_hidden_sizes,
        trunk_hidden_sizes=cfg.trunk_hidden_sizes,
        n_basis=cfg.n_basis,
    )


def create_fno(dataset_info, config):
    input_dims = tuple(dataset_info["input_spatial_dims"])
    output_dims = tuple(dataset_info["output_spatial_dims"])
    if input_dims != output_dims:
        raise ValueError(
            "The deterministic FNO inverse baseline requires matching input and output "
            f"grids; got input_spatial_dims={input_dims} and output_spatial_dims={output_dims}."
        )
    if dataset_info["X_size"] != dataset_info["Y_size"]:
        raise ValueError(
            "The deterministic FNO inverse baseline requires matching coordinate "
            f"dimensions; got X_size={dataset_info['X_size']} and Y_size={dataset_info['Y_size']}."
        )
    cfg = config.baselines.fno
    return DeterministicFNO(
        spatial_dims=input_dims,
        coordinate_dim=dataset_info["Y_size"],
        output_channels=dataset_info["output_function_channels"],
        input_channels=dataset_info["input_function_channels"],
        lifting_channels=cfg.lifting_channels,
        modes=cfg.modes,
        n_fourier_layers=cfg.n_fourier_layers,
        projection_channels=cfg.projection_channels,
    )
