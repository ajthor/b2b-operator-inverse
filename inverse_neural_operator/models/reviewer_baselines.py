"""Reviewer-requested raw function-space inverse baselines."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch


def _mlp(
    input_size: int,
    hidden_sizes: Iterable[int],
    output_size: int,
    *,
    activation: Optional[torch.nn.Module] = None,
) -> torch.nn.Sequential:
    activation = activation or torch.nn.ReLU()
    sizes = [input_size] + list(hidden_sizes) + [output_size]
    layers: list[torch.nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(activation.__class__())
    return torch.nn.Sequential(*layers)


class RealNVPCoupling(torch.nn.Module):
    def __init__(
        self,
        size: int,
        hidden_sizes: Iterable[int],
        mask: torch.Tensor,
        *,
        scale_clip: float = 5.0,
    ):
        super().__init__()
        self.size = size
        self.scale_clip = scale_clip
        self.register_buffer("mask", mask.float())
        self.register_buffer("inv_mask", 1.0 - mask.float())
        self.net = _mlp(size, hidden_sizes, 2 * size)

    def _scale(self, value: torch.Tensor) -> torch.Tensor:
        if self.scale_clip <= 0:
            return value
        return torch.tanh(value) * self.scale_clip

    def forward(self, x: torch.Tensor):
        mask = self.mask.to(dtype=x.dtype)
        inv_mask = self.inv_mask.to(dtype=x.dtype)
        x_masked = x * mask
        scale, shift = torch.chunk(self.net(x_masked), 2, dim=-1)
        scale = self._scale(scale)
        y = x_masked + inv_mask * (x * torch.exp(scale * inv_mask) + shift)
        log_det = (scale * inv_mask).sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor):
        mask = self.mask.to(dtype=y.dtype)
        inv_mask = self.inv_mask.to(dtype=y.dtype)
        y_masked = y * mask
        scale, shift = torch.chunk(self.net(y_masked), 2, dim=-1)
        scale = self._scale(scale)
        x = y_masked + inv_mask * ((y - shift) * torch.exp(-scale * inv_mask))
        return x


class RealNVPBranch(torch.nn.Module):
    def __init__(self, size: int, hidden_sizes: Iterable[int], n_coupling_layers: int):
        super().__init__()
        base_mask = (torch.arange(size) % 2).float()
        layers = []
        for index in range(n_coupling_layers):
            mask = base_mask if index % 2 == 0 else 1.0 - base_mask
            layers.append(RealNVPCoupling(size, hidden_sizes, mask))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        z = x
        log_det_total = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def inverse(self, z: torch.Tensor):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x


class InvertibleDeepONetBaseline(torch.nn.Module):
    """Supervised adaptation of RealNVP-branch invertible DeepONet."""

    def __init__(
        self,
        *,
        input_points: int,
        input_channels: int,
        output_channels: int,
        coordinate_dim: int,
        hidden_sizes: Iterable[int],
        trunk_hidden_sizes: Optional[Iterable[int]],
        n_coupling_layers: int,
        regularization: float,
    ):
        super().__init__()
        self.input_points = input_points
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_dim = input_points * input_channels
        self.regularization = regularization
        self.branch = RealNVPBranch(
            self.input_dim,
            hidden_sizes=hidden_sizes,
            n_coupling_layers=n_coupling_layers,
        )
        self.trunk = _mlp(
            coordinate_dim,
            trunk_hidden_sizes or hidden_sizes,
            self.input_dim * output_channels,
        )
        self.bias = torch.nn.Parameter(torch.zeros(output_channels))

    def trunk_matrix(self, y: torch.Tensor) -> torch.Tensor:
        batch, n_points, _ = y.shape
        values = self.trunk(y).view(batch, n_points, self.output_channels, self.input_dim)
        if self.output_channels == 1:
            return values[:, :, 0, :]
        return values.reshape(batch, n_points * self.output_channels, self.input_dim)

    def forward_from_input(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch = u.shape[0]
        coefficients, _ = self.branch(u.reshape(batch, -1))
        trunk = self.trunk_matrix(y)
        pred = torch.einsum("bkd,bd->bk", trunk, coefficients)
        pred = pred.view(batch, y.shape[1], self.output_channels)
        return pred + self.bias.view(1, 1, -1)

    def inverse_from_output(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch = s.shape[0]
        trunk = self.trunk_matrix(y)
        target = (s - self.bias.view(1, 1, -1)).reshape(batch, -1)
        gram = torch.bmm(trunk.transpose(1, 2), trunk)
        rhs = torch.bmm(trunk.transpose(1, 2), target.unsqueeze(-1)).squeeze(-1)
        eye = torch.eye(self.input_dim, device=s.device, dtype=s.dtype).unsqueeze(0)
        coefficients = torch.linalg.solve(
            gram + self.regularization * eye,
            rhs,
        )
        u = self.branch.inverse(coefficients)
        return u.view(batch, self.input_points, self.input_channels)

    def predict_inverse(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
        return self.inverse_from_output(s, y)


class SpectralConv1d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / math.sqrt(in_channels * out_channels)
        weight = scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, n_points = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            x_ft.shape[-1],
            dtype=torch.cfloat,
            device=x.device,
        )
        modes = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :modes],
            self.weight[:, :, :modes],
        )
        return torch.fft.irfft(out_ft, n=n_points, dim=-1)


class FNOBlock1d(torch.nn.Module):
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.pointwise = torch.nn.Conv1d(channels, channels, kernel_size=1)
        self.activation = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.pointwise(x))


class NIOBaseline(torch.nn.Module):
    """Supervised adaptation of DeepONet measurement aggregation plus FNO mixer."""

    def __init__(
        self,
        *,
        output_coordinate_dim: int,
        output_channels: int,
        input_coordinate_dim: int,
        input_channels: int,
        branch_hidden_sizes: Iterable[int],
        trunk_hidden_sizes: Iterable[int],
        n_basis: int,
        lifting_channels: int,
        modes: int,
        n_fourier_layers: int,
        measurement_points: Optional[int],
    ):
        super().__init__()
        self.input_channels = input_channels
        self.n_basis = n_basis
        self.lifting_channels = lifting_channels
        self.measurement_points = measurement_points
        self.branch = _mlp(output_coordinate_dim + output_channels, branch_hidden_sizes, n_basis)
        self.trunk = _mlp(input_coordinate_dim, trunk_hidden_sizes, n_basis)
        self.lift = torch.nn.Linear(n_basis + input_coordinate_dim, lifting_channels)
        self.blocks = torch.nn.ModuleList(
            [FNOBlock1d(lifting_channels, modes) for _ in range(n_fourier_layers)]
        )
        self.project = torch.nn.Sequential(
            torch.nn.Conv1d(lifting_channels, lifting_channels, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(lifting_channels, input_channels, kernel_size=1),
        )

    def _sample_measurements(self, y: torch.Tensor, s: torch.Tensor):
        if (
            not self.training
            or self.measurement_points is None
            or self.measurement_points >= y.shape[1]
        ):
            return y, s
        indices = torch.randperm(y.shape[1], device=y.device)[: self.measurement_points]
        return y[:, indices], s[:, indices]

    def predict_inverse(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
        y_sample, s_sample = self._sample_measurements(y, s)
        measurement_features = torch.cat([y_sample, s_sample], dim=-1)
        branch = self.branch(measurement_features)
        trunk = self.trunk(x)
        deeponet_features = trunk * branch.mean(dim=1).unsqueeze(1)
        lifted = self.lift(torch.cat([deeponet_features, x], dim=-1)).permute(0, 2, 1)
        for block in self.blocks:
            lifted = block(lifted)
        return self.project(lifted).permute(0, 2, 1)


def create_invertible_deeponet(dataset_info, config):
    cfg = config.baselines.invertible_deeponet
    return InvertibleDeepONetBaseline(
        input_points=dataset_info["u_len"],
        input_channels=dataset_info["input_function_channels"],
        output_channels=dataset_info["output_function_channels"],
        coordinate_dim=dataset_info["Y_size"],
        hidden_sizes=cfg.hidden_sizes,
        trunk_hidden_sizes=cfg.trunk_hidden_sizes,
        n_coupling_layers=cfg.n_coupling_layers,
        regularization=cfg.regularization,
    )


def create_nio(dataset_info, config):
    cfg = config.baselines.nio
    return NIOBaseline(
        output_coordinate_dim=dataset_info["Y_size"],
        output_channels=dataset_info["output_function_channels"],
        input_coordinate_dim=dataset_info["X_size"],
        input_channels=dataset_info["input_function_channels"],
        branch_hidden_sizes=cfg.branch_hidden_sizes,
        trunk_hidden_sizes=cfg.trunk_hidden_sizes,
        n_basis=cfg.n_basis,
        lifting_channels=cfg.lifting_channels,
        modes=cfg.modes,
        n_fourier_layers=cfg.n_fourier_layers,
        measurement_points=cfg.measurement_points,
    )
