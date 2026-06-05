"""Function encoder construction using public function_encoder APIs."""

from __future__ import annotations

import functools
import math
from typing import Callable, Iterable, List, Optional

import torch


class Sine(torch.nn.Module):
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


def _activation(name: str) -> torch.nn.Module:
    if name == "relu":
        return torch.nn.ReLU()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "tanh":
        return torch.nn.Tanh()
    raise ValueError(f"Unsupported MLP activation: {name}")


def _init_siren_linear(
    layer: torch.nn.Linear,
    *,
    layer_index: int,
    omega_0: float,
) -> torch.nn.Linear:
    with torch.no_grad():
        in_features = layer.weight.shape[1]
        if layer_index == 0:
            bound = 1 / in_features
        else:
            bound = math.sqrt(6 / in_features) / omega_0
        layer.weight.uniform_(-bound, bound)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound)
    return layer


def _make_mlp(
    input_size: int,
    hidden_sizes: Iterable[int],
    output_size: int,
    activation: str,
) -> torch.nn.Sequential:
    sizes = [input_size] + list(hidden_sizes) + [output_size]
    layers: List[torch.nn.Module] = []
    for idx, (in_features, out_features) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(torch.nn.Linear(in_features, out_features))
        if idx < len(sizes) - 2:
            layers.append(_activation(activation))
    return torch.nn.Sequential(*layers)


def _make_siren(
    input_size: int,
    hidden_sizes: Iterable[int],
    output_size: int,
    omega_0: float,
) -> torch.nn.Sequential:
    sizes = [input_size] + list(hidden_sizes) + [output_size]
    layers: List[torch.nn.Module] = []
    for idx, (in_features, out_features) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(
            _init_siren_linear(
                torch.nn.Linear(in_features, out_features),
                layer_index=idx,
                omega_0=omega_0,
            )
        )
        if idx < len(sizes) - 2:
            layers.append(Sine(omega_0=omega_0))
    return torch.nn.Sequential(*layers)


def memory_efficient_inner_product(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    b, m, d, k = f.shape
    l = g.shape[-1]
    f_flat = f.reshape(b, m * d, k)
    g_flat = g.reshape(b, m * d, l)
    return torch.matmul(f_flat.transpose(1, 2), g_flat) / m


class ChunkedFunctionEncoder(torch.nn.Module):
    """Function encoder that avoids materializing every basis output at once."""

    def __init__(
        self,
        basis_functions: torch.nn.Module,
        *,
        regularization: float,
        basis_chunk_size: int,
        residual_function: Optional[torch.nn.Module] = None,
        inner_product: Callable = memory_efficient_inner_product,
    ):
        super().__init__()
        self.basis_functions = basis_functions
        self.residual_function = residual_function
        self.regularization = regularization
        self.basis_chunk_size = basis_chunk_size
        self.inner_product = inner_product

    def _basis_chunks(self, x: torch.Tensor):
        basis_networks = self.basis_functions.basis_functions
        for start in range(0, len(basis_networks), self.basis_chunk_size):
            chunk = basis_networks[start : start + self.basis_chunk_size]
            yield start, torch.stack([basis(x) for basis in chunk], dim=-1)

    def compute_coefficients(self, x: torch.Tensor, y: torch.Tensor):
        f = y
        if self.residual_function is not None:
            f = f - self.residual_function(x).detach()

        f_chunks = []
        g_rows = []
        for _, g_left in self._basis_chunks(x):
            f_chunks.append(self.inner_product(g_left, f.unsqueeze(-1)).squeeze(-1))
            row_chunks = []
            for _, g_right in self._basis_chunks(x):
                row_chunks.append(self.inner_product(g_left, g_right))
            g_rows.append(torch.cat(row_chunks, dim=-1))
        F = torch.cat(f_chunks, dim=-1)
        G = torch.cat(g_rows, dim=-2)
        eye = torch.eye(G.size(-1), device=G.device, dtype=G.dtype)
        coefficients = torch.linalg.solve(G + self.regularization * eye, F)
        return coefficients, G

    def forward(self, x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        y = None
        for start, g in self._basis_chunks(x):
            stop = start + g.shape[-1]
            chunk_coefficients = coefficients[:, start:stop]
            chunk_y = torch.einsum("bmdk,bk->bmd", g, chunk_coefficients)
            y = chunk_y if y is None else y + chunk_y
        if y is None:
            raise ValueError("Function encoder has no basis functions.")
        if self.residual_function is not None:
            y = y + self.residual_function(x).detach()
        return y


def create_function_encoder(
    *,
    input_size: int,
    output_size: int,
    hidden_sizes: Iterable[int],
    n_basis: int,
    basis_kind: str,
    activation: str = "relu",
    omega_0: float = 30.0,
    regularization: float = 1e-3,
    inner_product: Optional[Callable] = None,
    basis_chunk_size: Optional[int] = None,
):
    """Create a FunctionEncoder from standard torch Sequential basis networks."""
    from function_encoder.coefficients import least_squares
    from function_encoder.function_encoder import BasisFunctions, FunctionEncoder

    basis_networks = []
    for _ in range(n_basis):
        if basis_kind == "mlp":
            basis_networks.append(
                _make_mlp(input_size, hidden_sizes, output_size, activation)
            )
        elif basis_kind == "siren":
            basis_networks.append(
                _make_siren(input_size, hidden_sizes, output_size, omega_0)
            )
        else:
            raise ValueError(f"Unsupported function encoder basis kind: {basis_kind}")

    basis_functions = BasisFunctions(*basis_networks)
    if basis_chunk_size is not None:
        return ChunkedFunctionEncoder(
            basis_functions,
            regularization=regularization,
            basis_chunk_size=basis_chunk_size,
            inner_product=inner_product or memory_efficient_inner_product,
        )

    coefficients_method = functools.partial(least_squares, regularization=regularization)
    kwargs = {"coefficients_method": coefficients_method}
    if inner_product is not None:
        kwargs["inner_product"] = inner_product

    return FunctionEncoder(basis_functions, **kwargs)
