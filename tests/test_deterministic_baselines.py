"""CPU checks for deterministic inverse DeepONet and FNO baselines."""

from __future__ import annotations

import types

import pytest
import torch

from inverse_neural_operator.baselines.train import _batch_metrics
from inverse_neural_operator.config.schema import DeepONetConfig, FNOConfig
from inverse_neural_operator.models.deterministic_baselines import (
    DeterministicDeepONet,
    DeterministicFNO,
    create_deeponet,
    create_fno,
)


def _info(length=32):
    return {
        "X_size": 1,
        "Y_size": 1,
        "input_function_channels": 1,
        "output_function_channels": 1,
        "input_spatial_dims": (length,),
        "output_spatial_dims": (length,),
    }


def _config():
    baselines = types.SimpleNamespace(
        deeponet=DeepONetConfig(
            branch_hidden_sizes=[16, 16],
            trunk_hidden_sizes=[16, 16],
            n_basis=8,
            loss="mse",
        ),
        fno=FNOConfig(
            lifting_channels=8,
            modes=6,
            n_fourier_layers=2,
            projection_channels=16,
            loss="mse",
        ),
    )
    return types.SimpleNamespace(baselines=baselines)


def _batch(batch=4, length=32):
    coords = torch.linspace(0, 1, length).view(1, length, 1).repeat(batch, 1, 1)
    s = torch.randn(batch, length, 1)
    u = 0.5 * s + torch.sin(2 * torch.pi * coords)
    return coords, u, coords.clone(), s


def test_deeponet_inverse_shape_and_gradients():
    model = create_deeponet(_info(), _config())
    assert isinstance(model, DeterministicDeepONet)
    x, u, y, s = _batch()
    prediction = model.predict_inverse(x, y, s)
    assert prediction.shape == u.shape
    torch.nn.functional.mse_loss(prediction, u).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("spatial_dims", [(32,), (8, 6)])
def test_fno_inverse_shape_and_gradients(spatial_dims):
    n_points = 1
    for dim in spatial_dims:
        n_points *= dim
    coordinate_dim = len(spatial_dims)
    model = DeterministicFNO(
        spatial_dims=spatial_dims,
        coordinate_dim=coordinate_dim,
        output_channels=1,
        input_channels=2,
        lifting_channels=8,
        modes=4,
        n_fourier_layers=2,
        projection_channels=12,
    )
    x = torch.randn(3, n_points, coordinate_dim)
    s = torch.randn(3, n_points, 1)
    target = torch.randn(3, n_points, 2)
    prediction = model.predict_inverse(x, x, s)
    assert prediction.shape == target.shape
    torch.nn.functional.mse_loss(prediction, target).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_fno_builder_rejects_mismatched_grids():
    info = _info()
    info["output_spatial_dims"] = (16,)
    with pytest.raises(ValueError, match="matching input and output grids"):
        create_fno(info, _config())


@pytest.mark.parametrize("model_name", ["deeponet", "fno"])
def test_shared_training_metrics(model_name):
    config = _config()
    builder = create_deeponet if model_name == "deeponet" else create_fno
    model = builder(_info(), config)
    metrics = _batch_metrics(model, _batch(), model_name, config)
    assert set(["loss", "input_loss", "input_l1", "input_mse", "input_relative_l2"]).issubset(metrics)
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
