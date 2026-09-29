"""Checks for configurable MLP function-encoder activations."""

import pytest
import torch

from inverse_neural_operator.function_encoders.build import _make_mlp


@pytest.mark.parametrize(
    ("name", "activation_type"),
    [("relu", torch.nn.ReLU), ("gelu", torch.nn.GELU), ("silu", torch.nn.SiLU)],
)
def test_mlp_activation_builds_and_backpropagates(name, activation_type):
    model = _make_mlp(1, [8, 8], 1, name)
    assert any(isinstance(module, activation_type) for module in model)
    x = torch.randn(4, 16, 1)
    loss = model(x).square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
