import torch

from inverse_neural_operator.config.schema import FunctionEncoderConfig
from inverse_neural_operator.function_encoders.train import _basis_unit_norm_loss


def test_basis_unit_norm_loss_matches_reference_expression():
    gram = torch.tensor(
        [[[2.0, 9.0], [9.0, 0.0]], [[1.0, -4.0], [-4.0, 1.0]]],
        requires_grad=True,
    )
    expected = ((torch.diagonal(gram, dim1=1, dim2=2) - 1) ** 2).mean()

    actual = _basis_unit_norm_loss(gram)

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert gram.grad is not None
    assert gram.grad[0, 0, 1] == 0


def test_reference_stability_safeguards_are_enabled_by_default():
    config = FunctionEncoderConfig()

    assert config.basis_unit_norm_loss_weight == 1.0
    assert config.gradient_clip_norm == 1.0
