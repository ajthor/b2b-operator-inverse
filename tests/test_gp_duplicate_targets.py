import pytest
import torch

from inverse_neural_operator.models.nystrom_gp import NystromGaussianProcess
from inverse_neural_operator.models.rff_gp import RandomFourierGaussianProcess


@pytest.mark.parametrize("model_kind", ["nystrom", "rff"])
def test_conflicting_duplicate_inputs_create_aleatoric_variance(model_kind):
    inputs = torch.cat([torch.zeros(40, 1), torch.ones(40, 1)])
    conflicting = torch.tensor([-1.0, 1.0]).repeat(20).unsqueeze(-1)
    targets = torch.cat([conflicting, torch.zeros(40, 1)])
    if model_kind == "nystrom":
        model = NystromGaussianProcess(
            1, 1, num_inducing=2, lengthscale=0.5, noise_variance=1e-3
        )
    else:
        model = RandomFourierGaussianProcess(
            1, 1, num_features=64, lengthscale=0.5, noise_variance=1e-3
        )
    model.set_normalization(
        inputs.mean(0),
        inputs.std(0, unbiased=False),
        targets.mean(0),
        targets.std(0, unbiased=False),
    )
    model.initialize_features(inputs, seed=5)
    model.accumulate(inputs, targets)
    model.finalize()

    prediction = model.predict_distribution(torch.tensor([[0.0], [1.0]]))
    conflicting_variance = prediction.aleatoric_variance[0, 0]
    consistent_variance = prediction.aleatoric_variance[1, 0]
    assert conflicting_variance > 0.5
    assert conflicting_variance > 10 * consistent_variance


@pytest.mark.parametrize("model_kind", ["nystrom", "rff"])
def test_legacy_gp_artifact_without_second_moment_loads(model_kind):
    if model_kind == "nystrom":
        model = NystromGaussianProcess(2, 3, num_inducing=4)
        restored = NystromGaussianProcess(2, 3, num_inducing=4)
    else:
        model = RandomFourierGaussianProcess(2, 3, num_features=4)
        restored = RandomFourierGaussianProcess(2, 3, num_features=4)
    legacy_state = dict(model.state_dict())
    legacy_state.pop("posterior_second_weights")
    restored.load_state_dict(legacy_state)
    assert torch.count_nonzero(restored.posterior_second_weights) == 0
