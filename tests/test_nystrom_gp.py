import torch

from inverse_neural_operator.config.schema import NystromGPConfig
from inverse_neural_operator.inverse.build import create_inverse_model
from inverse_neural_operator.models.nystrom_gp import NystromGaussianProcess


def _fit_model():
    torch.manual_seed(7)
    inputs = torch.linspace(-1, 1, 48).unsqueeze(-1)
    targets = torch.cat([torch.sin(2 * inputs), inputs.square()], dim=-1)
    model = NystromGaussianProcess(
        input_size=1,
        output_size=2,
        num_inducing=12,
        lengthscale=0.5,
        noise_variance=1e-3,
    )
    model.set_normalization(
        inputs.mean(0), inputs.std(0, unbiased=False), targets.mean(0), targets.std(0, unbiased=False)
    )
    model.initialize_inducing(inputs, seed=3)
    for start in range(0, len(inputs), 7):
        model.accumulate(inputs[start : start + 7], targets[start : start + 7])
    model.finalize()
    return model, inputs, targets


def test_nystrom_gp_streaming_fit_and_uncertainty():
    model, inputs, targets = _fit_model()
    prediction = model.predict_distribution(inputs)
    assert prediction.mean.shape == targets.shape
    assert prediction.epistemic_variance.shape == targets.shape
    assert torch.all(prediction.predictive_variance > prediction.epistemic_variance)
    covariance = model.predictive_covariance(inputs[:3])
    assert covariance.shape == (3, 2, 2)
    assert torch.all(torch.linalg.eigvalsh(covariance) > 0)
    assert torch.mean((prediction.mean - targets).square()) < 0.02
    assert int(model.num_observations) == len(inputs)
    assert model._feature_gram is None
    assert model._feature_cross is None


def test_nystrom_gp_uncertainty_increases_away_from_data():
    model, _, _ = _fit_model()
    near = model.predict_distribution(torch.tensor([[0.0]])).epistemic_variance.mean()
    far = model.predict_distribution(torch.tensor([[8.0]])).epistemic_variance.mean()
    assert far > near


def test_nystrom_gp_state_dict_round_trip():
    model, inputs, _ = _fit_model()
    restored = NystromGaussianProcess(
        input_size=1,
        output_size=2,
        num_inducing=12,
        lengthscale=0.5,
        noise_variance=1e-3,
    )
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(inputs), model(inputs))


def test_inverse_builder_creates_nystrom_gp():
    config = NystromGPConfig(num_inducing=8)
    model = create_inverse_model(
        "nystrom_gp",
        input_size=3,
        output_size=4,
        hidden_sizes=[],
        nystrom_gp_config=config,
    )
    assert model.input_size == 4
    assert model.output_size == 3
