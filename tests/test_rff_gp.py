import torch

from inverse_neural_operator.config.schema import RFFGPConfig
from inverse_neural_operator.inverse.build import create_inverse_model
from inverse_neural_operator.models.rff_gp import RandomFourierGaussianProcess


def _fit_model():
    torch.manual_seed(11)
    inputs = torch.linspace(-1, 1, 96).unsqueeze(-1)
    targets = torch.cat([torch.sin(2 * inputs), inputs.square()], dim=-1)
    model = RandomFourierGaussianProcess(
        input_size=1,
        output_size=2,
        num_features=64,
        lengthscale=0.6,
        noise_variance=1e-3,
    )
    model.set_normalization(
        inputs.mean(0),
        inputs.std(0, unbiased=False),
        targets.mean(0),
        targets.std(0, unbiased=False),
    )
    model.initialize_features(inputs, seed=4)
    for start in range(0, len(inputs), 13):
        model.accumulate(inputs[start : start + 13], targets[start : start + 13])
    model.finalize()
    return model, inputs, targets


def test_rff_gp_streaming_fit_and_uncertainty():
    model, inputs, targets = _fit_model()
    prediction = model.predict_distribution(inputs)
    assert prediction.mean.shape == targets.shape
    assert prediction.predictive_variance.shape == targets.shape
    assert torch.all(prediction.predictive_variance > prediction.epistemic_variance)
    assert torch.mean((prediction.mean - targets).square()) < 0.02
    assert int(model.num_observations) == len(inputs)
    assert model._feature_gram is None


def test_rff_gp_is_seed_deterministic_and_reloadable():
    model, inputs, _ = _fit_model()
    restored = RandomFourierGaussianProcess(
        1, 2, num_features=64, lengthscale=0.6, noise_variance=1e-3
    )
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(inputs), model(inputs))


def test_inverse_builder_creates_rff_gp():
    model = create_inverse_model(
        "rff_gp",
        input_size=3,
        output_size=4,
        hidden_sizes=[],
        rff_gp_config=RFFGPConfig(num_features=32),
    )
    assert model.input_size == 4
    assert model.output_size == 3
    assert model.feature_count == 32
