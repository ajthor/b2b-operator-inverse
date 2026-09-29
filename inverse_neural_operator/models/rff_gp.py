"""Fixed-memory random-Fourier-feature GP for inverse coefficients."""

from __future__ import annotations

import math

import torch

from inverse_neural_operator.models.nystrom_gp import GaussianPrediction


class RandomFourierGaussianProcess(torch.nn.Module):
    """RBF-kernel GP approximated by a fixed random Fourier feature map."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        num_features: int = 256,
        lengthscale: float = 0.0,
        noise_variance: float = 1e-2,
        jitter: float = 1e-5,
    ):
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features must be positive.")
        if noise_variance <= 0:
            raise ValueError("noise_variance must be positive.")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_features = int(num_features)
        self.feature_count = self.num_features
        self.configured_lengthscale = float(lengthscale)
        self.noise_variance = float(noise_variance)
        self.jitter = float(jitter)

        self.register_buffer("input_mean", torch.zeros(self.input_size))
        self.register_buffer("input_scale", torch.ones(self.input_size))
        self.register_buffer("target_mean", torch.zeros(self.output_size))
        self.register_buffer("target_scale", torch.ones(self.output_size))
        self.register_buffer(
            "random_weights", torch.zeros(self.num_features, self.input_size)
        )
        self.register_buffer("random_phases", torch.zeros(self.num_features))
        self.register_buffer("lengthscale", torch.ones(()))
        self.register_buffer("precision_cholesky", torch.eye(self.num_features))
        self.register_buffer(
            "posterior_weights", torch.zeros(self.num_features, self.output_size)
        )
        self.register_buffer(
            "posterior_second_weights",
            torch.zeros(self.num_features, self.output_size),
        )
        self.register_buffer("num_observations", torch.zeros((), dtype=torch.long))
        self.register_buffer("num_calibration", torch.zeros((), dtype=torch.long))
        self.register_buffer("variance_calibration", torch.ones(self.output_size))
        self.register_buffer("output_correlation", torch.eye(self.output_size))
        self.register_buffer("is_fitted", torch.zeros((), dtype=torch.bool))
        self._feature_gram: torch.Tensor | None = None
        self._feature_cross: torch.Tensor | None = None
        self._feature_second_cross: torch.Tensor | None = None

    @staticmethod
    def _safe_scale(values: torch.Tensor) -> torch.Tensor:
        return torch.clamp(values, min=1e-6)

    def set_normalization(self, input_mean, input_scale, target_mean, target_scale):
        self.input_mean.copy_(input_mean)
        self.input_scale.copy_(self._safe_scale(input_scale))
        self.target_mean.copy_(target_mean)
        self.target_scale.copy_(self._safe_scale(target_scale))

    def _standardize_input(self, values):
        return (values - self.input_mean) / self.input_scale

    def _standardize_target(self, values):
        return (values - self.target_mean) / self.target_scale

    def initialize_features(self, candidates: torch.Tensor, seed: int = 0) -> None:
        standardized = self._standardize_input(candidates)
        if self.configured_lengthscale > 0:
            selected_lengthscale = self.configured_lengthscale
        else:
            sample = standardized[: min(1024, standardized.shape[0])]
            distances = torch.pdist(sample)
            positive = distances[distances > 0]
            selected_lengthscale = float(positive.median()) if positive.numel() else 1.0
        self.lengthscale.fill_(max(selected_lengthscale, 1e-3))
        generator = torch.Generator(device=standardized.device)
        generator.manual_seed(seed)
        self.random_weights.copy_(
            torch.randn(
                self.random_weights.shape,
                generator=generator,
                device=standardized.device,
                dtype=standardized.dtype,
            )
            / self.lengthscale
        )
        self.random_phases.copy_(
            2
            * math.pi
            * torch.rand(
                self.random_phases.shape,
                generator=generator,
                device=standardized.device,
                dtype=standardized.dtype,
            )
        )
        self._feature_gram = torch.zeros(
            self.num_features,
            self.num_features,
            device=standardized.device,
            dtype=standardized.dtype,
        )
        self._feature_cross = torch.zeros(
            self.num_features,
            self.output_size,
            device=standardized.device,
            dtype=standardized.dtype,
        )
        self._feature_second_cross = torch.zeros_like(self._feature_cross)
        self.num_observations.zero_()
        self.num_calibration.zero_()
        self.variance_calibration.fill_(1.0)
        self.output_correlation.copy_(
            torch.eye(
                self.output_size,
                device=standardized.device,
                dtype=standardized.dtype,
            )
        )
        self.is_fitted.zero_()

    def _features(self, inputs: torch.Tensor) -> torch.Tensor:
        arguments = (
            self._standardize_input(inputs) @ self.random_weights.transpose(0, 1)
            + self.random_phases
        )
        return math.sqrt(2.0 / self.num_features) * torch.cos(arguments)

    def accumulate(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        if (
            self._feature_gram is None
            or self._feature_cross is None
            or self._feature_second_cross is None
        ):
            raise RuntimeError("Call initialize_features before accumulate.")
        features = self._features(inputs)
        targets = self._standardize_target(targets)
        self._feature_gram.add_(features.transpose(0, 1) @ features)
        self._feature_cross.add_(features.transpose(0, 1) @ targets)
        self._feature_second_cross.add_(features.transpose(0, 1) @ targets.square())
        self.num_observations.add_(inputs.shape[0])

    def finalize(self) -> None:
        if (
            self._feature_gram is None
            or self._feature_cross is None
            or self._feature_second_cross is None
        ):
            raise RuntimeError("No accumulated sufficient statistics to finalize.")
        identity = torch.eye(
            self.num_features,
            device=self._feature_gram.device,
            dtype=self._feature_gram.dtype,
        )
        precision = identity + self._feature_gram / self.noise_variance
        cholesky = torch.linalg.cholesky(precision + self.jitter * identity)
        self.precision_cholesky.copy_(cholesky)
        self.posterior_weights.copy_(
            torch.cholesky_solve(
                self._feature_cross / self.noise_variance, cholesky
            )
        )
        self.posterior_second_weights.copy_(
            torch.cholesky_solve(
                self._feature_second_cross / self.noise_variance, cholesky
            )
        )
        self.is_fitted.fill_(True)
        self._feature_gram = None
        self._feature_cross = None
        self._feature_second_cross = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not bool(self.is_fitted):
            raise RuntimeError("RFF GP has not been fitted.")
        mean = self._features(inputs) @ self.posterior_weights
        return mean * self.target_scale + self.target_mean

    def set_variance_calibration(self, scale: torch.Tensor, count: int) -> None:
        self.variance_calibration.copy_(torch.clamp(scale, min=0.05, max=100.0))
        self.num_calibration.fill_(count)

    def set_output_correlation(self, correlation: torch.Tensor) -> None:
        self.output_correlation.copy_(correlation)

    def predict_distribution(self, inputs: torch.Tensor) -> GaussianPrediction:
        features = self._features(inputs)
        standardized_mean = features @ self.posterior_weights
        mean = standardized_mean * self.target_scale + self.target_mean
        projected = torch.linalg.solve_triangular(
            self.precision_cholesky, features.transpose(0, 1), upper=False
        )
        latent = torch.clamp(projected.square().sum(dim=0), min=self.jitter)
        epistemic = latent.unsqueeze(-1) * self.target_scale.square()
        standardized_second = features @ self.posterior_second_weights
        conditional_variance = torch.clamp(
            standardized_second - standardized_mean.square(), min=0.0, max=100.0
        )
        aleatoric = conditional_variance * self.target_scale.square()
        predictive = self.variance_calibration.unsqueeze(0) * (
            latent.unsqueeze(-1) + self.noise_variance + conditional_variance
        ) * self.target_scale.square()
        return GaussianPrediction(mean, epistemic, aleatoric, predictive)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if "posterior_second_weights" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["posterior_second_weights"] = torch.zeros_like(
                self.posterior_second_weights
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def predictive_covariance(self, inputs: torch.Tensor) -> torch.Tensor:
        prediction = self.predict_distribution(inputs)
        std = torch.sqrt(prediction.predictive_variance)
        return (
            std.unsqueeze(-1)
            * self.output_correlation.unsqueeze(0)
            * std.unsqueeze(-2)
        )


def create_model(input_size: int, output_size: int, config):
    return RandomFourierGaussianProcess(
        input_size=input_size,
        output_size=output_size,
        num_features=config.num_features,
        lengthscale=config.lengthscale,
        noise_variance=config.noise_variance,
        jitter=config.jitter,
    )
