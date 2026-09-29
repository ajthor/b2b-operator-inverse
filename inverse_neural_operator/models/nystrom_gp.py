"""Fixed-memory Nyström Gaussian-process regression for inverse coefficients."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class GaussianPrediction:
    mean: torch.Tensor
    epistemic_variance: torch.Tensor
    aleatoric_variance: torch.Tensor
    predictive_variance: torch.Tensor


class NystromGaussianProcess(torch.nn.Module):
    """Multi-output GP with a shared kernel and independent output marginals.

    The model maps output-function coefficients ``beta`` to input-function
    coefficients ``alpha``.  Its posterior is represented by fixed-size
    Nyström sufficient statistics, not by the accumulated training examples.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        num_inducing: int = 256,
        lengthscale: float = 0.0,
        noise_variance: float = 1e-2,
        jitter: float = 1e-5,
    ):
        super().__init__()
        if num_inducing < 1:
            raise ValueError("num_inducing must be positive.")
        if noise_variance <= 0:
            raise ValueError("noise_variance must be positive.")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_inducing = int(num_inducing)
        self.feature_count = self.num_inducing
        self.configured_lengthscale = float(lengthscale)
        self.noise_variance = float(noise_variance)
        self.jitter = float(jitter)

        self.register_buffer("input_mean", torch.zeros(self.input_size))
        self.register_buffer("input_scale", torch.ones(self.input_size))
        self.register_buffer("target_mean", torch.zeros(self.output_size))
        self.register_buffer("target_scale", torch.ones(self.output_size))
        self.register_buffer(
            "inducing_points", torch.zeros(self.num_inducing, self.input_size)
        )
        self.register_buffer("lengthscale", torch.ones(()))
        self.register_buffer(
            "kernel_cholesky", torch.eye(self.num_inducing)
        )
        self.register_buffer(
            "precision_cholesky", torch.eye(self.num_inducing)
        )
        self.register_buffer(
            "posterior_weights", torch.zeros(self.num_inducing, self.output_size)
        )
        self.register_buffer(
            "posterior_second_weights",
            torch.zeros(self.num_inducing, self.output_size),
        )
        self.register_buffer("num_observations", torch.zeros((), dtype=torch.long))
        self.register_buffer("num_calibration", torch.zeros((), dtype=torch.long))
        self.register_buffer("variance_calibration", torch.ones(self.output_size))
        self.register_buffer("output_correlation", torch.eye(self.output_size))
        self.register_buffer("is_fitted", torch.zeros((), dtype=torch.bool))

        self._feature_cross: torch.Tensor | None = None
        self._feature_second_cross: torch.Tensor | None = None
        self._feature_gram: torch.Tensor | None = None

    @staticmethod
    def _matern52(left: torch.Tensor, right: torch.Tensor, lengthscale: torch.Tensor):
        distance = torch.cdist(left / lengthscale, right / lengthscale)
        root_five_distance = math.sqrt(5.0) * distance
        return (
            1.0 + root_five_distance + root_five_distance.square() / 3.0
        ) * torch.exp(-root_five_distance)

    @staticmethod
    def _safe_scale(values: torch.Tensor) -> torch.Tensor:
        return torch.clamp(values, min=1e-6)

    def set_normalization(
        self,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> None:
        self.input_mean.copy_(input_mean)
        self.input_scale.copy_(self._safe_scale(input_scale))
        self.target_mean.copy_(target_mean)
        self.target_scale.copy_(self._safe_scale(target_scale))

    def _standardize_input(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.input_mean) / self.input_scale

    def _standardize_target(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.target_mean) / self.target_scale

    def initialize_inducing(self, candidates: torch.Tensor, seed: int = 0) -> None:
        """Choose inducing points with deterministic k-means++ seeding."""
        candidates = self._standardize_input(candidates)
        if candidates.shape[0] < self.num_inducing:
            raise ValueError(
                f"Need at least {self.num_inducing} inducing candidates; "
                f"got {candidates.shape[0]}."
            )
        generator = torch.Generator(device=candidates.device)
        generator.manual_seed(seed)
        selected = [
            int(
                torch.randint(
                    candidates.shape[0],
                    (1,),
                    generator=generator,
                    device=candidates.device,
                )
            )
        ]
        minimum_distance = torch.full(
            (candidates.shape[0],), float("inf"), device=candidates.device
        )
        for _ in range(1, self.num_inducing):
            latest = candidates[selected[-1] : selected[-1] + 1]
            distance = torch.sum((candidates - latest).square(), dim=-1)
            minimum_distance = torch.minimum(minimum_distance, distance)
            total = minimum_distance.sum()
            if not torch.isfinite(total) or total <= 0:
                remaining = torch.ones_like(minimum_distance)
                remaining[selected] = 0
                next_index = int(torch.multinomial(remaining, 1, generator=generator))
            else:
                next_index = int(
                    torch.multinomial(minimum_distance / total, 1, generator=generator)
                )
            selected.append(next_index)
        inducing = candidates[selected]
        self.inducing_points.copy_(inducing)

        if self.configured_lengthscale > 0:
            selected_lengthscale = self.configured_lengthscale
        else:
            distances = torch.pdist(inducing)
            positive = distances[distances > 0]
            selected_lengthscale = (
                float(positive.median()) if positive.numel() else 1.0
            )
        self.lengthscale.fill_(max(selected_lengthscale, 1e-3))
        kernel = self._matern52(inducing, inducing, self.lengthscale)
        identity = torch.eye(
            self.num_inducing, device=kernel.device, dtype=kernel.dtype
        )
        self.kernel_cholesky.copy_(
            torch.linalg.cholesky(kernel + self.jitter * identity)
        )
        self._feature_gram = torch.zeros_like(kernel)
        self._feature_cross = torch.zeros(
            self.num_inducing,
            self.output_size,
            device=kernel.device,
            dtype=kernel.dtype,
        )
        self._feature_second_cross = torch.zeros_like(self._feature_cross)
        self.num_observations.zero_()
        self.num_calibration.zero_()
        self.variance_calibration.fill_(1.0)
        self.output_correlation.copy_(
            torch.eye(
                self.output_size,
                device=self.output_correlation.device,
                dtype=self.output_correlation.dtype,
            )
        )
        self.is_fitted.zero_()

    def initialize_features(self, candidates: torch.Tensor, seed: int = 0) -> None:
        self.initialize_inducing(candidates, seed=seed)

    def _features(self, values: torch.Tensor) -> torch.Tensor:
        standardized = self._standardize_input(values)
        kernel_zx = self._matern52(
            self.inducing_points, standardized, self.lengthscale
        )
        return torch.linalg.solve_triangular(
            self.kernel_cholesky, kernel_zx, upper=False
        ).transpose(0, 1)

    def accumulate(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        if (
            self._feature_gram is None
            or self._feature_cross is None
            or self._feature_second_cross is None
        ):
            raise RuntimeError("Call initialize_inducing before accumulate.")
        features = self._features(inputs)
        standardized_targets = self._standardize_target(targets)
        self._feature_gram.add_(features.transpose(0, 1) @ features)
        self._feature_cross.add_(features.transpose(0, 1) @ standardized_targets)
        self._feature_second_cross.add_(
            features.transpose(0, 1) @ standardized_targets.square()
        )
        self.num_observations.add_(inputs.shape[0])

    def finalize(self) -> None:
        if (
            self._feature_gram is None
            or self._feature_cross is None
            or self._feature_second_cross is None
        ):
            raise RuntimeError("No accumulated sufficient statistics to finalize.")
        identity = torch.eye(
            self.num_inducing,
            device=self._feature_gram.device,
            dtype=self._feature_gram.dtype,
        )
        precision = identity + self._feature_gram / self.noise_variance
        cholesky = torch.linalg.cholesky(precision + self.jitter * identity)
        weights = torch.cholesky_solve(
            self._feature_cross / self.noise_variance, cholesky
        )
        self.precision_cholesky.copy_(cholesky)
        self.posterior_weights.copy_(weights)
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
            raise RuntimeError("Nyström GP has not been fitted.")
        standardized_mean = self._features(inputs) @ self.posterior_weights
        return standardized_mean * self.target_scale + self.target_mean

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
            self.precision_cholesky,
            features.transpose(0, 1),
            upper=False,
        )
        posterior_diagonal = projected.square().sum(dim=0)
        nystrom_diagonal = features.square().sum(dim=-1)
        residual_diagonal = torch.clamp(1.0 - nystrom_diagonal, min=0.0)
        latent_diagonal = torch.clamp(
            residual_diagonal + posterior_diagonal, min=self.jitter
        )
        epistemic = latent_diagonal.unsqueeze(-1) * self.target_scale.square()
        standardized_second = features @ self.posterior_second_weights
        conditional_variance = torch.clamp(
            standardized_second - standardized_mean.square(), min=0.0, max=100.0
        )
        aleatoric = conditional_variance * self.target_scale.square()
        predictive = self.variance_calibration.unsqueeze(0) * (
            latent_diagonal.unsqueeze(-1)
            + self.noise_variance
            + conditional_variance
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
        standard_deviation = torch.sqrt(prediction.predictive_variance)
        return (
            standard_deviation.unsqueeze(-1)
            * self.output_correlation.unsqueeze(0)
            * standard_deviation.unsqueeze(-2)
        )


def create_model(input_size: int, output_size: int, config):
    return NystromGaussianProcess(
        input_size=input_size,
        output_size=output_size,
        num_inducing=config.num_inducing,
        lengthscale=config.lengthscale,
        noise_variance=config.noise_variance,
        jitter=config.jitter,
    )
