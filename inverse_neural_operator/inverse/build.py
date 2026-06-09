"""Inverse coefficient-map model builders."""

from __future__ import annotations

import torch


SUPPORTED_INVERSE_MODELS = [
    "linear_inverse",
    "nonlinear",
    "conditional_realnvp",
    "vae",
    "mixture_density",
    "cinn_additive",
    "cinn_affine",
    "cinn_additive_probabilistic",
    "cinn_affine_probabilistic",
    "inn_additive",
    "inn_affine",
]


class DeterministicInverseAdapter(torch.nn.Module):
    """Expose beta -> alpha for inverse architectures with custom APIs."""

    def __init__(self, model_name: str, base_model: torch.nn.Module):
        super().__init__()
        self.model_name = model_name
        self.base_model = base_model

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        if self.model_name == "mixture_density":
            pi, _, mu, _ = self.base_model(beta)
            return (pi.unsqueeze(-1) * mu).sum(dim=1)
        if self.model_name == "vae":
            z = torch.zeros(
                beta.shape[0],
                self.base_model.latent_size,
                device=beta.device,
                dtype=beta.dtype,
            )
            return self.base_model.inverse(beta, z)
        if self.model_name in {
            "conditional_realnvp",
            "cinn_additive",
            "cinn_affine",
            "cinn_additive_probabilistic",
            "cinn_affine_probabilistic",
        }:
            z = torch.zeros_like(beta)
            return self.base_model.inverse(z, beta)
        if self.model_name == "inn_additive":
            return self.base_model.inverse(beta)
        if self.model_name == "inn_affine":
            input_size = self.base_model.coupling_layers[0].input_size
            z_size = input_size - self.base_model.output_size
            z = torch.zeros(
                beta.shape[0],
                z_size,
                device=beta.device,
                dtype=beta.dtype,
            )
            return self.base_model.inverse(beta=beta, z=z)
        raise ValueError(f"Unsupported adapter model: {self.model_name}")

    def sample_posterior(self, beta: torch.Tensor, n_samples: int):
        if hasattr(self.base_model, "sample_posterior"):
            return self.base_model.sample_posterior(beta, n_samples)
        sample = self.forward(beta)
        return sample.unsqueeze(0).repeat(n_samples, 1, 1)


def create_inverse_model(
    model_name: str,
    *,
    input_size: int,
    output_size: int,
    hidden_sizes: list[int],
    latent_size: int = 128,
    n_coupling_layers: int = 6,
    n_components: int = 5,
):
    if model_name == "linear_inverse":
        from inverse_neural_operator.models.linear_inverse import create_model

        model = create_model(input_size=input_size, output_size=output_size)
        model.linear.weight.requires_grad_(True)
        return model
    if model_name == "nonlinear":
        from inverse_neural_operator.models.nonlinear import create_model

        return create_model(
            input_size=output_size,
            hidden_sizes=hidden_sizes,
            output_size=input_size,
        )
    if model_name == "conditional_realnvp":
        from inverse_neural_operator.models.conditional_realnvp import create_model

        model = create_model(
            alpha_size=input_size,
            beta_size=output_size,
            hidden_sizes=hidden_sizes,
            latent_size=latent_size,
            n_coupling_layers=n_coupling_layers,
        )
        return DeterministicInverseAdapter(model_name, model)
    if model_name == "vae":
        from inverse_neural_operator.models.variational_autoencoder import create_model

        model = create_model(
            alpha_size=input_size,
            beta_size=output_size,
            hidden_sizes=hidden_sizes,
            latent_size=latent_size,
        )
        return DeterministicInverseAdapter(model_name, model)
    if model_name == "mixture_density":
        from inverse_neural_operator.models.mixture_density_network import create_model

        model = create_model(
            input_size=output_size,
            output_size=input_size,
            hidden_sizes=hidden_sizes,
            n_components=n_components,
        )
        return DeterministicInverseAdapter(model_name, model)
    if model_name == "cinn_additive":
        from inverse_neural_operator.models.cinn_additive import create_model

        model = create_model(
            input_size=input_size,
            condition_size=output_size,
            hidden_sizes=hidden_sizes,
            n_coupling_layers=n_coupling_layers,
        )
        return DeterministicInverseAdapter(model_name, model)
    if model_name == "cinn_affine":
        from inverse_neural_operator.models.cinn_affine import create_model

        model = create_model(
            input_size=input_size,
            condition_size=output_size,
            hidden_sizes=hidden_sizes,
            n_coupling_layers=n_coupling_layers,
        )
        return DeterministicInverseAdapter(model_name, model)
    if model_name == "cinn_additive_probabilistic":
        from inverse_neural_operator.models.cinn_additive_probabilistic import (
            create_model,
        )

        model = create_model(
            input_size=input_size,
            condition_size=output_size,
            hidden_sizes=hidden_sizes,
            n_coupling_layers=n_coupling_layers,
        )
        return DeterministicInverseAdapter(model_name, model)
    if model_name == "cinn_affine_probabilistic":
        from inverse_neural_operator.models.cinn_affine_probabilistic import create_model

        model = create_model(
            input_size=input_size,
            condition_size=output_size,
            hidden_sizes=hidden_sizes,
            n_coupling_layers=n_coupling_layers,
        )
        return DeterministicInverseAdapter(model_name, model)
    if model_name == "inn_additive":
        from inverse_neural_operator.models.inn_additive import create_model

        model = create_model(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            n_coupling_layers=n_coupling_layers,
        )
        return DeterministicInverseAdapter(model_name, model)
    if model_name == "inn_affine":
        from inverse_neural_operator.models.inn_affine import create_model

        model = create_model(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            n_coupling_layers=n_coupling_layers,
        )
        return DeterministicInverseAdapter(model_name, model)
    raise ValueError(
        f"Unknown inverse model: {model_name}. Supported: "
        f"{', '.join(SUPPORTED_INVERSE_MODELS)}."
    )
