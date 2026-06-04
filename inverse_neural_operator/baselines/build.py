"""Baseline model builders."""

from __future__ import annotations


def create_baseline_model(model_name: str, dataset_info, config):
    if model_name != "ifno":
        raise ValueError(f"Unknown baseline model: {model_name}. Supported: ifno.")

    from inverse_neural_operator.models.ifno import create_model

    ifno = config.baselines.ifno
    return create_model(
        input_size=None,
        hidden_sizes=[256, 256, 256],
        n_coupling_layers=2,
        modes1=ifno.modes,
        modes2=ifno.modes,
        width=ifno.width,
        beta=ifno.beta,
        n_layers=ifno.n_layers,
        padding=ifno.padding,
        vae_latent_dim=ifno.vae_latent_dim,
        input_spatial_dims=dataset_info["input_spatial_dims"],
        output_spatial_dims=dataset_info["output_spatial_dims"],
        input_function_channels=dataset_info["input_function_channels"],
        output_function_channels=dataset_info["output_function_channels"],
        coordinate_dim=dataset_info["coordinate_dim"],
        intermediate_dim=ifno.intermediate_dim,
    )

