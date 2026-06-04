"""Inverse coefficient-map model builders."""

from __future__ import annotations


def create_inverse_model(
    model_name: str,
    *,
    input_size: int,
    output_size: int,
    hidden_sizes: list[int],
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
    raise ValueError(
        f"Unknown inverse model: {model_name}. Supported: linear_inverse, nonlinear."
    )
