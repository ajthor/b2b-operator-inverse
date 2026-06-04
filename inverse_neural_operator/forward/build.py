"""Forward coefficient-map model builders."""

from __future__ import annotations


def create_forward_model(
    model_name: str,
    *,
    input_size: int,
    output_size: int,
    hidden_sizes: list[int],
):
    if model_name == "b2b_linear":
        from inverse_neural_operator.b2b.b2b_linear import create_model

        return create_model(input_size=input_size, output_size=output_size)
    if model_name == "b2b_nonlinear":
        from inverse_neural_operator.b2b.b2b_nonlinear import create_model

        return create_model(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
        )
    raise ValueError(
        f"Unknown forward model: {model_name}. Supported: b2b_linear, b2b_nonlinear."
    )

