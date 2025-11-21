"""
Model creation utility for B2B forward models.

This module provides a unified interface for creating B2B forward models
(function encoders, deeponet, and forward operator models).
"""
import torch


def create_forward_model(model_name, params, input_size, output_size, device):
    """
    Create a forward model based on the model name and parameters.
    Forward models learn the mapping from input coefficients (alpha) to output coefficients (beta).

    Args:
        model_name (str): Name of the forward model to create
        params: Parameters object containing model configuration
        input_size (int): Size of input (alpha coefficients)
        output_size (int): Size of output (beta coefficients)
        device (str): Device to create model on

    Returns:
        tuple: (model, optimizer) - Created forward model and optimizer

    Raises:
        ValueError: If model_name is not supported for forward models
    """
    model = None
    optimizer = None

    match model_name:
        case "b2b_nonlinear":
            from inverse_neural_operator.b2b.b2b_nonlinear import create_model

            model = create_model(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=params.hidden_sizes,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        case "b2b_linear":
            from inverse_neural_operator.b2b.b2b_linear import create_model

            model = create_model(
                input_size=input_size,
                output_size=output_size,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        case _:
            raise ValueError(
                f"Unknown forward model: {model_name}. "
                f"Supported forward models: b2b_nonlinear, b2b_linear"
            )

    return model, optimizer
