"""
Model creation utility for inverse models.

This module provides a unified interface for creating all supported inverse models
with their appropriate configurations and optimizers.
"""

import torch


def create_model(
    model_name, params, dataset_info, device, input_size=None, output_size=None
):
    """
    Create a model based on the model name and parameters.

    Args:
        model_name (str): Name of the model to create
        params: Parameters object containing model configuration
        dataset_info (dict): Dataset information from dataset.get_info()
        device (str): Device to create model on
        input_size (int, optional): Size of input for b2b models (alpha coefficients)
        output_size (int, optional): Size of output for b2b models (beta coefficients)

    Returns:
        tuple: (model, optimizer) - Created model and optimizer (None for linear models)

    Raises:
        ValueError: If model_name is not supported
    """
    model = None
    optimizer = None

    match model_name:
        case "linear_inverse":
            from models.linear_inverse import create_model

            if input_size is None or output_size is None:
                raise ValueError(
                    "input_size and output_size are required for linear_inverse model"
                )

            model = create_model(
                input_size=input_size,
                output_size=output_size,
            ).to(device)
            optimizer = None

        case "linear":
            from models.linear import create_model

            if input_size is None or output_size is None:
                raise ValueError(
                    "input_size and output_size are required for linear model"
                )

            model = create_model(
                input_size=input_size,
                output_size=output_size,
            ).to(device)
            optimizer = None

        case "nonlinear":
            from models.nonlinear import create_model

            if input_size is None or output_size is None:
                raise ValueError(
                    "input_size and output_size are required for nonlinear model"
                )

            model = create_model(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=params.hidden_sizes,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        case "variational_autoencoder":
            from models.variational_autoencoder import create_model

            if input_size is None or output_size is None:
                raise ValueError(
                    "input_size and output_size are required for variational_autoencoder"
                )

            model = create_model(
                alpha_size=input_size,
                beta_size=output_size,
                hidden_sizes=params.hidden_sizes,
                latent_size=output_size,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        case "inn_additive":
            from models.inn_additive import create_model

            if input_size is None or output_size is None:
                raise ValueError(
                    "input_size and output_size are required for inn_additive"
                )

            model = create_model(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=params.hidden_sizes,
                n_coupling_layers=2,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        case "inn_affine":
            from models.inn_affine import create_model

            if input_size is None or output_size is None:
                raise ValueError(
                    "input_size and output_size are required for inn_affine"
                )

            model = create_model(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=params.hidden_sizes,
                n_coupling_layers=2,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        case "cinn_additive":
            from models.cinn_additive import create_model

            if input_size is None or output_size is None:
                raise ValueError(
                    "input_size and output_size are required for cinn_additive"
                )

            model = create_model(
                input_size=input_size,
                condition_size=output_size,  # Condition on output coefficients (beta)
                hidden_sizes=params.hidden_sizes,
                n_coupling_layers=2,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        case "cinn_affine":
            from models.cinn_affine import create_model

            if input_size is None or output_size is None:
                raise ValueError(
                    "input_size and output_size are required for cinn_affine"
                )

            model = create_model(
                input_size=input_size,
                condition_size=output_size,  # Condition on output coefficients (beta)
                hidden_sizes=params.hidden_sizes,
                n_coupling_layers=2,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        case "cinn_additive_probabilistic":
            from models.cinn_additive_probabilistic import create_model

            if input_size is None or output_size is None:
                raise ValueError(
                    "input_size and output_size are required for cinn_additive_probabilistic"
                )

            model = create_model(
                input_size=input_size,
                condition_size=output_size,  # Condition on output coefficients (beta)
                hidden_sizes=params.hidden_sizes,
                n_coupling_layers=2,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        case "cinn_affine_probabilistic":
            from models.cinn_affine_probabilistic import create_model

            if input_size is None or output_size is None:
                raise ValueError(
                    "input_size and output_size are required for cinn_affine_probabilistic"
                )

            model = create_model(
                input_size=input_size,
                condition_size=output_size,  # Condition on output coefficients (beta)
                hidden_sizes=params.hidden_sizes,
                n_coupling_layers=2,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        case "ifno":
            from models.ifno import create_model

            if input_size is None:
                raise ValueError("input_size is required for ifno")

            model = create_model(
                input_size=input_size,
                hidden_sizes=params.hidden_sizes,
                n_coupling_layers=2,
                modes1=16,
                modes2=16,
                width=64,
                beta=2.0,
                n_layers=4,
                padding=20,
                vae_latent_dim=24,
                intermediate_dim=64,
                # iFNO-specific parameters from dataset info
                input_spatial_dims=dataset_info["input_spatial_dims"],
                output_spatial_dims=dataset_info["output_spatial_dims"],
                input_function_channels=dataset_info["input_function_channels"],
                output_function_channels=dataset_info["output_function_channels"],
                coordinate_dim=dataset_info["coordinate_dim"],
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        case "mixture_density_network":
            from models.mixture_density_network import create_model

            if input_size is None or output_size is None:
                raise ValueError(
                    "input_size and output_size are required for mixture_density_network"
                )

            model = create_model(
                input_size=output_size,  # Takes beta coefficients as input
                output_size=input_size,  # Outputs alpha coefficients
                hidden_sizes=params.hidden_sizes,
                n_components=getattr(params, "n_components", 5),
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        case "conditional_realnvp":
            from models.conditional_realnvp import create_model

            if input_size is None or output_size is None:
                raise ValueError(
                    "input_size and output_size are required for conditional_realnvp"
                )

            model = create_model(
                alpha_size=input_size,
                beta_size=output_size,
                hidden_sizes=params.hidden_sizes,
                latent_size=input_size,
                n_coupling_layers=getattr(params, "n_coupling_layers", 6),
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        case _:
            raise ValueError(f"Unknown model: {model_name}")

    return model, optimizer
