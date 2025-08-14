import os

import torch

from inverse_neural_operator.models.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)


def load_models(
    log_dir: str,
    dataset_info: dict,
    params: dict,
    device: str = "cpu",
):
    # Load the input function encoder

    input_function_encoder_params = torch.load(
        os.path.join(log_dir, "input_function_encoder_params.pth"), weights_only=False
    )
    input_function_encoder = create_function_encoder(
        input_size=dataset_info["X_size"],
        hidden_sizes=input_function_encoder_params.hidden_sizes,
        output_size=dataset_info["u_size"],
        n_basis=input_function_encoder_params.n_basis,
        inner_product=(
            memory_efficient_inner_product
            if params.dataset in ["fwi_flat", "fwi_curve"]
            else None
        ),
    )
    # input_function_encoder = torch.compile(input_function_encoder)
    input_function_encoder.to(device)
    input_function_encoder = load_function_encoder(
        input_function_encoder,
        os.path.join(log_dir, "input_function_encoder.pth"),
        device=device,
    )

    # Load the output function encoder

    output_function_encoder_params = torch.load(
        os.path.join(log_dir, "output_function_encoder_params.pth"), weights_only=False
    )
    output_function_encoder = create_function_encoder(
        input_size=dataset_info["Y_size"],
        hidden_sizes=output_function_encoder_params.hidden_sizes,
        output_size=dataset_info["s_size"],
        n_basis=output_function_encoder_params.n_basis,
        inner_product=(
            memory_efficient_inner_product
            if params.dataset in ["fwi_flat", "fwi_curve"]
            else None
        ),
    )
    # output_function_encoder = torch.compile(output_function_encoder)
    output_function_encoder.to(device)
    output_function_encoder = load_function_encoder(
        output_function_encoder,
        os.path.join(log_dir, "output_function_encoder.pth"),
        device=device,
    )

    # Load model

    match params.model:
        case "b2b_linear":
            from inverse_neural_operator.models.b2b_operator_linear import (
                create_model,
                load,
                evaluate,
            )

            model = create_model(
                input_size=input_function_encoder_params.n_basis,
                output_size=output_function_encoder_params.n_basis,
            ).to(device)
            model = load(
                model=model, path=os.path.join(log_dir, "model.pth"), device=device
            )

        case "b2b_nonlinear":
            from inverse_neural_operator.models.b2b_operator_nonlinear import (
                create_model,
                load,
                evaluate,
            )

            model = create_model(
                input_size=input_function_encoder_params.n_basis,
                output_size=output_function_encoder_params.n_basis,
                hidden_sizes=params.hidden_sizes,
            ).to(device)
            model = load(
                model=model, path=os.path.join(log_dir, "model.pth"), device=device
            )

        case "deeponet":
            from inverse_neural_operator.models.deeponet import (
                create_model,
                load,
                evaluate,
            )

            model = create_model(
                branch_input_size=dataset_info["Y_size"] *
                dataset_info["Y_len"],
                trunk_input_size=dataset_info["X_size"],
                output_size=dataset_info["u_size"],
                hidden_sizes=params.hidden_sizes,
            ).to(device)
            model = load(
                model=model, path=os.path.join(log_dir, "model.pth"), device=device
            )

        case "variational_autoencoder":
            from inverse_neural_operator.models.variational_autoencoder import (
                create_model,
                load,
                evaluate,
            )

            model = create_model(
                alpha_size=input_function_encoder_params.n_basis,
                beta_size=output_function_encoder_params.n_basis,
                hidden_sizes=params.hidden_sizes,
                latent_size=output_function_encoder_params.n_basis,
            ).to(device)
            model = load(
                model=model, path=os.path.join(log_dir, "model.pth"), device=device
            )

        case "realnvp":
            from models.realnvp import (
                create_model,
                load,
                evaluate,
            )

            model = create_model(
                input_size=input_function_encoder_params.n_basis,
                hidden_sizes=params.hidden_sizes,
                n_coupling_layers=4,
            ).to(device)
            model = load(
                model=model, path=os.path.join(log_dir, "model.pth"), device=device
            )

        case "invertible_network":
            from inverse_neural_operator.models.invertible_network import (
                create_model,
                load,
                evaluate,
            )

            model = create_model(
                input_size=input_function_encoder_params.n_basis,
                hidden_sizes=params.hidden_sizes,
                n_coupling_layers=2,
            ).to(device)
            model = load(
                model=model, path=os.path.join(log_dir, "model.pth"), device=device
            )

        case "ifno":
            from inverse_neural_operator.models.ifno import (
                create_model,
                load,
                evaluate,
            )

            # Load dataset info for iFNO configuration
<<<<<<< Updated upstream
            from inverse_neural_operator.data import get_data_loaders
            dataset_info = get_data_loaders(params.dataset, 1, 1)[2]  # Get dataset info
=======
            from data import get_data_loaders
            dataset_info = get_data_loaders(params.dataset, 1, 1)[
                2]  # Get dataset info
>>>>>>> Stashed changes

            model = create_model(
                input_size=input_function_encoder_params.n_basis,
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
                # iFNO-specific parameters from dataset info (auto-computed in process_data.py)
                input_spatial_dims=dataset_info["input_spatial_dims"],
                output_spatial_dims=dataset_info["output_spatial_dims"],
                input_function_channels=dataset_info["input_function_channels"],
                output_function_channels=dataset_info["output_function_channels"],
                coordinate_dim=dataset_info["coordinate_dim"],
            ).to(device)
            model = load(
                model=model, path=os.path.join(log_dir, "model.pth"), device=device
            )

        case _:
            raise ValueError(f"Unknown model: {params.model}")

    return input_function_encoder, output_function_encoder, model
