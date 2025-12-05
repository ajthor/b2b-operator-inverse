"""
Model loading utility for B2B models (function encoders and forward models).

This module provides functionality to load trained B2B models from disk.
"""

import os
import torch
from safetensors.torch import load_file

from inverse_neural_operator.b2b.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)
from inverse_neural_operator.b2b.create_model import create_forward_model


def load_function_encoder_params(model_dir: str):
    """
    Load function encoder parameters from disk.

    Args:
        model_dir (str): Directory containing saved function encoder parameter files
                        (models/dataset/shared/seed_X/ or models/dataset/model/seed_X/)

    Returns:
        tuple: (input_function_encoder_params, output_function_encoder_params)
    """
    # Load input function encoder parameters
    input_function_encoder_params = torch.load(
        os.path.join(model_dir, "input_function_encoder_params.pth"), weights_only=False
    )

    # Load output function encoder parameters
    output_function_encoder_params = torch.load(
        os.path.join(model_dir, "output_function_encoder_params.pth"), weights_only=False
    )

    return input_function_encoder_params, output_function_encoder_params


def load_function_encoders(
    model_dir: str, dataset_info: dict, params: dict, device: str = "cpu"
):
    """
    Load pre-trained function encoders from disk.

    Args:
        model_dir (str): Directory containing saved function encoder files
                        (models/dataset/shared/seed_X/ or models/dataset/model/seed_X/)
        dataset_info (dict): Dataset information from dataset.get_info()
        params: Parameters object containing dataset information
        device (str): Device to load encoders on

    Returns:
        tuple: (input_function_encoder, output_function_encoder)
    """
    # Load the input function encoder
    input_function_encoder_params = torch.load(
        os.path.join(model_dir, "input_function_encoder_params.pth"), weights_only=False
    )
    input_function_encoder = create_function_encoder(
        input_size=dataset_info["X_size"],
        hidden_sizes=input_function_encoder_params.hidden_sizes,
        output_size=dataset_info["u_size"],
        n_basis=input_function_encoder_params.n_basis,
        inner_product=(
            memory_efficient_inner_product if params.dataset in ["fwi"] else None
        ),
    )
    input_function_encoder.to(device)
    input_function_encoder = load_function_encoder(
        input_function_encoder,
        os.path.join(model_dir, "input_function_encoder.safetensors"),
        device=device,
    )

    # Load the output function encoder
    output_function_encoder_params = torch.load(
        os.path.join(model_dir, "output_function_encoder_params.pth"), weights_only=False
    )
    output_function_encoder = create_function_encoder(
        input_size=dataset_info["Y_size"],
        hidden_sizes=output_function_encoder_params.hidden_sizes,
        output_size=dataset_info["s_size"],
        n_basis=output_function_encoder_params.n_basis,
        inner_product=(
            memory_efficient_inner_product if params.dataset in ["fwi"] else None
        ),
    )
    output_function_encoder.to(device)
    output_function_encoder = load_function_encoder(
        output_function_encoder,
        os.path.join(model_dir, "output_function_encoder.safetensors"),
        device=device,
    )

    return input_function_encoder, output_function_encoder


def load_forward_model(model_dir: str, forward_model_name: str, device: str = "cpu"):
    """
    Load the pre-trained forward B2B operator from the specified directory.

    Args:
        model_dir (str): Directory containing the forward model checkpoint and params
                        (models/dataset/shared/seed_X/)
        forward_model_name (str): Forward model name (e.g., 'b2b_nonlinear', 'b2b_linear')
        device (str): Device to load the model on

    Returns:
        torch.nn.Module: Loaded forward B2B operator

    Raises:
        FileNotFoundError: If the forward model checkpoint is not found
    """
    # Load function encoder parameters to get sizes
    input_encoder_params, output_encoder_params = load_function_encoder_params(model_dir)

    # Load params for model configuration
    params_path = os.path.join(model_dir, "params.pth")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Model parameters not found at {params_path}")

    params = torch.load(params_path, weights_only=False)
    # Override the model name with the forward model name
    params.model = forward_model_name

    # The forward model checkpoint is named forward_{model_name}.safetensors
    forward_model_path = os.path.join(model_dir, f"forward_{forward_model_name}.safetensors")
    if not os.path.exists(forward_model_path):
        raise FileNotFoundError(
            f"Forward model checkpoint not found at {forward_model_path}"
        )

    # Create forward model
    forward_model, _ = create_forward_model(
        model_name=forward_model_name,
        params=params,
        input_size=input_encoder_params.n_basis,
        output_size=output_encoder_params.n_basis,
        device=device,
    )

    # Load forward model weights (safetensors format)
    state_dict = load_file(forward_model_path, device=str(device))
    forward_model.load_state_dict(state_dict)
    forward_model.eval()

    return forward_model
