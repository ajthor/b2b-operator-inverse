"""
Model loading utility for B2B models (function encoders and forward models).

This module provides functionality to load trained B2B models from disk.
"""

import os
import torch

from b2b.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)
from b2b.create_model import create_forward_model


def load_function_encoder_params(log_dir: str):
    """
    Load function encoder parameters from disk.

    Args:
        log_dir (str): Directory containing saved function encoder parameter files

    Returns:
        tuple: (input_function_encoder_params, output_function_encoder_params)
    """
    # Load input function encoder parameters
    input_function_encoder_params = torch.load(
        os.path.join(log_dir, "input_function_encoder_params.pth"), weights_only=False
    )

    # Load output function encoder parameters
    output_function_encoder_params = torch.load(
        os.path.join(log_dir, "output_function_encoder_params.pth"), weights_only=False
    )

    return input_function_encoder_params, output_function_encoder_params


def load_function_encoders(
    log_dir: str, dataset_info: dict, params: dict, device: str = "cpu"
):
    """
    Load pre-trained function encoders from disk.

    Args:
        log_dir (str): Directory containing saved function encoder files
        dataset_info (dict): Dataset information from dataset.get_info()
        params: Parameters object containing dataset information
        device (str): Device to load encoders on

    Returns:
        tuple: (input_function_encoder, output_function_encoder)
    """
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
            memory_efficient_inner_product if params.dataset in ["fwi"] else None
        ),
    )
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
            memory_efficient_inner_product if params.dataset in ["fwi"] else None
        ),
    )
    output_function_encoder.to(device)
    output_function_encoder = load_function_encoder(
        output_function_encoder,
        os.path.join(log_dir, "output_function_encoder.pth"),
        device=device,
    )

    return input_function_encoder, output_function_encoder


def load_forward_model(log_dir: str, device: str = "cpu"):
    """
    Load the pre-trained forward B2B operator from the shared directory.

    Args:
        log_dir (str): Directory containing the forward model
        device (str): Device to load the model on

    Returns:
        torch.nn.Module: Loaded forward B2B operator
    """
    # Load function encoder parameters to get sizes
    input_encoder_params, output_encoder_params = load_function_encoder_params(log_dir)

    # Load saved forward model parameters from disk
    forward_params_path = os.path.join(log_dir, "params.pth")
    if not os.path.exists(forward_params_path):
        raise FileNotFoundError(
            f"Forward model params not found at {forward_params_path}"
        )

    forward_params = torch.load(forward_params_path, weights_only=False)

    # Create forward model
    forward_model, _ = create_forward_model(
        model_name=forward_params.model,
        params=forward_params,
        input_size=input_encoder_params.n_basis,
        output_size=output_encoder_params.n_basis,
        device=device,
    )

    # Load forward model weights
    forward_model_path = os.path.join(log_dir, f"forward_{forward_params.model}.pth")
    if os.path.exists(forward_model_path):
        forward_model.load_state_dict(
            torch.load(forward_model_path, map_location=device)
        )
        forward_model.eval()
    else:
        raise FileNotFoundError(
            f"Forward model weights not found at {forward_model_path}"
        )

    return forward_model
