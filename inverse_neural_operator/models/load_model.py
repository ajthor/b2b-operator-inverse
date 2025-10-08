"""
Model loading utility for loading pre-trained models and function encoders.

This module provides functionality to load trained models from disk for evaluation and plotting.
"""

import os
import torch

from b2b.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)
from b2b.load_model import (
    load_function_encoder_params,
    load_function_encoders,
    load_forward_model,
)
from b2b.create_model import create_forward_model
from models.create_model import create_model
from utils.imports import import_model_functions


def load_forward_models(
    log_dir: str,
    dataset_info: dict,
    params: dict,
    device: str = "cpu",
):
    """
    Load pre-trained forward models and function encoders from disk.

    Args:
        log_dir (str): Directory containing saved model files
        dataset_info (dict): Dataset information from dataset.get_info()
        params: Parameters object containing model configuration
        device (str): Device to load models on

    Returns:
        tuple: (input_function_encoder, output_function_encoder, model, evaluate_function)
    """
    # Load function encoders using shared utility
    input_function_encoder, output_function_encoder = load_function_encoders(
        log_dir, dataset_info, params, device
    )

    # Load function encoder parameters to get sizes for model creation
    input_params, output_params = load_function_encoder_params(log_dir)

    # Create forward model using the create_forward_model utility
    model, _ = create_forward_model(
        params.model,
        params,
        input_params.n_basis,  # input size (alpha coefficients)
        output_params.n_basis,  # output size (beta coefficients)
        device,
    )

    # Load the trained forward model weights and get the appropriate load/evaluate functions
    forward_model_path = os.path.join(log_dir, f"forward_{params.model}.pth")

    # Get the appropriate load and evaluate functions for the forward model type
    load, evaluate = import_model_functions(params.model, "load", "evaluate")

    # Load the trained weights into the model
    model = load(model=model, path=forward_model_path, device=device)

    return input_function_encoder, output_function_encoder, model, evaluate


def load_models(
    log_dir: str,
    dataset_info: dict,
    params: dict,
    device: str = "cpu",
):
    """
    Load pre-trained models and function encoders from disk.

    Args:
        log_dir (str): Directory containing saved model files
        dataset_info (dict): Dataset information from dataset.get_info()
        params: Parameters object containing model configuration
        device (str): Device to load models on

    Returns:
        tuple: (input_function_encoder, output_function_encoder, model, evaluate_function)
    """
    # Load function encoders using shared utility
    input_function_encoder, output_function_encoder = load_function_encoders(
        log_dir, dataset_info, params, device
    )

    # For models that need encoder sizes, we need to get the parameter sizes
    input_size = None
    output_size = None
    if params.model in [
        "linear_inverse",
        "linear",
        "nonlinear",
        "variational_autoencoder",
        "inn_additive",
        "cinn_additive",
        "inn_affine",
        "cinn_affine",
        "cinn_additive_probabilistic",
        "cinn_affine_probabilistic",
        "ifno",
        "mixture_density_network",
    ]:
        input_params, output_params = load_function_encoder_params(log_dir)
        input_size = input_params.n_basis
        output_size = output_params.n_basis

    # Create model using the create_model utility (without optimizer since we're loading)
    model, _ = create_model(
        params.model, params, dataset_info, device, input_size, output_size
    )

    # Load the trained model weights and get the appropriate load/evaluate functions
    model_path = os.path.join(log_dir, "model.pth")

    # Get the appropriate load and evaluate functions for the model type
    load, evaluate = import_model_functions(params.model, "load", "evaluate")

    # Load the trained weights into the model
    model = load(model=model, path=model_path, device=device)

    return input_function_encoder, output_function_encoder, model, evaluate
