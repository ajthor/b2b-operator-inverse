"""
Model loading utility for loading pre-trained models and function encoders.

This module provides functionality to load trained models from disk for evaluation and plotting.
"""

import os
import torch

from inverse_neural_operator.b2b.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)
from inverse_neural_operator.b2b.load_model import (
    load_function_encoder_params,
    load_function_encoders,
)
from inverse_neural_operator.models.create_model import create_model
from inverse_neural_operator.utils.imports import import_model_functions

# Note: Forward model loading (load_forward_model) should be imported directly from b2b.load_model
# This keeps the responsibility clear - all B2B-specific loading logic lives in b2b.load_model


def load_models(
    base_dir: str,
    dataset: str,
    model_name: str,
    seed: int,
    device: str = "cpu",
):
    """
    Load pre-trained models and function encoders from disk.

    Args:
        base_dir (str): Base directory for models (e.g., ./results or /store/...)
        dataset (str): Dataset name (e.g., 'burgers_1d', 'darcy_1d')
        model_name (str): Model name (e.g., 'nonlinear', 'cinn_affine')
        seed (int): Random seed used for training
        device (str): Device to load models on

    Returns:
        tuple: (input_function_encoder, output_function_encoder, model, evaluate_function)
    """
    # Construct model_dir where models and params are saved
    model_dir = os.path.join(base_dir, "models", dataset, model_name, f"seed_{seed}")
    shared_dir = os.path.join(base_dir, "models", dataset, "shared", f"seed_{seed}")

    # Load params from model_dir
    params_path = os.path.join(model_dir, "params.pth")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Model parameters not found at {params_path}")

    params = torch.load(params_path, weights_only=False)

    # Load dataset to get dataset_info
    from inverse_neural_operator.data.load_dataset import load_dataset
    dataset_obj = load_dataset(dataset, params, device, split="test")
    dataset_info = dataset_obj.get_info()

    # Load function encoders from shared directory
    input_function_encoder, output_function_encoder = load_function_encoders(
        shared_dir, dataset_info, params, device
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
        "conditional_realnvp",
    ]:
        input_params, output_params = load_function_encoder_params(shared_dir)
        input_size = input_params.n_basis
        output_size = output_params.n_basis

    # Create model using the create_model utility (without optimizer since we're loading)
    model, _ = create_model(
        params.model, params, dataset_info, device, input_size, output_size
    )

    # Load the trained model weights
    model_path = os.path.join(model_dir, "model.safetensors")

    # Get the appropriate load and evaluate functions for the model type
    load, evaluate = import_model_functions(params.model, "load", "evaluate")

    # Load the trained weights into the model
    model = load(model=model, path=model_path, device=device)

    return input_function_encoder, output_function_encoder, model, evaluate
