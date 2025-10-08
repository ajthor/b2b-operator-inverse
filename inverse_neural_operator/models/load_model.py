"""
Model loading utility for loading pre-trained models and function encoders.

This module provides functionality to load trained models from disk for evaluation and plotting.
"""
import os
import torch

from models.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)
from models.create_model import create_model, create_forward_model


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
        os.path.join(log_dir, "input_function_encoder_params.pth"), 
        weights_only=False
    )
    
    # Load output function encoder parameters
    output_function_encoder_params = torch.load(
        os.path.join(log_dir, "output_function_encoder_params.pth"), 
        weights_only=False
    )
    
    return input_function_encoder_params, output_function_encoder_params


def load_forward_model(log_dir: str, device: str = "cpu"):
    """
    Load the pre-trained forward B2B operator from the shared directory.

    Args:
        log_dir (str): Directory containing the forward model
        device (str): Device to load the model on

    Returns:
        torch.nn.Module: Loaded forward B2B operator
    """
    import torch

    # Load function encoder parameters to get sizes
    input_encoder_params, output_encoder_params = load_function_encoder_params(log_dir)

    # Load saved forward model parameters from disk
    forward_params_path = os.path.join(log_dir, "params.pth")
    if not os.path.exists(forward_params_path):
        raise FileNotFoundError(f"Forward model params not found at {forward_params_path}")

    forward_params = torch.load(forward_params_path, weights_only=False)

    # Create forward model
    forward_model, _ = create_forward_model(
        model_name="b2b_nonlinear_fwd",
        params=forward_params,
        input_size=input_encoder_params.n_basis,
        output_size=output_encoder_params.n_basis,
        device=device,
    )

    # Load forward model weights
    forward_model_path = os.path.join(log_dir, "forward_b2b_nonlinear_fwd.pth")
    if os.path.exists(forward_model_path):
        forward_model.load_state_dict(torch.load(forward_model_path, map_location=device))
        forward_model.eval()
    else:
        raise FileNotFoundError(f"Forward model weights not found at {forward_model_path}")

    return forward_model


def load_function_encoders(log_dir: str, dataset_info: dict, params: dict, device: str = "cpu"):
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
            memory_efficient_inner_product
            if params.dataset in ["fwi"]
            else None
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
            memory_efficient_inner_product
            if params.dataset in ["fwi"]
            else None
        ),
    )
    output_function_encoder.to(device)
    output_function_encoder = load_function_encoder(
        output_function_encoder,
        os.path.join(log_dir, "output_function_encoder.pth"),
        device=device,
    )

    return input_function_encoder, output_function_encoder


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
        device
    )

    # Load the trained forward model weights and get the appropriate load/evaluate functions
    forward_model_path = os.path.join(log_dir, f"forward_{params.model}.pth")
    
    # Get the appropriate load and evaluate functions for the forward model type
    match params.model:
        case "b2b_nonlinear_fwd":
            from models.b2b_operator_nonlinear_fwd import load, evaluate
        case _:
            raise ValueError(f"Unknown forward model: {params.model}")

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

    # For b2b models, we need to get the parameter sizes
    input_size = None
    output_size = None
    if params.model.startswith("b2b") or params.model in ["variational_autoencoder", "inn_additive", "cinn_additive", "inn_affine", "cinn_affine", "ifno", "mixture_density_network", "b2b_linear_deterministic"]:
        input_params, output_params = load_function_encoder_params(log_dir)
        input_size = input_params.n_basis
        output_size = output_params.n_basis

    # Create model using the create_model utility (without optimizer since we're loading)
    model, _ = create_model(params.model, params, dataset_info, device, input_size, output_size)

    # Load the trained model weights and get the appropriate load/evaluate functions
    model_path = os.path.join(log_dir, "model.pth")
    
    # Get the appropriate load and evaluate functions for the model type
    match params.model:
        case "b2b_linear":
            from models.b2b_operator_linear import load, evaluate
        case "b2b_linear_deterministic":
            from models.b2b_operator_linear_deterministic import load, evaluate
        case "b2b_nonlinear":
            from models.b2b_operator_nonlinear import load, evaluate
        case "deeponet":
            from models.deeponet import load, evaluate
        case "variational_autoencoder":
            from models.variational_autoencoder import load, evaluate
        case "inn_affine":
            from models.inn_affine import load, evaluate
        case "inn_additive":
            from models.inn_additive import load, evaluate
        case "cinn_additive":
            from models.cinn_additive import load, evaluate
            
        case "cinn_affine":
            from models.cinn_affine import load, evaluate
        case "ifno":
            from models.ifno import load, evaluate
        case "mixture_density_network":
            from models.mixture_density_network import load, evaluate
        case _:
            raise ValueError(f"Unknown model: {params.model}")

    # Load the trained weights into the model
    model = load(model=model, path=model_path, device=device)

    return input_function_encoder, output_function_encoder, model, evaluate