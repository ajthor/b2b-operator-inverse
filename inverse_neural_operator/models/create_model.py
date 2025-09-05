"""
Model creation utility for centralized model management.

This module provides a unified interface for creating all supported models
with their appropriate configurations and optimizers.
"""
import torch
import os
from models.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)


def create_model(model_name, params, dataset_info, device, log_dir=None):
    """
    Create a model based on the model name and parameters.
    
    Args:
        model_name (str): Name of the model to create
        params: Parameters object containing model configuration
        dataset_info (dict): Dataset information from dataset.get_info()
        device (str): Device to create model on
        log_dir (str, optional): Directory containing function encoder checkpoints
        
    Returns:
        tuple: (model, optimizer) - Created model and optimizer (None for linear models)
        
    Raises:
        ValueError: If model_name is not supported
    """
    model = None
    optimizer = None
    
    # Helper function to load function encoders for b2b models
    def load_function_encoders():
        if log_dir is None:
            raise ValueError("log_dir is required for b2b models to load function encoders")
            
        # Load input function encoder
        input_function_encoder_params = torch.load(
            os.path.join(log_dir, "input_function_encoder_params.pth"), 
            weights_only=False
        )
        input_function_encoder = create_function_encoder(
            input_size=dataset_info["X_size"],
            hidden_sizes=input_function_encoder_params.hidden_sizes,
            output_size=dataset_info["u_size"],
            n_basis=input_function_encoder_params.n_basis,
            inner_product=(
                memory_efficient_inner_product
                if hasattr(params, 'dataset') and params.dataset in ["fwi"]
                else None
            ),
        )
        input_function_encoder.to(device)
        input_function_encoder = load_function_encoder(
            input_function_encoder,
            os.path.join(log_dir, "input_function_encoder.pth"),
            device=device,
        )
        
        # Load output function encoder
        output_function_encoder_params = torch.load(
            os.path.join(log_dir, "output_function_encoder_params.pth"), 
            weights_only=False
        )
        output_function_encoder = create_function_encoder(
            input_size=dataset_info["Y_size"],
            hidden_sizes=output_function_encoder_params.hidden_sizes,
            output_size=dataset_info["s_size"],
            n_basis=output_function_encoder_params.n_basis,
            inner_product=(
                memory_efficient_inner_product
                if hasattr(params, 'dataset') and params.dataset in ["fwi"]
                else None
            ),
        )
        output_function_encoder.to(device)
        output_function_encoder = load_function_encoder(
            output_function_encoder,
            os.path.join(log_dir, "output_function_encoder.pth"),
            device=device,
        )
        
        return input_function_encoder_params, output_function_encoder_params
    
    match model_name:
        case "b2b_linear":
            from models.b2b_operator_linear import create_model
            
            input_params, output_params = load_function_encoders()
            model = create_model(
                input_size=input_params.n_basis,
                output_size=output_params.n_basis,
            ).to(device)
            optimizer = None
            
        case "b2b_nonlinear":
            from models.b2b_operator_nonlinear import create_model
            
            input_params, output_params = load_function_encoders()
            model = create_model(
                input_size=input_params.n_basis,
                output_size=output_params.n_basis,
                hidden_sizes=params.hidden_sizes,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
            
        case "b2b_nonlinear_fwd":
            from models.b2b_operator_nonlinear_fwd import create_model
            
            input_params, output_params = load_function_encoders()
            model = create_model(
                input_size=input_params.n_basis,
                output_size=output_params.n_basis,
                hidden_sizes=params.hidden_sizes,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
            
        case "deeponet":
            from models.deeponet import create_model
            
            model = create_model(
                branch_input_size=dataset_info["Y_size"] * dataset_info["Y_len"],
                trunk_input_size=dataset_info["X_size"],
                output_size=dataset_info["u_size"],
                hidden_sizes=params.hidden_sizes,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
            
        case "variational_autoencoder":
            from models.variational_autoencoder import create_model
            
            input_params, output_params = load_function_encoders()
            model = create_model(
                alpha_size=input_params.n_basis,
                beta_size=output_params.n_basis,
                hidden_sizes=params.hidden_sizes,
                latent_size=output_params.n_basis,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
            
        case "invertible_network":
            from models.invertible_network import create_model
            
            input_params, output_params = load_function_encoders()
            model = create_model(
                input_size=input_params.n_basis,
                hidden_sizes=params.hidden_sizes,
                n_coupling_layers=2,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
            
        case "realnvp":
            from models.realnvp import create_model
            
            input_params, output_params = load_function_encoders()
            model = create_model(
                input_size=input_params.n_basis,
                hidden_sizes=params.hidden_sizes,
                n_coupling_layers=2,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
            
        case "ifno":
            from models.ifno import create_model
            
            input_params, output_params = load_function_encoders()
            model = create_model(
                input_size=input_params.n_basis,
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
            
        case _:
            raise ValueError(f"Unknown model: {model_name}")
    
    return model, optimizer