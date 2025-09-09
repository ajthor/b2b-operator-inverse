"""
Model creation utility for centralized model management.

This module provides a unified interface for creating all supported models
with their appropriate configurations and optimizers.
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
        case "b2b_nonlinear_fwd":
            from models.b2b_operator_nonlinear_fwd import create_model
            
            model = create_model(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=params.hidden_sizes,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
            
        case _:
            raise ValueError(f"Unknown forward model: {model_name}. Supported forward models: b2b_nonlinear_fwd")
    
    return model, optimizer


def create_model(model_name, params, dataset_info, device, input_size=None, output_size=None):
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
        case "b2b_linear":
            from models.b2b_operator_linear import create_model
            
            if input_size is None or output_size is None:
                raise ValueError("input_size and output_size are required for b2b models")
            
            model = create_model(
                input_size=input_size,
                output_size=output_size,
            ).to(device)
            optimizer = None
            
        case "b2b_nonlinear":
            from models.b2b_operator_nonlinear import create_model
            
            if input_size is None or output_size is None:
                raise ValueError("input_size and output_size are required for b2b models")
            
            model = create_model(
                input_size=input_size,
                output_size=output_size,
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
            
            if input_size is None or output_size is None:
                raise ValueError("input_size and output_size are required for variational_autoencoder")
            
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
                raise ValueError("input_size and output_size are required for inn_additive")
            
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
                raise ValueError("input_size and output_size are required for inn_affine")
            
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
                raise ValueError("input_size and output_size are required for cinn_additive")
            
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
                raise ValueError("input_size and output_size are required for cinn_affine")
            
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
                raise ValueError("input_size and output_size are required for mixture_density_network")
            
            model = create_model(
                input_size=output_size,  # Takes beta coefficients as input
                output_size=input_size,  # Outputs alpha coefficients 
                hidden_sizes=params.hidden_sizes,
                n_components=getattr(params, 'n_components', 5),
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
            
        case _:
            raise ValueError(f"Unknown model: {model_name}")
    
    return model, optimizer