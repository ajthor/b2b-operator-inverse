"""
Plot the results of the Burgers 1D dataset for all models.
To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_burgers
"""
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random

import torch

from inverse_neural_operator.models.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)

from inverse_neural_operator.plots.load_dataset import load_dataset
from inverse_neural_operator.plots.load_model import load_models

device = "cpu"

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Available models to plot
MODELS = ['b2b_linear', 'b2b_nonlinear', 'b2b_nonlinear_fwd', 'variational_autoencoder', 'invertible_network']


def get_evaluate_function(model_name):
    """Get the appropriate evaluate function for the model."""
    if model_name == "b2b_linear":
        from inverse_neural_operator.models.b2b_operator_linear import evaluate
    elif model_name == "b2b_nonlinear":
        from inverse_neural_operator.models.b2b_operator_nonlinear import evaluate
    elif model_name == "b2b_nonlinear_fwd":
        from inverse_neural_operator.models.b2b_operator_nonlinear_fwd import evaluate
    elif model_name == "variational_autoencoder":
        from inverse_neural_operator.models.variational_autoencoder import evaluate
    elif model_name == "invertible_network":
        from inverse_neural_operator.models.invertible_network import evaluate
    elif model_name == "deeponet":
        from inverse_neural_operator.models.deeponet import evaluate
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return evaluate


def plot_forward_model_sample(model, input_function_encoder, output_function_encoder,
                              sample, sample_idx, model_name, save_dir=None):
    """
    Plot a single sample for the forward model (alpha -> beta -> s prediction).
    """
    model.eval()
    
    X, u_true, Y, s_true = sample
    
    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_true = s_true.to(device)
    
    with torch.no_grad():
        # Add batch dimension for encoders
        X_batch = X.unsqueeze(0)
        u_batch = u_true.unsqueeze(0)
        Y_batch = Y.unsqueeze(0)
        
        # Compute alpha from true input
        alpha_result = input_function_encoder.compute_coefficients(X_batch, u_batch)
        alpha = alpha_result[0] if isinstance(alpha_result, tuple) else alpha_result
        
        # Forward pass through model
        beta_pred = model.forward(alpha)
        
        # Reconstruct predicted output
        s_pred = output_function_encoder(Y_batch, beta_pred)
        s_pred = s_pred.squeeze(0)  # Remove batch dimension
    
    # Convert to numpy for plotting
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    s_true_np = s_true.squeeze(-1).cpu().numpy()
    s_pred_np = s_pred.squeeze(-1).cpu().numpy()
    X_np = X.squeeze(-1).cpu().numpy()
    Y_np = Y.squeeze(-1).cpu().numpy()
    
    # Extract x coordinates for 1D plotting
    if X_np.ndim == 1:
        x_coords = X_np
        y_coords = Y_np
    else:
        x_coords = X_np[:, 0]
        y_coords = Y_np[:, 0]
    
    # Create plot with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Input function (what we start with)
    axes[0].plot(x_coords, u_true_np, 'b-', label='Input u(x)', linewidth=2)
    axes[0].set_title('Input Function u(x)', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Output function comparison
    axes[1].plot(y_coords, s_true_np, 'g-', label='True s(y)', linewidth=2)
    axes[1].plot(y_coords, s_pred_np, 'r--', label='Predicted s(y)', linewidth=2)
    axes[1].set_title('Forward Model: Output Prediction', fontsize=12)
    axes[1].set_xlabel('y')
    axes[1].set_ylabel('s(y)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}_forward_sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved forward model plot → {save_path}")
    
    plt.close()


def plot_burgers_sample(model, evaluate_fn, input_function_encoder, output_function_encoder, 
                        sample, sample_idx, model_name, save_dir=None,
                        num_inverse_samples: int = 3,
                        num_forward_samples: int = 2,
                        params: dict | None = None):
    """
    Plot a single Burgers sample with input, prediction, ground truth, and error.
    """
    model.eval()
    
    X, u_true, Y, s = sample
    
    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s = s.to(device)
    
    # Prepare batched point
    point = (X.unsqueeze(0), u_true.unsqueeze(0), Y.unsqueeze(0), s.unsqueeze(0))

    # Compute predictions
    inverse_preds = []
    forward_preds_s = []

    with torch.no_grad():
        if model_name == "variational_autoencoder":
            # For VAE, draw multiple z samples to get multiple inverse preds
            # Also optionally compute forward predictions via output encoder if available
            # Compute beta from (Y, s)
            beta_result = output_function_encoder.compute_coefficients(point[2], point[3])
            beta = beta_result[0] if isinstance(beta_result, tuple) else beta_result

            for i in range(max(1, int(num_inverse_samples))):
                z = model.sample_prior(1, device=X.device)
                alpha_pred = model.inverse(beta, z)
                u_pred_i = input_function_encoder(point[0], alpha_pred).squeeze(0)
                inverse_preds.append(u_pred_i)

                # For the first num_forward_samples, also compute forward s-pred via output encoder
                if i < max(0, int(num_forward_samples)):
                    beta_pred = None
                    # If a learned forward model exists (saved alongside VAE), try to load it lazily
                    # Otherwise, fall back to using output encoder directly with beta (identity)
                    try:
                        # Attempt to load nonlinear forward model if not already loaded
                        # Infer hidden sizes from params if provided
                        if params is not None and hasattr(params, "hidden_sizes"):
                            from inverse_neural_operator.models.b2b_operator_nonlinear import create_model as create_forward
                            fwd = create_forward(
                                input_size=alpha_pred.shape[-1],
                                hidden_sizes=params.hidden_sizes,
                                output_size=beta.shape[-1],
                            ).to(X.device)
                            fwd.load_state_dict(torch.load(os.path.join(params.log_dir, "forward_model.pth"), map_location=X.device))
                            fwd.eval()
                            beta_pred = fwd(alpha_pred)
                        else:
                            beta_pred = beta  # fallback
                    except Exception:
                        beta_pred = beta

                    s_pred_i = output_function_encoder(point[2], beta_pred).squeeze(0)
                    forward_preds_s.append(s_pred_i)
        else:
            # Non-VAE models: single prediction via evaluate_fn
            u_pred = evaluate_fn(model, point, input_function_encoder, output_function_encoder)
            u_pred = u_pred.squeeze(0)
            inverse_preds = [u_pred]
    
    # Convert to numpy for plotting
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    # Choose a representative inverse prediction for plots that need a single field (e.g., 2D heatmaps)
    u_pred_main = inverse_preds[0] if len(inverse_preds) > 0 else u_true
    u_pred_np = u_pred_main.squeeze(-1).cpu().numpy()
    s_np = s.squeeze(-1).cpu().numpy()
    X_np = X.squeeze(-1).cpu().numpy()
    
    # Try to determine grid size (assuming square grid)
    n_points = len(X_np)
    grid_size = int(np.sqrt(n_points))
    
    # If not a perfect square, use the data as-is for 1D case
    if grid_size * grid_size != n_points:
        # Create a simple 1D plot with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract x coordinates for 1D plotting
        if X_np.ndim == 1:
            x_coords = X_np  # Already 1D coordinates
        else:
            x_coords = X_np[:, 0]  # Use the first coordinate (x)
        
        # Plot 1: Observed output function (what we can measure)
        axes[0].plot(x_coords, s_np, 'g-', label='Observed Output Function')
        # Overlay forward-model predictions of s from inverse outputs
        if len(forward_preds_s) > 0:
            for j, s_pred_j in enumerate(forward_preds_s):
                s_pred_np = s_pred_j.squeeze(-1).cpu().numpy()
                if j == 0:
                    # First forward prediction: plot as dots to distinguish from dashed line
                    axes[0].plot(x_coords, s_pred_np, linestyle='None', marker='o', markersize=2, label='Forward Pred 1')
                else:
                    axes[0].plot(x_coords, s_pred_np, linestyle='--', label=f'Forward Pred {j+1}')
        axes[0].set_title('Observed Output Function s(x)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('s(x)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Input function comparison (what we want to predict)
        axes[1].plot(x_coords, u_true_np, 'b-', label='True Input', alpha=0.7)
        # Overlay multiple inverse predictions
        if len(inverse_preds) > 0:
            for i, u_pred_i in enumerate(inverse_preds):
                u_pred_np_i = u_pred_i.squeeze(-1).cpu().numpy()
                axes[1].plot(x_coords, u_pred_np_i, 'r--', alpha=0.7, label=(f'Inverse Pred {i+1}' if i == 0 else None))
        axes[1].set_title('Input Function: True vs Predicted u(x)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('u(x)')
        axes[1].legend()
        axes[1].grid(True)
        
    else:
        # 2D visualization
        # Reshape to 2D grids
        u_true_2d = u_true_np.reshape(grid_size, grid_size)
        u_pred_2d = u_pred_np.reshape(grid_size, grid_size)
        s_2d = s_np.reshape(grid_size, grid_size)
        
        # Calculate error
        error_2d = np.abs(u_pred_2d - u_true_2d)
        
        # Create the plot with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Physical domain extent (assuming normalized coordinates)
        extent = [0, 1, 0, 1]
        
        # Plot 1: Observed output function
        im1 = axes[0].imshow(s_2d, cmap='viridis', extent=extent, origin='lower')
        axes[0].set_title('Observed Output Function s(x,y)', fontsize=12)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # Plot 2: Absolute error for input function prediction
        im2 = axes[1].imshow(error_2d, cmap='Reds', extent=extent, origin='lower')
        axes[1].set_title('Input Prediction Error |u_pred - u_true|', fontsize=12)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    plt.tight_layout()
    
    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}_sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot → {save_path}")
    
    plt.close()


def plot_multiple_samples(model, evaluate_fn, input_function_encoder, output_function_encoder, 
                          test_dataset, model_name, n_samples=3, save_dir=None,
                          params=None):
    """Plot multiple random samples from the test set."""
    
    # Select random samples
    test_indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))
    
    for i, idx in enumerate(test_indices):
        sample = test_dataset[idx]
        plot_burgers_sample(
            model, evaluate_fn, input_function_encoder, output_function_encoder,
            sample, idx, model_name, save_dir,
            num_inverse_samples=3, num_forward_samples=2, params=params,
        )


def plot_model_results(model_name, log_dir, results_dir, test_dataset, dataset_info, n_samples=3):
    """Plot results for a single model."""
    
    model_log_dir = os.path.join(log_dir, model_name, "seed_1")
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_log_dir, "params.pth")):
        return False
    
    # Load model parameters
    params = torch.load(os.path.join(model_log_dir, "params.pth"), weights_only=False)
    
    # Load models
    input_function_encoder, output_function_encoder, model = load_models(
        log_dir=model_log_dir,
        dataset_info=dataset_info,
        params=params,
        device=device,
    )
    
    # Create model-specific results directory
    model_results_dir = os.path.join(results_dir, model_name)
    
    # Handle forward model separately
    if model_name == "b2b_nonlinear_fwd":
        # Select random samples for forward model plots
        test_indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))
        
        for i, idx in enumerate(test_indices):
            sample = test_dataset[idx]
            plot_forward_model_sample(
                model=model,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
                sample=sample,
                sample_idx=idx,
                model_name=model_name,
                save_dir=model_results_dir
            )
    else:
        # Get evaluation function for inverse models
        evaluate_fn = get_evaluate_function(model_name)
        
        # Plot results for inverse models
        plot_multiple_samples(
            model=model,
            evaluate_fn=evaluate_fn,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
            test_dataset=test_dataset,
            model_name=model_name,
            n_samples=n_samples,
            save_dir=model_results_dir,
            params=params,
        )
    
    return True


# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot Burgers 1D results for all models.")
parser.add_argument("--log_dir", type=str, default="/workspaces/b2b-operator-inverse/logs", 
                   help="Base log directory")
parser.add_argument("--results_dir", type=str, default="results/burgers_plots_VAE1", 
                   help="Results directory for saving plots")
parser.add_argument("--n_samples", type=int, default=3, 
                   help="Number of random samples to plot per model")
parser.add_argument("--seed", type=int, default=1, 
                   help="Random seed for reproducibility")

args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Hardcoded dataset
dataset = "burgers_1d"

# Construct paths
log_dir = os.path.join(args.log_dir, dataset)
results_dir = os.path.join(args.results_dir, dataset)

# Check which models are available
available_models = []
for model_name in MODELS:
    model_path = os.path.join(log_dir, model_name, f"seed_{args.seed}", "params.pth")
    if os.path.exists(model_path):
        available_models.append(model_name)

if not available_models:
    print("❌ No trained models found!")
    exit(1)

# Load dataset once (we'll reuse it for all models)
# Use the first available model to get params for dataset loading
first_model = available_models[0]
temp_log_dir = os.path.join(log_dir, first_model, f"seed_{args.seed}")
temp_params = torch.load(os.path.join(temp_log_dir, "params.pth"), weights_only=False)

test_dataset, dataset_info = load_dataset(temp_params, device)

# Process each available model
successful_models = []
for model_name in available_models:
    success = plot_model_results(
        model_name=model_name,
        log_dir=log_dir,
        results_dir=results_dir,
        test_dataset=test_dataset,
        dataset_info=dataset_info,
        n_samples=args.n_samples
    )
    if success:
        successful_models.append(model_name)

print(f"✅ Plotted {len(successful_models)} models, {len(successful_models) * args.n_samples} total plots")