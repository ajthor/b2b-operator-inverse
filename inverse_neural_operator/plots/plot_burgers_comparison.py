"""
Publication-quality comparison plotting script for Burgers 1D inverse problem results.

Creates a single comprehensive figure with all model predictions in a 2x4 grid layout:
- 6.5 inch width for journal publications
- 5-7pt sans serif fonts
- Professional styling and layout
- High-resolution output (300 DPI)
- Direct model comparison in grid format

To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_burgers_comparison
"""

import os
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random

import torch

from inverse_neural_operator.models.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)

from data.load_dataset import load_dataset
from models.load_model import load_models, load_forward_model

device = "cpu"

# Available models to plot
MODELS = [
    "b2b_linear", 
    "b2b_nonlinear",
    "variational_autoencoder",
    "invertible_network",
    "realnvp",
    "deeponet",
    "mixture_density_network",
    "b2b_nonlinear_fwd"  # Forward model for reference
]

# Model display names for publication
MODEL_NAMES = {
    "b2b_linear": "B2B Linear",
    "b2b_nonlinear": "B2B Nonlinear", 
    "variational_autoencoder": "VAE",
    "invertible_network": "INN",
    "realnvp": "RealNVP",
    "deeponet": "DeepONet",
    "mixture_density_network": "MDN",
    "b2b_nonlinear_fwd": "Forward Model"
}


def setup_publication_style():
    """Configure matplotlib for publication-quality output."""
    plt.style.use('default')
    
    # Font configuration
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    mpl.rcParams['font.size'] = 5
    mpl.rcParams['axes.labelsize'] = 5
    mpl.rcParams['axes.titlesize'] = 6
    mpl.rcParams['xtick.labelsize'] = 4
    mpl.rcParams['ytick.labelsize'] = 4
    mpl.rcParams['legend.fontsize'] = 4
    
    # Line and marker configuration
    mpl.rcParams['lines.linewidth'] = 0.8
    mpl.rcParams['lines.markersize'] = 2
    
    # Figure and subplot configuration
    mpl.rcParams['figure.figsize'] = [6.5, 3.5]  # 6.5" width, taller for 2x4 grid
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.05
    
    # Grid and axes
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.linewidth'] = 0.3
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['ytick.major.width'] = 0.5


def load_model_prediction(model_name, log_dir, sample, dataset_info, device):
    """Load a model and get its prediction for a single sample."""
    model_log_dir = os.path.join(log_dir, model_name, "seed_1")
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_log_dir, "params.pth")):
        return None, None, None
    
    try:
        # Load model parameters
        params = torch.load(os.path.join(model_log_dir, "params.pth"), weights_only=False)
        
        # Load models
        input_function_encoder, output_function_encoder, model, evaluate_fn = load_models(
            log_dir=model_log_dir,
            dataset_info=dataset_info,
            params=params,
            device=device,
        )
        
        # Load forward model for re-simulation
        forward_model = load_forward_model(log_dir=model_log_dir, device=device)
        
        X, u_true, Y, s_observed = sample
        
        # Ensure tensors are on correct device
        X = X.to(device)
        u_true = u_true.to(device)
        Y = Y.to(device)
        s_observed = s_observed.to(device)
        
        if model_name == "b2b_nonlinear_fwd":
            # Forward model: predict output from true input
            with torch.no_grad():
                X_batch = X.unsqueeze(0)
                u_batch = u_true.unsqueeze(0)
                Y_batch = Y.unsqueeze(0)
                
                alpha, _ = input_function_encoder.compute_coefficients(X_batch, u_batch)
                beta_pred = model.forward(alpha)
                s_pred = output_function_encoder(Y_batch, beta_pred)
                s_pred = s_pred.squeeze(0)
                
            return u_true, s_pred, None  # No re-simulation for forward model
        else:
            # Inverse model: predict input from observed output
            with torch.no_grad():
                point = (X.unsqueeze(0), u_true.unsqueeze(0), Y.unsqueeze(0), s_observed.unsqueeze(0))
                u_pred, _ = evaluate_fn(model, point, input_function_encoder, output_function_encoder)
                u_pred = u_pred.squeeze(0)
                
                # Re-simulate using forward model
                X_batch = X.unsqueeze(0)
                u_pred_batch = u_pred.unsqueeze(0)
                Y_batch = Y.unsqueeze(0)
                
                alpha, _ = input_function_encoder.compute_coefficients(X_batch, u_pred_batch)
                beta_pred = forward_model.forward(alpha)
                s_resim = output_function_encoder(Y_batch, beta_pred)
                s_resim = s_resim.squeeze(0)
                
            return u_pred, s_resim, np.mean((s_observed.cpu().numpy() - s_resim.cpu().numpy()) ** 2)
            
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None, None


def plot_burgers_comparison(log_dir, results_dir, test_dataset, dataset_info, sample_idx=0, seed=1):
    """Create comprehensive comparison plot for all models."""
    
    # Get a single sample for comparison
    sample = test_dataset[sample_idx]
    X, u_true, Y, s_observed = sample
    
    # Convert to numpy for plotting
    X_np = X.squeeze(-1).cpu().numpy()
    Y_np = Y.squeeze(-1).cpu().numpy() 
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    s_observed_np = s_observed.squeeze(-1).cpu().numpy()
    
    # Extract coordinates for 1D plotting
    if X_np.ndim == 1:
        x_coords = X_np
        y_coords = Y_np
    else:
        x_coords = X_np[:, 0]
        y_coords = Y_np[:, 0]
    
    # Create 2x4 subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.5))
    axes = axes.flatten()
    
    # Professional color scheme
    color_true = '#1f77b4'      # Blue
    color_observed = '#2E8B57'  # Sea green
    color_pred = '#d62728'      # Red
    color_resim = '#9467bd'     # Purple
    
    # Plot each model
    for idx, model_name in enumerate(MODELS):
        if idx >= 8:  # Only 8 subplots available
            break
            
        ax = axes[idx]
        display_name = MODEL_NAMES.get(model_name, model_name)
        
        # Load model prediction
        u_pred, s_pred, mse = load_model_prediction(
            model_name, log_dir, sample, dataset_info, device
        )
        
        if u_pred is None:
            # Model not available
            ax.text(0.5, 0.5, f'{display_name}\n(Not Available)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Convert predictions to numpy
        u_pred_np = u_pred.squeeze(-1).cpu().numpy()
        s_pred_np = s_pred.squeeze(-1).cpu().numpy()
        
        if model_name == "b2b_nonlinear_fwd":
            # Forward model: show input -> output
            ax.plot(y_coords, s_observed_np, color=color_observed, linewidth=0.8, 
                   label='True', alpha=0.7)
            ax.plot(y_coords, s_pred_np, color=color_pred, linewidth=0.8, 
                   linestyle='--', label='Pred', alpha=0.9)
            ax.set_xlabel('$y$', fontsize=5)
            ax.set_ylabel('$s(y)$', fontsize=5)
            mse_val = np.mean((s_observed_np - s_pred_np) ** 2)
            ax.set_title(f'{display_name}\nMSE: {mse_val:.2e}', fontsize=6)
        else:
            # Inverse model: show predicted input and re-simulated output
            # Create inset for input comparison
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            
            # Main plot: re-simulated vs observed output
            ax.plot(y_coords, s_observed_np, color=color_observed, linewidth=0.8, 
                   label='Obs', alpha=0.7)
            ax.plot(y_coords, s_pred_np, color=color_resim, linewidth=0.8, 
                   linestyle='--', label='Resim', alpha=0.9)
            ax.set_xlabel('$y$', fontsize=5)
            ax.set_ylabel('$s(y)$', fontsize=5)
            ax.set_title(f'{display_name}\nMSE: {mse:.2e}', fontsize=6)
            
            # Small inset for input comparison
            inset = inset_axes(ax, width="35%", height="35%", loc='upper right', 
                             bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes)
            inset.plot(x_coords, u_true_np, color=color_true, linewidth=0.6, alpha=0.7)
            inset.plot(x_coords, u_pred_np, color=color_pred, linewidth=0.6, 
                      linestyle='--', alpha=0.9)
            inset.set_xticks([])
            inset.set_yticks([])
            inset.text(0.5, 0.05, '$u(x)$', ha='center', transform=inset.transAxes, fontsize=4)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='lower right', frameon=False, fontsize=4)
        
        ax.tick_params(labelsize=4)
    
    # Hide any unused subplots
    for idx in range(len(MODELS), 8):
        axes[idx].axis('off')
    
    plt.tight_layout(pad=0.5, w_pad=0.3, h_pad=0.4)
    
    # Save plot
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"burgers_comparison_sample_{sample_idx}.pdf")
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    # Also save PNG for quick viewing
    png_path = os.path.join(results_dir, f"burgers_comparison_sample_{sample_idx}.png")
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    plt.close()
    
    return save_path


if __name__ == "__main__":
    # Setup publication style
    setup_publication_style()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create Burgers model comparison plot.")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/workspaces/b2b-operator-inverse/logs",
        help="Base log directory",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/burgers_comparison",
        help="Results directory for saving comparison plot",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Sample index to use for comparison",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Hardcoded dataset
    dataset = "burgers_1d"

    # Construct paths
    log_dir = os.path.join(args.log_dir, dataset)
    results_dir = args.results_dir

    # Load dataset using first available model's parameters
    available_models = []
    for model_name in MODELS:
        model_path = os.path.join(log_dir, model_name, f"seed_{args.seed}", "params.pth")
        if os.path.exists(model_path):
            available_models.append(model_name)
    
    if not available_models:
        print(f"ERROR: No trained models found in {log_dir}")
        exit(1)
    
    # Load dataset using first available model
    first_model = available_models[0]
    temp_log_dir = os.path.join(log_dir, first_model, f"seed_{args.seed}")
    temp_params = torch.load(os.path.join(temp_log_dir, "params.pth"), weights_only=False)

    test_dataset, dataset_info = load_dataset(
        temp_params.dataset, temp_params, device, split="test", return_info=True
    )

    # Create comparison plot
    save_path = plot_burgers_comparison(
        log_dir=log_dir,
        results_dir=results_dir,
        test_dataset=test_dataset,
        dataset_info=dataset_info,
        sample_idx=args.sample_idx,
        seed=args.seed,
    )

    print(f"SUCCESS: Created Burgers comparison plot → {save_path}")
    print(f"Found {len(available_models)} available models: {', '.join(available_models)}")