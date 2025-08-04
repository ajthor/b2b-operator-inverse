"""
Plot the results of the Parametric Heat dataset for all models.
To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_parametric_heat

NOTE: The parametric heat dataset has challenging signal characteristics where the output 
function 's' contains mostly near-zero values, making the inverse problem ill-posed.
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
MODELS = ['b2b_linear', 'b2b_nonlinear', 'variational_autoencoder', 'invertible_network']


def get_evaluate_function(model_name):
    """Get the appropriate evaluate function for the model."""
    if model_name == "b2b_linear":
        from inverse_neural_operator.models.b2b_operator_linear import evaluate
    elif model_name == "b2b_nonlinear":
        from inverse_neural_operator.models.b2b_operator_nonlinear import evaluate
    elif model_name == "variational_autoencoder":
        from inverse_neural_operator.models.variational_autoencoder import evaluate
    elif model_name == "invertible_network":
        from inverse_neural_operator.models.invertible_network import evaluate
    elif model_name == "deeponet":
        from inverse_neural_operator.models.deeponet import evaluate
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return evaluate


def plot_parametric_heat_sample(model, evaluate_fn, input_function_encoder, output_function_encoder, 
                                sample, sample_idx, model_name, save_dir=None):
    """
    Plot a single parametric heat sample with 3 subplots: prediction, ground truth, and error.
    """
    model.eval()
    
    X, u_true, Y, s = sample
    
    # Ensure tensors are on correct device
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s = s.to(device)
    
    # Get model prediction
    with torch.no_grad():
        point = (X.unsqueeze(0), u_true.unsqueeze(0), Y.unsqueeze(0), s.unsqueeze(0))
        u_pred = evaluate_fn(model, point, input_function_encoder, output_function_encoder)
        u_pred = u_pred.squeeze(0)
    
    # Convert to numpy for plotting
    u_true_np = u_true.squeeze(-1).cpu().numpy()
    u_pred_np = u_pred.squeeze(-1).cpu().numpy()
    s_np = s.squeeze(-1).cpu().numpy()
    X_np = X.cpu().numpy()
    
    # For parametric heat, we know it's a 51x51 grid (2601 points)
    grid_size = 51
    n_points = len(u_true_np)
    
    if n_points != grid_size * grid_size:
        print(f"Warning: Expected {grid_size*grid_size} points but got {n_points}")
        grid_size = int(np.sqrt(n_points))
    
    # Reshape to 2D grids
    u_true_2d = u_true_np.reshape(grid_size, grid_size)
    u_pred_2d = u_pred_np.reshape(grid_size, grid_size)
    s_2d = s_np.reshape(grid_size, grid_size)
    
    # Calculate error
    error_2d = np.abs(u_pred_2d - u_true_2d)
    
    # Calculate diagnostics
    s_nonzero = np.sum(s_np != 0)
    s_max = np.max(np.abs(s_np))
    u_range = [u_true_np.min(), u_true_np.max()]
    pred_range = [u_pred_np.min(), u_pred_np.max()]
    mse_error = np.mean((u_pred_np - u_true_np)**2)
    
    # Create the plot with 4 subplots (adding s field visualization)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Physical domain extent (normalized coordinates)
    extent = [0, 1, 0, 1]
    
    # Plot 1: Predicted input function
    im1 = axes[0,0].imshow(u_pred_2d, cmap='viridis', extent=extent, origin='lower')
    axes[0,0].set_title(f'Predicted Input Function u(x,y)\nRange: [{pred_range[0]:.3f}, {pred_range[1]:.3f}]', fontsize=12)
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0], fraction=0.046)
    
    # Plot 2: Ground truth input function
    im2 = axes[0,1].imshow(u_true_2d, cmap='viridis', extent=extent, origin='lower')
    axes[0,1].set_title(f'Ground Truth Input Function u(x,y)\nRange: [{u_range[0]:.3f}, {u_range[1]:.3f}]', fontsize=12)
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1], fraction=0.046)
    
    # Plot 3: Output function s (what the model sees as input)
    im3 = axes[1,0].imshow(s_2d, cmap='plasma', extent=extent, origin='lower')
    axes[1,0].set_title(f'Output Function s(x,y) [Model Input]\nNon-zero: {s_nonzero}/{n_points}, Max: {s_max:.1e}', fontsize=12)
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1,0], fraction=0.046)
    
    # Plot 4: Absolute error
    im4 = axes[1,1].imshow(error_2d, cmap='Reds', extent=extent, origin='lower')
    axes[1,1].set_title(f'Absolute Error |u_pred - u_true|\nMSE: {mse_error:.3e}', fontsize=12)
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1,1], fraction=0.046)
    
    # Add overall title with diagnostics
    fig.suptitle(f'{model_name} - Sample {sample_idx}\n' + 
                f'Dataset Issue: s field has low signal (max={s_max:.1e}, {s_nonzero} non-zero points)',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}_sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        print(f"   📊 Signal quality: s_max={s_max:.1e}, u_range={u_range[0]:.3f}-{u_range[1]:.3f}, MSE={mse_error:.3e}")
    
    plt.close()


def plot_multiple_samples(model, evaluate_fn, input_function_encoder, output_function_encoder, 
                          dataset, model_name, n_samples=3, save_dir=None, split="test"):
    """Plot multiple samples from the dataset."""
    
    # Select samples
    if split == "test":
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    else:
        # For train split, select first n_samples
        indices = list(range(min(n_samples, len(dataset))))
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        plot_parametric_heat_sample(model, evaluate_fn, input_function_encoder, output_function_encoder, 
                                   sample, idx, model_name, save_dir)


def plot_model_results(model_name, log_dir, results_dir, test_dataset, train_dataset, dataset_info, n_test_samples=3, n_train_samples=2):
    """Plot results for a single model."""
    
    model_log_dir = os.path.join(log_dir, model_name, "seed_1")
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_log_dir, "params.pth")):
        print(f"⚠️  Model {model_name} not found, skipping...")
        return False
    
    print(f"📊 Plotting {model_name}...")
    
    # Load model parameters
    params = torch.load(os.path.join(model_log_dir, "params.pth"), weights_only=False)
    
    # Load models
    input_function_encoder, output_function_encoder, model = load_models(
        log_dir=model_log_dir,
        dataset_info=dataset_info,
        params=params,
        device=device,
    )
    
    # Get evaluation function
    evaluate_fn = get_evaluate_function(model_name)
    
    # Create model-specific results directory
    model_results_dir = os.path.join(results_dir, model_name)
    
    # Plot test samples
    test_results_dir = os.path.join(model_results_dir, "test")
    plot_multiple_samples(
        model=model,
        evaluate_fn=evaluate_fn,
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
        dataset=test_dataset,
        model_name=model_name,
        n_samples=n_test_samples,
        save_dir=test_results_dir,
        split="test"
    )
    
    # Plot train samples
    train_results_dir = os.path.join(model_results_dir, "train")
    plot_multiple_samples(
        model=model,
        evaluate_fn=evaluate_fn,
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
        dataset=train_dataset,
        model_name=model_name,
        n_samples=n_train_samples,
        save_dir=train_results_dir,
        split="train"
    )
    
    return True


# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot Parametric Heat results for all models.")
parser.add_argument("--log_dir", type=str, default="/workspaces/b2b-operator-inverse/logs", 
                   help="Base log directory")
parser.add_argument("--results_dir", type=str, default="results/parametric_heat_plots", 
                   help="Results directory for saving plots")
parser.add_argument("--n_test_samples", type=int, default=3, 
                   help="Number of test samples to plot per model")
parser.add_argument("--n_train_samples", type=int, default=2, 
                   help="Number of train samples to plot per model")
parser.add_argument("--seed", type=int, default=1, 
                   help="Random seed for reproducibility")

args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Hardcoded dataset
dataset = "parametric_heat"

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
    print("🔍 Expected models in:", log_dir)
    exit(1)

print(f"✅ Found {len(available_models)} trained models: {available_models}")

# Load dataset once (we'll reuse it for all models)
# Use the first available model to get params for dataset loading
first_model = available_models[0]
temp_log_dir = os.path.join(log_dir, first_model, f"seed_{args.seed}")
temp_params = torch.load(os.path.join(temp_log_dir, "params.pth"), weights_only=False)

test_dataset, dataset_info = load_dataset(temp_params, device)

# Load train dataset for train samples
from inverse_neural_operator.data.parametric_heat import load_data
train_dataset = load_data(temp_params, device=device, split="train")

print(f"📊 Dataset info: {len(test_dataset)} test samples, {len(train_dataset)} train samples")

# Process each available model
successful_models = []
for model_name in available_models:
    success = plot_model_results(
        model_name=model_name,
        log_dir=log_dir,
        results_dir=results_dir,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        dataset_info=dataset_info,
        n_test_samples=args.n_test_samples,
        n_train_samples=args.n_train_samples
    )
    if success:
        successful_models.append(model_name)

print(f"\n✅ Successfully plotted {len(successful_models)} models: {successful_models}")
print(f"📊 Generated {args.n_test_samples} test + {args.n_train_samples} train samples per model")
print(f"📁 Results saved in: {results_dir}")

# Show dataset analysis
print(f"\n📋 Parametric Heat Dataset Analysis:")
print(f"   - This is a challenging inverse problem where models predict input field u(x,y) from output field s(x,y)")
print(f"   - The s field contains mostly near-zero values, making the inverse problem ill-posed")
print(f"   - Poor model performance is expected due to insufficient signal in the s field")
print(f"   - The plots show both the low-signal input s and the predicted/true output u fields")
