import os
import sys
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to Python path for direct execution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

from inverse_neural_operator.models.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)

from inverse_neural_operator.plots.load_dataset import load_dataset
from inverse_neural_operator.plots.load_model import load_models

from inverse_neural_operator.models.model_evaluation import evaluate_random, find_best_case, find_worst_case

device = "cpu"

torch.manual_seed(1)


# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot results for Chladni 2D dataset.")
parser.add_argument("--model", type=str, default="variational_autoencoder", 
                   help="Specific model to plot (default: variational_autoencoder)")
parser.add_argument("--dataset", type=str, default="chladni_2d", 
                   help="Dataset name (default: chladni_2d)")
parser.add_argument("--log_dir", type=str, default="/workspaces/b2b-operator-inverse/logs",
                   help="Base log directory containing trained models")
parser.add_argument("--results_dir", type=str, default="results/chladni_2d",
                   help="Base results directory for saving plots")
parser.add_argument("--all-models", action="store_true", 
                   help="Plot results for all available trained models")
parser.add_argument("--n-samples", type=int, default=4,
                   help="Number of test samples to plot per model (default: 4)")
parser.add_argument("--eval-encoders", action="store_true", 
                   help="Evaluate and plot function encoder reconstruction performance")

args = parser.parse_args()

# Available models to check for
ALL_MODELS = ['b2b_linear', 'b2b_nonlinear', 'variational_autoencoder', 'invertible_network']

def get_evaluate_function(model_name):
    """Get the appropriate evaluate function for a model."""
    match model_name:
        case "b2b_linear":
            from inverse_neural_operator.models.b2b_operator_linear import evaluate
        case "b2b_nonlinear":
            from inverse_neural_operator.models.b2b_operator_nonlinear import evaluate
        case "variational_autoencoder":
            from inverse_neural_operator.models.variational_autoencoder import evaluate
        case "invertible_network":
            from inverse_neural_operator.models.invertible_network import evaluate
        case _:
            raise ValueError(f"Unknown model: {model_name}")
    return evaluate

def plot_function_encoder_results(model_name, log_dir, results_dir, test_dataset, dataset_info, test_indices):
    """Evaluate and plot function encoder reconstruction performance."""
    print(f"\n🔍 Evaluating Function Encoders for: {model_name}")
    
    model_log_dir = os.path.join(log_dir, args.dataset, model_name, "seed_1")
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_log_dir, "params.pth")):
        print(f"  ⚠️  Model {model_name} not found or incomplete training")
        return None
    
    # Load model parameters and models
    params = torch.load(f"{model_log_dir}/params.pth", weights_only=False)
    input_function_encoder, output_function_encoder, model = load_models(
        log_dir=model_log_dir,
        dataset_info=dataset_info,
        params=params,
        device=device,
    )
    
    # Ensure all models are in evaluation mode
    input_function_encoder.eval()
    output_function_encoder.eval()
    
    # Create encoder-specific results directory
    encoder_results_dir = os.path.join(results_dir, f"{model_name}_function_encoders")
    os.makedirs(encoder_results_dir, exist_ok=True)
    
    # Use the provided test indices (same for all models)
    # Calculate reconstruction errors
    input_encoder_errors = []
    output_encoder_errors = []
    
    for i, idx in enumerate(test_indices):
        # Set random seed for consistent results
        torch.manual_seed(42 + i)
        
        # Get the test sample
        sample = test_dataset[idx]
        X, u_true, Y, s_true = sample
        
        # Ensure all tensors are on the correct device
        X = X.to(device)
        u_true = u_true.to(device)
        Y = Y.to(device)
        s_true = s_true.to(device)
        
        # Validate data
        if torch.isnan(u_true).any() or torch.isnan(s_true).any():
            print(f"  ⚠️  Warning: NaN values detected in sample {i+1}")
            continue
        
        # Evaluate input function encoder (forces reconstruction)
        with torch.no_grad():
            # Encode to coefficients and decode back
            alpha = input_function_encoder.compute_coefficients(X.unsqueeze(0), u_true.unsqueeze(0))
            if isinstance(alpha, tuple):
                alpha = alpha[0]
            u_reconstructed = input_function_encoder(X.unsqueeze(0), alpha).squeeze(0)
            
            # Calculate reconstruction MSE
            input_mse = torch.mean((u_true - u_reconstructed)**2).item()
            if not (math.isnan(input_mse) or math.isinf(input_mse)):
                input_encoder_errors.append(input_mse)
        
        # Evaluate output function encoder (displacements reconstruction)
        with torch.no_grad():
            # Encode to coefficients and decode back
            beta = output_function_encoder.compute_coefficients(Y.unsqueeze(0), s_true.unsqueeze(0))
            if isinstance(beta, tuple):
                beta = beta[0]
            s_reconstructed = output_function_encoder(Y.unsqueeze(0), beta).squeeze(0)
            
            # Calculate reconstruction MSE
            output_mse = torch.mean((s_true - s_reconstructed)**2).item()
            if not (math.isnan(output_mse) or math.isinf(output_mse)):
                output_encoder_errors.append(output_mse)
        
        # Convert to numpy for plotting
        u_true_np = u_true.squeeze(-1).cpu().numpy()
        u_reconstructed_np = u_reconstructed.squeeze(-1).cpu().numpy()
        s_true_np = s_true.squeeze(-1).cpu().numpy()
        s_reconstructed_np = s_reconstructed.squeeze(-1).cpu().numpy()
        X_np = X.squeeze(-1).cpu().numpy()
        
        # Infer grid size and reshape
        grid_size = int(np.sqrt(len(X_np)))
        u_true_2d = u_true_np.reshape(grid_size, grid_size)
        u_reconstructed_2d = u_reconstructed_np.reshape(grid_size, grid_size)
        s_true_2d = s_true_np.reshape(grid_size, grid_size)
        s_reconstructed_2d = s_reconstructed_np.reshape(grid_size, grid_size)
        x_coords = X_np[:, 0].reshape(grid_size, grid_size)
        y_coords = X_np[:, 1].reshape(grid_size, grid_size)
        
        # Create figure with 4 subplots: Original vs Reconstructed for both input and output
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
        fig.suptitle(f'Function Encoder Evaluation: {model_name} - Sample {idx} (#{i+1})', fontsize=16, fontweight='bold')
        
        # Input Forces: Original vs Reconstructed
        ax1 = axes[0, 0]
        ax1.set_facecolor('white')
        contour1 = ax1.contourf(x_coords, y_coords, u_true_2d, levels=50, cmap='viridis')
        ax1.set_xlabel('X axis', fontsize=12)
        ax1.set_ylabel('Y axis', fontsize=12)
        ax1.set_title('Original Forces\n(True Input)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        cbar1 = plt.colorbar(contour1, ax=ax1)
        cbar1.set_label('Force Amplitude', rotation=270, labelpad=20, fontsize=12)
        
        ax2 = axes[0, 1]
        ax2.set_facecolor('white')
        contour2 = ax2.contourf(x_coords, y_coords, u_reconstructed_2d, levels=50, cmap='viridis')
        ax2.set_xlabel('X axis', fontsize=12)
        ax2.set_ylabel('Y axis', fontsize=12)
        ax2.set_title('Reconstructed Forces\n(Input Encoder)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        cbar2 = plt.colorbar(contour2, ax=ax2)
        cbar2.set_label('Force Amplitude', rotation=270, labelpad=20, fontsize=12)
        ax2.text(0.02, 0.98, f'MSE: {input_mse:.6f}', transform=ax2.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Output Displacements: Original vs Reconstructed
        ax3 = axes[1, 0]
        ax3.set_facecolor('white')
        contour3 = ax3.contourf(x_coords, y_coords, s_true_2d, levels=50, cmap='plasma')
        ax3.set_xlabel('X axis', fontsize=12)
        ax3.set_ylabel('Y axis', fontsize=12)
        ax3.set_title('Original Displacements\n(True Output)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        cbar3 = plt.colorbar(contour3, ax=ax3)
        cbar3.set_label('Displacement Amplitude', rotation=270, labelpad=20, fontsize=12)
        
        ax4 = axes[1, 1]
        ax4.set_facecolor('white')
        contour4 = ax4.contourf(x_coords, y_coords, s_reconstructed_2d, levels=50, cmap='plasma')
        ax4.set_xlabel('X axis', fontsize=12)
        ax4.set_ylabel('Y axis', fontsize=12)
        ax4.set_title('Reconstructed Displacements\n(Output Encoder)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        cbar4 = plt.colorbar(contour4, ax=ax4)
        cbar4.set_label('Displacement Amplitude', rotation=270, labelpad=20, fontsize=12)
        ax4.text(0.02, 0.98, f'MSE: {output_mse:.6f}', transform=ax4.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = os.path.join(encoder_results_dir, f"function_encoders_sample_{idx}_{model_name}.png")
        plt.savefig(output_file, facecolor='white', dpi=150, bbox_inches='tight')
        print(f"  📊 Saved encoder evaluation plot: {output_file}")
        
        plt.close(fig)
    
    # Calculate average reconstruction errors
    avg_input_mse = np.mean(input_encoder_errors) if input_encoder_errors else float('inf')
    avg_output_mse = np.mean(output_encoder_errors) if output_encoder_errors else float('inf')
    
    print(f"  ✅ Function Encoder Evaluation Complete:")
    print(f"    📊 Input Encoder (Forces) - Average MSE: {avg_input_mse:.6f}")
    print(f"    📊 Output Encoder (Displacements) - Average MSE: {avg_output_mse:.6f}")
    
    return {
        'input_mse': avg_input_mse,
        'output_mse': avg_output_mse,
        'results_dir': encoder_results_dir
    }

def plot_model_results(model_name, log_dir, results_dir, test_dataset, dataset_info, test_indices):
    """Plot results for a single model."""
    print(f"\n📈 Plotting results for: {model_name}")
    
    model_log_dir = os.path.join(log_dir, args.dataset, model_name, "seed_1")
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_log_dir, "params.pth")):
        print(f"  ⚠️  Model {model_name} not found or incomplete training")
        return None
    
    # Load model parameters and models
    params = torch.load(f"{model_log_dir}/params.pth", weights_only=False)
    input_function_encoder, output_function_encoder, model = load_models(
        log_dir=model_log_dir,
        dataset_info=dataset_info,
        params=params,
        device=device,
    )
    
    # Ensure all models are in evaluation mode
    model.eval()
    input_function_encoder.eval()
    output_function_encoder.eval()
    
    # Create model-specific results directory
    model_results_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    
    # Use the provided test indices (same for all models)
    # Get the appropriate evaluate function
    evaluate = get_evaluate_function(model_name)
    
    # Calculate average MSE for this model
    model_errors = []
    
    for i, idx in enumerate(test_indices):
        # Set random seed for consistent results across runs (important for variational models)
        torch.manual_seed(42 + i)
        
        # Get the test sample
        sample = test_dataset[idx]
        X, u_true, Y, s_true = sample
        
        # Ensure all tensors are on the correct device
        X = X.to(device)
        u_true = u_true.to(device)
        Y = Y.to(device)
        s_true = s_true.to(device)
        
        # Validate data shapes and detect potential issues
        if X.shape != Y.shape:
            print(f"  ⚠️  Warning: X and Y coordinate shapes differ: {X.shape} vs {Y.shape}")
        
        if torch.isnan(u_true).any() or torch.isnan(s_true).any():
            print(f"  ⚠️  Warning: NaN values detected in sample {i+1}")
            continue
            
        # Add batch dimension for evaluation
        point = (X.unsqueeze(0), u_true.unsqueeze(0), Y.unsqueeze(0), s_true.unsqueeze(0))
        
        # Get model prediction (predicted forces from observed displacements)
        u_pred = evaluate(model, point, input_function_encoder, output_function_encoder)
        u_pred = u_pred.squeeze(0)  # Remove batch dimension
        
        # Validate prediction results
        if torch.isnan(u_pred).any() or torch.isinf(u_pred).any():
            print(f"  ⚠️  Warning: Invalid prediction (NaN/Inf) for sample {i+1}, skipping...")
            continue
            
        # Calculate MSE for this sample
        mse = torch.mean((u_true - u_pred)**2).item()
        if math.isnan(mse) or math.isinf(mse):
            print(f"  ⚠️  Warning: Invalid MSE for sample {i+1}, skipping...")
            continue
            
        model_errors.append(mse)
        
        # Convert to numpy for plotting
        u_true_np = u_true.squeeze(-1).cpu().numpy()  # Remove function dimension
        u_pred_np = u_pred.squeeze(-1).cpu().numpy()  # Remove function dimension
        X_np = X.squeeze(-1).cpu().numpy()  # Remove coordinate dimension
        
        # Assuming square grid - infer grid size from data
        grid_size = int(np.sqrt(len(X_np)))
        
        # Reshape to 2D grids
        u_true_2d = u_true_np.reshape(grid_size, grid_size)
        u_pred_2d = u_pred_np.reshape(grid_size, grid_size)
        
        # Extract coordinates for plotting
        x_coords = X_np[:, 0].reshape(grid_size, grid_size)
        y_coords = X_np[:, 1].reshape(grid_size, grid_size)
        
        # Create figure with 2 subplots: Ground Truth Forces vs Predicted Forces
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
        fig.suptitle(f'Chladni 2D: {model_name} - Sample {idx} (#{i+1})', fontsize=16, fontweight='bold')
        
        # Plot 1: Ground Truth Forces (what we want to predict)
        ax1 = axes[0]
        ax1.set_facecolor('white')
        contour1 = ax1.contourf(x_coords, y_coords, u_true_2d, levels=50, cmap='viridis')
        ax1.set_xlabel('X axis', fontsize=12)
        ax1.set_ylabel('Y axis', fontsize=12)
        ax1.set_title('Ground Truth\n(True Forces)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        cbar1 = plt.colorbar(contour1, ax=ax1)
        cbar1.set_label('Force Amplitude', rotation=270, labelpad=20, fontsize=12)
        
        # Plot 2: Predicted Forces (what the model predicts)
        ax2 = axes[1]
        ax2.set_facecolor('white')
        contour2 = ax2.contourf(x_coords, y_coords, u_pred_2d, levels=50, cmap='viridis')
        ax2.set_xlabel('X axis', fontsize=12)
        ax2.set_ylabel('Y axis', fontsize=12)
        ax2.set_title('Model Prediction\n(Predicted Forces)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        cbar2 = plt.colorbar(contour2, ax=ax2)
        cbar2.set_label('Force Amplitude', rotation=270, labelpad=20, fontsize=12)
        
        # Add MSE text
        ax2.text(0.02, 0.98, f'MSE: {mse:.6f}', transform=ax2.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = os.path.join(model_results_dir, f"chladni_2d_sample_{idx}_{model_name}.png")
        plt.savefig(output_file, facecolor='white', dpi=150, bbox_inches='tight')
        print(f"  📊 Saved plot: {output_file}")
        
        plt.close(fig)
    
    avg_mse = np.mean(model_errors)
    print(f"  ✅ Completed plots for {model_name} (Average MSE: {avg_mse:.6f})")
    
    return avg_mse, model_results_dir

def create_comparison_plot(comparison_results, results_dir):
    """Create a comparison plot for all models."""
    if len(comparison_results) <= 1:
        print("📊 Skipping comparison plot (need at least 2 models)")
        return
        
    print(f"\n📊 Creating model comparison plot...")
    
    plt.figure(figsize=(12, 8), facecolor='white')
    models = list(comparison_results.keys())
    errors = list(comparison_results.values())
    
    # Create bar chart
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum'][:len(models)]
    bars = plt.bar(models, errors, color=colors)
    
    plt.ylabel('Average MSE', fontsize=14)
    plt.title('Model Performance Comparison - Chladni 2D Dataset', fontsize=16, fontweight='bold')
    plt.yscale('log')  # Use log scale for better visualization
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, error in zip(bars, errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f'{error:.3e}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Rotate x labels if needed
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    comparison_file = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(comparison_file, facecolor='white', dpi=150, bbox_inches='tight')
    print(f"📊 Model comparison saved: {comparison_file}")
    plt.close()

# Main execution
def main():
    print("🎨 Chladni 2D Results Plotting Script")
    print("=" * 50)
    
    # Create base results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Determine which models to plot
    if args.all_models:
        models_to_plot = []
        print(f"\n🔍 Checking for available models in {args.log_dir}/{args.dataset}/")
        
        for model_name in ALL_MODELS:
            model_dir = os.path.join(args.log_dir, args.dataset, model_name, "seed_1")
            if os.path.exists(os.path.join(model_dir, "params.pth")):
                models_to_plot.append(model_name)
                print(f"  ✅ {model_name}")
            else:
                print(f"  ❌ {model_name} (not found or incomplete)")
        
        if not models_to_plot:
            print("❌ No trained models found!")
            return
            
    else:
        models_to_plot = [args.model]
        print(f"\n📈 Plotting results for single model: {args.model}")
    
    # Load dataset once (we'll reuse it for all models)
    print(f"\n📊 Loading {args.dataset} test dataset...")
    # We need to load params from any available model to get dataset info
    first_model = models_to_plot[0]
    temp_log_dir = os.path.join(args.log_dir, args.dataset, first_model, "seed_1")
    temp_params = torch.load(f"{temp_log_dir}/params.pth", weights_only=False)
    
    test_dataset, dataset_info = load_dataset(temp_params, device)
    
    # Select test samples once - all models will use the same samples for fair comparison
    n_samples = min(args.n_samples, len(test_dataset))
    np.random.seed(42)  # Fixed seed for reproducible sample selection
    test_indices = np.random.choice(len(test_dataset), n_samples, replace=False)
    print(f"📊 Selected test samples: {test_indices.tolist()}")
    
    # Plot results for each model
    comparison_results = {}
    encoder_results = {}
    
    for model_name in models_to_plot:
        try:
            # Plot regular model results
            result = plot_model_results(
                model_name, args.log_dir, args.results_dir, 
                test_dataset, dataset_info, test_indices
            )
            if result is not None:
                avg_mse, model_results_dir = result
                comparison_results[model_name] = avg_mse
            
            # Plot function encoder results if requested
            if args.eval_encoders:
                encoder_result = plot_function_encoder_results(
                    model_name, args.log_dir, args.results_dir,
                    test_dataset, dataset_info, test_indices
                )
                if encoder_result is not None:
                    encoder_results[model_name] = encoder_result
                    
        except Exception as e:
            print(f"  ❌ Failed to plot {model_name}: {e}")
    
    # Create comparison plot if we have multiple models
    if len(comparison_results) > 1:
        create_comparison_plot(comparison_results, args.results_dir)
    
    # Summary
    print(f"\n🎉 Plotting complete!")
    print(f"📁 Results saved in: {args.results_dir}")
    print(f"📊 Models plotted: {list(comparison_results.keys())}")
    
    if comparison_results:
        print(f"\n📈 Performance Summary (Average MSE):")
        sorted_results = sorted(comparison_results.items(), key=lambda x: x[1])
        for i, (model, mse) in enumerate(sorted_results, 1):
            print(f"  {i}. {model}: {mse:.6f}")
    
    if encoder_results:
        print(f"\n🔍 Function Encoder Performance Summary:")
        # Since all models use the same function encoders and same test samples,
        # the reconstruction performance should be identical
        if len(encoder_results) > 1:
            print("  (Note: All models use the same function encoders, so performance should be identical)")
        for model_name, results in encoder_results.items():
            print(f"  {model_name}:")
            print(f"    Input Encoder (Forces): {results['input_mse']:.6f}")
            print(f"    Output Encoder (Displacements): {results['output_mse']:.6f}")

if __name__ == "__main__":
    main() 