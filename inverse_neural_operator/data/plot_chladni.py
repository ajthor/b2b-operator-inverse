import numpy as np
import matplotlib.pyplot as plt
import os

from chladni_2d import load_chladni_data, load_normalization_stats

if __name__ == "__main__":

    np.random.seed(42)
    
    # Load the dataset
    try:
        ds = load_chladni_data()
        print("Dataset loaded successfully!")
        print(f"Training samples: {len(ds['train'])}")
        print(f"Testing samples: {len(ds['test'])}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please run 'python chladni_2d.py' first to generate the dataset.")
        exit(1)
    
    # Load normalization stats for reference
    try:
        norm_stats = load_normalization_stats()
        print("Normalization stats loaded")
    except:
        print("No normalization stats found")
        norm_stats = None
    
    # Get a sample from the training set
    sample_idx = 0
    sample = ds['train'][sample_idx]
    
    # Extract data - note that data is normalized in the dataset
    X_coords = np.array(sample["X"])  # (n_points, 2) coordinates
    u_forces = np.array(sample["u"])  # (n_points,) forcing function values
    Y_coords = np.array(sample["Y"])  # (n_points, 2) coordinates (same as X)
    s_displacements = np.array(sample["s"])  # (n_points,) displacement values
    
    # Get unique x and y coordinates for meshgrid
    # Since data is on a regular grid, we can extract unique values
    n_points = len(X_coords)
    grid_size = int(np.sqrt(n_points))
    
    # Reshape coordinates and data back to 2D
    X_2d = X_coords[:, 0].reshape(grid_size, grid_size)
    Y_2d = X_coords[:, 1].reshape(grid_size, grid_size)
    u_2d = u_forces.reshape(grid_size, grid_size)
    s_2d = s_displacements.reshape(grid_size, grid_size)
    
    # Extract 1D coordinate arrays for labeling
    x_1d = X_2d[:, 0]
    y_1d = Y_2d[0, :]
    
    # Create output directory
    os.makedirs("visualizations/output", exist_ok=True)
    
    # Create figure with subplots (white background, wider for colorbars, taller for square plots)
    fig, axs = plt.subplots(1, 2, figsize=(18, 7.5), facecolor='white')
    
    # Plot 1: Forcing function u
    ax1 = axs[0]
    ax1.set_facecolor('white')
    contour1 = ax1.contourf(X_2d, Y_2d, u_2d, levels=50, cmap='viridis')
    ax1.set_xlabel('X axis (normalized)', fontsize=14)
    ax1.set_ylabel('Y axis (normalized)', fontsize=14)
    ax1.set_title('Forcing Function u(x,y)', fontsize=16, fontweight='bold')
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar with proper labels
    cbar1 = plt.colorbar(contour1, ax=ax1)
    cbar1.set_label('Normalized Forcing Amplitude', rotation=270, labelpad=25, fontsize=14)
    cbar1.ax.tick_params(labelsize=12)
    
    # Plot 2: Displacement response s 
    ax2 = axs[1]
    ax2.set_facecolor('white')
    
    # Nice filled contours
    contour2 = ax2.contourf(X_2d, Y_2d, s_2d, levels=50, cmap='RdBu_r')
    
    ax2.set_xlabel('X axis (normalized)', fontsize=14)
    ax2.set_ylabel('Y axis (normalized)', fontsize=14)
    ax2.set_title(f'Displacement Response s(x,y) - Sample #{sample_idx + 1}', fontsize=16, fontweight='bold')
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar2 = plt.colorbar(contour2, ax=ax2)
    cbar2.set_label('Normalized Displacement Amplitude', rotation=270, labelpad=25, fontsize=14)
    cbar2.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    # Save the plot 
    output_file = "visualizations/output/chladni_2d_sample.png"
    plt.savefig(output_file, facecolor='white', dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    
    plt.close(fig)
    
    # Also create individual plots for multiple samples
    for plot_num in range(4):
        sample = ds['train'][plot_num]
        s_displacements = np.array(sample["s"])
        s_2d = s_displacements.reshape(grid_size, grid_size)
        
        plt.figure(figsize=(8, 6), facecolor='black')
        ax = plt.gca()
        ax.set_facecolor('black')
        
        # Create zero-level contours in golden color
        contour = plt.contour(X_2d, Y_2d, s_2d, levels=[0], 
                            colors=[(0.85, 0.65, 0.13)], linewidths=3)
        
        plt.xlabel('X axis (normalized)', color='white')
        plt.ylabel('Y axis (normalized)', color='white')
        plt.title(f'Sample #{plot_num + 1} - Zero-level Contours', color='white')
        
        # Set axis colors
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        
        plt.tight_layout()
        
        # Save individual plot
        plot_filename = f"visualizations/output/chladni_sample_{plot_num+1}.png"
        plt.savefig(plot_filename, facecolor='black', dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot_filename}")
        
        plt.close()
    
    print("Visualization complete!") 