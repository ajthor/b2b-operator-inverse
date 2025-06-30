import numpy as np
import matplotlib.pyplot as plt
import os

from chladni_2d import load_chladni_original, generate_chladni_data


class ChladniSolver:
    """Iterator wrapper for the generated Chladni data."""
    
    def __init__(self, numPoints=25, n_range=5, m_range=5):
        self.numPoints = numPoints
        self.n_range = n_range
        self.m_range = m_range
        
        # Try to load existing data, generate if not available
        try:
            self.data = load_chladni_original()
            print("Loaded existing Chladni data")
        except:
            print("No existing data found, generating new data...")
            generate_chladni_data()
            self.data = load_chladni_original()
            
        self.current_idx = 0
        self.total_samples = self.data['S_full'].shape[2]
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.current_idx >= self.total_samples:
            self.current_idx = 0  # Reset for continuous iteration
            
        idx = self.current_idx
        self.current_idx += 1
        
        # Extract coordinate arrays
        x = self.data['x']
        y = self.data['y']
        
        # Extract data for this sample
        S_2d = self.data['S_full'][:, :, idx]
        Z_2d = self.data['Z_full'][:, :, idx]
        alpha = self.data['alpha_full'][:, :, idx]
        
        # Calculate omega from the saved parameters
        L = self.data['L']
        M = self.data['M']
        omega = 600 * np.pi / M  # Same formula as in data generation
        
        return {
            "X": x,
            "Y": y, 
            "alpha": alpha,
            "S": S_2d.flatten(),  # Flatten to match expected format
            "Z": Z_2d.flatten(),  # Flatten to match expected format
            "omega": omega
        }


if __name__ == "__main__":

    np.random.seed(42)
    
    # Create solver with default parameters
    solver = ChladniSolver(numPoints=25, n_range=5, m_range=5)
    
    # Create output directory
    os.makedirs("visualizations/output", exist_ok=True)
    
    # Create 4 main plots
    for plot_num in range(4):
        # Generate a sample
        sample = next(iter(solver))
        
        # Extract data
        X = sample["X"]
        Y = sample["Y"]
        alpha = sample["alpha"]
        S = sample["S"]  # Forcing function
        Z = sample["Z"]  # Displacement response
        omega = sample["omega"]
        
        # Reshape 1D arrays back to 2D grids
        numPoints = len(X)
        S_2d = S.reshape(numPoints, numPoints)
        Z_2d = Z.reshape(numPoints, numPoints)
        
        # Create meshgrid for plotting
        X_grid, Y_grid = np.meshgrid(X, Y)
        
        # Create figure with subplots (white background, wider for colorbars, taller for square plots)
        fig, axs = plt.subplots(1, 2, figsize=(18, 7.5), facecolor='white')
        
        # Plot 1: Forcing function S
        ax1 = axs[0]
        ax1.set_facecolor('white')
        contour1 = ax1.contourf(X_grid, Y_grid, S_2d.T, levels=50, cmap='viridis')
        ax1.set_xlabel('X axis (m)', fontsize=14)
        ax1.set_ylabel('Y axis (m)', fontsize=14)
        ax1.set_title('Forcing Function S(x,y)', fontsize=16, fontweight='bold')
        ax1.tick_params(labelsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar with proper labels
        cbar1 = plt.colorbar(contour1, ax=ax1)
        cbar1.set_label('Forcing Amplitude', rotation=270, labelpad=25, fontsize=14)
        cbar1.ax.tick_params(labelsize=12)
        
        # Plot 2: Displacement response Z 
        ax2 = axs[1]
        ax2.set_facecolor('white')
        
        # Nice filled contours
        contour2 = ax2.contourf(X_grid, Y_grid, Z_2d.T, levels=50, cmap='RdBu_r')
        
        ax2.set_xlabel('X axis (m)', fontsize=14)
        ax2.set_ylabel('Y axis (m)', fontsize=14)
        ax2.set_title(f'Displacement Response Z(x,y), ω = {omega:.2f} Hz', fontsize=16, fontweight='bold')
        ax2.tick_params(labelsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar2 = plt.colorbar(contour2, ax=ax2)
        cbar2.set_label('Displacement Amplitude', rotation=270, labelpad=25, fontsize=14)
        cbar2.ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        # Save the plot with numbered filename
        output_file = f"visualizations/output/chladni_2d_sample_{plot_num+1}.png"
        plt.savefig(output_file, facecolor='white', dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_file}")
        
        plt.close(fig)
    
    print("Visualization complete!") 