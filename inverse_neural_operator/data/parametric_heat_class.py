from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import IterableDataset

import dolfinx
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem


class ParametricHeatDataset(IterableDataset):
    def __init__(self, domain, V, dt=0.01, T=1.0, seed=None):
        super().__init__()
        self.domain = domain
        self.V = V
        self.dt = dt
        self.T = T
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Define boundary condition (u=0 on the entire boundary)
        fdim = domain.topology.dim - 1
        facets = dolfinx.mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.full(x.shape[1], True)
        )
        self.bc = dolfinx.fem.dirichletbc(
            ScalarType(0), dolfinx.fem.locate_dofs_topological(V, fdim, facets), V
        )

        dof_coordinates = V.tabulate_dof_coordinates()
        x_coords = dof_coordinates[:, 0]
        y_coords = dof_coordinates[:, 1]
        self.sorted_dofs = np.lexsort((y_coords, x_coords))
        
        # Get spatial coordinates for output
        self.x_coords = dof_coordinates[self.sorted_dofs]

    def __iter__(self):
        """Generate infinite samples from the dataset."""
        while True:
            # Sample-varying parameter
            alpha = np.random.uniform(0.01, 1.0)
            
            # Generate random initial condition using a sum of Gaussian bumps
            u = dolfinx.fem.Function(self.V)
            
            # Generate random parameters for Gaussian bumps
            num_bumps = np.random.randint(1, 4)
            bump_amplitudes = np.random.uniform(0.5, 2.0, size=num_bumps)
            bump_x = np.random.uniform(0.2, 0.8, size=num_bumps)
            bump_y = np.random.uniform(0.2, 0.8, size=num_bumps)
            bump_sigma = np.random.uniform(0.05, 0.2, size=num_bumps)
            
            # Function to create a sum of Gaussian bumps
            def multi_bump_function(x, y, centers_x, centers_y, amplitudes, sigmas):
                result = np.zeros_like(x)
                for i in range(len(centers_x)):
                    # Calculate the Gaussian bump
                    exponent = -((x - centers_x[i]) ** 2 + (y - centers_y[i]) ** 2) / (
                        2 * sigmas[i] ** 2
                    )
                    result += amplitudes[i] * np.exp(exponent)
                return result
            
            # Set initial condition
            u.interpolate(
                lambda x: multi_bump_function(
                    x[0], x[1], bump_x, bump_y, bump_amplitudes, bump_sigma
                )
            )
            
            # Solve the heat equation
            un = dolfinx.fem.Function(self.V)
            un.x.array[:] = u.x.array
            
            s = dolfinx.fem.Function(self.V)
            
            # Define variational problem
            v = ufl.TestFunction(self.V)
            ds = ufl.TrialFunction(self.V)
            
            # Time stepping loop to solve heat equation
            num_steps = int(self.T / self.dt)
            for _ in range(num_steps):
                # Weak form of the heat equation
                F = (s - un) / self.dt * v * ufl.dx + alpha * ufl.dot(
                    ufl.grad(s), ufl.grad(v)
                ) * ufl.dx
                
                J = ufl.derivative(F, s, ds)
                
                # Create and solve the linear problem
                problem = NonlinearProblem(F, s, bcs=[self.bc], J=J)
                solver = NewtonSolver(MPI.COMM_WORLD, problem)
                solver.atol = 1e-8
                solver.rtol = 1e-8
                solver.max_it = 50
                
                n, converged = solver.solve(s)
                
                # Update for next time step
                un.x.array[:] = s.x.array[:]
            
            yield {
                'x': self.x_coords,
                'u': u.x.array[self.sorted_dofs],
                'y': self.x_coords,
                's': s.x.array[self.sorted_dofs],
                'alpha': alpha
            }


def plot_parametric_heat_sample(sample, save_path="parametric_heat_sample.png"):
    """Plot a single sample from the Parametric Heat dataset"""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    u = sample["u"]  # Initial condition
    s = sample["s"]  # Solution
    alpha = sample["alpha"]  # Diffusion coefficient

    n = 51  # Grid size
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    # Plot initial condition
    im1 = axs[0].pcolormesh(X, Y, u.reshape(n, n), shading="auto", cmap="viridis")
    axs[0].set_title("Initial Condition")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    plt.colorbar(im1, ax=axs[0])

    # Plot solution
    im2 = axs[1].pcolormesh(X, Y, s.reshape(n, n), shading="auto", cmap="viridis")
    axs[1].set_title(f"Solution at T=1.0, α={alpha:.4f}")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    plt.colorbar(im2, ax=axs[1])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    # Create mesh
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx=50, ny=50)
    V = dolfinx.fem.functionspace(domain, ("CG", 1))

    # Time stepping parameters
    dt = 0.01
    T = 1.0

    # Create dataset
    dataset = ParametricHeatDataset(domain, V, dt=dt, T=T, seed=42)

    # Generate and visualize a single sample
    sample = next(iter(dataset))

    plot_parametric_heat_sample(sample, save_path="parametric_heat_sample.png")
    print("Sample visualization saved to parametric_heat_sample.png")