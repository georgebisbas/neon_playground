from devito import Grid, TimeFunction, Eq, Operator, norm
import numpy as np

# Define 2D grid
nx, ny = 10000, 10000  # Grid points in x and y
dx, dy = 1.0, 1.0  # Grid spacing
nt = 100  # Number of timesteps
dt = 0.1  # Time step size
D = 0.1  # Diffusion coefficient

grid = Grid(shape=(nx, ny), extent=(nx*dx, ny*dy))

# Define time-dependent function u
u = TimeFunction(name="u", grid=grid, time_order=1, space_order=2)

# Define finite-difference stencil for diffusion
laplacian = u.laplace  # ∇²u
diffusion_eq = Eq(u.forward, u + dt * D * laplacian)  # Forward-time scheme

# Create and apply operator
op = Operator(diffusion_eq)
u.data[0, nx//2, ny//2] = 1.0  # Initial condition: Point source in center

# Run the simulation
op.apply(time=nt)

print("Norm u :", norm(u))
# Plot the final state

