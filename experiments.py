# Import necessary libraries
import numpy as np
import torch
import torch.optim as optim

from utils.plotting import plot_density_subplots, plot_2d_particle_trajectories
from utils.data import sample_initial_density, sample_initial_density_random
from utils.porous_medium import m, barenplatt_constants
from deep_jko import PhiNet, deep_jko

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# problem dimension
d = 2 

# Barenplatt constants 
alpha, beta, C = barenplatt_constants(d, m=2)

# Number of particles
N = 1000 

# JKO time step
Delta_t = 0.001 # 0.0005 for 6D

# Inner dynamic time step
N_tau = 2
d_tau = 1.0 / N_tau

# Number of JKO steps
K = 10 

# Neural network parameters
hidden_dim = 128
num_layers = 3 # hidden layers
learning_rate = 10e-5
num_iterations_per_inner_step = 10 #000  # used for convergence, 20000 for 6D

# initial distribution
x0, rho0 = sample_initial_density(N, d, alpha, beta, C, device)
model = PhiNet(input_dim=d + 1, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# run Deep JKO
x_list, rho_list, t_list, mse_list, iteration_losses, final_losses = deep_jko(device, N, K, d, x0, rho0, model, optimizer, Delta_t, N_tau, d_tau, num_iterations_per_inner_step, alpha, beta, C, t0=1e-3)

# times for density evolution plot
times_to_plot = [0.000, 0.002, 0.005, 0.008, 0.01] # adjust based on Delta_t 
# plot density evolution
plot_density_subplots(x_list, rho_list, t_list, times_to_plot, alpha, beta, C, d, filename="density_evolution.png")

# plot particle trajectories in 2D
if d == 2:
    plot_2d_particle_trajectories(x_list, t_list, filename="particle_trajectories_2D.html")
