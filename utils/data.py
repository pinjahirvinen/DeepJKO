import torch
import numpy as np
import math

#####################################################
# FUNCTIONS FOR INITIAL DENSITIES
#####################################################
def sample_initial_density(N, d, alpha, beta, C, device, m=2.0, t0=1e-3):
    """
    Samples initial density of PME from the Barenplatt solution
    """
    # Barenblatt solution parameters
    #alpha = d / (d * (m - 1) + 2)
    #beta = ((m - 1) * alpha) / (2 * d * m)
    r_max = np.sqrt(C / beta) * t0 ** (alpha / d)

    x = np.zeros((N, d))
    rho = np.zeros(N)
    generated = 0

    while generated < N:
        candidates = np.random.uniform(-r_max, r_max, size=(2 * (N - generated), d))
        norms = np.linalg.norm(candidates, axis=1)
        valid = norms <= r_max

        take = min(np.sum(valid), N - generated)
        x[generated:generated + take] = candidates[valid][:take]

        r = norms[valid][:take]
        rho[generated:generated + take] = t0 ** (-alpha) * (1 - beta * (r ** 2) * t0 ** (-2 * alpha / d)) ** (1 / (m - 1))
        generated += take

    # convert to tensors
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    rho_tensor = torch.tensor(rho, dtype=torch.float32, device=device)  # shape: (N,)

    # normalize total mass
    volume = (np.pi ** (d / 2)) / math.gamma(d / 2 + 1) * r_max ** d
    rho_tensor *= volume / rho_tensor.sum()

    return x_tensor, rho_tensor.unsqueeze(1)  # return shape (N, d), (N, 1)

def sample_initial_density_random(N, d, alpha, beta, C, device, m=2.0, t0=1e-3):
    # Barenblatt solution parameters (used only to compute r_max)
    #alpha = d / (d * (m - 1) + 2)
    #beta = ((m - 1) * alpha) / (2 * d * m)
    r_max = np.sqrt(C / beta) * t0 ** (alpha / d)

    x = np.zeros((N, d))
    generated = 0

    # Sample positions uniformly within the ball of radius r_max
    while generated < N:
        candidates = np.random.uniform(-r_max, r_max, size=(2 * (N - generated), d))
        norms = np.linalg.norm(candidates, axis=1)
        valid = norms <= r_max

        take = min(np.sum(valid), N - generated)
        x[generated:generated + take] = candidates[valid][:take]
        generated += take

    # Assign random densities (positive values)
    rho = np.random.rand(N)  # uniform random values in [0, 1)

    # Normalize total mass
    volume = (np.pi ** (d / 2)) / math.gamma(d / 2 + 1) * r_max ** d
    rho *= volume / np.sum(rho)

    # Convert to tensors
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    rho_tensor = torch.tensor(rho, dtype=torch.float32, device=device)

    return x_tensor, rho_tensor.unsqueeze(1)  # shape: (N, d), (N, 1)
