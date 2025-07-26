import torch
import numpy as np

# porous medium
m=2 # m>1

#####################################################
# internal energy of PME
#####################################################
def U(rho, m=2):
    # Internal energy function U(rho) for porous medium
    return (1/(m-1))*(rho**m).mean() 

def barenplatt_constants(d, m):
    # Barenplatt constants
    alpha = d / (d * (m - 1) + 2)
    beta = ((m - 1) * alpha) / (2 * d * m)
    if d==2:
        C = np.sqrt((2*beta)/np.pi) # numerical solution for 2D
    elif d==6:
        C = ((24*beta**3)/(np.pi**3))**0.25 # numerical solution for 6D
    return alpha, beta, C

def analytical_solution(x: torch.Tensor, t: float, alpha: float, beta: float, C: float, dim: int,
                      t0: float = 1e-3, m: float = 2.0):

    t_eff = max(t, 0.0)
    denom_time = t_eff + t0
    r_sq = (x**2).sum(dim=1)
    factor_inside = C - beta * r_sq * (denom_time ** (-2.0 * alpha / dim))
    factor_clamped = torch.clamp(factor_inside, min=0.0)
    return (denom_time ** (-alpha)) * factor_clamped