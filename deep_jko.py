# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import time as realtime
from utils.plotting import plot_iteration_loss, plot_solution_boundary, plot_mse
from utils.porous_medium import U, analytical_solution

#####################################################
# POTENTIAL NETWORK
#####################################################
# Neural network for approximating the potential phi_theta
class PhiNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(PhiNet, self).__init__()
        self.input_dim = input_dim  # d + 1 (space + time)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.W0 = nn.Linear(self.input_dim, hidden_dim, bias=True)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(num_layers - 1)])
        self.w_final = nn.Parameter(torch.randn(hidden_dim))

    def activation(self, x):
        #return torch.abs(x)
        return torch.log(torch.exp(x)+torch.exp(-x))
        #return torch.tanh(x)
        
    def forward(self, tau, x):
        if len(tau.shape) == 1:
            tau = tau.unsqueeze(-1)
        s = torch.cat([tau, x], dim=-1)
        u = self.activation(self.W0(s))

        for layer in self.hidden_layers:
            u = u + self.activation(layer(u))

        phi = torch.matmul(u, self.w_final)

        return phi

# Function to compute gradients and trace of Hessian
def compute_grad_and_trace_hessian(model, tau, x, num_vectors=10):
    x_temp = x.detach().clone()
    x_temp.requires_grad_(True)
    phi = model(tau, x_temp) 
    grad_phi = torch.autograd.grad(phi.sum(), x_temp, create_graph=True)[0]

    # Use Hutchinson's estimator for trace of Hessian
    trace_hessian = torch.zeros_like(grad_phi[:,0])
    for i in range(num_vectors):
        v = torch.randn_like(x_temp)  # random vector v
        Hv = torch.autograd.grad((grad_phi*v).sum(), x_temp, create_graph=True, retain_graph=True)[0]
        trace_hessian += (v * Hv).sum(dim=1)  # approximate trace
    trace_hessian /= num_vectors
    
    return grad_phi, trace_hessian


def loss_and_forward(device, model, x_init, rho_init, N_tau, d_tau, N, Delta_t, num_vectors):

    N = x_init.shape[0]

    x_temp = x_init
    rho_init = rho_init.view(-1)
    log_rho = torch.log(rho_init + 1e-12)
    kinetic_term = torch.tensor(0.0, device=device)
    
    for n in range(N_tau+1): 
        tau = torch.full((N,), n*d_tau, device=device)
        grad, trace = compute_grad_and_trace_hessian(model, tau, x_temp, num_vectors)

        velocity_term = - grad 

        # k1
        grad1, _ = compute_grad_and_trace_hessian(model, tau, x_temp, num_vectors)
        k1 = -grad1

        # k2
        x2 = x_temp + 0.5 * d_tau * k1
        grad2, _ = compute_grad_and_trace_hessian(model, tau, x2, num_vectors)
        k2 = -grad2

        # k3
        x3 = x_temp + 0.5 * d_tau * k2
        grad3, _ = compute_grad_and_trace_hessian(model, tau, x3, num_vectors)
        k3 = -grad3

        # k4
        x4 = x_temp + d_tau * k3
        grad4, _ = compute_grad_and_trace_hessian(model, tau, x4, num_vectors)
        k4 = -grad4

        # RK4 update particle positions
        x_temp = x_temp + (d_tau / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # update logarithmic density
        log_rho = log_rho - d_tau * trace 

        # update kinetic term 
        kinetic_term = kinetic_term + d_tau * ((velocity_term**2).sum(dim=1)).mean()

    rho_new = rho_init / torch.exp(log_rho) 
    
    loss = kinetic_term + 2.0 * Delta_t * (U(rho_new) / (rho_new + 1e-12)).mean() 
    return loss, x_temp.detach(), rho_new.detach()



#####################################################
# THE DEEP JKO ALGORITHM
#####################################################
def deep_jko(device, N, K, d, x0, rho0, model, optimizer, Delta_t, N_tau, d_tau, num_iterations_per_inner_step, alpha, beta, C, t0=1e-3):
    step_start_time = realtime.time()
    
    # lists for solutions
    x_list = []
    rho_list = []
    t_list = []
    mse_list = []
    final_losses = []

    time = 0.0

    x_list.append(x0.clone())
    rho_list.append(rho0.clone())
    t_list.append(time)

    # Visualize initial particles
    plot_solution_boundary(t=0, x=x0, rho=rho0, alpha=alpha, beta=beta, C=C, dim=d, fig_title=f"initial_particles.png")
    
    # outer step (JKO step)
    for n in range(K):
        print(f"Starting JKO step {n + 1}/{K}, PDE time {time:.4f}->{time+Delta_t:.4f}")

        time += Delta_t
        
        iteration_losses = []

        # inner optimization loop for current JKO step
        # for-loop used for convergence
        for iteration in range(num_iterations_per_inner_step):
            
            optimizer.zero_grad()

            loss, _, _ = loss_and_forward(device, model, x0, rho0, N_tau, d_tau, N, Delta_t, num_vectors=10) 
            loss.backward()
            
            optimizer.step() # update theta

            # store the loss for plotting
            iteration_losses.append(loss.item())

            if (iteration + 1) % (num_iterations_per_inner_step//2) == 0:
                print(f"Inner iteration {iteration + 1}/{num_iterations_per_inner_step}, Loss: {loss.item():.6f}")

        # plot NN iteration loss
        plot_iteration_loss(iteration_losses, n+1) 
        
        _, x_next, rho_next = loss_and_forward(device, model, x0, rho0, N_tau, d_tau, N, Delta_t, num_vectors=10) 
        
        rho0 = rho_next.clone()
        x0 = x_next.clone()
        
        final_losses.append((loss.item(), time))
        
        # calculate MSE
        with torch.no_grad():
            rho_barenplatt = analytical_solution(x_next, time, alpha, beta, C, d)
            mse = torch.mean((rho_next - rho_barenplatt)**2).item()
            mse_list.append((n+1, time, mse))
            print(f"  => PDE time t={time:.4f}, MSE vs Barenblatt={mse:.6f}")
        
        # After optimization, update particles and densities
        x_list.append(x_next.detach())
        rho_list.append(rho_next.detach())
        t_list.append(time)

        # Visualize particles at current step
        print(f"rho passed to plot: shape {rho0.shape}")
        plot_solution_boundary(t=time, x=x0, rho=rho0, alpha=alpha, beta=beta, C=C, dim=d, fig_title=f"particles_step_{n+1}.png")
        print(f"  => Outer step {n+1}, final loss={loss.item():.6f}")

        # End real time timer and print duration
        step_end_time = realtime.time()
        elapsed = step_end_time - step_start_time
        print(f"  => Real time for step {n + 1}: {elapsed:.2f} seconds")

    # plot computed vs Barenplatt density mse
    plot_mse(mse_list)

    return x_list, rho_list, t_list, mse_list, iteration_losses, final_losses
