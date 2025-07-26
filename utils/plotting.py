# %%
# Import necessary libraries
import numpy as np
import torch
#import torch.utils.checkpoint as checkpoint
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.gridspec as gridspec
from utils.porous_medium import analytical_solution


#####################################################
# PLOTTING FUNCTIONS
#####################################################
# function to plot boundary and particles
def plot_solution_boundary(t, x, rho, alpha, beta, C, dim, fig_title, x_range=(-1, 1), y_range=(-1, 1), num_points=1000, t0=1e-3):
    
    xx = np.linspace(x_range[0], x_range[1], num_points)
    yy = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(xx, yy)
    points = np.stack([X, Y], axis=-1) 

    fig = plt.figure(figsize=(8, 6))

    points_tensor = torch.tensor(points.reshape(-1, 2), dtype=torch.float32, device=x.device)
    rho_analyt = analytical_solution(points_tensor, t, alpha, beta, C, dim)  # returns (N^2,)
    rho_analyt = rho_analyt.view(num_points, num_points).detach().cpu().numpy()
    contour = plt.contour(X, Y, rho_analyt, levels=0, colors="green", linewidths=2)

    # overlay the particles
    x_np = x.detach().cpu().numpy()
    rho_np = rho.detach().cpu().numpy().squeeze()
    scatter = plt.scatter(x_np[:, 0], x_np[:, 1], c=rho_np, cmap='inferno', s=15)
    plt.colorbar(scatter, label="Density")
    plt.title(f"Particle Positions at Step t = {t:.3f}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha=0.3)
    fig.savefig(fig_title, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_mse(mse_list, filename="mse_plot.png"):

    steps, times, mses = zip(*mse_list)
    
    # Print MSE
    for step, time, mse in mse_list:
        print(f"MSE at time t = {time:.4f} (step {step}): {mse:.6e}")

    fig, ax1 = plt.subplots()

    # Plot MSE vs Step
    ax1.set_xlabel('JKO step')
    ax1.set_ylabel('MSE')
    ax1.plot(steps, mses)
    ax1.tick_params(axis='y')

    # Title and legend
    plt.title('Computed vs. analytical density over time')
    fig.tight_layout() 
    fig.legend(loc='upper right')
    plt.savefig(filename) 
    plt.close() 

def plot_iteration_loss(iteration_losses, step_number):
    """
    Plots and saves the loss over iterations for each JKO step.
    
    Parameters:
        iteration_losses (list): List of loss values recorded at each iteration.
        step_number (int): The current JKO step number, used to save the figure with a unique name.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_losses, label=f'JKO Step {step_number}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss over Iterations - JKO Step {step_number}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure with a unique name for each JKO step
    filename = f'iteration_loss_step_{step_number}.png'
    plt.savefig(filename)
    plt.close()  

def plot_density_subplots(x_list, rho_list, t_list, times_to_plot, alpha, beta, C, dim, filename="density_evolution.png", 
                          x_range=(-1, 1), y_range=(-1, 1), num_points=1000):
    """
    Plots and saves density at the specified outer times using outputs
    with overlaid analytical solution
    
    Parameters:
        x_list: List of tensors with particle positions at each outer time.
        rho_list: List of tensors with densities at each outer time.
        t_list: List of outer times corresponding to the positions and densities.
        times_to_plot: List of outer times to plot densities for.
        filename: The name of the file to save the figure as.
        x_range, y_range: The range for x and y axes.
        num_points: Resolution of the analytical solution grid.
    """

    # Setup figure and GridSpec
    fig = plt.figure(figsize=(5 * len(times_to_plot), 5))

    gs = gridspec.GridSpec(1, len(times_to_plot))
    fig.suptitle('Density Evolution of The 2D Porous Medium Equation', fontsize=22, fontweight='bold')

    xx = np.linspace(x_range[0], x_range[1], num_points)
    yy = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(xx, yy)
    points = np.stack([X, Y], axis=-1)
    points_tensor = torch.tensor(points.reshape(-1, 2), dtype=torch.float32, device=x_list[0].device)
    t_list_rounded = [round(t.item() if isinstance(t, torch.Tensor) else t, 3) for t in t_list]
    
    scatter = None
    for i, target_time in enumerate(times_to_plot):
        ax = fig.add_subplot(gs[0, i])
        time_index = t_list_rounded.index(target_time) 
        
        t = t_list[time_index]
        x = x_list[time_index]
        rho = rho_list[time_index]

        # Overlay analytical solution
        rho_analyt = analytical_solution(points_tensor, t, alpha, beta, C, dim)
        rho_analyt = rho_analyt.view(num_points, num_points).detach().cpu().numpy()
        ax.contour(X, Y, rho_analyt, levels=0, colors="green", linewidths=2)

        # Plot particles
        x_np = x.detach().cpu().numpy()
        rho_np = rho.detach().cpu().numpy()
        scatter = ax.scatter(x_np[:, 0], x_np[:, 1], c=rho_np, cmap='inferno', s=15)
        
        ax.set_title(f"t = {t:.3f}", fontsize=20)
        ax.set_xlabel(r"$x_1$", fontsize=18)
        ax.set_ylabel(r"$x_2$", fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.86)  
    plt.savefig(filename, dpi=300)
    print(f"Figure saved as {filename}")
    plt.close()

def plot_2d_particle_trajectories(x_list, t_list, filename):
    """
    Plots 2D particle trajectories over time in a 3D space (x, y, t).
    
    Parameters:
    - particle_positions: np.ndarray of shape (T, N, 2), 2D positions over time
    - dt: time step between each outer step
    """

    T = len(x_list)
    N = x_list[0].shape[0]

    fig = go.Figure()

    for i in range(N):
        x_traj = [x_list[t][i, 0].item() for t in range(T)]
        y_traj = [x_list[t][i, 1].item() for t in range(T)]
        t_vals = t_list # time axis

        fig.add_trace(go.Scatter3d(
            x=x_traj,
            y=y_traj,
            z=t_vals,
            mode='lines',
            line=dict(color='blue', width=2),
            opacity=0.4
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='time',
        ),
        title="2D Particle Trajectories Over Time"
    )

    fig.write_html(filename)
    print(f"Plot saved as {filename}")