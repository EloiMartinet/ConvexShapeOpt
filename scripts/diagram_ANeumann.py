"""Compute the Blaschke-Santalo diagram of volume and first two Neumann eigenvalues
"""

import torch
from tqdm import tqdm
import shutil
import os
import math
import numpy as np

from shapes.invertible_nn import ConvexDiffeo
from shapes.plot_utils import plot_all_shapes, plot_diagram
from shapes.repulsion_energy import repulsion_energy, pointwise_repulsion_energy

# Important for eigenvalue computation
torch.set_default_dtype(torch.float64)

N_MODELS = 200
N_UNITS = 30
OUTPUT_FOLDER = 'res/Neumann'
N_ITER = 1000            # Maximum number of iterations
N_SOURCES = 800          # Number of sources for the Galerkin RBF
N_POINTS_QUAD = 15_000   # Number of points for the quadratures
N_POINTS_REG = 500       # Number of points for the regularization
PLOT_EVERY = 10          # Plot all the pictures and save data every
REPULSION = 2.5

# Loss function weights
REG_MULTPLIER = 1e-6
COG_MULTPLIER = 1.

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')

# Create the output foler
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Draw a bunch of models
models = [
    ConvexDiffeo(input_size=2, n_unit=N_UNITS).to(device)
    for _ in range(N_MODELS)
]

# Collect parameters from all models
all_parameters = []
for model in models:
    all_parameters += list(model.parameters())

# Set up the optimizer
optimizer = torch.optim.LBFGS(
    all_parameters,
    lr=.0001,
    history_size=20, #TODO: TAKE THIS SMALLER ?
    max_iter=20,
    line_search_fn="strong_wolfe"
)


def compute_x(model):
    eig = model.neumann_eigenvalues(
        n_ev=4,
        n_quad_points=N_POINTS_QUAD,
        n_sources=N_SOURCES,
        normalize=True
    )[0]
    mu1, mu2 = eig[1], eig[2]

    x1 = mu1/10.65
    x2 = mu2/21.30
    x = torch.hstack([x1.reshape(-1), x2.reshape(-1)])

    return x


def closure():
    optimizer.zero_grad()

    # Compute the global diagram (no grad)
    with torch.no_grad():
        X_list = [compute_x(m) for m in models]
        X = torch.stack(X_list, dim=0)

    # Compute repulsion energy gradient w.r.t. diagram points
    X_for_grad = X.clone().detach().requires_grad_(True)
    E = repulsion_energy(X_for_grad, s=REPULSION)
    g = torch.autograd.grad(E, X_for_grad, create_graph=False, retain_graph=False)[0]
    E_val = E.detach()  # scalar energy value (no grad)

    # Compute per-model Jacobian regularization and apply chain rule
    regs = []
    cog_pens = []

    for i, m in enumerate(models):
        # compute_x(m) builds a small, temporary graph for that model only
        x_i = compute_x(m)

        # Add regularizer
        reg_i = m.jacobian_regularizer(N_POINTS_REG)
        regs.append(reg_i.detach())

        # Add the CoG penalization
        cog_pen_i = torch.sum(m.center_of_gravity(n_points=10)**2)
        cog_pens.append(cog_pen_i.detach())

        # scalar projection of energy gradient onto model output direction
        s_i = torch.dot(x_i.reshape(-1), g[i].reshape(-1))
        total_scalar = s_i + REG_MULTPLIER / X.shape[0] * reg_i + COG_MULTPLIER / X.shape[0] * cog_pen_i

        # Compute gradient of that scalar w.r.t. model parameters
        grads = torch.autograd.grad(
            total_scalar,
            list(m.parameters()),
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )

        # Accumulate parameter grads manually (detach to save memory)
        for p, gp in zip(m.parameters(), grads):
            if gp is None:
                continue
            if p.grad is None:
                p.grad = gp.detach().clone()
            else:
                p.grad.add_(gp.detach())

    reg_mean = torch.stack(regs).mean()
    cog_pen_mean = torch.stack(cog_pens).mean()
    loss_scalar = E_val + REG_MULTPLIER * reg_mean + COG_MULTPLIER * cog_pen_mean

    return loss_scalar



# === Optimization Loop ===
for step in tqdm(range(N_ITER)):
    # Evaluate current energy for logging/plotting
    with torch.no_grad():
        X = torch.stack([compute_x(m) for m in models])
        energy_now = repulsion_energy(X, s=REPULSION)

    # Plot
    print(f"Iter {step}: energy={energy_now.item():.6f}")
    
    # Plot the diagram
    color = pointwise_repulsion_energy(X, s=REPULSION).detach().cpu().numpy()    
    plot_diagram(X.cpu().numpy(), os.path.join(OUTPUT_FOLDER, f'{step}.png'), color=color)

    if step % PLOT_EVERY == 0:
        plot_all_shapes(models, X.cpu().numpy(), OUTPUT_FOLDER, video=False)

    # Take a step
    loss_val = optimizer.step(closure)