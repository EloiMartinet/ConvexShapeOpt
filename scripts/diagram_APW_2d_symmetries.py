"""Compute the Blaschke-Santalo diagram of Volume, Perimeter
and Momentum in 2D for doubly symmetric convex bodies
"""

import torch
from tqdm import tqdm
import shutil
import os

from shapes.invertible_nn import ConvexDiffeo
from shapes.plot_utils import plot_all_shapes, plot_diagram
from shapes.repulsion_energy import repulsion_energy, pointwise_repulsion_energy

N_MODELS = 200
N_UNITS = 30
OUTPUT_FOLDER = 'res/APW_2d_symmetries'
N_ITER = 1000
BARRIER_MULTPLIER = 1e-1
N_POINTS_QUAD = 50_000     # Number of points for the quadratures
N_POINTS_REG = 2_000     # Number of points for the quadratures
PLOT_EVERY = 10         # Plot all the pictures and save data every
REPULSION = 2


# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')

# Create the output foler
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Define the symmetries
def id(x):
    return x


def opposite(x):
    return -x


def x_reflexion(x):
    x_reflected = x.clone()            # make a copy
    x_reflected[:, 0] = -x_reflected[:, 0]
    return x_reflected


def y_reflexion(x):
    x_reflected = x.clone()            # make a copy
    x_reflected[:, 1] = -x_reflected[:, 1]
    return x_reflected


# symmetries = [id, x_reflexion, y_reflexion]
symmetries = [id, x_reflexion, y_reflexion, opposite]

# Draw a bunch of models
models = [
    ConvexDiffeo(input_size=2, n_unit=N_UNITS, symmetries=symmetries).to(device)
    for _ in range(N_MODELS)
]



# Collect parameters from all models
all_parameters = []
for model in models:
    all_parameters += list(model.parameters())

# Set up the optimizer
optimizer = torch.optim.LBFGS(
    all_parameters,
    lr=.02,
    history_size=100,
    max_iter=20,
    line_search_fn="strong_wolfe"
)


def compute_x(model):
    vol = model.volume(N_POINTS_QUAD).to(device)
    per = model.perimeter(N_POINTS_QUAD).to(device)
    moi = model.moment_of_inertia(N_POINTS_QUAD).to(device)
    
    x1 = (0.5/torch.pi)*(vol**2)/moi
    x2 = 4*torch.pi*vol/(per**2)
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
    for i, m in enumerate(models):
        # compute_x(m) builds a small, temporary graph for that model only
        x_i = compute_x(m)
        reg_i = m.jacobian_regularizer(N_POINTS_REG)
        regs.append(reg_i.detach())

        # scalar projection of energy gradient onto model output direction
        s_i = torch.dot(x_i.reshape(-1), g[i].reshape(-1))
        total_scalar = s_i + BARRIER_MULTPLIER / X.shape[0] * reg_i

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
    loss_scalar = E_val + BARRIER_MULTPLIER * reg_mean

    return loss_scalar



# === Optimization Loop ===
for step in tqdm(range(N_ITER)):
    # Evaluate current energy for logging/plotting
    with torch.no_grad():
        X = torch.stack([compute_x(m) for m in models])
        energy_now = repulsion_energy(X, s=REPULSION)

    # Plot
    print(f"Iter {step}: energy={energy_now.item():.6f}")
    
    color = pointwise_repulsion_energy(X, s=REPULSION).detach().cpu().numpy()
    plot_diagram(X.cpu().numpy(), os.path.join(OUTPUT_FOLDER, f'{step}.png'), color=color)

    if step % PLOT_EVERY == 0:
        plot_all_shapes(models, X.cpu().numpy(), OUTPUT_FOLDER, video=False)

    # Take a step
    loss_val = optimizer.step(closure)