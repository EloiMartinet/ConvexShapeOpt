"""An example file for the numerical exploration of the Blasche-Santalo
diagram of Volume, Perimeter and Moment of inertia. 

The purpose of this file is to demonstrate how the ones in the script/ folder works,
while making it simpler and lighter. Hence, it should be able to run on a
standard laptop in a few hours"""

import torch
import shutil
import os

from shapes.invertible_nn import ConvexDiffeo
from shapes.plot_utils import plot_all_shapes, plot_diagram
from shapes.repulsion_energy import repulsion_energy, pointwise_repulsion_energy

# PARAMETERS
N_MODELS = 50               # Number of neural nets (= number of shapes)
N_UNITS = 30                # Size of each NN
OUTPUT_FOLDER = 'res/APW_2d_example'
N_ITER = 1000               # Maximal number of iterations
BARRIER_MULTPLIER = 5e-3    # The regularization multiplier
N_POINTS_QUAD = 50_000      # Number of points for the quadratures
N_POINTS_REG = 100          # Number of points for the quadratures
PLOT_EVERY = 10             # Plot all the pictures and save data every
REPULSION = 2.5             # Electronic repulsion parameter

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

# Collect parameters from all models to pass to the optimizer
all_parameters = []
for model in models:
    all_parameters += list(model.parameters())

# Set up the optimizer
optimizer = torch.optim.LBFGS(
    all_parameters,
    lr=.005,
    history_size=100,
    max_iter=20,
    line_search_fn="strong_wolfe"
)

# The functions of interest: volume, perimeter, momentum
def compute_x(model):
    vol = model.volume(N_POINTS_QUAD)
    per = model.perimeter(N_POINTS_QUAD)
    moi = model.moment_of_inertia(N_POINTS_QUAD)

    x1 = (0.5/torch.pi)*(vol**2)/moi
    x2 = 4*torch.pi*vol/(per**2)
    x = torch.hstack([x1.reshape(-1), x2.reshape(-1)])

    return x

# The closure for the lbfgs optimizer. The closure in
# the files inside of the script/ folder is more complex,
# in order to be more emory-friendly
def closure():
    optimizer.zero_grad()

    # Compute the global diagram (no grad)
    X_list = [compute_x(m) for m in models]
    X = torch.stack(X_list, dim=0).to(device)
    E = repulsion_energy(X, s=REPULSION)

    # Compute the jacobian regularizer
    regs = [m.jacobian_regularizer(N_POINTS_REG) for m in models]
    regs = torch.stack(regs, dim=0).to(device)

    # Compute the loss and compute the gradients
    loss = E + BARRIER_MULTPLIER * regs.mean()
    loss.backward()

    return loss


# Optimization Loop
for step in range(N_ITER):
    # Evaluate current energy for logging/plotting
    X = torch.stack([compute_x(m).detach() for m in models])
    energy_now = repulsion_energy(X, s=REPULSION)

    # Plot
    print(f"Iter {step}: energy={energy_now.item():.6f}")
    plot_diagram(X.cpu().numpy(), os.path.join(OUTPUT_FOLDER, f'{step}.png'))

    if step % PLOT_EVERY == 0:
        plot_all_shapes(models, X.cpu().numpy(), OUTPUT_FOLDER, video=False)

    # Take a step
    loss_val = optimizer.step(closure)