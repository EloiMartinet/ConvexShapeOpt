"""A simple script for the minimization of the k-th Dirichlet 
eigenvalue among convex sets of unit measure.

One can compare the obtained results with 
https://arxiv.org/pdf/1809.00254, Figure 1
"""

import torch
import shutil
import os
from tqdm import tqdm

from shapes.invertible_nn import ConvexDiffeo
from shapes.plot_utils import plot_shape


# Important for eigenvalue problems
torch.set_default_dtype(torch.float64)

# Parameters
NUM_EV = 2
N_UNIT = 30
N_QUAD = 10_000
N_SOURCES = 500
OUTPUT_FOLDER = 'res/dirichlet'

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')

# Create the output foler
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Create the model
model = ConvexDiffeo(input_size=2, n_unit=N_UNIT)

# Set up the optimizer
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=0.01,
    history_size=100,
    max_iter=20,
    line_search_fn="strong_wolfe"
)

# The closure for lbfgs
def closure():
    optimizer.zero_grad()
    
    vol = model.volume(n_points=N_QUAD)
    
    ev, _, _ = model.dirichlet_eigenvalues(
        n_ev=NUM_EV+1,
        n_sources = N_SOURCES,
        n_quad_points = N_QUAD,
        normalize=True
    )
    loss = ev[NUM_EV]
    loss.backward()

    return loss

# Optimization loop
for step in tqdm(range(1000)):
    # Plot
    plot_shape(model, output=os.path.join(OUTPUT_FOLDER, f'{step}.png'))

    # Compute the quantites for printing
    loss = closure()

    # Print history
    print(f'Iter {step} :\tloss = {loss.item()}')

    # scheduler.step(loss)
    optimizer.step(closure)


