"""An example file for the numerical approximation 
of the isoperimetric inequality for convex bodies
with the symmetries of the equilateral triangle.
"""

import torch
import shutil
import os
from math import sqrt

from shapes.invertible_nn import ConvexDiffeo
from shapes.plot_utils import plot_shape

# Parameters
DIM = 2
N_QUAD = 100_000
OUTPUT_FOLDER = 'res/isoperimetric_symmetries'


# Define the symmetries
def id(x):
    return x


def rot_1(x):
    x_cpy = x.clone()            # make a copy
    x_cpy[:, 0] = -0.5*x[:, 0] - sqrt(3)/2*x[:,1]
    x_cpy[:, 1] = sqrt(3)/2*x[:, 0] - 0.5*x[:,1]
    return x_cpy

def rot_2(x):
    x_cpy = x.clone()            # make a copy
    x_cpy[:, 0] = -0.5*x[:, 0] + sqrt(3)/2*x[:,1]
    x_cpy[:, 1] = -sqrt(3)/2*x[:, 0] - 0.5*x[:,1]
    return x_cpy

symmetries = [id, rot_1, rot_2]


# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')

# Create the output foler
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Create the model and pass the symmetries
model = ConvexDiffeo(input_size=DIM, n_unit=20, symmetries=symmetries).to(device)

# Set up optimizer
optimizer = torch.optim.LBFGS(model.parameters(), lr=.02)

# For the l-bfgs optimizer
def closure():
    optimizer.zero_grad()

    per = model.perimeter(n_points=N_QUAD)
    vol = model.volume(n_points=N_QUAD)
    loss = per/vol**((DIM-1)/DIM)

    loss.backward()

    return loss

# Optimize
for step in range(1_000):
    plot_shape(model, output=os.path.join(OUTPUT_FOLDER, f'{step}.png'))
    loss = optimizer.step(closure)
    print(f'Loss = {loss.item()}')
