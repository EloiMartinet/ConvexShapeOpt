# Convex shape optimization using neural networks

This repository implements a way to perform shape optimization among convex sets using neural networks. The approac is then used to numerically explore *Blashcke-Santalò diagrams*, i.e. the image of shape functionals.

This library is implemented almost fully in PyTorch, with few dependencies. 

Disclaimer: this repository is linked to the paper [link]. One should first read the paper before trying to run this code.


## Installation

1. Clone the repository
```bash
git clone https://github.com/EloiMartinet/ConvexShapeOpt
```

2. Go to the project folder
```bash
cd ConvexShapeOpt
```

3. Install dependencies
```bash
pip install matplotlib numpy pyvista fiblat
```
For the PyTorch install, please consult https://pytorch.org/get-started/locally/ to get the more suitable installation for you. Cuda is recommended for running the Blaschke-Santalò-related scripts, but the example scripts (simple shape optimization problems) runs well on CPU.

Once you installed everything, run
```bash
pip install -e .
```


## Usage

In the main folder, execute
```bash
python3 examples/isoperimetric.py
```
This code simulates the minimization of perimeter of a convex set of unit measure. If the installation went ok, you should see the shapes at each iteration under the folder `res/isoperimetric`, that hopefully converges to a ball. I encourage you to open the file to see how simple it is ! You can also, for instance, change the variable `DIM` from `2`to `3` to perform 3d shape optimization.

The Blaschke-Santalò scripts are under the `scripts/` folder. For instance, you can run
```bash
python3 scripts/diagram_APW_2d.py
```
which run the diagram of volume, perimeter and moment of inertia. The picture of the diagram at each iteration should be under `res/APW_2d/`. The images of the shapes and a csv file with the corresponding values are under the folder `res/APW_2d/shapes/`, while the models are saved under `res/APW_2d/models/`. 


In order to visualize the convex sets on the diagram, you can use the `plot_diagram.html` file. First, from the root folder, execute
```bash
python3 -m http.server 8000
```
then open your browser at http://localhost:8000/plot_diagram.html?folder=res/APW_2d/shapes.


## Code structure

The main file is `src/shapes/invertible_nn.py`. It implements the main model, `ConvexDiffeo`, of which the `fowrad` method is a diffeomorphism from the ball to a convex set. It relies on the gauge function of a smoothed polygon, implemented in the class `LSEGauge` in the file `src/shapes/gauge_functions.py`

It also implements various geometric and spectral shape quantities as:
- Volume
- Perimeter
- Normal vector
- Mean curvature
- Willmore energy
- Integral mean curvature
- Torsional rigidity
- Dirichlet and Neumann eigenvalues

The computation of these quantites takes great advantage of PyTorch's automatic differentiation. Please consult the paper for more information.

WARNING: Use double precision for computing PDE-related quantities.

The file `src/shapes/repulsion_energy.py` only implements a simple Riesz potential, that is used in all the scripts. The `src/shapes/plot_utils.py` file implements some plotting routines in 2 and 3 dimensions.


## Documentation


## Cite


## License

This project is under the MIT License