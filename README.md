# Dynamics stuff

## Setting

Navier-Stokes plus `[sin(4y),0,0]` (Kolmogorov) forcing on a 128-by-128 grid.
We are especially interested in defects (points where the flow velocity vanishes, such as centers of vorticies).

## doLearnFlow.py

Tries to reconstruct a flow field from it's defect locations and some information about them.
Fails spectacularly.

## latentRepr

Based on [this paper](https://arxiv.org/pdf/2008.07515.pdf).
Squeezes the vorticity of the flow into a lower dimensional embedding that it learns.
Exciting because this dimension can be as low as 3 to 8.

## symbolic

Does symbolic regression to find a differential equation describing the time evolution of the defect locations.
