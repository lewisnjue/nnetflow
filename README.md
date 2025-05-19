# nnetflow

A minimal neural network framework with autodiff, inspired by micrograd and pytorch.

## Installation

pip install nnetflow

## Usage

from nnetflow.nn import MLP, SGD, MSELoss
from nnetflow.engine import Tensor

model = MLP(nin=3, nouts=[8, 2])
# ...

See the docs/ folder for more details.