# Usage

from chainflow.nn import MLP, SGD, MSELoss
from chainflow.engine import Tensor

model = MLP(nin=3, nouts=[8, 2])
# ...
