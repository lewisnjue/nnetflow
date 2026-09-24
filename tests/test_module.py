import numpy as np

from nnetflow.layers import Linear
from nnetflow.module import Module
from nnetflow.optim import Adam
from nnetflow.engine import Tensor


class TwoLayerModel(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(3, 4)
        self.linear2 = Linear(4, 1)

    def forward(self, x):
        return self.linear2(self.linear1(x))


def test_parameters_returns_only_tensor_parameters():
    model = TwoLayerModel()
    parameters = model.parameters()
    assert len(parameters) == 4
    assert all(isinstance(parameter, Tensor) for parameter in parameters)
    assert all(parameter.requires_grad for parameter in parameters)


def test_adam_accepts_nested_module_parameters():
    model = TwoLayerModel()
    optimizer = Adam(model.parameters(), lr=0.001)
    x = Tensor(np.ones((2, 3), dtype=np.float32), requires_grad=False)
    model(x).sum().backward()
    optimizer.step()