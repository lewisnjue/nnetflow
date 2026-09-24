import numpy as np
import torch

from nnetflow.engine import Tensor
from nnetflow.layers import MaxPool2d


def test_maxpool2d_forward_and_backward_match_torch():
    data = np.array([[[[1.0, 4.0, 2.0], [3.0, 5.0, 0.0], [2.0, 1.0, 6.0]]]])
    layer = MaxPool2d(2, stride=1, padding=1)
    x = Tensor(data.copy(), requires_grad=True)
    out = layer(x)
    out.sum().backward()

    torch_x = torch.tensor(data, requires_grad=True)
    torch_out = torch.nn.functional.max_pool2d(torch_x, 2, stride=1, padding=1)
    torch_out.sum().backward()
    assert out.shape == tuple(torch_out.shape)
    assert np.allclose(out.data, torch_out.detach().numpy())
    assert np.allclose(x.grad, torch_x.grad.numpy())


def test_maxpool2d_accepts_tuple_kernel_and_stride():
    layer = MaxPool2d((2, 3), stride=(1, 2))
    out = layer(Tensor(np.ones((2, 1, 4, 6))))
    assert out.shape == (2, 1, 3, 2)