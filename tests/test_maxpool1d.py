import numpy as np
import torch

from nnetflow.engine import Tensor
from nnetflow.layers import MaxPool1d


def test_maxpool1d_forward_and_backward_match_torch():
    data = np.array([[[1.0, 4.0, 2.0, 3.0, 5.0]]])
    layer = MaxPool1d(3, stride=2, padding=1)
    x = Tensor(data.copy(), requires_grad=True)
    out = layer(x)
    out.sum().backward()

    torch_x = torch.tensor(data, requires_grad=True)
    torch_out = torch.nn.functional.max_pool1d(torch_x, 3, stride=2, padding=1)
    torch_out.sum().backward()
    assert out.shape == tuple(torch_out.shape)
    assert np.allclose(out.data, torch_out.detach().numpy())
    assert np.allclose(x.grad, torch_x.grad.numpy())


def test_maxpool1d_defaults_stride_to_kernel_size():
    layer = MaxPool1d(2)
    assert layer(Tensor(np.ones((1, 1, 4)))).shape == (1, 1, 2)