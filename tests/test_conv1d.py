import numpy as np
import torch

from nnetflow.engine import Tensor
from nnetflow.layers import Conv1d


def test_conv1d_forward_and_backward_match_torch():
    np.random.seed(1)
    layer = Conv1d(2, 3, kernel_size=3, stride=2, padding=1, dtype=np.float64)
    x_data = np.random.randn(2, 2, 6)
    x = Tensor(x_data.copy(), requires_grad=True, dtype=np.float64)
    out = layer(x)
    out.sum().backward()

    torch_layer = torch.nn.Conv1d(2, 3, 3, stride=2, padding=1, dtype=torch.float64)
    torch_layer.weight.data = torch.tensor(layer.weight.data)
    torch_layer.bias.data = torch.tensor(layer.bias.data.reshape(-1))
    torch_x = torch.tensor(x_data, requires_grad=True)
    torch_out = torch_layer(torch_x)
    torch_out.sum().backward()

    assert out.shape == (2, 3, 3)
    assert np.allclose(out.data, torch_out.detach().numpy())
    assert np.allclose(x.grad, torch_x.grad.numpy())
    assert np.allclose(layer.weight.grad, torch_layer.weight.grad.numpy())
    assert np.allclose(layer.bias.grad.reshape(-1), torch_layer.bias.grad.numpy())


def test_conv1d_without_bias_and_repr():
    layer = Conv1d(1, 2, 2, bias=False)
    out = layer(Tensor(np.ones((1, 1, 4))))
    assert out.shape == (1, 2, 3)
    assert layer.bias is None
    assert "Conv1d" in repr(layer)


def test_conv1d_rejects_wrong_rank_and_channels():
    layer = Conv1d(2, 3, 3)
    with np.testing.assert_raises(AssertionError):
        layer(Tensor(np.ones((1, 2, 5, 1))))
    with np.testing.assert_raises(AssertionError):
        layer(Tensor(np.ones((1, 1, 5))))