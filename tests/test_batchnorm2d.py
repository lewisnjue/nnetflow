import numpy as np
import torch

from nnetflow.engine import Tensor
from nnetflow.layers import BatchNorm2d


def test_batchnorm2d_training_matches_torch():
    np.random.seed(4)
    data = np.random.randn(4, 3, 2, 2)
    layer = BatchNorm2d(3, momentum=0.1, eps=1e-5)
    x = Tensor(data.copy(), requires_grad=True)
    out = layer(x)
    out.sum().backward()

    reference = torch.nn.BatchNorm2d(3, momentum=0.1, eps=1e-5, dtype=torch.float64)
    torch_x = torch.tensor(data, requires_grad=True)
    torch_out = reference(torch_x)
    torch_out.sum().backward()

    assert np.allclose(out.data, torch_out.detach().numpy(), atol=1e-6)
    assert np.allclose(x.grad, torch_x.grad.numpy(), atol=1e-5)
    assert np.allclose(layer.running_mean.data.reshape(-1), reference.running_mean.numpy())
    assert np.allclose(layer.running_var.data.reshape(-1), reference.running_var.numpy())


def test_batchnorm2d_eval_uses_running_statistics():
    layer = BatchNorm2d(2)
    layer.running_mean.data[...] = [[[1]], [[2]]]
    layer.running_var.data[...] = [[[4]], [[9]]]
    layer.training = False
    data = np.array([[[[3.0]], [[5.0]]]])
    out = layer(Tensor(data))
    expected = (data - layer.running_mean.data) / np.sqrt(layer.running_var.data + layer.eps)
    assert np.allclose(out.data, expected)


def test_batchnorm2d_rejects_invalid_rank():
    with np.testing.assert_raises(AssertionError):
        BatchNorm2d(2)(Tensor(np.ones((2, 2, 3))))