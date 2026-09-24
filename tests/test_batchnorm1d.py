import numpy as np
import torch

from nnetflow.engine import Tensor
from nnetflow.layers import BatchNorm1d


def test_batchnorm1d_training_2d_matches_torch():
    np.random.seed(3)
    data = np.random.randn(5, 3)
    layer = BatchNorm1d(3, momentum=0.1, eps=1e-5)
    x = Tensor(data.copy(), requires_grad=True)
    out = layer(x)
    out.sum().backward()

    reference = torch.nn.BatchNorm1d(3, momentum=0.1, eps=1e-5, dtype=torch.float64)
    torch_x = torch.tensor(data, requires_grad=True)
    torch_out = reference(torch_x)
    torch_out.sum().backward()

    assert np.allclose(out.data, torch_out.detach().numpy(), atol=1e-6)
    assert np.allclose(x.grad, torch_x.grad.numpy(), atol=1e-5)
    assert np.allclose(layer.running_mean.data, reference.running_mean.numpy())
    assert np.allclose(layer.running_var.data, reference.running_var.numpy())


def test_batchnorm1d_supports_3d_input_and_eval_mode():
    layer = BatchNorm1d(2)
    x = Tensor(np.arange(12, dtype=np.float64).reshape(2, 2, 3))
    out = layer(x)
    assert out.shape == x.shape
    layer.training = False
    eval_out = layer(x)
    expected = (x.data - layer.running_mean.data.reshape(1, 2, 1)) / np.sqrt(
        layer.running_var.data.reshape(1, 2, 1) + layer.eps
    )
    assert np.allclose(eval_out.data, expected)


def test_batchnorm1d_rejects_invalid_rank_and_channels():
    layer = BatchNorm1d(2)
    with np.testing.assert_raises(AssertionError):
        layer(Tensor(np.ones((2, 2, 3, 1))))
    with np.testing.assert_raises(AssertionError):
        layer(Tensor(np.ones((2, 3))))