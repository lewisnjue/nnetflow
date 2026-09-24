import numpy as np

from nnetflow.engine import Tensor
from nnetflow.layers import GlobalAveragePool2d


def test_global_averagepool2d_forward_and_backward():
    data = np.arange(24, dtype=np.float64).reshape(2, 3, 2, 2)
    layer = GlobalAveragePool2d()
    x = Tensor(data.copy(), requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert out.shape == (2, 3, 1, 1)
    assert np.allclose(out.data, data.mean(axis=(2, 3), keepdims=True))
    assert np.allclose(x.grad, np.full_like(data, 0.25))


def test_global_averagepool2d_rejects_non_image_input():
    with np.testing.assert_raises(AssertionError):
        GlobalAveragePool2d()(Tensor(np.ones((2, 3, 4))))