import numpy as np

from nnetflow.engine import Tensor
from nnetflow.layers import AveragePool2d


def test_averagepool2d_forward_and_backward_without_padding():
    data = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4)
    layer = AveragePool2d(2)
    x = Tensor(data.copy(), requires_grad=True)
    out = layer(x)
    out.sum().backward()
    expected = np.array([[[[2.5, 4.5], [10.5, 12.5]]]])
    assert np.allclose(out.data, expected)
    assert np.allclose(x.grad, np.full_like(data, 0.25))


def test_averagepool2d_count_excluding_padding_matches_valid_window_means():
    data = np.arange(1, 10, dtype=np.float64).reshape(1, 1, 3, 3)
    layer = AveragePool2d(2, stride=1, padding=1, count_include_pad=False)
    out = layer(Tensor(data))
    expected = np.array([[[[1, 1.5, 2.5, 3], [2.5, 3, 4, 4.5], [5.5, 6, 7, 7.5], [7, 7.5, 8.5, 9]]]])
    assert np.allclose(out.data, expected)