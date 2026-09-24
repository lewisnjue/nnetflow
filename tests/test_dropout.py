import numpy as np
import pytest

from nnetflow.engine import Tensor
from nnetflow.layers import Dropout


def test_dropout_training_uses_inverted_scaling_and_backward_mask():
    np.random.seed(10)
    layer = Dropout(0.5)
    x_data = np.ones((20, 20), dtype=np.float64)
    x = Tensor(x_data, requires_grad=True)
    out = layer(x)
    assert set(np.unique(out.data)).issubset({0.0, 2.0})
    out.sum().backward()
    assert np.array_equal(x.grad, out.data)


def test_dropout_eval_is_identity():
    layer = Dropout(0.5)
    layer.training = False
    x = Tensor(np.arange(6, dtype=np.float64).reshape(2, 3), requires_grad=True)
    out = layer(x)
    assert out is x


def test_dropout_rejects_invalid_probability():
    with pytest.raises(AssertionError):
        Dropout(-0.1)
    with pytest.raises(AssertionError):
        Dropout(1.0)