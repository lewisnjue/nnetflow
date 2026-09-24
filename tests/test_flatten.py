import numpy as np

from nnetflow.engine import Tensor
from nnetflow.layers import Flatten


def test_flatten_preserves_batch_and_flattens_remaining_dimensions():
    x = Tensor(np.arange(24, dtype=np.float64).reshape(2, 3, 4), requires_grad=True)
    layer = Flatten()
    out = layer(x)
    assert out.shape == (2, 12)
    assert np.array_equal(out.data, x.data.reshape(2, 12))
    out.sum().backward()
    assert np.allclose(x.grad, np.ones_like(x.data))


def test_flatten_repr():
    assert repr(Flatten()) == "Flatten()"