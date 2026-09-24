import numpy as np

from nnetflow.engine import Tensor
from nnetflow.layers import Embedding


def test_embedding_lookup_matches_weight_rows_and_shape():
    layer = Embedding(6, 4, dtype=np.float64)
    indices = np.array([[0, 3], [5, 1]])
    out = layer(indices)
    assert out.shape == (2, 2, 4)
    assert np.allclose(out.data, layer.weight.data[indices])


def test_embedding_backward_accumulates_duplicate_indices():
    layer = Embedding(5, 3, dtype=np.float64)
    out = layer(np.array([2, 2, 4]))
    out.sum().backward()
    assert np.allclose(layer.weight.grad[2], 2.0)
    assert np.allclose(layer.weight.grad[4], 1.0)
    assert np.allclose(layer.weight.grad[[0, 1, 3]], 0.0)


def test_embedding_supports_scalar_and_slice_indices():
    layer = Embedding(5, 2)
    assert layer(1).shape == (2,)
    assert layer(slice(1, 4)).shape == (3, 2)