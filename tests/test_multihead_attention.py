import numpy as np
import pytest

from nnetflow.engine import Tensor
from nnetflow.layers import MultiHeadAttention


def test_attention_output_shape_and_causal_mask():
    np.random.seed(12)
    layer = MultiHeadAttention(4, 4, 2, dropout=0.0, causal=True, dtype=np.float64)
    x = Tensor(np.random.randn(2, 5, 4), requires_grad=True, dtype=np.float64)
    out = layer(x)
    assert out.shape == (2, 5, 4)
    assert np.all(np.isfinite(out.data))
    out.sum().backward()
    assert x.grad.shape == x.shape
    projection_parameters = [
        layer.W_query.weight,
        layer.W_key.weight,
        layer.W_value.weight,
        layer.out_proj.weight,
    ]
    assert all(parameter.grad is not None for parameter in projection_parameters)


def test_attention_noncausal_and_cache_reset():
    layer = MultiHeadAttention(4, 4, 2, dropout=0.0, causal=False, dtype=np.float64)
    x = Tensor(np.ones((1, 3, 4)), dtype=np.float64)
    assert layer(x).shape == (1, 3, 4)
    first = layer(x[:, :1], use_cache=True)
    second = layer(x[:, 1:2], use_cache=True)
    assert first.shape == second.shape == (1, 1, 4)
    assert layer.cache_k.shape[1] == 2
    layer.reset_cache()
    assert layer.cache_k is None and layer.cache_v is None


def test_attention_eval_propagates_to_nested_dropout():
    layer = MultiHeadAttention(4, 4, 2, dropout=0.5, causal=False)
    assert layer.training is True
    assert layer.dropout_layer.training is True
    layer.eval()
    assert layer.training is False
    assert layer.dropout_layer.training is False
    layer.train()
    assert layer.training is True
    assert layer.dropout_layer.training is True


def test_attention_requires_divisible_output_dimension():
    with pytest.raises(ValueError):
        MultiHeadAttention(4, 5, 2)