"""Tests for dtype support across Tensor and Layers."""
import numpy as np
from nnetflow.engine import Tensor
from nnetflow.layers import Linear, BatchNorm1d, LayerNorm


def test_tensor_default_dtype_is_float64():
    t = Tensor([1.0, 2.0, 3.0])
    assert t.dtype == np.float64


def test_tensor_preserve_requested_dtype():
    a = np.array([1.0, 2.0], dtype=np.float32)
    t = Tensor(a, dtype=np.float32)
    assert t.dtype == np.float32

    b = np.array([1.0, 2.0], dtype=np.float64)
    t2 = Tensor(b, dtype=np.float64)
    assert t2.dtype == np.float64


def test_operation_dtype_propagation_float32():
    a = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    b = Tensor(np.array([3.0, 4.0], dtype=np.float32))
    c = a + b
    assert c.dtype == np.float32


def test_linear_forward_preserves_dtype():
    layer = Linear(3, 2, dtype=np.float32)
    x = Tensor(np.random.randn(4, 3).astype(np.float32))
    out = layer(x)
    assert out.dtype == np.float32


def test_batchnorm_dtype_and_running_stats():
    bn = BatchNorm1d(5)
    x = Tensor(np.random.randn(3, 5).astype(np.float32))
    out = bn(x)
    assert out.dtype == np.float32
    assert bn.running_mean.data.dtype == np.float32


def test_layernorm_preserves_dtype():
    ln = LayerNorm(dim=6)
    x = Tensor(np.random.randn(2, 6).astype(np.float32))
    out = ln(x)
    assert out.dtype == np.float32

