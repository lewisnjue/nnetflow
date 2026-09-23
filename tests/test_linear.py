import pytest
import torch
import numpy as np
from nnetflow.engine import Tensor
from nnetflow.layers import Linear


def _torch_linear_ref(x_np, w_np, b_np=None):
    """Manual torch reference using nnetflow's (in_features, out_features)
    weight convention: y = x @ w (+ b). Note this is NOT torch.nn.Linear,
    whose weight is (out_features, in_features) and does x @ w.T + b."""
    x_tt = torch.tensor(x_np, requires_grad=True)
    w_tt = torch.tensor(w_np, requires_grad=True)
    if b_np is not None:
        b_tt = torch.tensor(b_np, requires_grad=True)
        y_tt = x_tt @ w_tt + b_tt
        return x_tt, w_tt, b_tt, y_tt
    y_tt = x_tt @ w_tt
    return x_tt, w_tt, None, y_tt


class TestLinear:
    # ---------------------------------------------------------------
    # Construction / shapes
    # ---------------------------------------------------------------

    def test_weight_and_bias_shapes(self):
        layer = Linear(4, 3, bias=True, dtype=np.float64)
        assert layer.weight.shape == (4, 3)
        assert layer.bias.shape == (1, 3)
        assert layer.weight.requires_grad is True
        assert layer.bias.requires_grad is True

    def test_bias_initialized_to_zero(self):
        layer = Linear(5, 2, bias=True, dtype=np.float64)
        assert np.all(layer.bias.data == 0.0)

    def test_no_bias_has_no_bias_attribute(self):
        layer = Linear(4, 3, bias=False, dtype=np.float64)
        assert layer.has_bias is False
        assert not hasattr(layer, "bias")

    def test_weight_dtype_follows_constructor_arg(self):
        layer = Linear(4, 3, dtype=np.float32)
        assert layer.weight.dtype == np.float32
        assert layer.bias.dtype == np.float32


    def test_repr_contains_dimensions(self):
        layer = Linear(4, 3)
        r = repr(layer)
        assert "4" in r and "3" in r

    # ---------------------------------------------------------------
    # Forward correctness
    # ---------------------------------------------------------------

    def test_forward_matches_manual_matmul_with_bias(self):
        layer = Linear(4, 3, bias=True, dtype=np.float64)
        x_np = np.random.randn(6, 4)
        x = Tensor(x_np, requires_grad=True, dtype=np.float64)
        out = layer.forward(x)
        expected = x_np @ layer.weight.data + layer.bias.data
        assert out.shape == (6, 3)
        assert np.allclose(out.data, expected)

    def test_forward_matches_manual_matmul_without_bias(self):
        layer = Linear(4, 3, bias=False, dtype=np.float64)
        x_np = np.random.randn(6, 4)
        x = Tensor(x_np, requires_grad=True, dtype=np.float64)
        out = layer.forward(x)
        expected = x_np @ layer.weight.data
        assert np.allclose(out.data, expected)

    def test_forward_supports_extra_leading_batch_dims(self):
        layer = Linear(4, 3, bias=True, dtype=np.float64)
        x_np = np.random.randn(2, 5, 4)
        x = Tensor(x_np, requires_grad=True, dtype=np.float64)
        out = layer.forward(x)
        expected = x_np @ layer.weight.data + layer.bias.data
        assert out.shape == (2, 5, 3)
        assert np.allclose(out.data, expected)

    def test_input_feature_mismatch_raises(self):
        layer = Linear(4, 3, dtype=np.float64)
        x = Tensor(np.random.randn(6, 5), dtype=np.float64)
        with pytest.raises(AssertionError):
            layer.forward(x)

    def test_dtype_mismatch_between_input_and_layer_raises(self):
        layer = Linear(4, 3) 
        x = Tensor.randn(6, 4)  
        with pytest.raises(ValueError):
            layer.forward(x)

        layer32 = Linear(4, 3, dtype=np.float32)
        x32 = Tensor.randn(6, 4)
        out = layer32.forward(x32)
        assert out.shape == (6, 3)

    # ---------------------------------------------------------------
    # Backward correctness (compared against a manual torch reference)
    # ---------------------------------------------------------------

    def test_backward_gradients_with_bias(self):
        layer = Linear(4, 3, bias=True, dtype=np.float64)
        x_np = np.random.randn(6, 4)
        x = Tensor(x_np, requires_grad=True, dtype=np.float64)

        out = layer.forward(x)
        loss = out.sum()
        loss.backward()

        x_tt, w_tt, b_tt, y_tt = _torch_linear_ref(x_np, layer.weight.data, layer.bias.data)
        y_tt.sum().backward()

        assert np.allclose(out.data, y_tt.detach().numpy())
        assert np.allclose(x.grad, x_tt.grad.numpy())
        assert np.allclose(layer.weight.grad, w_tt.grad.numpy())
        assert np.allclose(layer.bias.grad, b_tt.grad.numpy())

    def test_backward_gradients_without_bias(self):
        layer = Linear(4, 3, bias=False, dtype=np.float64)
        x_np = np.random.randn(6, 4)
        x = Tensor(x_np, requires_grad=True, dtype=np.float64)

        out = layer.forward(x)
        loss = out.sum()
        loss.backward()

        x_tt, w_tt, _, y_tt = _torch_linear_ref(x_np, layer.weight.data)
        y_tt.sum().backward()

        assert np.allclose(x.grad, x_tt.grad.numpy())
        assert np.allclose(layer.weight.grad, w_tt.grad.numpy())

    def test_bias_gradient_is_unbroadcast_correctly(self):
        layer = Linear(4, 3, bias=True, dtype=np.float64)
        x = Tensor(np.random.randn(10, 4), requires_grad=True, dtype=np.float64)
        out = layer.forward(x)
        out.sum().backward()
        assert layer.bias.grad.shape == (1, 3)
        # d(sum(x@w + b))/db = sum over batch of ones = batch_size, per column
        assert np.allclose(layer.bias.grad, np.full((1, 3), 10.0))

    def test_weight_grad_accumulates_across_forward_passes(self):
        layer = Linear(4, 3, bias=True, dtype=np.float64)
        x = Tensor(np.random.randn(6, 4), requires_grad=True, dtype=np.float64)

        layer.forward(x).sum().backward()
        first_grad = layer.weight.grad.copy()

        layer.forward(x).sum().backward()
        assert np.allclose(layer.weight.grad, first_grad * 2)

        layer.weight.zero_grad()
        assert np.all(layer.weight.grad == 0)