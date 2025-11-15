"""Tests for Tensor engine and autograd operations."""
import numpy as np
from nnetflow.engine import Tensor
import pytest


class TestTensorBasic:
    """Test basic Tensor operations."""
    
    def test_tensor_creation(self):
        """Test Tensor creation from various inputs."""
        # From numpy array
        t1 = Tensor(np.array([1.0, 2.0, 3.0]))
        assert t1.shape == (3,)
        
        # From list
        t2 = Tensor([1.0, 2.0, 3.0])
        assert t2.shape == (3,)
        
        # From scalar
        t3 = Tensor(5.0)
        assert t3.shape == ()
    
    def test_tensor_requires_grad(self):
        """Test requires_grad flag."""
        t1 = Tensor([1.0, 2.0], requires_grad=True)
        assert t1.requires_grad
        assert t1.grad is not None
        
        t2 = Tensor([1.0, 2.0], requires_grad=False)
        assert not t2.requires_grad
        assert t2.grad is None

    def test_tensor_shape_size_ndim_numel_dim_helpers(self):
        """Tensor helper APIs delegate correctly to underlying numpy array."""
        data = np.random.randn(2, 3, 4)
        t = Tensor(data)

        # Attribute-style helpers
        assert t.shape == data.shape
        assert t.size == data.size
        assert t.ndim == data.ndim

        # Method-style helpers
        assert t.numel() == data.size
        assert t.dim() == data.ndim


class TestTensorArithmetic:
    """Test Tensor arithmetic operations."""
    
    def test_addition(self):
        """Test addition operation."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        
        c = a + b
        assert np.allclose(c.data, [4.0, 6.0])
    
    def test_multiplication(self):
        """Test multiplication operation."""
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        
        c = a * b
        assert np.allclose(c.data, [8.0, 15.0])
    
    def test_division(self):
        """Test division operation."""
        a = Tensor([6.0, 8.0], requires_grad=True)
        b = Tensor([2.0, 4.0], requires_grad=True)
        
        c = a / b
        assert np.allclose(c.data, [3.0, 2.0])
    
    def test_division_with_scalar(self):
        """Test division with scalar."""
        a = Tensor([6.0, 8.0], requires_grad=True)
        c = a / 2.0
        assert np.allclose(c.data, [3.0, 4.0])


class TestTensorBackward:
    """Test backward pass and gradient computation."""
    
    def test_simple_backward(self):
        """Test simple backward pass."""
        x = Tensor([2.0], requires_grad=True)
        y = x * 3.0
        y.backward()
        
        assert x.grad is not None
        assert np.allclose(x.grad, [3.0])
    
    def test_division_backward_numerator(self):
        """Test division backward pass for numerator."""
        x = Tensor([6.0], requires_grad=True)
        y = Tensor([2.0], requires_grad=False)
        
        z = x / y
        z.backward()
        
        # d/dx (x/y) = 1/y = 1/2 = 0.5
        assert np.allclose(x.grad, [0.5])
        assert y.grad is None
    
    def test_division_backward_denominator(self):
        """Test division backward pass for denominator - CRITICAL TEST."""
        x = Tensor([6.0], requires_grad=False)
        y = Tensor([2.0], requires_grad=True)
        
        z = x / y
        z.backward()
        
        # d/dy (x/y) = -x/y^2 = -6/4 = -1.5
        assert np.allclose(y.grad, [-1.5])
    
    def test_division_backward_both(self):
        """Test division backward pass when both tensors require grad."""
        x = Tensor([6.0], requires_grad=True)
        y = Tensor([2.0], requires_grad=True)
        
        z = x / y
        z.backward()
        
        # d/dx (x/y) = 1/y = 0.5
        assert np.allclose(x.grad, [0.5])
        # d/dy (x/y) = -x/y^2 = -1.5
        assert np.allclose(y.grad, [-1.5])
    
    def test_division_backward_numerical_check(self):
        """Numerical gradient check for division."""
        eps = 1e-5
        
        # Test numerator gradient
        x = Tensor([6.0], requires_grad=True)
        y = Tensor([2.0], requires_grad=False)
        
        z = x / y
        z.backward()
        analytical_grad_x = x.grad[0]
        
        # Numerical gradient
        x_plus = Tensor([6.0 + eps], requires_grad=False)
        z_plus = (x_plus / y).item()
        x_minus = Tensor([6.0 - eps], requires_grad=False)
        z_minus = (x_minus / y).item()
        numerical_grad_x = (z_plus - z_minus) / (2 * eps)
        
        np.testing.assert_allclose(analytical_grad_x, numerical_grad_x, rtol=1e-4)
        
        # Test denominator gradient
        x = Tensor([6.0], requires_grad=False)
        y = Tensor([2.0], requires_grad=True)
        
        z = x / y
        z.backward()
        analytical_grad_y = y.grad[0]
        
        # Numerical gradient
        y_plus = Tensor([2.0 + eps], requires_grad=False)
        z_plus = (x / y_plus).item()
        y_minus = Tensor([2.0 - eps], requires_grad=False)
        z_minus = (x / y_minus).item()
        numerical_grad_y = (z_plus - z_minus) / (2 * eps)
        
        np.testing.assert_allclose(analytical_grad_y, numerical_grad_y, rtol=1e-4)
    
    def test_chain_rule_backward(self):
        """Test backward pass with chain rule."""
        x = Tensor([2.0], requires_grad=True)
        y = x * x  # x^2
        z = y / x  # x^2 / x = x
        
        z.backward()
        # d/dx (x) = 1
        assert np.allclose(x.grad, [1.0])


class TestTensorActivations:
    """Test activation functions."""
    
    def test_relu(self):
        """Test ReLU activation."""
        x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        y = x.relu()
        
        assert np.allclose(y.data, [0.0, 0.0, 1.0])
    
    def test_sigmoid(self):
        """Test sigmoid activation."""
        x = Tensor([0.0], requires_grad=True)
        y = x.sigmoid()
        
        # sigmoid(0) = 0.5
        assert np.allclose(y.data, [0.5], rtol=1e-5)
    
    def test_tanh(self):
        """Test tanh activation."""
        x = Tensor([0.0], requires_grad=True)
        y = x.tanh()
        
        # tanh(0) = 0
        assert np.allclose(y.data, [0.0])


class TestTensorReductions:
    """Test reduction operations."""
    
    def test_sum(self):
        """Test sum operation."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.sum()
        
        assert y.item() == 6.0
    
    def test_mean(self):
        """Test mean operation."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.mean()
        
        assert y.item() == 2.0
    
    def test_sum_backward(self):
        """Test sum backward pass."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.sum()
        y.backward()
        
        # Gradient of sum is all ones
        assert np.allclose(x.grad, [1.0, 1.0, 1.0])

    def test_var_std_match_numpy_all_elements(self):
        """Tensor.var/std should match NumPy sample variance/std over all elements."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x = Tensor(data, requires_grad=False)

        t_var = x.var(axis=None, keepdims=False)
        t_std = x.std(axis=None, keepdims=False)

        np_var = data.var(axis=None, ddof=1)
        np_std = data.std(axis=None, ddof=1)

        np.testing.assert_allclose(t_var.data, np_var, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(t_std.data, np_std, rtol=1e-6, atol=1e-8)

    def test_var_std_axes_and_keepdims(self):
        """Tensor.var/std should support axis and keepdims like NumPy (sample)."""
        data = np.arange(12, dtype=float).reshape(3, 4)
        x = Tensor(data, requires_grad=False)

        # axis=0
        t_var_0 = x.var(axis=0, keepdims=False)
        t_std_0 = x.std(axis=0, keepdims=False)
        np_var_0 = data.var(axis=0, ddof=1)
        np_std_0 = data.std(axis=0, ddof=1)

        np.testing.assert_allclose(t_var_0.data, np_var_0, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(t_std_0.data, np_std_0, rtol=1e-6, atol=1e-8)

        # axis=1 with keepdims
        t_var_1 = x.var(axis=1, keepdims=True)
        t_std_1 = x.std(axis=1, keepdims=True)
        np_var_1 = data.var(axis=1, ddof=1, keepdims=True)
        np_std_1 = data.std(axis=1, ddof=1, keepdims=True)

        np.testing.assert_allclose(t_var_1.data, np_var_1, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(t_std_1.data, np_std_1, rtol=1e-6, atol=1e-8)

    def test_view_behaves_like_reshape_and_supports_tuple(self):
        """Tensor.view should match reshape semantics and accept tuple or varargs."""
        data = np.arange(12, dtype=float)
        x = Tensor(data.copy(), requires_grad=False)

        v1 = x.view(3, 4)
        v2 = x.view((3, 4))
        v3 = x.view(2, -1)

        np.testing.assert_allclose(v1.data, data.reshape(3, 4))
        np.testing.assert_allclose(v2.data, data.reshape(3, 4))
        np.testing.assert_allclose(v3.data, data.reshape(2, -1))


class TestTensorMatmul:
    """Test matrix multiplication."""
    
    def test_matmul(self):
        """Test matrix multiplication."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        
        c = a @ b
        
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(c.data, expected)
    
    def test_matmul_backward(self):
        """Test matrix multiplication backward pass."""
        a = Tensor([[1.0, 2.0]], requires_grad=True)
        b = Tensor([[3.0], [4.0]], requires_grad=True)
        
        c = a @ b
        c.backward()
        
        # Gradient shapes should match
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

