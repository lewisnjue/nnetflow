"""Tests for Tensor engine and autograd operations."""
import numpy as np
import pytest
import torch
from nnetflow.engine import Tensor
from nnetflow.layers import Linear, BatchNorm1d, LayerNorm

class TestTensorBroadcasting:
    """Test whether broadcasting is working as expected"""
    def test_unbroadcast_method(self):
        grad = np.random.randn(3, 4, 5)
        shape = (4, 5)
        # Assuming unbroadcast is a static method exposed on Tensor
        assert Tensor.unbroadcast(grad, shape).shape == shape
        
        grad_bad = np.random.randn(3, 4, 5)
        shape_bad = (2, 7, 2)
        with pytest.raises(ValueError):
            Tensor.unbroadcast(grad_bad, shape_bad)

class TestGrad:
    """Test gradient Tensor properties and zero_grad method."""
    def test_grad_initialization(self):
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert a.requires_grad == True
        assert a.grad is not None
        assert a.grad.shape == a.data.shape
        assert a.grad.dtype == a.data.dtype
        assert a.grad.sum() == 0.0
        assert np.allclose(a.grad, np.zeros_like(a.data))

class TestTensorDtype:
    """Tests for dtype support across Tensor and Layers."""

    def test_astype_returns_new_tensor(self):
        t = Tensor([1.0, 2.0, 3.0], dtype=np.float32)
        t2 = t.astype(np.float64)
        assert t2.dtype == np.float64
        assert t.dtype == np.float32
        assert t2 is not t

    def test_tensor_default_dtype_is_float64(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.dtype == np.float64

    def test_could_not_convert_data_to_tensor(self):
        with pytest.raises(TypeError):
            Tensor("invalid data type")

    def test_tensor_preserve_requested_dtype(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        t = Tensor(a, dtype=np.float32)
        assert t.dtype == np.float32

        b = np.array([1.0, 2.0], dtype=np.float64)
        t2 = Tensor(b, dtype=np.float64)
        assert t2.dtype == np.float64

    def test_operation_dtype_propagation_float32(self):
        a = Tensor(np.array([1.0, 2.0], dtype=np.float32))
        b = Tensor(np.array([3.0, 4.0], dtype=np.float32))
        c = a + b
        assert c.dtype == np.float32

    def test_linear_forward_preserves_dtype(self):
        layer = Linear(3, 2, dtype=np.float32)
        x = Tensor(np.random.randn(4, 3).astype(np.float32))
        out = layer(x)
        assert out.dtype == np.float32

    def test_batchnorm_dtype_and_running_stats(self):
        bn = BatchNorm1d(5)
        x = Tensor(np.random.randn(3, 5).astype(np.float32))
        out = bn(x)
        assert out.dtype == np.float32
        assert bn.running_mean.data.dtype == np.float32

    def test_layernorm_preserves_dtype(self):
        ln = LayerNorm(dim=6)
        x = Tensor(np.random.randn(2, 6).astype(np.float32))
        out = ln(x)
        assert out.dtype == np.float32

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
        assert t.numel == data.size
        assert t.dim == data.ndim


class TestTensorArithmetic:
    """Test Tensor arithmetic operations."""
    
    def test_addition(self):
        """Test addition operation."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        
        c = a + b
        assert np.allclose(c.data, [4.0, 6.0])
    
    def test_addition_backward(self):
        """Test addition backward pass.""" 
        # Reduced size to prevent float drift
        a = np.random.randn(2, 50, 60)
        b = np.random.randn(2, 50, 60)
        
        a_n = Tensor(a, requires_grad=True, dtype=np.float16)
        b_n = Tensor(b, requires_grad=True, dtype=np.float16)
        c_n = a_n + b_n
        loss_n = c_n.sum()
        loss_n.backward()
        
        a_t = torch.tensor(a, requires_grad=True, dtype=torch.float16)
        b_t = torch.tensor(b, requires_grad=True, dtype=torch.float16)
        c_t = a_t + b_t
        loss_t = c_t.sum()
        loss_t.backward()
        
        # Relaxed tolerance for float16
        np.testing.assert_allclose(a_n.grad, a_t.grad.numpy(), rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(b_n.grad, b_t.grad.numpy(), rtol=1e-2, atol=1e-3)
        
        assert c_n.dtype == np.float16
        assert b_n.dtype == np.float16
        assert a_n.dtype == np.float16
        assert loss_n.dtype == np.float16   
        assert loss_t.dtype == torch.float16  
    
    def test_mean_operation(self): 
        """Test mean operation.""" 
        a = np.random.randn(2, 40, 50)   
        a_n = Tensor(a, requires_grad=True, dtype=np.float32) 
        c_n = a_n.mean(axis=2, keepdims=False)
        
        a_t = torch.tensor(a, requires_grad=True, dtype=torch.float32) 
        c_t = a_t.mean(dim=2, keepdim=False) 
        
        np.testing.assert_allclose(c_n.data, c_t.detach().numpy(), rtol=1e-3, atol=1e-4)

    def test_multiplication(self):
        """Test multiplication operation."""
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        
        c = a * b
        assert np.allclose(c.data, [8.0, 15.0])

    def test_multiplication_backward(self):
        a = np.random.randn(2, 40, 50) # Reduced size
        W = np.random.randn(2, 40, 50) 
        
        a_n = Tensor(a, requires_grad=True, dtype=np.float32) 
        W_n = Tensor(W, requires_grad=True, dtype=np.float32) 
        c_n = a_n * W_n 
        loss_n = c_n.sum() 
        loss_n.backward() 
        
        a_t = torch.tensor(a, requires_grad=True, dtype=torch.float32) 
        W_t = torch.tensor(W, requires_grad=True, dtype=torch.float32) 
        c_t = a_t * W_t 
        loss_t = c_t.sum() 
        loss_t.backward() 
        
        np.testing.assert_allclose(a_n.grad, a_t.grad.numpy(), rtol=1e-4) 
        np.testing.assert_allclose(W_n.grad, W_t.grad.numpy(), rtol=1e-4) 
        assert c_n.dtype == np.float32

    def test_tensor_power(self):
        """Test power operation."""
        a = np.random.randn(2, 40, 50) # Reduced size
        a_n = Tensor(a, requires_grad=True, dtype=np.float32) 
        c_n = a_n ** 3.0  
        
        a_t = torch.tensor(a, requires_grad=True, dtype=torch.float32) 
        c_t = a_t ** 3.0 
        
        np.testing.assert_allclose(c_n.data, c_t.detach().numpy(), rtol=1e-4) 
    
    def test_tensor_power_backward(self): 
        """Test power backward operation."""
        a = np.random.randn(2, 40, 50)  
        a_n = Tensor(a, requires_grad=True, dtype=np.float32) 
        c_n = a_n ** 3.0  
        loss_n = c_n.sum() 
        loss_n.backward() 
        
        a_t = torch.tensor(a, requires_grad=True, dtype=torch.float32) 
        c_t = a_t ** 3.0 
        loss_t = c_t.sum() 
        loss_t.backward() 
        
        np.testing.assert_allclose(a_n.grad, a_t.grad.numpy(), rtol=1e-4)
    
    def test_division(self):
        """Test division operation."""
        a = Tensor([6.0, 8.0], requires_grad=True)
        b = Tensor([2.0, 4.0], requires_grad=True)
        
        c = a / b
        assert np.allclose(c.data, [3.0, 2.0])

    def test_division_backward(self): 
        """Test division backward operation."""
        a = np.random.randn(2, 40, 50)  
        b = np.random.randn(2, 40, 50) + 1.0 # Ensure no division by zero
        
        a_n = Tensor(a, requires_grad=True, dtype=np.float32) 
        b_n = Tensor(b, requires_grad=True, dtype=np.float32) 
        c_n = a_n / b_n  
        loss_n = c_n.sum() 
        loss_n.backward() 
        
        a_t = torch.tensor(a, requires_grad=True, dtype=torch.float32) 
        b_t = torch.tensor(b, requires_grad=True, dtype=torch.float32) 
        c_t = a_t / b_t 
        loss_t = c_t.sum() 
        loss_t.backward() 
        
        np.testing.assert_allclose(a_n.grad, a_t.grad.numpy(), rtol=1e-4) 
        np.testing.assert_allclose(b_n.grad, b_t.grad.numpy(), rtol=1e-4)
    
    def test_division_with_scalar(self):
        """Test division with scalar."""
        a = Tensor([6.0, 8.0], requires_grad=True)
        c = a / 2.0
        assert np.allclose(c.data, [3.0, 4.0])

    def test_exponential_operation(self): 
        """Test exponential operation.""" 
        # Reduced size and magnitude to prevent overflow/drift
        a = np.random.randn(2, 40, 50) * 0.1   
        a_n = Tensor(a, requires_grad=True, dtype=np.float32) 
        c_n = a_n.exp() 
        
        a_t = torch.tensor(a, requires_grad=True, dtype=torch.float32) 
        c_t = a_t.exp() 
        
        np.testing.assert_allclose(c_n.data, c_t.detach().numpy(), rtol=1e-3, atol=1e-3)


class TestTensorBackward:
    """Test backward pass and gradient computation."""

    def test_exponential_backward(self): 
        """Test exponential backward operation."""
        # Fixed syntax error here
        # Reduced size to (2, 40, 50)
        a = np.random.randn(2, 40, 50)  
        
        a_n = Tensor(a, requires_grad=True, dtype=np.float16) 
        c_n = a_n.exp() 
        loss_n = c_n.sum() 
        loss_n.backward() 
        
        a_t = torch.tensor(a, requires_grad=True, dtype=torch.float16) 
        c_t = a_t.exp() 
        loss_t = c_t.sum() 
        loss_t.backward() 
        
        # Relaxed tolerance for float16 and exp explosive growth
        np.testing.assert_allclose(a_n.grad, a_t.grad.numpy(), rtol=1e-2, atol=1e-2) 

    def test_addition_backward(self): 
        """Test addition backward operation."""
        a = np.random.randn(2, 40, 50) 
        b = np.random.randn(2, 40, 50) 
        
        a_n = Tensor(a, requires_grad=True, dtype=np.float16) 
        b_n = Tensor(b, requires_grad=True, dtype=np.float16) 
        c_n = a_n + b_n 
        loss_n = c_n.sum() 
        loss_n.backward() 
        
        a_t = torch.tensor(a, requires_grad=True, dtype=torch.float16) 
        b_t = torch.tensor(b, requires_grad=True, dtype=torch.float16) 
        c_t = a_t + b_t 
        loss_t = c_t.sum() 
        loss_t.backward() 
        
        np.testing.assert_allclose(a_n.grad, a_t.grad.numpy(), rtol=1e-2, atol=1e-2) 
        np.testing.assert_allclose(b_n.grad, b_t.grad.numpy(), rtol=1e-2, atol=1e-2)

    def test_sum_backward(self): 
        """Test sum backward operation."""
        a = np.random.randn(2, 40, 50)  
        a_n = Tensor(a, requires_grad=True, dtype=np.float16) 
        c_n = a_n.sum()  
        loss_n = c_n 
        loss_n.backward() 
        
        a_t = torch.tensor(a, requires_grad=True, dtype=torch.float16) 
        c_t = a_t.sum() 
        loss_t = c_t 
        loss_t.backward() 
        
        np.testing.assert_allclose(a_n.grad, a_t.grad.numpy(), rtol=1e-2)
    
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
        # Reduced size to prevent accumulation error
        a = np.random.randn(2, 32, 64) 
        x_n = Tensor(a, requires_grad=True, dtype=np.float16) 
        x_t = torch.tensor(a, requires_grad=True, dtype=torch.float16) 
        
        y_n = x_n.sum() 
        y_t = x_t.sum() 
        
        y_n_ax = x_n.sum(axis=2) 
        y_t_ax = x_t.sum(axis=2) 
        
        # Relaxed tolerance for float16 summation
        np.testing.assert_allclose(y_n_ax.data, y_t_ax.detach().numpy(), rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(y_n.data, y_t.detach().numpy(), rtol=1e-2, atol=1e-3) 

    def test_mean(self):
        """Test mean operation."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.mean()
        
        assert y.item() == 2.0
    
    def test_sum_backward_reduction(self):
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
        a_data = np.array([1.0, 2.0])
        b_data = np.array([[3.0], [4.0]]) 
        
        a_t = Tensor(a_data, requires_grad=True) 
        b_t = Tensor(b_data, requires_grad=True) 
        c_t  = a_t @ b_t 
        c_t.backward()
        
        a_p = torch.tensor(a_data, requires_grad=True)
        b_p = torch.tensor(b_data, requires_grad=True)
        c_p = a_p @ b_p  
        c_p.backward() 

        np.testing.assert_allclose(a_t.grad, a_p.grad.numpy(), rtol=1e-5)
        np.testing.assert_allclose(b_t.grad, b_p.grad.numpy(), rtol=1e-5)