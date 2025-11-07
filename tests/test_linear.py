"""Tests for Linear layer."""
import numpy as np 
from nnetflow.layers import Linear
from nnetflow.engine import Tensor
import pytest


class TestLinearLayer:
    """Test suite for Linear layer."""
    
    def test_linear_initialization(self):
        """Test that Linear layer initializes with correct shapes."""
        linear = Linear(in_features=10, out_features=5)
        
        assert linear.weight.shape == (10, 5), "Weight shape mismatch"
        assert linear.bias.shape == (1, 5), "Bias shape mismatch"
        assert linear.weight.requires_grad, "Weights should require gradients"
        assert linear.bias.requires_grad, "Bias should require gradients"
    
    def test_linear_forward_pass(self):
        """Test forward pass produces correct output shape."""
        linear = Linear(in_features=3, out_features=5)
        x = Tensor(np.random.randn(10, 3), requires_grad=False)
        
        output = linear(x)
        
        assert output.shape == (10, 5), "Output shape mismatch"
        assert output.requires_grad, "Output should require gradients"
    
    def test_linear_forward_computation(self):
        """Test that forward pass computation is correct."""
        linear = Linear(in_features=2, out_features=3)
        
        # Set known weights and bias
        linear.weight.data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        linear.bias.data = np.array([[0.1, 0.2, 0.3]])
        
        x = Tensor(np.array([[1.0, 2.0]]), requires_grad=False)
        output = linear(x)
        
        # Expected: [1, 2] @ [[1, 2, 3], [4, 5, 6]] + [0.1, 0.2, 0.3]
        #         = [9, 12, 15] + [0.1, 0.2, 0.3] = [9.1, 12.2, 15.3]
        expected = np.array([[9.1, 12.2, 15.3]])
        
        np.testing.assert_allclose(output.data, expected, rtol=1e-5)
    
    def test_linear_backward_pass(self):
        """Test that backward pass computes gradients."""
        np.random.seed(42)
        linear = Linear(in_features=3, out_features=1)
        
        x = Tensor(np.random.randn(5, 3), requires_grad=False)
        y = Tensor(np.random.randn(5, 1), requires_grad=False)
        
        output = linear(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()
        
        assert linear.weight.grad is not None, "Weight gradients should not be None"
        assert linear.bias.grad is not None, "Bias gradients should not be None"
        assert linear.weight.grad.shape == linear.weight.shape, "Weight gradient shape mismatch"
        assert linear.bias.grad.shape == linear.bias.shape, "Bias gradient shape mismatch"
    
    def test_linear_gradient_numerical_check(self):
        """Test gradients using numerical gradient checking."""
        np.random.seed(42)
        linear = Linear(in_features=2, out_features=1)
        
        x = Tensor(np.array([[1.0, 2.0]]), requires_grad=False)
        
        # Forward and backward
        output = linear(x)
        loss = output.sum()
        loss.backward()
        
        # Numerical gradient for weight[0, 0]
        eps = 1e-5
        original_val = linear.weight.data[0, 0]
        
        linear.weight.data[0, 0] = original_val + eps
        loss_plus = linear(x).sum().item()
        
        linear.weight.data[0, 0] = original_val - eps
        loss_minus = linear(x).sum().item()
        
        numerical_grad = (loss_plus - loss_minus) / (2 * eps)
        analytical_grad = linear.weight.grad[0, 0]
        
        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6)
    
    def test_linear_input_shape_assertion(self):
        """Test that Linear layer raises error for incorrect input shapes."""
        linear = Linear(in_features=3, out_features=5)
        
        # Wrong number of features
        with pytest.raises(AssertionError, match="Input feature size mismatch"):
            x = Tensor(np.random.randn(10, 5), requires_grad=False)
            linear(x)
        
        # Wrong number of dimensions (note: this checks feature size first)
        with pytest.raises(AssertionError):
            x = Tensor(np.random.randn(10, 3, 2), requires_grad=False)
            linear(x)
    
    def test_linear_multiple_forward_passes(self):
        """Test that multiple forward passes work correctly."""
        linear = Linear(in_features=3, out_features=2)
        
        x1 = Tensor(np.random.randn(5, 3), requires_grad=False)
        x2 = Tensor(np.random.randn(3, 3), requires_grad=False)
        
        output1 = linear(x1)
        output2 = linear(x2)
        
        assert output1.shape == (5, 2)
        assert output2.shape == (3, 2)
    
    def test_linear_zero_grad(self):
        """Test that gradients can be zeroed."""
        linear = Linear(in_features=3, out_features=1)
        
        x = Tensor(np.random.randn(5, 3), requires_grad=False)
        output = linear(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert np.any(linear.weight.grad != 0)
        assert np.any(linear.bias.grad != 0)
        
        # Zero gradients
        linear.weight.zero_grad()
        linear.bias.zero_grad()
        
        # Check gradients are zero
        np.testing.assert_array_equal(linear.weight.grad, np.zeros_like(linear.weight.data))
        np.testing.assert_array_equal(linear.bias.grad, np.zeros_like(linear.bias.data))
    
    def test_linear_repr(self):
        """Test string representation of Linear layer."""
        linear = Linear(in_features=10, out_features=5)
        repr_str = repr(linear)
        
        assert "Linear" in repr_str
        assert "in_features=10" in repr_str
        assert "out_features=5" in repr_str
