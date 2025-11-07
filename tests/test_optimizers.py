"""Tests for optimizers."""
import numpy as np
from nnetflow.engine import Tensor
from nnetflow.optim import SGD, Adam
import pytest


class TestSGD:
    """Test suite for SGD optimizer."""
    
    def test_sgd_initialization(self):
        """Test SGD initializes correctly."""
        params = [Tensor(np.random.randn(3, 2), requires_grad=True)]
        optimizer = SGD(params, lr=0.01, momentum=0.9)
        
        assert optimizer.lr == 0.01
        assert optimizer.momentum == 0.9
        assert len(optimizer.velocities) == 1
    
    def test_sgd_step_no_momentum(self):
        """Test SGD step without momentum."""
        param = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        optimizer = SGD([param], lr=0.1, momentum=0.0)
        
        # Simulate a gradient
        param.grad = np.array([[0.5, 1.0]])
        
        # Take optimizer step
        optimizer.step()
        
        # Expected: param = param - lr * grad = [1.0, 2.0] - 0.1 * [0.5, 1.0] = [0.95, 1.9]
        expected = np.array([[0.95, 1.9]])
        np.testing.assert_allclose(param.data, expected, rtol=1e-5)
    
    def test_sgd_step_with_momentum(self):
        """Test SGD step with momentum."""
        param = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        optimizer = SGD([param], lr=0.1, momentum=0.9)
        
        # First step
        param.grad = np.array([[1.0, 1.0]])
        optimizer.step()
        
        # v = 0.9 * 0 - 0.1 * 1 = -0.1
        # param = 1.0 + (-0.1) = 0.9
        expected_step1 = np.array([[0.9, 1.9]])
        np.testing.assert_allclose(param.data, expected_step1, rtol=1e-5)
        
        # Second step
        param.grad = np.array([[1.0, 1.0]])
        optimizer.step()
        
        # v = 0.9 * (-0.1) - 0.1 * 1 = -0.19
        # param = 0.9 + (-0.19) = 0.71
        expected_step2 = np.array([[0.71, 1.71]])
        np.testing.assert_allclose(param.data, expected_step2, rtol=1e-5)
    
    def test_sgd_zero_grad(self):
        """Test SGD zero_grad functionality."""
        param = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        optimizer = SGD([param], lr=0.1)
        
        # Set gradient
        param.grad = np.array([[0.5, 1.0]])
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Check gradient is zeroed
        np.testing.assert_array_equal(param.grad, np.zeros_like(param.data))
    
    def test_sgd_multiple_parameters(self):
        """Test SGD with multiple parameters."""
        param1 = Tensor(np.array([[1.0]]), requires_grad=True)
        param2 = Tensor(np.array([[2.0]]), requires_grad=True)
        optimizer = SGD([param1, param2], lr=0.1)
        
        param1.grad = np.array([[1.0]])
        param2.grad = np.array([[2.0]])
        
        optimizer.step()
        
        np.testing.assert_allclose(param1.data, np.array([[0.9]]), rtol=1e-5)
        np.testing.assert_allclose(param2.data, np.array([[1.8]]), rtol=1e-5)
    
    def test_sgd_repr(self):
        """Test SGD string representation."""
        params = [Tensor(np.random.randn(2, 2), requires_grad=True)]
        optimizer = SGD(params, lr=0.01, momentum=0.9)
        
        repr_str = repr(optimizer)
        assert "SGD" in repr_str
        assert "0.01" in repr_str
        assert "0.9" in repr_str


class TestAdam:
    """Test suite for Adam optimizer."""
    
    def test_adam_initialization(self):
        """Test Adam initializes correctly."""
        params = [Tensor(np.random.randn(3, 2), requires_grad=True)]
        optimizer = Adam(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
        
        assert optimizer.lr == 0.001
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert optimizer.eps == 1e-8
        assert optimizer.t == 0
        assert len(optimizer.m) == 1
        assert len(optimizer.v) == 1
    
    def test_adam_step(self):
        """Test Adam step."""
        param = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        optimizer = Adam([param], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
        
        # First step
        param.grad = np.array([[0.1, 0.2]])
        initial_value = param.data.copy()
        optimizer.step()
        
        # Parameter should have changed
        assert not np.allclose(param.data, initial_value)
        assert optimizer.t == 1
    
    def test_adam_bias_correction(self):
        """Test Adam bias correction."""
        param = Tensor(np.array([[1.0]]), requires_grad=True)
        optimizer = Adam([param], lr=0.001, beta1=0.9, beta2=0.999)
        
        # First step
        param.grad = np.array([[1.0]])
        optimizer.step()
        
        # Check that timestep is incremented
        assert optimizer.t == 1
        
        # Second step
        param.grad = np.array([[1.0]])
        optimizer.step()
        
        assert optimizer.t == 2
    
    def test_adam_zero_grad(self):
        """Test Adam zero_grad functionality."""
        param = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        optimizer = Adam([param], lr=0.001)
        
        # Set gradient
        param.grad = np.array([[0.5, 1.0]])
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Check gradient is zeroed
        np.testing.assert_array_equal(param.grad, np.zeros_like(param.data))
    
    def test_adam_multiple_parameters(self):
        """Test Adam with multiple parameters."""
        param1 = Tensor(np.array([[1.0]]), requires_grad=True)
        param2 = Tensor(np.array([[2.0]]), requires_grad=True)
        optimizer = Adam([param1, param2], lr=0.01)
        
        param1.grad = np.array([[0.1]])
        param2.grad = np.array([[0.2]])
        
        initial_value1 = param1.data.copy()
        initial_value2 = param2.data.copy()
        
        optimizer.step()
        
        # Parameters should have changed
        assert not np.allclose(param1.data, initial_value1)
        assert not np.allclose(param2.data, initial_value2)
    
    def test_adam_convergence_simple_problem(self):
        """Test that Adam can optimize a simple quadratic function."""
        # Minimize f(x) = x^2, starting from x=5
        param = Tensor(np.array([[5.0]]), requires_grad=True)
        optimizer = Adam([param], lr=0.1)
        
        initial_value = param.data[0, 0]
        
        # Run optimization for several steps
        for _ in range(50):
            # Gradient of x^2 is 2x
            param.grad = 2 * param.data
            optimizer.step()
        
        # Should move closer to 0
        assert abs(param.data[0, 0]) < abs(initial_value)
    
    def test_adam_repr(self):
        """Test Adam string representation."""
        params = [Tensor(np.random.randn(2, 2), requires_grad=True)]
        optimizer = Adam(params, lr=0.001, beta1=0.9, beta2=0.999)
        
        repr_str = repr(optimizer)
        assert "Adam" in repr_str
        assert "0.001" in repr_str


class TestOptimizerComparison:
    """Test comparison between optimizers."""
    
    def test_sgd_vs_adam_on_simple_problem(self):
        """Test that both optimizers can optimize a simple problem."""
        # Create identical starting parameters
        param_sgd = Tensor(np.array([[10.0, -5.0]]), requires_grad=True)
        param_adam = Tensor(np.array([[10.0, -5.0]]), requires_grad=True)
        
        optimizer_sgd = SGD([param_sgd], lr=0.01)
        optimizer_adam = Adam([param_adam], lr=0.01)
        
        # Optimize f(x, y) = x^2 + y^2 for 100 steps
        for _ in range(100):
            # Gradients: df/dx = 2x, df/dy = 2y
            param_sgd.grad = 2 * param_sgd.data
            optimizer_sgd.step()
            
            param_adam.grad = 2 * param_adam.data
            optimizer_adam.step()
        
        # Both should move towards [0, 0]
        sgd_distance = np.linalg.norm(param_sgd.data)
        adam_distance = np.linalg.norm(param_adam.data)
        
        # Both should be closer to origin than initial position
        initial_distance = np.linalg.norm(np.array([[10.0, -5.0]]))
        assert sgd_distance < initial_distance
        assert adam_distance < initial_distance
