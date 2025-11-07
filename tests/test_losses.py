"""Tests for loss functions."""
import numpy as np
from nnetflow.engine import Tensor
from nnetflow.losses import (
    mse_loss,
    rmse_loss,
    cross_entropy_loss,
    binary_cross_entropy_loss,
    logits_binary_cross_entropy_loss
)
import pytest


class TestMSELoss:
    """Test suite for MSE loss."""
    
    def test_mse_loss_computation(self):
        """Test that MSE loss computes correctly."""
        predictions = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
        targets = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=False)
        
        loss = mse_loss(predictions, targets)
        
        assert loss.item() == 0.0, "MSE should be 0 for identical predictions and targets"
    
    def test_mse_loss_nonzero(self):
        """Test MSE loss with non-zero error."""
        predictions = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        targets = Tensor(np.array([[2.0, 4.0]]), requires_grad=False)
        
        loss = mse_loss(predictions, targets)
        
        # MSE = mean((1-2)^2 + (2-4)^2) = mean(1 + 4) = 2.5
        expected = 2.5
        np.testing.assert_allclose(loss.item(), expected, rtol=1e-5)
    
    def test_mse_loss_gradient(self):
        """Test that MSE loss computes gradients correctly."""
        predictions = Tensor(np.array([[3.0, 5.0]]), requires_grad=True)
        targets = Tensor(np.array([[1.0, 2.0]]), requires_grad=False)
        
        loss = mse_loss(predictions, targets)
        loss.backward()
        
        # d/dx MSE = 2 * (pred - target) / n
        # For pred=[3, 5], target=[1, 2]: grad = 2 * [2, 3] / 2 = [2, 3]
        expected_grad = np.array([[2.0, 3.0]])
        np.testing.assert_allclose(predictions.grad, expected_grad, rtol=1e-5)
    
    def test_mse_loss_batch(self):
        """Test MSE loss with batch of samples."""
        np.random.seed(42)
        predictions = Tensor(np.random.randn(10, 5), requires_grad=True)
        targets = Tensor(np.random.randn(10, 5), requires_grad=False)
        
        loss = mse_loss(predictions, targets)
        loss.backward()
        
        assert loss.shape == (), "Loss should be scalar"
        assert predictions.grad is not None
        assert predictions.grad.shape == predictions.shape


class TestRMSELoss:
    """Test suite for RMSE loss."""
    
    def test_rmse_loss_computation(self):
        """Test that RMSE loss computes correctly."""
        predictions = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        targets = Tensor(np.array([[2.0, 4.0]]), requires_grad=False)
        
        loss = rmse_loss(predictions, targets)
        
        # RMSE = sqrt(mean((1-2)^2 + (2-4)^2)) = sqrt(2.5) ≈ 1.581
        expected = np.sqrt(2.5)
        np.testing.assert_allclose(loss.item(), expected, rtol=1e-5)
    
    def test_rmse_loss_gradient(self):
        """Test that RMSE loss computes gradients."""
        predictions = Tensor(np.array([[3.0, 5.0]]), requires_grad=True)
        targets = Tensor(np.array([[1.0, 2.0]]), requires_grad=False)
        
        loss = rmse_loss(predictions, targets)
        loss.backward()
        
        assert predictions.grad is not None
        assert predictions.grad.shape == predictions.shape


class TestCrossEntropyLoss:
    """Test suite for cross-entropy loss."""
    
    def test_cross_entropy_loss_computation(self):
        """Test cross-entropy loss computation."""
        # Logits for 2 samples, 3 classes
        logits = Tensor(np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]]), requires_grad=True)
        # One-hot targets
        targets = Tensor(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), requires_grad=False)
        
        loss = cross_entropy_loss(logits, targets)
        
        assert loss.shape == (), "Loss should be scalar"
        assert loss.item() > 0, "Cross-entropy should be positive"
    
    def test_cross_entropy_loss_gradient(self):
        """Test that cross-entropy computes gradients."""
        logits = Tensor(np.array([[2.0, 1.0, 0.1]]), requires_grad=True)
        targets = Tensor(np.array([[1.0, 0.0, 0.0]]), requires_grad=False)
        
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape
    
    def test_cross_entropy_perfect_prediction(self):
        """Test cross-entropy with near-perfect prediction."""
        # Very high logit for correct class
        logits = Tensor(np.array([[10.0, -10.0, -10.0]]), requires_grad=True)
        targets = Tensor(np.array([[1.0, 0.0, 0.0]]), requires_grad=False)
        
        loss = cross_entropy_loss(logits, targets)
        
        # Should be very close to 0
        assert loss.item() < 0.1, "Loss should be very small for confident correct prediction"


class TestBinaryCrossEntropyLoss:
    """Test suite for binary cross-entropy loss."""
    
    def test_bce_loss_computation(self):
        """Test BCE loss computation."""
        # Predictions after sigmoid (probabilities)
        predictions = Tensor(np.array([[0.9, 0.1, 0.8]]), requires_grad=True)
        targets = Tensor(np.array([[1.0, 0.0, 1.0]]), requires_grad=False)
        
        loss = binary_cross_entropy_loss(predictions, targets)
        
        assert loss.shape == (), "Loss should be scalar"
        assert loss.item() > 0, "BCE should be positive"
    
    def test_bce_loss_gradient(self):
        """Test that BCE computes gradients."""
        predictions = Tensor(np.array([[0.7, 0.3]]), requires_grad=True)
        targets = Tensor(np.array([[1.0, 0.0]]), requires_grad=False)
        
        loss = binary_cross_entropy_loss(predictions, targets)
        loss.backward()
        
        assert predictions.grad is not None
        assert predictions.grad.shape == predictions.shape
    
    def test_bce_perfect_prediction(self):
        """Test BCE with near-perfect prediction."""
        predictions = Tensor(np.array([[0.99, 0.01]]), requires_grad=True)
        targets = Tensor(np.array([[1.0, 0.0]]), requires_grad=False)
        
        loss = binary_cross_entropy_loss(predictions, targets)
        
        # Should be very close to 0
        assert loss.item() < 0.1, "Loss should be small for confident correct prediction"


class TestLogitsBinaryCrossEntropyLoss:
    """Test suite for BCE loss with logits."""
    
    def test_logits_bce_loss_computation(self):
        """Test BCE with logits computation."""
        # Raw logits (before sigmoid)
        logits = Tensor(np.array([[2.0, -2.0, 1.5]]), requires_grad=True)
        targets = Tensor(np.array([[1.0, 0.0, 1.0]]), requires_grad=False)
        
        loss = logits_binary_cross_entropy_loss(logits, targets)
        
        assert loss.shape == (), "Loss should be scalar"
        assert loss.item() > 0, "BCE should be positive"
    
    def test_logits_bce_loss_gradient(self):
        """Test that BCE with logits computes gradients."""
        logits = Tensor(np.array([[1.0, -1.0]]), requires_grad=True)
        targets = Tensor(np.array([[1.0, 0.0]]), requires_grad=False)
        
        loss = logits_binary_cross_entropy_loss(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape
    
    def test_logits_bce_with_reasonable_values(self):
        """Test BCE with logits with reasonable values."""
        # Reasonable logits
        logits = Tensor(np.array([[5.0, -5.0]]), requires_grad=True)
        targets = Tensor(np.array([[1.0, 0.0]]), requires_grad=False)
        
        loss = logits_binary_cross_entropy_loss(logits, targets)
        
        # Should produce finite loss
        assert np.isfinite(loss.item()), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"
