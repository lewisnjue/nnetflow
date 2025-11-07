"""
Simple Regression Example using nnetflow

This example demonstrates how to build and train a neural network
for regression using the nnetflow library.
"""

import numpy as np
from nnetflow.engine import Tensor
from nnetflow.layers import Linear
from nnetflow.losses import mse_loss
from nnetflow.optim import SGD, Adam


def generate_regression_data(n_samples=100, seed=42):
    """Generate synthetic regression data: y = 2x1 - 3x2 + 1.5x3 + noise"""
    np.random.seed(seed)
    
    X = np.random.randn(n_samples, 3)
    true_weights = np.array([[2.0], [-3.0], [1.5]])
    true_bias = 0.5
    noise = 0.1 * np.random.randn(n_samples, 1)
    
    y = X @ true_weights + true_bias + noise
    
    return X, y


def main():
    print("=" * 60)
    print("Simple Regression with nnetflow")
    print("=" * 60)
    
    # Generate data
    X_train, y_train = generate_regression_data(n_samples=100)
    X_test, y_test = generate_regression_data(n_samples=20, seed=999)
    
    print(f"\nDataset:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Build model: Simple linear regression (no hidden layers)
    layer = Linear(in_features=3, out_features=1)
    
    # Create optimizer
    optimizer = SGD([layer.weight, layer.bias], lr=0.01)
    
    # Convert to Tensors
    X_train_tensor = Tensor(X_train, requires_grad=False)
    y_train_tensor = Tensor(y_train, requires_grad=False)
    X_test_tensor = Tensor(X_test, requires_grad=False)
    y_test_tensor = Tensor(y_test, requires_grad=False)
    
    # Training loop
    n_epochs = 1000
    print(f"\nTraining for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        # Forward pass
        predictions = layer(X_train_tensor)
        
        # Compute loss
        loss = mse_loss(predictions, y_train_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            # Test loss
            predictions_test = layer(X_test_tensor)
            test_loss = mse_loss(predictions_test, y_test_tensor)
            
            print(f"Epoch {epoch + 1:4d} | Train Loss: {loss.item():.6f} | "
                  f"Test Loss: {test_loss.item():.6f}")
    
    # Print learned weights
    print(f"\n{'=' * 60}")
    print("Learned parameters:")
    print(f"  Weights: {layer.weight.data.flatten()}")
    print(f"  Bias: {layer.bias.data.flatten()[0]:.4f}")
    print(f"\nTrue parameters:")
    print(f"  Weights: [ 2.0 -3.0  1.5]")
    print(f"  Bias: 0.5000")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
