"""
Binary Classification Example using nnetflow

This example demonstrates how to build and train a simple neural network
for binary classification using the nnetflow library.
"""

import numpy as np
from nnetflow.engine import Tensor
from nnetflow.layers import Linear
from nnetflow.losses import logits_binary_cross_entropy_loss
from nnetflow.optim import Adam


def generate_binary_data(n_samples=1000, n_features=2, seed=42):
    """Generate synthetic binary classification data."""
    np.random.seed(seed)
    
    # Generate two clusters
    cluster_0 = np.random.randn(n_samples // 2, n_features) + np.array([2, 2])
    cluster_1 = np.random.randn(n_samples // 2, n_features) + np.array([-2, -2])
    
    X = np.vstack([cluster_0, cluster_1])
    y = np.vstack([np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))])
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def main():
    print("=" * 60)
    print("Binary Classification with nnetflow")
    print("=" * 60)
    
    # Generate data
    X_train, y_train = generate_binary_data(n_samples=800)
    X_test, y_test = generate_binary_data(n_samples=200, seed=123)
    
    print(f"\nDataset:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Build model: Input(2) -> Hidden(16) -> Hidden(8) -> Output(1)
    layer1 = Linear(in_features=2, out_features=16)
    layer2 = Linear(in_features=16, out_features=8)
    layer3 = Linear(in_features=8, out_features=1)
    
    # Collect all parameters
    params = [layer1.weight, layer1.bias, layer2.weight, layer2.bias, 
              layer3.weight, layer3.bias]
    
    # Create optimizer
    optimizer = Adam(params, lr=0.01)
    
    # Convert to Tensors
    X_train_tensor = Tensor(X_train, requires_grad=False)
    y_train_tensor = Tensor(y_train, requires_grad=False)
    X_test_tensor = Tensor(X_test, requires_grad=False)
    y_test_tensor = Tensor(y_test, requires_grad=False)
    
    # Training loop
    n_epochs = 100
    print(f"\nTraining for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        # Forward pass
        h1 = layer1(X_train_tensor).relu()
        h2 = layer2(h1).relu()
        logits = layer3(h2)
        
        # Compute loss
        loss = logits_binary_cross_entropy_loss(logits, y_train_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Test accuracy
            h1_test = layer1(X_test_tensor).relu()
            h2_test = layer2(h1_test).relu()
            logits_test = layer3(h2_test)
            probs_test = logits_test.sigmoid()
            predictions = (probs_test.data > 0.5).astype(float)
            accuracy = (predictions == y_test).mean()
            
            print(f"Epoch {epoch + 1:3d} | Loss: {loss.item():.4f} | "
                  f"Test Accuracy: {accuracy:.4f}")
    
    # Final evaluation
    h1_test = layer1(X_test_tensor).relu()
    h2_test = layer2(h1_test).relu()
    logits_test = layer3(h2_test)
    probs_test = logits_test.sigmoid()
    predictions = (probs_test.data > 0.5).astype(float)
    accuracy = (predictions == y_test).mean()
    
    print(f"\n{'=' * 60}")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
