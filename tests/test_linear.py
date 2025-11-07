import numpy as np 
from nnetflow.layers import Linear
from nnetflow.engine import Tensor
import pytest
# test linear layer by creating a regression task and a classification task 


def test_linear_regression():
    # create a simple dataset for regression 
    np.random.seed(0)
    X = np.random.rand(100, 3) 
    true_weights = np.array([[2.0], [-3.0], [1.5]])
    true_bias = np.array([[0.5]])
    y = X @ true_weights + true_bias + 0.1 * np.random.randn(100, 1)  # add some noise 

    # create linear layer 
    linear = Linear(in_features=3, out_features=1)


    # forward pass 
    inputs = Tensor(X, requires_grad=False)
    targets = Tensor(y, requires_grad=False)
    outputs = linear(inputs)

    # compute mean squared error loss 
    loss = ((outputs - targets) ** 2).mean()

    # backward pass 
    loss.backward() 

    assert linear.weight.grad is not None, "Weight gradients should not be None" 
    assert linear.bias.grad is not None, "Bias gradients should not be None" 