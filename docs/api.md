# API Reference

## nnetflow.engine

### Tensor
- `Tensor(data, shape=(1,))`: Core tensor class with autodiff. Supports NumPy-like operations and gradients.
- Methods: `.backward()`, `.zero_grad()`, `.relu()`, `.tanh()`, `.sigmoid()`, `.sum()`, `.log()`, and arithmetic operators.

## nnetflow.nn

### Layers
- `Linear(in_features, out_features, bias=True)`: Fully connected layer.
- `MLP(nin, nouts)`: Multi-layer perceptron (see README for usage).

### Loss Functions
- `mse_loss(input, target)`: Mean squared error loss.
- `cross_entropy(input, target)`: Cross-entropy loss for classification.
- `bce_loss(input, target)`: Binary cross-entropy loss.

### Activations
- `.relu()`, `.tanh()`, `.sigmoid()` on Tensor objects.
- `softmax(input, dim)`: Softmax activation.

## nnetflow.optim

- `SGD(params, lr=0.01)`: Stochastic Gradient Descent optimizer.

---
See the [usage guide](usage.md) and [examples](../regression_problem_example.py, ../classification_example.py) for practical code.
