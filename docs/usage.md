# Usage Guide

nnetflow is designed for easy experimentation with neural networks and autodiff. Below are typical usage patterns for regression and classification problems.

## Regression Example

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nnetflow.engine import Tensor
from nnetflow.nn import Linear, mse_loss
from nnetflow.optim import SGD

# Load and preprocess data
X, y = fetch_california_housing(return_X_y=True)
y = y.reshape(-1, 1)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X).astype(np.float32)
y = scaler_y.fit_transform(y).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model and training setup
model = [Linear(X.shape[1], 32), Linear(32, 1)]
params = [layer.weight for layer in model] + [layer.bias for layer in model if layer.bias is not None]
optimizer = SGD(params, lr=0.01)

def forward(x_batch):
    out = Tensor(x_batch, shape=x_batch.shape)
    out = model[0](out).relu()
    out = model[1](out)
    return out

# Training loop
for epoch in range(1, 101):
    ... # batching, forward, loss, backward, optimizer.step

# Evaluation
preds_test = forward(X_test).data
```

## Classification Example

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nnetflow.engine import Tensor
from nnetflow.nn import Linear, cross_entropy, softmax
from nnetflow.optim import SGD

# Load and preprocess data
X, y = load_iris(return_X_y=True)
y = y.reshape(-1, 1)
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X).astype(np.float32)
y = np.eye(3)[y.astype(int).reshape(-1)]  # One-hot encoding
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model and training setup
model = [Linear(X.shape[1], 10), Linear(10, 3)]
params = [layer.weight for layer in model] + [layer.bias for layer in model if layer.bias is not None]
optimizer = SGD(params, lr=0.01)

def forward(x_batch):
    out = Tensor(x_batch, shape=x_batch.shape)
    out = model[0](out).relu()
    out = model[1](out)
    return out

# Training loop
for epoch in range(1, 101):
    ... # batching, forward, loss, backward, optimizer.step

# Evaluation
preds_test = forward(X_test).data
preds_test_class = np.argmax(preds_test, axis=1)
```

## Tensor API

- `Tensor(data, shape=(1,))`: Core data structure supporting autodiff.
- Supports operations: `+`, `-`, `*`, `/`, `@` (matmul), `.relu()`, `.tanh()`, `.sigmoid()`, `.sum()`, `.backward()`

## Layers and Losses

- `Linear(in_features, out_features)`: Fully connected layer.
- `mse_loss`, `cross_entropy`, `bce_loss`: Loss functions for regression/classification.

## Optimizers

- `SGD(params, lr=0.01)`: Stochastic Gradient Descent.

---
See the [API Reference](api.md) for more details.
