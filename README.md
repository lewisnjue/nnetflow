# nnetflow

A minimal neural network framework with autodiff, inspired by micrograd and pytorch.

## Installation

pip install nnetflow

## Usage

from nnetflow.nn import MLP, SGD, MSELoss
from nnetflow.engine import Tensor

model = MLP(nin=3, nouts=[8, 2])

**example**
```py
import numpy as np
from sklearn.datasets import make_regression
from nnetflow.nn import MLP, SGD, MSELoss
from nnetflow.engine import Tensor

X, Y = make_regression(n_samples=100, n_features=3, n_targets=2, noise=0.1, random_state=42)


data = []
for x, y in zip(X, Y):
    x_tensors = [Tensor([float(val)]) for val in x]
    y_tensors = [Tensor([float(val)]) for val in y]
    data.append((x_tensors, y_tensors))

model = MLP(nin=3, nouts=[8, 2])

loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.001)

for epoch in range(200):
    total_loss = 0.0
    for x, y_true in data:
        y_pred = model(x)
        if not isinstance(y_pred, list):
            y_pred = [y_pred]
        loss = loss_fn(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

param = model.parameters()[0]
orig = param.data.copy()
eps = 1e-4
param.data += eps
plus_loss = 0.0
for x, y_true in data:
    y_pred = model(x)
    if not isinstance(y_pred, list):
        y_pred = [y_pred]
    plus_loss += loss_fn(y_pred, y_true).data.item()
param.data = orig - eps
minus_loss = 0.0
for x, y_true in data:
    y_pred = model(x)
    if not isinstance(y_pred, list):
        y_pred = [y_pred]
    minus_loss += loss_fn(y_pred, y_true).data.item()
param.data = orig
fd_grad = (plus_loss - minus_loss) / (2 * eps)
optimizer.zero_grad()
for x, y_true in data:
    y_pred = model(x)
    if not isinstance(y_pred, list):
        y_pred = [y_pred]
    loss = loss_fn(y_pred, y_true)
    loss.backward()
print(f"Finite diff grad: {fd_grad:.6f}, Backprop grad: {param.grad.flatten()[0]:.6f}")

x_test = [Tensor([float(val)]) for val in X[0]]
y_pred = model(x_test)
print("Prediction:", [p.data for p in y_pred])
print("True:", Y[0])

```


# ...

See the docs/ folder for more details.