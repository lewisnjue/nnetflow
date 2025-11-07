# Getting Started with nnetflow

This short tutorial shows how to build and train a simple linear regression model using `nnetflow`.

## 1. Prepare data

```python
import numpy as np
from nnetflow.engine import Tensor
from nnetflow.layers import Linear
from nnetflow.losses import mse_loss
from nnetflow.optim import SGD

# Toy data
np.random.seed(0)
X = np.random.randn(200, 3)
true_w = np.array([[2.0], [-3.0], [1.5]])
true_b = np.array([[0.5]])
y = X @ true_w + true_b + 0.1 * np.random.randn(200, 1)

# Convert to tensors
X_t = Tensor(X, requires_grad=False)
y_t = Tensor(y, requires_grad=False)
```

## 2. Build model and optimizer

```python
layer = Linear(in_features=3, out_features=1)
opt = SGD([layer.weight, layer.bias], lr=0.01)
```

## 3. Train

```python
for epoch in range(200):
    preds = layer(X_t)
    loss = mse_loss(preds, y_t)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if (epoch + 1) % 50 == 0:
        print(f"epoch {epoch+1}, loss={loss.item():.4f}")
```

## 4. Inspect learned parameters

```python
print('Learned weights:', layer.weight.data.flatten())
print('Learned bias:', layer.bias.data.flatten())
```
