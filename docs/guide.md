# Guide

This section explains the main ideas behind `nnetflow` and how the library is structured.

## 1. Tensors and autograd

`nnetflow` uses a minimal autograd engine built around the `Tensor` class. Each tensor stores:

- a NumPy-backed value,
- a `requires_grad` flag,
- a gradient buffer,
- references to parent tensors and the backward closure that defines the gradient rule.

Whenever an operation creates a new value, it records the dependency and later uses it during `backward()`.

```python
import numpy as np
from nnetflow import Tensor

x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
y = x * 2
z = y.sum()
z.backward()
print(x.grad)
```

## 2. Modules and layers

Most models are built from `Module` subclasses. A module groups parameters and exposes a `forward()` function.

```python
from nnetflow import Linear

layer = Linear(10, 5)
```

The `Linear` layer is a simple example of a reusable block whose parameters are tracked by the autograd engine.

## 3. Loss functions

Losses produce scalar values that represent model error. A typical training loop looks like this:

```python
pred = layer(x)
loss = loss_fn(pred, target)

loss.backward()
```

The library includes a few common objectives such as MSE and cross-entropy.

## 4. Optimizers

Optimizers update trainable weights using gradients from `backward()`.

```python
from nnetflow import Adam

opt = Adam(layer.parameters(), lr=1e-3)
opt.zero_grad()
loss.backward()
opt.step()
```

The optimizer logic is intentionally simple and easy to read, which makes it useful for learning.

## 5. Training loop pattern

```python
for epoch in range(num_epochs):
    opt.zero_grad()
    predictions = model(x)
    loss = loss_fn(predictions, y)
    loss.backward()
    opt.step()
```

## 6. Visualization

You can inspect the computation graph for a tensor using the visualization utilities.

```python
from nnetflow import visualize_model

visualize_model(loss)
```

This is useful for debugging forward passes and understanding the graph structure behind a model.
