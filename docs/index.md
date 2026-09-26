# nnetflow

`nnetflow` is a lightweight, NumPy-backed deep learning library built for learning and experimentation. It provides a minimal autodiff engine, common neural-network layers, loss functions, and optimizers in a small, readable codebase.

## Why use nnetflow?

- Small and readable implementation
- Reverse-mode autodiff through a custom `Tensor` class
- Minimal layer API for building models
- Easy to inspect and extend for learning purposes
- Built around NumPy instead of a large framework backend

## Core concepts

- `Tensor`: stores data and gradients and tracks the computation graph.
- `Module`: base class for layers and models.
- `Layer`: reusable building block such as `Linear`, `Conv2d`, `Embedding`, etc.
- `Loss`: objective function such as MSE or cross-entropy.
- `Optimizer`: updates parameters using gradients, such as `SGD` or `Adam`.

## Quick Start

```bash
pip install -e .
```

Then:

```python
import numpy as np
from nnetflow import Tensor, Linear, MSELoss, Adam

x = Tensor(np.random.randn(32, 4), requires_grad=False)
y = Tensor(np.random.randn(32, 1), requires_grad=False)

layer = Linear(4, 1)
opt = Adam(layer.parameters(), lr=1e-3)
loss_fn = MSELoss()

for _ in range(50):
    pred = layer(x)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

print("training complete")
```

## Documentation sections

- [Quick Start](quickstart.md)
- [Guide](guide.md)
- [API Reference](reference.md)

## Project structure

```text
nnetflow/
├── engine.py
├── layers.py
├── losses.py
├── module.py
├── optim.py
├── visualize.py
└── __init__.py
```

This documentation is designed to help you understand both the runtime behavior and the public API exposed by the library.
