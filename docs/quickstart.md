# Quick Start

This guide shows the fastest way to get a simple model training loop running with `nnetflow`.

## Install

```bash
pip install -e .
```

## Minimal example

```python
import numpy as np
from nnetflow import Tensor, Linear, MSELoss, Adam

X = Tensor(np.random.randn(32, 4), requires_grad=False)
y = Tensor(np.random.randn(32, 1), requires_grad=False)

layer = Linear(4, 1)
opt = Adam(layer.parameters(), lr=1e-2)
loss_fn = MSELoss()

for step in range(100):
    pred = layer(X)
    loss = loss_fn(pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 10 == 0:
        print(f"step={step} loss={loss.item():.4f}")
```

## Working with Tensor objects

```python
import numpy as np
from nnetflow import Tensor

x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
y = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)

z = (x * y).sum()
z.backward()

print(x.grad)
print(y.grad)
```

## Common patterns

- Use `requires_grad=True` for trainable parameters.
- Call `zero_grad()` before every optimizer step.
- Call `backward()` on a scalar or reduced loss tensor.
- For custom model logic, subclass `Module` and implement `forward()`.
