# nnetflow

A minimal neural network framework with autodiff, inspired by micrograd and pytorch.

## Installation

```bash
pip install nnetflow
```

### From source

```bash
git clone https://github.com/lewisnjue/nnetflow.git
cd nnetflow
pip install -e .
```

```bash

from nnetflow.engine import Tensor
from nnetflow.layers import Linear
from nnetflow.module import Module
from nnetflow.optim import SGD
import numpy as np

# Define a simple MLP
class MLP(Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

# Generate dummy data
np.random.seed(0)
X = np.random.randn(100, 3).astype(np.float32)
y = (np.random.randn(100, 1) > 0).astype(np.float32)

# Convert to Tensor
X_tensor = Tensor(X, require_grad=False)
y_tensor = Tensor(y, require_grad=False)

# Instantiate model, loss, optimizer
model = MLP(3, 8, 1)
optimizer = SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    out = model(X_tensor)
    # Simple MSE loss
    loss = ((out - y_tensor) ** 2).mean()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.save("mlp_model.pkl")

# Load the model
loaded_model = Module.load("mlp_model.pkl")
# Verify loaded model
print(f"Loaded model: {loaded_model}")
# Check if the loaded model can still perform inference
test_out = loaded_model(X_tensor)
print(f"Test output from loaded model: {test_out.data[:5]}")  # Print first 5 outputs
# Check if the loaded model's parameters match the original model's parameters
for original_param, loaded_param in zip(model.parameters(), loaded_model.parameters()):
    assert np.array_equal(original_param.data, loaded_param.data), "Loaded parameters do not match original parameters"
print("All parameters match successfully after loading the model.")
```


# Documentation

- See `docs/index.md` for a full guide and API overview.
- See `CONTRIBUTING.md` for contribution guidelines.
- See `CHANGELOG.md` for release notes.
