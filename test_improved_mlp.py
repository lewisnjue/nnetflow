from nnetflow.engine import Tensor
from nnetflow.layers import Linear
from nnetflow.module import Module
from nnetflow.optim import SGD
import numpy as np

# Define a simple MLP with improved initialization
class MLP(Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # Use He initialization for ReLU activations
        self.fc1 = Linear(in_dim, hidden_dim, device='cpu', init='he_normal')
        self.fc2 = Linear(hidden_dim, out_dim, device='cpu', init='xavier_normal')

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

# Instantiate model with improved initialization
print("Training with improved weight initialization...")
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

print("Training completed!")