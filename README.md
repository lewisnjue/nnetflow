# nnetflow

A minimal neural network framework with autodiff, inspired by micrograd and pytorch.

## Installation
```bash
pip install nnetflow
```

## Usage

from nnetflow.nn import MLP, SGD, MSELoss
from nnetflow.engine import Tensor

model = MLP(nin=3, nouts=[8, 2])

**example**
```py
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nnetflow.engine import Tensor
from nnetflow.nn import Linear, mse_loss
from nnetflow.optim import SGD


X, y = fetch_california_housing(return_X_y=True)
y = y.reshape(-1, 1)  


scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X).astype(np.float32)
y = scaler_y.fit_transform(y).astype(np.float32)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr = 0.01
epochs = 100
batch_size = 64

input_dim = X.shape[1]
hidden_dim = 32
output_dim = 1


model = [
    Linear(input_dim, hidden_dim),
    Linear(hidden_dim, output_dim)
]

# Collect trainable parameters
params = []
for layer in model:
    params.append(layer.weight)
    if layer.bias:
        params.append(layer.bias)
optimizer = SGD(params, lr=lr)


def forward(x_batch):
    out = Tensor(x_batch, shape=x_batch.shape)
    out = model[0](out).relu()
    out = model[1](out)
    return out


for epoch in range(1, epochs + 1):
    perm = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[perm], y_train[perm]
    total_loss = 0.0

    for i in range(0, X_train.shape[0], batch_size):
        xb = X_train[i:i + batch_size]
        yb = y_train[i:i + batch_size]

        preds = forward(xb)
        loss = mse_loss(preds, Tensor(yb))
        total_loss += loss.data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / (X_train.shape[0] / batch_size)
    print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss.item():.4f}")


preds_test = forward(X_test).data
mse = np.mean((preds_test - y_test) ** 2)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse:.4f}")

```


# ...

See the docs/ folder for more details.
