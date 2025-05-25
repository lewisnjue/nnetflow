import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nnetflow.engine import Tensor
from nnetflow.nn import MLP, mse_loss
from nnetflow.optim import SGD,Adam
from nnetflow import cuda

# Generate regression data
X, y = make_regression(n_samples=20000, n_features=10, noise=0.1)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X).astype(np.float32)
y = scaler_y.fit_transform(y.reshape(-1, 1)).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: 2 hidden layers, ReLU, no output activation
device = 'cuda' if cuda.is_available() else 'cpu'
print(f"[nnetflow] Using device: {device.upper()}")
model = MLP(nin=10, nouts=[32, 16, 1], activation='relu').to(device)
params = model.parameters()
optimizer = Adam(params, lr=0.01)

batch_size = 1024
epochs = 50
for epoch in range(1, epochs + 1):
    perm = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[perm], y_train[perm]
    total_loss = 0.0
    for i in range(0, X_train.shape[0], batch_size):
        xb = X_train[i:i+batch_size]
        yb = y_train[i:i+batch_size]
        preds = model(Tensor(xb, shape=xb.shape, device=device))
        loss = mse_loss(preds, Tensor(yb, shape=yb.shape, device=device))
        total_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / (X_train.shape[0] / batch_size)
    print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss.item():.4f}")

# Test
preds_test = model(Tensor(X_test, shape=X_test.shape, device=device)).data
# Ensure both preds_test and y_test are NumPy arrays for metric computation
cp = getattr(cuda, 'cp', None)
if cp is not None and isinstance(preds_test, cp.ndarray):
    preds_test_np = cp.asnumpy(preds_test)
else:
    preds_test_np = preds_test
if cp is not None and isinstance(y_test, cp.ndarray):
    y_test_np = cp.asnumpy(y_test)
else:
    y_test_np = y_test
mse = np.mean((preds_test_np - y_test_np) ** 2)
print(f"Test MSE: {mse:.4f}")
