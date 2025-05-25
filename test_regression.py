import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nnetflow.engine import Tensor
from nnetflow.nn import MLP, mse_loss
from nnetflow.optim import SGD,Adam

# Generate regression data
X, y = make_regression(n_samples=20000, n_features=10, noise=0.1)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X).astype(np.float32)
y = scaler_y.fit_transform(y.reshape(-1, 1)).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: 2 hidden layers, ReLU, no output activation
model = MLP(nin=10, nouts=[32, 16, 1], activation='relu')
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
        preds = model(Tensor(xb, shape=xb.shape))
        loss = mse_loss(preds, Tensor(yb, shape=yb.shape))
        total_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / (X_train.shape[0] / batch_size)
    print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss.item():.4f}")

# Test
preds_test = model(Tensor(X_test, shape=X_test.shape)).data
mse = np.mean((preds_test - y_test) ** 2)
print(f"Test MSE: {mse:.4f}")
