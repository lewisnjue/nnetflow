import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nnetflow.engine import Tensor
from nnetflow.nn import MLP, mse_loss
from nnetflow.optim import SGD
import torch
import torch.nn as nn
import torch.optim as optim

# --- Data Preparation ---
X, y = fetch_california_housing(return_X_y=True)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X).astype(np.float32)
y = scaler_y.fit_transform(y.reshape(-1, 1)).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- nnetflow Model ---
nf_model = MLP(nin=X.shape[1], nouts=[32, 16, 1], activation='relu')
nf_params = nf_model.parameters()
nf_optimizer = SGD(nf_params, lr=0.01, momentum=0.9)

batch_size = 128
epochs = 20
for epoch in range(1, epochs + 1):
    perm = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[perm], y_train[perm]
    total_loss = 0.0
    for i in range(0, X_train.shape[0], batch_size):
        xb = X_train[i:i+batch_size]
        yb = y_train[i:i+batch_size]
        preds = nf_model(Tensor(xb, shape=xb.shape))
        loss = mse_loss(preds, Tensor(yb, shape=yb.shape))
        total_loss += loss.data
        nf_optimizer.zero_grad()
        loss.backward()
        # Print max grad for debugging
        max_grad = max([np.abs(p.grad).max() for p in nf_params])
        if i == 0:
            print(f"[nnetflow] Max grad: {max_grad}")
        # Clip gradients to prevent explosion
        for p in nf_params:
            np.clip(p.grad, -1.0, 1.0, out=p.grad)
        nf_optimizer.step()
    avg_loss = total_loss / (X_train.shape[0] / batch_size)
    print(f"[nnetflow] Epoch {epoch}/{epochs} - Loss: {avg_loss.item():.4f}")

nf_preds_test = nf_model(Tensor(X_test, shape=X_test.shape)).data
nf_mse = np.mean((nf_preds_test - y_test) ** 2)
print(f"[nnetflow] Test MSE: {nf_mse:.4f}")

# --- PyTorch Model ---
class TorchMLP(nn.Module):
    def __init__(self, nin, nouts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nouts[0]),
            nn.ReLU(),
            nn.Linear(nouts[0], nouts[1]),
            nn.ReLU(),
            nn.Linear(nouts[1], nouts[2])
        )
    def forward(self, x):
        return self.net(x)

torch_model = TorchMLP(X.shape[1], [32, 16, 1])
torch_optimizer = optim.SGD(torch_model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.MSELoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

for epoch in range(1, epochs + 1):
    perm = torch.randperm(X_train_t.size(0))
    X_train_t, y_train_t = X_train_t[perm], y_train_t[perm]
    total_loss = 0.0
    for i in range(0, X_train_t.size(0), batch_size):
        xb = X_train_t[i:i+batch_size]
        yb = y_train_t[i:i+batch_size]
        preds = torch_model(xb)
        loss = loss_fn(preds, yb)
        torch_optimizer.zero_grad()
        loss.backward()
        torch_optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / (X_train_t.size(0) / batch_size)
    print(f"[PyTorch] Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

torch_model.eval()
with torch.no_grad():
    torch_preds_test = torch_model(X_test_t).numpy()
torch_mse = np.mean((torch_preds_test - y_test) ** 2)
print(f"[PyTorch] Test MSE: {torch_mse:.4f}")

print(f"\nComparison:\n[nnetflow] Test MSE: {nf_mse:.4f}\n[PyTorch]  Test MSE: {torch_mse:.4f}")
