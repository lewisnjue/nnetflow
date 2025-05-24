import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nnetflow.engine import Tensor
from nnetflow.nn import MLP, Linear, cross_entropy, mse_loss
from nnetflow.optim import SGD, Adam

# --- 1. Regression Task ---
print("\n=== REGRESSION TASK ===")
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X).astype(np.float32)
y = scaler_y.fit_transform(y.reshape(-1, 1)).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# nnetflow
nf_model = MLP(nin=10, nouts=[32, 16, 1], activation='relu')
nf_params = []
for p in nf_model.parameters():
    if hasattr(p, 'data') and isinstance(p.data, np.ndarray):
        nf_params.append(p)
nf_optimizer = SGD(nf_params, lr=0.01, momentum=0.9)
batch_size = 64
for epoch in range(1, 6):
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
        # Debug: print first param and grad before step
        if epoch == 1 and i == 0:
            if len(nf_params) == 0:
                print('[nnetflow][DEBUG] No parameters found in model!')
            else:
                print('[nnetflow][DEBUG] First param before step:', nf_params[0].data.flatten()[:5])
                print('[nnetflow][DEBUG] First grad:', nf_params[0].grad.flatten()[:5])
        nf_optimizer.step()
        if len(nf_params) > 0 and epoch == 1 and i == 0:
            print('[nnetflow][DEBUG] First param after step:', nf_params[0].data.flatten()[:5])
    avg_loss = total_loss / (X_train.shape[0] / batch_size)
    if isinstance(avg_loss, np.ndarray):
        avg_loss = avg_loss.item() if avg_loss.size == 1 else float(np.mean(avg_loss))
    print(f"[nnetflow] Epoch {epoch} - Loss: {avg_loss:.4f}")
nf_preds_test = nf_model(Tensor(X_test, shape=X_test.shape)).data
nf_mse = np.mean((nf_preds_test - y_test) ** 2)
print(f"[nnetflow] Test MSE: {nf_mse:.4f}")

# PyTorch
class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)
torch_model = TorchMLP()
torch_optimizer = optim.SGD(torch_model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.MSELoss()
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)
for epoch in range(1, 6):
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
    print(f"[PyTorch] Epoch {epoch} - Loss: {avg_loss:.4f}")
torch_model.eval()
with torch.no_grad():
    torch_preds_test = torch_model(X_test_t).numpy()
torch_mse = np.mean((torch_preds_test - y_test) ** 2)
print(f"[PyTorch] Test MSE: {torch_mse:.4f}")

# --- 2. Classification Task ---
print("\n=== CLASSIFICATION TASK ===")
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)
y = y.astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# nnetflow
nf_model = MLP(nin=10, nouts=[32, 16, 2], activation='relu')
nf_params = []
for p in nf_model.parameters():
    if hasattr(p, 'data') and isinstance(p.data, np.ndarray):
        nf_params.append(p)
nf_optimizer = SGD(nf_params, lr=0.01, momentum=0.9)
def nf_ce_loss(logits, targets):
    targets_onehot = np.eye(2, dtype=np.float32)[targets]
    return cross_entropy(logits, Tensor(targets_onehot))
for epoch in range(1, 6):
    perm = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[perm], y_train[perm]
    total_loss = 0.0
    for i in range(0, X_train.shape[0], batch_size):
        xb = X_train[i:i+batch_size]
        yb = y_train[i:i+batch_size]
        logits = nf_model(Tensor(xb, shape=xb.shape))
        loss = nf_ce_loss(logits, yb)
        total_loss += loss.data
        nf_optimizer.zero_grad()
        loss.backward()
        nf_optimizer.step()
    avg_loss = total_loss / (X_train.shape[0] / batch_size)
    if isinstance(avg_loss, np.ndarray):
        avg_loss = avg_loss.item() if avg_loss.size == 1 else float(np.mean(avg_loss))
    print(f"[nnetflow] Epoch {epoch} - Loss: {avg_loss:.4f}")
nf_logits_test = nf_model(Tensor(X_test, shape=X_test.shape))
nf_preds = np.argmax(nf_logits_test.data, axis=1)
nf_acc = (nf_preds == y_test).mean()
print(f"[nnetflow] Test Accuracy: {nf_acc:.4f}")

# PyTorch
class TorchMLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)
torch_model = TorchMLP2()
torch_optimizer = optim.SGD(torch_model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)
for epoch in range(1, 6):
    perm = torch.randperm(X_train_t.size(0))
    X_train_t, y_train_t = X_train_t[perm], y_train_t[perm]
    total_loss = 0.0
    for i in range(0, X_train_t.size(0), batch_size):
        xb = X_train_t[i:i+batch_size]
        yb = y_train_t[i:i+batch_size]
        logits = torch_model(xb)
        loss = loss_fn(logits, yb)
        torch_optimizer.zero_grad()
        loss.backward()
        torch_optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / (X_train_t.size(0) / batch_size)
    print(f"[PyTorch] Epoch {epoch} - Loss: {avg_loss:.4f}")
torch_model.eval()
with torch.no_grad():
    torch_logits_test = torch_model(X_test_t)
    torch_preds = torch_logits_test.argmax(dim=1).numpy()
torch_acc = (torch_preds == y_test).mean()
print(f"[PyTorch] Test Accuracy: {torch_acc:.4f}")

# --- 3. Image Classification Task (MNIST subset) ---
print("\n=== IMAGE CLASSIFICATION TASK (MNIST) ===")
from torchvision import datasets, transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
small_train_size = 500
small_test_size = 100
X_train = train_dataset.data.numpy().astype(np.float32)[:small_train_size] / 255.0
X_test = test_dataset.data.numpy().astype(np.float32)[:small_test_size] / 255.0
y_train = train_dataset.targets.numpy().astype(np.int64)[:small_train_size]
y_test = test_dataset.targets.numpy().astype(np.int64)[:small_test_size]
X_train = (X_train - 0.1307) / 0.3081
X_test = (X_test - 0.1307) / 0.3081
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

from nnetflow.nn import Module, Conv2D, MaxPool2D
class NFConvNet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(1, 8, 3, stride=1, padding=1)
        self.pool1 = MaxPool2D(2)
        self.conv2 = Conv2D(8, 16, 3, stride=1, padding=1)
        self.pool2 = MaxPool2D(2)
        self.fc = Linear(16*7*7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.pool1(x)
        x = self.conv2(x)
        x = x.relu()
        x = self.pool2(x)
        if hasattr(x, 'data') and hasattr(x.data, 'reshape') and len(x.data.shape) > 1:
            x = Tensor(x.data.reshape(x.data.shape[0], -1))
        x = self.fc(x)
        return x
nf_model = NFConvNet()
nf_params = [p for p in nf_model.parameters() if hasattr(p, 'data')]
nf_optimizer = Adam(nf_params, lr=0.001)
def nf_ce_loss_img(logits, targets):
    targets_onehot = np.eye(10, dtype=np.float32)[targets]
    return cross_entropy(logits, Tensor(targets_onehot))
batch_size = 64
for epoch in range(1, 4):
    perm = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[perm], y_train[perm]
    total_loss = 0.0
    for i in range(0, X_train.shape[0], batch_size):
        xb = X_train[i:i+batch_size]
        yb = y_train[i:i+batch_size]
        logits = nf_model(Tensor(xb, shape=xb.shape))
        loss = nf_ce_loss_img(logits, yb)
        total_loss += loss.data
        nf_optimizer.zero_grad()
        loss.backward()
        nf_optimizer.step()
    avg_loss = total_loss / (X_train.shape[0] / batch_size)
    if isinstance(avg_loss, np.ndarray):
        avg_loss = avg_loss.item() if avg_loss.size == 1 else float(np.mean(avg_loss))
    print(f"[nnetflow] Epoch {epoch} - Loss: {avg_loss:.4f}")
nf_logits_test = nf_model(Tensor(X_test, shape=X_test.shape))
nf_preds = np.argmax(nf_logits_test.data, axis=1)
nf_acc = (nf_preds == y_test).mean()
print(f"[nnetflow] Test Accuracy: {nf_acc:.4f}")

# PyTorch
class TorchConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(16*7*7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
torch_model = TorchConvNet()
torch_optimizer = optim.Adam(torch_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)
for epoch in range(1, 4):
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
    print(f"[PyTorch] Epoch {epoch} - Loss: {avg_loss:.4f}")
torch_model.eval()
with torch.no_grad():
    torch_preds = torch_model(X_test_t).argmax(dim=1).numpy()
torch_acc = (torch_preds == y_test).mean()
print(f"[PyTorch] Test Accuracy: {torch_acc:.4f}")
