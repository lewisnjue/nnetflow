# Usage Guide

nnetflow is designed for easy experimentation with neural networks and autodiff. Below are typical usage patterns for regression, classification, and image tasks. All code is comment-free for clarity.

## Basic Regression Example

```python
import numpy as np
from nnetflow.engine import Tensor
from nnetflow.nn import MLP, mse_loss
from nnetflow.optim import Adam
from nnetflow import cuda
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = make_regression(n_samples=2000, n_features=10, noise=0.1)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X).astype(np.float32)
y = scaler_y.fit_transform(y.reshape(-1, 1)).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
device = 'cuda' if cuda.is_available() else 'cpu'
model = MLP(nin=10, nouts=[32, 16, 1], activation='relu').to(device)
optimizer = Adam(model.parameters(), lr=0.01)
batch_size = 128
epochs = 10
for epoch in range(epochs):
    perm = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[perm], y_train[perm]
    for i in range(0, X_train.shape[0], batch_size):
        xb = X_train[i:i+batch_size]
        yb = y_train[i:i+batch_size]
        preds = model(Tensor(xb, shape=xb.shape, device=device))
        loss = mse_loss(preds, Tensor(yb, shape=yb.shape, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
preds_test = model(Tensor(X_test, shape=X_test.shape, device=device)).data
cp = getattr(cuda, 'cp', None)
if cp is not None and isinstance(preds_test, cp.ndarray):
    preds_test = cp.asnumpy(preds_test)
mse = np.mean((preds_test - y_test) ** 2)
print(f"Test MSE: {mse:.4f}")
```

## Classification Example

```python
import numpy as np
from sklearn.datasets import make_blobs
from nnetflow.nn import Tensor, Linear, Module, CrossEntropyLoss
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)
X = X.astype(np.float32)
y_onehot = np.eye(3)[y]
def get_batches(X, y, batch_size=32):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for i in range(0, len(X), batch_size):
        idx = indices[i:i+batch_size]
        yield Tensor(X[idx]), Tensor(y[idx])
class MLPClassifier(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 16, activation='relu')
        self.fc2 = Linear(16, 3)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
model = MLPClassifier()
for epoch in range(10):
    for x_batch, y_batch in get_batches(X, y_onehot):
        out = model(x_batch)
        loss = CrossEntropyLoss()(out, y_batch)
        for p in model.parameters():
            p.grad = np.zeros_like(p.data)
        loss.backward()
        for p in model.parameters():
            p.data -= 0.1 * p.grad
logits = model(Tensor(X)).data
preds = np.argmax(logits, axis=-1)
true = np.argmax(y_onehot, axis=-1)
acc = np.mean(preds == true)
print(f"Accuracy: {acc * 100:.2f}%")
```

## Image Classification (CIFAR-10)

```python
import numpy as np
from nnetflow.nn import Conv2D, MaxPool2D, Linear, Module, CrossEntropyLoss, softmax
from nnetflow.engine import Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from nnetflow.optim import Adam
def numpy_dataloader(batch_size=32, train=True):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy()),
    ])
    cifar = datasets.CIFAR10(root='./data', train=train, download=True, transform=tf)
    loader = DataLoader(cifar, batch_size=batch_size, shuffle=True)
    for imgs, labels in loader:
        imgs = imgs.numpy()
        yield Tensor(imgs), Tensor(labels.numpy().astype(int))
class SimpleCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(3, 8, kernel_size=3, stride=1, padding=1)
        self.pool = MaxPool2D(kernel_size=2)
        self.conv2 = Conv2D(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = Linear(16 * 8 * 8, 64, activation='relu')
        self.fc2 = Linear(64, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        B, C, H, W = tuple(x.data.shape)
        x = x.reshape(B, C * H * W)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
def train(model, epochs=5, lr=0.01, batch_size=32):
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for x, y in numpy_dataloader(batch_size=batch_size, train=True):
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
def evaluate(model):
    correct = 0
    total = 0
    for x, y in numpy_dataloader(train=False):
        out = model(x)
        out = softmax(out, dim=-1)
        preds = np.argmax(out.data, axis=-1)
        labels = y.data.astype(int)
        correct += np.sum(preds == labels)
        total += x.data.shape[0] if len(x.data.shape) > 0 else x.data.size
    print(f"Accuracy: {(correct / total) * 100:.2f}%")
```

## Device Management

nnetflow supports both CPU and GPU (CuPy) automatically. Use `.to('cuda')` or `.to('cpu')` on models and tensors. All operations and training loops will use the selected device.

---
