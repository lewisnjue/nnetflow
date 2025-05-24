import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from nnetflow.engine import Tensor
from nnetflow.nn import Module, Conv2D, MaxPool2D, Linear
from nnetflow.optim import Adam
from nnetflow.nn import cross_entropy

# --- Data Preparation (MNIST, but as numpy for nnetflow) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Use a much smaller subset for fast CPU testing
small_train_size = 500
small_test_size = 100
X_train = train_dataset.data.numpy().astype(np.float32)[:small_train_size] / 255.0
X_test = test_dataset.data.numpy().astype(np.float32)[:small_test_size] / 255.0
y_train = train_dataset.targets.numpy().astype(np.int64)[:small_train_size]
y_test = test_dataset.targets.numpy().astype(np.int64)[:small_test_size]

# Normalize as in PyTorch
X_train = (X_train - 0.1307) / 0.3081
X_test = (X_test - 0.1307) / 0.3081

# Reshape for Conv2D: (batch, channels, height, width)
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

# --- nnetflow Model ---
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
        # Flatten only if shape is correct
        if hasattr(x, 'data') and hasattr(x.data, 'reshape') and len(x.data.shape) > 1:
            x = Tensor(x.data.reshape(x.data.shape[0], -1))
        x = self.fc(x)
        return x


def nf_cross_entropy(logits, targets):
    # targets: (batch,) integer class labels
    # Convert to one-hot for nnetflow cross_entropy, ensure float32 dtype
    num_classes = logits.data.shape[1]
    targets_onehot = np.eye(num_classes, dtype=np.float32)[targets]
    return cross_entropy(logits, Tensor(targets_onehot))


nf_model = NFConvNet()
nf_params = nf_model.parameters()
nf_optimizer = Adam(nf_params, lr=0.001)
batch_size = 64
epochs = 3
for epoch in range(1, epochs + 1):
    perm = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[perm], y_train[perm]
    total_loss = 0.0
    for i in range(0, X_train.shape[0], batch_size):
        xb = X_train[i:i+batch_size]
        yb = y_train[i:i+batch_size]
        logits = nf_model(Tensor(xb, shape=xb.shape))
        loss = nf_cross_entropy(logits, yb)
        total_loss += loss.data
        nf_optimizer.zero_grad()
        loss.backward()
        nf_optimizer.step()
    avg_loss = total_loss / (X_train.shape[0] / batch_size)
    if isinstance(avg_loss, np.ndarray):
        avg_loss = avg_loss.item() if avg_loss.size == 1 else float(np.mean(avg_loss))
    print(f"[nnetflow] Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
# Test accuracy for nnetflow
nf_logits = nf_model(Tensor(X_test, shape=X_test.shape))
nf_preds = np.argmax(nf_logits.data, axis=1)
nf_accuracy = (nf_preds == y_test).mean()
print(f"[nnetflow] Test Accuracy: {nf_accuracy:.4f}")

# --- PyTorch Model ---
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
# Test accuracy for PyTorch
with torch.no_grad():
    torch_preds = torch_model(X_test_t).argmax(dim=1).numpy()
torch_accuracy = (torch_preds == y_test).mean()
print(f"[PyTorch] Test Accuracy: {torch_accuracy:.4f}")

print("\nComparison (Image Classification):")
print(f"[nnetflow] Test Accuracy: {nf_accuracy:.4f}")
print(f"[PyTorch]  Test Accuracy: {torch_accuracy:.4f}")
