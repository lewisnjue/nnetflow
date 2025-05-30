# nnetflow

A minimal neural network framework with autodiff, inspired by micrograd and pytorch.

## Installation

```bash
pip install nnetflow
```

```bash
import numpy as np
from nnetflow.nn import Conv2D, MaxPool2D, Linear, MLP, CrossEntropyLoss, Module
from nnetflow.engine import Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# ----------- DataLoader -----------
def numpy_dataloader(batch_size=32, train=True):
    tf = transforms.Compose([
        transforms.ToTensor(),  # (0,1)
        transforms.Lambda(lambda x: x.numpy()),
    ])
    cifar = datasets.CIFAR10(root='./data', train=train, download=True, transform=tf)
    loader = DataLoader(cifar, batch_size=batch_size, shuffle=True)
    for imgs, labels in loader:
        imgs = imgs.numpy()
        labels = np.eye(10)[labels.numpy()]  # One-hot
        yield Tensor(imgs), Tensor(labels)

# ----------- Model Definition -----------
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
        B, C, H, W = x.data.shape
        x = x.reshape(B, C * H * W)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# ----------- Training Function -----------
def train(model, epochs=5, lr=0.01, batch_size=32):
    loss_fn = CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        start = time.time()

        for x, y in numpy_dataloader(batch_size=batch_size, train=True):
            out = model(x)
            loss = loss_fn(out, y)

            # Backward pass
            for p in model.parameters():
                p.grad = np.zeros_like(p.data)
            loss.backward()

            # SGD step
            for p in model.parameters():
                p.data -= lr * p.grad

            total_loss += loss.data.item() if hasattr(loss.data, 'item') else loss.data
            num_batches += 1

        print(f"[Epoch {epoch+1}] Loss: {total_loss/num_batches:.4f} Time: {time.time()-start:.2f}s")

# ----------- Accuracy Evaluation -----------
def evaluate(model):
    correct = 0
    total = 0
    for x, y in numpy_dataloader(train=False):
        out = model(x)
        preds = np.argmax(out.data, axis=-1)
        labels = np.argmax(y.data, axis=-1)
        correct += np.sum(preds == labels)
        total += x.data.shape[0]
    print(f"Accuracy: {(correct / total) * 100:.2f}%")

# ----------- Run Training -----------
if __name__ == "__main__":
    model = SimpleCNN()
    train(model, epochs=5, lr=0.01, batch_size=64)
    evaluate(model)

```


# ...

See the docs/ folder for more details.
