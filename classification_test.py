import numpy as np
from sklearn.datasets import make_blobs
from nnetflow.nn import Tensor, Linear, Module, CrossEntropyLoss
from nnetflow.engine import Tensor 
import time

# Dataset
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)
X = X.astype(np.float32)
y_onehot = np.eye(3)[y]  # One-hot encoding

# Dataloader
def get_batches(X, y, batch_size=32):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for i in range(0, len(X), batch_size):
        idx = indices[i:i+batch_size]
        yield Tensor(X[idx]), Tensor(y[idx])

# Model
class MLPClassifier(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 16, activation='relu')
        self.fc2 = Linear(16, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Train
def train(model, X, y, epochs=20, lr=0.1):
    loss_fn = CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in get_batches(X, y):
            out = model(x_batch)
            loss = loss_fn(out, y_batch)

            for p in model.parameters():
                p.grad = np.zeros_like(p.data)
            loss.backward()

            for p in model.parameters():
                p.data -= lr * p.grad

            total_loss += loss.data
        print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}")

# Eval
def evaluate(model, X, y):
    logits = model(Tensor(X)).data
    preds = np.argmax(logits, axis=-1)
    true = np.argmax(y, axis=-1)
    acc = np.mean(preds == true)
    print(f"Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    model = MLPClassifier()
    train(model, X, y_onehot)
    evaluate(model, X, y_onehot)
