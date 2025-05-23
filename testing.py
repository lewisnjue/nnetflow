import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from nnetflow.engine import Tensor
from nnetflow.nn import Linear, Softmax, cross_entropy
from nnetflow.optim import SGD


X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=4,
    random_state=42
)


num_classes = 4
y_onehot = np.eye(num_classes)[y]


X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)


lr = 0.01
epochs = 200
batch_size = 32

input_dim = X.shape[1]
hidden_dim = 16
model = [
    Linear(input_dim, hidden_dim),
    Linear(hidden_dim, num_classes)
]


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

        logits = forward(xb)
        loss = cross_entropy(logits, Tensor(yb))
        total_loss += loss.data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / (X_train.shape[0] / batch_size)
    print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss.item():.4f}")


logits_test = forward(X_test)
probs = Softmax(dim=-1)(logits_test).data
preds = np.argmax(probs, axis=1)
labels = np.argmax(y_test, axis=1)
accuracy = np.mean(preds == labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
