import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nnetflow.engine import Tensor
from nnetflow.nn import MLP, bce_loss
from nnetflow.optim import SGD

# Generate classification data
X, y = make_classification(n_classes=2, n_samples=20000000, n_features=20)
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)
y = y.astype(np.float32).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: 2 hidden layers, ReLU, sigmoid output
model = MLP(nin=20, nouts=[32, 16, 1], activation='relu', last_activation='sigmoid')
params = model.parameters()
optimizer = SGD(params, lr=0.01, momentum=0.9)

batch_size = 10000
epochs = 50
for epoch in range(1, epochs + 1):
    perm = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[perm], y_train[perm]
    total_loss = 0.0
    for i in range(0, X_train.shape[0], batch_size):
        xb = X_train[i:i+batch_size]
        yb = y_train[i:i+batch_size]
        preds = model(Tensor(xb, shape=xb.shape))
        loss = bce_loss(preds, Tensor(yb, shape=yb.shape))
        total_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / (X_train.shape[0] / batch_size)
    print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss.item():.4f}")

# Test
preds_test = model(Tensor(X_test, shape=X_test.shape)).data.squeeze()
preds_labels = (preds_test > 0.5).astype(np.int32)
accuracy = (preds_labels == y_test.squeeze()).mean()
print(f"Test Accuracy: {accuracy:.4f}")
