import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nnetflow.engine import Tensor
from nnetflow.nn import  mse_loss
from nnetflow.optim import Adam
import nnetflow.nn as nn

# Generate regression data
X, y = make_regression(n_samples=20000, n_features=10, noise=0.1)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X).astype(np.float32)
y = scaler_y.fit_transform(y.reshape(-1, 1)).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class SimpleModel(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = (self.fc1(x))
        x = x.relu()
        x = (self.fc2(x))
        x = x.relu()
        return self.fc3(x)


model = SimpleModel()



params = model.parameters()
optimizer = Adam(params, lr=0.01)

batch_size = 1024
epochs = 50


# Training loop
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        X_batch = Tensor(X_train[i:i + batch_size])
        y_batch = Tensor(y_train[i:i + batch_size])

        # Forward pass
        y_pred = model(X_batch)

        # Compute loss
        loss = mse_loss(y_pred, y_batch)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.data.item()}')
# Evaluation


y_test_tensor = Tensor(X_test)
y_pred_test = model(y_test_tensor)
test_loss = mse_loss(y_pred_test, Tensor(y_test))
print(f'Test Loss: {test_loss.data.item()}')