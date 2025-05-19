import numpy as np
from sklearn.datasets import make_regression
from chainflow.nn import MLP, SGD, MSELoss
from chainflow.engine import Tensor

X, Y = make_regression(n_samples=100, n_features=3, n_targets=2, noise=0.1, random_state=42)


data = []
for x, y in zip(X, Y):
    x_tensors = [Tensor([float(val)]) for val in x]
    y_tensors = [Tensor([float(val)]) for val in y]
    data.append((x_tensors, y_tensors))

model = MLP(nin=3, nouts=[8,16,16,4, 2])

"""
if you want a model with 2 hidden layers, you can change the nouts to [8, 8, 2]
and the number of neurons in the hidden layers to 8 
"""


loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.001)  # Lower learning rate to avoid overflow

for epoch in range(200):
    total_loss = 0.0
    for x, y_true in data:
        y_pred = model(x)
        if not isinstance(y_pred, list):
            y_pred = [y_pred]
        loss = loss_fn(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Gradient check: finite difference for first parameter
param = model.parameters()[0]
orig = param.data.copy()
eps = 1e-4
param.data += eps
plus_loss = 0.0
for x, y_true in data:
    y_pred = model(x)
    if not isinstance(y_pred, list):
        y_pred = [y_pred]
    plus_loss += loss_fn(y_pred, y_true).data.item()
param.data = orig - eps
minus_loss = 0.0
for x, y_true in data:
    y_pred = model(x)
    if not isinstance(y_pred, list):
        y_pred = [y_pred]
    minus_loss += loss_fn(y_pred, y_true).data.item()
param.data = orig
fd_grad = (plus_loss - minus_loss) / (2 * eps)
# Backprop gradient
optimizer.zero_grad()
for x, y_true in data:
    y_pred = model(x)
    if not isinstance(y_pred, list):
        y_pred = [y_pred]
    loss = loss_fn(y_pred, y_true)
    loss.backward()
print(f"Finite diff grad: {fd_grad:.6f}, Backprop grad: {param.grad.flatten()[0]:.6f}")

# Test on a random sample from the dataset
x_test = [Tensor([float(val)]) for val in X[0]]
y_pred = model(x_test)
print("Prediction:", [p.data for p in y_pred])
print("True:", Y[0])
