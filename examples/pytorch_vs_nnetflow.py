import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from nnetflow.engine import Tensor
from nnetflow.layers import Linear
from nnetflow.module import Module
from nnetflow.optim import SGD as NNSGD


def make_data(n=512, in_dim=10, out_dim=1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, in_dim)).astype(np.float32)
    true_W = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    y = (X @ true_W + 0.1 * rng.standard_normal((n, out_dim))).astype(np.float32)
    return X, y


class TorchMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class NFMLP(Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden, device='cpu', init='he_normal')
        self.fc2 = Linear(hidden, out_dim, device='cpu', init='xavier_normal')

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu())


def train_torch(X, y, epochs=50, lr=0.1):
    model = TorchMLP(X.shape[1], 32, y.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    t0 = time.time()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
    t1 = time.time()
    return float(loss.item()), t1 - t0


def train_nf(X, y, epochs=50, lr=0.1):
    X_t = Tensor(X, device='cpu', require_grad=False)
    y_t = Tensor(y, device='cpu', require_grad=False)
    model = NFMLP(X.shape[1], 32, y.shape[1])
    optimizer = NNSGD(model.parameters(), lr=lr)

    def mse(pred: Tensor, target: Tensor) -> Tensor:
        return ((pred - target) ** 2).mean()

    t0 = time.time()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = mse(pred, y_t)
        loss.backward()
        optimizer.step()
    t1 = time.time()
    return float(loss.item()), t1 - t0


def main():
    X, y = make_data()
    torch_loss, torch_time = train_torch(X, y)
    nf_loss, nf_time = train_nf(X, y)
    print({
        'torch': {'loss': round(torch_loss, 6), 'time_s': round(torch_time, 4)},
        'nnetflow': {'loss': round(nf_loss, 6), 'time_s': round(nf_time, 4)},
    })


if __name__ == '__main__':
    main()

