import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from nnetflow.engine import Tensor
from nnetflow.layers import Linear
from nnetflow.module import Module
from nnetflow.optim import SGD as NNSGD


def make_regression(n=512, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 1)).astype(np.float32)
    y = (np.sin(X) + 0.3 * np.cos(2*X) + noise*rng.standard_normal((n,1))).astype(np.float32)
    return X, y


class TorchReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)


class NFReg(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(1, 64, device='cpu', init='he_normal')
        self.fc2 = Linear(64, 64, device='cpu', init='he_normal')
        self.fc3 = Linear(64, 1, device='cpu', init='xavier_normal')
    def forward(self, x: Tensor) -> Tensor:
        return self.fc3(self.fc2(self.fc1(x).relu()).relu())


def train_torch(X, y, epochs=200, lr=0.05):
    model = TorchReg()
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    losses = []
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    t1 = time.time()
    return model, losses, t1 - t0


def train_nf(X, y, epochs=200, lr=0.05):
    X_t = Tensor(X, device='cpu', require_grad=False)
    y_t = Tensor(y, device='cpu', require_grad=False)
    model = NFReg()
    opt = NNSGD(model.parameters(), lr=lr)
    losses = []
    def mse(a: Tensor, b: Tensor) -> Tensor:
        return ((a - b) ** 2).mean()
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = mse(pred, y_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    t1 = time.time()
    return model, losses, t1 - t0


def plot_curves(model_t, model_nf, X, y, losses_t, losses_nf, out_dir="examples/outputs"):
    os.makedirs(out_dir, exist_ok=True)
    xs = np.linspace(-3, 3, 400).astype(np.float32).reshape(-1,1)
    with torch.no_grad():
        pred_t = model_t(torch.from_numpy(xs)).numpy()
    pred_nf = model_nf(Tensor(xs, device='cpu', require_grad=False)).numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(X[:,0], y[:,0], s=8, alpha=0.5, label='data')
    axs[0].plot(xs, pred_t, label='torch')
    axs[0].plot(xs, pred_nf, label='nnetflow')
    axs[0].legend(); axs[0].set_title('Fit')

    axs[1].plot(losses_t, label='torch')
    axs[1].plot(losses_nf, label='nnetflow')
    axs[1].set_title('Loss'); axs[1].set_xlabel('epoch'); axs[1].set_ylabel('MSE')
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'regression_fit_and_loss.png'), dpi=150)
    plt.close(fig)


def main():
    X, y = make_regression()
    t_model, t_losses, t_time = train_torch(X, y)
    nf_model, nf_losses, nf_time = train_nf(X, y)
    plot_curves(t_model, nf_model, X, y, t_losses, nf_losses)
    print({'torch': {'final_loss': round(t_losses[-1], 4), 'time_s': round(t_time, 3)},
           'nnetflow': {'final_loss': round(nf_losses[-1], 4), 'time_s': round(nf_time, 3)}})


if __name__ == '__main__':
    main()

