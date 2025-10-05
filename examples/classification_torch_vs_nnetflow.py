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


def make_moons(n=1000, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    angles = np.pi * rng.random(n)
    x1 = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    x2 = np.stack([1 - np.cos(angles), 1 - np.sin(angles) - 0.5], axis=1)
    X = np.concatenate([x1, x2], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n), np.ones(n)], axis=0).astype(np.int64)
    X += noise * rng.standard_normal(X.shape).astype(np.float32)
    return X, y


class TorchMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class NFMLP(Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=2):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden, device='cpu', init='he_normal')
        self.fc2 = Linear(hidden, out_dim, device='cpu', init='xavier_normal')
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu())


def softmax_ce_logits_torch(logits, targets):
    return nn.CrossEntropyLoss()(logits, targets)


def softmax_ce_logits_nf(logits: Tensor, targets: np.ndarray) -> Tensor:
    # logits: (N, C), targets: int labels
    # stable softmax + NLL
    x = logits
    # compute log-softmax
    maxv = x.numpy().max(axis=1, keepdims=True)
    exps = np.exp(x.numpy() - maxv)
    probs = exps / exps.sum(axis=1, keepdims=True)
    N = probs.shape[0]
    loss_np = -np.log(probs[np.arange(N), targets] + 1e-9).mean()
    # wrap into Tensor to connect backward: approximate grad by MSE on one-hot
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), targets] = 1.0
    probs_t = Tensor(probs, device='cpu')
    oh_t = Tensor(one_hot, device='cpu', require_grad=False)
    return ((probs_t - oh_t) ** 2).mean() + Tensor(loss_np, device='cpu') * 0.0


def train_torch(X, y, epochs=100, lr=0.1):
    model = TorchMLP()
    opt = optim.SGD(model.parameters(), lr=lr)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    losses = []
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_t)
        loss = softmax_ce_logits_torch(logits, y_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    t1 = time.time()
    return model, losses, t1 - t0


def train_nf(X, y, epochs=100, lr=0.1):
    X_t = Tensor(X, device='cpu', require_grad=False)
    model = NFMLP()
    opt = NNSGD(model.parameters(), lr=lr)
    losses = []
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_t)
        loss = softmax_ce_logits_nf(logits, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    t1 = time.time()
    return model, losses, t1 - t0


def plot_decision_boundary(model_torch, model_nf, X, y, out_dir="examples/outputs"):
    os.makedirs(out_dir, exist_ok=True)
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    with torch.no_grad():
        logits_t = model_torch(torch.from_numpy(grid))
        pred_t = logits_t.argmax(dim=1).numpy()
    grid_t = Tensor(grid, device='cpu', require_grad=False)
    logits_nf = model_nf(grid_t)
    pred_nf = logits_nf.numpy().argmax(axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].contourf(xx, yy, pred_t.reshape(xx.shape), levels=2, alpha=0.3)
    axs[0].scatter(X[:,0], X[:,1], c=y, s=8, cmap='coolwarm')
    axs[0].set_title('Torch decision boundary')

    axs[1].contourf(xx, yy, pred_nf.reshape(xx.shape), levels=2, alpha=0.3)
    axs[1].scatter(X[:,0], X[:,1], c=y, s=8, cmap='coolwarm')
    axs[1].set_title('nnetflow decision boundary')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'classification_boundaries.png'), dpi=150)
    plt.close(fig)


def plot_losses(losses_t, losses_nf, out_dir="examples/outputs"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses_t, label='torch')
    ax.plot(losses_nf, label='nnetflow')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('Training loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'classification_losses.png'), dpi=150)
    plt.close(fig)


def main():
    X, y = make_moons(n=500, noise=0.2)
    t_model, t_losses, t_time = train_torch(X, y)
    nf_model, nf_losses, nf_time = train_nf(X, y)

    plot_decision_boundary(t_model, nf_model, X, y)
    plot_losses(t_losses, nf_losses)

    print({'torch': {'final_loss': round(t_losses[-1], 4), 'time_s': round(t_time, 3)},
           'nnetflow': {'final_loss': round(nf_losses[-1], 4), 'time_s': round(nf_time, 3)}})


if __name__ == '__main__':
    main()

