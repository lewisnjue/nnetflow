from .engine import Tensor as tensor 
import numpy as np
from typing import List

"""add optimizers here"""

class SGD:
    def __init__(self, params:List[tensor], lr=0.01, momentum=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in params]
    def step(self):
        for i, p in enumerate(self.params):
            if self.momentum:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * p.grad
                p.data += self.velocities[i]  # PyTorch uses += for momentum update
            else:
                p.data -= self.lr * p.grad
    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (p.grad ** 2)
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    def zero_grad(self):
        for p in self.params:
            p.zero_grad()




