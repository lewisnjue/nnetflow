from .engine import Tensor as tensor 
import numpy as np

"""add optimizers here"""

class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
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




