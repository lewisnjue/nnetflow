from .engine import Tensor as tensor 
import numpy as np
from . import cuda
class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        if params:
            xp = cuda.get_array_module(params[0].device if hasattr(params[0], 'device') else 'cpu')
            if xp is not None and hasattr(xp, 'zeros_like'):
                self.velocities = [xp.zeros_like(p.data) for p in params]
            else:
                self.velocities = [np.zeros_like(p.data) for p in params]
        else:
            self.velocities = []
    def step(self):
        for i, p in enumerate(self.params):
            xp = cuda.get_array_module(p.device if hasattr(p, 'device') else 'cpu')
            if self.momentum:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * p.grad
                p.data += self.velocities[i]
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
        if params:
            xp = cuda.get_array_module(params[0].device if hasattr(params[0], 'device') else 'cpu')
            if xp is not None and hasattr(xp, 'zeros_like'):
                self.m = [xp.zeros_like(p.data) for p in params]
                self.v = [xp.zeros_like(p.data) for p in params]
            else:
                self.m = [np.zeros_like(p.data) for p in params]
                self.v = [np.zeros_like(p.data) for p in params]
        else:
            self.m = []
            self.v = []
        self.t = 0
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            xp = cuda.get_array_module(p.device if hasattr(p, 'device') else 'cpu')
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (p.grad ** 2)
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            sqrt_fn = getattr(xp, 'sqrt', np.sqrt)
            p.data -= self.lr * m_hat / (sqrt_fn(v_hat) + self.eps)
    def zero_grad(self):
        for p in self.params:
            p.zero_grad()




