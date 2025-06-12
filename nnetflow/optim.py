from .engine import Tensor
from typing import List
class SGD:
    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0) -> None:
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [p.zeros_like() for p in params]

    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * param.grad
                param.data += self.velocities[i].data
            else:
                param.data -= self.lr * param.grad

    def zero_grad(self) -> None:
        for param in self.params:
            if hasattr(param, 'zero_grad'):
                param.zero_grad()

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.momentum})"
    def __str__(self) -> str:
        return f"SGD Optimizer: lr={self.lr}, momentum={self.momentum}"

class Adam:
    def __init__(self, params: List[Tensor], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [p.zeros_like() for p in params]
        self.v = [p.zeros_like() for p in params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            # Use np or cp depending on param.device
            xp = __import__('cupy') if hasattr(param.data, 'get') else __import__('numpy')
            self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * param.grad
            self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * (param.grad ** 2)
            m_hat = self.m[i].data / (1 - self.beta1 ** self.t)
            v_hat = self.v[i].data / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for param in self.params:
            if hasattr(param, 'zero_grad'):
                param.zero_grad()
    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps})"
    def __str__(self) -> str:
        return f"Adam Optimizer: lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}"