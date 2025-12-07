from nnetflow.engine import Tensor
from typing import List
from nnetflow.module import Module
class SGD(Module):
    """Stochastic Gradient Descent optimizer with optional momentum."""
    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0) -> None:
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [Tensor.zeros_like(p) for p in params]
    

    def forward(self, *args, **kwargs):
        return None

    def step(self) -> None:
        """Apply one optimization step to all parameters."""
        for i, param in enumerate(self.params):
            if param.requires_grad is False:
                continue # just for safety because some will not require grad in training model like batchnorm layer :) 
            if param.grad is None:
                continue
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * param.grad
                param.data += self.velocities[i].data
            else:
                param.data -= self.lr * param.grad

    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        for param in self.params:
            if hasattr(param, 'zero_grad'):
                param.zero_grad()

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.momentum})"
    def __str__(self) -> str:
        return f"SGD Optimizer: lr={self.lr}, momentum={self.momentum}"


class Adam(Module):
    """
    Adam optimizer.
    link to the paper: https://arxiv.org/abs/1412.6980
    """
    def __init__(self, params: List[Tensor], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [Tensor.zeros_like(p) for p in params]
        self.v = [Tensor.zeros_like(p) for p in params]
        self.t = 0
    
    def forward(self, *args, **kwargs):
        return None 
    

    def state_dict(self, prefix=""):
        state = {}
        state[f"{prefix}.lr"] = self.lr 
        state[f"{prefix}.beta1"] = self.beta1
        state[f"{prefix}.beta2"] = self.beta2
        state[f"{prefix}.eps"] = self.eps
        state[f"{prefix}.t"] = self.t
        for i, (m_i, v_i) in enumerate(zip(self.m, self.v)):
            state[f"{prefix}.m.{i}"] = m_i.data
            state[f"{prefix}.v.{i}"] = v_i.data
        return state

    def step(self) -> None:
        """Apply one optimization step to all parameters with bias correction."""
        self.t += 1
        for i, param in enumerate(self.params):
            if param.requires_grad is False:
                continue 
            if param.grad is None:
                continue
            self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * param.grad
            self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * (param.grad ** 2)
            m_hat = self.m[i].data / (1 - self.beta1 ** self.t)
            v_hat = self.v[i].data / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)

    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        for param in self.params:
            if hasattr(param, 'zero_grad'):
                param.zero_grad()
    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps})"
    def __str__(self) -> str:
        return f"Adam Optimizer: lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}"