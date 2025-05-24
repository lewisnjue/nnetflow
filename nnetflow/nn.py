import random
import numpy as np
from nnetflow.engine import Tensor

class Linear:
    def __init__(self, in_features: int, out_features: int, bias=True, dtype=None, activation=None):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.activation = activation
        # Use PyTorch-like Kaiming (He) initialization for ReLU, Xavier for tanh
        if activation == 'relu':
            std = np.sqrt(2.0 / in_features)
        elif activation == 'tanh':
            std = np.sqrt(1.0 / in_features)
        else:
            std = np.sqrt(2.0 / (in_features + out_features))
        weight = np.random.randn(out_features, in_features) * std
        if dtype:
            weight = weight.astype(dtype)
        self.weight = Tensor(weight.T)
        if bias:
            b = np.zeros(out_features, dtype=dtype if dtype else float)
            self.bias = Tensor(b)
        else:
            self.bias = None

    def __call__(self, x: Tensor):
        # Accept both 1D and 2D input, handle scalar edge case
        shape = x.data.shape
        if not shape or shape == ():
            raise ValueError("Input tensor has no shape (scalar), expected at least 1D array.")
        if len(shape) == 1:
            if shape[0] != self.in_features:
                raise ValueError(f"Shape mismatch: got {shape}, expected ({self.in_features},)")
        else:
            if shape[-1] != self.in_features:
                raise ValueError(f"Shape mismatch: got {shape}, expected (..., {self.in_features})")
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        if self.activation == 'relu':
            out = out.relu()
        elif self.activation == 'tanh':
            out = out.tanh()
        return out

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

class CrossEntropyLoss:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        max_logits = Tensor(np.max(input.data, axis=-1, keepdims=True))
        shifted = input - max_logits
        exp_shifted = shifted.exp()
        sum_exp = exp_shifted.sum(axis=-1, keepdims=True)
        logsumexp = sum_exp.log()
        log_probs = shifted - logsumexp
        nll = -(target * log_probs).sum(axis=-1)
        # Fix for scalar input
        if nll.data.shape == ():
            return nll
        return nll.sum() * (1.0 / nll.data.shape[0])

def cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    return CrossEntropyLoss()(input, target)

def softmax(input: Tensor, dim: int) -> Tensor:
    data = input.data
    shifted = data - np.max(data, axis=dim, keepdims=True)
    exp_data = np.exp(shifted)
    exp_sum = exp_data.sum(axis=dim, keepdims=True)
    probs = exp_data / exp_sum
    return Tensor(probs)

class Softmax:
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, input: Tensor) -> Tensor:
        return softmax(input, self.dim)

class BCELoss:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        # Ensure input is in (0,1) for log
        data = np.clip(input.data, self.eps, 1 - self.eps)
        # dL/dx = (x - y) / (x * (1-x))
        bce = -(target.data * np.log(data) + (1 - target.data) * np.log(1 - data))
        out = Tensor(np.array(bce.mean()), _children=(input, target), _op='bce')

        def _backward():
            # Gradient w.r.t. input: (input - target) / (input * (1-input) * N)
            grad = (data - target.data) / (data * (1 - data) * target.data.size)
            input.grad += grad.reshape(input.grad.shape)
            # No grad for target (labels)
        out._backward = _backward
        return out

def bce_loss(input: Tensor, target: Tensor) -> Tensor:
    return BCELoss()(input, target)

# === Regression Losses ===

class MSELoss:
    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        mse = (diff * diff).sum() * (1.0 / input.data.size)
        return mse

def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    return MSELoss()(input, target)

class RMSELoss:
    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        mse = (diff * diff).sum() * (1.0 / input.data.size)
        rmse = mse ** 0.5
        return rmse

def rmse_loss(input: Tensor, target: Tensor) -> Tensor:
    return RMSELoss()(input, target)

class MLP:
    def __init__(self, nin, nouts, activation='relu', last_activation=None):
        self.layers = []
        sz = [nin] + nouts
        for i in range(len(nouts)):
            act = activation if i < len(nouts) - 1 else last_activation
            self.layers.append(Linear(sz[i], sz[i+1], activation=act))
        self.last_activation = last_activation

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        # Apply last activation if specified
        if self.last_activation == 'sigmoid':
            x = x.sigmoid()
        elif self.last_activation == 'softmax':
            x = softmax(x, dim=-1)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
