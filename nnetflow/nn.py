import random
import numpy as np
from nnetflow.engine import Tensor

class Linear:
    def __init__(self, in_features: int, out_features: int, bias=True, dtype=None):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        weight = np.random.randn(out_features, in_features)
        if dtype:
            weight = weight.astype(dtype)
        # store as (in_features, out_features) for matmul compatibility
        self.weight = Tensor(weight.T)

        if bias:
            b = np.random.randn(out_features)
            if dtype:
                b = b.astype(dtype)
            self.bias = Tensor(b)
        else:
            self.bias = None
        del weight, b

    def __call__(self, x: Tensor):
        assert x.data.shape[-1] == self.in_features, "shape mismatch"
        out = x @ self.weight
        if self.bias:
            out = out + self.bias
        return out

# Numerically stable cross-entropy implemented with Tensor ops for proper autograd
class CrossEntropyLoss:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        # input: logits shape (batch, classes)
        # shift for numerical stability
        max_logits = Tensor(np.max(input.data, axis=-1, keepdims=True))
        shifted = input - max_logits
        exp_shifted = shifted.exp()
        sum_exp = exp_shifted.sum(axis=-1, keepdims=True)
        logsumexp = sum_exp.log()
        # log-probs: shifted - logsumexp
        log_probs = shifted - logsumexp
        # negative log-likelihood
        nll = -(target * log_probs).sum(axis=-1)
        # mean over batch
        return nll.sum() * (1.0 / input.data.shape[0])

# Functional wrapper
def cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    return CrossEntropyLoss()(input, target)


def softmax(input: Tensor, dim: int) -> Tensor:
    """Functional softmax: returns probabilities along specified dim"""
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
    """Binary Cross Entropy Loss as a callable class"""
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        data = np.clip(input.data, self.eps, 1 - self.eps)
        bce = -(target.data * np.log(data) + (1 - target.data) * np.log(1 - data))
        return Tensor(np.array(bce.mean()))


def bce_loss(input: Tensor, target: Tensor) -> Tensor:
    """Functional BCE loss"""
    return BCELoss()(input, target)
