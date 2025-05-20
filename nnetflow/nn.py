import random
import numpy as np
from typing import List, Optional
from nnetflow.engine import Tensor

class Module:
    def zero_grad(self) -> None:
        """Reset gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def parameters(self) -> List[Tensor]:
        """Return list of all trainable Tensors in the module."""
        return []

class Neuron(Module):
    """
    A single neuron performing a weighted sum plus bias,
    followed by an optional activation.

    activation: 'relu', 'sigmoid', or None for linear.
    """
    def __init__(self, nin: int, activation: Optional[str] = 'relu'):
        assert activation in ('relu', 'sigmoid', None), f"Unsupported activation: {activation}"
        self.w: List[Tensor] = [Tensor([random.uniform(-1, 1)]) for _ in range(nin)]
        self.b: Tensor = Tensor([0.0])
        self.activation = activation

    def __call__(self, x: List[Tensor]) -> Tensor:
        assert len(x) == len(self.w), "Input length mismatch"
        # Linear combination (fixed generator syntax issue)
        z = sum(((wi * xi) for wi, xi in zip(self.w, x)), self.b)
        # Activation
        if self.activation == 'relu':
            return z.relu()
        elif self.activation == 'sigmoid':
            return z.sigmoid()
        else:
            return z

    def parameters(self) -> List[Tensor]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        act = self.activation or 'linear'
        return f"Neuron(nin={len(self.w)}, activation={act})"

class Layer(Module):
    """
    Fully-connected layer: applies multiple neurons to the same input.
    """
    def __init__(self, nin: int, nout: int, activation: Optional[str] = 'relu'):
        self.neurons: List[Neuron] = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x: List[Tensor]) -> List[Tensor]:
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

    def __repr__(self) -> str:
        return f"Layer(nin={len(self.neurons[0].w)}, nout={len(self.neurons)}, activation={self.neurons[0].activation})"

class MLP(Module):
    """
    Multi-Layer Perceptron: stack of fully-connected layers.
    Hidden layers use ReLU by default; output layer is linear.
    """
    def __init__(self, nin: int, nouts: List[int]):
        sizes = [nin] + nouts
        self.layers: List[Layer] = []
        for i, nout in enumerate(nouts):
            act = 'relu' if i < len(nouts) - 1 else None
            self.layers.append(Layer(sizes[i], nout, activation=act))

    def __call__(self, x: List[Tensor]) -> List[Tensor]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self) -> str:
        sizes = [len(layer.neurons) for layer in self.layers]
        return f"MLP(layers={sizes})"

# ----- Loss functions -----
class MSELoss:
    """Mean Squared Error: (1/n) * sum((y_pred - y_true)^2)"""
    def __call__(self, pred: List[Tensor], target: List[Tensor]) -> Tensor:
        assert len(pred) == len(target), "Prediction/target length mismatch"
        losses = [(p - t) * (p - t) for p, t in zip(pred, target)]
        return sum(losses, Tensor([0.0])) * Tensor([1.0 / len(losses)])

class MAELoss:
    """Mean Absolute Error: (1/n) * sum(|y_pred - y_true|)"""
    def __call__(self, pred: List[Tensor], target: List[Tensor]) -> Tensor:
        assert len(pred) == len(target), "Prediction/target length mismatch"
        losses = [abs(p - t) for p, t in zip(pred, target)]
        return sum(losses, Tensor([0.0])) * Tensor([1.0 / len(losses)])

class BCEWithLogitsLoss:
    """Binary Cross-Entropy with logits. Input are raw logits."""
    def __call__(self, logits: List[Tensor], labels: List[float]) -> Tensor:
        assert len(logits) == len(labels), "Logits/labels length mismatch"
        losses: List[Tensor] = []
        for logit, y in zip(logits, labels):
            x = logit
            y_tensor = Tensor([y])
            # numerically stable BCE
            loss = x.relu() - x * y_tensor + ( (-x).exp() + Tensor([1.0]) ).log()
            losses.append(loss)
        return sum(losses, Tensor([0.0])) * Tensor([1.0 / len(losses)])

class CrossEntropyLoss:
    """Cross-entropy for multi-class with raw logits and integer labels."""
    def __call__(self, logits: List[List[Tensor]], labels: List[int]) -> Tensor:
        # Stable log-softmax: subtract max logit to prevent overflow
        batch_losses: List[Tensor] = []
        for logit_vec, label in zip(logits, labels):
            # find max logit value
            max_val = max([t.data.item() for t in logit_vec])
            # shift logits
            shifted = [l + Tensor([-max_val]) for l in logit_vec]
            # exponentiate shifted values (now <= 0)
            exps = [l.exp() for l in shifted]
            sum_exp = sum(exps, Tensor([0.0]))
            # log-softmax: log(exp(shifted)/sum_exp) = shifted - log(sum_exp)
            log_probs = [l - sum_exp.log() for l in shifted]
            # negative log-likelihood of the true class
            nll = log_probs[label] * Tensor([-1.0])
            batch_losses.append(nll)
        # mean over batch
        return sum(batch_losses, Tensor([0.0])) * Tensor([1.0 / len(batch_losses)])

# Optimizer
class SGD:
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = params
        self.lr = lr

    def step(self) -> None:
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = np.zeros_like(p.grad)
