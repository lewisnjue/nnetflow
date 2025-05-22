import random
import numpy as np
from nnetflow.engine import Tensor



"""start here not impremented"""


class Linear:
    def __init__(self, in_features: int, out_features: int, bias=True, dtype=None):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        weight = np.random.randn(out_features, in_features)

        if dtype:
            weight = weight.astype(dtype)
        self.weight = Tensor(weight.T) 


        if bias:
            b = np.random.randn(out_features)
            if dtype:
                b = b.astype(dtype)
            self.bias = Tensor(b)
            del b
        else:
            self.bias = None
        
        del weight

    def __call__(self,x:Tensor):
        assert x.data.shape[-1] == self.in_features , "shape mismatch"

        if self.bias:
            return x @ self.weight + self.bias
        else:
            return x @ self.weight 
            





"""end here """

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Tensor([random.uniform(-1, 1)]) for _ in range(nin)]
        self.b = Tensor([0.0])  # Bias as Tensor with shape (1,)
        self.nonlin = nonlin

    def __call__(self, x):
        assert len(x) == len(self.w), "Input size mismatch"
        act = sum([wi * xi for wi, xi in zip(self.w, x)], self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i != len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


# --------------------

class MSELoss:
    def __call__(self, pred, target):
        # Mean Squared Error: (1/n) * sum((y_pred - y_true)^2)
        assert isinstance(pred, list) and isinstance(target, list), "Inputs must be lists of Tensors"
        assert len(pred) == len(target), "Mismatch in prediction and target length"
        
        losses = [(p - t) * (p - t) for p, t in zip(pred, target)]
        return sum(losses, Tensor([0.0])) * Tensor([1.0 / len(losses)])


class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.grad)


# -----------



