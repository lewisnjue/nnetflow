from typing import List, Tuple
import numpy as np

class Tensor:
    def __init__(self, data: List[float] | np.ndarray, shape: Tuple = (1,), _children=(), _op=''):
        self.data = np.array(data).reshape(shape)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.shape = self.data.shape

    @staticmethod
    def _unbroadcast(grad, shape):
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
            if s_dim == 1 and g_dim != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __add__(self, other):
        assert isinstance(other, (float, Tensor)), "unsupported operation"
        other = other if isinstance(other, Tensor) else Tensor([other])
        out_shape = np.broadcast_shapes(self.data.shape, other.data.shape)
        out = Tensor((self.data + other.data).flatten().tolist(), shape=out_shape, _children=(self, other), _op='+')

        def _backward():
            self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __mul__(self, other):
        # Allow int as well as float and Tensor
        if isinstance(other, int):
            other = float(other)
        assert isinstance(other, (float, Tensor)), "unsupported operation"
        other = other if isinstance(other, Tensor) else Tensor([other])
        out_shape = np.broadcast_shapes(self.data.shape, other.data.shape)
        out = Tensor((self.data * other.data).flatten().tolist(), shape=out_shape, _children=(self, other), _op='*')

        def _backward():
            self.grad += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward

        return out

    def relu(self):
        out_data = np.where(self.data < 0, 0, self.data)
        out = Tensor(out_data.flatten().tolist(), shape=self.data.shape, _children=(self,), _op='ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def __matmul__(self, other):
        assert isinstance(other, Tensor), "unsupported operation"
        assert self.data.shape[-1] == other.data.shape[0], "shape mismatch"

        result = self.data @ other.data
        out = Tensor(result.flatten().tolist(), shape=result.shape, _children=(self, other), _op='@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        # Allow negation to work with int/float
        return self * -1.0

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"




