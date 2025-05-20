from typing import List, Tuple
import numpy as np

class Tensor:
    def __init__(self, data: List[float] | np.ndarray, shape: Tuple = None, _children=(), _op=''):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        if shape is not None:
            self.data = self.data.reshape(shape)

        self.shape = self.data.shape
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        

    @staticmethod
    def _unbroadcast(grad, shape):
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
            if s_dim == 1 and g_dim != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def exp(self):
        out_data = np.exp(self.data)
        out = Tensor(out_data, _children=(self,), _op='exp')
        def _backward():
            # d/dx e^x = e^x
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def log(self):
        out_data = np.log(self.data)
        out = Tensor(out_data, _children=(self,), _op='log')
        def _backward():
            # d/dx ln(x) = 1/x
            self.grad += out.grad * (1 / self.data)
        out._backward = _backward
        return out
    def abs(self):
        out_data = np.abs(self.data)
        out = Tensor(out_data, _children=(self,), _op='abs')
        def _backward():
            # d/dx |x| = sign(x)
            self.grad += out.grad * np.sign(self.data)
        out._backward = _backward
        return out

    def log1p(self):
        out_data = np.log1p(self.data)
        out = Tensor(out_data, _children=(self,), _op='log1p')
        def _backward():
            # d/dx log(1+x) = 1/(1+x)
            self.grad += out.grad * (1.0 / (1.0 + self.data))
        out._backward = _backward
        return out



    def __add__(self, other):
        if isinstance(other,int):
            other = float(other)
        assert isinstance(other, (float, Tensor)), "unsupported operation"
        other = other if isinstance(other, Tensor) else Tensor([other])
        out_shape = np.broadcast_shapes(self.data.shape, other.data.shape)
        out = Tensor((self.data + other.data), _children=(self, other), _op='+')

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
        out = Tensor((self.data * other.data), _children=(self, other), _op='*')

        def _backward():
            self.grad += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supports scalar powers"
        out = Tensor(self.data ** power, _children=(self,), _op='pow')
        def _backward():
            self.grad += out.grad * (power * self.data ** (power - 1))
        out._backward = _backward
        return out

    # activation functions 
    def relu(self):
        out_data = np.where(self.data < 0, 0, self.data)
        # also here 
        out = Tensor(out_data, _children=(self,), _op='ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    # -----------
    def tanh(self):
        out_data = np.tanh(self.data)
        out = Tensor(out_data, _children=(self,), _op='tanh')
        def _backward():
            self.grad += out.grad * (1 - out.data**2)
        out._backward = _backward
        return out
    # ------ 
    def sigmoid(self):
        """
        Applies the sigmoid activation: sigmoid(x) = 1 / (1 + exp(-x)).
        """
        out_data = 1 / (1 + np.exp(-self.data))
        out = Tensor(out_data, _children=(self,), _op='sigmoid')
        def _backward():
            # derivative: sigmoid(x) * (1 - sigmoid(x))
            self.grad += out.grad * (out.data * (1 - out.data))
        out._backward = _backward
        return out





    def __matmul__(self, other):
        assert isinstance(other, Tensor), "unsupported operation"

        try:
            result = self.data @ other.data
            out = Tensor(result, _children=(self, other), _op='@')
        except Exception as e: 
            raise ValueError(e)

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def backward(self):
        if self.data.size != 1:
            raise RuntimeError("Backward can only be called on scalar outputs.")
        """
        this is building a topological sort
        """
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




