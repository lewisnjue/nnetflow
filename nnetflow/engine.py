from typing import List, Tuple
import numpy as np
from . import cuda

class Tensor:
    def __init__(self, data, shape: Tuple = None, _children=(), _op='', dtype=None, device=None):
        self.device = device or cuda.get_default_device()
        xp = cuda.get_array_module(self.device)
        # Always convert to array on the correct device
        if hasattr(xp, 'ndarray') and isinstance(data, xp.ndarray):
            arr = data
        elif hasattr(data, 'shape') and hasattr(data, 'dtype'):
            arr = cuda.as_device_array(data, self.device)
        else:
            arr = cuda.as_device_array(data, self.device)
        # Only reshape if shape is provided and not None and not equal to arr.shape
        if shape is not None and shape != arr.shape:
            arr = arr.reshape(shape)
        self.data = arr
        self.grad = xp.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.shape = self.data.shape

    def to(self, device):
        device = cuda.Device(device) if not isinstance(device, cuda.Device) else device
        if device == self.device:
            return self
        xp = cuda.get_array_module(device)
        if xp is not cuda.get_array_module(self.device):
            # Transfer between CPU/GPU
            if device.type == 'cuda':
                data = cuda.cp.asarray(self.data)
                grad = cuda.cp.asarray(self.grad)
            else:
                data = cuda.np.asarray(self.data)
                grad = cuda.np.asarray(self.grad)
        else:
            data = xp.asarray(self.data)
            grad = xp.asarray(self.grad)
        t = Tensor(data, shape=data.shape, _children=self._prev, _op=self._op, dtype=self.data.dtype, device=device)
        t.grad = grad
        return t

    @staticmethod
    def _unbroadcast(grad, shape):
        xp = cuda.get_array_module('cuda' if hasattr(grad, 'device') and getattr(grad, 'device', None) == 'cuda' else 'cpu')
        while len(grad.shape) > len(shape):
            grad = xp.sum(grad, axis=0)
        for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
            if s_dim == 1 and g_dim != 1:
                grad = xp.sum(grad, axis=i, keepdims=True)
        return grad

    def sum(self, axis=None, keepdims=False):
        xp = cuda.get_array_module(self.device)
        out_data = xp.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, _children=(self,), _op='sum', device=self.device)
        def _backward():
            grad = out.grad
            if axis is None:
                grad = xp.ones_like(self.data) * grad
            else:
                axes = axis if isinstance(axis, (tuple, list)) else (axis,)
                if not keepdims:
                    for ax in sorted(axes):
                        grad = xp.expand_dims(grad, axis=ax)
                grad = xp.broadcast_to(grad, self.data.shape)
            self.grad += grad
        out._backward = _backward
        return out

    def __add__(self, other):
        xp = cuda.get_array_module(self.device)
        other = other if isinstance(other, Tensor) else Tensor([other], device=self.device)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+', device=self.device)
        def _backward():
            self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        xp = cuda.get_array_module(self.device)
        assert isinstance(other, Tensor), "unsupported operation"
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@', device=self.device)
        def _backward():
            self_grad = xp.matmul(out.grad, xp.swapaxes(other.data, -1, -2))
            other_grad = xp.matmul(xp.swapaxes(self.data, -1, -2), out.grad)
            if len(self.data.shape) > 0 and self.data.shape[0] == 1 and out.data.shape[0] > 1:
                self_grad = xp.sum(self_grad, axis=0, keepdims=True)
            if len(other.data.shape) > 0 and other.data.shape[0] == 1 and out.data.shape[0] > 1:
                other_grad = xp.sum(other_grad, axis=0, keepdims=True)
            self.grad += self_grad
            other.grad += other_grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        xp = cuda.get_array_module(self.device)
        if isinstance(other, int): other = float(other)
        other = other if isinstance(other, Tensor) else Tensor([other], device=self.device)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*', device=self.device)
        def _backward():
            self.grad += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def relu(self):
        xp = cuda.get_array_module(self.device)
        out_data = xp.where(self.data < 0, 0, self.data)
        out = Tensor(out_data, _children=(self,), _op='ReLU', device=self.device)
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        xp = cuda.get_array_module(self.device)
        clipped = xp.clip(self.data, -60, 60)
        out_data = 1 / (1 + xp.exp(-clipped))
        out = Tensor(out_data, _children=(self,), _op='sigmoid', device=self.device)
        def _backward():
            self.grad += out.grad * out.data * (1 - out.data)
        out._backward = _backward
        return out

    def exp(self):
        xp = cuda.get_array_module(self.device)
        clipped = xp.clip(self.data, -60, 60)
        out_data = xp.exp(clipped)
        out = Tensor(out_data, _children=(self,), _op='exp', device=self.device)
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self, eps=1e-8):
        xp = cuda.get_array_module(self.device)
        safe_data = xp.clip(self.data, eps, None)
        out = Tensor(xp.log(safe_data), _children=(self,), _op='log', device=self.device)
        def _backward():
            self.grad += out.grad / safe_data
        out._backward = _backward
        return out

    def tanh(self):
        xp = cuda.get_array_module(self.device)
        out_data = xp.tanh(self.data)
        out = Tensor(out_data, _children=(self,), _op='tanh', device=self.device)
        def _backward():
            self.grad += out.grad * (1 - out.data**2)
        out._backward = _backward
        return out

    def __pow__(self, power):
        xp = cuda.get_array_module(self.device)
        assert isinstance(power, (int, float)), "only supports scalar powers"
        out_data = xp.power(self.data, power)
        out = Tensor(out_data, _children=(self,), _op='pow', device=self.device)
        def _backward():
            self.grad += out.grad * (power * xp.power(self.data, power - 1))
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(shape), _children=(self,), _op='reshape', device=self.device)
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def zero_grad(self):
        xp = cuda.get_array_module(self.device) if hasattr(self, 'device') else np
        self.grad = xp.zeros_like(self.data)
        # Explicitly break reference cycles to help GC
        self._backward = lambda: None
        self._prev = set()
        # If on GPU, try to free memory
        if hasattr(xp, 'get_default_memory_pool'):
            try:
                xp.get_default_memory_pool().free_bytes()
            except Exception:
                pass

    def backward(self, grad_clip=None):
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
            if np.isnan(v.grad).any():  # i dont want NaN to get away with it 
                print(f"NaN in gradients of node with op: {v._op}")
            v.grad = np.nan_to_num(v.grad, nan=0.0, posinf=1e5, neginf=-1e5)
            if grad_clip is not None:
                if isinstance(v.grad, np.ndarray) and np.issubdtype(v.grad.dtype, np.floating):
                    np.clip(v.grad, -grad_clip, grad_clip, out=v.grad)
                elif isinstance(v.grad, float):
                    v.grad = float(np.clip(v.grad, -grad_clip, grad_clip))
        # After backward, break reference cycles to help GC
        self._backward = lambda: None
        self._prev = set()
        # If on GPU, try to free memory
        xp = cuda.get_array_module(self.device) if hasattr(self, 'device') else np
        if hasattr(xp, 'get_default_memory_pool'):
            try:
                xp.get_default_memory_pool().free_bytes()
            except Exception:
                pass

    def __neg__(self): return self * -1.0
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __repr__(self): return f"Tensor(data={self.data}, grad={self.grad})"