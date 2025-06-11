from typing import List, Tuple, Union
import numpy as np

class Tensor:
    def __init__(self, data: np.ndarray, _children = (), _op: str = '') -> None:
        self.data = data
        self.dtype = data.dtype
        self.shape = self.data.shape
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    @staticmethod
    def _unbroadcast(grad, shape) -> np.ndarray:
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
            if s_dim == 1 and g_dim != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, _children=(self,), _op='sum')
        def _backward():
            grad = out.grad
            if axis is None:
                grad = np.ones_like(self.data) * grad
            else:
                axes = axis if isinstance(axis, (tuple, list)) else (axis,)
                if not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False) -> 'Tensor':
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, _children=(self,), _op='mean')
        def _backward():
            grad = out.grad / np.prod(self.data.shape) if axis is None else np.prod([self.data.shape[i] for i in (axis if isinstance(axis, (tuple, list)) else [axis])])
            if not keepdims:
                grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad
        out._backward = _backward
        return out
    
    def var(self, axis=None, keepdims=False) -> 'Tensor':
        out_data = self.data.var(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, _children=(self,), _op='var')

        def _backward():
            n = np.prod(self.data.shape) if axis is None else np.prod([self.data.shape[i] for i in (axis if isinstance(axis, (tuple, list)) else [axis])])
            grad = out.grad / n
            if not keepdims:
                grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad
        out._backward = _backward
        return out
    
    def std(self, axis=None, keepdims=False) -> 'Tensor':
        out_data = self.data.std(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, _children=(self,), _op='std')

        def _backward():
            n = np.prod(self.data.shape) if axis is None else np.prod([self.data.shape[i] for i in (axis if isinstance(axis, (tuple, list)) else [axis])])
            grad = out.grad / (n * np.sqrt(np.var(self.data, axis=axis, keepdims=keepdims) + 1e-8))
            if not keepdims:
                grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad
        out._backward = _backward
        return out
    
    def __add__(self, other: 'Tensor') -> 'Tensor':
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')
        def _backward():
            self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other:'Tensor') -> 'Tensor':
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@')
        def _backward():
            self.grad =  np.swapaxes(other.data,-1,-2) @ out.grad 
            other.grad = np.swapaxes(self.data,-1,-2) @ out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')
        def _backward():
            self.grad += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out
    
    def __pow__(self, power: Union[int, float]) -> 'Tensor':
        out_data = np.power(self.data, power)
        out = Tensor(out_data, _children=(self,), _op='pow')

        def _backward():
            self.grad += out.grad * (power * np.power(self.data, power - 1))
        out._backward = _backward
        return out

    def __truediv__(self, other: Union[float, int, 'Tensor']) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        return self * (other ** -1)

    def reshape(self, *shape) -> 'Tensor':
        out = Tensor(self.data.reshape(shape), _children=(self,), _op='reshape')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out 
    
    def permute(self, *dims):
        out = Tensor(self.data.transpose(dims), _children=(self,), _op='permute')
        def _backward():
            self.grad += out.grad.transpose(np.argsort(dims))
        out._backward = _backward

        return out
    def mask_fill(self, mask:'Tensor', value=0):
        out_data = np.where(mask.data, value, self.data)
        out = Tensor(out_data, _children=(self, mask), _op='mask_fill')
        def _backward():
            self.grad += out.grad * mask.data
            mask.grad += out.grad * (self.data == value)
        out._backward = _backward
        return out
    

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

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
            if np.isnan(v.grad).any():  
                print(f"NaN in gradients of node with op: {v._op}")
            v.grad = np.nan_to_num(v.grad, nan=0.0, posinf=1e5, neginf=-1e5)
            if grad_clip is not None:
                if isinstance(v.grad, np.ndarray) and np.issubdtype(v.grad.dtype, np.floating):
                    np.clip(v.grad, -grad_clip, grad_clip, out=v.grad)
                elif isinstance(v.grad, float):
                    v.grad = float(np.clip(v.grad, -grad_clip, grad_clip))


    def __getitem__(self, idx: Union[int, slice, Tuple]) -> 'Tensor':
        if isinstance(idx, tuple) or isinstance(idx, int) or isinstance(idx, slice) or (hasattr(idx, '__iter__') and not isinstance(idx, str)):
            out = Tensor(self.data[idx], _children=(self,), _op='slice')
            def _backward():
                grad = np.zeros_like(self.data)
                grad_idx = grad[idx]
                if grad_idx.shape == out.grad.shape:
                    grad[idx] = out.grad
                elif out.grad.shape[0] < grad_idx.shape[0] and grad_idx.shape[1:] == out.grad.shape[1:]:
                    grad[idx] = out.grad.sum(axis=0)
                else:
                    grad[idx] = np.broadcast_to(out.grad, grad_idx.shape)
                self.grad += grad
            out._backward = _backward
            return out
        raise TypeError(f"Invalid index type: {type(idx)}")
    
    def item(self) -> float:
        if self.data.size != 1:
            raise ValueError("Cannot convert a tensor with more than one element to a scalar.")
        return float(self.data.item())


    def __repr__(self): return f"Tensor(data={self.data}, grad={self.grad})"
    def __neg__(self): return self * -1.0
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other

