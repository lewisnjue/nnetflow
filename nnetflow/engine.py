import numpy as np
import cupy as cp
from typing import Union, List, Literal, Tuple, Optional
from .utils import is_cuda_available

DEVICE  = 'cuda'  if is_cuda_available() else 'cpu' # i want to use this as the default device
class Tensor:
    @staticmethod
    def _noop_backward():
        pass
    def __init__(
        self,
        data: Union[List[float], List[int], np.ndarray, cp.ndarray,int,float],
        _children: Tuple['Tensor', ...] = (),
        _op: str = '',
        device: str = DEVICE,
        dtype: str = 'float32',
        require_grad: bool = True
    ) -> None:
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=dtype) if device == 'cpu' else cp.array(data, dtype=dtype)
            _children = ()
        self.require_grad = require_grad
        if not self.require_grad:
            assert _children == (), "Children must be empty if require_grad is False"
            self._children = () 
            self.grad = None
        else:
            self._children = _children
            self.grad = np.zeros_like(data) if device == 'cpu' else cp.zeros_like(data)
        self.device = device
        self.dtype = dtype
        self._op = _op
        self.require_grad = require_grad
        # Tensor data initialization
        if self.device == 'cpu':
            if isinstance(data, list):
                data = np.array(data, dtype=self.dtype)
            elif isinstance(data, np.ndarray):
                data = data.astype(self.dtype)
            elif np.isscalar(data):
                data = np.array(data, dtype=self.dtype)
            else:
                raise TypeError("Data must be a list or np.ndarray for CPU tensors.")
            self.data = data
        else:
            if isinstance(data, list):
                data = cp.array(data, dtype=self.dtype)
            elif isinstance(data, np.ndarray):
                data = cp.asarray(data, dtype=self.dtype)
            elif isinstance(data, cp.ndarray):
                data = data.astype(self.dtype)
            elif cp.isscalar(data):
                data = cp.array(data, dtype=self.dtype)
            else:
                raise TypeError("Data must be a list, np.ndarray, or cp.ndarray for CUDA tensors.")
            self.data = data
        self.data: np.ndarray | cp.ndarray = self.data
        self.grad: np.ndarray | cp.ndarray | None = self.grad
        self.shape = self.data.shape
        self._backward = Tensor._noop_backward

    @staticmethod
    def _unbroadcast(
        grad: Union[np.ndarray, cp.ndarray],
        shape: Tuple[int, ...]
    ) -> Union[np.ndarray, cp.ndarray]:
        # Sum out broadcasted dims
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, (g, s) in enumerate(zip(grad.shape, shape)):
            if s == 1 and g != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    @staticmethod
    def zeros(shape: Tuple[int, ...], device: str = DEVICE, dtype: str = 'float32', require_grad: bool = True) -> 'Tensor':
        if device == 'cpu':
            data = np.zeros(shape, dtype=dtype)
        else:
            data = cp.zeros(shape, dtype=dtype)
        return Tensor(data, (), 'zeros', device, dtype, require_grad=require_grad)

    @staticmethod
    def ones(shape: Tuple[int, ...], device: str = DEVICE, dtype: str = 'float32', require_grad: bool = True) -> 'Tensor':
        if device == 'cpu':
            data = np.ones(shape, dtype=dtype)
        else:
            data = cp.ones(shape, dtype=dtype)
        return Tensor(data, (), 'ones', device, dtype, require_grad=require_grad)
    
    @property
    def requires_grad(self) -> bool:
        return self.require_grad
    
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        if not value:
            self.grad = None
            self._children = ()
        self.require_grad = value

    def zero_grad(self) -> None:
        if self.require_grad:
            if self.device == 'cpu':
                self.grad = np.zeros_like(self.data)
            else:
                self.grad = cp.zeros_like(self.data)
        else:
            raise RuntimeError("Cannot zero_grad on a tensor that does not require gradients.")
        
    @property
    def T(self) -> 'Tensor':
        out_data = self.data.T
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'transpose', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    self.grad += out.grad.T
            out._backward = _backward
        return out
    

    def backward(self, grad_clip: Optional[float] = None) -> None:
        topo, visited = [], set()

        def build(v: 'Tensor'):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build(child)
                topo.append(v)
        build(self)

        # Seed gradient
        if self.device == 'cpu':
            self.grad = np.ones_like(self.data)
        else:
            self.grad = cp.ones_like(self.data)

        # Backprop
        for v in reversed(topo):
            v._backward()
            # Safe-guard
            if v.device == 'cpu':
                v.grad = np.nan_to_num(v.grad, nan=0.0, posinf=1e5, neginf=-1e5)
                if grad_clip is not None:
                    np.clip(v.grad, -grad_clip, grad_clip, out=v.grad)
            else:
                v.grad = cp.nan_to_num(v.grad, nan=0.0, posinf=1e5, neginf=-1e5)
                if grad_clip is not None:
                    cp.clip(v.grad, -grad_clip, grad_clip, out=v.grad)

    def to(self, device: Literal['cpu', 'cuda']) -> 'Tensor':
        data = self.data
        if device == self.device:
            return self
        if device == 'cpu':
            arr = cp.asnumpy(data) if isinstance(data, cp.ndarray) else data
            return Tensor(arr, device='cpu', dtype=self.dtype)
        else:
            arr = cp.asarray(data) if isinstance(data, np.ndarray) else data
            return Tensor(arr, device='cuda', dtype=self.dtype)
    
    def cpu(self) -> 'Tensor':
        return self.to('cpu')
    
    def cuda(self) -> 'Tensor':
        return self.to('cuda')
    
    def numpy(self) -> np.ndarray:
        if self.device == 'cpu':
            return np.asarray(self.data)
        else:
            return cp.asnumpy(self.data)

    def sum(self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> 'Tensor':
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,) if self.require_grad else (), 'sum', self.device, self.dtype, require_grad=self.require_grad)
        n = int(np.prod(self.shape)) if axis is None else int(np.prod([self.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))]))
        if self.require_grad:
            def _backward():
                if out.grad is not None:
                    grad = out.grad / n
                    if axis is None:
                        grad = (np.ones_like(self.data) if self.device=='cpu' else cp.ones_like(self.data)) * grad
                    else:
                        if not keepdims:
                            shape_bd = list(self.shape)
                            for ax in (axis if isinstance(axis, tuple) else (axis,)):
                                shape_bd[ax] = 1
                            grad = grad.reshape(shape_bd)
                        grad = (np.broadcast_to if self.device=='cpu' else cp.broadcast_to)(grad, self.shape)
                    if self.grad is not None:
                        self.grad += grad
            out._backward = _backward
        return out

    def mean(self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> 'Tensor':
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,) if self.require_grad else (), 'mean', self.device, self.dtype, require_grad=self.require_grad)
        
        n = int(np.prod(self.shape)) if axis is None else int(np.prod([self.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))]))
        if self.require_grad:
            def _backward():
                if out.grad is not None:
                    grad = out.grad / n
                    if axis is None:
                        grad = (np.ones_like(self.data) if self.device=='cpu' else cp.ones_like(self.data)) * grad
                    else:
                        if not keepdims:
                            shape_bd = list(self.shape)
                            for ax in (axis if isinstance(axis, tuple) else (axis,)):
                                shape_bd[ax] = 1
                            grad = grad.reshape(shape_bd)
                        grad = (np.broadcast_to if self.device=='cpu' else cp.broadcast_to)(grad, self.shape)
                    if self.grad is not None:
                        self.grad += grad
            out._backward = _backward
        return out


    def var(self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> 'Tensor':
        out_data = self.data.var(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,) if self.require_grad else (), 'var', self.device, self.dtype, require_grad=self.require_grad)
        if self.require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    mu = self.data.mean(axis=axis, keepdims=True)
                    n = int(np.prod(self.shape)) if axis is None else int(np.prod([self.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))]))
                    diff = self.data - mu
                    grad = out.grad * (2.0 / n) * diff
                    if not keepdims:
                        grad = (np.broadcast_to if self.device=='cpu' else cp.broadcast_to)(grad, self.shape)
                    self.grad += grad
            out._backward = _backward
        return out

    def std(self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> 'Tensor':
        out_data = self.data.std(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,) if self.require_grad else (), 'std', self.device, self.dtype, require_grad=self.require_grad)
        if self.require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    mu = self.data.mean(axis=axis, keepdims=True)
                    stdv = out.data
                    n = int(np.prod(self.shape)) if axis is None else int(np.prod([self.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))]))
                    diff = self.data - mu
                    grad = out.grad * diff / (n * stdv + 1e-8)
                    if not keepdims:
                        grad = (np.broadcast_to if self.device=='cpu' else cp.broadcast_to)(grad, self.shape)
                    self.grad += grad
            out._backward = _backward
        return out
    
    def softmax(self, axis: int = -1) -> 'Tensor':
        if self.device == 'cpu':
            max_val = np.max(self.data, axis=axis, keepdims=True)
            exp_data = np.exp(self.data - max_val)
            out_data = exp_data / np.sum(exp_data, axis=axis, keepdims=True)
        else:
            max_val = cp.max(self.data, axis=axis, keepdims=True)
            exp_data = cp.exp(self.data - max_val)
            out_data = exp_data / cp.sum(exp_data, axis=axis, keepdims=True)
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'softmax', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    # Compute the gradient of softmax
                    grad = out.grad * out.data * (1 - out.data)
                    if axis != -1:
                        grad = np.moveaxis(grad, -1, axis) if self.device == 'cpu' else cp.moveaxis(grad, -1, axis)
                    self.grad += grad.sum(axis=axis, keepdims=True)
            out._backward = _backward
        return out
    
    def log(self) -> 'Tensor':
        if self.device == 'cpu':
            out_data = np.log(self.data)
        else:
            out_data = cp.log(self.data)
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'log', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    self.grad += out.grad / self.data
            out._backward = _backward
        return out
    
    def abs(self) -> 'Tensor':
        if self.device == 'cpu':
            out_data = np.abs(self.data)
        else:
            out_data = cp.abs(self.data)
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'abs', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    self.grad += out.grad * np.sign(self.data) if self.device == 'cpu' else cp.sign(self.data)
            out._backward = _backward
        return out
    
    def zeros_like(self) -> 'Tensor':
        if self.device == 'cpu':
            out_data = np.zeros_like(self.data, dtype=self.dtype)
        else:
            out_data = cp.zeros_like(self.data, dtype=self.dtype)
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'zeros_like', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    self.grad += out.grad
            out._backward = _backward
        return out

    def ones_like(self) -> 'Tensor':
        if self.device == 'cpu':
            out_data = np.ones_like(self.data, dtype=self.dtype)
        else:
            out_data = cp.ones_like(self.data, dtype=self.dtype)
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'ones_like', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    self.grad += out.grad
            out._backward = _backward
        return out
    
    def exp(self) -> 'Tensor':
        if self.device == 'cpu':
            out_data = np.exp(self.data)
        else:
            out_data = cp.exp(self.data)
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'exp', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    self.grad += out.grad * out.data
            out._backward = _backward
        return out

    def relu(self) -> 'Tensor':
        if self.device == 'cpu':
            out_data = np.maximum(self.data, 0)
        else:
            out_data = cp.maximum(self.data, 0)
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'relu', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    mask = (self.data > 0).astype(self.data.dtype)
                    self.grad += out.grad * mask
            out._backward = _backward
        return out

    def __add__(self, other: 'Tensor') -> 'Tensor':
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = other
        out = Tensor(self.data + other_data, (self, other) if isinstance(other, Tensor) else (self,), '+', self.device, self.dtype, require_grad=self.require_grad or (other.require_grad if isinstance(other, Tensor) else False))
        if out.require_grad:
            def _backward():
                if self.grad is not None and out.grad is not None:
                    self.grad += Tensor._unbroadcast(out.grad, self.shape)
                if isinstance(other, Tensor) and other.grad is not None and out.grad is not None:
                    other.grad += Tensor._unbroadcast(out.grad, other.shape)
            out._backward = _backward
        return out

    def __sub__(self, other: Union['Tensor',int,float]) -> 'Tensor':
        assert isinstance(other, (Tensor, int, float)), "Subtraction only supported with Tensor or scalar"
        if not isinstance(other, Tensor):
            other = Tensor(np.array(other, dtype=self.dtype), (), 'scalar', self.device, self.dtype)
        
        out_data = self.data - other.data # this is a numpy array or cupy array 
        out = Tensor(out_data, (self, other), '-', self.device, self.dtype)

        if out.require_grad:
            def _backward():
                if self.grad is not None and out.grad is not None:
                    self.grad += Tensor._unbroadcast(out.grad, self.shape)
                if other.grad is not None and out.grad is not None:
                    other.grad -= Tensor._unbroadcast(out.grad, other.shape)
            out._backward = _backward

        return out

    def __mul__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = other
        out = Tensor(self.data * other_data, (self, other) if isinstance(other, Tensor) else (self,), '*', self.device, self.dtype, require_grad=self.require_grad or (other.require_grad if isinstance(other, Tensor) else False))
        if out.require_grad:
            def _backward():
                if self.grad is not None and out.grad is not None:
                    self.grad += Tensor._unbroadcast(other_data * out.grad, self.shape)
                if isinstance(other, Tensor) and other.grad is not None and out.grad is not None:
                    other.grad += Tensor._unbroadcast(self.data * out.grad, other.shape)
            out._backward = _backward
        return out

    def __pow__(self, power: Union[int, float]) -> 'Tensor':
        assert isinstance(power, (int, float)), "Power must be an integer or float"
        out_data = self.data ** power # this is is a numpy or cupy array 
        out = Tensor(out_data, (self,), '**', self.device, self.dtype, require_grad=self.require_grad)
        if out.require_grad:
            def _backward():
                if self.grad is not None and out.grad is not None:
                    self.grad += Tensor._unbroadcast(power * (self.data ** (power - 1)) * out.grad, self.shape)
            out._backward = _backward
        return out

    def __truediv__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = other
        out = Tensor(self.data / other_data, (self, other) if isinstance(other, Tensor) else (self,), '/', self.device, self.dtype, require_grad=self.require_grad or (other.require_grad if isinstance(other, Tensor) else False))
        if out.require_grad:
            def _backward():
                if self.grad is not None and out.grad is not None:
                    self.grad += Tensor._unbroadcast((1 / other_data) * out.grad, self.shape)
                if isinstance(other, Tensor) and other.grad is not None and out.grad is not None:
                    other.grad -= Tensor._unbroadcast((self.data / (other_data ** 2)) * out.grad, other.shape)
            out._backward = _backward
        return out

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            raise TypeError('Matmul only supported with another Tensor')
        out = Tensor(self.data @ other_data, (self, other), '@', self.device, self.dtype, require_grad=self.require_grad or other.require_grad)
        if out.require_grad:
            def _backward():
                grad = out.grad
                # For x @ weight: dL/dx = grad @ weight.T, dL/dweight = x.T @ grad
                if self.grad is not None and grad is not None:
                    self.grad += grad @ other_data.T
                if other.grad is not None and grad is not None:
                    other.grad += self.data.T @ grad
            out._backward = _backward
        return out

    def __getitem__(self, idx:
        Union[int, slice, Tuple[Union[int, slice, Tuple[int, ...]], ...]]
    ) -> 'Tensor':
        out_data = self.data[idx]
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'slice', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if self.grad is not None:
                    grad_full = np.zeros_like(self.data) if self.device=='cpu' else cp.zeros_like(self.data)
                    grad_full[idx] = out.grad
                    self.grad += grad_full
            out._backward = _backward
        return out
    
    def reshape(self, *shape: int) -> 'Tensor':
        out_data = self.data.reshape(shape)
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'reshape', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if self.grad is not None and out.grad is not None:
                    self.grad += out.grad.reshape(self.shape)
            out._backward = _backward
        return out
    

    def item(self) -> float:
        if self.data.size != 1:
            raise ValueError("Tensor must have exactly one element to convert to scalar.")
        return float(self.data.item())

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"
    def __neg__(self): return self * -1.0
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other

