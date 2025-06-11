import numpy as np
import cupy as cp
from typing import Union, List, Literal, Tuple, Optional
from .utils import is_cuda_available

DEVICE  = 'cuda'  if is_cuda_available() else 'cpu' # i want to use this as the default device
class Tensor:
    def __init__(
        self,
        data: Union[List[float], List[int], np.ndarray, cp.ndarray],
        _children: Tuple['Tensor', ...] = (),
        _op: str = '',
        device: Literal['cpu', 'cuda'] = DEVICE,
        dtype: Literal['float32', 'float64', 'int32', 'int64', 'float16', 'int16', 'uint8', 'int8'] = 'float32'
    ) -> None:
        # Metadata
        self.device = device
        self.dtype = dtype
        self._children = set(_children)
        self._op = _op

        # Tensor data initialization
        if self.device == 'cpu':
            assert isinstance(data, (list, np.ndarray)), \
                "Data must be a list or NumPy array for CPU tensors."
            if isinstance(data, list):
                data = np.array(data)
            self.data = data.astype(self.dtype)
            self.grad = np.zeros_like(self.data)
        else:
            assert isinstance(data, (list, np.ndarray, cp.ndarray)), \
                "Data must be a list, NumPy array, or CuPy array for CUDA tensors."
            if isinstance(data, list):
                data = cp.array(data)
            elif isinstance(data, np.ndarray):
                data = cp.asarray(data)
            self.data = data.astype(self.dtype)
            self.grad = cp.zeros_like(self.data)

        self.shape = self.data.shape
        self._backward = lambda: None

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

    def zero_grad(self) -> None:
        if self.device == 'cpu':
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = cp.zeros_like(self.data)

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
            return self.data
        else:
            return cp.asnumpy(self.data)

    def sum(self, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> 'Tensor':
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), 'sum', self.device, self.dtype)
        n = int(np.prod(self.shape)) if axis is None else int(np.prod([self.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))]))

        def _backward():
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
            self.grad += grad
        out._backward = _backward
        return out

    def mean(self, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> 'Tensor':
        out = self.sum(axis, keepdims)
        # scale sum gradient by 1/n
        n = int(np.prod(self.shape)) if axis is None else int(np.prod([self.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))]))
        def _backward():
            self.grad += out.grad * (1.0 / n)
        out._backward = _backward
        out._op = 'mean'
        return out

    def var(self, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> 'Tensor':
        out_data = self.data.var(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), 'var', self.device, self.dtype)

        def _backward():
            mu = self.data.mean(axis=axis, keepdims=True)
            n = int(np.prod(self.shape)) if axis is None else int(np.prod([self.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))]))
            diff = self.data - mu
            grad = out.grad * (2.0 / n) * diff
            if not keepdims:
                grad = (np.broadcast_to if self.device=='cpu' else cp.broadcast_to)(grad, self.shape)
            self.grad += grad
        out._backward = _backward
        return out

    def std(self, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> 'Tensor':
        out_data = self.data.std(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), 'std', self.device, self.dtype)

        def _backward():
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

    def __add__(self, other: 'Tensor') -> 'Tensor':
        assert self.device == other.device, "Devices must match"
        out = Tensor(self.data + other.data, (self, other), '+', self.device, self.dtype)
        def _backward():
            self.grad += Tensor._unbroadcast(out.grad, self.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out

    def __mul__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(other, (), 'const', self.device, self.dtype)
        assert self.device == other.device, "Devices must match"
        out = Tensor(self.data * other.data, (self, other), '*', self.device, self.dtype)
        def _backward():
            self.grad += Tensor._unbroadcast(other.data * out.grad, self.shape)
            other.grad += Tensor._unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out

    def __pow__(self, power: Union[int, float]) -> 'Tensor':
        out_data = (np.power if self.device=='cpu' else cp.power)(self.data, power)
        out = Tensor(out_data, (self,), 'pow', self.device, self.dtype)
        def _backward():
            grad = out.grad * power * (np.power if self.device=='cpu' else cp.power)(self.data, power - 1)
            self.grad += grad
        out._backward = _backward
        return out

    def __truediv__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(other, (), 'const', self.device, self.dtype)
        return self * other**-1

    def reshape(self, *shape: int) -> 'Tensor':
        out_data = self.data.reshape(shape)
        out = Tensor(out_data, (self,), 'reshape', self.device, self.dtype)
        def _backward():
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out

    def permute(self, *dims: int) -> 'Tensor':
        out_data = self.data.transpose(dims)
        out = Tensor(out_data, (self,), 'permute', self.device, self.dtype)
        def _backward():
            inv = np.argsort(dims) if self.device=='cpu' else cp.argsort(dims)
            self.grad += out.grad.transpose(tuple(inv))
        out._backward = _backward
        return out

    def mask_fill(self, mask: 'Tensor', value: Union[int, float]=0) -> 'Tensor':
        assert self.device == mask.device, "Devices must match"
        if self.device=='cpu':
            out_data = np.where(mask.data, value, self.data)
        else:
            out_data = cp.where(mask.data, value, self.data)
        out = Tensor(out_data, (self, mask), 'mask_fill', self.device, self.dtype)
        def _backward():
            self.grad += out.grad * mask.data
            mask.grad += out.grad * ((self.data == value).astype(self.dtype))
        out._backward = _backward
        return out

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        assert self.device == other.device, "Devices must match"
        out_data = self.data @ other.data
        out = Tensor(out_data, (self, other), '@', self.device, self.dtype)
        def _backward():
            grad = out.grad
            if self.device=='cpu':
                self.grad += np.swapaxes(other.data, -1, -2) @ grad
                other.grad += np.swapaxes(self.data, -1, -2) @ grad
            else:
                self.grad += cp.swapaxes(other.data, -1, -2) @ grad
                other.grad += cp.swapaxes(self.data, -1, -2) @ grad
        out._backward = _backward
        return out

    def __getitem__(self, idx:
        Union[int, slice, Tuple[Union[int, slice, Tuple[int, ...]], ...]]
    ) -> 'Tensor':
        out_data = self.data[idx]
        out = Tensor(out_data, (self,), 'slice', self.device, self.dtype)
        def _backward():
            grad_full = np.zeros_like(self.data) if self.device=='cpu' else cp.zeros_like(self.data)
            grad_full[idx] = out.grad
            self.grad += grad_full
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
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other

