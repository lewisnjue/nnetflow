import numpy as np
from typing import Union, List, Literal, Tuple, Optional
import importlib

# Try to import CuPy for CUDA support
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

# Try to import the C++/CUDA backend
try:
    tensor_cuda = importlib.import_module('tensor_cuda')
    _HAS_CUDA_BACKEND = True
except ImportError:
    tensor_cuda = None
    _HAS_CUDA_BACKEND = False

def is_cuda_available():
    # First check C++/CUDA backend
    if _HAS_CUDA_BACKEND:
        try:
            return tensor_cuda.Device.detect_best().type == tensor_cuda.DeviceType.CUDA
        except Exception:
            pass
    # Fallback to CuPy detection
    if _HAS_CUPY:
        try:
            cp.cuda.runtime.getDeviceCount()
            return True
        except Exception:
            pass
    return False

DEVICE = 'cuda' if is_cuda_available() else 'cpu'

# Helper functions for array operations
def _get_array_module(device: str):
    """Get the appropriate array module for the device."""
    if device == 'cpu':
        return np
    else:
        if not _HAS_CUPY:
            raise RuntimeError("CuPy not available for CUDA operations")
        return cp

def _zeros_like(arr, device: str = 'cpu'):
    """Create zeros_like array on appropriate device."""
    xp = _get_array_module(device)
    return xp.zeros_like(arr)

def _ones_like(arr, device: str = 'cpu'):
    """Create ones_like array on appropriate device."""
    xp = _get_array_module(device)
    return xp.ones_like(arr)

def _randn(shape: Tuple[int, ...], device: str = 'cpu'):
    """Create random normal array on appropriate device."""
    if device == 'cpu':
        return np.random.randn(*shape)
    else:
        if not _HAS_CUPY:
            raise RuntimeError("CuPy not available for CUDA operations")
        return cp.random.randn(*shape)

class Tensor:
    @staticmethod
    def _noop_backward():
        pass
    def __init__(
        self,
        data: Union[List[float], List[int], np.ndarray, int, float],
        _children: Tuple['Tensor', ...] = (),
        _op: str = '',
        device: str = DEVICE,
        dtype: str = 'float32',
        require_grad: bool = True
    ) -> None:
        self.require_grad = require_grad
        self.device = device
        self.dtype = dtype
        self._op = _op
        self._children = _children if require_grad else ()
        # Use C++/CUDA backend if available
        if _HAS_CUDA_BACKEND:
            dev = tensor_cuda.Device(tensor_cuda.DeviceType.CUDA if device == 'cuda' else tensor_cuda.DeviceType.CPU, 0)
            if isinstance(data, (int, float)):
                arr = np.array([data], dtype=dtype)
            elif isinstance(data, np.ndarray):
                arr = data.astype(dtype)
            elif isinstance(data, list):
                arr = np.array(data, dtype=dtype)
            else:
                raise TypeError("Unsupported data type for Tensor")
            shape = list(arr.shape)
            self._tensor = tensor_cuda.CudaTensor(shape, require_grad, dev)
            self._tensor.from_host(arr.flatten().tolist())
            self.shape = tuple(shape)
            self.grad = None # Expose grad if needed
            self.data = arr # For compatibility
        else:
            # Fallback to numpy
            if isinstance(data, (int, float)):
                arr = np.array([data], dtype=dtype)
            elif isinstance(data, np.ndarray):
                arr = data.astype(dtype)
            elif isinstance(data, list):
                arr = np.array(data, dtype=dtype)
            elif hasattr(data, 'shape') and hasattr(data, 'dtype'):  # numpy scalar or array-like
                arr = np.array(data, dtype=dtype)
            else:
                raise TypeError(f"Unsupported data type for Tensor: {type(data)}")
            self.data = arr
            self.shape = arr.shape
            self.grad = np.zeros_like(arr) if require_grad else None
        self._backward = Tensor._noop_backward

    @staticmethod
    def _unbroadcast(
        grad: Union[np.ndarray, 'cp.ndarray'],
        shape: Tuple[int, ...]
    ) -> Union[np.ndarray, 'cp.ndarray']:
        # Sum out broadcasted dims
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0) # after here the shape have the shame ndim 
        for i, (g, s) in enumerate(zip(grad.shape, shape)): 
            if s == 1 and g != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    @staticmethod
    def zeros(shape: Tuple[int, ...], device: str = DEVICE, dtype: str = 'float32', require_grad: bool = True) -> 'Tensor':
        if device == 'cpu':
            data = np.zeros(shape, dtype=dtype)
        else:
            if not _HAS_CUPY:
                raise RuntimeError("CuPy not available for CUDA operations")
            data = cp.zeros(shape, dtype=dtype)
        return Tensor(data, (), 'zeros', device, dtype, require_grad=require_grad)

    @staticmethod
    def ones(shape: Tuple[int, ...], device: str = DEVICE, dtype: str = 'float32', require_grad: bool = True) -> 'Tensor':
        if device == 'cpu':
            data = np.ones(shape, dtype=dtype)
        else:
            if not _HAS_CUPY:
                raise RuntimeError("CuPy not available for CUDA operations")
            data = cp.ones(shape, dtype=dtype)
        return Tensor(data, (), 'ones', device, dtype, require_grad=require_grad)
    
    @staticmethod
    def xavier_uniform(shape: Tuple[int, ...], device: str = DEVICE, dtype: str = 'float32', require_grad: bool = True) -> 'Tensor':
        """Initialize tensor with Xavier/Glorot uniform distribution."""
        fan_in = shape[0] if len(shape) > 0 else 1
        fan_out = shape[1] if len(shape) > 1 else 1
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
        if device == 'cpu':
            data = np.random.uniform(-limit, limit, shape).astype(dtype)
        else:
            if not _HAS_CUPY:
                raise RuntimeError("CuPy not available for CUDA operations")
            data = cp.random.uniform(-limit, limit, shape).astype(dtype)
        return Tensor(data, (), 'xavier_uniform', device, dtype, require_grad=require_grad)
    
    @staticmethod
    def xavier_normal(shape: Tuple[int, ...], device: str = DEVICE, dtype: str = 'float32', require_grad: bool = True) -> 'Tensor':
        """Initialize tensor with Xavier/Glorot normal distribution."""
        fan_in = shape[0] if len(shape) > 0 else 1
        fan_out = shape[1] if len(shape) > 1 else 1
        std = np.sqrt(2.0 / (fan_in + fan_out))
        
        if device == 'cpu':
            data = np.random.normal(0.0, std, shape).astype(dtype)
        else:
            if not _HAS_CUPY:
                raise RuntimeError("CuPy not available for CUDA operations")
            data = cp.random.normal(0.0, std, shape).astype(dtype)
        return Tensor(data, (), 'xavier_normal', device, dtype, require_grad=require_grad)
    
    @staticmethod
    def he_uniform(shape: Tuple[int, ...], device: str = DEVICE, dtype: str = 'float32', require_grad: bool = True) -> 'Tensor':
        """Initialize tensor with He uniform distribution (for ReLU)."""
        fan_in = shape[0] if len(shape) > 0 else 1
        limit = np.sqrt(6.0 / fan_in)
        
        if device == 'cpu':
            data = np.random.uniform(-limit, limit, shape).astype(dtype)
        else:
            if not _HAS_CUPY:
                raise RuntimeError("CuPy not available for CUDA operations")
            data = cp.random.uniform(-limit, limit, shape).astype(dtype)
        return Tensor(data, (), 'he_uniform', device, dtype, require_grad=require_grad)
    
    @staticmethod
    def he_normal(shape: Tuple[int, ...], device: str = DEVICE, dtype: str = 'float32', require_grad: bool = True) -> 'Tensor':
        """Initialize tensor with He normal distribution (for ReLU)."""
        fan_in = shape[0] if len(shape) > 0 else 1
        std = np.sqrt(2.0 / fan_in)
        
        if device == 'cpu':
            data = np.random.normal(0.0, std, shape).astype(dtype)
        else:
            if not _HAS_CUPY:
                raise RuntimeError("CuPy not available for CUDA operations")
            data = cp.random.normal(0.0, std, shape).astype(dtype)
        return Tensor(data, (), 'he_normal', device, dtype, require_grad=require_grad)
    
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
                if not _HAS_CUPY:
                    raise RuntimeError("CuPy not available for CUDA operations")
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

        if self.device == 'cpu':
            self.grad = np.ones_like(self.data)
        else:
            if not _HAS_CUPY:
                raise RuntimeError("CuPy not available for CUDA operations")
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
                if not _HAS_CUPY:
                    raise RuntimeError("CuPy not available for CUDA operations")
                v.grad = cp.nan_to_num(v.grad, nan=0.0, posinf=1e5, neginf=-1e5)
                if grad_clip is not None:
                    cp.clip(v.grad, -grad_clip, grad_clip, out=v.grad)

    def to(self, device: Literal['cpu', 'cuda']) -> 'Tensor':
        if _HAS_CUDA_BACKEND:
            dev = tensor_cuda.Device(tensor_cuda.DeviceType.CUDA if device == 'cuda' else tensor_cuda.DeviceType.CPU, 0)
            t2 = self._tensor.to(dev.type)
            arr = np.array(t2.to_host(), dtype=self.dtype).reshape(self.shape)
            return Tensor(arr, device=device, dtype=self.dtype, require_grad=self.require_grad)
        else:
            if device == self.device:
                return self
            arr = self.data.copy()
            return Tensor(arr, device=device, dtype=self.dtype, require_grad=self.require_grad)
    
    def cpu(self) -> 'Tensor':
        return self.to('cpu')

    def cuda(self) -> 'Tensor':
        return self.to('cuda')
    
    def numpy(self) -> np.ndarray:
        if _HAS_CUDA_BACKEND:
            return np.array(self._tensor.to_host(), dtype=self.dtype).reshape(self.shape)
        else:
            return np.asarray(self.data)

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

    def sigmoid(self) -> 'Tensor':
        if self.device == 'cpu':
            out_data = 1.0 / (1.0 + np.exp(-self.data))
        else:
            out_data = 1.0 / (1.0 + cp.exp(-self.data))
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'sigmoid', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    grad = out.grad * out.data * (1.0 - out.data)
                    self.grad += grad
            out._backward = _backward
        return out

    def tanh(self) -> 'Tensor':
        if self.device == 'cpu':
            out_data = np.tanh(self.data)
        else:
            out_data = cp.tanh(self.data)
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'tanh', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    grad = out.grad * (1.0 - out.data**2)
                    self.grad += grad
            out._backward = _backward
        return out

    def leaky_relu(self, alpha: float = 0.01) -> 'Tensor':
        if self.device == 'cpu':
            out_data = np.where(self.data > 0, self.data, alpha * self.data)
        else:
            out_data = cp.where(self.data > 0, self.data, alpha * self.data)
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'leaky_relu', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    if self.device == 'cpu':
                        slope = np.where(self.data > 0, 1.0, alpha).astype(self.data.dtype)
                    else:
                        slope = cp.where(self.data > 0, 1.0, alpha).astype(self.data.dtype)
                    self.grad += out.grad * slope
            out._backward = _backward
        return out

    def gelu(self, approximate: bool = True) -> 'Tensor':
        # Gaussian Error Linear Unit
        if approximate:
            # tanh approximation (Hendrycks & Gimpel)
            if self.device == 'cpu':
                c = np.sqrt(2 / np.pi)
                out_data = 0.5 * self.data * (1.0 + np.tanh(c * (self.data + 0.044715 * (self.data ** 3))))
            else:
                c = np.sqrt(2 / np.pi)
                out_data = 0.5 * self.data * (1.0 + cp.tanh(c * (self.data + 0.044715 * (self.data ** 3))))
        else:
            # exact via erf
            if self.device == 'cpu':
                out_data = 0.5 * self.data * (1.0 + (2/np.sqrt(np.pi)) * np.vectorize(lambda v: np.math.erf(v/np.sqrt(2)))(self.data))
            else:
                out_data = 0.5 * self.data * (1.0 + (2/np.sqrt(np.pi)) * cp.erf(self.data / np.sqrt(2)))
        require_grad = self.require_grad
        children = (self,) if require_grad else ()
        out = Tensor(out_data, children, 'gelu', self.device, self.dtype, require_grad=require_grad)
        if require_grad:
            def _backward():
                if out.grad is not None and self.grad is not None:
                    if self.device == 'cpu':
                        c = np.sqrt(2 / np.pi)
                        tanh_arg = c * (self.data + 0.044715 * (self.data ** 3))
                        sech2 = 1.0 / (np.cosh(tanh_arg) ** 2)
                        dgelu = 0.5 * (1.0 + np.tanh(tanh_arg)) + 0.5 * self.data * sech2 * c * (1 + 3 * 0.044715 * (self.data ** 2))
                    else:
                        c = np.sqrt(2 / np.pi)
                        tanh_arg = c * (self.data + 0.044715 * (self.data ** 3))
                        sech2 = 1.0 / (cp.cosh(tanh_arg) ** 2)
                        dgelu = 0.5 * (1.0 + cp.tanh(tanh_arg)) + 0.5 * self.data * sech2 * c * (1 + 3 * 0.044715 * (self.data ** 2))
                    self.grad += out.grad * dgelu
            out._backward = _backward
        return out

    def swish(self) -> 'Tensor':
        sig = self.sigmoid()
        out = self * sig
        # backward is handled via autograd chain (mul + sigmoid)
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
        if _HAS_CUDA_BACKEND:
            out_tensor = self._tensor.matmul(other._tensor)
            arr = np.array(out_tensor.to_host(), dtype=self.dtype).reshape(out_tensor.shape)
            return Tensor(arr, device=self.device, dtype=self.dtype, require_grad=self.require_grad or other.require_grad)
        else:
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

