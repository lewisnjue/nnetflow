import numpy as np
import numpy.typing as npt
import warnings
from typing import Any, Union, Tuple, Optional, Set

import scipy.special as sp


class Tensor:
    """A simple autograd Tensor class supporting dynamic computation graphs
    and backpropagation.

    This is the core data structure of nnetflow.  Every arithmetic operation
    on tensors builds a directed acyclic graph (DAG) that records *how* the
    output was produced.  Calling :meth:`backward` on a scalar output walks
    this graph in reverse (topological) order and accumulates gradients via
    the chain rule — this is reverse-mode automatic differentiation.
    """

    def __init__(
        self,
        data: Union[np.ndarray, float, int, list, tuple],
        _children: Tuple['Tensor', ...] = (),
        _op: str = '',
        requires_grad: Optional[bool] = None,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> None:
        """Create a new Tensor.

        Args:
            data: Raw data — a NumPy array, Python scalar, list, or tuple.
            _children: Tensors that were used to produce this tensor (used
                internally by operations to build the computation graph).
            _op: A short label for the operation that created this tensor
                (e.g. ``'+'``, ``'@'``).  Used only for debugging/display.
            requires_grad: Whether to track gradients for this tensor.  If
                ``None``, gradient tracking is enabled automatically when
                any parent tensor requires gradients.
            dtype: Data type of the tensor (e.g. ``np.float32``).  If
                ``None``, the dtype is inferred from *data* (or defaults
                to ``np.float64`` for plain Python scalars/lists).
        """
        if dtype is not None:
            target_dtype = np.dtype(dtype)
        elif hasattr(data, 'dtype'):  
            target_dtype = data.dtype
        else:
            target_dtype = np.float64

        if hasattr(data, 'dtype'):  
            self.data = data.astype(target_dtype, copy=False)
        else:
            try:
                self.data = np.array(data, dtype=target_dtype)
            except Exception as e:
                raise TypeError(f"Could not convert data to Tensor. Error: {e}")

        self._op = _op

        self._prev: Set['Tensor'] = set(c for c in _children if isinstance(c, Tensor))

        if requires_grad is None:
            self.requires_grad = any(c.requires_grad for c in self._prev)
        else:
            self.requires_grad = bool(requires_grad)

        if self.requires_grad:
            grad_dtype = np.float64 if self.data.dtype not in [np.float64, np.float32] else self.data.dtype
            self.grad: Optional[Any] = np.zeros(self.data.shape, dtype=grad_dtype)
        else:
            self.grad: Optional[Any] = None

        self._backward = lambda: None

    def __getstate__(self):
        """Called when pickling — removes the backward closure which cannot be serialized."""
        state = self.__dict__.copy()
        if '_backward' in state:
            del state['_backward']
        return state
    
    @classmethod 
    def _check_dtype(cls,A:'Tensor',B:'Tensor') -> None: 
        """ Used to check if the dypes of two tensors are compatible for 
        arithmetic operations""" 
        if A.dtype != B.dtype: 
            raise ValueError(f"Tensor dtype mismatch: {A.dtype} != {B.dtype}")
        return True 

    def __setstate__(self, state):
        """Called when unpickling — restores a no-op backward closure."""
        self.__dict__.update(state)
        self._backward = lambda: None

    @property
    def dtype(self):
        """dtype property"""
        return self.data.dtype 
    def to(self, dtype: npt.DTypeLike) -> 'Tensor':
        """Cast the tensor to a new dtype, returning a **detached** copy.

        The returned tensor is independent of the original — it does not
        share the computation graph.  This is intentional: casting in the
        middle of a forward pass would break the backward chain.

        Args:
            dtype: Target data type (e.g. ``np.float32``, ``np.int32``).

        Returns:
            A new ``Tensor`` with the specified dtype.
        """
        new_data = self.data.astype(dtype)  # creates a copy
        return Tensor(new_data, requires_grad=self.requires_grad)

    def astype(self, dtype: npt.DTypeLike) -> 'Tensor':
        """Alias for :meth:`to`.  Returns a new tensor with the given dtype."""
        return self.to(dtype)

    @classmethod  
    def unbroadcast(cls, grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Sums a gradient to match the original shape before a broadcasting operation.
        Args:
            grad: The incoming gradient (with the broadcasted shape).
            shape: The target shape (the original tensor's shape).
        Returns:
            The unbroadcasted gradient.
        """
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)  
        for i, (grad_dim, shape_dim) in enumerate(zip(grad.shape, shape)):
            if grad_dim != shape_dim:
                if shape_dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
                else:
                    raise ValueError(f"Cannot unbroadcast shape {grad.shape} to {shape}")
        return grad

    def __repr__(self) -> str:
        """Return a human-readable string representation of the Tensor."""
        # Convert to numpy for display purposes
        data_np = np.asarray(self.data)
        data_str = np.array2string(data_np, max_line_width=70, precision=4, suppress_small=True)
        if '\n' in data_str:
            data_str = data_str.split('\n')[0] + '...]' # Show first line only if multi-line
        grad_info = ", grad_fn" if self._op else "" # Simplified grad_fn indicator
        return f"Tensor(data={data_str}, shape={self.shape}, requires_grad={self.requires_grad}{grad_info})"

    def zero_grad(self) -> None:
        """Resets the gradient of this tensor to zero."""
        if self.requires_grad:
            grad_dtype = np.float64 if self.data.dtype not in [np.float64, np.float32] else self.data.dtype
            self.grad = np.zeros_like(self.data, dtype=grad_dtype)

    
    @classmethod
    def zeros(cls, *shape: int, requires_grad: bool = False, dtype: Optional[npt.DTypeLike] = None) -> 'Tensor':
        """Create a tensor filled with zeros."""
        return cls(np.zeros(shape), requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def ones(cls, *shape: int, requires_grad: bool = False, dtype: Optional[npt.DTypeLike] = None) -> 'Tensor':
        """Create a tensor filled with ones."""
        return cls(np.ones(shape), requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def randn(cls, *shape: int, requires_grad: bool = False, dtype: Optional[npt.DTypeLike] = None) -> 'Tensor':
        """Create a tensor filled with random numbers from a standard normal distribution."""
        data = np.random.randn(*shape).astype(dtype=dtype if dtype else np.float32)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def zeros_like(cls, tensor: 'Tensor', requires_grad: Optional[bool] = None, dtype: Optional[npt.DTypeLike] = None) -> 'Tensor':
        """Create a tensor of zeros with the same shape as *tensor*."""
        if requires_grad is None:
            requires_grad = tensor.requires_grad
        return cls(np.zeros_like(tensor.data), requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def ones_like(cls, tensor: 'Tensor', requires_grad: Optional[bool] = None, dtype: Optional[npt.DTypeLike] = None) -> 'Tensor':
        """Create a tensor of ones with the same shape as *tensor*."""
        if requires_grad is None:
            requires_grad = tensor.requires_grad
        return cls(np.ones_like(tensor.data), requires_grad=requires_grad, dtype=dtype)

    def __add__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        """Element-wise addition (``self + other``).

        Supports adding a Tensor to another Tensor, a scalar, or a NumPy array.
        Broadcasting is handled automatically during the backward pass.
        """
        if isinstance(other, Tensor): 
            Tensor._check_dtype(self, other)
        other_val = other.data if isinstance(other, Tensor) else other
        children = (self, other) if isinstance(other, Tensor) else (self,)
        # Preserve dtype from self (or other if both are tensors)
        dtype = self.data.dtype if not isinstance(other, Tensor) else self.data.dtype
        out = Tensor(self.data + other_val, children, '+', dtype=dtype) 
        
        def _backward():
            if self.requires_grad:
                self.grad += Tensor.unbroadcast(out.grad, self.data.shape)
            if isinstance(other, Tensor) and other.requires_grad:
                other.grad += Tensor.unbroadcast(out.grad, other.data.shape)
        
        if out.requires_grad:
            out._backward = _backward
        return out
    
    def add(self, x: 'Tensor') -> 'Tensor':
        """Functional form of addition.  Equivalent to ``self + x``."""
        return self.__add__(x)


    def __mul__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        """Element-wise multiplication (``self * other``).

        The local gradient rule is:
            d(a * b)/da = b  and  d(a * b)/db = a
        so in the backward pass the incoming gradient is scaled by the *other*
        operand's data.
        """
        if isinstance(other, Tensor): 
            Tensor._check_dtype(self, other)
        other_val = other.data if isinstance(other, Tensor) else other
        children = (self, other) if isinstance(other, Tensor) else (self,)
        
        out = Tensor(self.data * other_val, children, '*')
        
        def _backward():
            if self.requires_grad:
                self.grad += Tensor.unbroadcast((other_val * out.grad), self.data.shape)
            if isinstance(other, Tensor) and other.requires_grad:
                other.grad += Tensor.unbroadcast((self.data * out.grad), other.data.shape)
                
        if out.requires_grad:
            out._backward = _backward
        return out
    
    def matmul(self, x: 'Tensor') -> 'Tensor':
        """Functional form of matrix multiplication.  Equivalent to ``self @ x``."""
        return self.__matmul__(x)

    def __pow__(self, other: Union[float, int]) -> 'Tensor':
        """Element-wise power (``self ** other``).  Only scalar exponents are supported."""
        assert isinstance(other, (float, int)), "Only support float and int power for Tensor"
        out = Tensor(self.data ** other, (self,), f'**{other}') 
        
        def _backward():
            if self.requires_grad:
                self.grad += (other * (self.data ** (other - 1))) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def __truediv__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        """Element-wise division (``self / other``).

        Local gradient rules:
            d(a/b)/da =  1 / b
            d(a/b)/db = -a / b**2
        """
        if isinstance(other, Tensor): 
            Tensor._check_dtype(self, other)
        other_val = other.data if isinstance(other, Tensor) else other
        children = (self, other) if isinstance(other, Tensor) else (self,)

        out = Tensor(self.data / other_val, children, '/')
        
        def _backward():
            if self.requires_grad:
                self.grad += Tensor.unbroadcast((1 / other_val) * out.grad, self.data.shape)
            if isinstance(other, Tensor) and other.requires_grad:
                other.grad += Tensor.unbroadcast((-self.data / (other_val ** 2)) * out.grad, other.data.shape)
                
        if out.requires_grad:
            out._backward = _backward
        return out

    def __neg__(self) -> 'Tensor':
        return self * -1

    def __sub__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        return self + (other * -1)

    def __radd__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        return self + other

    def __rmul__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        return self * other

    def __rsub__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        return (self * -1) + other

    def __rtruediv__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        return other * (self ** -1)

    @classmethod 
    def can_matmul(cls,shape_a, shape_b):
        """ used to check if two shapes can be matrix multiplied together"""
        ndim_a, ndim_b = len(shape_a), len(shape_b)
        if ndim_a == 0 or ndim_b == 0:
            return False 
        dim_inner_a = shape_a[-1]
        dim_inner_b = shape_b[-2] if ndim_b > 1 else shape_b[0]
        
        if dim_inner_a != dim_inner_b:
            return False
        batch_shape_a = shape_a[:-1] if ndim_a == 1 else shape_a[:-2]
        batch_shape_b = shape_b[:-1] if ndim_b == 1 else shape_b[:-2]
        try:
            # Use numpy for shape checking (doesn't need device)
            np.broadcast_shapes(batch_shape_a, batch_shape_b)
            return True
        except ValueError:
            return False



    def __matmul__(self, other: 'Tensor') -> 'Tensor':
            assert isinstance(other, Tensor), "Only support Tensor type for matmul operation"
            Tensor._check_dtype(self, other)
            can_matmul = Tensor.can_matmul(self.data.shape, other.data.shape) 
            if not can_matmul:
                raise ValueError(f"Shapes {self.data.shape} and {other.data.shape} not aligned for matmul") 
            # Preserve dtype from inputs (use self's dtype as primary)
            dtype = self.data.dtype
            out = Tensor(self.data @ other.data, (self, other), '@', dtype=dtype)

            def _backward():
               if self.requires_grad:
                    # CASE A: 'other' is a Matrix (2D+)
                    if other.data.ndim > 1:
                        other_transposed = np.swapaxes(other.data, -1, -2)
                        self_grad_contrib = out.grad @ other_transposed
                    
                    # CASE B: 'other' is a Vector (1D)
                    else:
                        # If result is scalar (Vector @ Vector), simple scaling
                        if out.grad.ndim == 0:
                            self_grad_contrib = out.grad * other.data
                        # If result is vector (Matrix @ Vector), outer product
                        else:
                            self_grad_contrib = np.outer(out.grad, other.data)

                    self.grad += Tensor.unbroadcast(self_grad_contrib, self.data.shape)

               if other.requires_grad:
                    # CASE A: 'self' is a Matrix (2D+)
                    if self.data.ndim > 1:
                        self_transposed = np.swapaxes(self.data, -1, -2)
                        other_grad_contrib = self_transposed @ out.grad
                    
                    # CASE B: 'self' is a Vector (1D)
                    else:
                        # If result is scalar (Vector @ Vector), simple scaling
                        if out.grad.ndim == 0:
                            other_grad_contrib = out.grad * self.data
                        # If result is vector (Vector @ Matrix), outer product
                        else:
                            # Note: This case (Vector @ Matrix) usually results in a vector,
                            # requiring the outer product of self and grad.
                            other_grad_contrib = np.outer(self.data, out.grad)

                    other.grad += Tensor.unbroadcast(other_grad_contrib, other.data.shape)

            if out.requires_grad:
                out._backward = _backward
            return out 
 
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), 'sum',dtype=self.data.dtype)
        
        def _backward():
            if self.requires_grad:
                if axis is None: # in this case only one number is in the out `Tensor`
                    grad_expanded = np.ones_like(self.data) * out.grad
                else:
                    if keepdims:
                        grad_to_expand = out.grad 
                    else:
                        grad_to_expand = np.expand_dims(out.grad, axis=axis)
                    grad_expanded = np.ones_like(self.data) * grad_to_expand
                
                self.grad += grad_expanded
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        if axis is None:
            n = self.data.size 
        elif isinstance(axis, int): 
            n = self.data.shape[axis]
        else: 
            # Use numpy for shape operations (doesn't need device)
            n = np.prod([self.data.shape[i] for i in axis])         

        sum_out = self.sum(axis=axis, keepdims=keepdims)
        out = sum_out * (1.0 / n) 
        out._op = 'mean'
        return out
        

    def exp(self) -> 'Tensor':
        out_data = np.exp(self.data)
        out = Tensor(out_data, (self,), 'exp') 
        
        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out
    def log(self) -> 'Tensor':
        """Natural logarithm (ln)."""
        # Convert to numpy for checking (scalar check)
        if not np.all(np.asarray(self.data) > 0):
            warnings.warn("Log applied to non-positive elements",RuntimeWarning,stacklevel=2)
        
        out = Tensor(np.log(self.data), (self,), 'ln') 
        
        def _backward():
            if self.requires_grad:
                # Add epsilon for numerical stability in gradient
                self.grad += (1 / (self.data + 1e-8)) * out.grad 
        
        if out.requires_grad:
            out._backward = _backward
        return out
    
    def sqrt(self) -> 'Tensor':
        """Square root"""
        out = Tensor(np.sqrt(self.data), (self,), 'sqrt')
        
        def _backward():
            if self.requires_grad:
                # d/dx(sqrt(x)) = 1 / (2 * sqrt(x))
                self.grad += (0.5 / (np.sqrt(self.data) + 1e-8)) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def clip(self, min_val: float, max_val: float) -> 'Tensor':
        """
        Clips the tensor values to be within [min_val, max_val].
        """
        out = Tensor(np.clip(self.data, min_val, max_val), (self,), 'clip')

        def _backward():
            if self.requires_grad:
                mask = (self.data >= min_val) & (self.data <= max_val)
                self.grad += out.grad * mask

        if out.requires_grad:
            out._backward = _backward
            
        return out

    def log10(self) -> 'Tensor':
        """Base-10 logarithm."""
        if not np.all(np.asarray(self.data) > 0):
            warnings.warn("Log10 applied to non-positive elements", RuntimeWarning, stacklevel=2)
        
        out = Tensor(np.log10(self.data), (self,), 'log10') 
        
        def _backward():
            if self.requires_grad:
                # Add epsilon for numerical stability in gradient
                # np.log(10) is a constant, so it's fine to use numpy
                self.grad += (1 / ((self.data + 1e-8) * np.log(10))) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out


    def relu(self) -> 'Tensor':
        """Rectified Linear Unit: ``max(0, x)``.

        Reference: https://arxiv.org/abs/1803.08375
        """
        out = Tensor(np.maximum(self.data, 0), (self,), 'relu')
        
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def leaky_relu(self, alpha: float = 0.01) -> 'Tensor':
        """Leaky ReLU: ``x`` if ``x > 0``, else ``alpha * x``.

        Reference: https://arxiv.org/abs/1505.00853
        """
        out = Tensor(np.where(self.data > 0, self.data, alpha * self.data), (self,), 'leaky_relu')
        
        def _backward():
            if self.requires_grad:
                self.grad += np.where(self.data > 0, 1, alpha) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def elu(self, alpha: float = 1.0) -> 'Tensor':
        """Enponential Linear Unit: ``x`` if ``x > 0``, else ``alpha * (exp(x) - 1)``.

        Reference: https://arxiv.org/abs/1511.07289
        """
        out = Tensor(np.where(self.data > 0, self.data, alpha * (np.exp(self.data) - 1)), (self,), 'elu')
        
        def _backward():
            if self.requires_grad:
                # d/dx(alpha * (exp(x) - 1)) = alpha * exp(x)
                self.grad += np.where(self.data > 0, 1, alpha * np.exp(self.data)) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def selu(self, alpha: float = 1.67326, scale: float = 1.0507) -> 'Tensor':
        """Scaled Exponential Linear Unit.

        Reference: https://arxiv.org/abs/1706.02515
        """
        out = Tensor(scale * np.where(self.data > 0, self.data, alpha * (np.exp(self.data) - 1)), (self,), 'selu')
        
        def _backward():
            if self.requires_grad:
                self.grad += scale * np.where(self.data > 0, 1, alpha * np.exp(self.data)) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def gelu(self) -> 'Tensor':
        """Gaussian Error Linear Unit.

        Reference: https://arxiv.org/abs/1606.08415
        """
        # scipy.special.erf works with numpy arrays, so convert if needed
        data_np = np.asarray(self.data)
        erf_result = sp.erf(data_np / np.sqrt(2))
        out_data = 0.5 * self.data * (1 + erf_result)
        out = Tensor(out_data, (self,), 'gelu')
        
        def _backward():
            if self.requires_grad:
                # np.sqrt and np.pi are constants
                sqrt_2pi = np.sqrt(2 * np.pi)
                data_np = np.asarray(self.data)
                cdf_np = 0.5 * (1 + sp.erf(data_np / np.sqrt(2)))
                pdf_np = (1 / sqrt_2pi) * np.exp(-0.5 * data_np ** 2)
                cdf = cdf_np
                pdf = pdf_np
                self.grad += (cdf + self.data * pdf) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def sigmoid(self) -> 'Tensor':
        """Sigmoid activation: ``1 / (1 + exp(-x))``.

        Uses a numerically stable formulation that avoids overflow for
        large negative values.
        """
        # Numerically stable sigmoid
        sig = np.where(self.data >= 0, 
                       1 / (1 + np.exp(-self.data)), 
                       np.exp(self.data) / (1 + np.exp(self.data)))
        out = Tensor(sig, (self,), 'sigmoid')
        
        def _backward():
            if self.requires_grad:
                self.grad += sig * (1 - sig) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def swish(self) -> 'Tensor':
        """Swish activation: ``x * sigmoid(x)``.

        Reference: https://arxiv.org/abs/1710.05941
        """
        # swish(x) = x * sigmoid(x)
        # We can re-use our stable sigmoid
        sig = self.sigmoid() 
        out = self * sig # This builds the graph!
        out._op = 'swish'
        return out

    def tanh(self) -> 'Tensor':
        """Hyperbolic tangent activation."""
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        
        def _backward():
            if self.requires_grad:
                self.grad += (1 - t ** 2) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> 'Tensor':
        """Softmax: ``exp(x_i) / sum(exp(x_j))`` along *axis*.

        Uses the log-sum-exp trick for numerical stability.
        """
        # Log-sum-exp trick for numerical stability
        max_val = self.data.max(axis=axis, keepdims=True)
        e_x = np.exp(self.data - max_val) # Subtract max for stability
        sum_e_x = e_x.sum(axis=axis, keepdims=True)
        sm = e_x / (sum_e_x + 1e-8) # Add epsilon for safety
        
        out = Tensor(sm, (self,), 'softmax')
        
        def _backward():
            if self.requires_grad:
                # VJP (Vector-Jacobian Product) for softmax:
                # Let y = out.data, g = out.grad
                # dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
                y = out.data
                g = out.grad
                
                sum_gy = (g * y).sum(axis=axis, keepdims=True)
                grad_contrib = y * (g - sum_gy)
                
                self.grad += grad_contrib
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> 'Tensor':
        """Log-softmax: numerically stable ``log(softmax(x))`` along *axis*."""
        # Stable LogSoftmax
        max_val = self.data.max(axis=axis, keepdims=True)
        x_minus_max = self.data - max_val
        log_sum_exp = np.log(np.exp(x_minus_max).sum(axis=axis, keepdims=True) + 1e-8)
        log_sm = x_minus_max - log_sum_exp
        
        out = Tensor(log_sm, (self,), 'log_softmax')
        
        def _backward():
            if self.requires_grad:
                # VJP for LogSoftmax:
                # dL/dx_i = dL/dy_i - exp(y_i) * sum_j(dL/dy_j)
                g = out.grad
                sm = np.exp(out.data) # = softmax(x)
                grad_contrib = g - sm * g.sum(axis=axis, keepdims=True)
                self.grad += grad_contrib
                
        if out.requires_grad:
            out._backward = _backward
        return out


    def reshape(self, *new_shape: int) -> 'Tensor':
        """Return a tensor with the same data but a different shape.

        Note: this creates a new node in the computation graph.
        """
        if -1 in new_shape:
            # Calculate the -1 dimension
            new_shape = list(new_shape)
            # Use numpy for shape operations (doesn't need device)
            known_prod = np.prod([d for d in new_shape if d != -1])
            new_shape[new_shape.index(-1)] = self.data.size // known_prod
        
        # Use numpy for shape operations (doesn't need device)
        assert np.prod(new_shape) == self.data.size, "Invalid shape for reshape"
        
        out = Tensor(self.data.reshape(new_shape), (self,), 'reshape')
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        
        if out.requires_grad:
            out._backward = _backward
        return out


    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the underlying data (tuple of ints)."""
        return self.data.shape

    @property
    def size(self) -> int:
        """Total number of elements in the tensor (alias for ``numel``)."""
        return int(self.data.size)

    @property
    def ndim(self) -> int:
        """Number of dimensions of the tensor."""
        return int(self.data.ndim)

    @property
    def numel(self) -> int:
        """PyTorch-style alias for the total number of elements in the tensor."""
        return int(self.data.size)

    @property
    def dim(self) -> int:
        """PyTorch-style alias for the number of dimensions of the tensor."""
        return int(self.data.ndim)

    def bool(self) -> 'Tensor':
        """
        Casts the tensor's data to a boolean data type.
        
        This is a non-differentiable operation and will detach
        the new tensor from the computation graph.
        """
        bool_data = self.data.astype(bool)
        
        out = Tensor(bool_data, requires_grad=False)
        return out
    
    def __bool__(self) -> bool:
        """
        Defines the behavior of the Tensor in a boolean context (e.g., `if tensor:`).
        
        Raises an error for multi-element tensors because their truth
        value is ambiguous.
        """
        if self.data.size == 1:
            # .item() extracts the single scalar value from the numpy array
            return bool(self.data.item())
        
        raise ValueError(
            "The truth value of a Tensor with more than one element is ambiguous. "
            "Use .any() or .all() if you want to check for element-wise truth."
        )
    
    def masked_fill(self, mask: 'Tensor', fill_value: float) -> 'Tensor':
        """
        Fills elements of self tensor with fill_value where mask is True.
        
        The mask tensor must be broadcastable to the shape of this tensor
        and should contain boolean values.
        """
        out_data = np.where(mask.data, fill_value, self.data)
        out = Tensor(out_data, (self,), 'masked_fill')
        
        def _backward():
            # 3. Backward Pass
            if self.requires_grad:
                grad_for_self = np.where(mask.data, 0.0, out.grad)
                
                # Add the gradient to the parent.
                self.grad += grad_for_self
    
        if self.requires_grad:
            out._backward = _backward
            
        return out
    
    def view(self, *new_shape):   
        """ 
        Reshape the tensor using a view-like API.

        This is a thin wrapper around :meth:`reshape` that accepts a
        variable number of dimensions or a single tuple, e.g.::

            x.view(2, 3)
            x.view((2, 3))
        """ 
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        return self.reshape(*new_shape)

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """Permute the dimensions of the tensor.

        Args:
            axes: Order of axes for the transposition.  If ``None``,
                reverses the order of all dimensions.
        """
        out = Tensor(np.transpose(self.data, axes=axes), (self,), 'transpose')
        
        def _backward():
            if self.requires_grad:
                if axes is None:
                    inverse_axes = None 
                else:
                    inverse_axes = tuple(np.argsort(axes))
                self.grad += np.transpose(out.grad, axes=inverse_axes)
        
        if out.requires_grad:
            out._backward = _backward
        return out


    def var(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = True) -> 'Tensor':
        """Sample variance of tensor elements (unbiased, denominator N-1).

        Args:
            axis: Axis or axes along which to compute the variance. If ``None``,
                variance is computed over all elements.
            keepdims: Whether to keep the reduced dimensions.
        """
        # Compute mean along the given axis.
        mean = self.mean(axis=axis, keepdims=True) # mu 
        diff = self - mean # (x_i - mu)  
        sq_diff = diff ** 2 # (x_i - mu)**2 

        # Number of elements along the reduction axes
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:  # tuple of axes
            # Use numpy for shape operations (doesn't need device)
            n = int(np.prod([self.data.shape[a] for a in axis]))

        # Use sample variance (N - 1 in the denominator) with a safe minimum of 1
        denom = max(n - 1, 1)
        var = sq_diff.sum(axis=axis, keepdims=keepdims) / denom  # sum(x_i - my)**2 / N 
        return var
    
    def std(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = True) -> 'Tensor':
        """Sample standard deviation of tensor elements (sqrt of sample variance)."""
        variance = self.var(axis=axis, keepdims=keepdims)
        std = variance.sqrt()
        return std
    
    def item(self) -> float: 
        """Returns the value of this tensor as a standard Python float.
        Only works for single-element tensors.
        """
        if self.data.size != 1:
            raise ValueError("item() can only be called on tensors with one element.")
        return float(self.data.flatten()[0])  
    

    def __getitem__(self, slices: Union[int, slice, Tuple]) -> 'Tensor':
        out = Tensor(self.data[slices], (self,), 'slice')
        
        def _backward():
            if self.requires_grad:
                # Create a grad array of zeros and "scatter" out.grad
                # into the locations specified by the slice
                grad_slice = np.zeros_like(self.data)
                grad_slice[slices] = out.grad
                self.grad += grad_slice
        
        if out.requires_grad:
            out._backward = _backward
        return out

    # --- Backward Pass ---
    
    def backward(self) -> None:
        """
        Performs backpropagation starting from this tensor.
        Assumes this tensor is the final output (e.g., a scalar loss).
        """
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on tensor that does not require_grad")
        
        # Build topological sort
        topo = []
        visited = set()
        def build_topo(v: 'Tensor'):
            if v not in visited and v.requires_grad:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # --- Initialize Gradients ---
        # 1. Set the seed gradient for the output tensor to 1
        self.grad = np.ones_like(self.data)
        
        # 2. Ensure all other tensors in the graph have zeroed gradients
        #    (This is technically optional if zero_grad() is used, but safer)
        for node in topo:
            if node is not self and node.grad is not None:
                node.grad.fill(0.0)
            elif node.grad is None and node.requires_grad: # Should not happen, but safeguard
                node.grad = np.zeros_like(node.data)

        # --- Propagate Gradients ---
        for node in reversed(topo):
            node._backward()
