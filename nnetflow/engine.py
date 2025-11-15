import numpy as np
from typing import Union, List, Literal, Tuple, Optional, Set
import scipy.special as sp 

class Tensor:
    """
    A simple autograd Tensor class supporting dynamic computation graphs
    and backpropagation.
    """
    def __init__(self, 
                 data: Union[np.ndarray, float, int, list, tuple], 
                 _children: Tuple['Tensor', ...] = (), 
                 _op: str = '', 
                 requires_grad: Optional[bool] = None) -> None:
        
        if not isinstance(data, np.ndarray):
            # If it's not an array (e.g., list, tuple, float, int), try to convert it.
            try:
                data = np.array(data, dtype=np.float64)
            except Exception as e:
                # This will catch truly weird inputs (like dicts)
                raise TypeError(f"Could not convert data of type {type(data)} to np.ndarray. Error: {e}")
        
        # Now we know 'data' is an ndarray.
        # We must ensure it's a float type for gradients.
        if not np.issubdtype(data.dtype, np.floating):
            # print(f"Warning: Converting non-float ndarray ({data.dtype}) to float64.")
            data = data.astype(np.float64)
        
        self.data = data
        self._op = _op
        self._prev: Set['Tensor'] = set(c for c in _children if isinstance(c, Tensor))

        # --- Grad Propagation Logic ---
        if requires_grad is None:
            # Infer: True if ANY child requires_grad
            self.requires_grad = any(c.requires_grad for c in self._prev)
        else:
            # Explicitly set (for leaf nodes)
            self.requires_grad = bool(requires_grad)

        # Initialize grad only if needed
        self.grad: Optional[np.ndarray] = np.zeros_like(self.data) if self.requires_grad else None
        
        # This function will be populated by the op that creates this Tensor
        self._backward = lambda: None

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
        # 1. Sum away extra dimensions added by broadcasting
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)  
            
        # 2. Sum along dimensions that were broadcasted (size 1)
        for i, (grad_dim, shape_dim) in enumerate(zip(grad.shape, shape)):
            if grad_dim != shape_dim:
                if shape_dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
                else:
                    # This should not happen if broadcasting was valid
                    raise ValueError(f"Cannot unbroadcast shape {grad.shape} to {shape}")
        return grad

    def __repr__(self) -> str:
        # Truncate data for cleaner printing
        data_str = np.array2string(self.data, max_line_width=70, precision=4, suppress_small=True)
        if '\n' in data_str:
            data_str = data_str.split('\n')[0] + '...]' # Show first line only if multi-line
        
        grad_info = ", grad_fn" if self._op else "" # Simplified grad_fn indicator
        return f"Tensor(data={data_str}, shape={self.shape}, requires_grad={self.requires_grad}{grad_info})"

    def zero_grad(self) -> None:
        """Resets the gradient of this tensor to zero."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    # --- Factory Methods ---
    
    @classmethod  
    def zeros(cls, *shape: int, requires_grad: bool = False) -> 'Tensor':
        """ 
        Args: 
            shape: the shape of your tensor 
            requires_grad: if the tensor requires gradient tracking 
        Returns: 
            Tensor filled with zero 
        """ 
        return cls(np.zeros(shape), requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape: int, requires_grad: bool = False) -> 'Tensor':
        """ 
        Args: 
            shape: the shape of the Tensor 
            requires_grad: if the Tensor require gradient tracking 
        Returns: 
            Tensor filled with ones 
        """ 
        return cls(np.ones(shape), requires_grad=requires_grad)

    @classmethod  
    def randn(cls, *shape: int, requires_grad: bool = False) -> 'Tensor':
        """ 
        creates a tensor filled with random numbers 
        Args: 
            shape: the shape of the Tensor 
            requires_grad: if the Tensor require gradient tracking 
        Returns: 
            Tensor 
        """ 
        data = np.random.randn(*shape).astype(np.float64) 
        return cls(data, requires_grad=requires_grad)

    @classmethod  
    def zeros_like(cls, tensor: 'Tensor', requires_grad: Optional[bool] = None) -> 'Tensor':
        """ 
        creates a Tensor filled with zeros of the shape of a given Tensor 
        Args: 
            tensor: tensor of the shape you want to create a new Tensor based on its shape 
            requires_grad: if the new created Tensor requires gradient tracking 
        Returns: 
            Tensor 
        """ 
        if requires_grad is None:
            requires_grad = tensor.requires_grad
        return cls(np.zeros_like(tensor.data), requires_grad=requires_grad)

    @classmethod
    def ones_like(cls, tensor: 'Tensor', requires_grad: Optional[bool] = None) -> 'Tensor':
        """ 
        create a tensor filled with ones of the shape of the passed tensor 
        Args: 
            tensor: Tensor of the shape you want to create 
            requires_grad: if the new tensor created will require gradient tracking 
        Returns: 
            Tensor 
        """ 
        if requires_grad is None:
            requires_grad = tensor.requires_grad
        return cls(np.ones_like(tensor.data), requires_grad=requires_grad)

    # --- Basic Arithmetic Ops ---

    def __add__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':  
        other_val = other.data if isinstance(other, Tensor) else other
        children = (self, other) if isinstance(other, Tensor) else (self,)
        
        out = Tensor(self.data + other_val, children, '+')
        
        def _backward():
            if self.requires_grad:
                self.grad += Tensor.unbroadcast(out.grad, self.data.shape)
            if isinstance(other, Tensor) and other.requires_grad:
                other.grad += Tensor.unbroadcast(out.grad, other.data.shape)
        
        if out.requires_grad:
            out._backward = _backward
        return out
    
    def add(self,x:'Tensor'): 
        """ 
        add two tensors to create a new tensor 
        Args: 
            x: Tensor to which to add to self tensor 
        """
        return self.__add__(x) 


    def __mul__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
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
    
    def matmul(self,x:'Tensor'): 
        """ 
        perform matrix multiplication of two tensors 
        Args: 
            x: the tensor on which self will perform matrix multiplication 
        """ 
        return self.__matmul__(x) 

    def __pow__(self, other: Union[float, int]) -> 'Tensor':
        assert isinstance(other, (float, int)), "Only support float and int power for Tensor"
        out = Tensor(self.data ** other, (self,), f'**{other}') 
        
        def _backward():
            if self.requires_grad:
                self.grad += (other * (self.data ** (other - 1))) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def __truediv__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        other_val = other.data if isinstance(other, Tensor) else other
        children = (self, other) if isinstance(other, Tensor) else (self,)
        
        # Note: We don't add epsilon here; user is responsible for 0-division.
        # Stability is added in log/sigmoid/softmax where it's unambiguous.
        out = Tensor(self.data / other_val, children, '/')
        
        def _backward():
            # Add epsilon to grad calculation to avoid 1/0
            other_val_safe = other_val + 1e-8
            if self.requires_grad:
                self.grad += Tensor.unbroadcast((1 / other_val_safe) * out.grad, self.data.shape)
            if isinstance(other, Tensor) and other.requires_grad:
                other.grad += Tensor.unbroadcast((-self.data / (other_val_safe ** 2)) * out.grad, other.data.shape)
                
        if out.requires_grad:
            out._backward = _backward
        return out

    def __neg__(self) -> 'Tensor':
        return self * -1

    def __sub__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        return self + (other * -1)

    # Reflected ops (for `5 + tensor`, etc.)
    def __radd__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        return self + other

    def __rmul__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        return self * other

    def __rsub__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        return (self * -1) + other

    def __rtruediv__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        return other * (self ** -1)

    # --- Matrix/Reduction Ops ---

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        assert isinstance(other, Tensor), "Only support Tensor type for matmul operation"
        out = Tensor(self.data @ other.data, (self, other), '@')
        
        def _backward():
            if self.requires_grad:
                # (dL/dC) @ B^T
                other_transposed = np.swapaxes(other.data, -1, -2)
                self_grad_contrib = out.grad @ other_transposed
                self.grad += Tensor.unbroadcast(self_grad_contrib, self.data.shape)
            
            if other.requires_grad:
                # A^T @ (dL/dC)
                self_transposed = np.swapaxes(self.data, -1, -2)
                other_grad_contrib = self_transposed @ out.grad
                other.grad += Tensor.unbroadcast(other_grad_contrib, other.data.shape)
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), 'sum')
        
        def _backward():
            if self.requires_grad:
                # The gradient needs to be broadcasted back to the original shape
                if axis is None:
                    # Scalar sum, grad is repeated across all elements
                    grad_expanded = np.ones_like(self.data) * out.grad
                else:
                    # Sum along axis, grad is repeated along that axis
                    # We can use np.ones * expanded_grad for a general solution
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
        # Determine the number of elements being averaged over
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else: # axis is a tuple
            n = np.prod([self.data.shape[i] for i in axis])
        
        # Implement mean as sum * (1/n) so (1/n) is part of the graph
        sum_out = self.sum(axis=axis, keepdims=keepdims)
        out = sum_out * (1.0 / n) # This creates a Mul node
        out._op = 'mean' # Override op label
        return out
        
    # --- Unary Ops (Activations, etc.) ---

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
        """Natural logarithm (ln)"""
        if not np.all(self.data > 0):
            print("Warning: log applied to non-positive elements.")
        
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

    def log10(self) -> 'Tensor':
        """Base-10 logarithm"""
        if not np.all(self.data > 0):
            print("Warning: log10 applied to non-positive elements.")
        
        out = Tensor(np.log10(self.data), (self,), 'log10') 
        
        def _backward():
            if self.requires_grad:
                # Add epsilon for numerical stability in gradient
                self.grad += (1 / ((self.data + 1e-8) * np.log(10))) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    # --- Activation Functions ---

    def relu(self) -> 'Tensor':
        """ 
        Perform relu activation pased on this paper : https://arxiv.org/abs/1803.08375
        """ 
        out = Tensor(np.maximum(self.data, 0), (self,), 'relu')
        
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def leaky_relu(self, alpha: float = 0.01) -> 'Tensor': 
        """ 
        perform activation pased on this paper : https://arxiv.org/abs/1505.00853 
        """ 
        out = Tensor(np.where(self.data > 0, self.data, alpha * self.data), (self,), 'leaky_relu')
        
        def _backward():
            if self.requires_grad:
                self.grad += np.where(self.data > 0, 1, alpha) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def elu(self, alpha: float = 1.0) -> 'Tensor':
        """ 
        perform activation based on this paper : https://arxiv.org/abs/1511.07289 
        """ 
        out = Tensor(np.where(self.data > 0, self.data, alpha * (np.exp(self.data) - 1)), (self,), 'elu')
        
        def _backward():
            if self.requires_grad:
                # d/dx(alpha * (exp(x) - 1)) = alpha * exp(x)
                self.grad += np.where(self.data > 0, 1, alpha * np.exp(self.data)) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def selu(self, alpha: float = 1.67326, scale: float = 1.0507) -> 'Tensor': # Renamed beta to scale
        """ 
        perform activation based on this paper : https://arxiv.org/abs/1706.02515 
        """ 
        out = Tensor(scale * np.where(self.data > 0, self.data, alpha * (np.exp(self.data) - 1)), (self,), 'selu')
        
        def _backward():
            if self.requires_grad:
                self.grad += scale * np.where(self.data > 0, 1, alpha * np.exp(self.data)) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def gelu(self) -> 'Tensor':
        """ 
        perform gelu activation based on this paper : https://arxiv.org/abs/1606.08415 
        """ 
        # Using the scipy.special.erf implementation
        out_data = 0.5 * self.data * (1 + sp.erf(self.data / np.sqrt(2)))
        out = Tensor(out_data, (self,), 'gelu')
        
        def _backward():
            if self.requires_grad:
                sqrt_2pi = np.sqrt(2 * np.pi)
                cdf = 0.5 * (1 + sp.erf(self.data / np.sqrt(2)))
                pdf = (1 / sqrt_2pi) * np.exp(-0.5 * self.data ** 2)
                self.grad += (cdf + self.data * pdf) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def sigmoid(self) -> 'Tensor':
        """ 
        perform sigmoid activation  
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
        """ 
        perform swish activation pased on this paper : https://arxiv.org/pdf/1710.05941v1 
        """ 
        # swish(x) = x * sigmoid(x)
        # We can re-use our stable sigmoid
        sig = self.sigmoid() 
        out = self * sig # This builds the graph!
        out._op = 'swish'
        return out

    def tanh(self) -> 'Tensor':
        """
        perform tanh to the Tensor 
        """ 
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        
        def _backward():
            if self.requires_grad:
                self.grad += (1 - t ** 2) * out.grad
        
        if out.requires_grad:
            out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> 'Tensor':
        """ 
        perform softmax operation  to the Tensor 
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
        """ 
        perform log_sotmax 
        """ 
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

    # --- Reshaping/Indexing Ops ---

    def reshape(self, *new_shape: int) -> 'Tensor':
        """ 
        reshape the tensor to a new shape , rember that this creates a new tensor not efficient :( 
        """ 
        if -1 in new_shape:
            # Calculate the -1 dimension
            new_shape = list(new_shape)
            known_prod = np.prod([d for d in new_shape if d != -1])
            new_shape[new_shape.index(-1)] = self.data.size // known_prod
        
        assert np.prod(new_shape) == self.data.size, "Invalid shape for reshape"
        
        out = Tensor(self.data.reshape(new_shape), (self,), 'reshape')
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        
        if out.requires_grad:
            out._backward = _backward
        return out

    # --- Shape / size helpers ---

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

    def numel(self) -> int:
        """PyTorch-style alias for the total number of elements in the tensor."""
        return int(self.data.size)

    def dim(self) -> int:
        """PyTorch-style alias for the number of dimensions of the tensor."""
        return int(self.data.ndim)

    def bool(self) -> 'Tensor':
        """
        Casts the tensor's data to a boolean data type.
        
        This is a non-differentiable operation and will detach
        the new tensor from the computation graph.
        """
        # We use .data to get the numpy array and .astype() to cast it
        bool_data = self.data.astype(bool)
        
        # Create a new *leaf* Tensor with no gradient history.
        # This is correct because the operation is not differentiable.
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
    
    def view(self,new_shape):  # mybad :( i am inefficient but that okay 
        """ 
        reshape the tensor to the new shape but creates a new tensor not efficient  
        """ 
        return self.reshape(new_shape)

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """ 
        transpose a tensor on the given axes but create a new tensor , not efficient 
        """ 
        out = Tensor(np.transpose(self.data, axes=axes), (self,), 'transpose')
        
        def _backward():
            if self.requires_grad:
                # The inverse of a transpose is a transpose with the inverse permutation
                if axes is None:
                    inverse_axes = None # Standard matrix transpose
                else:
                    inverse_axes = tuple(np.argsort(axes))
                self.grad += np.transpose(out.grad, axes=inverse_axes)
        
        if out.requires_grad:
            out._backward = _backward
        return out


    def var(self,axis:int=0,keepdims=True) -> 'Tensor':
        """variance of all elements in the tensor,this is a sample variance (N-1 in denominator)""" 
        mean = self.mean(axis=axis,keepdims=True)
        diff = self - mean 
        sq_diff = diff ** 2
        var = sq_diff.sum(axis=axis,keepdims=keepdims) / (self.shape[axis] -1)
        return var
    
    def std(self,axis:int=0,keepdim=True) -> 'Tensor':
        """standard deviation of all elements in the tensor,this is a sample std (N-1 in denominator)""" 
        variance = self.var(axis=axis,keepdim=keepdim)
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