from typing import List, Tuple
import numpy as np

class Tensor:
    __doc__ = """"Tensor class for automatic differentiation and basic tensor operations.
    This class supports operations like addition, multiplication, matrix multiplication,
    summation, reshaping, and activation functions (ReLU, sigmoid, tanh, etc.).
    It also implements backward propagation to compute gradients.
    Attributes:
        data (np.ndarray): The tensor data.
        grad (np.ndarray): The gradient of the tensor.
        _backward (callable): The function to compute gradients during backpropagation.
        _prev (set): Previous tensors in the computation graph.
        _op (str): The operation that produced this tensor.
        shape (Tuple): The shape of the tensor data.
    """
    def __init__(self, data: List[float] | np.ndarray, shape: Tuple = (1,), _children=(), _op='', dtype=None):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data, dtype=dtype if dtype else float).reshape(shape)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.shape = self.data.shape

    @staticmethod
    def _unbroadcast(grad, shape):
        """"Unbroadcast the gradient to match the shape of the original tensor.
        This function reduces the dimensions of the gradient to match the shape of the tensor
        by summing over dimensions where the tensor shape is 1.
        Args:
            grad (np.ndarray): The gradient to unbroadcast.
            shape (Tuple): The shape of the original tensor.
        Returns:
            np.ndarray: The unbroadcasted gradient.
        """
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
            if s_dim == 1 and g_dim != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def sum(self, axis=None, keepdims=False):
        """Sum the tensor along specified axes.
        Args:
            axis (int, tuple, list, optional): Axis or axes along which to sum. If None, sums all dimensions.
            keepdims (bool, optional): If True, retains reduced dimensions with size 1.
        Returns:
            Tensor: A new tensor with the summed data.
        """
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

    def __add__(self, other):
        """Add another tensor or a scalar to this tensor.
        Args:
            other (Tensor, int, float): The tensor or scalar to add.
        Returns:
            Tensor: A new tensor with the result of the addition.
        """
        other = other if isinstance(other, Tensor) else Tensor([other])
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        """Matrix multiplication with another tensor.
        Args:
            other (Tensor): The tensor to multiply with.
        Returns:
            Tensor: A new tensor with the result of the matrix multiplication.
        """
        assert isinstance(other, Tensor), "unsupported operation"
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@')

        def _backward():
            self_grad = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
            other_grad = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
            # Sum over broadcasted batch dimension for self
            if self.data.shape[0] == 1 and out.data.shape[0] > 1:
                self_grad = self_grad.sum(axis=0, keepdims=True)
            # Sum over broadcasted batch dimension for other, if applicable
            if other.data.shape[0] == 1 and out.data.shape[0] > 1:
                other_grad = other_grad.sum(axis=0, keepdims=True)
            self.grad += self_grad
            other.grad += other_grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        """Element-wise multiplication with another tensor or a scalar.
        Args:
            other (Tensor, int, float): The tensor or scalar to multiply with.
        Returns:
            Tensor: A new tensor with the result of the multiplication.
        """
        if isinstance(other, int): other = float(other)
        other = other if isinstance(other, Tensor) else Tensor([other])
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def relu(self):
        """Apply the ReLU activation function.
        Returns:
            Tensor: A new tensor with the result of the ReLU activation.
        """
        out_data = np.where(self.data < 0, 0, self.data)
        out = Tensor(out_data, _children=(self,), _op='ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        """Apply the sigmoid activation function.
        Returns:
            Tensor: A new tensor with the result of the sigmoid activation.
        """
        clipped = np.clip(self.data, -60, 60)
        out_data = 1 / (1 + np.exp(-clipped))
        out = Tensor(out_data, _children=(self,), _op='sigmoid')

        def _backward():
            self.grad += out.grad * out.data * (1 - out.data)
        out._backward = _backward
        return out

    def exp(self):
        """Apply the exponential activation function.
        Returns:
            Tensor: A new tensor with the result of the exponential activation.
        """
        clipped = np.clip(self.data, -60, 60)
        out_data = np.exp(clipped)
        out = Tensor(out_data, _children=(self,), _op='exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self, eps=1e-8):
        """Apply the logarithm activation function.
        Args:
            eps (float, optional): A small value to avoid log(0). Defaults to 1e-8.
        Returns:
            Tensor: A new tensor with the result of the logarithm activation.
        """
        safe_data = np.clip(self.data, eps, None)
        out = Tensor(np.log(safe_data), _children=(self,), _op='log')

        def _backward():
            self.grad += out.grad / safe_data
        out._backward = _backward
        return out

    def tanh(self):
        """Apply the hyperbolic tangent activation function.
        Returns:
            Tensor: A new tensor with the result of the tanh activation.
        """
        out_data = np.tanh(self.data)
        out = Tensor(out_data, _children=(self,), _op='tanh')

        def _backward():
            self.grad += out.grad * (1 - out.data**2)
        out._backward = _backward
        return out

    def __pow__(self, power):
        """Raise the tensor to a scalar power.
        Args:
            power (int, float): The scalar power to raise the tensor to.
        Returns:
            Tensor: A new tensor with the result of the power operation.
        """
        assert isinstance(power, (int, float)), "only supports scalar powers"
        out_data = np.power(self.data, power)
        out = Tensor(out_data, _children=(self,), _op='pow')

        def _backward():
            self.grad += out.grad * (power * np.power(self.data, power - 1))
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def reshape(self, *shape):
        """Reshape the tensor to a new shape.
        Args:
            shape (int, ...): The new shape to reshape the tensor to.
        Returns:
            Tensor: A new tensor with the reshaped data.
        """
        out = Tensor(self.data.reshape(shape), _children=(self,), _op='reshape')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def zero_grad(self):
        """Reset the gradient of the tensor to zero.
        This is useful before starting a new backward pass to avoid accumulating gradients.
        """
        self.grad = np.zeros_like(self.data)

    def backward(self, grad_clip=None):
        """Perform backpropagation to compute gradients.
        This method traverses the computation graph in reverse order and computes gradients
        for each tensor using the stored backward functions.
        Args:
            grad_clip (float, optional): If provided, clips the gradients to this value.
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
            if np.isnan(v.grad).any():  # i dont want NaN to get away with it 
                print(f"NaN in gradients of node with op: {v._op}")
            v.grad = np.nan_to_num(v.grad, nan=0.0, posinf=1e5, neginf=-1e5)
            if grad_clip is not None:
                if isinstance(v.grad, np.ndarray) and np.issubdtype(v.grad.dtype, np.floating):
                    np.clip(v.grad, -grad_clip, grad_clip, out=v.grad)
                elif isinstance(v.grad, float):
                    v.grad = float(np.clip(v.grad, -grad_clip, grad_clip))

    def __neg__(self): return self * -1.0
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __repr__(self): return f"Tensor(data={self.data}, grad={self.grad})"