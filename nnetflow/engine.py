"""
tensor
======

A minimal, NumPy-backed autograd engine.

This module implements a single class, :class:`Tensor`, which wraps a
``numpy.ndarray`` and records the operations performed on it in a dynamic
computation graph. Calling :meth:`Tensor.backward` on a scalar (or
single-element) output walks that graph in reverse topological order and
accumulates gradients into each participating tensor's ``.grad`` attribute,
in the same spirit as PyTorch's ``autograd`` (though far simpler and
intended primarily for learning / small-scale experimentation).

Typical usage::

    >>> import numpy as np
    >>> from tensor import Tensor
    >>> a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    >>> b = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
    >>> c = (a * b).sum()
    >>> c.backward()
    >>> a.grad
    array([4., 5., 6.], dtype=float32)

Key concepts
------------
* **Graph construction**: every operation (``+``, ``*``, ``@``, ``exp``,
  ``relu``, ...) returns a *new* :class:`Tensor` whose ``_prev`` set
  references the input tensor(s) that produced it, and whose ``_backward``
  closure knows how to push gradients back to those inputs (a
  vector-Jacobian product).
* **requires_grad propagation**: a tensor requires gradients if it was
  explicitly asked to, or if *any* of its parents do.
* **Gradient accumulation**: gradients are *added into* ``.grad`` (via
  ``np.add(..., out=...)``) rather than overwritten, matching PyTorch
  semantics. Call :meth:`Tensor.zero_grad` between optimization steps to
  reset them.
* **Broadcasting**: NumPy broadcasting is supported transparently on the
  forward pass; :meth:`Tensor.unbroadcast` reduces an incoming gradient back
  down to the original (pre-broadcast) shape on the backward pass.

Notes on numerical stability
-----------------------------
Several ops that involve a division or a logarithm (``log``, ``log10``,
``sqrt``, ``__truediv__``'s local gradient, ``softmax``, ``log_softmax``)
add a small ``epsilon`` (``1e-8``) to denominators to avoid division by
zero / ``-inf`` results when inputs are exactly zero.
"""

import numpy as np
import numpy.typing as npt
import warnings
from typing import Any, Union, Tuple, Optional, Set

import scipy.special as sp

epsilon = 1e-8


class Tensor:
    """
    The base class that represents `Tensor`, it wraps a numpy array and provides
    a computation graph for automatic differentiation.
    """

    def __init__(
        self,
        data: npt.ArrayLike = None,
        _children: Tuple['Tensor', ...] = (),
        _op: str = '',
        requires_grad: Optional[bool] = None,
        dtype: Optional[npt.DTypeLike] = None,
        copy: bool = False
    ) -> None:
        """
        Initializes a new Tensor instance.
        Args:
            data: The data to be stored in the tensor. Can be a numpy array, list, or scalar.
            _children: A tuple of parent tensors that this tensor depends on (for autograd).
            _op: A string representing the operation that produced this tensor (for autograd).
            requires_grad: If True, gradients will be computed for this tensor during backpropagation.
            dtype: The desired data type for the tensor. If None, it will be inferred from the data.
            copy: If True, and you used a numpy array as input, the data will be copied to avoid unexpected changes in the original array.
        Returns:
            None
        Raises:
            TypeError: If the data cannot be converted to a numpy array.
        Examples:
            >>> a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
            >>> b = Tensor([3.0, 4.0], requires_grad=False)
            >>> c = Tensor(5.0, requires_grad=True)
            >>> d = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True,dtype=np.float64,copy=True)
        """
        if isinstance(data, Tensor):
            if requires_grad is None:
                requires_grad = data.requires_grad
            data = data.data
        if dtype is not None:
            target_dtype = np.dtype(dtype)
        elif hasattr(data, 'dtype'):
            target_dtype = data.dtype
        else:
            target_dtype = np.float32

        if hasattr(data, 'dtype'):
            self.data = data.astype(target_dtype, copy=copy)
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
            # Gradients are always accumulated in a floating-point buffer.
            # If the tensor's own dtype is already float32 or float64 we
            # reuse it so grad and data share precision; any other dtype
            # (int, bool, float16, ...) falls back to float32.
            grad_dtype = self.data.dtype if self.data.dtype in (np.float32, np.float64) else np.float32
            self.grad: Optional[np.ndarray] = np.zeros_like(self.data, dtype=grad_dtype)
        else:
            self.grad: Optional[np.ndarray] = None

        self._backward = lambda: None

    def __getstate__(self) -> dict:
        """
        Called when pickling the Tensor.  Excludes the _backward closure
        Returns:
            dict: The state of the Tensor, excluding the _backward closure.
        Raises:
            None
        Examples:
            >>> a = Tensor(np.random.rand(2,3), requires_grad=True)
            >>> a.__getstate__()
            {'data': array([[0.24210482, 0.7320231 , 0.89306345],
                    [0.50106301, 0.37299244, 0.94203871]]),
            '_op': '',
            '_prev': set(),
            'requires_grad': False,
            'grad': None}
        """
        state = self.__dict__.copy()
        if '_backward' in state:
            del state['_backward']
        return state

    @classmethod
    def _check_dtype(cls, A: 'Tensor', B: 'Tensor') -> bool:
        """
        Checks if two `Tensor` instances have the same data type
        Args:
            A: first `Tensor` instance
            B: second `Tensor` instance
        Returns:
            bool: True if both tensors have the same data type
        Raises:
            ValueError: If the data types of the two tensors do not match
        Examples:
            >>> a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
            >>> b = Tensor(np.array([3.0, 4.0]), requires_grad=False)
            >>> Tensor._check_dtype(a, b)
            True
            >>> c = Tensor(np.array([5, 6]), requires_grad=True, dtype=np.int32)
            >>> Tensor._check_dtype(a,c)
            Traceback (most recent call last):
                ...
            ValueError: Tensor dtype mismatch: float32 != int32
        """
        if A.dtype != B.dtype:
            raise ValueError(f"Tensor dtype mismatch: {A.dtype} != {B.dtype}")
        return True

    def __setstate__(self, state: dict) -> None:
        """
        Called when unpickling — restores a no-op backward closure.

        A pickled tensor has, by definition, already finished participating
        in whatever forward pass created it, so there is nothing meaningful
        for ``_backward`` to do; it is restored as a no-op rather than
        raising, so the unpickled tensor can still safely appear as a leaf
        (e.g. re-loaded model weights).

        Args:
            state: The ``__dict__`` produced by :meth:`__getstate__`.
        Returns:
            None
        Raises:
            None
        """
        self.__dict__.update(state)
        self._backward = lambda: None

    @property
    def dtype(self) -> npt.DTypeLike:
        """dtype property
        Returns:
            np.dtype
        Raises:
            None
        Examples:
            a = Tensor(np.random.randn(2,3))
            a.dtype
            dtype('float64')
        """
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
        Examples:
            >>> a = Tensor(np.random.randn(2,3))
            >>> b = a.to(np.float16)
        """
        new_data = self.data.astype(dtype)
        return Tensor(new_data, requires_grad=False)

    def astype(self, dtype: npt.DTypeLike) -> 'Tensor':
        """Alias for :meth:`to`.  Returns a new tensor with the given dtype.
        see :meth: `to` for more information
        """
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
        Raises:
            ValueError : if the gradient cannot be unbroadcasted to the target shape
        """
        grad = np.asarray(grad)
        if grad.shape == shape:
            return grad

        ndim_diff = grad.ndim - len(shape)
        if ndim_diff < 0:
            raise ValueError(f"Cannot unbroadcast shape {grad.shape} to target shape {shape}")

        padded_shape = (1,) * ndim_diff + shape
        axes = []

        for axis, (g_dim, p_dim) in enumerate(zip(grad.shape, padded_shape)):
            if g_dim == p_dim:
                continue
            elif p_dim == 1:
                axes.append(axis)
            else:
                raise ValueError(f"Cannot unbroadcast shape {grad.shape} to target shape {shape}")

        if axes:
            grad = grad.sum(axis=tuple(axes), keepdims=True)

        return grad.reshape(shape)

    def __repr__(self) -> str:
        """Return a human-readable string representation of the Tensor.

        The underlying data is truncated to its first line when it spans
        multiple lines, so large tensors don't flood the console; shape,
        ``requires_grad`` and (if this tensor was produced by an op) a
        ``grad_fn`` marker are always shown.

        Returns:
            str: A single-line summary of the tensor.
        """
        # Convert to numpy for display purposes
        data_np = np.asarray(self.data)
        data_str = np.array2string(data_np, max_line_width=70, precision=4, suppress_small=True)
        if '\n' in data_str:
            data_str = data_str.split('\n')[0] + '...]'  # Show first line only if multi-line
        grad_info = ", grad_fn" if self._op else ""  # Simplified grad_fn indicator
        return f"Tensor(data={data_str}, shape={self.shape}, requires_grad={self.requires_grad}{grad_info})"

    def zero_grad(self) -> None:
        """Resets the gradient of this tensor to zero (in place).

        No-op if ``requires_grad`` is ``False`` (there is no gradient
        buffer to reset). Call this on every leaf/parameter tensor before
        each new ``backward()`` call if you are *not* relying on
        :meth:`backward`'s own internal zeroing of intermediate nodes —
        e.g. between optimizer steps in a training loop.

        Returns:
            None
        Example:
            >>> a = Tensor(np.random.randn(2, 3), requires_grad=True)
            >>> b = Tensor(np.random.randn(3, 2), requires_grad=True)
            >>> c = a @ b
            >>> out = c.sum()
            >>> out.backward()
            >>> a.zero_grad()
        """
        if self.requires_grad:
            grad_dtype = self.data.dtype if self.data.dtype in (np.float32, np.float64) else np.float32
            self.grad = np.zeros_like(self.data, dtype=grad_dtype)

    @classmethod
    def zeros(cls, *shape: int, requires_grad: bool = False, dtype: Optional[npt.DTypeLike] = None) -> 'Tensor':
        """Create a tensor filled with zeros.
        Args:
            *shape: shape of the `Tensor` to create
            requires_grad: bool of if the `Tensor` being created will require `grad` tracking
            dtype: data type of the `data` in the `Tensor`
        Returns:
            Tensor
        Examples:
            >>> a = Tensor.zeros(2,3)
        """
        return cls(np.zeros(shape), requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def ones(cls, *shape: int, requires_grad: bool = False, dtype: Optional[npt.DTypeLike] = None) -> 'Tensor':
        """Create a tensor filled with ones.
         Args:
            *shape: shape of the `Tensor` to create
            requires_grad: bool of if the `Tensor` being created will require `grad` tracking
            dtype: data type of the `data` in the `Tensor`
        Returns:
            Tensor
        Examples:
            >>> a = Tensor.ones(2,3)
        """
        return cls(np.ones(shape), requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def randn(cls, *shape: int, requires_grad: bool = False, dtype: Optional[npt.DTypeLike] = None) -> 'Tensor':
        """Create a tensor filled with random numbers from a standard normal distribution.
        Args:
            *shape: shape of the `Tensor` to create
            requires_grad: bool of if the `Tensor` being created will require `grad` tracking
            dtype: data type of the `data` in the `Tensor`
        Returns:
            Tensor
        Examples:
            >>> a = Tensor.randn(2,3)
        """
        data = np.random.randn(*shape).astype(dtype=dtype if dtype else np.float32)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def rand(cls, *shape: int, requires_grad: bool = False, dtype: Optional[npt.DTypeLike] = None) -> 'Tensor':
        """Create a tensor filled with random numbers from a uniform distribution over [0, 1).
        Args:
            *shape: shape of the `Tensor` to create
            requires_grad: bool of if the `Tensor` being created will require `grad` tracking
            dtype: data type of the `data` in the `Tensor`
        Returns:
            Tensor
        Examples:
            >>> a = Tensor.rand(2,3)
        """
        data = np.random.rand(*shape).astype(dtype=dtype if dtype else np.float32)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def zeros_like(cls, tensor: 'Tensor', requires_grad: Optional[bool] = None, dtype: Optional[npt.DTypeLike] = None) -> 'Tensor':
        """Create a `Tensor` of zeros with the same shape as *tensor*.
        Args:
            tensor: `Tensor` for which shape will be inferenced for new `Tensor`
            requires_grad: bool of if the `Tensor` being created will require `grad` tracking
            dtype: data type of the `data` in the `Tensor`
        Returns:
            Tensor
        Examples:
            >>> a = Tensor.randn(2,3)
            >>> b = Tensor.zeros_like(a)
        """
        if requires_grad is None:
            requires_grad = tensor.requires_grad
        return cls(np.zeros_like(tensor.data), requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def ones_like(cls, tensor: 'Tensor', requires_grad: Optional[bool] = None, dtype: Optional[npt.DTypeLike] = None) -> 'Tensor':
        """Create a tensor of ones with the same shape as *tensor*.
        Args:
            tensor: `Tensor` for which shape will be inferenced for new `Tensor`
            requires_grad: bool of if the `Tensor` being created will require `grad` tracking
            dtype: data type of the `data` in the `Tensor`
        Returns:
            Tensor
        Examples:
            >>> a = Tensor.randn(2,3)
            >>> b = Tensor.ones_like(a)
        """
        if requires_grad is None:
            requires_grad = tensor.requires_grad
        return cls(np.ones_like(tensor.data), requires_grad=requires_grad, dtype=dtype)

    # --- operations
    def __add__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        """Element-wise addition (``self + other``).

        Supports adding a Tensor to another Tensor, a scalar, or a NumPy array.
        Broadcasting is handled automatically during the backward pass.
        Args:
            other: a `Tensor`,float,int or numpy array to add to the this `Tensor`
        Returns:
            `Tensor`
        Raises:
            ValueError: if `other` is a `Tensor` with a different dtype than `self`.
        Examples:
            a = Tensor.randn(2,3)
            c = a + 23
        """
        if isinstance(other, Tensor):
            Tensor._check_dtype(self, other)
        other_val = other.data if isinstance(other, Tensor) else other
        children = (self, other) if isinstance(other, Tensor) else (self,)
        out = Tensor(self.data + other_val, children, '+')

        def _backward():
            if self.requires_grad:
                np.add(self.grad, Tensor.unbroadcast(out.grad, self.data.shape), out=self.grad)
            if isinstance(other, Tensor) and other.requires_grad:
                np.add(other.grad, Tensor.unbroadcast(out.grad, other.data.shape), out=other.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def add(self, x: 'Tensor') -> 'Tensor':
        """Functional form of addition.  Equivalent to ``self + x``.
        see :meth: `__add__`
        """
        return self.__add__(x)

    def __mul__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        """Element-wise multiplication (``self * other``).

        The local gradient rule is:
            d(a * b)/da = b  and  d(a * b)/db = a
        so in the backward pass the incoming gradient is scaled by the *other*
        operand's data.

        Args:
            other: a `Tensor`, float, int, or numpy array to multiply with this `Tensor`.
        Returns:
            `Tensor`
        Raises:
            ValueError: if `other` is a `Tensor` with a different dtype than `self`.
        Examples:
            >>> a = Tensor.randn(2, 3)
            >>> c = a * 2.0
        """
        if isinstance(other, Tensor):
            Tensor._check_dtype(self, other)
        other_val = other.data if isinstance(other, Tensor) else other
        children = (self, other) if isinstance(other, Tensor) else (self,)

        out = Tensor(self.data * other_val, children, '*')

        def _backward():
            if self.requires_grad:
                np.add(self.grad, Tensor.unbroadcast(other_val * out.grad, self.data.shape), out=self.grad)
            if isinstance(other, Tensor) and other.requires_grad:
                np.add(other.grad, Tensor.unbroadcast(self.data * out.grad, other.data.shape), out=other.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def matmul(self, x: 'Tensor') -> 'Tensor':
        """Functional form of matrix multiplication.  Equivalent to ``self @ x``."""
        return self.__matmul__(x)

    def __pow__(self, other: Union[float, int]) -> 'Tensor':
        """Element-wise power (``self ** other``).  Only scalar exponents are supported.

        Local gradient rule:
            d(x**p)/dx = p * x**(p - 1)

        Args:
            other: The (scalar, non-tensor) exponent to raise every element to.
        Returns:
            `Tensor`
        Raises:
            AssertionError: if `other` is not a Python ``float`` or ``int``.
        Examples:
            >>> a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
            >>> b = a ** 2
        """
        assert isinstance(other, (float, int)), "Only support float and int power for Tensor"
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            if self.requires_grad:
                np.add(self.grad, (other * (self.data ** (other - 1))) * out.grad, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def __truediv__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        """Element-wise division (``self / other``).

        Local gradient rules:
            d(a/b)/da =  1 / b
            d(a/b)/db = -a / b**2

        Args:
            other: a `Tensor`, float, int, or numpy array to divide this `Tensor` by.
        Returns:
            `Tensor`
        Raises:
            ValueError: if `other` is a `Tensor` with a different dtype than `self`.
        Examples:
            >>> a = Tensor(np.array([4.0, 9.0]), requires_grad=True)
            >>> b = a / 2.0
        """
        if isinstance(other, Tensor):
            Tensor._check_dtype(self, other)
        other_val = other.data if isinstance(other, Tensor) else other
        children = (self, other) if isinstance(other, Tensor) else (self,)

        out = Tensor(self.data / other_val, children, '/')

        def _backward():
            if self.requires_grad:
                np.add(self.grad, Tensor.unbroadcast((1 / other_val) * out.grad, self.data.shape), out=self.grad)
            if isinstance(other, Tensor) and other.requires_grad:
                np.add(other.grad, Tensor.unbroadcast((-self.data / (other_val ** 2)) * out.grad, other.data.shape), out=other.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def __neg__(self) -> 'Tensor':
        """Element-wise negation (``-self``).  Equivalent to ``self * -1``.

        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([1.0, -2.0]), requires_grad=True)
            >>> b = -a
        """
        return self * -1

    def __sub__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        """Element-wise subtraction (``self - other``).  Equivalent to ``self + (other * -1)``.

        Args:
            other: a `Tensor`, float, int, or numpy array to subtract from this `Tensor`.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([5.0, 6.0]), requires_grad=True)
            >>> b = a - 1.0
        """
        return self + (other * -1)

    def __radd__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        """Reflected addition (``other + self``), for ``other`` on the left of a non-Tensor operand.

        Args:
            other: a float, int, or numpy array.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
            >>> b = 5 + a
        """
        return self + other

    def __rmul__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        """Reflected multiplication (``other * self``), for ``other`` on the left of a non-Tensor operand.

        Args:
            other: a float, int, or numpy array.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
            >>> b = 3 * a
        """
        return self * other

    def __rsub__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        """Reflected subtraction (``other - self``), for ``other`` on the left of a non-Tensor operand.

        Args:
            other: a float, int, or numpy array.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
            >>> b = 5 - a
        """
        return (self * -1) + other

    def __rtruediv__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        """Reflected division (``other / self``), for ``other`` on the left of a non-Tensor operand.

        Implemented as ``other * self**-1`` so it reuses the ``__pow__``
        and ``__mul__`` backward rules rather than defining a new one.

        Args:
            other: a float, int, or numpy array.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([2.0, 4.0]), requires_grad=True)
            >>> b = 1.0 / a
        """
        return other * (self ** -1)

    @classmethod
    def can_matmul(cls, shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> bool:
        """Check if two shapes can be matrix multiplied together.

        Validates NumPy ``@``-style matmul compatibility: the last
        dimension of ``shape_a`` must match the second-to-last dimension
        of ``shape_b`` (or its only dimension, if ``shape_b`` is 1-D), and
        any leading batch dimensions must be broadcastable against each
        other.

        Args:
            shape_a: Shape of the left-hand operand.
            shape_b: Shape of the right-hand operand.
        Returns:
            bool: True if the two shapes are compatible for matmul, False otherwise.
        Examples:
            >>> Tensor.can_matmul((2, 3), (3, 4))
            True
            >>> Tensor.can_matmul((2, 3), (4, 4))
            False
        """
        if not shape_a or not shape_b:
            return False

        # Inner matrix dimensions must match (A's last dim vs B's second-to-last dim)
        inner_b = shape_b[-2] if len(shape_b) > 1 else shape_b[-1]
        if shape_a[-1] != inner_b:
            return False

        try:
            np.broadcast_shapes(shape_a[:-2], shape_b[:-2])
            return True
        except ValueError:
            return False

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication (``self @ other``).

        Supports batched matmul (leading broadcastable batch dimensions)
        as well as matrix-vector and vector-matrix products, mirroring
        ``numpy.matmul`` semantics. See :meth:`can_matmul` for the shape
        compatibility rule.

        Args:
            other: The right-hand `Tensor` operand. Must be a `Tensor`
                (unlike the other arithmetic ops, raw arrays/scalars are
                not accepted here).
        Returns:
            `Tensor`
        Raises:
            AssertionError: if `other` is not a `Tensor`.
            ValueError: if the dtypes of `self` and `other` do not match, or
                if their shapes are not aligned for matmul.
        Examples:
            >>> a = Tensor.randn(2, 3, requires_grad=True)
            >>> b = Tensor.randn(3, 4, requires_grad=True)
            >>> c = a @ b
        """
        assert isinstance(other, Tensor), "Only support Tensor type for matmul operation"
        Tensor._check_dtype(self, other)

        can_matmul = Tensor.can_matmul(self.data.shape, other.data.shape)
        if not can_matmul:
            raise ValueError(f"Shapes {self.data.shape} and {other.data.shape} not aligned for matmul")

        dtype = self.data.dtype
        out = Tensor(self.data @ other.data, (self, other), '@', dtype=dtype)

        def _backward():
            if self.requires_grad:
                if other.data.ndim > 1:
                    other_transposed = np.swapaxes(other.data, -1, -2)
                    self_grad_contrib = out.grad @ other_transposed

                else:
                    if out.grad.ndim == 0:
                        self_grad_contrib = out.grad * other.data
                    else:
                        self_grad_contrib = np.expand_dims(out.grad, -1) * other.data

                np.add(self.grad, Tensor.unbroadcast(self_grad_contrib, self.data.shape), out=self.grad)

            if other.requires_grad:
                if self.data.ndim > 1:
                    self_transposed = np.swapaxes(self.data, -1, -2)
                    other_grad_contrib = self_transposed @ out.grad

                else:
                    if out.grad.ndim == 0:
                        other_grad_contrib = out.grad * self.data
                    else:
                        other_grad_contrib = np.expand_dims(self.data, -1) * out.grad

                np.add(other.grad, Tensor.unbroadcast(other_grad_contrib, other.data.shape), out=other.grad)

        if out.requires_grad:
            out._backward = _backward

        return out

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """Sum of tensor elements along the given axis (or all axes).

        Args:
            axis: Axis or axes along which to sum. If ``None``, sums over
                all elements, producing a scalar tensor.
            keepdims: If True, the reduced axes are retained with size 1
                instead of being removed.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.ones((2, 3)), requires_grad=True)
            >>> a.sum()
            Tensor(data=6.0, shape=(), requires_grad=True, grad_fn)
            >>> a.sum(axis=0)
        """
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), 'sum', dtype=self.data.dtype)

        def _backward():
            if self.requires_grad:
                if axis is None:
                    grad_to_expand = out.grad
                else:
                    grad_to_expand = out.grad if keepdims else np.expand_dims(out.grad, axis=axis)
                np.add(self.grad, grad_to_expand, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """Arithmetic mean of tensor elements along the given axis (or all axes).

        Implemented as ``sum(axis) * (1 / n)`` so it reuses :meth:`sum`'s
        backward rule (each element simply receives ``grad / n``).

        Args:
            axis: Axis or axes along which to average. If ``None``,
                averages over all elements, producing a scalar tensor.
            keepdims: If True, the reduced axes are retained with size 1
                instead of being removed.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
            >>> a.mean()
        """
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = np.prod([self.data.shape[i] for i in axis])

        sum_out = self.sum(axis=axis, keepdims=keepdims)
        out = sum_out * (1.0 / n)
        out._op = 'mean'
        return out

    def exp(self) -> 'Tensor':
        """Element-wise exponential (``e**x``).

        Local gradient rule: ``d(e**x)/dx = e**x``, so the backward pass
        reuses the already-computed ``out.data`` rather than recomputing
        ``exp(self.data)``.

        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([0.0, 1.0]), requires_grad=True)
            >>> b = a.exp()
        """
        out_data = np.exp(self.data)
        out = Tensor(out_data, (self,), 'exp')

        def _backward():
            if self.requires_grad:
                np.add(self.grad, out.data * out.grad, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def log(self) -> 'Tensor':
        """Natural logarithm (ln).

        Returns:
            `Tensor`
        Warns:
            RuntimeWarning: if any element of the tensor is ``<= 0``, since
                ``log`` of a non-positive number is undefined (``nan``) or
                ``-inf``.
        Examples:
            >>> a = Tensor(np.array([1.0, np.e]), requires_grad=True)
            >>> b = a.log()
        """
        if not np.all(np.asarray(self.data) > 0):
            warnings.warn("Log applied to non-positive elements", RuntimeWarning, stacklevel=2)

        out = Tensor(np.log(self.data), (self,), 'ln')

        def _backward():
            if self.requires_grad:
                np.add(self.grad, (1 / (self.data + epsilon)) * out.grad, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def sqrt(self) -> 'Tensor':
        """Element-wise square root.

        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([4.0, 9.0]), requires_grad=True)
            >>> b = a.sqrt()
        """
        out = Tensor(np.sqrt(self.data), (self,), 'sqrt')

        def _backward():
            if self.requires_grad:
                # d/dx(sqrt(x)) = 1 / (2 * sqrt(x))
                np.add(self.grad, (0.5 / (np.sqrt(self.data) + epsilon)) * out.grad, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def clip(self, min_val: float, max_val: float) -> 'Tensor':
        """
        Clips the tensor values to be within [min_val, max_val].

        The backward pass zeroes the gradient for any element that was
        actually clipped (i.e. fell outside ``[min_val, max_val]``), since
        those elements' outputs are locally constant with respect to the
        input.

        Args:
            min_val: Lower bound of the clipping range (inclusive).
            max_val: Upper bound of the clipping range (inclusive).
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([-2.0, 0.5, 3.0]), requires_grad=True)
            >>> b = a.clip(0.0, 1.0)
        """
        out = Tensor(np.clip(self.data, min_val, max_val), (self,), 'clip')

        def _backward():
            if self.requires_grad:
                mask = (self.data >= min_val) & (self.data <= max_val)
                np.add(self.grad, out.grad * mask, out=self.grad)

        if out.requires_grad:
            out._backward = _backward

        return out

    def log10(self) -> 'Tensor':
        """Base-10 logarithm.

        Returns:
            `Tensor`
        Warns:
            RuntimeWarning: if any element of the tensor is ``<= 0``.
        Examples:
            >>> a = Tensor(np.array([1.0, 100.0]), requires_grad=True)
            >>> b = a.log10()
        """
        if not np.all(np.asarray(self.data) > 0):
            warnings.warn("Log10 applied to non-positive elements", RuntimeWarning, stacklevel=2)

        out = Tensor(np.log10(self.data), (self,), 'log10')

        def _backward():
            if self.requires_grad:
                np.add(self.grad, (1 / ((self.data + epsilon) * np.log(10))) * out.grad, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def relu(self) -> 'Tensor':
        """Rectified Linear Unit: ``max(0, x)``.

        Reference: https://arxiv.org/abs/1803.08375

        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([-1.0, 2.0]), requires_grad=True)
            >>> b = a.relu()
        """
        out = Tensor(np.maximum(self.data, 0), (self,), 'relu')

        def _backward():
            if self.requires_grad:
                np.add(self.grad, (self.data > 0) * out.grad, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def leaky_relu(self, alpha: float = 0.01) -> 'Tensor':
        """Leaky ReLU: ``x`` if ``x > 0``, else ``alpha * x``.

        Reference: https://arxiv.org/abs/1505.00853

        Args:
            alpha: Slope for negative inputs. Defaults to ``0.01``.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([-1.0, 2.0]), requires_grad=True)
            >>> b = a.leaky_relu(0.1)
        """
        out = Tensor(np.where(self.data > 0, self.data, alpha * self.data), (self,), 'leaky_relu')

        def _backward():
            if self.requires_grad:
                np.add(self.grad, np.where(self.data > 0, 1, alpha) * out.grad, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def elu(self, alpha: float = 1.0) -> 'Tensor':
        """Exponential Linear Unit: ``x`` if ``x > 0``, else ``alpha * (exp(x) - 1)``.

        Reference: https://arxiv.org/abs/1511.07289

        Args:
            alpha: Scale for the negative-input branch. Defaults to ``1.0``.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([-1.0, 2.0]), requires_grad=True)
            >>> b = a.elu()
        """
        out = Tensor(np.where(self.data > 0, self.data, alpha * (np.exp(self.data) - 1)), (self,), 'elu')

        def _backward():
            if self.requires_grad:
                # d/dx(alpha * (exp(x) - 1)) = alpha * exp(x)
                np.add(self.grad, np.where(self.data > 0, 1, alpha * np.exp(self.data)) * out.grad, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def selu(self, alpha: float = 1.67326, scale: float = 1.0507) -> 'Tensor':
        """Scaled Exponential Linear Unit.

        Reference: https://arxiv.org/abs/1706.02515

        Args:
            alpha: Scale for the negative-input branch (default is the
                value derived in the SELU paper for self-normalizing nets).
            scale: Overall output scale (default from the SELU paper).
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([-1.0, 2.0]), requires_grad=True)
            >>> b = a.selu()
        """
        out = Tensor(scale * np.where(self.data > 0, self.data, alpha * (np.exp(self.data) - 1)), (self,), 'selu')

        def _backward():
            if self.requires_grad:
                np.add(self.grad, scale * np.where(self.data > 0, 1, alpha * np.exp(self.data)) * out.grad, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def gelu(self) -> 'Tensor':
        """Gaussian Error Linear Unit.

        Reference: https://arxiv.org/abs/1606.08415

        Computed via ``0.5 * x * (1 + erf(x / sqrt(2)))``, using
        ``scipy.special.erf`` for the error function.

        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([-1.0, 2.0]), requires_grad=True)
            >>> b = a.gelu()
        """
        # scipy.special.erf works with numpy arrays, so convert if needed
        data_np = np.asarray(self.data)
        erf_result = sp.erf(data_np / np.sqrt(2))
        input_dtype = self.data.dtype
        out_data = np.asarray(
            0.5 * self.data * (1 + erf_result), dtype=input_dtype
        )
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
                np.add(self.grad, (cdf + self.data * pdf) * out.grad, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def sigmoid(self) -> 'Tensor':
        """Sigmoid activation: ``1 / (1 + exp(-x))``.

        Uses a numerically stable formulation that avoids overflow for
        large negative values.

        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([-100.0, 0.0, 100.0]), requires_grad=True)
            >>> b = a.sigmoid()
        """
        # Numerically stable sigmoid
        sig = np.where(self.data >= 0,
                        1 / (1 + np.exp(-self.data)),
                        np.exp(self.data) / (1 + np.exp(self.data)))
        out = Tensor(sig, (self,), 'sigmoid')

        def _backward():
            if self.requires_grad:
                np.add(self.grad, sig * (1 - sig) * out.grad, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def swish(self) -> 'Tensor':
        """Swish activation: ``x * sigmoid(x)``.

        Reference: https://arxiv.org/abs/1710.05941

        Implemented in terms of :meth:`sigmoid` and ``__mul__`` so it
        picks up their backward rules automatically rather than defining
        a new gradient formula.

        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([-1.0, 2.0]), requires_grad=True)
            >>> b = a.swish()
        """
        # swish(x) = x * sigmoid(x)
        # We can re-use our stable sigmoid
        sig = self.sigmoid()
        out = self * sig  # This builds the graph!
        out._op = 'swish'
        return out

    def tanh(self) -> 'Tensor':
        """Hyperbolic tangent activation.

        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=True)
            >>> b = a.tanh()
        """
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            if self.requires_grad:
                np.add(self.grad, (1 - t ** 2) * out.grad, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> 'Tensor':
        """Softmax: ``exp(x_i) / sum(exp(x_j))`` along *axis*.

        Uses the log-sum-exp trick for numerical stability.

        Args:
            axis: Axis along which to normalize. Defaults to the last axis.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
            >>> b = a.softmax()
        """
        # Log-sum-exp trick for numerical stability
        max_val = self.data.max(axis=axis, keepdims=True)
        e_x = np.exp(self.data - max_val)  # Subtract max for stability
        sum_e_x = e_x.sum(axis=axis, keepdims=True)
        sm = e_x / (sum_e_x + 1e-8)  # Add epsilon for safety

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

                np.add(self.grad, grad_contrib, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> 'Tensor':
        """Log-softmax: numerically stable ``log(softmax(x))`` along *axis*.

        Preferred over calling ``.softmax().log()`` separately, since it
        avoids the intermediate ``exp`` -> ``log`` round trip that can lose
        precision (and is the standard building block for a stable
        cross-entropy loss).

        Args:
            axis: Axis along which to normalize. Defaults to the last axis.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
            >>> b = a.log_softmax()
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
                sm = np.exp(out.data)  # = softmax(x)
                grad_contrib = g - sm * g.sum(axis=axis, keepdims=True)
                np.add(self.grad, grad_contrib, out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def reshape(self, *new_shape: int) -> 'Tensor':
        """Return a tensor with the same data but a different shape.

        A single ``-1`` entry in ``new_shape`` is inferred from the
        tensor's total number of elements, mirroring ``numpy.reshape``.

        Note: this creates a new node in the computation graph.

        Args:
            new_shape: The target shape, given as separate positional
                dimensions (e.g. ``a.reshape(2, 3)``). May contain at most
                one ``-1``.
        Returns:
            `Tensor`
        Raises:
            AssertionError: if the requested shape's element count doesn't
                match this tensor's ``size``.
        Examples:
            >>> a = Tensor(np.arange(6.0), requires_grad=True)
            >>> b = a.reshape(2, 3)
            >>> c = a.reshape(-1, 2)
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
                np.add(self.grad, out.grad.reshape(self.data.shape), out=self.grad)

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

        Returns:
            `Tensor`: A new, detached tensor (``requires_grad=False``) with
                boolean data.
        Examples:
            >>> a = Tensor(np.array([0.0, 1.0, -2.0]))
            >>> a.bool()
            Tensor(data=[False  True  True], shape=(3,), requires_grad=False)
        """
        bool_data = self.data.astype(bool)

        out = Tensor(bool_data, requires_grad=False)
        return out

    def __bool__(self) -> bool:
        """
        Defines the behavior of the Tensor in a boolean context (e.g., `if tensor:`).

        Raises an error for multi-element tensors because their truth
        value is ambiguous.

        Returns:
            bool: The truth value of a single-element tensor.
        Raises:
            ValueError: if the tensor has more than one element.
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

        Args:
            mask: A `Tensor` of booleans (broadcastable to ``self.shape``)
                indicating which positions to overwrite.
            fill_value: The constant value to write wherever ``mask`` is True.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
            >>> mask = Tensor(np.array([True, False, True]))
            >>> b = a.masked_fill(mask, 0.0)
        """
        out_data = np.where(mask.data, fill_value, self.data)
        out = Tensor(out_data, (self,), 'masked_fill')

        def _backward():
            # Positions that were overwritten with fill_value are locally
            # constant with respect to self, so they receive zero gradient;
            # everywhere else the gradient passes straight through.
            if self.requires_grad:
                grad_for_self = np.where(mask.data, 0.0, out.grad)

                # Add the gradient to the parent.
                np.add(self.grad, grad_for_self, out=self.grad)

        if out.requires_grad:
            out._backward = _backward

        return out

    def view(self, *new_shape: int) -> 'Tensor':
        """
        Reshape the tensor using a view-like API.

        This is a thin wrapper around :meth:`reshape` that accepts a
        variable number of dimensions or a single tuple, e.g.::

            x.view(2, 3)
            x.view((2, 3))

        Args:
            new_shape: The target shape, either as separate positional
                ints or as a single tuple/list.
        Returns:
            `Tensor`
        """
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        return self.reshape(*new_shape)

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """Permute the dimensions of the tensor.

        Args:
            axes: Order of axes for the transposition.  If ``None``,
                reverses the order of all dimensions.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.random.randn(2, 3), requires_grad=True)
            >>> b = a.transpose()          # shape (3, 2)
            >>> c = a.transpose((1, 0))    # same as above, explicit
        """
        out = Tensor(np.transpose(self.data, axes=axes), (self,), 'transpose')

        def _backward():
            if self.requires_grad:
                if axes is None:
                    inverse_axes = None
                else:
                    inverse_axes = tuple(np.argsort(axes))
                np.add(self.grad, np.transpose(out.grad, axes=inverse_axes), out=self.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def var(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = True) -> 'Tensor':
        """Sample variance of tensor elements (unbiased, denominator N-1).

        Args:
            axis: Axis or axes along which to compute the variance. If ``None``,
                variance is computed over all elements.
            keepdims: Whether to keep the reduced dimensions.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([1.0, 2.0, 3.0, 4.0]), requires_grad=True)
            >>> a.var()
        """
        # Compute mean along the given axis.
        mean = self.mean(axis=axis, keepdims=True)  # mu
        diff = self - mean  # (x_i - mu)
        sq_diff = diff ** 2  # (x_i - mu)**2

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
        var = sq_diff.sum(axis=axis, keepdims=keepdims) / denom  # sum(x_i - mu)**2 / (N - 1)
        return var

    def std(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = True) -> 'Tensor':
        """Sample standard deviation of tensor elements (sqrt of sample variance).

        Args:
            axis: Axis or axes along which to compute the standard deviation.
                If ``None``, computed over all elements.
            keepdims: Whether to keep the reduced dimensions.
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.array([1.0, 2.0, 3.0, 4.0]), requires_grad=True)
            >>> a.std()
        """
        variance = self.var(axis=axis, keepdims=keepdims)
        std = variance.sqrt()
        return std

    def item(self) -> float:
        """Returns the value of this tensor as a standard Python float.
        Only works for single-element tensors.

        Returns:
            float
        Raises:
            ValueError: if the tensor has more than one element.
        Examples:
            >>> a = Tensor(3.5)
            >>> a.item()
            3.5
        """
        if self.data.size != 1:
            raise ValueError("item() can only be called on tensors with one element.")
        return float(self.data.flatten()[0])

    def __getitem__(self, slices: Union[int, slice, Tuple]) -> 'Tensor':
        """Index or slice the tensor (``self[slices]``), preserving the graph.

        On the backward pass, the incoming gradient is scattered back into
        a zero-filled buffer at the same positions that were selected on
        the forward pass, so unselected elements correctly receive no
        gradient. ``np.add.at`` is used (rather than plain assignment) so
        that indices which are repeated (e.g. via fancy/advanced indexing)
        accumulate correctly rather than overwriting one another.

        Args:
            slices: Any valid NumPy index/slice expression (an int, a
                ``slice``, a tuple of these, a boolean mask, etc.).
        Returns:
            `Tensor`
        Examples:
            >>> a = Tensor(np.arange(5.0), requires_grad=True)
            >>> b = a[1:3]
            >>> c = a[0]
        """
        out = Tensor(self.data[slices], (self,), 'slice')

        def _backward():
            if self.requires_grad:
                # Create a grad array of zeros and "scatter" out.grad
                # into the locations specified by the slice
                grad_slice = np.zeros_like(self.data)
                np.add.at(grad_slice, slices, out.grad)
                np.add(self.grad, grad_slice, out=self.grad)
        if out.requires_grad:
            out._backward = _backward
        return out

    @staticmethod
    def concatenate(tensors: list, axis: int = 0) -> 'Tensor':
        """Concatenate tensors along an axis while preserving gradients.

        On the backward pass, the incoming gradient is split back into
        per-input slices along ``axis`` (using cumulative offsets of each
        input's size along that axis) and routed to each original tensor.

        Args:
            tensors: A non-empty list of `Tensor` instances, all sharing
                the same dtype, to concatenate.
            axis: The axis along which to concatenate. Defaults to ``0``.
        Returns:
            `Tensor`
        Raises:
            ValueError: if `tensors` is empty, or if the tensors do not all
                share the same dtype.
        Examples:
            >>> a = Tensor(np.ones((2, 3)), requires_grad=True)
            >>> b = Tensor(np.zeros((2, 3)), requires_grad=True)
            >>> c = Tensor.concatenate([a, b], axis=0)  # shape (4, 3)
        """
        if not tensors:
            raise ValueError("concatenate requires at least one tensor")
        dtype = tensors[0].dtype
        if any(t.dtype != dtype for t in tensors[1:]):
            raise ValueError("All tensors must have the same dtype")

        out = Tensor(np.concatenate([t.data for t in tensors], axis=axis),
                      tuple(tensors), 'concatenate')
        offsets = np.cumsum([0] + [t.shape[axis] for t in tensors])

        def _backward():
            for index, tensor in enumerate(tensors):
                if tensor.requires_grad:
                    slices = [slice(None)] * out.data.ndim
                    slices[axis] = slice(offsets[index], offsets[index + 1])
                    tensor.grad += out.grad[tuple(slices)]

        if out.requires_grad:
            out._backward = _backward
        return out

    # --- Backward Pass ---
    def backward(self) -> None:
        """
        Performs backpropagation starting from this tensor.

        Assumes this tensor is the final output (e.g., a scalar loss),
        though it will also work for any non-scalar tensor — in that case
        every element implicitly receives an incoming gradient of ``1``,
        equivalent to first calling ``.sum()``.

        The algorithm:
            1. Build a topological ordering of every ``requires_grad``
               tensor reachable from ``self`` via ``_prev``, via a
               depth-first search (:func:`build_topo`).
            2. Seed this tensor's own gradient with ones (the base case of
               the chain rule, ``dL/dL = 1``).
            3. Zero out the gradient of every *non-leaf* node in the graph
               (any tensor produced by an operation, i.e. one that has
               parents) so that gradients from a previous ``backward()``
               call don't silently accumulate into this one. Leaf tensors
               (parameters, inputs) are left untouched, since callers are
               expected to manage their accumulation themselves (see
               :meth:`zero_grad`), which is what enables patterns like
               gradient accumulation across multiple mini-batches.
            4. Walk the topological order in reverse, calling each node's
               ``_backward`` closure to push its gradient onto its parents.

        Returns:
            None
        Raises:
            RuntimeError: if this tensor does not have ``requires_grad=True``.
        Examples:
            >>> a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
            >>> b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
            >>> loss = (a * b).sum()
            >>> loss.backward()
            >>> a.grad
            array([3., 4.], dtype=float32)
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
        if self._prev:
            self.grad = np.ones_like(self.grad)
        else:
            np.add(self.grad, np.ones_like(self.grad), out=self.grad)

        for node in topo:
            if node is not self and node._prev and node.grad is not None:
                node.grad.fill(0.0)
            elif node.grad is None and node.requires_grad:
                node.grad = np.zeros_like(node.data)

        # --- Propagate Gradients ---
        for node in reversed(topo):
            node._backward()
