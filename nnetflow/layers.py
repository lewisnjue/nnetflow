import numpy as np 
from nnetflow.engine import Tensor 
from typing import Union, List, Tuple, Optional, Dict, Any
from nnetflow.module import Module
import numpy.typing as npt

class Linear(Module):
    """Fully-connected (dense) layer: ``output = input @ weight + bias``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> None:
        """Create a Linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features (neurons).
            bias: If ``True``, a learnable bias vector is added.
            dtype: Data type for parameters (e.g. ``np.float32``). Defaults
                to ``np.float32`` so the layer will naturally reject input
                tensors with a different dtype unless you pass an explicit
                dtype to match the data.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype if dtype is not None else np.float32

        self.weight = Tensor(
            np.random.randn(in_features, out_features),
            requires_grad=True,
            dtype=self.dtype,
        )
        self.has_bias = bool(bias)
        if self.has_bias:
            _bias = np.zeros((1, out_features))
            self.bias = Tensor(_bias, requires_grad=True, dtype=self.dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``x @ weight + bias``.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        assert x.shape[-1] == self.in_features, (
            f"Input feature size mismatch, expected {self.in_features}, got {x.shape[-1]}"
        )
        if x.dtype != self.weight.dtype:
            raise ValueError(
                f"Tensor dtype mismatch: input dtype {x.dtype} != layer dtype {self.weight.dtype}"
            )
        if self.has_bias:
            return x @ self.weight + self.bias
        else:
            return x @ self.weight

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.has_bias})"
        )

    def __str__(self) -> str:
        return self.__repr__()

class Conv2d(Module):
    """2D convolution layer.

    Input format: ``(batch_size, in_channels, height, width)``.
    Weights are initialized with He normal initialization by default.

    Reference: https://arxiv.org/abs/1511.08458
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True, dtype: Optional[npt.DTypeLike] = None) -> None:
        """Create a Conv2d layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (filters).
            kernel_size: Size of the square kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides.
            bias: If ``True``, adds a learnable bias term.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        self.dtype = dtype if dtype is not None else np.float32

        _weight = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size)

        self.weight = Tensor(_weight, requires_grad=True, dtype=self.dtype)
        if self.has_bias:
            _bias = np.zeros((1, out_channels))
            self.bias = Tensor(_bias, requires_grad=True, dtype=self.dtype)
        else:
            self.bias = None

    def _get_patches_strided(self, x_data: np.ndarray, K: int, S: int) -> np.ndarray:
        """
        Helper function to create a strided view of input data (no-copy).
        This will be used for both forward pass (on x.data)
        and backward pass (on grad_x_padded).
        """
        B, C_in, H_in_pad, W_in_pad = x_data.shape
        H_out = (H_in_pad - K) // S + 1
        W_out = (W_in_pad - K) // S + 1

        B_stride, C_stride, H_stride, W_stride = x_data.strides

        return np.lib.stride_tricks.as_strided(
            x_data,
            shape=(B, C_in, H_out, W_out, K, K),
            strides=(B_stride, C_stride, H_stride * S, W_stride * S, H_stride, W_stride)
        )

    @staticmethod
    def _col2im_accumulate(grad_patches: np.ndarray, x_padded_shape: Tuple[int, int, int, int], K: int, S: int) -> np.ndarray:
        """
        Scatter-add grad_patches back into a zero array of x_padded_shape.

        grad_patches: (B, C_in, H_out, W_out, K, K)
        Vectorized over B, C_in, H_out, W_out — only loops over the K x K
        kernel offsets, so cost is O(K^2) Python-level iterations instead
        of O(B * C_in * H_out * W_out).
        """
        H_out, W_out = grad_patches.shape[2], grad_patches.shape[3]
        grad_x_padded = np.zeros(x_padded_shape, dtype=grad_patches.dtype)

        row0 = np.arange(H_out) * S  # top-left row of each output window
        col0 = np.arange(W_out) * S  # top-left col of each output window

        for kh in range(K):
            for kw in range(K):
                rows = row0 + kh  # (H_out,)
                cols = col0 + kw  # (W_out,)
                # For a fixed (kh, kw), (rows, cols) are all distinct
                # positions across H_out x W_out, so this += is safe
                # (no duplicate-index accumulation issue).
                grad_x_padded[:, :, rows[:, None], cols[None, :]] += grad_patches[:, :, :, :, kh, kw]

        return grad_x_padded

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass and builds the computation graph.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor.
        """
        assert len(x.shape) == 4, f"Input tensor must be 4D, got {len(x.shape)}D"
        assert x.shape[1] == self.in_channels, f"Input channel size mismatch, expected {self.in_channels}, got {x.shape[1]}"

        # Get dimensions and parameters
        B, C_in, H_in, W_in = x.shape
        K, S, P = self.kernel_size, self.stride, self.padding

        # Calculate output dimensions
        H_out = (H_in - K + 2 * P) // S + 1
        W_out = (W_in - K + 2 * P) // S + 1

        # --- 1. Forward Pass (Numpy/CuPy land) ---
        x_padded_data = np.pad(
            x.data, ((0, 0), (0, 0), (P, P), (P, P)), 'constant')
        patches = self._get_patches_strided(x_padded_data, K, S)
        output_data = np.einsum(
            'bchwkl, ockl -> bohw', patches, self.weight.data)

        if self.has_bias:
            output_data = output_data + self.bias.data.reshape(1, self.out_channels, 1, 1)

        # --- 2. Create Output Tensor (Autograd land) ---
        children = [x, self.weight]
        if self.has_bias:
            children.append(self.bias)

        out = Tensor(output_data, _children=tuple(children), _op='Conv2d')

        # --- 3. Define Backward Pass ---

        if out.requires_grad:
            def _backward():
                grad_output = out.grad  # Shape (B, O, H_out, W_out)

                # --- 3a. Calculate dL/db ---
                if self.has_bias and self.bias.requires_grad:
                    grad_bias = grad_output.sum(axis=(0, 2, 3))
                    self.bias.grad += grad_bias.reshape(self.bias.data.shape)

                # --- 3b. Calculate dL/dw ---
                if self.weight.requires_grad:
                    # 'patches' is from the forward pass
                    grad_weight = np.einsum('bohw, bchwkl -> ockl', grad_output, patches)
                    self.weight.grad += grad_weight

                # --- 3c. Calculate dL/dx (vectorized col2im-style scatter-add) ---
                if x.requires_grad:
                    grad_patches = np.einsum('bohw, ockl -> bchwkl', grad_output, self.weight.data)

                    grad_x_padded = self._col2im_accumulate(
                        grad_patches, x_padded_data.shape, K, S
                    )

                    # Un-pad the gradient to get dL/dx
                    if P > 0:
                        grad_x = grad_x_padded[:, :, P:-P, P:-P]
                    else:
                        grad_x = grad_x_padded

                    assert grad_x.shape == x.data.shape
                    x.grad += grad_x

            out._backward = _backward

        return out

    def __repr__(self) -> str:
        return (f"Conv2d(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"stride={self.stride}, "
                f"padding={self.padding}, "
                f"bias={self.has_bias})")

    def __str__(self):
        return self.__repr__()


class Conv1d(Module):
    """1D convolution layer.

    Input format: ``(batch_size, in_channels, length)``.
    Weights are initialized with He normal initialization by default.

    Reference: https://arxiv.org/abs/1511.08458
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True, dtype: Optional[npt.DTypeLike] = None) -> None:
        """Create a Conv1d layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (filters).
            kernel_size: Size of the kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides.
            bias: If ``True``, adds a learnable bias term.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        self.dtype = dtype if dtype is not None else np.float32

        _weight = np.random.randn(
            out_channels, in_channels, kernel_size)
        self.weight = Tensor(_weight, requires_grad=True, dtype=self.dtype)

        if self.has_bias:
            _bias = np.zeros((1, out_channels))
            self.bias = Tensor(_bias, requires_grad=True, dtype=self.dtype)
        else:
            self.bias = None

    def _get_patches_strided(self, x_data: np.ndarray, K: int, S: int) -> np.ndarray:
        """
        Helper function to create a strided view of input data (no-copy).
        'x_data' is assumed to be the *padded* input.
        """
        B, C_in, L_in_pad = x_data.shape
        L_out = (L_in_pad - K) // S + 1

        B_stride, C_stride, L_stride = x_data.strides

        return np.lib.stride_tricks.as_strided(
            x_data,
            shape=(B, C_in, L_out, K),
            strides=(B_stride, C_stride, L_stride * S, L_stride)
        )

    @staticmethod
    def _col2im_accumulate(grad_patches: np.ndarray, x_padded_shape: Tuple[int, int, int], K: int, S: int) -> np.ndarray:
        """
        Scatter-add grad_patches back into a zero array of x_padded_shape.

        grad_patches: (B, C_in, L_out, K)
        Vectorized over B, C_in, L_out — only loops over the K kernel
        offsets, so cost is O(K) Python-level iterations instead of
        O(B * C_in * L_out).
        """
        L_out = grad_patches.shape[2]
        grad_x_padded = np.zeros(x_padded_shape, dtype=grad_patches.dtype)

        pos0 = np.arange(L_out) * S  # start position of each output window

        for k in range(K):
            idx = pos0 + k  # (L_out,) — distinct positions for a fixed k
            grad_x_padded[:, :, idx] += grad_patches[:, :, :, k]

        return grad_x_padded

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass and builds the computation graph.

        Args:
            x: Input tensor of shape (batch_size, in_channels, length).
        Returns:
            Output tensor.
        """
        assert len(x.shape) == 3, f"Input tensor must be 3D, got {len(x.shape)}D"
        assert x.shape[1] == self.in_channels, f"Input channel size mismatch, expected {self.in_channels}, got {x.shape[1]}"

        # Get dimensions and parameters
        B, C_in, L_in = x.shape
        K, S, P = self.kernel_size, self.stride, self.padding

        # Calculate output dimensions
        L_out = (L_in - K + 2 * P) // S + 1

        # --- 1. Forward Pass (Numpy/CuPy land) ---
        x_padded_data = np.pad(
            x.data, ((0, 0), (0, 0), (P, P)), 'constant')
        patches = self._get_patches_strided(x_padded_data, K, S)
        output_data = np.einsum(
            'bclk, ock -> bol', patches, self.weight.data)

        if self.has_bias:
            output_data = output_data + self.bias.data.reshape(1, self.out_channels, 1)

        # --- 2. Create Output Tensor (Autograd land) ---
        children = [x, self.weight]
        if self.has_bias:
            children.append(self.bias)

        out = Tensor(output_data, _children=tuple(children), _op='Conv1d')

        # --- 3. Define Backward Pass ---

        if out.requires_grad:
            def _backward():
                grad_output = out.grad  # Shape (B, O, L_out)

                # --- 3a. Calculate dL/db ---
                if self.has_bias and self.bias.requires_grad:
                    grad_bias = grad_output.sum(axis=(0, 2))  # Shape (O,)
                    self.bias.grad += grad_bias.reshape(self.bias.data.shape)

                # --- 3b. Calculate dL/dw ---
                if self.weight.requires_grad:
                    # 'patches' is from the forward pass
                    grad_weight = np.einsum('bol, bclk -> ock', grad_output, patches)
                    self.weight.grad += grad_weight

                # --- 3c. Calculate dL/dx (vectorized col2im-style scatter-add) ---
                if x.requires_grad:
                    grad_patches = np.einsum('bol, ock -> bclk', grad_output, self.weight.data)

                    grad_x_padded = self._col2im_accumulate(
                        grad_patches, x_padded_data.shape, K, S
                    )

                    # Un-pad the gradient to get dL/dx
                    if P > 0:
                        grad_x = grad_x_padded[:, :, P:-P]
                    else:
                        grad_x = grad_x_padded

                    assert grad_x.shape == x.data.shape
                    x.grad += grad_x

            out._backward = _backward

        return out

    def __repr__(self) -> str:
        return (f"Conv1d(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"stride={self.stride}, "
                f"padding={self.padding}, "
                f"bias={self.has_bias})")

    def __str__(self):
        return self.__repr__()

class BatchNorm2d(Module):
    """Batch Normalization over a batch of 4-D inputs.

    Typically used for spatial inputs (e.g., images) of shape (N, C, H, W).
    Normalizes over the batch, height, and width dimensions.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True) -> None:
        """Create a BatchNorm2d layer.

        Args:
            num_features: Number of channels (C) to normalize.
            eps: Small constant added to the denominator for numerical stability.
            momentum: Momentum factor for exponential moving average of running stats.
            affine: If ``True``, learnable scale (gamma) and shift (beta) are added.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        # Shape (1, C, 1, 1) allows automatic broadcasting over (N, C, H, W)
        shape = (1, num_features, 1, 1)
        
        if affine:
            self.gamma = Tensor(np.ones(shape), requires_grad=True)
            self.beta = Tensor(np.zeros(shape), requires_grad=True)
        else:
            self.gamma = Tensor(np.ones(shape), requires_grad=False)
            self.beta = Tensor(np.zeros(shape), requires_grad=False)
            
        self.running_mean = Tensor(np.zeros(shape), requires_grad=False)
        self.running_var = Tensor(np.ones(shape), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the 4D input tensor."""
        assert len(x.shape) == 4, f"Input tensor must be 4D (N, C, H, W), got {x.shape}"
        
        # Match parameter dtype to input dtype
        target_dtype = x.data.dtype
        if self.gamma.data.dtype != target_dtype:
            self.gamma.data = self.gamma.data.astype(target_dtype)
            self.beta.data = self.beta.data.astype(target_dtype)
            self.running_mean.data = self.running_mean.data.astype(target_dtype)
            self.running_var.data = self.running_var.data.astype(target_dtype)
            
        if self.training:
            # Calculate mean and var across N, H, and W (axes 0, 2, and 3)
            # Resulting shape will automatically be (1, C, 1, 1) if keepdims=True
            batch_mean = x.mean(axis=(0, 2, 3), keepdims=True)
            centered = x - batch_mean
            batch_var = (centered ** 2).mean(axis=(0, 2, 3), keepdims=True)
            
            x_normalized = centered / (batch_var + self.eps).sqrt()
            
            # Update running stats
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + \
                                   self.momentum * batch_mean.data
            
            # Unbiased variance update (Bessel's correction)
            # m = N * H * W
            m = x.shape[0] * x.shape[2] * x.shape[3]
            unbiased_var_data = batch_var.data * (m / (m - 1)) if m > 1 else batch_var.data
            
            self.running_var.data = (1 - self.momentum) * self.running_var.data + \
                                  self.momentum * unbiased_var_data
        else:
            # Eval mode: use running statistics
            x_normalized = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        
        # Affine transformation
        out = self.gamma * x_normalized + self.beta
        return out
    
    def __repr__(self) -> str:
        return f"BatchNorm2d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum})"
    
    def __str__(self) -> str:
        return self.__repr__()



class BatchNorm1d(Module):
    """Batch Normalization over a batch of 2-D or 3-D inputs.
    
    Matches PyTorch exactly:
    - Input shape: (N, C) or (N, C, L) where C = num_features
    - Normalizes across the batch (and sequence L, if 3D) dimensions.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        # Initialize flat arrays, we will reshape them dynamically in forward 
        # depending on whether the input is 2D or 3D
        if affine:
            self.gamma = Tensor(np.ones((num_features,)), requires_grad=True)
            self.beta = Tensor(np.zeros((num_features,)), requires_grad=True)
        else:
            self.gamma = Tensor(np.ones((num_features,)), requires_grad=False)
            self.beta = Tensor(np.zeros((num_features,)), requires_grad=False)
            
        self.running_mean = Tensor(np.zeros((num_features,)), requires_grad=False)
        self.running_var = Tensor(np.ones((num_features,)), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        dim = len(x.shape)
        assert dim in (2, 3), f"BatchNorm1d requires 2D or 3D input, got {dim}D"
        assert x.shape[1] == self.num_features, f"Expected {self.num_features} channels, got {x.shape[1]}"
        
        # Match dtypes to prevent upcasting
        target_dtype = x.data.dtype
        if self.gamma.data.dtype != target_dtype:
            self.gamma.data = self.gamma.data.astype(target_dtype)
            self.beta.data = self.beta.data.astype(target_dtype)
            self.running_mean.data = self.running_mean.data.astype(target_dtype)
            self.running_var.data = self.running_var.data.astype(target_dtype)

        # Set up broadcast shape for parameters: (1, C) for 2D, (1, C, 1) for 3D
        view_shape = (1, self.num_features) if dim == 2 else (1, self.num_features, 1)
        reduce_axes = (0,) if dim == 2 else (0, 2)
        
        # Reshape parameters for broadcasting
        gamma_view = self.gamma.reshape(*view_shape)
        beta_view = self.beta.reshape(*view_shape)
        running_mean_view = self.running_mean.reshape(*view_shape)
        running_var_view = self.running_var.reshape(*view_shape)
        
        if self.training:
            # 1. Batch Mean and Variance
            batch_mean = x.mean(axis=reduce_axes, keepdims=True)
            centered = x - batch_mean
            
            # PyTorch uses biased variance for normalizing the forward pass
            batch_var = (centered ** 2).mean(axis=reduce_axes, keepdims=True)
            
            x_normalized = centered / (batch_var + self.eps).sqrt()
            
            # 2. Update Running Statistics
            # Calculate number of elements we are normalizing over (m)
            m = x.shape[0] if dim == 2 else x.shape[0] * x.shape[2]
            
            # Use unbiased variance for the running stats (Bessel's correction)
            unbiased_var = batch_var.data * (m / (m - 1)) if m > 1 else batch_var.data
            
            # Update EMA (using squeezed arrays to update the 1D buffers)
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + \
                                     self.momentum * batch_mean.data.reshape(self.num_features)
                                     
            self.running_var.data = (1 - self.momentum) * self.running_var.data + \
                                    self.momentum * unbiased_var.reshape(self.num_features)
        else:
            # Evaluation mode: use running statistics
            x_normalized = (x - running_mean_view) / (running_var_view + self.eps).sqrt()
            
        out = gamma_view * x_normalized + beta_view
        return out
        
    def __repr__(self) -> str:
        return f"BatchNorm1d({self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine})"



class LayerNorm(Module):
    """Layer Normalization over the last dimension.

    Unlike BatchNorm, this normalizes each sample independently, making
    it suitable for variable-length sequences (e.g. in Transformers).

    Reference: https://arxiv.org/abs/1607.06450
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Create a LayerNorm layer.

        Args:
            dim: Size of the last dimension of input tensors.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.gamma = Tensor(np.ones((1, dim)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, dim)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization over the last dimension.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Normalized tensor of the same shape.
        """
        # Match parameter dtype to input dtype to avoid upcasting
        target_dtype = x.data.dtype
        if self.gamma.data.dtype != target_dtype:
            self.gamma.data = self.gamma.data.astype(target_dtype)
            self.beta.data = self.beta.data.astype(target_dtype)

        mean = x.mean(axis=-1, keepdims=True)  # Shape: (..., 1)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)  # Shape: (..., 1)
        
        x_normalized = (x - mean) / (var + self.eps).sqrt()
        
        out = self.gamma * x_normalized + self.beta
        return out 
    
class Embedding(Module):
    """A lookup table that maps integer indices to dense vectors.s
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: Optional[npt.DTypeLike] = None) -> None:
        """Create an Embedding layer.

        Args:
            num_embeddings: Size of the vocabulary (number of rows).
            embedding_dim: Dimension of each embedding vector.
            dtype: Data type for the embedding weight matrix.
        """
        super().__init__()
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim  
        weight = np.random.randn(num_embeddings, embedding_dim)
        self.dtype = dtype if dtype is not None else np.float32
        self.weight = Tensor(weight, requires_grad=True,dtype=self.dtype)

    def forward(self, indices: Union[int, slice, tuple]) -> Tensor:
        """Look up embeddings for the given indices.

        Args:
            indices: Integer indices, slices, or a tuple of indices into
                the embedding table.

        Returns:
            Tensor containing the selected embedding vectors.
        """
        return self.weight[indices]


class Dropout(Module):
    """Inverted Dropout layer.

    During training, randomly zeroes elements of the input with probability
    ``p`` and scales the remaining values by ``1 / (1 - p)`` so that the
    expected sum is unchanged.  During evaluation, this layer is a no-op.
    
    You need to make sure you dont apply dropout to the output Layer of the Model 

    """
    def __init__(self, p: float = 0.5) -> None:
        """
        Args:
            p: probability of dropping a neuron (setting it to zero).
        """
        super().__init__() 
        assert 0.0 <= p < 1.0, "Dropout probability must be in [0.0, 1.0) range"
        self.p = p

    def forward(self, x: Tensor) -> Tensor: # self.Training is set in the Module init 
        if not self.training:
            return x
        mask = (np.random.rand(*x.data.shape) > self.p).astype(x.data.dtype)
        mask_tensor = Tensor(mask, requires_grad=False, dtype=x.data.dtype)
        scale = 1.0 / (1.0 - self.p)
        return (x * mask_tensor) * scale


class MCDropout(Dropout): 
    """ 
    in MCDroptout training is always true during training and inference 
    """ 
    def __init__(self,p=0.5):
        super().__init__(p)  
    
    def forward(self,x:Tensor)  -> Tensor:
        was_training = self.training
        self.training = True
        try:
            return super().forward(x)
        finally:
            self.training = was_training

class Flatten(Module):
    """Flatten all dimensions except the batch dimension.

    Reshapes input from ``(batch_size, ...)`` to ``(batch_size, -1)``.
    """

    def __init__(self) -> None:
        pass
    
    def forward(self,x:Tensor) -> Tensor:
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
    
    def __repr__(self) -> str:
        return "Flatten()"
    
    def __str__(self) -> str:
        return self.__repr__()
    



def _to_pair(x: Union[int, Tuple[int, ...]]) -> Tuple[int, int]:
    """Converts an int or a 2-tuple into a 2-tuple."""
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, (tuple, list)) and len(x) == 2:
        return tuple(x)
    raise ValueError("MaxPool2d: kernel_size/stride must be an int or a 2-tuple")


class MaxPool2d(Module):
    """
    Applies a 2D max pooling over an input tensor.
    Input format: (batch_size, in_channels, height, width)
    """
    def __init__(self, 
                 kernel_size: Union[int, Tuple[int, int]], 
                 stride: Optional[Union[int, Tuple[int, int]]] = None, 
                 padding: int = 0):
        
        self.kernel_size: Tuple[int, int] = _to_pair(kernel_size)
        self.stride: Tuple[int, int] = _to_pair(stride) if stride is not None else _to_pair(kernel_size)
        self.padding: int = padding

        self.cache: Dict[str, Any] = {}

    def _get_patches_strided(self, x_data: np.ndarray, K_h: int, K_w: int, S_h: int, S_w: int) -> np.ndarray:
        """Helper to create a strided view of input data."""
        B, C, H_in_pad, W_in_pad = x_data.shape
        H_out = (H_in_pad - K_h) // S_h + 1
        W_out = (W_in_pad - K_w) // S_w + 1

        B_stride, C_stride, H_stride, W_stride = x_data.strides

        return np.lib.stride_tricks.as_strided(
            x_data,
            shape=(B, C, H_out, W_out, K_h, K_w),
            strides=(B_stride, C_stride, H_stride * S_h, W_stride * S_w, H_stride, W_stride)
        )

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4, "MaxPool2d input must be 4D (B, C, H, W)"
        
        B, C, H_in, W_in = x.shape
        K_h, K_w = self.kernel_size
        S_h, S_w = self.stride
        P = self.padding

        # --- 1. Forward Pass ---
        x_padded_data = np.pad(
            x.data, 
            ((0, 0), (0, 0), (P, P), (P, P)), 
            'constant', 
            constant_values=-np.inf
        )
        
        # Get strided windows: Shape (B, C, H_out, W_out, K_h, K_w)
        patches = self._get_patches_strided(x_padded_data, K_h, K_w, S_h, S_w)
        
        # Max along the kernel dimensions
        output_data = np.max(patches, axis=(4, 5))
        
        # Find argmax over the flattened kernel area to get 1D index within window
        patches_flat = patches.reshape(B, C, output_data.shape[2], output_data.shape[3], K_h * K_w)
        argmax_idx = np.argmax(patches_flat, axis=-1)  # Shape (B, C, H_out, W_out)

        # --- 2. Autograd Setup ---
        out = Tensor(output_data, _children=(x,), _op='MaxPool2d')
        
        self.cache['input_padded_shape'] = x_padded_data.shape
        self.cache['argmax_idx'] = argmax_idx

        # --- 3. Backward Pass ---
        if out.requires_grad:
            def _backward():
                if not x.requires_grad:
                    return

                grad_output = out.grad  # (B, C, H_out, W_out)
                argmax_idx = self.cache['argmax_idx']
                input_padded_shape = self.cache['input_padded_shape']
                
                grad_x_padded = np.zeros(input_padded_shape)
                
                # To vectorize the scatter, we use NumPy's advanced indexing / add.at
                # Generate grids for B, C, H_out, W_out
                B_dim, C_dim, H_out, W_out = grad_output.shape
                
                b_idx, c_idx, h_idx, w_idx = np.indices((B_dim, C_dim, H_out, W_out))
                
                # Convert flattened argmax inside window back to h and w offsets
                h_offset = argmax_idx // K_w
                w_offset = argmax_idx % K_w
                
                # Calculate absolute padded coordinates
                abs_h = (h_idx * S_h) + h_offset
                abs_w = (w_idx * S_w) + w_offset
                
                # Fast unbuffered scatter-add (Equivalent to your nested for-loops)
                np.add.at(
                    grad_x_padded, 
                    (b_idx, c_idx, abs_h, abs_w), 
                    grad_output
                )
                
                # Un-pad the gradient
                if P > 0:
                    grad_x = grad_x_padded[:, :, P:-P, P:-P]
                else:
                    grad_x = grad_x_padded
                
                x.grad += grad_x

            out._backward = _backward
            
        return out

    def __repr__(self) -> str:
        return (f"MaxPool2d(kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding})")

class MaxPool1d(Module):
    """
    Applies a 1D max pooling over an input tensor.
    Input format: (batch_size, in_channels, length)
    """
    def __init__(self, 
                 kernel_size: int, 
                 stride: Optional[int] = None, 
                 padding: int = 0):
        
        self.kernel_size: int = kernel_size
        self.stride: int = stride if stride is not None else kernel_size
        self.padding: int = padding

        # Cache to store information for backward pass
        self.cache: Dict[str, Any] = {}

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass and builds the computation graph.
        """
        assert len(x.shape) == 3, "MaxPool1d input must be 3D (B, C, L)"
        
        B, C, L_in = x.shape
        K, S, P = self.kernel_size, self.stride, self.padding

        # --- 1. Forward Pass (Numpy/CuPy land) ---
        
        # Pad with -infinity
        x_padded_data = np.pad(
            x.data, 
            ((0, 0), (0, 0), (P, P)), 
            'constant', 
            constant_values=-np.inf
        )
        
        padded_shape = x_padded_data.shape # (B, C, L_pad)

        # Calculate output dimensions
        L_out = (L_in - K + 2 * P) // S + 1

        # Create output arrays
        output_data = np.zeros((B, C, L_out))
        
        # 'indices' will store the (l) coordinate from the *padded*
        # input array for each max value.
        # Shape: (B, C, L_out)
        indices = np.zeros((B, C, L_out), dtype=int)

        # Loop-based forward pass
        for b in range(B):
            for c in range(C):
                for l in range(L_out):
                    # Find the window in the padded input
                    l_start = l * S
                    l_end = l_start + K
                    
                    window = x_padded_data[b, c, l_start:l_end]
                    
                    # Get the max value
                    output_data[b, c, l] = np.max(window)
                    
                    # Get the 1D index *within the window*
                    l_idx_window = np.argmax(window)
                    
                    # Convert to index in the *padded* array and store
                    indices[b, c, l] = l_start + l_idx_window

        # --- 2. Create Output Tensor (Autograd land) ---
        out = Tensor(output_data, _children=(x,), _op='MaxPool1d')
        
        # Save context for backward pass
        self.cache['input_padded_shape'] = padded_shape
        self.cache['indices'] = indices

        # --- 3. Define Backward Pass ---
        if out.requires_grad:
            def _backward():
                if not x.requires_grad:
                    return

                # Get incoming gradient
                grad_output = out.grad  # (B, C, L_out)
                
                # Get saved context
                indices = self.cache['indices'] # (B, C, L_out)
                input_padded_shape = self.cache['input_padded_shape']
                
                # Create the gradient for the padded input
                grad_x_padded = np.zeros(input_padded_shape)
                
                B, C, L_out = grad_output.shape

                # Loop and "scatter" the gradients
                for b in range(B):
                    for c in range(C):
                        for l in range(L_out):
                            # Get the (l) coordinate from the forward pass
                            l_idx = indices[b, c, l]
                            
                            # Get the gradient value
                            grad_val = grad_output[b, c, l]
                            
                            # Add it to the single max location.
                            grad_x_padded[b, c, l_idx] += grad_val
                
                # Un-pad the gradient
                if P > 0:
                    grad_x = grad_x_padded[:, :, P:-P]
                else:
                    grad_x = grad_x_padded
                
                # Accumulate gradient in the input tensor
                x.grad += grad_x

            out._backward = _backward
            
        return out


    def __repr__(self) -> str:
        return (f"MaxPool1d(kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding})")

class MultiHeadAttention(Module):
    """Multi-Head Attention (Vaswani et al., 2017).

    Splits queries, keys and values into ``num_heads`` parallel attention
    heads, computes scaled dot-product attention independently for each
    head, and concatenates the results.

    Features:
        * Causal masking for autoregressive models.
        * Dropout regularization on attention weights.
        * Optional QKV bias.
        * KV-Caching for efficient autoregressive decoding.

    Reference: https://arxiv.org/abs/1706.03762
    """
    
    def __init__(
        self, 
        d_in: int, 
        d_out: int, 
        num_heads: int, 
        dropout: float = 0.1,
        bias: bool = True,
        causal: bool = True,
        max_seq_len: Optional[int] = None,
        dtype: Optional[npt.DTypeLike] = None
    ) -> None:
        """
        Initialize Multi-Head Attention layer.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability (default: 0.1)
            bias: Whether to use bias in QKV projections (default: True)
            causal: Whether to apply causal masking (default: True)
            max_seq_len: Maximum sequence length for causal mask (if None, mask is created dynamically)
            dtype: Data type for parameters (default: None, uses device default)
        
        Raises:
            ValueError: If d_out is not divisible by num_heads
        """
        super().__init__()
        
        if d_out % num_heads != 0:
            raise ValueError(f"d_out ({d_out}) must be divisible by num_heads ({num_heads})")
        
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.dropout = dropout
        self.causal = causal
        self.max_seq_len = max_seq_len
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        self.W_query = Linear(d_in, d_out, bias=bias, dtype=dtype)
        self.W_key = Linear(d_in, d_out, bias=bias, dtype=dtype)
        self.W_value = Linear(d_in, d_out, bias=bias, dtype=dtype)
        self.out_proj = Linear(d_out, d_out, bias=True, dtype=dtype)
        self.dropout_layer = Dropout(dropout)
        
        self._causal_mask: Optional[Tensor] = None
        if causal and max_seq_len is not None:
            # Memory Optimization: Initialize directly as boolean instead of float32
            mask = np.triu(np.ones((max_seq_len, max_seq_len), dtype=bool), k=1)
            self._causal_mask = Tensor(mask, requires_grad=False)
            
        self.ptr_current_pos = 0
        self.cache_k: Optional[Tensor] = None
        self.cache_v: Optional[Tensor] = None
    
    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        """
        Forward pass of Multi-Head Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
            use_cache: Whether to use cached values (default: False)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_out)
        """
        B, T, _ = x.shape
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        Q = Q.reshape(B, T, self.num_heads, self.head_dim)
        K = K.reshape(B, T, self.num_heads, self.head_dim)
        V = V.reshape(B, T, self.num_heads, self.head_dim)

        if use_cache:
            if self.cache_k is None or self.cache_v is None:
                self.cache_k = K
                self.cache_v = V
            else:
                self.cache_k = Tensor.concatenate([self.cache_k, K], axis=1)
                self.cache_v = Tensor.concatenate([self.cache_v, V], axis=1)
            K = self.cache_k
            V = self.cache_v
        else:
            # Reset the pointer if we aren't using cache
            self.ptr_current_pos = 0

        # Calculate the total key sequence length (important for the mask shape)
        T_k = K.shape[1]
        
        Q = Q.transpose((0, 2, 1, 3))
        K = K.transpose((0, 2, 1, 3)) 
        V = V.transpose((0, 2, 1, 3))  
        
        attn_scores = (Q @ K.transpose((0, 1, 3, 2))) * self.scale

        mask = None
        if self.causal:
            if self._causal_mask is not None:
                if use_cache:
                    # Queries correspond to: ptr_current_pos -> ptr_current_pos + T
                    # Keys correspond to: 0 -> ptr_current_pos + T
                    mask = self._causal_mask[ 
                        self.ptr_current_pos : self.ptr_current_pos + T,
                        : self.ptr_current_pos + T 
                    ]
                    self.ptr_current_pos += T
                else:
                    mask = self._causal_mask[:T, :T]
                    self.ptr_current_pos = T # Advance pointer in case cache is used next pass
            else:
                # Dynamic masking if max_seq_len was not provided
                # k = T_k - T + 1 ensures the diagonal aligns correctly 
                # whether we are passing a full sequence or a single cached token
                dynamic_mask = np.triu(np.ones((T, T_k), dtype=bool), k=(T_k - T + 1))
                mask = Tensor(dynamic_mask, requires_grad=False)
        
        if mask is not None:
            # Broadcast mask from (T, T_k) to match attn_scores (B, num_heads, T, T_k)
            mask_broadcast = mask.reshape(1, 1, T, T_k)
            attn_scores = attn_scores.masked_fill(mask_broadcast, float('-inf'))
            
        attn_weights = attn_scores.softmax(axis=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        context = attn_weights @ V
        context = context.transpose((0, 2, 1, 3)).reshape(B, T, self.d_out)
        
        out = self.out_proj(context)
        return out
    
    def reset_cache(self) -> None:
        """Reset the cached keys and values for autoregressive decoding."""
        self.cache_k = None
        self.cache_v = None
        self.ptr_current_pos = 0
    
    def __repr__(self) -> str:
        return (
            f"MultiHeadAttention(d_in={self.d_in}, d_out={self.d_out}, "
            f"num_heads={self.num_heads}, dropout={self.dropout}, "
            f"causal={self.causal})"
        )
    
    def __str__(self) -> str:
        return self.__repr__()

class AveragePool2d(Module):
    """
    2D Average Pooling layer (channel-first: N,C,H,W)

    Reduces spatial dimensions by taking the average value in each pooling window.

    Supports:
    - arbitrary kernel_size
    - stride (defaults to kernel_size if not given)
    - padding (usually 0 for average pooling)
    - count_include_pad (whether to include padded zeros in the average)
    """
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        count_include_pad: bool = True,
        dtype: Optional[npt.DTypeLike] = None
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.count_include_pad = count_include_pad
        self.dtype = dtype

    def _get_patches_strided(self, x_data: np.ndarray) -> np.ndarray:
        """Create strided view of all pooling windows (zero-copy when possible)."""
        B, C, H_in_pad, W_in_pad = x_data.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride

        H_out = (H_in_pad - kh) // sh + 1
        W_out = (W_in_pad - kw) // sw + 1

        strides = x_data.strides
        shape = (B, C, H_out, W_out, kh, kw)

        new_strides = (
            strides[0],
            strides[1],
            strides[2] * sh,
            strides[3] * sw,
            strides[2],
            strides[3],
        )

        return np.lib.stride_tricks.as_strided(
            x_data,
            shape=shape,
            strides=new_strides
        )

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4, "Expected 4D input (N,C,H,W)"

        B, C, H_in, W_in = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        H_out = (H_in + 2 * ph - kh) // sh + 1
        W_out = (W_in + 2 * pw - kw) // sw + 1

        x_padded = np.pad(
            x.data,
            ((0, 0), (0, 0), (ph, ph), (pw, pw)),
            mode="constant",
            constant_values=0.0,
        )

        patches = self._get_patches_strided(x_padded)

        if self.count_include_pad:
            pooled = patches.mean(axis=(-2, -1))
        else:
            valid = np.pad(
                np.ones((H_in, W_in), dtype=x.data.dtype),
                ((ph, ph), (pw, pw)),
                mode="constant",
                constant_values=0.0,
            )
            valid_patches = self._get_patches_strided(valid[None, None, ...])
            counts = valid_patches.sum(axis=(-2, -1))
            pooled = patches.sum(axis=(-2, -1)) / counts

        out = Tensor(pooled, _children=(x,), _op="AvgPool2d")

        if out.requires_grad:

            def _backward():
                grad_output = out.grad

                if x.requires_grad:
                    kh, kw = self.kernel_size
                    area = kh * kw if self.count_include_pad else None

                    valid = np.pad(
                        np.ones((H_in, W_in), dtype=x.data.dtype),
                        ((ph, ph), (pw, pw)),
                        mode="constant",
                        constant_values=0.0,
                    )
                    valid_patches = self._get_patches_strided(valid[None, None, ...])
                    counts = valid_patches.sum(axis=(-2, -1))

                    grad_x_padded = np.zeros_like(x_padded)

                    for b in range(B):
                        for c in range(C):
                            for ho in range(H_out):
                                for wo in range(W_out):

                                    h_start = ho * sh
                                    w_start = wo * sw

                                    window = grad_x_padded[
                                        b, c,
                                        h_start:h_start + kh,
                                        w_start:w_start + kw
                                    ]

                                    divisor = area if self.count_include_pad else counts[ho, wo]
                                    window += grad_output[b, c, ho, wo] / divisor

                    if ph > 0 or pw > 0:
                        grad_x = grad_x_padded[:, :, ph:-ph if ph else None, pw:-pw if pw else None]
                    else:
                        grad_x = grad_x_padded

                    x.grad += grad_x

            out._backward = _backward

        return out

    def __repr__(self) -> str:
        return (
            f"AveragePool2d(kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"count_include_pad={self.count_include_pad})"
        )


class GlobalAveragePool2d(Module):
    """
    Global Average Pooling 2D – reduces H and W to 1×1 by averaging each channel.

    Input:  (N, C, H, W)
    Output: (N, C, 1, 1)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4, "Expected input shape (N, C, H, W)"

        out_data = x.data.mean(axis=(2, 3), keepdims=True)

        out = Tensor(out_data, _children=(x,), _op="GlobalAvgPool2d")

        if out.requires_grad:

            def _backward():
                if x.requires_grad:
                    H, W = x.shape[2], x.shape[3]
                    spatial_area = H * W

                    grad_x = np.ones_like(x.data) * (out.grad / spatial_area)

                    x.grad += grad_x

            out._backward = _backward

        return out

    def __repr__(self) -> str:
        return "GlobalAveragePool2d()"