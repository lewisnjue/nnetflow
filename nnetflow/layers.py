import numpy as np 
from nnetflow.engine import Tensor 
from typing import Union, List, Tuple, Optional, Dict, Any
from nnetflow.init import initializers
from nnetflow.module import Module
import numpy.typing as npt

class Linear(Module):
    """Fully-connected (dense) layer: ``output = input @ weight + bias``.

    Weights are initialized with He uniform initialization by default.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: Optional[npt.DTypeLike] = None) -> None:
        """Create a Linear layer.

        Args:
            in_features: Number of input features (last dimension of input).
            out_features: Number of output features (neurons).
            bias: If ``True``, a learnable bias vector is added.
            dtype: Data type for parameters (e.g. ``np.float32``).
        """
        self.in_features = in_features 
        self.out_features = out_features 
        _weight = np.random.randn(in_features, out_features) 
        _bias = np.zeros((1, out_features)) 
        self.weight = Tensor(_weight, requires_grad=True,dtype=dtype)
        initializers.He_uniform(self.weight, nonlinearity='relu') 
        self.has_bias = bias
        if bias:
            self.bias = Tensor(_bias, requires_grad=True,dtype=dtype)
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute ``x @ weight + bias``.

        Args:
            x: Input tensor of shape ``(batch_size, in_features)``.

        Returns:
            Output tensor of shape ``(batch_size, out_features)``.
        """
        assert x.shape[-1] == self.in_features, f"Input feature size mismatch, expected {self.in_features}, got {x.shape[-1]}"
        if self.has_bias:
             return x @ self.weight + self.bias 
        else:
            return x @ self.weight 

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        self.dtype = dtype

        _weight = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size) 
        
        self.weight = Tensor(_weight, requires_grad=True,dtype=dtype)
        initializers.He_normal(self.weight, nonlinearity='relu')

        if self.has_bias:

            _bias = np.zeros((1, out_channels))
            self.bias = Tensor(_bias, requires_grad=True,dtype=dtype)
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

                # --- 3c. Calculate dL/dx ---
                if x.requires_grad:
                    # Calculate dL/d(patches)
                    # 'bohw, ockl -> bchwkl'
                    grad_patches = np.einsum('bohw, ockl -> bchwkl', grad_output, self.weight.data)
                    
                    # Create a zero-padded array for the gradient
                    grad_x_padded = np.zeros_like(x_padded_data)
                    
                    for b in range(B):
                        for c in range(C_in):
                            for h in range(H_out):
                                for w in range(W_out):
                                    # Find the window in the padded gradient array
                                    h_start, w_start = h * S, w * S
                                    h_end, w_end = h_start + K, w_start + K
                                    
                                    # Add the gradient from this patch
                                    grad_x_padded[b, c, h_start:h_end, w_start:w_end] += grad_patches[b, c, h, w, :, :]
                    
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

        _weight = np.random.randn(
            out_channels, in_channels, kernel_size) 
        self.weight = Tensor(_weight, requires_grad=True,dtype=dtype)
        initializers.He_normal(self.weight, nonlinearity='relu')

        if self.has_bias:
            _bias = np.zeros((1, out_channels))
            self.bias = Tensor(_bias, requires_grad=True, dtype=dtype)
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
                    grad_bias = grad_output.sum(axis=(0, 2)) # Shape (O,)
                    self.bias.grad += grad_bias.reshape(self.bias.data.shape)

                # --- 3b. Calculate dL/dw ---
                if self.weight.requires_grad:
                    # 'patches' is from the forward pass
                    grad_weight = np.einsum('bol, bclk -> ock', grad_output, patches)
                    self.weight.grad += grad_weight

                # --- 3c. Calculate dL/dx ---
                if x.requires_grad:
                    # Calculate dL/d(patches)
                    # 'bol, ock -> bclk'
                    grad_patches = np.einsum('bol, ock -> bclk', grad_output, self.weight.data)
                    
                    # Create a zero-padded array for the gradient
                    grad_x_padded = np.zeros_like(x_padded_data)
                    
                    # We cannot use the strided view for a scatter-add.
                    # We must loop manually.
                    for b in range(B):
                        for c in range(C_in):
                            for l in range(L_out):
                                # Find the window in the padded gradient array
                                l_start = l * S
                                l_end = l_start + K
                                
                                # Add the gradient from this patch
                                grad_x_padded[b, c, l_start:l_end] += grad_patches[b, c, l, :]
                    
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


class BatchNorm1d(Module):
    """Batch Normalization over a batch of 2-D or 3-D inputs.

    During training, normalizes each feature across the batch and maintains
    running statistics for use at evaluation time.

    Reference: https://arxiv.org/abs/1502.03167
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True) -> None:
        """Create a BatchNorm1d layer.

        Args:
            num_features: Number of features/channels to normalize.
            eps: Small constant added to the denominator for numerical stability.
            momentum: Momentum factor for exponential moving average of
                running mean and variance.
            affine: If ``True``, learnable scale (gamma) and shift (beta)
                parameters are added.
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.affine = affine
        
        if affine:
            self.gamma = Tensor(np.ones((1, num_features)), requires_grad=True)
            self.beta = Tensor(np.zeros((1, num_features)), requires_grad=True)
        else:
            self.gamma = Tensor(np.ones((1, num_features)), requires_grad=False)
            self.beta = Tensor(np.zeros((1, num_features)), requires_grad=False)
            
        self.running_mean = Tensor(np.zeros((1, num_features)), requires_grad=False)
        self.running_var = Tensor(np.ones((1, num_features)), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the input tensor.

        Args:
            x: Input tensor of shape ``(batch_size, num_features)`` or
                ``(batch_size, seq_len, num_features)``.

        Returns:
            Normalized tensor of the same shape.
        """
        orig_shape = x.shape
        # Ensure parameters run in the same dtype as the input to avoid
        # upcasting during arithmetic. If parameters were initialized with a
        # different dtype (e.g., float64) we cast them to the input dtype.
        target_dtype = x.data.dtype
        if self.gamma.data.dtype != target_dtype:
            self.gamma.data = self.gamma.data.astype(target_dtype)
            self.beta.data = self.beta.data.astype(target_dtype)
            self.running_mean.data = self.running_mean.data.astype(target_dtype)
            self.running_var.data = self.running_var.data.astype(target_dtype)
        if len(x.shape) == 3:
            x = x.reshape((-1, x.shape[-1]))
        
        assert len(x.shape) == 2, f"Input tensor must be 2D or 3D, got shape {orig_shape}"
        
        if self.training:
            batch_mean = x.mean(axis=0, keepdims=True)
            centered = x - batch_mean
            batch_var = (centered ** 2).mean(axis=0, keepdims=True)
            
            x_normalized = centered / (batch_var + self.eps).sqrt()
            
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + \
                                   self.momentum * batch_mean.data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + \
                                  self.momentum * batch_var.data
        else:
            x_normalized = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        
        out = self.gamma * x_normalized + self.beta
        
        if len(orig_shape) == 3:
            out = out.reshape(orig_shape)
            
        return out
    

    def __repr__(self) -> str:
        num_features = self.gamma.shape[1]
        return f"BatchNorm1d(num_features={num_features}, eps={self.eps}, momentum={self.momentum})"
    
    def __str__(self) -> str:
        return self.__repr__()


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
    """A lookup table that maps integer indices to dense vectors.

    Weights are initialized with He normal initialization.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: Optional[npt.DTypeLike] = None) -> None:
        """Create an Embedding layer.

        Args:
            num_embeddings: Size of the vocabulary (number of rows).
            embedding_dim: Dimension of each embedding vector.
            dtype: Data type for the embedding weight matrix.
        """
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim  
        weight = np.random.randn(num_embeddings, embedding_dim)  
        self.weight = Tensor(weight, requires_grad=True,dtype=dtype) 
        initializers.He_normal(self.weight, nonlinearity='relu')


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

    def forward(self, x: Tensor, training: bool=False) -> Tensor:
        self.training = training 
        if not self.training:
            return x
        mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32)
        mask_tensor = Tensor(mask, requires_grad=False)
        scale = 1.0 / (1.0 - self.p)
        return (x * mask_tensor) * scale


class MCDropout(Dropout): 
    """ 
    in MCDroptout training is always true during training and inference 
    """ 
    def __init__(self,p=0.5):
        super().__init__(p)  
    
    def forward(self,x:Tensor)  -> Tensor: 
        super().forward(x,training = True)

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

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass and builds the computation graph.
        """
        assert len(x.shape) == 4, "MaxPool2d input must be 4D (B, C, H, W)"
        
        B, C, H_in, W_in = x.shape
        K_h, K_w = self.kernel_size
        S_h, S_w = self.stride
        P = self.padding

        # --- 1. Forward Pass (Numpy/CuPy land) ---
        
        # Apply padding. We pad with -infinity so that padded values
        # are never chosen as the maximum.
        x_padded_data = np.pad(
            x.data, 
            ((0, 0), (0, 0), (P, P), (P, P)), 
            'constant', 
            constant_values=-np.inf
        )
        
        padded_shape = x_padded_data.shape # (B, C, H_pad, W_pad)

        # Calculate output dimensions
        H_out = (H_in - K_h + 2 * P) // S_h + 1
        W_out = (W_in - K_w + 2 * P) // S_w + 1

        # Create output arrays
        output_data = np.zeros((B, C, H_out, W_out))
        
        indices = np.zeros((B, C, H_out, W_out, 2), dtype=int)

        # Loop-based forward pass to find maxes and store indices
        for b in range(B):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start, w_start = h * S_h, w * S_w
                        h_end, w_end = h_start + K_h, w_start + K_w
                        
                        window = x_padded_data[b, c, h_start:h_end, w_start:w_end]
                        
                        output_data[b, c, h, w] = np.max(window)
                        
                        # Convert to numpy for unravel_index (CuPy doesn't have it)
                        window_np = window if np is np else np.asarray(window)
                        h_idx_window, w_idx_window = np.unravel_index(np.argmax(window_np), window_np.shape)
                        
                        indices[b, c, h, w, 0] = h_start + h_idx_window
                        indices[b, c, h, w, 1] = w_start + w_idx_window

        # --- 2. Create Output Tensor (Autograd land) ---
        out = Tensor(output_data, _children=(x,), _op='MaxPool2d')
        
        # Save context for backward pass
        self.cache['input_padded_shape'] = padded_shape
        self.cache['indices'] = indices

        # --- 3. Define Backward Pass ---
        if out.requires_grad:
            def _backward():
                if not x.requires_grad:
                    return

                # Get incoming gradient
                grad_output = out.grad  # (B, C, H_out, W_out)
                
                # Get saved context
                indices = self.cache['indices'] # (B, C, H_out, W_out, 2)
                input_padded_shape = self.cache['input_padded_shape']
                
                # Create the gradient for the padded input
                grad_x_padded = np.zeros(input_padded_shape)
                
                B, C, H_out, W_out = grad_output.shape

                # Loop and "scatter" the gradients
                for b in range(B):
                    for c in range(C):
                        for h in range(H_out):
                            for w in range(W_out):
                                # Get the (h, w) coordinate from the forward pass
                                h_idx = indices[b, c, h, w, 0]
                                w_idx = indices[b, c, h, w, 1]
                                
                                # Get the gradient value
                                grad_val = grad_output[b, c, h, w]
                                
                                # Add it to the single max location.
                                # We use += in case multiple output windows
                                # (from overlapping strides) picked the same
                                # input element as their max.
                                grad_x_padded[b, c, h_idx, w_idx] += grad_val
                
                # Un-pad the gradient
                if P > 0:
                    grad_x = grad_x_padded[:, :, P:-P, P:-P]
                else:
                    grad_x = grad_x_padded
                
                # Accumulate gradient in the input tensor
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



class RNN(Module):
    """Vanilla (Elman) RNN layer with tanh non-linearity.

    At each time step *t* the hidden state is updated as::

        h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)

    Notes:
        * Expects input of shape ``(batch_size, time_steps, n_features)``.
        * Parameters are lazily initialized on the first forward pass
          once the input feature dimension is known.
    """

    def __init__(self, n_neurons: int = 1, return_sequence: bool = False, dtype: Optional[npt.DTypeLike] = None) -> None:
        """Create an RNN layer.

        Args:
            n_neurons: Number of hidden units.
            return_sequence: If ``True``, returns the hidden state at every
                time step ``(batch_size, time_steps, n_neurons)``.
                If ``False``, returns only the final hidden state
                ``(batch_size, n_neurons)``.
            dtype: Data type for parameters.
        """
        self.n_neurons = n_neurons
        self.return_sequence = return_sequence
        self.dtype = dtype

        # Parameters will be lazily initialized on the first forward pass
        # once we know the input feature dimension.
        self._initialized = False
        self.input_size: Optional[int] = None

    def _init_parameters(self, n_features: int) -> None:
        """Initialize RNN parameters based on the input feature size."""
        self.input_size = n_features
        # Xavier/Glorot-like scaling for stability
        limit = np.sqrt(1.0 / max(1, n_features))  # np.sqrt for constant

        Wxh = np.random.randn(n_features, self.n_neurons) * limit
        Whh = np.random.randn(self.n_neurons, self.n_neurons) * limit
        bh = np.zeros((1, self.n_neurons))

        self.Wxh = Tensor(Wxh, requires_grad=True, dtype=self.dtype)
        self.Whh = Tensor(Whh, requires_grad=True, dtype=self.dtype)
        self.bh = Tensor(bh, requires_grad=True, dtype=self.dtype)

        self._initialized = True

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass of the RNN.

        Args:
            x: Input tensor of shape (batch_size, time_steps, n_features).
        Returns:
            Tensor of shape (batch_size, time_steps, n_neurons) if
            ``return_sequence=True`` else (batch_size, n_neurons).
        """
        assert len(x.shape) == 3, f"Expected input to be 3D got {len(x.shape)}"
        batch_size, time_steps, n_features = x.shape

        if not self._initialized:
            self._init_parameters(n_features)
        else:
            assert (
                n_features == self.input_size
            ), f"RNN expected input feature size {self.input_size}, got {n_features}"

        # Initial hidden state h_0 = 0
        h_t = Tensor.zeros(batch_size, self.n_neurons, requires_grad=False)
        outputs: List[Tensor] = []

        for t in range(time_steps):
            x_t = x[:, t, :]  # (batch_size, n_features)
            h_t = (x_t @ self.Wxh + h_t @ self.Whh + self.bh).tanh()
            if self.return_sequence:
                outputs.append(h_t)

        if self.return_sequence:
            out_data = np.stack([h.data for h in outputs], axis=1)
            out = Tensor(out_data, _children=tuple(outputs), _op='RNNSequence')
            if out.requires_grad:
                def _backward():
                    grad_out = out.grad  # (B, T, H)
                    for t, h in enumerate(outputs):
                        if h.requires_grad:
                            h.grad += grad_out[:, t, :]

                out._backward = _backward

            return out
        else:
            # Final hidden state already carries the full recurrent graph.
            # No custom backward hook is needed in the non-sequence case.
            return h_t


    def __repr__(self) -> str:
        base = f"RNN(n_neurons={self.n_neurons}, return_sequence={self.return_sequence}"
        if self.input_size is not None:
            base += f", input_size={self.input_size}"
        return base + ")"

    def __str__(self) -> str:
        return self.__repr__()
    
class LSTM(Module):
    """Basic (single-layer) LSTM over sequences.

    Expects input of shape ``(batch_size, time_steps, input_size)``.

    This implementation is intentionally minimal and supports:
    - lazy initialization of weights on first forward pass
    - optional initial states ``h_t`` and ``c_t``
    - returning either the final hidden state or the full hidden sequence
    """

    def __init__(
        self,
        return_sequence: bool = False,
        hidden_size: int = 1,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.is_initialized = False
        self.return_sequence = return_sequence
        self.hidden_size = hidden_size

        # Set on first forward pass.
        self.input_size: Optional[int] = None

    def _init_parameters(self, input_size: int, effective_dtype: npt.DTypeLike) -> None:
        """Initialize LSTM parameters for a given input feature size."""
        self.input_size = input_size

        # Use small uniform scaling for stability.
        limit = np.sqrt(1.0 / max(1, input_size))

        def rand_w(in_dim: int, out_dim: int) -> Tensor:
            data = (np.random.uniform(-limit, limit, size=(in_dim, out_dim))).astype(effective_dtype)
            return Tensor(data, requires_grad=True, dtype=effective_dtype)

        def rand_b(out_dim: int) -> Tensor:
            data = np.zeros((1, out_dim), dtype=effective_dtype)
            return Tensor(data, requires_grad=True, dtype=effective_dtype)

        # Forget gate parameters
        self.w_xf = rand_w(input_size, self.hidden_size)
        self.w_hf = rand_w(self.hidden_size, self.hidden_size)
        self.b_f = rand_b(self.hidden_size)

        # Input gate parameters
        self.w_xi = rand_w(input_size, self.hidden_size)
        self.w_hi = rand_w(self.hidden_size, self.hidden_size)
        self.b_i = rand_b(self.hidden_size)

        # Output gate parameters
        self.w_xo = rand_w(input_size, self.hidden_size)
        self.w_ho = rand_w(self.hidden_size, self.hidden_size)
        self.b_o = rand_b(self.hidden_size)

        # Cell candidate parameters
        self.w_xg = rand_w(input_size, self.hidden_size)
        self.w_hg = rand_w(self.hidden_size, self.hidden_size)
        self.b_g = rand_b(self.hidden_size)

        self.is_initialized = True

    def forward(
        self,
        x: Tensor,
        c_t: Optional[Tensor] = None,
        h_t: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, input_size)
            c_t: Optional initial cell state, (B, hidden_size)
            h_t: Optional initial hidden state, (B, hidden_size)

        Returns:
            - If ``return_sequence=False``: (h_T, c_T)
            - If ``return_sequence=True``: (H_seq, c_T)
        """
        assert len(x.shape) == 3, f"Expected input to be 3D, got shape {x.shape}"
        batch_size, time_steps, input_size = x.shape

        effective_dtype: npt.DTypeLike = x.data.dtype if self.dtype is None else self.dtype
        if self.dtype is not None and x.data.dtype != np.dtype(self.dtype):
            raise ValueError(f"LSTM dtype mismatch: x dtype {x.data.dtype} != layer dtype {np.dtype(self.dtype)}")
        if not self.is_initialized:
            self._init_parameters(input_size, effective_dtype=effective_dtype)
        else:
            assert (
                input_size == self.input_size
            ), f"LSTM expected input feature size {self.input_size}, got {input_size}"

        if c_t is not None:
            assert c_t.shape == (batch_size, self.hidden_size), "c_t shape must be (batch_size,hidden_size)"
            if c_t.dtype != np.dtype(effective_dtype):
                raise ValueError(f"c_t dtype mismatch: {c_t.dtype} != {np.dtype(effective_dtype)}")
        if h_t is not None:
            assert h_t.shape == (batch_size, self.hidden_size), "h_t shape must be (batch_size,hidden_size)"
            if h_t.dtype != np.dtype(effective_dtype):
                raise ValueError(f"h_t dtype mismatch: {h_t.dtype} != {np.dtype(effective_dtype)}")

        if h_t is None:
            h_t = Tensor.zeros(batch_size, self.hidden_size, requires_grad=False, dtype=effective_dtype)
        if c_t is None:
            c_t = Tensor.zeros(batch_size, self.hidden_size, requires_grad=False, dtype=effective_dtype)

        outputs: List[Tensor] = []

        for t in range(time_steps):
            x_t = x[:, t, :]  # (B, input_size)

            # Gate computations
            f_t = (x_t @ self.w_xf + h_t @ self.w_hf + self.b_f).sigmoid()
            i_t = (x_t @ self.w_xi + h_t @ self.w_hi + self.b_i).sigmoid()
            o_t = (x_t @ self.w_xo + h_t @ self.w_ho + self.b_o).sigmoid()
            g_t = (x_t @ self.w_xg + h_t @ self.w_hg + self.b_g).tanh()

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * c_t.tanh()

            if self.return_sequence:
                outputs.append(h_t)

        if self.return_sequence:
            out_data = np.stack([h.data for h in outputs], axis=1)
            h_seq = Tensor(out_data, _children=tuple(outputs), _op="LSTMSequence")
            if h_seq.requires_grad:
                def _backward() -> None:
                    grad_out = h_seq.grad  # (B, T, H)
                    for t, h in enumerate(outputs):
                        if h.requires_grad:
                            h.grad += grad_out[:, t, :]

                h_seq._backward = _backward
            return h_seq, c_t

        return h_t, c_t

class MultiHeadAttention(Module):
    """Multi-Head Attention (Vaswani et al., 2017).

    Splits queries, keys and values into ``num_heads`` parallel attention
    heads, computes scaled dot-product attention independently for each
    head, and concatenates the results.

    Features:
        * Causal masking for autoregressive models.
        * Dropout regularization on attention weights.
        * Optional QKV bias.

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
            mask = np.triu(np.ones((max_seq_len, max_seq_len), dtype=np.float32), k=1)
            self._causal_mask = Tensor(mask, requires_grad=False)
    
    def _get_causal_mask(self, seq_len: int) -> Optional[Tensor]:
        """
        Get or create causal mask for the given sequence length.
        
        Args:
            seq_len: Current sequence length
        
        Returns:
            Causal mask tensor of shape (seq_len, seq_len) or None if not causal
        """
        if not self.causal:
            return None
        
        if self._causal_mask is not None and seq_len <= self._causal_mask.shape[0]:
            mask_slice = self._causal_mask[:seq_len, :seq_len]
            return mask_slice.bool()
        
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
        mask_tensor = Tensor(mask, requires_grad=False)
        return mask_tensor.bool()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of Multi-Head Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_out)
        """
        B, T, _ = x.shape
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)
        
        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        K = K.reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        V = V.reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        
        attn_scores = (Q @ K.transpose((0, 1, 3, 2))) * self.scale
        
        if self.causal:
            causal_mask = self._get_causal_mask(T)
            if causal_mask is not None:
                mask_broadcast = causal_mask.data[None, None, :, :]
                attn_scores = attn_scores.masked_fill(
                    Tensor(mask_broadcast, requires_grad=False), 
                    float('-inf')
                )
        
        attn_weights = attn_scores.softmax(axis=-1)
        attn_weights = self.dropout_layer(attn_weights)
        context = attn_weights @ V
        context = context.transpose((0, 2, 1, 3)).reshape(B, T, self.d_out)
        out = self.out_proj(context)
        return out
    
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
            pooled = patches.mean(axis=(-2, -1))

        out = Tensor(pooled, _children=(x,), _op="AvgPool2d")

        if out.requires_grad:

            def _backward():
                grad_output = out.grad

                if x.requires_grad:
                    kh, kw = self.kernel_size
                    area = kh * kw if self.count_include_pad else None

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

                                    if self.count_include_pad:
                                        window += grad_output[b, c, ho, wo] / area
                                    else:
                                        # If not counting pad we'd need a mask
                                        pass

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