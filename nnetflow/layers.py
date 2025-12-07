import numpy as np 
from nnetflow.engine import Tensor 
from typing import Union, List, Tuple, Optional, Dict, Any
from nnetflow.init import initializers
from nnetflow.module import Module
from nnetflow.device import get_array_module
import numpy.typing as npt

class Linear(Module):
    """
    This class creates a Linear Layer. This is a dense layer 
    it perform the following mathematical calculation: 
    out_put = input @ weight + bias 
    by default weights will be initalized using He uniform initialization 
    """
    def __init__(self,in_features:int, out_features:int,bias = True,dtype:Optional[npt.DTypeLike] = None) -> None: 
        """ 
        Args: 
            in_features: this is the number of features in your input tensor 
            out_features: this is the number of neuron to create  which are all fully connected to the in_features 
            bias: if true , bias is added if not it is not 
            dtype: the data type of the weights and bias tensors 
        Returns: 
            None 
        """
        self.in_features = in_features 
        self.out_features = out_features 
        xp = get_array_module()
        _weight = xp.random.randn(in_features, out_features) 
        _bias = xp.zeros((1, out_features)) 
        self.weight = Tensor(_weight, requires_grad=True,dtype=dtype)
        initializers.He_uniform(self.weight, nonlinearity='relu') 
        self.has_bias = bias
        if bias:
            self.bias = Tensor(_bias, requires_grad=True,dtype=dtype)
    
    def forward(self,x:Tensor) -> Tensor:
        """ 
        This is the forward pass of the layer 
        Args: 
            x: the input tensor of shape (batchsize,in_features)
        Returns: 
            Tensor 
        
        """
        assert x.shape[-1] == self.in_features, f"Input feature size mismatch, expected {self.in_features}, got {x.shape[-1]}"
        if self.has_bias:
             return x @ self.weight + self.bias 
        else:
            return x @ self.weight 

    def __repr__(self) -> str:
        """ 
        prints a string description of the layer when you try to print the layer 
        """ 
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})" 
    
    def __str__(self):
        """ 
        just call the __repr__ 
        """
        return self.__repr__()

class Conv2d(Module):
    """
    Implements a 2D convolution layer for your autograd Tensor class.
    Input format: (batch_size, in_channels, height, width)
    Link to the paper : https://arxiv.org/abs/1511.08458
    by default weights will be initialized using He normal initialization 
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True,dtype:Optional[npt.DTypeLike] = None) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (filters).
            kernel_size: Size of the kernel (assumed square).
            stride: Stride for the convolution.
            padding: Padding for the convolution.
            bias: If True, adds a bias term.
        Returns: 
            None
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        self.dtype = dtype

        fan_in = in_channels * kernel_size * kernel_size
        xp = get_array_module()
        _weight = xp.random.randn(
            out_channels, in_channels, kernel_size, kernel_size) 
        
        self.weight = Tensor(_weight, requires_grad=True,dtype=dtype)
        initializers.He_normal(self.weight, nonlinearity='relu')

        if self.has_bias:

            _bias = xp.zeros((1, out_channels))
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
        xp = get_array_module()
        x_padded_data = xp.pad(
            x.data, ((0, 0), (0, 0), (P, P), (P, P)), 'constant')
        patches = self._get_patches_strided(x_padded_data, K, S)
        output_data = xp.einsum(
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
                    xp = get_array_module()
                    # 'patches' is from the forward pass
                    grad_weight = xp.einsum('bohw, bchwkl -> ockl', grad_output, patches)
                    self.weight.grad += grad_weight

                # --- 3c. Calculate dL/dx ---
                if x.requires_grad:
                    xp = get_array_module()
                    # Calculate dL/d(patches)
                    # 'bohw, ockl -> bchwkl'
                    grad_patches = xp.einsum('bohw, ockl -> bchwkl', grad_output, self.weight.data)
                    
                    # Create a zero-padded array for the gradient
                    grad_x_padded = xp.zeros_like(x_padded_data)
                    
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
    """
    Implements a 1D convolution layer for your autograd Tensor class.
    Input format: (batch_size, in_channels, length)
    Link to the paper : https://arxiv.org/abs/1511.08458
    by default weights will be initialized using He normal initialization
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True, dtype: Optional[npt.DTypeLike] = None) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (filters).
            kernel_size: Size of the kernel.
            stride: Stride for the convolution.
            padding: Padding for the convolution.
            bias: If True, adds a bias term.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

        fan_in = in_channels * kernel_size
        xp = get_array_module()
        _weight = xp.random.randn(
            out_channels, in_channels, kernel_size) 
        self.weight = Tensor(_weight, requires_grad=True,dtype=dtype)
        initializers.He_normal(self.weight, nonlinearity='relu')

        if self.has_bias:
            _bias = xp.zeros((1, out_channels))
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
        xp = get_array_module()
        x_padded_data = xp.pad(
            x.data, ((0, 0), (0, 0), (P, P)), 'constant')
        patches = self._get_patches_strided(x_padded_data, K, S)
        output_data = xp.einsum(
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
                    xp = get_array_module()
                    # 'patches' is from the forward pass
                    grad_weight = xp.einsum('bol, bclk -> ock', grad_output, patches)
                    self.weight.grad += grad_weight

                # --- 3c. Calculate dL/dx ---
                if x.requires_grad:
                    xp = get_array_module()
                    # Calculate dL/d(patches)
                    # 'bol, ock -> bclk'
                    grad_patches = xp.einsum('bol, ock -> bclk', grad_output, self.weight.data)
                    
                    # Create a zero-padded array for the gradient
                    grad_x_padded = xp.zeros_like(x_padded_data)
                    
                    # ******** START OF FIX ********
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
                    # ******** END OF FIX ********
                    
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
    """
    Batch Normalization layer that normalizes inputs across the batch dimension.
    Supports both training and evaluation modes.
    Link to the paper : https://arxiv.org/abs/1502.03167 
    """ 
    
 
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True) -> None:
        """ 
        Args:
            num_features: Number of features/channels to normalize
            eps: Small constant for numerical stability (default: 1e-5)
            momentum: Momentum factor for running stats (default: 0.1)
            affine: If True, has learnable affine parameters (default: True)
        Returns: 
            None 
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.affine = affine
        
        xp = get_array_module()
        if affine:
            self.gamma = Tensor(xp.ones((1, num_features)), requires_grad=True)
            self.beta = Tensor(xp.zeros((1, num_features)), requires_grad=True)
        else:
            self.gamma = Tensor(xp.ones((1, num_features)), requires_grad=False)
            self.beta = Tensor(xp.zeros((1, num_features)), requires_grad=False)
            
        self.running_mean = Tensor(xp.zeros((1, num_features)), requires_grad=False)
        self.running_var = Tensor(xp.ones((1, num_features)), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """ 
        Args: 
            x: the input to be normalized 
        Returns: 
         x normalized
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
    """ 
    This layer perform Layer normalization based on this paper  : https://arxiv.org/abs/1607.06450
    """ 
    def __init__(self, dim: int, eps: float = 1e-5) -> None: 
        """Initialize LayerNorm.
        Args:
            dim: The size of the last dimension of input tensors
            eps: Small constant for numerical stability
        Returns: 
            None 
        """
        self.eps = eps
        xp = get_array_module()
        self.gamma = Tensor(xp.ones((1, dim)), requires_grad=True)
        self.beta = Tensor(xp.zeros((1, dim)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor: 
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
    """ 
    Creates an embedding layer 
    by default weights are initialized using He  normal 
    """ 

    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: Optional[npt.DTypeLike] = None) -> None:
        """ 
        Args: 
            num_embeddings: number of embeddings you want to create 
            embedding_dim: the embedding dimention 
        Returns: 
            None 
        """ 
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim  
        xp = get_array_module()
        weight = xp.random.randn(num_embeddings, embedding_dim)  
        self.weight = Tensor(weight, requires_grad=True,dtype=dtype) 
        initializers.He_normal(self.weight, nonlinearity='relu')

    
    def forward(self, indices: Union[int, slice, tuple]) -> Tensor:
        embedded = self.weight[indices]
        return embedded


class Dropout(Module): 
    """ 
    applies dropout based on this paper : https://arxiv.org/pdf/1904.13310
    """ 
    def __init__(self,p:float=0.5,training=True) ->None: 
        """ 
        Args: 
            p: probability of droping a neuron 
            training: if the Dropout layer is in training model 
        Returns: 
            None 
        """ 
        assert 0.0 <= p < 1.0 , "Dropout probability must be in [0.0,1.0] range" 
        self.p = p 
        self.training = training 
    
    def forward(self,x:Tensor) -> Tensor: 
        if self.training:
            xp = get_array_module()
            mask = (xp.random.rand(*x.shape) >= self.p).astype(xp.float32) / (1.0 - self.p)
            return x * Tensor(mask, requires_grad=False)
        else:
            return x
    
    def __repr__(self) -> str:
        return f"Dropout(p={self.p}, training={self.training})"

    def __str__(self) -> str:
        return self.__repr__()

class Flatten(Module):
    """ 
    reshapes the tensor to shape of (batch_size,-1) 
    """  
    def __init__(self) -> None:
        """ 
        init is empty 
        """ 
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
        xp = get_array_module()
        
        # Apply padding. We pad with -infinity so that padded values
        # are never chosen as the maximum.
        x_padded_data = xp.pad(
            x.data, 
            ((0, 0), (0, 0), (P, P), (P, P)), 
            'constant', 
            constant_values=-xp.inf
        )
        
        padded_shape = x_padded_data.shape # (B, C, H_pad, W_pad)

        # Calculate output dimensions
        H_out = (H_in - K_h + 2 * P) // S_h + 1
        W_out = (W_in - K_w + 2 * P) // S_w + 1

        # Create output arrays
        output_data = xp.zeros((B, C, H_out, W_out))
        
        indices = xp.zeros((B, C, H_out, W_out, 2), dtype=int)

        # Loop-based forward pass to find maxes and store indices
        for b in range(B):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start, w_start = h * S_h, w * S_w
                        h_end, w_end = h_start + K_h, w_start + K_w
                        
                        window = x_padded_data[b, c, h_start:h_end, w_start:w_end]
                        
                        output_data[b, c, h, w] = xp.max(window)
                        
                        # Convert to numpy for unravel_index (CuPy doesn't have it)
                        window_np = window if xp is np else np.asarray(window)
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
                xp = get_array_module()
                grad_x_padded = xp.zeros(input_padded_shape)
                
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
        xp = get_array_module()
        
        # Pad with -infinity
        x_padded_data = xp.pad(
            x.data, 
            ((0, 0), (0, 0), (P, P)), 
            'constant', 
            constant_values=-xp.inf
        )
        
        padded_shape = x_padded_data.shape # (B, C, L_pad)

        # Calculate output dimensions
        L_out = (L_in - K + 2 * P) // S + 1

        # Create output arrays
        output_data = xp.zeros((B, C, L_out))
        
        # 'indices' will store the (l) coordinate from the *padded*
        # input array for each max value.
        # Shape: (B, C, L_out)
        indices = xp.zeros((B, C, L_out), dtype=int)

        # Loop-based forward pass
        for b in range(B):
            for c in range(C):
                for l in range(L_out):
                    # Find the window in the padded input
                    l_start = l * S
                    l_end = l_start + K
                    
                    window = x_padded_data[b, c, l_start:l_end]
                    
                    # Get the max value
                    output_data[b, c, l] = xp.max(window)
                    
                    # Get the 1D index *within the window*
                    l_idx_window = xp.argmax(window)
                    
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
                xp = get_array_module()
                grad_x_padded = xp.zeros(input_padded_shape)
                
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
    """ 
    A simple RNN layer capable of handling sequential data.

    Notes:
        * Expects input of shape ``(batch_size, time_steps, n_features)``.
        * Uses a vanilla tanh RNN cell.
        * This is a simplified implementation and currently assumes a fixed
          input feature size inferred on the first forward pass.
    """

    def __init__(self, n_neurons: int = 1, return_sequence: bool = False, dtype: Optional[npt.DTypeLike] = None) -> None:
        """ 
        Args:
            n_neurons: Number of hidden units in this RNN layer.
            return_sequence: If True, returns the hidden state at every
                time step with shape ``(batch_size, time_steps, n_neurons)``.
                If False, returns only the final hidden state with shape
                ``(batch_size, n_neurons)``.
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
        xp = get_array_module()
        limit = np.sqrt(1.0 / max(1, n_features))  # np.sqrt for constant

        Wxh = xp.random.randn(n_features, self.n_neurons) * limit
        Whh = xp.random.randn(self.n_neurons, self.n_neurons) * limit
        bh = xp.zeros((1, self.n_neurons))

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
            xp = get_array_module()
            out_data = xp.stack([h.data for h in outputs], axis=1)
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
            return h_t


    def __repr__(self) -> str:
        base = f"RNN(n_neurons={self.n_neurons}, return_sequence={self.return_sequence}"
        if self.input_size is not None:
            base += f", input_size={self.input_size}"
        return base + ")"

    def __str__(self) -> str:
        return self.__repr__()
    


class MultiHeadAttention(Module):
    """
    Multi-Head Attention mechanism as described in "Attention Is All You Need" (Vaswani et al., 2017).
    
    This implementation supports:
    - Causal masking for autoregressive models
    - Dropout for regularization
    - Optional QKV bias
    - Optimized for both CPU and GPU
    
    Link to paper: https://arxiv.org/abs/1706.03762
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
        
        # Scale factor for attention scores (1/sqrt(head_dim))
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        # QKV projections - can be combined for efficiency, but separate is clearer
        self.W_query = Linear(d_in, d_out, bias=bias, dtype=dtype)
        self.W_key = Linear(d_in, d_out, bias=bias, dtype=dtype)
        self.W_value = Linear(d_in, d_out, bias=bias, dtype=dtype)
        
        # Output projection
        self.out_proj = Linear(d_out, d_out, bias=True, dtype=dtype)
        
        # Dropout layer
        self.dropout_layer = Dropout(dropout)
        
        # Causal mask (created lazily or pre-computed)
        self._causal_mask: Optional[Tensor] = None
        if causal and max_seq_len is not None:
            # Pre-compute mask if max_seq_len is provided
            xp = get_array_module()
            # Use float for mask, bool can cause issues with some operations
            mask = xp.triu(xp.ones((max_seq_len, max_seq_len), dtype=xp.float32), k=1)
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
        
        # If mask was pre-computed and seq_len is within bounds, use it
        if self._causal_mask is not None and seq_len <= self._causal_mask.shape[0]:
            mask_slice = self._causal_mask[:seq_len, :seq_len]
            return mask_slice.bool()
        
        # Otherwise, create mask dynamically
        xp = get_array_module()
        # Create mask as float, then convert to bool when needed
        mask = xp.triu(xp.ones((seq_len, seq_len), dtype=xp.float32), k=1)
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
        
        # Project to Q, K, V: (B, T, d_out)
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)
        
        # Reshape and transpose for multi-head attention
        # (B, T, d_out) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        K = K.reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        V = V.reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        
        # Compute attention scores: (B, num_heads, T, T)
        # Q @ K^T / sqrt(head_dim)
        attn_scores = (Q @ K.transpose((0, 1, 3, 2))) * self.scale
        
        # Apply causal mask if enabled
        if self.causal:
            causal_mask = self._get_causal_mask(T)
            if causal_mask is not None:
                # Expand mask to match attention scores shape: (1, 1, T, T)
                # The mask is already bool, just need to broadcast it
                mask_broadcast = causal_mask.data[None, None, :, :]
                # Apply mask: set masked positions to -inf
                attn_scores = attn_scores.masked_fill(
                    Tensor(mask_broadcast, requires_grad=False), 
                    float('-inf')
                )
        
        # Apply softmax to get attention weights
        attn_weights = attn_scores.softmax(axis=-1)
        
        # Apply dropout to attention weights
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values: (B, num_heads, T, head_dim)
        context = attn_weights @ V
        
        # Reshape back: (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) -> (B, T, d_out)
        context = context.transpose((0, 2, 1, 3)).reshape(B, T, self.d_out)
        
        # Final output projection
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