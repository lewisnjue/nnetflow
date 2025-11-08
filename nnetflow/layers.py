import numpy as np 
from typing import Union
from nnetflow.engine import Tensor 

class Linear:
    def __init__(self,in_features:int, out_features:int,bias = True) -> None: 
        self.in_features = in_features 
        self.out_features = out_features 
        _weight = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features) 
        _bias = np.zeros((1, out_features)) 
        self.weight = Tensor(_weight, requires_grad=True)
        self.has_bias = bias
        if bias:
            self.bias = Tensor(_bias, requires_grad=True)
    
    def __call__(self,x:Tensor) -> Tensor:
        assert x.shape[-1] == self.in_features, f"Input feature size mismatch, expected {self.in_features}, got {x.shape[-1]}"
        # x : (batch_size, in_features), or x: (b,t,in) @  (in,out)  
        # weight : (in_features, out_features) 
        # x @ weight (batch_size,in_features) @ (in_features, out_features) = (batch_size, out_features)
        if self.has_bias:
             return x @ self.weight + self.bias 
        else:
            return x @ self.weight 
    
    def parameters(self):
        if self.has_bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]
 

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})" 
    
    def __str__(self):
        return self.__repr__()
    

import numpy as np
# Assuming 'Tensor', 'Union', 'Tuple' are defined
# from your_tensor_library import Tensor
# from typing import Union, Tuple

class BatchNorm1d:
    """
    Batch Normalization layer that normalizes inputs across the batch dimension.
    Supports both training and evaluation modes.
    
    Args:
        num_features: Number of features/channels to normalize
        eps: Small constant for numerical stability (default: 1e-5)
        momentum: Momentum factor for running stats (default: 0.1)
        affine: If True, has learnable affine parameters (default: True)
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True) -> None:
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
            
        # Running statistics (not trainable)
        self.running_mean = Tensor(np.zeros((1, num_features)), requires_grad=False)
        self.running_var = Tensor(np.ones((1, num_features)), requires_grad=False)

    def __call__(self, x: Tensor) -> Tensor:
        # Handle both 2D and 3D inputs
        orig_shape = x.shape
        if len(x.shape) == 3:
            # (batch, seq_len, features) -> (batch * seq_len, features)
            x = x.reshape((-1, x.shape[-1]))
        
        assert len(x.shape) == 2, f"Input tensor must be 2D or 3D, got shape {orig_shape}"
        
        if self.training:
            # Compute mean and variance over batch dimension
            batch_mean = x.mean(axis=0, keepdims=True)
            # Use more numerically stable variance computation
            centered = x - batch_mean
            batch_var = (centered ** 2).mean(axis=0, keepdims=True)
            
            # Normalize
            x_normalized = centered / (batch_var + self.eps).sqrt()
            
            # Update running statistics
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + \
                                   self.momentum * batch_mean.data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + \
                                  self.momentum * batch_var.data
        else:
            # Use running statistics in eval mode
            x_normalized = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        
        # Apply affine transform
        out = self.gamma * x_normalized + self.beta
        
        # Restore original shape if input was 3D
        if len(orig_shape) == 3:
            out = out.reshape(orig_shape)
            
        return out
    
    def parameters(self):
        if self.affine:
            return [self.gamma, self.beta]
        return []

    def __repr__(self) -> str:
        return (f"BatchNorm1d(num_features={self.num_features}, "
                f"eps={self.eps}, momentum={self.momentum}, affine={self.affine})")
    
    def parameters(self):
        return [self.gamma, self.beta]

    def __repr__(self) -> str:
        num_features = self.gamma.shape[1]
        return f"BatchNorm1d(num_features={num_features}, eps={self.eps}, momentum={self.momentum})"
    
    def __str__(self) -> str:
        return self.__repr__()

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5) -> None: 
        """Initialize LayerNorm.
        Args:
            dim: The size of the last dimension of input tensors
            eps: Small constant for numerical stability
        """
        self.eps = eps
        # Create parameters with shape (1, dim) for proper broadcasting
        self.gamma = Tensor(np.ones((1, dim)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, dim)), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor: 
        # Compute mean and variance along last dimension
        mean = x.mean(axis=-1, keepdims=True)  # Shape: (..., 1)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)  # Shape: (..., 1)
        
        # Normalize
        x_normalized = (x - mean) / (var + self.eps).sqrt()
        
        # Scale and shift (gamma and beta will broadcast automatically)
        out = self.gamma * x_normalized + self.beta
        return out 
    
    def parameters(self):
        return [self.gamma, self.beta]
    
class Embedding: 
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None: 
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim  
        weight = np.random.randn(num_embeddings, embedding_dim) * 0.01  
        self.weight = Tensor(weight, requires_grad=True) 
    
    def __call__(self, indices: Union[int, slice, tuple]) -> Tensor:
        embedded = self.weight[indices]
        return embedded

    def parameters(self):
        return [self.weight]


class Dropout: 
    def __init__(self,p:float=0.5,training=True) ->None: 
        assert 0.0 <= p < 1.0 , "Dropout probability must be in [0.0,1.0] range" 
        self.p = p 
        self.training = training 
    
    def __call__(self,x:Tensor) -> Tensor: 
        if self.training:
            mask = (np.random.rand(*x.shape) >= self.p).astype(np.float32) / (1.0 - self.p)
            return x * Tensor(mask, requires_grad=False)
        else:
            return x
    
    def __repr__(self) -> str:
        return f"Dropout(p={self.p}, training={self.training})"

    def __str__(self) -> str:
        return self.__repr__()

    def parameters(self):
        return []




class Flatten:
    def __init__(self) -> None:
        pass 
    
    def __call__(self,x:Tensor) -> Tensor:
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
    
    def __repr__(self) -> str:
        return "Flatten()"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def parameters(self):
        return []

