import numpy as np
from .engine import Tensor
from typing import List, Tuple, Optional, Union
from numpy.lib.stride_tricks import as_strided


def im2col_2d(arr: np.ndarray,
                      kernel_size: Tuple[int,int],
                      stride: int) -> np.ndarray:
    B, C, H, W = arr.shape
    kH, kW = kernel_size

    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1

  
    new_shape = (B, out_h, out_w, C, kH, kW)
    new_strides = (
        arr.strides[0],           # batch
        arr.strides[2] * stride,  # window row step
        arr.strides[3] * stride,  # window col step
        arr.strides[1],           # channel
        arr.strides[2],           # kernel row
        arr.strides[3],           # kernel col
    )

    windows = as_strided(arr, shape=new_shape, strides=new_strides)
    patches = windows.reshape(B, out_h*out_w, C*kH*kW).transpose(0,2,1)
    return patches

class Module:
    __doc__ = "Base class for all neural network modules. Modules can contain other modules, and parameters can be added to them." 
    def __init__(self):
        self._modules = {}
        self._parameters = []

    def __setattr__(self, name, value):
        if '_modules' not in self.__dict__:
            object.__setattr__(self, '_modules', {})
        if '_parameters' not in self.__dict__:
            object.__setattr__(self, '_parameters', [])
        if isinstance(value, Module):
            self.__dict__['_modules'][name] = value
        elif isinstance(value, (list, tuple)):
            for idx, v in enumerate(value):
                if isinstance(v, Module):
                    self.__dict__['_modules'][f'{name}[{idx}]'] = v
                elif hasattr(v, 'parameters') and callable(v.parameters):
                    self.__dict__['_parameters'].extend(v.parameters())
        elif hasattr(value, 'parameters') and callable(value.parameters):
            self.__dict__['_parameters'].extend(value.parameters())
        object.__setattr__(self, name, value)

    def parameters(self):
        params = list(self.__dict__.get('_parameters', []))
        for module in self.__dict__.get('_modules', {}).values():
            params.extend(module.parameters())
        for v in self.__dict__.values():
            if isinstance(v, (list, tuple)):
                for item in v:
                    if hasattr(item, 'parameters') and callable(item.parameters):
                        p = item.parameters()
                        if isinstance(p, (list, tuple)):
                            params.extend(p)
                        else:
                            params.append(p)
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        std = 1 / np.sqrt(num_embeddings)
        w = np.random.randn(num_embeddings, embedding_dim) * std
        self.weight = Tensor(w, shape=w.shape)

    def __call__(self, idx):
        # idx: numpy array or int or list of indices
        out_data = self.weight.data[idx]
        out = Tensor(out_data, _children=(self.weight,), _op='embedding')
        def _backward():
            # Accumulate gradients for the selected indices
            if self.weight.grad.shape != self.weight.data.shape:
                self.weight.grad = np.zeros_like(self.weight.data)
            np.add.at(self.weight.grad, idx, out.grad)
        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight]
    
    

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()
        std = 1 / np.sqrt(in_features) if activation in ['tanh', None] else np.sqrt(2.0 / in_features)
        w = np.random.randn(in_features, out_features) * std
        self.weight = Tensor(w)
        self.bias = Tensor(np.zeros(out_features)) if bias else None
        self.activation = activation

    def __call__(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        if self.activation == 'relu':
            out = out.relu()
        elif self.activation == 'tanh':
            out = out.tanh()
        elif self.activation == 'sigmoid':
            out = out.sigmoid()
        return out

    def parameters(self):
        return [self.weight, self.bias] if self.bias is not None else [self.weight]


class RNN(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nonlinearity: str = 'tanh',
        bias: bool = True,
        batch_first: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.use_bias = bias
        self.batch_first = batch_first

        # Weight initialization
        std = 1.0 / np.sqrt(hidden_size)
        self.weight_ih = Tensor(np.random.randn(input_size, hidden_size) * std)
        self.weight_hh = Tensor(np.random.randn(hidden_size, hidden_size) * std)
        self.bias_ih = Tensor(np.zeros(hidden_size)) if bias else None
        self.bias_hh = Tensor(np.zeros(hidden_size)) if bias else None

    def forward(
        self,
        x: Tensor,
        hx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        x: Tensor of shape (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first=True
        hx: Optional Tensor of shape (batch, hidden_size)
        returns: (output, h_n)
          - output: Tensor of shape (seq_len, batch, hidden_size)
          - h_n: Tensor of shape (batch, hidden_size)
        """
        data = x.data
        # if batch_first, convert to (seq_len, batch, input_size)
        if self.batch_first:
            data = data.transpose(1, 0, 2)

        seq_len, batch_size, _ = data.shape
        # initialize hidden state
        if hx is None:
            h_t = Tensor(np.zeros((batch_size, self.hidden_size)))
        else:
            h_t = hx

        outputs = []
        for t in range(seq_len):
            # get time-step tensor
            x_t = Tensor(data[t], _children=(x,), _op='slice')  # (batch, input_size)

            # RNN cell: linear + nonlinearity
            h_linear = x_t @ self.weight_ih + h_t @ self.weight_hh
            if self.use_bias:
                h_linear = h_linear + self.bias_ih + self.bias_hh

            if self.nonlinearity == 'tanh':
                h_t = h_linear.tanh()
            elif self.nonlinearity == 'relu':
                h_t = h_linear.relu()
            else:
                raise ValueError(f"Unknown nonlinearity {self.nonlinearity}")

            outputs.append(h_t)

        # stack outputs into one Tensor (seq_len, batch, hidden)
        stacked_data = np.stack([o.data for o in outputs], axis=0)
        output = Tensor(stacked_data, _children=tuple(outputs), _op='stack')

        def _backward_stack():
            for idx, o in enumerate(outputs):
                o.grad += output.grad[idx]
        output._backward = _backward_stack

        # final hidden state
        h_n = h_t
        return output, h_n

    def parameters(self):
        params = [self.weight_ih, self.weight_hh]
        if self.use_bias:
            params.extend([self.bias_ih, self.bias_hh])
        return params





class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        std = np.sqrt(2.0 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = Tensor(np.random.randn(out_channels, in_channels, *kernel_size) * std)
        self.bias = Tensor(np.zeros(out_channels)) if bias else None

    def __call__(self, x):
        if self.padding > 0:
            x_padded = Tensor(np.pad(x.data, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant'), _children=(x,), _op='pad')
            def _pad_backward():
                x.grad += x_padded.grad[:, :, self.padding:-self.padding, self.padding:-self.padding]
            x_padded._backward = _pad_backward
        else:
            x_padded = x

        B, C, H, W = x_padded.data.shape
        kH, kW = self.kernel_size
        out_h = (H - kH) // self.stride + 1
        out_w = (W - kW) // self.stride + 1

        cols_data = im2col_2d(x_padded.data, self.kernel_size, self.stride)
        cols = Tensor(cols_data.reshape(B, C * kH * kW, out_h * out_w), _children=(x_padded,), _op='im2col')
        weight_reshaped = self.weight.reshape(1, self.out_channels, C * kH * kW)

        out = weight_reshaped @ cols  # (B, out_channels, out_h*out_w)
        out = out.reshape(B, self.out_channels, out_h, out_w)

        if self.bias is not None:
            # Reshape bias to (1, out_channels, 1, 1) for proper broadcasting
            bias_reshaped = self.bias.reshape(1, self.out_channels, 1, 1)
            out = out + bias_reshaped

        return out

    def parameters(self):
        return [self.weight, self.bias] if self.bias is not None else [self.weight]


class MaxPool2D(Module):
    def __init__(self, kernel_size: int, stride: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def __call__(self, x: Tensor) -> Tensor:
        B, C, H, W = x.data.shape
        k = self.kernel_size
        s = self.stride

        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1

        # Create sliding-window view
        shape = (B, C, out_h, out_w, k, k)
        strides = (
            x.data.strides[0],           # batch step
            x.data.strides[1],           # channel step
            x.data.strides[2] * s,       # window row step
            x.data.strides[3] * s,       # window col step
            x.data.strides[2],           # within-window row
            x.data.strides[3],           # within-window col
        )
        windows = as_strided(x.data, shape=shape, strides=strides)

        out_data = windows.max(axis=(4, 5))
        argmax = windows.reshape(B, C, out_h, out_w, k * k).argmax(axis=4)

        out_tensor = Tensor(out_data, _children=(x,), _op='maxpool2d')
        out_tensor._argmax = argmax
        # Store kernel_size and stride for use in backward
        out_tensor._k = k
        out_tensor._s = s

        def _backward():
            k = out_tensor._k
            s = out_tensor._s
            B, C, out_h, out_w = out_tensor.data.shape

            # Ensure x.grad is initialized
            if x.grad is None:
                x.grad = np.zeros_like(x.data, dtype=out_tensor.grad.dtype)

            # Compute the position of max within each window
            argmax = out_tensor._argmax  # Shape: (B, C, out_h, out_w)
            kh = argmax // k  # Row index within kernel
            kw = argmax % k   # Column index within kernel

            # Accumulate gradients for each batch and channel
            for b in range(B):
                for c in range(C):
                    # Compute input indices where gradients should go
                    input_rows = (np.arange(out_h)[:, None] * s + kh[b, c]).ravel()
                    input_cols = (np.arange(out_w)[None, :] * s + kw[b, c]).ravel()
                    # Accumulate gradients at those positions
                    np.add.at(
                        x.grad[b, c],
                        (input_rows, input_cols),
                        out_tensor.grad[b, c].ravel()
                    )

        out_tensor._backward = _backward
        return out_tensor
        


class CrossEntropyLoss:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        # input: (B, C), raw logits
        # target: (B,), class indices
        B = input.data.shape[0]
        shifted = input - Tensor(np.max(input.data, axis=1, keepdims=True))  # for numerical stability
        exp_shifted = shifted.exp()
        sum_exp = exp_shifted.sum(axis=1, keepdims=True)
        log_probs = shifted - sum_exp.log()

        # Pick log-prob of correct class using advanced indexing
        idx = (np.arange(B), target.data.astype(np.int64))
        nll_data = -log_probs.data[idx]  # shape (B,)
        loss_data = nll_data.mean()
        out = Tensor(np.array(loss_data), _children=(input, target), _op='cross_entropy')

        def _backward():
            grad = np.exp(log_probs.data)
            grad[idx] -= 1  # subtract 1 at the true class
            grad /= B
            input.grad += grad

        out._backward = _backward
        return out


def cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    return CrossEntropyLoss()(input, target)

def softmax(input: Tensor, dim: int) -> Tensor:
    data = input.data
    shifted = data - np.max(data, axis=dim, keepdims=True)
    exp_data = np.exp(shifted)
    exp_sum = exp_data.sum(axis=dim, keepdims=True)
    probs = exp_data / exp_sum
    out = Tensor(probs, _children=(input,), _op='softmax')
    def _backward():
        # Jacobian-vector product for softmax
        grad = out.grad
        s = out.data
        axis = dim
        # For each sample in batch, compute grad = s * (grad - sum(grad * s))
        dx = grad - (grad * s).sum(axis=axis, keepdims=True)
        dx = s * dx
        input.grad += dx
    out._backward = _backward
    return out

class BCELoss:
    """input to this should be sigmod output"""
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        data = np.clip(input.data, self.eps, 1 - self.eps)
        bce = -(target.data * np.log(data) + (1 - target.data) * np.log(1 - data))
        out = Tensor(np.array(bce.mean()), _children=(input, target), _op='bce')

        def _backward():
            grad = (data - target.data) / (data * (1 - data) * target.data.size)
            input.grad += grad.reshape(input.grad.shape)
        out._backward = _backward
        return out

def bce_loss(input: Tensor, target: Tensor) -> Tensor:
    return BCELoss()(input, target)


class MSELoss:
    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        mse = (diff * diff).sum() * (1.0 / input.data.size)
        out = Tensor(mse.data, _children=(input, target), _op='mse')
        def _backward():
            grad = 2 * (input.data - target.data) / input.data.size
            input.grad += grad * out.grad
            # Optionally, target.grad -= grad * out.grad
        out._backward = _backward
        return out

def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    return MSELoss()(input, target)

class RMSELoss:
    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        mse = (diff * diff).sum() * (1.0 / input.data.size)
        rmse = mse ** 0.5
        out = Tensor(rmse.data, _children=(input, target), _op='rmse')
        def _backward():
            grad = (input.data - target.data) / (input.data.size * (rmse.data + 1e-12))
            input.grad += grad * out.grad
            # Optionally, target.grad -= grad * out.grad
        out._backward = _backward
        return out

def rmse_loss(input: Tensor, target: Tensor) -> Tensor:
    return RMSELoss()(input, target)


class LayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, Tuple[int]], eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Tensor(np.ones(normalized_shape))
        self.bias = Tensor(np.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        # Use Tensor's sqrt method for autograd compatibility
        denom = (var + self.eps)
        if not isinstance(denom, Tensor):
            denom = Tensor(denom)
        denom = denom.sqrt()
        normed = (x - mean) / denom
        out_data = normed * self.weight.data + self.bias.data
        out = Tensor(out_data, _children=(x,), _op='layer_norm')

        def _backward():
            grad_normed = out.grad * self.weight.data
            grad_mean = -grad_normed.mean(axis=-1, keepdims=True)
            denom3 = denom.data ** 3
            grad_var = -0.5 * grad_normed * (x.data - mean.data) / denom3
            grad_var += 2 * (x.data - mean.data) * grad_var.mean(axis=-1, keepdims=True)
            x.grad += (grad_normed / denom.data) + grad_mean + grad_var

        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight, self.bias]


class ModuleList(Module):
    def __init__(self, modules: Optional[List[Module]] = None):
        super().__init__()
        self._modules = {}
        if modules is not None:
            for i, module in enumerate(modules):
                self.add_module(f'layer_{i}', module)

    def add_module(self, name: str, module: Module):
        if not isinstance(module, Module):
            raise TypeError(f"Expected a Module instance, got {type(module)}")
        self._modules[name] = module
    
    def __iter__(self):
        return iter(self._modules.values())
    
    def __next__(self):
        return next(iter(self._modules.values()))

    def parameters(self):
        params = []
        for module in self._modules.values():
            params.extend(module.parameters())
        return params


TRAINING = False

def train():
    global TRAINING
    TRAINING = True

def eval():
    global TRAINING
    TRAINING = False

def dropout(x: Tensor, p: float = 0.5) -> Tensor:
    if p < 0 or p >= 1:
        raise ValueError("Dropout probability must be in the range [0, 1).")
    if not TRAINING or p == 0:
        return x  # No dropout, return input as is

    mask = np.random.binomial(1, 1 - p, size=x.data.shape)
    out_data = x.data * mask / (1 - p)  # Scale by keep probability
    out = Tensor(out_data, _children=(x,), _op='dropout')

    def _backward():
        x.grad += out.grad * mask / (1 - p)

    out._backward = _backward
    return out

class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return dropout(x, self.p)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        out_data = np.where(x.data < 0, 0, x.data)
        out = Tensor(out_data, _children=(x,), _op='ReLU')

        def _backward():
            x.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
class Sequential(Module):
    def __init__(self, *args: Module):
        super().__init__()
        self._modules = {f'layer_{i}': layer for i, layer in enumerate(args)}

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._modules.values():
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self._modules.values():
            params.extend(layer.parameters())
        return params
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

