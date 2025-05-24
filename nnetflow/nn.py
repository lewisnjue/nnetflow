import random
import numpy as np
from nnetflow.engine import Tensor

class Module:
    def __init__(self):
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, '_parameters', [])

    def __setattr__(self, name, value):
        if '_modules' not in self.__dict__:
            object.__setattr__(self, '_modules', {})
        if '_parameters' not in self.__dict__:
            object.__setattr__(self, '_parameters', [])
        if isinstance(value, Module):
            self.__dict__['_modules'][name] = value
        elif isinstance(value, (list, tuple)):
            # Recursively register modules/parameters in lists/tuples
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
        # Also check for lists/tuples of parameters
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
        raise NotImplementedError("Module subclasses must implement forward()")

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True, dtype=None, activation=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.activation = activation
        # Use PyTorch-like Kaiming (He) initialization for ReLU, Xavier for tanh
        if activation == 'relu':
            std = np.sqrt(2.0 / in_features)
        elif activation == 'tanh':
            std = np.sqrt(1.0 / in_features)
        else:
            std = np.sqrt(2.0 / (in_features + out_features))
        weight = np.random.randn(out_features, in_features) * std
        if dtype:
            weight = weight.astype(dtype)
        self.weight = Tensor(weight.T)
        if bias:
            b = np.zeros(out_features, dtype=dtype if dtype else float)
            self.bias = Tensor(b)
        else:
            self.bias = None

    def __call__(self, x: Tensor):
        # Accept both 1D and 2D input, handle scalar edge case
        shape = x.data.shape
        if not shape or shape == ():
            raise ValueError("Input tensor has no shape (scalar), expected at least 1D array.")
        if len(shape) == 1:
            if shape[0] != self.in_features:
                raise ValueError(f"Shape mismatch: got {shape}, expected ({self.in_features},)")
        else:
            if shape[-1] != self.in_features:
                raise ValueError(f"Shape mismatch: got {shape}, expected (..., {self.in_features})")
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        if self.activation == 'relu':
            out = out.relu()
        elif self.activation == 'tanh':
            out = out.tanh()
        return out

class CrossEntropyLoss:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        # Numerically stable log-softmax
        max_logits = Tensor(np.max(input.data, axis=-1, keepdims=True))
        shifted = input - max_logits
        exp_shifted = shifted.exp()
        sum_exp = exp_shifted.sum(axis=-1, keepdims=True)
        logsumexp = sum_exp.log()
        log_probs = shifted - logsumexp
        nll = -(target * log_probs).sum(axis=-1)
        # Fix for scalar input
        if nll.data.shape == ():
            loss = nll
        else:
            loss = nll.sum() * (1.0 / nll.data.shape[0])
        out = Tensor(np.array(loss.data), _children=(input, target), _op='cross_entropy')

        def _backward():
            # Gradient: softmax(input) - target
            exps = np.exp(input.data - np.max(input.data, axis=-1, keepdims=True))
            softmax = exps / np.sum(exps, axis=-1, keepdims=True)
            grad = (softmax - target.data).astype(np.float32)
            grad = np.broadcast_to(grad, input.grad.shape)
            # Divide by batch size to match PyTorch's reduction (mean)
            batch_size = input.data.shape[0] if len(input.data.shape) > 0 else 1
            input.grad += grad / batch_size
            # No grad for target
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
    return Tensor(probs)

class Softmax:
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, input: Tensor) -> Tensor:
        return softmax(input, self.dim)

class BCELoss:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        # Ensure input is in (0,1) for log
        data = np.clip(input.data, self.eps, 1 - self.eps)
        # dL/dx = (x - y) / (x * (1-x))
        bce = -(target.data * np.log(data) + (1 - target.data) * np.log(1 - data))
        out = Tensor(np.array(bce.mean()), _children=(input, target), _op='bce')

        def _backward():
            # Gradient w.r.t. input: (input - target) / (input * (1-input) * N)
            grad = (data - target.data) / (data * (1 - data) * target.data.size)
            input.grad += grad.reshape(input.grad.shape)
            # No grad for target (labels)
        out._backward = _backward
        return out

def bce_loss(input: Tensor, target: Tensor) -> Tensor:
    return BCELoss()(input, target)

# === Regression Losses ===

class MSELoss:
    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        mse = (diff * diff).sum() * (1.0 / input.data.size)
        return mse

def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    return MSELoss()(input, target)

class RMSELoss:
    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        mse = (diff * diff).sum() * (1.0 / input.data.size)
        rmse = mse ** 0.5
        return rmse

def rmse_loss(input: Tensor, target: Tensor) -> Tensor:
    return RMSELoss()(input, target)

class MLP(Module):
    def __init__(self, nin, nouts, activation='relu', last_activation=None):
        super().__init__()
        self.layers = []
        sz = [nin] + nouts
        for i in range(len(nouts)):
            act = activation if i < len(nouts) - 1 else last_activation
            self.layers.append(Linear(sz[i], sz[i+1], activation=act))
        self.last_activation = last_activation

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        # Apply last activation if specified
        if self.last_activation == 'sigmoid':
            x = x.sigmoid()
        elif self.last_activation == 'softmax':
            x = softmax(x, dim=-1)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

def im2col_1d(x, kernel_size, stride):
    # x: (batch, channels, width)
    B, C, W = x.shape
    out_w = (W - kernel_size) // stride + 1
    shape = (B, C, kernel_size, out_w)
    strides = (
        x.strides[0],
        x.strides[1],
        x.strides[2],
        x.strides[2] * stride,
    )
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

class Conv1D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        std = np.sqrt(2.0 / (in_channels * kernel_size))
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size) * std)
        self.bias = Tensor(np.zeros(out_channels)) if bias else None

    def __call__(self, x):
        # x: Tensor with shape (batch, in_channels, width)
        x_data = x.data
        if self.padding > 0:
            x_data = np.pad(x_data, ((0,0), (0,0), (self.padding, self.padding)), mode='constant')
        B, C, W = x_data.shape
        cols = im2col_1d(x_data, self.kernel_size, self.stride)  # shape: (B, C, K, L)
        cols = cols.reshape(B, -1, cols.shape[-1])               # (B, C*K, L)
        weight_reshaped = self.weight.data.reshape(self.out_channels, -1)  # (out_ch, C*K)
        out = np.matmul(weight_reshaped[None, :, :], cols)       # (B, out_ch, L)
        if self.bias is not None:
            out += self.bias.data[None, :, None]
        return Tensor(out)

def im2col_2d(x, kernel_size, stride):
    # x: (batch, channels, height, width)
    B, C, H, W = x.shape
    kH, kW = kernel_size
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    shape = (B, C, kH, kW, out_h, out_w)
    strides = (
        x.strides[0],
        x.strides[1],
        x.strides[2],
        x.strides[3],
        x.strides[2] * stride,
        x.strides[3] * stride,
    )
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

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
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) * std)
        self.bias = Tensor(np.zeros(out_channels)) if bias else None

    def __call__(self, x):
        # x: Tensor with shape (batch, in_channels, height, width)
        x_data = x.data
        if self.padding > 0:
            x_data = np.pad(x_data, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        B, C, H, W = x_data.shape
        kH, kW = self.kernel_size
        out_h = (H - kH) // self.stride + 1
        out_w = (W - kW) // self.stride + 1
        cols = im2col_2d(x_data, self.kernel_size, self.stride)  # (B, C, kH, kW, out_h, out_w)
        cols = cols.reshape(B, -1, out_h * out_w)  # (B, C*kH*kW, out_h*out_w)
        weight_reshaped = self.weight.data.reshape(self.out_channels, -1)  # (out_ch, C*kH*kW)
        out = np.matmul(weight_reshaped[None, :, :], cols)  # (B, out_ch, out_h*out_w)
        out = out.reshape(B, self.out_channels, out_h, out_w)  # (B, out_ch, out_h, out_w)
        if self.bias is not None:
            out += self.bias.data[None, :, None, None]
        return Tensor(out)
    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

def im2col_pool2d(x, kernel_size, stride):
    # x: (batch, channels, height, width)
    B, C, H, W = x.shape
    kH, kW = kernel_size
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    shape = (B, C, kH, kW, out_h, out_w)
    strides = (
        x.strides[0],
        x.strides[1],
        x.strides[2],
        x.strides[3],
        x.strides[2] * stride,
        x.strides[3] * stride,
    )
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

class MaxPool1D:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
    def __call__(self, x):
        # x: (batch, channels, width)
        batch, ch, width = x.data.shape
        out_width = (width - self.kernel_size) // self.stride + 1
        out = np.zeros((batch, ch, out_width))
        for b in range(batch):
            for c in range(ch):
                for i in range(out_width):
                    out[b, c, i] = np.max(x.data[b, c, i*self.stride:i*self.stride+self.kernel_size])
        return Tensor(out)

class MaxPool2D:
    def __init__(self, kernel_size, stride=None):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size[0]
    def __call__(self, x):
        # x: (batch, channels, height, width)
        x_data = x.data
        kH, kW = self.kernel_size
        B, C, H, W = x_data.shape
        out_h = (H - kH) // self.stride + 1
        out_w = (W - kW) // self.stride + 1
        cols = im2col_pool2d(x_data, self.kernel_size, self.stride)  # (B, C, kH, kW, out_h, out_w)
        cols = cols.reshape(B, C, kH * kW, out_h, out_w)
        out = np.max(cols, axis=2)  # (B, C, out_h, out_w)
        return Tensor(out)
