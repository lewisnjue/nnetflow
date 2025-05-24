import numpy as np
from .engine import Tensor
from typing import List, Tuple, Optional, Union

def im2col_2d(arr: np.ndarray, kernel_size: Tuple[int,int], stride: int) -> np.ndarray:
    B, C, H, W = arr.shape
    kH, kW = kernel_size
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    patches = np.zeros((B, C*kH*kW, out_h*out_w))
    for b in range(B):
        patch_idx = 0
        for i in range(0, H - kH + 1, stride):
            for j in range(0, W - kW + 1, stride):
                patch = arr[b, :, i:i+kH, j:j+kW].reshape(C*kH*kW)
                patches[b, :, patch_idx] = patch
                patch_idx += 1
    return patches

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    exp_x = (x - x.max(dim=dim)).exp()
    return exp_x / exp_x.sum(dim=dim)

class Module:
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
        elif hasattr(value, 'parameters') and callable(v.parameters):
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
        B, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1
        out = np.zeros((B, C, out_h, out_w))
        indices = np.zeros((B, C, out_h, out_w), dtype=np.int64)
        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * s
                        h_end = h_start + k
                        w_start = j * s
                        w_end = w_start + k
                        window = x.data[b, c, h_start:h_end, w_start:w_end]
                        out[b, c, i, j] = np.max(window)
                        indices[b, c, i, j] = np.argmax(window)
        out_tensor = Tensor(out, _children=(x,), _op='maxpool2d')
        def _backward():
            x.grad += np.zeros_like(x.data)
            for b in range(B):
                for c in range(C):
                    for i in range(out_h):
                        for j in range(out_w):
                            h_start = i * s
                            w_start = j * s
                            idx = indices[b, c, i, j]
                            ih = idx // k
                            iw = idx % k
                            x.grad[b, c, h_start + ih, w_start + iw] += out_tensor.grad[b, c, i, j]
        out_tensor._backward = _backward
        return out_tensor

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
        for layer in self.layers:
            x = layer(x)
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
