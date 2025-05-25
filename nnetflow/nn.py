import numpy as np
from .engine import Tensor
from typing import List, Tuple, Optional, Union
from numpy.lib.stride_tricks import as_strided
from . import cuda


def im2col_2d(arr, kernel_size: Tuple[int,int], stride: int):
    # Ensure arr is a numpy array for as_strided
    import numpy as np
    if hasattr(arr, 'get'):
        arr = arr.get()  # Convert CuPy array to NumPy array for as_strided
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

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    exp_x = (x - x.data.max(dim=dim)).exp() 
    return exp_x / exp_x.sum(dim=dim)

class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = []
        self.device = None

    def to(self, device):
        device = cuda.Device(device) if not isinstance(device, cuda.Device) else device
        self.device = device
        # Move all submodules
        for module in self._modules.values():
            module.to(device)
        # Move all parameters
        for idx, param in enumerate(self._parameters):
            if hasattr(param, 'to'):
                self._parameters[idx] = param.to(device)
        # Move parameters in lists/tuples
        for attr, value in self.__dict__.items():
            if isinstance(value, (list, tuple)):
                new_list = []
                for item in value:
                    if hasattr(item, 'to'):
                        new_list.append(item.to(device))
                    else:
                        new_list.append(item)
                if isinstance(value, list):
                    setattr(self, attr, new_list)
                else:
                    setattr(self, attr, tuple(new_list))
        return self

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

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, activation=None, device=None):
        super().__init__()
        self.device = device or cuda.get_default_device()
        std = 1 / np.sqrt(in_features) if activation in ['tanh', None] else np.sqrt(2.0 / in_features)
        w = np.random.randn(in_features, out_features) * std
        self.weight = Tensor(w, device=self.device)
        self.bias = Tensor(np.zeros(out_features), device=self.device) if bias else None
        self.activation = activation

    def __call__(self, x):
        if hasattr(x, 'device') and x.device != self.device:
            x = x.to(self.device)
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, device=None):
        super().__init__()
        self.device = device or cuda.get_default_device()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        std = np.sqrt(2.0 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = Tensor(np.random.randn(out_channels, in_channels, *kernel_size) * std, device=self.device)
        self.bias = Tensor(np.zeros(out_channels), device=self.device) if bias else None

    def __call__(self, x):
        if hasattr(x, 'device') and x.device != self.device:
            x = x.to(self.device)
        if self.padding > 0:
            xp = cuda.get_array_module(self.device)
            pad_fn = getattr(xp, 'pad', np.pad)  # fallback to np.pad if not present
            x_padded = Tensor(pad_fn(x.data, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant'), _children=(x,), _op='pad', device=self.device)
            def _pad_backward():
                x.grad += x_padded.grad[:, :, self.padding:-self.padding, self.padding:-self.padding]
            x_padded._backward = _pad_backward
        else:
            x_padded = x
        B, C, H, W = tuple(x_padded.data.shape)
        kH, kW = self.kernel_size
        out_h = (H - kH) // self.stride + 1
        out_w = (W - kW) // self.stride + 1
        cols_data = im2col_2d(x_padded.data, self.kernel_size, self.stride)
        cols = Tensor(cols_data.reshape(B, C * kH * kW, out_h * out_w), _children=(x_padded,), _op='im2col', device=self.device)
        weight_reshaped = self.weight.reshape(1, self.out_channels, C * kH * kW)
        out = weight_reshaped @ cols  # (B, out_channels, out_h*out_w)
        out = out.reshape(B, self.out_channels, out_h, out_w)
        if self.bias is not None:
            bias_reshaped = self.bias.reshape(1, self.out_channels, 1, 1)
            out = out + bias_reshaped
        return out

    def parameters(self):
        return [self.weight, self.bias] if self.bias is not None else [self.weight]


class MaxPool2D(Module):
    def __init__(self, kernel_size: int, stride: Optional[int] = None, device=None):
        super().__init__()
        self.device = device or cuda.get_default_device()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def __call__(self, x: Tensor) -> Tensor:
        if hasattr(x, 'device') and x.device != self.device:
            x = x.to(self.device)
        B, C, H, W = tuple(x.data.shape)
        k = self.kernel_size
        s = self.stride
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1
        # Ensure x.data is a numpy array for as_strided
        arr = x.data
        if hasattr(arr, 'get'):
            arr = arr.get()
        shape = (B, C, out_h, out_w, k, k)
        strides = (
            arr.strides[0],           # batch step
            arr.strides[1],           # channel step
            arr.strides[2] * s,       # window row step
            arr.strides[3] * s,       # window col step
            arr.strides[2],           # within-window row
            arr.strides[3],           # within-window col
        )
        windows = as_strided(arr, shape=shape, strides=strides)
        out_data = windows.max(axis=(4, 5))
        argmax = windows.reshape(B, C, out_h, out_w, k * k).argmax(axis=4)
        out_tensor = Tensor(out_data, _children=(x,), _op='maxpool2d', device=self.device)
        out_tensor._argmax = argmax
        out_tensor._k = k
        out_tensor._s = s
        def _backward():
            k = out_tensor._k
            s = out_tensor._s
            B, C, out_h, out_w = out_tensor.data.shape
            import numpy as np
            if x.grad is None:
                x.grad = np.zeros_like(x.data, dtype=out_tensor.grad.dtype)
            argmax = out_tensor._argmax
            kh = argmax // k
            kw = argmax % k
            for b in range(B):
                for c in range(C):
                    input_rows = (np.arange(out_h)[:, None] * s + kh[b, c]).ravel()
                    input_cols = (np.arange(out_w)[None, :] * s + kw[b, c]).ravel()
                    np.add.at(
                        x.grad[b, c],
                        (input_rows, input_cols),
                        out_tensor.grad[b, c].ravel()
                    )
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
    return Tensor(probs)

class Softmax:
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, input: Tensor) -> Tensor:
        return softmax(input, self.dim)

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
