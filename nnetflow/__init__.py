from .engine import Tensor
from .layers import Linear, BatchNorm1d, LayerNorm, Embedding, Dropout, Flatten, MultiHeadAttention 
from .losses import (
    mse_loss, 
    rmse_loss, 
    cross_entropy_loss, 
    binary_cross_entropy_loss,
    logits_binary_cross_entropy_loss
)
from .optim import SGD, Adam
from .device import (
    Device,
    get_device,
    set_device,
    get_array_module,
    is_gpu_available,
    get_gpu_count,
    get_gpu_name,
    gpu_supports_dtype,
    to_numpy,
    to_cupy
)
try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:  # pragma: no cover
    version = None
    PackageNotFoundError = Exception

try:
    __version__ = version("nnetflow") if version is not None else "2.0.5"
except PackageNotFoundError:
    __version__ = "2.0.5"

__all__ = [
    'Tensor',
    '__version__', 
    'Linear', 
    'BatchNorm1d',
    'LayerNorm', 
    'Embedding',
    'Dropout', 
    'Flatten',
    'MultiHeadAttention', 
    'mse_loss',
    'rmse_loss', 
    'cross_entropy_loss', 
    'binary_cross_entropy_loss',
    'logits_binary_cross_entropy_loss',
    'SGD', 
    'Adam',
    'Device',
    'get_device',
    'set_device',
    'get_array_module',
    'is_gpu_available',
    'get_gpu_count',
    'get_gpu_name',
    'gpu_supports_dtype',
    'to_numpy',
    'to_cupy'
]
