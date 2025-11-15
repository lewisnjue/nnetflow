from .engine import Tensor
from .layers import Linear, BatchNorm1d, LayerNorm, Embedding, Dropout, Flatten 
from .losses import (
    mse_loss, 
    rmse_loss, 
    cross_entropy_loss, 
    binary_cross_entropy_loss,
    logits_binary_cross_entropy_loss
)
from .optim import SGD, Adam
try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:  # pragma: no cover
    version = None
    PackageNotFoundError = Exception

try:
    __version__ = version("nnetflow") if version is not None else "2.0.3"
except PackageNotFoundError:
    __version__ = "2.0.3"

__all__ = [
    'Tensor',
    '__version__', 
    'Linear', 
    'BatchNorm1d',
    'LayerNorm', 
    'Embedding',
    'Dropout', 
    'Flatten', 
    'mse_loss',
    'rmse_loss', 
    'cross_entropy_loss', 
    'binary_cross_entropy_loss',
    'logits_binary_cross_entropy_loss',
    'SGD', 
    'Adam'
]
