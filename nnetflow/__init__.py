from .engine import Tensor
from .layers import (
    Linear,
    Conv1d,
    Conv2d,
    BatchNorm1d,
    BatchNorm2d,
    LayerNorm,
    Embedding,
    Dropout,
    MCDropout,
    Flatten,
    MaxPool1d,
    MaxPool2d,
    MultiHeadAttention,
    AveragePool2d,
    GlobalAveragePool2d,
)
from .losses import (
    MSELoss,
    RMSELoss,
    CrossEntropyLoss,
)
from .visualize import draw_dot, visualize_model
from .optim import SGD, Adam

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
    'Conv1d',
    'Conv2d',
    'BatchNorm1d',
    'BatchNorm2d',
    'LayerNorm',
    'Embedding',
    'Dropout',
    'MCDropout',
    'Flatten',
    'MaxPool1d',
    'MaxPool2d',
    'MultiHeadAttention',
    'AveragePool2d',
    'GlobalAveragePool2d',
    'MSELoss',
    'RMSELoss',
    'CrossEntropyLoss',
    'SGD',
    'Adam',
    'draw_dot',
    'visualize_model',
]
