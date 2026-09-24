from .engine import Tensor
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
    'MSELoss',
    'RMSELoss', 
    'CrossEntropyLoss', 
    'SGD', 
    'Adam',
    'draw_dot',
    'visualize_model'
]
