from .engine import Tensor
try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:  # pragma: no cover
    version = None
    PackageNotFoundError = Exception

try:
    __version__ = version("nnetflow") if version is not None else "0.0.0"
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ['Tensor', '__version__']
