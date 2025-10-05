import importlib

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

# Try to import the C++/CUDA backend
try:
    tensor_cuda = importlib.import_module('tensor_cuda')
    _HAS_CUDA_BACKEND = True
except ImportError:
    tensor_cuda = None
    _HAS_CUDA_BACKEND = False

def is_cuda_available() -> bool:
    """Check if CUDA is available using the C++/CUDA backend or CuPy."""
    # First check C++/CUDA backend
    if _HAS_CUDA_BACKEND:
        try:
            return tensor_cuda.Device.detect_best().type == tensor_cuda.DeviceType.CUDA
        except Exception:
            pass
    # Fallback to CuPy detection
    if _HAS_CUPY:
        try:
            cp.cuda.runtime.getDeviceCount()
            return True
        except Exception:
            pass
    return False
