def is_cuda_available() -> bool:
import importlib

def is_cuda_available() -> bool:
    """Check if CUDA is available using the C++/CUDA backend."""
    try:
        tensor_cuda = importlib.import_module('tensor_cuda')
        return tensor_cuda.Device.detect_best().type == tensor_cuda.DeviceType.CUDA
    except Exception as e:
        print(f"CUDA check failed: {e}")
        return False
