"""
Device and backend abstraction for CPU/GPU support.

This module provides a unified interface for NumPy (CPU) and CuPy (GPU) operations,
allowing users to seamlessly switch between CPU and GPU computation.
"""

import os
from typing import Optional, Union, Literal, Any
import warnings

# Try to import CuPy, but don't fail if it's not available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

import numpy as np
import numpy.typing as npt


# Global device state
_current_device: Optional['Device'] = None


class Device:
    """
    Device abstraction for CPU/GPU computation.
    
    This class provides a unified interface that works with both NumPy (CPU)
    and CuPy (GPU) backends.
    """
    
    def __init__(self, device_type: Literal['cpu', 'cuda'] = 'cpu', device_id: int = 0):
        """
        Initialize a device.
        
        Args:
            device_type: 'cpu' for CPU computation, 'cuda' for GPU computation
            device_id: Device ID for CUDA devices (ignored for CPU)
        
        Raises:
            ValueError: If device_type is 'cuda' but CuPy is not available
        """
        if device_type == 'cuda':
            if not CUPY_AVAILABLE:
                raise ValueError(
                    "CuPy is not available. Please install CuPy with the appropriate "
                    "CUDA version (e.g., pip install cupy-cuda12x)."
                )
            self.device_type = 'cuda'
            self.device_id = device_id
            self._xp = cp  # CuPy
            # Set the current CUDA device
            cp.cuda.Device(device_id).use()
        else:
            self.device_type = 'cpu'
            self.device_id = 0
            self._xp = np  # NumPy
    
    @property
    def xp(self):
        """
        Returns the array module (NumPy or CuPy) for this device.
        Use this for all array operations.
        """
        return self._xp
    
    def __repr__(self) -> str:
        if self.device_type == 'cuda':
            return f"Device(type='cuda', device_id={self.device_id})"
        return "Device(type='cpu')"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Device):
            return False
        return self.device_type == other.device_type and self.device_id == other.device_id


def get_device() -> Device:
    """
    Get the current global device.
    
    Returns:
        The current Device instance
    """
    global _current_device
    if _current_device is None:
        # Default to CPU if no device is set
        _current_device = Device('cpu')
    return _current_device


def set_device(device: Union[Device, str, Literal['cpu', 'cuda']], device_id: int = 0) -> Device:
    """
    Set the global device for computation.
    
    Args:
        device: Either a Device instance, or 'cpu'/'cuda' string
        device_id: Device ID for CUDA devices (only used if device is a string)
    
    Returns:
        The Device instance that was set
    
    Examples:
        >>> set_device('cpu')  # Use CPU
        >>> set_device('cuda', device_id=0)  # Use GPU 0
        >>> set_device(Device('cuda', 1))  # Use GPU 1
    """
    global _current_device
    
    if isinstance(device, Device):
        _current_device = device
    elif isinstance(device, str):
        if device.lower() == 'cpu':
            _current_device = Device('cpu')
        elif device.lower() == 'cuda':
            _current_device = Device('cuda', device_id)
        else:
            raise ValueError(f"Unknown device type: {device}. Use 'cpu' or 'cuda'.")
    else:
        raise TypeError(f"device must be a Device instance or string, got {type(device)}")
    
    return _current_device


def get_array_module():
    """
    Get the current array module (NumPy or CuPy) based on the current device.
    
    Returns:
        numpy or cupy module
    
    Examples:
        >>> xp = get_array_module()
        >>> arr = xp.array([1, 2, 3])
    """
    return get_device().xp


def is_gpu_available() -> bool:
    """
    Check if GPU (CuPy) is available.
    
    Returns:
        True if CuPy is available, False otherwise
    """
    if not CUPY_AVAILABLE:
        return False
    
    try:
        # Try to get device count
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def get_gpu_count() -> int:
    """
    Get the number of available GPU devices.
    
    Returns:
        Number of GPUs available, or 0 if no GPUs or CuPy not available
    """
    if not CUPY_AVAILABLE:
        return 0
    
    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def get_gpu_name(device_id: int = 0) -> Optional[str]:
    """
    Get the name of a GPU device.
    
    Args:
        device_id: GPU device ID
    
    Returns:
        GPU name string, or None if not available
    """
    if not CUPY_AVAILABLE:
        return None
    
    try:
        with cp.cuda.Device(device_id):
            return cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode('utf-8')
    except Exception:
        return None


def gpu_supports_dtype(dtype: npt.DTypeLike, device_id: int = 0) -> bool:
    """
    Check if a GPU supports a given data type.
    
    This function checks if the GPU has compute capability to support
    certain data types like float16 (fp16), bfloat16, etc.
    
    Args:
        dtype: The data type to check (e.g., np.float16, np.float32)
        device_id: GPU device ID to check
    
    Returns:
        True if the GPU supports the dtype, False otherwise
    
    Examples:
        >>> # Check if GPU supports float16 (useful for A100, etc.)
        >>> if gpu_supports_dtype(np.float16):
        ...     print("GPU supports float16!")
    """
    if not CUPY_AVAILABLE:
        return False
    
    try:
        with cp.cuda.Device(device_id):
            # Get compute capability
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            major = props['major']
            minor = props['minor']
            compute_capability = major * 10 + minor
            
            # Convert dtype to string for comparison
            dtype_str = str(np.dtype(dtype))
            
            # Check support based on compute capability
            # Float16 (fp16) is supported on compute capability 5.3+ (Pascal and later)
            if dtype_str in ['float16', 'half']:
                return compute_capability >= 53
            
            # BFloat16 is supported on compute capability 8.0+ (Ampere and later)
            if dtype_str in ['bfloat16']:
                return compute_capability >= 80
            
            # Float32 and Float64 are supported on all modern GPUs
            if dtype_str in ['float32', 'float64', 'double']:
                return True
            
            # For other dtypes, assume supported if we can create an array
            try:
                test_arr = cp.array([1.0], dtype=dtype)
                return True
            except Exception:
                return False
                
    except Exception:
        return False


def to_numpy(array: Any) -> np.ndarray:
    """
    Convert a CuPy array to NumPy array, or return NumPy array as-is.
    
    Args:
        array: NumPy or CuPy array
    
    Returns:
        NumPy array
    
    This is useful when you need to convert GPU arrays back to CPU
    for operations that don't support GPU (e.g., some scipy functions).
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def to_cupy(array: Any, device_id: int = 0) -> Any:
    """
    Convert a NumPy array to CuPy array, or return CuPy array as-is.
    
    Args:
        array: NumPy or CuPy array
        device_id: Target GPU device ID
    
    Returns:
        CuPy array (if CuPy available), otherwise NumPy array
    
    Raises:
        ValueError: If CuPy is not available
    """
    if not CUPY_AVAILABLE:
        raise ValueError("CuPy is not available. Cannot convert to CuPy array.")
    
    if isinstance(array, cp.ndarray):
        return array
    return cp.asarray(array)


# Initialize default device to CPU
_current_device = Device('cpu')

