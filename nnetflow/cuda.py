# nnetflow/cuda.py
"""
CuPy-based GPU device management for nnetflow.
Provides is_available(), device context, and device string API.
"""
try:
    import cupy as cp
    _cupy_available = True
except ImportError:
    cp = None
    _cupy_available = False

import numpy as np

class Device:
    def __init__(self, device_str):
        assert device_str in ('cpu', 'cuda'), f"Unknown device: {device_str}"
        self.type = device_str
    def __str__(self):
        return self.type
    def __eq__(self, other):
        if isinstance(other, Device):
            return self.type == other.type
        return self.type == other

def is_available():
    return _cupy_available

def get_array_module(device):
    if device == 'cuda' and _cupy_available:
        return cp
    return np

def as_device_array(data, device):
    if device == 'cuda' and _cupy_available:
        return cp.asarray(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.asarray(data)

# Global default device
_default_device = Device('cuda' if _cupy_available else 'cpu')

def get_default_device():
    return _default_device

def set_default_device(device_str):
    global _default_device
    _default_device = Device(device_str)
