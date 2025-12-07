# Release Notes - nnetflow v2.0.5

## Overview

Version 2.0.5 is a major feature release that introduces GPU support via CuPy, implements the Multi-Head Attention mechanism, and includes significant improvements to device abstraction and dtype handling. This release makes nnetflow production-ready for both CPU and GPU training workflows.

## What's New

### 🚀 GPU Support with CuPy

* **Device Abstraction Module**:
    * New `nnetflow.device` module provides unified interface for CPU (NumPy) and GPU (CuPy) computation
    * Automatic device detection and seamless switching between CPU and GPU
    * Users can install any CuPy version (e.g., `cupy-cuda11x`, `cupy-cuda12x`) without framework constraints
    * All operations automatically use the appropriate backend based on current device

* **GPU Utilities**:
    * `gpu_supports_dtype()` function to check GPU data type support (e.g., float16 for A100)
    * `is_gpu_available()` to detect GPU availability
    * `get_gpu_count()` and `get_gpu_name()` for GPU information
    * `to_numpy()` and `to_cupy()` helpers for array conversion

* **Device-Aware Operations**:
    * All Tensor operations now use device abstraction
    * All layers (Linear, Conv, BatchNorm, etc.) work seamlessly on CPU and GPU
    * Optimizers and loss functions are device-aware
    * No conflicts between NumPy and CuPy when using CPU

### ✨ New Features

* **Multi-Head Attention Layer**:
    * Implemented `MultiHeadAttention` class following "Attention Is All You Need" paper
    * Supports causal masking for autoregressive models
    * Configurable dropout, bias, and sequence length
    * Optimized for both CPU and GPU with pre-computed mask option
    * Fully integrated with autograd system

* **Enhanced Examples**:
    * Updated `gpt2.py` example to automatically use GPU when available
    * Automatic float16 support detection for compatible GPUs
    * Device-aware data processing throughout

### 🛠 Improvements & Fixes

* **Dtype Preservation**:
    * Fixed dtype preservation in Tensor operations (`__add__`, `__matmul__`)
    * Operations now maintain input dtype instead of defaulting to float64
    * Improved compatibility with float32 and float16 workflows

* **Layer Fixes**:
    * Fixed `Conv2d` and `Conv1d` to inherit from `Module` (now properly callable)
    * All convolution tests now passing (14 tests fixed)
    * Improved layer initialization with device-aware operations

* **Code Quality**:
    * All 112 tests passing
    * Comprehensive device abstraction throughout codebase
    * Better error handling for GPU operations

### 📦 API Additions

New exports in `nnetflow`:
- `MultiHeadAttention` - Multi-head attention layer
- `Device` - Device abstraction class
- `get_device()`, `set_device()` - Device management
- `get_array_module()` - Get current array module (NumPy/CuPy)
- `is_gpu_available()`, `get_gpu_count()`, `get_gpu_name()` - GPU utilities
- `gpu_supports_dtype()` - Check GPU data type support
- `to_numpy()`, `to_cupy()` - Array conversion helpers

## Migration Guide

This release is **backwards compatible** for CPU users. GPU support is **optional** and requires CuPy installation.

### For CPU Users
- No changes required - all existing code works as before
- Framework automatically uses NumPy (CPU) backend

### For GPU Users
- Install CuPy for your CUDA version:
  ```bash
  pip install cupy-cuda12x  # For CUDA 12.x
  # or
  pip install cupy-cuda11x  # For CUDA 11.x
  ```
- Set device before training:
  ```python
  from nnetflow import set_device
  set_device('cuda', device_id=0)
  ```
- Check GPU data type support:
  ```python
  from nnetflow import gpu_supports_dtype
  import numpy as np
  if gpu_supports_dtype(np.float16):
      # Use float16 for faster training
  ```

### Breaking Changes
- None - fully backwards compatible

### Deprecations
- None

## Performance

- **GPU Training**: Significant speedup for large models when using GPU
- **Float16 Support**: Automatic detection and use of float16 on compatible GPUs (A100, etc.)
- **Optimized Operations**: All operations use device-optimized backends

## Testing

- All 112 tests passing
- Comprehensive GPU/CPU compatibility verified
- MultiHeadAttention fully tested
- Convolution layers fixed and tested

## Contributors

* Lewis Njue (maintainer)

