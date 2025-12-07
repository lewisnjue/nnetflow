# Changelog

All notable changes to this project are documented in this file.

## [2.0.5] - 2025-12-07

### Added
- **GPU Support**: Complete GPU support via CuPy with device abstraction module
  - New `nnetflow.device` module for CPU/GPU management
  - Automatic device detection and seamless switching
  - Support for any CuPy version (cupy-cuda11x, cupy-cuda12x, etc.)
  - GPU data type support checking (`gpu_supports_dtype()`)
  - GPU information utilities (`get_gpu_count()`, `get_gpu_name()`)
- **MultiHeadAttention Layer**: Full implementation of multi-head attention mechanism
  - Causal masking for autoregressive models
  - Configurable dropout, bias, and sequence length
  - Optimized for both CPU and GPU
  - Pre-computed mask option for performance
- **Device-Aware Operations**: All Tensor operations now use device abstraction
  - All layers work seamlessly on CPU and GPU
  - Optimizers and losses are device-aware
  - No NumPy/CuPy conflicts

### Fixed
- **Dtype Preservation**: Fixed dtype preservation in Tensor operations (`__add__`, `__matmul__`)
  - Operations now maintain input dtype instead of defaulting to float64
- **Layer Callability**: Fixed `Conv2d` and `Conv1d` to inherit from `Module`
  - Both layers are now properly callable
  - All 14 convolution tests now passing
- **Test Failures**: Fixed all test failures (112/112 tests now passing)

### Changed
- **Examples**: Updated `gpt2.py` to automatically use GPU when available
  - Automatic float16 support detection
  - Device-aware data processing
- **Initializers**: Updated to use device abstraction for GPU compatibility
- **Codebase**: Comprehensive device abstraction throughout

### Performance
- GPU training support for significant speedup on large models
- Automatic float16 support on compatible GPUs (A100, etc.)
- Optimized operations using device-optimized backends

##[2.0.4] - 2025-11-21 

### Fixed 
- **Numerical Stability**: `cross_entropy_loss` now uses `log_sofmax` internally to prevent underflow. 
- **Numerical Stability**: ``binary_cross_entropy_loss` now clamps predictions ( default eps=1e-7) to prevent `NaN` on exact 0/1 inputs 
- **Gradient Correctness**:  Fixed inconsistent eplison handing in division backward pass. 
- **BatchNorm**: `BatchNorm1d.parameters()` now correctly returns an empty list when `affine = False` 
- **Packaging**: `MANIFEST.in` now correctly includes `nnetflow` assets  
- **Cleanup**: Removed duplicate imports and stale code in `layers.py` 

### Changed 
- **Logging**: `Tensor.log()` and `Tensor.log10()` now raise `RuntimeWarning` instead of printing 
- **API`**: Resolved naming conflicts between Tensor helper methods and numpy API 


## [2.0.3] - 2025-11-15

### Added
- Simple tanh-based `RNN` layer for sequential data with tests.
- Tests for GELU paths and SciPy-backed activations.

### Fixed
- Tensor helper methods (`shape`, `size`, `ndim`, `numel`, `dim`) now
  delegate correctly to the underlying NumPy array.
- `Tensor.var`/`Tensor.std` now match NumPy sample variance/std
  semantics and support `axis` + `keepdims`.
- `Tensor.view` now accepts both varargs and a tuple shape and forwards
  them correctly to `reshape`.

### Packaging
- Declare `scipy>=1.9` as a runtime dependency in both `pyproject.toml`
  and `setup.py`.

##[2.0.2] - 2025-11-12 

### Added 
- convolution layers (conv1d and conv2d) 
- pooling layers (maxpool1d and maxpool2d )
- test for pooling layers and convolutin layers 
- doc string improvent 

## [2.0.1] - 2025-11-09

### Fixed
- Fixed version handling in `__init__.py` to correctly report version 2.0.1
- Verified and confirmed division backward pass correctness (was already correct)
- Removed duplicate method definitions in BatchNorm1d class

### Added
- Comprehensive test suite for Tensor engine operations (`test_engine.py`)
- Tests for division backward pass with both numerator and denominator gradients
- Numerical gradient checking for division operation
- Pytest configuration in `pyproject.toml` for consistent test execution
- Enhanced API reference documentation in README
- Added `logits_binary_cross_entropy_loss` to package exports

### Changed
- Updated all version references to 2.0.1
- Enhanced README with comprehensive API reference section
- Improved import examples in README to show both direct and module-level imports
- Updated project structure documentation

### Improved
- Better code organization with cleaner exports in `__init__.py`
- More comprehensive test coverage for core Tensor operations
- Enhanced documentation with practical examples

## [2.0.0] - 2025-11-07
### Added
- Major project cleanup and professionalization.
- Comprehensive unit tests for core components (Tensor, Linear layer, losses, optimizers).
- CI updated to run tests on push & pull requests.
- Pre-commit configuration to run `pytest` before pushing.
- Examples updated and documented.
- README rewritten with clearer usage and contribution instructions.

### Changed
- Bumped package version to `2.0.0`.
- Improved packaging metadata in `pyproject.toml` and `setup.py`.

### Fixed
- Addressed various small API inconsistencies and test-suite issues discovered during refactor.


## [0.1.2] - (previous)
- Initial prototype release.
