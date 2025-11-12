# Changelog

All notable changes to this project are documented in this file.


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
