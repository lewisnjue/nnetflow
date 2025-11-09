# Release Notes - nnetflow v2.0.1

## Overview

Version 2.0.1 is a bug fix and improvement release that enhances the stability, testing, and documentation of the nnetflow deep learning framework.

## What's New

### 🐛 Bug Fixes
- **Version Handling**: Fixed version reporting in `__init__.py` to correctly display version 2.0.1
- **Code Cleanup**: Removed duplicate method definitions in `BatchNorm1d` class
- **Verified Correctness**: Confirmed division backward pass implementation is correct

### ✨ New Features
- **Comprehensive Test Suite**: Added `test_engine.py` with extensive tests for Tensor operations
- **Division Gradient Tests**: Added tests for division backward pass covering both numerator and denominator gradients
- **Numerical Gradient Checking**: Implemented numerical gradient verification for division operation
- **Pytest Configuration**: Added pytest settings to `pyproject.toml` for consistent test execution
- **Enhanced Exports**: Added `logits_binary_cross_entropy_loss` to package exports

### 📚 Documentation Improvements
- **API Reference**: Added comprehensive API reference section to README
- **Import Examples**: Updated examples to show both direct and module-level imports
- **Better Structure**: Improved project structure documentation

### 🔧 Code Quality
- **Better Organization**: Cleaner exports in `__init__.py` with proper formatting
- **Test Coverage**: Expanded test coverage for core Tensor operations
- **Code Consistency**: Improved code organization and consistency

## Migration Guide

No breaking changes in this release. All existing code should work without modification.

### Updated Imports (Optional)

You can now use more convenient direct imports:

```python
# New style (recommended)
from nnetflow import Tensor, Linear, mse_loss, Adam

# Old style (still works)
from nnetflow.engine import Tensor
from nnetflow.layers import Linear
from nnetflow.losses import mse_loss
from nnetflow.optim import Adam
```

## Testing

All tests pass successfully. The test suite now includes:
- Tensor creation and basic operations
- Arithmetic operations (addition, multiplication, division)
- Backward pass and gradient computation
- Division backward pass (numerator and denominator)
- Numerical gradient checking
- Activation functions
- Reduction operations
- Matrix multiplication

## Installation

```bash
pip install nnetflow==2.0.1
```

Or upgrade from a previous version:

```bash
pip install --upgrade nnetflow
```

## Full Changelog

See `CHANGELOG.md` for complete details of all changes.

## Contributors

- Lewis Njue (maintainer)

## Next Steps

Future releases will focus on:
- Additional layer types
- Performance optimizations
- Extended documentation
- More examples and tutorials

---

**Release Date**: November 9, 2025  
**Version**: 2.0.1  
**Python Support**: 3.8+

