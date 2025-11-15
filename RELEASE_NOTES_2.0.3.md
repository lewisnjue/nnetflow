# Release Notes - nnetflow v2.0.3

## Overview

Version 2.0.3 is a small but important maintenance release that
focuses on numerical correctness, API consistency, and packaging
fixes. It also introduces a simple RNN layer for sequential data.

## What's New

### ✨ New Features
- **RNN layer for sequential data**
  - Added `nnetflow.layers.RNN`, a simple tanh-based RNN layer that
    handles inputs of shape `(batch_size, time_steps, n_features)`.
  - Supports `return_sequence` to return either the full sequence of
    hidden states or just the final hidden state.
  - Includes tests comparing forward outputs to PyTorch `nn.RNN` and
    verifying gradients.

### 🧮 Numerical & API Fixes
- **Tensor helper methods**
  - Fixed `Tensor.shape`, `Tensor.size`, and `Tensor.ndim` by exposing
    them as properties delegating to the underlying NumPy array.
  - Re-implemented `Tensor.numel()` and `Tensor.dim()` to use
    `self.data.size` and `self.data.ndim`, avoiding reliance on
    non-existent NumPy methods.

- **Variance and standard deviation**
  - Updated `Tensor.var` and `Tensor.std` to:
    - Support `axis=None`, single axis, and tuples of axes.
    - Respect `keepdims` semantics aligned with NumPy.
    - Compute *sample* variance/std with `ddof=1` (denominator `N-1`).
  - Added tests that compare results against NumPy for multiple axis
    and `keepdims` combinations.

- **Tensor.view semantics**
  - Fixed `Tensor.view` so it behaves like `reshape` and correctly
    unpacks shape arguments:
    - Supports both `x.view(2, 3)` and `x.view((2, 3))`.
    - Preserves `-1` inference behavior via the underlying `reshape`.
  - Added tests to ensure parity with NumPy `reshape`.

### 📦 Packaging
- **SciPy dependency for GELU**
  - `Tensor.gelu` and `nnetflow.experimental.activations.GELU` depend
    on `scipy.special.erf`. This release explicitly declares SciPy as a
    runtime dependency:
    - Added `scipy>=1.9` to `project.dependencies` in `pyproject.toml`.
    - Added `scipy>=1.9` to `install_requires` in `setup.py`.
  - New tests exercise both GELU paths so a missing SciPy installation
    would be caught during testing.

## Migration Guide

This release is backwards compatible for most users.

- If you use `Tensor.var` or `Tensor.std`:
  - Results now match NumPy's *sample* variance/std (`ddof=1`) instead
    of using a simple `N` denominator. This is a numerically more
    standard choice but may slightly change reported values.
  - The API now accepts `axis=None | int | tuple[int, ...]` and
    supports `keepdims` consistently.

- If you use `Tensor.view`:
  - You can now safely call `x.view((2, 3))` in addition to
    `x.view(2, 3)`. Existing correct usages will continue to work.

- If you rely on GELU or experimental activations:
  - Ensure `scipy>=1.9` is available in your environment. Installing
    `nnetflow` via the updated packaging metadata will pull this in
    automatically.

No breaking API removals were introduced in v2.0.3.
