# Release Notes - nnetflow v2.0.4

## Overview

Version 2.0.4 is a comprehensive maintenance release focused on numerical stability, API hygiene, and packaging correctness. It eliminates potential NaN sources in loss functions and cleans up the user experience by replacing debug prints with proper Python warnings.

## What's New

### 🧮 Numerical Stability & Correctness

* **Robust Cross Entropy**:
    * `cross_entropy_loss` now internally uses `log_softmax` (Log-Sum-Exp trick) instead of `softmax().log()`. This prevents numerical underflow when probabilities are near zero.
* **NaN-Safe Binary Cross Entropy**:
    * `binary_cross_entropy_loss` now clamps predictions to the range `[eps, 1-eps]` (default `1e-7`) before taking logarithms. This prevents infinite loss values when models predict exact `0.0` or `1.0`.
* **Gradient Consistency**:
    * Fixed inconsistent epsilon handling in the division backward pass, ensuring smoother gradient propagation.

### 🛠 API & Core Engine

* **Quieter Logging**:
    * `Tensor.log()` and `Tensor.log10()` now emit standard `RuntimeWarning` instead of printing to stdout when handling non-positive values. This allows users to filter or catch these warnings programmatically.
* **BatchNorm Logic**:
    * `BatchNorm1d` now correctly handles `affine=False`. Calling `.parameters()` on such layers now returns an empty list instead of erroneously returning non-learnable weights.
* **Numpy Compatibility**:
    * Resolved naming conflicts where `Tensor` helper methods were shadowing native numpy attributes, improving interoperability.

### 📦 Packaging & Housekeeping

* **MANIFEST Fixes**:
    * Corrected the `MANIFEST.in` file to properly reference the `nnetflow` package directory, ensuring all non-code assets are included in source distributions.
* **Code Hygiene**:
    * Removed duplicate imports and stale commented-out code from `layers.py`.
    * Removed empty `.gitmodules` files to clean up the repository structure.

## Migration Guide

This release is backwards compatible. However, you may notice the following changes in behavior:

* **Loss Values**: You might observe slightly different (more accurate) loss values in `cross_entropy_loss` due to the improved numerical precision of `log_softmax`.
* **Warnings**: If your code relies on capturing stdout to check for "log applied to non-positive elements" messages, you should update it to catch `RuntimeWarning` instead.

## Contributors

* Lewis Njue (maintainer)