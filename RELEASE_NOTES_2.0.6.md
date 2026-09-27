# Release Notes - nnetflow v2.0.6

## Overview

Version 2.0.6 is a significant release focused on **simplicity over scope**. GPU/CuPy support, which was introduced in 2.0.5, has been removed. nnetflow is meant to be a small, readable tool for learning how autodiff and neural networks work under the hood — not a production framework — and a device-abstraction layer added complexity that worked against that goal. This release also adds a large set of new layers, a proper `Module` base class for building and persisting models, and a graph-visualization utility.

## What's New

### 🔻 Removed: GPU / CuPy support

* Dropped the `nnetflow.device` module and all CuPy-based device abstraction that was introduced in 2.0.5.
* All tensor operations are now plain NumPy/SciPy only, with no device-switching logic anywhere in the codebase.
* This is a deliberate simplification, not an oversight: nnetflow is meant to stay small enough to read end-to-end. If you need GPU acceleration, reach for PyTorch or JAX instead.
* No functional regression for CPU users — everything that worked before continues to work.

### ✨ New Layers

* **Convolutions**: `Conv1d`, `Conv2d` — strided, padded convolutions with a vectorized (as-strided) forward pass and a scatter-add backward pass, verified against `torch.nn.Conv1d` / `Conv2d`.
* **Pooling**: `MaxPool1d`, `MaxPool2d`, `AveragePool2d`, `GlobalAveragePool2d` — verified against PyTorch's `max_pool1d` / `max_pool2d`.
* **Normalization**: `BatchNorm1d` (2D/3D input), `BatchNorm2d`, `LayerNorm` — all matched numerically against their PyTorch counterparts, including running-statistics updates with Bessel's correction.
* **Embedding**: `Embedding` lookup table with correct duplicate-index gradient accumulation.
* **Regularization**: `Dropout` (inverted dropout) and `MCDropout` (dropout that stays active at inference time for Monte-Carlo estimates).
* **Utility**: `Flatten` for reshaping `(batch, ...)` → `(batch, -1)`.
* **Attention**: `MultiHeadAttention` with causal masking, dropout, optional QKV bias, and a simple KV-cache for autoregressive decoding.

### 🧱 New: `Module` base class

* Added `nnetflow.module.Module`, the base class every layer and model now inherits from.
* `parameters()` recursively collects every `Tensor` with `requires_grad=True` across nested modules, lists, and tuples of modules — no manual bookkeeping required.
* `train()` / `eval()` propagate through the full module tree, so layers like `Dropout` and `BatchNorm` automatically switch behavior.
* `to(dtype)` casts every tensor in a model (in place) to a new dtype.
* `state_dict()` / `load_state_dict()` and `save()` / `load()` support both a safe "weights only" mode and a full-object pickle mode.

### 📊 New: Graph visualization

* Added `nnetflow.visualize.draw_dot` and its alias `visualize_model`, which render the autograd computation graph (operations and tensors) with Graphviz — useful for debugging a forward pass or understanding how a model is wired together.

### 🧪 Testing

* Added a much larger test suite covering every new layer (`test_conv1d.py`, `test_conv2d.py`, `test_batchnorm1d.py`, `test_batchnorm2d.py`, `test_layernorm.py`, `test_embedding.py`, `test_dropout.py`, `test_mcdropout.py`, `test_flatten.py`, `test_maxpool1d.py`, `test_maxpool2d.py`, `test_averagepool2d.py`, `test_global_averagepool2d.py`, `test_multihead_attention.py`, `test_module.py`), most of which cross-check forward and backward results directly against PyTorch.

### 📚 Documentation

* Added a small MkDocs documentation site under `docs/` (`index.md`, `quickstart.md`, `guide.md`, `reference.md`), with the API reference auto-generated from docstrings via `mkdocstrings`.
* Added a GitHub Actions workflow to deploy the docs site to GitHub Pages on every push to `main`.

### 📦 Examples

* Added `examples/gpt.py`, a small character-level GPT built entirely from `Embedding`, `MultiHeadAttention`, `LayerNorm`, and `Linear`, including a minimal `Dataset`/`DataLoader` and a text-generation loop.

## Migration Guide

This release is backwards compatible for anyone using nnetflow on CPU (which was always the primary supported path).

* **If you used the 2.0.5 device-abstraction APIs** (`set_device`, `get_device`, `is_gpu_available`, `gpu_supports_dtype`, etc.): these were only ever documented, not something most users had adopted, and are no longer present. Remove any references to them — nnetflow now always runs on NumPy.
* **If you subclass layers directly**: layers now consistently inherit from `Module`. If you had custom layers that duck-typed the old interface, inheriting from `nnetflow.module.Module` instead is the recommended path going forward — it gives you `parameters()`, `train()`/`eval()`, and `save()`/`load()` for free.
* **Loss functions are callable classes**, not bare functions: use `MSELoss()(pred, target)` rather than `mse_loss(pred, target)`.

## Contributors

* Lewis Njue (maintainer)
