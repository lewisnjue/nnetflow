# nnetflow Documentation

nnetflow is a minimal neural network framework with autodiff, inspired by micrograd and PyTorch. It provides a simple API for building and training neural networks using NumPy arrays and custom autodiff. It supports both CPU and GPU (via CuPy) with automatic device management.

## Installation

```bash
pip install nnetflow
```

## Modules

- nnetflow.engine: Core Tensor and autodiff engine
- nnetflow.nn: Layers, MLP, optimizers, loss functions
- nnetflow.optim: Optimizers (SGD, Adam, etc.)
- nnetflow.cuda: Device management and GPU support
- nnetflow.utils: Utility functions

## Getting Started

- See the [usage guide](usage.md) for step-by-step instructions.
- Example scripts: regression, classification, and image tasks are in the main repo.

## Features

- Minimal, readable codebase
- PyTorch-like API for tensors, modules, and optimizers
- Automatic device (CPU/GPU) support
- Custom layers and loss functions
- Easy extensibility for research and learning

---
