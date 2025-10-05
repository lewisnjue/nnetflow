# nnetflow

A minimal neural network framework with autodiff, inspired by micrograd and pytorch.

## Installation

```bash
pip install nnetflow
```

### From source

```bash
git clone https://github.com/lewisnjue/nnetflow.git
cd nnetflow
pip install -e .
```

For a complete, runnable example, see:

- `examples/pytorch_vs_nnetflow.py` (side-by-side training comparison)


# Documentation

- See `docs/index.md` for a full guide and API overview.
- See `CONTRIBUTING.md` for contribution guidelines.
- See `CHANGELOG.md` for release notes.

## Examples

- PyTorch vs nnetflow simple regression: `examples/pytorch_vs_nnetflow.py`
- Classification comparison with decision boundaries: `examples/classification_torch_vs_nnetflow.py`
  - Outputs: `examples/outputs/classification_boundaries.png`, `examples/outputs/classification_losses.png`
- Regression comparison with fit and loss curves: `examples/regression_torch_vs_nnetflow.py`
  - Outputs: `examples/outputs/regression_fit_and_loss.png`

## Supported layers and model types

- Layers
  - `Linear` (fully-connected)
  - Activations: ReLU, Sigmoid, Tanh, LeakyReLU, GELU, Swish
  - Regularization/Normalization: Dropout, BatchNorm1d

- Model types you can build
  - Feedforward MLPs for classification and regression
  - Logistic regression (single `Linear` + Sigmoid)
  - Deep fully-connected regressors/classifiers with Dropout/BatchNorm
  - Note: Convolutional and recurrent layers are not included (yet)
