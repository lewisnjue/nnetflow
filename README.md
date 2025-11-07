# nnetflow — lightweight neural networks for learning

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/lewisnjue/nnetflow/actions)

nnetflow is a small, opinionated deep learning library implemented with NumPy for education and experimentation. It focuses on readability and a small, correct autodiff core so you can learn how deep learning frameworks work under the hood.

Key design goals:
- Minimal API surface: easy to read and reason about
- Correct reverse-mode autodiff (dynamic graphs)
- Small, focused feature set for learning (Linear layer, losses, optimizers)
- Well-tested: unit tests exercise core pieces (Tensor, Linear, losses, optimizers)

This repository represents the v2.0.0 release — a cleaned-up, documented, and tested baseline.

## Highlights

- Tensor: NumPy-backed tensor with reverse-mode autodiff and many activations
- Linear layer: fully-connected layer with sensible initialization
- Losses: MSE, RMSE, Cross-Entropy, Binary Cross-Entropy (logits and probs)
- Optimizers: SGD (+momentum), Adam
- Examples: runnable scripts under `examples/`
- CI: GitHub Actions runs full test matrix on push and PRs
- Local checks: `pre-commit` configured to run tests before pushing changes

## Install

Install from PyPI (when published):

```bash
pip install nnetflow
```

Or install editable from source (recommended for contributors):

```bash
git clone https://github.com/lewisnjue/nnetflow.git
cd nnetflow
pip install -e .
pip install -r requirements.dev
pre-commit install --install-hooks
```

Note: `pre-commit install` sets up git hooks locally. This repo includes a pre-push hook that runs `pytest` to help prevent regressions before pushing.

## Examples

Examples are included in the `examples/` folder and are runnable directly:

```bash
python examples/simple_regression.py
python examples/binary_classification.py
```

They demonstrate model definition, training loops, loss computation and parameter updates using the library primitives.

## Quick usage

```python
import numpy as np
from nnetflow.engine import Tensor
from nnetflow.layers import Linear
from nnetflow.losses import mse_loss
from nnetflow.optim import Adam

X = np.random.randn(128, 3)
y = np.random.randn(128, 1)

layer = Linear(3, 1)
opt = Adam([layer.weight, layer.bias], lr=1e-2)

X_t = Tensor(X, requires_grad=False)
y_t = Tensor(y, requires_grad=False)

for epoch in range(100):
    preds = layer(X_t)
    loss = mse_loss(preds, y_t)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1}: loss={loss.item():.4f}")
```

## Testing

Run unit tests locally with:

```bash
pytest tests/ -q
```

CI runs tests automatically on push and pull requests.

## Pre-commit

This project uses `pre-commit` to run basic checks and to run `pytest` before pushing. After cloning run:

```bash
pip install pre-commit
pre-commit install --install-hooks
```

To run the hooks locally (including the pytest hook configured for pre-push):

```bash
pre-commit run --all-files
```

## Project structure

```
nnetflow/                 # package source
  engine.py               # Tensor & autodiff engine
  layers.py               # Linear, Flatten
  losses.py               # loss functions
  optim.py                # optimizers
examples/                 # runnable examples
tests/                    # unit tests
```

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository and create a feature branch
2. Write tests for your change
3. Run `pytest` and `pre-commit` locally
4. Open a pull request with a clear description

See `CONTRIBUTING.md` for more details.

## Changelog

See `CHANGELOG.md` for details on releases. The repository is now at v2.0.0.

## License

MIT — see `LICENSE`.

---

Maintained by Lewis Njue — aimed at learners and educators building intuition about how neural networks work.
