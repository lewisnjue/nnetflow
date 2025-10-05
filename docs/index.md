# nnetflow Documentation

Welcome to the nnetflow docs.

## Contents

- Getting Started
- API Reference
- Tutorials
- Development

## Getting Started

Install from PyPI:

```bash
pip install nnetflow
```

Or from source:

```bash
git clone https://github.com/lewisnjue/nnetflow.git
cd nnetflow
pip install -e .
```

## API Reference

- `nnetflow.engine.Tensor`: Autodiff-enabled tensor with NumPy/CuPy backend.
- `nnetflow.layers.Linear`: Fully-connected layer with Xavier/He initialization.
- `nnetflow.optim.SGD`, `nnetflow.optim.Adam`: Optimizers.
- `nnetflow.module.Module`: Base class for models; handles parameters and save/load.

## Tutorials

- See `nbs/` for example notebooks.

## Development

- Run tests: `pip install .[test] && pytest -q`
- Lint/type check: coming soon.

