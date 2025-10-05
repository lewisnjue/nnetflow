# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-10-05
## [0.1.2] - 2025-10-05
### Changed
- Removed C++/CUDA backend files and extern; Python-only package
- Added new examples with visualizations; updated README
- Switched CI to release-only deploy and verified 0.1.1 publish

### Added
- Xavier/He initializers in `Linear` layer and documentation
- Basic pytest suite covering Tensor, Linear, and SGD
- Documentation: docs/ index, CONTRIBUTING, README improvements, CHANGELOG

### Fixed
- Version mismatch between `setup.py` and `pyproject.toml`
