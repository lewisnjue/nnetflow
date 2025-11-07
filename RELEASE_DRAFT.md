# Release draft — nnetflow v2.0.0

Release date: 2025-11-07

Summary
-------
This release (v2.0.0) professionalizes the project: improved docs, CI, pre-commit checks, and testing infrastructure. The core API was kept stable for existing primitives (Tensor, Linear, losses, optimizers).

Highlights
----------
- Full test suite (38 tests) validated across the codebase.
- Pre-commit configured to run `pytest` on pre-push, plus common hygiene hooks.
- GitHub Actions workflows:
  - `tests.yml`: runs pytest on push and PRs across multiple Python versions.
  - `pypi-release.yml`: publishes to PyPI only on GitHub Release events.
- Professional README and CHANGELOG.

Notes for maintainers
---------------------
- To publish a release from GitHub UI: create a Release (choose tag `v2.0.0`), fill notes with content from `CHANGELOG.md` and `RELEASE_DRAFT.md`, then press Publish.
- If publishing from CLI, create an annotated tag:

```bash
git tag -a v2.0.0 -m "Release v2.0.0"
git push origin v2.0.0
```

Then create the Release on GitHub (or use `gh release create v2.0.0 --notes-file RELEASE_DRAFT.md`).

Security / tokens
-----------------
- The publish workflow expects a `PYPI_API_TOKEN` secret in repository settings. Add this before publishing.

Checklist before publishing
--------------------------
- [ ] All tests pass locally and in CI
- [ ] `CHANGELOG.md` updated with notable changes
- [ ] `pyproject.toml` / `setup.py` versions are correct (2.0.0)
- [ ] Confirm PyPI package metadata and classifiers
