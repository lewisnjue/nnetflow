# Contributing to nnetflow

Thank you for your interest in contributing! Please follow these guidelines to get started.

## Setup

- Fork and clone the repo
- Create a branch per change: `git checkout -b feat/your-change`
- Install deps: `pip install -e .[test]`

## Development

- Run tests: `pytest -q`
- Write type hints and docstrings (NumPy style)
- Keep changes focused and readable

## Commit Messages

- Conventional commits are preferred: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, etc.

## Pull Requests

- Link related issues with `Closes #<number>`
- Include description and motivation
- Ensure CI passes
