# Contributing to Spectra

We welcome contributions! See the [full contributing guide](docs/contributing.md) for details on:

- Adding new metrics
- Adding new scenarios
- Adding new model adapters
- Development setup
- Testing conventions

## Quick Start

```bash
git clone https://github.com/StrangeStorm243-bit/when-metrics-lie.git
cd when-metrics-lie
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check src tests
ruff format src tests
```

## Pull Request Process

1. Fork the repo and create a feature branch
2. Write tests for your changes
3. Ensure all tests pass: `pytest`
4. Ensure lint passes: `ruff check src tests`
5. Submit a PR with a clear description
