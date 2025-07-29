.PHONY: help install install-dev test test-cov lint format type-check clean docs docs-serve

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=counterfactual_lab --cov-report=html --cov-report=term

lint:  ## Run linting
	ruff check src tests
	black --check src tests

format:  ## Format code
	black src tests
	ruff --fix src tests

type-check:  ## Run type checking
	mypy src

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

docs:  ## Build documentation
	mkdocs build

docs-serve:  ## Serve documentation locally
	mkdocs serve