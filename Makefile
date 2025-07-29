.PHONY: help install install-dev test test-all test-integration test-e2e test-performance test-slow test-fast test-cov lint format type-check clean docs docs-serve benchmark security-scan build release

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run unit tests
	pytest tests/unit/

test-all:  ## Run all tests
	pytest

test-integration:  ## Run integration tests
	pytest tests/integration/ -m integration

test-e2e:  ## Run end-to-end tests
	pytest tests/e2e/ -m e2e

test-performance:  ## Run performance tests
	pytest tests/performance/ -m performance

test-slow:  ## Run slow tests
	pytest -m slow

test-fast:  ## Run fast tests only
	pytest -m "not slow and not performance"

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

benchmark:  ## Run performance benchmarks
	python benchmarks/benchmark_generation.py
	python benchmarks/benchmark_evaluation.py

security-scan:  ## Run security scans
	./scripts/security_scan.sh

build:  ## Build package for distribution
	python -m build
	
release:  ## Prepare a new release (requires bump type: major/minor/patch)
	@echo "Usage: make release BUMP=patch|minor|major"
	@if [ -z "$(BUMP)" ]; then echo "Please specify BUMP type"; exit 1; fi
	python scripts/prepare_release.py $(BUMP)

docker-build:  ## Build Docker image
	docker build -t counterfactual-lab .

docker-run:  ## Run Docker container
	docker-compose up counterfactual-lab

docker-dev:  ## Run development Docker environment
	docker-compose --profile dev up counterfactual-lab-dev