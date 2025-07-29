# Development Environment Setup

## Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for model training/inference)
- Docker and Docker Compose
- Git

## Local Development Setup

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/terragon-labs/multimodal-counterfactual-lab.git
cd multimodal-counterfactual-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

### 2. Environment Configuration

Create `.env` file in project root:

```bash
# Model configuration
DEFAULT_DEVICE=cuda  # or cpu
MODEL_CACHE_DIR=./models
DATA_CACHE_DIR=./data

# Development settings
DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_PROFILING=true

# Optional: API keys for external services
HUGGINGFACE_TOKEN=your_token_here
WANDB_API_KEY=your_key_here
```

### 3. IDE Configuration

#### VS Code Settings

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm Configuration

1. Set Python interpreter to `./venv/bin/python`
2. Configure code style to use Black
3. Enable pytest as test runner
4. Install Python Security plugin

## Docker Development

### Development Container

```bash
# Build development image
docker-compose --profile dev build

# Start development environment
docker-compose --profile dev up -d

# Access development container
docker-compose exec counterfactual-lab-dev bash
```

### Docker Compose Services

The development environment includes:
- **Main application**: Counterfactual generation service
- **Jupyter**: For interactive development
- **PostgreSQL**: For experiment tracking
- **Redis**: For caching and task queues
- **Prometheus**: For metrics collection
- **Grafana**: For monitoring dashboards

## Testing

### Running Tests

```bash
# Run all tests
make test-all

# Run unit tests only
make test

# Run with coverage
make test-cov

# Run performance tests
make test-performance

# Run specific test file
pytest tests/unit/test_core.py -v

# Run tests matching pattern
pytest -k "test_generation" -v
```

### Test Categories

- **Unit tests**: Fast tests for individual components
- **Integration tests**: Tests for component interactions
- **E2E tests**: End-to-end workflow testing
- **Performance tests**: Memory and speed benchmarks

### Test Data

Test data is managed through fixtures:

```python
# tests/conftest.py
import pytest
from PIL import Image
import numpy as np

@pytest.fixture
def sample_image():
    """Generate a sample test image"""
    return Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

@pytest.fixture
def sample_text():
    """Provide sample text for testing"""
    return "A person working at a computer"
```

## Code Quality

### Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:
- Code formatting (Black)
- Import sorting (Ruff)
- Type checking (MyPy)
- Security scanning (Bandit)
- General linting

### Manual Code Quality Checks

```bash
# Format code
make format

# Lint code
make lint

# Type check
make type-check

# Security scan
make security-scan
```

### Code Review Guidelines

1. **All changes require PR review**
2. **Tests must pass CI checks**
3. **Code coverage should not decrease**
4. **Security scans must pass**
5. **Documentation updates for API changes**

## Debugging

### Local Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use pdb for interactive debugging
import pdb; pdb.set_trace()

# Profile performance
import cProfile
cProfile.run('your_function()')
```

### GPU Debugging

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Debug memory issues
export CUDA_LAUNCH_BLOCKING=1
```

## Model Development

### Model Management

```bash
# Download pretrained models
python scripts/download_models.py --model stable-diffusion-v2

# Validate model setup
python -c "from counterfactual_lab import CounterfactualGenerator; g = CounterfactualGenerator(); print('Models loaded successfully')"
```

### Experiment Tracking

Integration with Weights & Biases:

```python
import wandb

# Initialize experiment
wandb.init(project="counterfactual-lab", name="experiment-1")

# Log metrics
wandb.log({"loss": loss_value, "accuracy": acc_value})

# Log artifacts
wandb.log_artifact("model.pth")
```

## Performance Optimization

### Profiling

```bash
# Profile generation pipeline
python -m cProfile -o profile.prof scripts/benchmark_generation.py

# Analyze profile
python -c "import pstats; p = pstats.Stats('profile.prof'); p.sort_stats('cumulative').print_stats(20)"
```

### Memory Management

```python
# Monitor memory usage
import psutil
import torch

def log_memory_usage():
    # System memory
    mem = psutil.virtual_memory()
    print(f"System memory: {mem.percent}% used")
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory: {gpu_mem:.2f} GB allocated")
```

## Database Development

### Migrations

```bash
# Create migration
alembic revision --autogenerate -m "Add experiment tracking"

# Run migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Database Testing

Use separate test database:

```python
# tests/conftest.py
@pytest.fixture(scope="session")
def test_db():
    # Setup test database
    test_db_url = "postgresql://test:test@localhost:5433/test_counterfactual"
    engine = create_engine(test_db_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
```

## Documentation Development

### Building Documentation

```bash
# Build docs
make docs

# Serve docs locally
make docs-serve

# Auto-rebuild on changes
mkdocs serve --dev-addr=0.0.0.0:8000
```

### Documentation Standards

- Use Google-style docstrings
- Include type hints
- Provide usage examples
- Document breaking changes
- Keep README.md updated

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""
# Or reduce model precision
export TORCH_USE_CUDA_DSA=1
```

#### Import Errors
```bash
# Reinstall in development mode
pip install -e ".[dev]"

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Permission Errors
```bash
# Fix file permissions
chmod +x scripts/*.sh

# Fix Docker permissions
sudo chown -R $USER:$USER .
```

### Getting Help

1. Check existing GitHub issues
2. Search documentation
3. Ask in Discord community
4. Create detailed bug report with reproduction steps

## Contribution Workflow

1. **Create feature branch** from main
2. **Make changes** following code standards
3. **Write/update tests** for new functionality
4. **Run full test suite** locally
5. **Submit pull request** with clear description
6. **Address review feedback**
7. **Merge after approval**

## Development Tools

### Recommended Extensions

#### VS Code
- Python
- PyLance  
- Python Docstring Generator
- GitLens
- Docker
- Python Security

#### Browser
- React DevTools (for web interface)
- JSON Viewer
- CORS toggle (for development)

### Useful Scripts

```bash
# Quick development setup
./scripts/dev-setup.sh

# Reset development environment
./scripts/dev-reset.sh

# Run development server with hot reload
./scripts/dev-server.sh
```