# Testing Guide

This document provides comprehensive information about testing in the Multimodal Counterfactual Lab project.

## Testing Philosophy

Our testing strategy follows a multi-layered approach:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Validate performance characteristics
5. **Mutation Tests**: Verify test quality and coverage

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── methods/            # Generation method tests
│   ├── metrics/            # Evaluation metric tests
│   ├── utils/              # Utility function tests
│   └── models/             # Model integration tests
├── integration/            # Integration tests
│   ├── pipelines/          # End-to-end pipeline tests
│   ├── api/                # API integration tests
│   └── data/               # Data processing tests
├── e2e/                    # End-to-end workflow tests
│   ├── workflows/          # Complete workflow tests
│   └── cli/                # CLI interface tests
├── performance/            # Performance and benchmarking tests
│   ├── benchmarks/         # Performance benchmarks
│   └── stress/             # Stress testing
└── fixtures/               # Test data and fixtures
    ├── images/             # Sample images
    ├── models/             # Mock model files
    └── data/               # Test datasets
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
make test

# Run fast tests only (exclude slow and performance tests)
make test-fast

# Run tests with coverage
make test-cov

# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest tests/e2e/                     # End-to-end tests only
pytest tests/performance/             # Performance tests only
```

### Test Markers

Use pytest markers to categorize and filter tests:

```bash
# Run tests by marker
pytest -m unit                       # Unit tests
pytest -m integration                # Integration tests
pytest -m e2e                        # End-to-end tests
pytest -m performance                # Performance tests
pytest -m slow                       # Slow tests
pytest -m gpu                        # GPU-dependent tests
pytest -m "not slow"                 # Exclude slow tests
pytest -m "not performance"          # Exclude performance tests
```

### Parallel Test Execution

```bash
# Run tests in parallel (faster execution)
pytest -n auto                       # Auto-detect CPU cores
pytest -n 4                          # Use 4 parallel workers
```

## Test Configuration

### Environment Variables

Set these environment variables for testing:

```bash
export COUNTERFACTUAL_LAB_ENV=test
export ENABLE_TELEMETRY=false
export CUDA_VISIBLE_DEVICES=-1        # Force CPU for most tests
export TRANSFORMERS_CACHE=./tests/fixtures/models
```

### Configuration Files

- `pytest.ini`: Main pytest configuration
- `pyproject.toml`: Additional test settings and coverage configuration
- `tests/conftest.py`: Shared fixtures and test setup

## Writing Tests

### Test Structure

Follow this structure for writing tests:

```python
"""Test module docstring describing what is being tested."""

import pytest
from unittest.mock import Mock, patch

# Apply appropriate markers
pytestmark = pytest.mark.unit  # or integration, e2e, performance


class TestComponentName:
    """Test class for specific component."""

    @pytest.fixture
    def component_instance(self):
        """Create component instance for testing."""
        return ComponentClass()

    def test_basic_functionality(self, component_instance):
        """Test basic functionality."""
        result = component_instance.method()
        assert result is not None

    def test_error_handling(self, component_instance):
        """Test error conditions."""
        with pytest.raises(ValueError, match="Expected error message"):
            component_instance.method_with_error()

    @pytest.mark.slow
    def test_expensive_operation(self, component_instance):
        """Test that takes significant time."""
        # Mark slow tests appropriately
        pass
```

### Using Fixtures

Leverage shared fixtures from `conftest.py`:

```python
def test_generation_with_sample_data(self, sample_image, sample_text, 
                                   mock_diffusion_model):
    """Test using shared fixtures."""
    generator = CounterfactualGenerator(model=mock_diffusion_model)
    result = generator.generate(image=sample_image, text=sample_text)
    pytest.assert_valid_counterfactual_result(result)
```

### Mocking Best Practices

1. **Mock External Dependencies**: Mock APIs, models, and file I/O
2. **Use Appropriate Mock Types**: `Mock`, `MagicMock`, or `patch`
3. **Configure Realistic Behavior**: Mock return values should match expected types
4. **Verify Mock Calls**: Assert that mocks were called with expected parameters

```python
def test_with_mocking(self, sample_image, sample_text):
    """Example of proper mocking."""
    with patch('counterfactual_lab.models.load_model') as mock_load:
        mock_model = Mock()
        mock_model.generate.return_value = sample_image
        mock_load.return_value = mock_model
        
        generator = CounterfactualGenerator()
        result = generator.generate(image=sample_image, text=sample_text)
        
        # Verify mock was called correctly
        mock_load.assert_called_once()
        mock_model.generate.assert_called_once()
```

## Test Categories

### Unit Tests

Test individual functions and classes in isolation:

- **Purpose**: Verify component behavior
- **Scope**: Single function/class
- **Dependencies**: Mocked
- **Speed**: Fast (< 1 second each)

Example:
```python
def test_demographic_parity_calculation(self, fairness_evaluator):
    """Test demographic parity metric calculation."""
    predictions = np.array([1, 0, 1, 0])
    groups = np.array([0, 0, 1, 1])
    
    dp_score = fairness_evaluator.demographic_parity(predictions, groups)
    
    assert 0 <= dp_score <= 1
    assert isinstance(dp_score, float)
```

### Integration Tests

Test component interactions:

- **Purpose**: Verify component integration
- **Scope**: Multiple components
- **Dependencies**: Partially mocked
- **Speed**: Medium (1-10 seconds each)

Example:
```python
def test_generation_evaluation_pipeline(self, mock_generator, mock_evaluator):
    """Test integration between generation and evaluation."""
    pipeline = Pipeline(generator=mock_generator, evaluator=mock_evaluator)
    result = pipeline.run(image=sample_image, text=sample_text)
    
    assert "generation_results" in result
    assert "evaluation_results" in result
```

### End-to-End Tests

Test complete workflows:

- **Purpose**: Verify full system behavior
- **Scope**: Complete workflows
- **Dependencies**: Minimally mocked
- **Speed**: Slow (10+ seconds each)

Example:
```python
def test_cli_generation_workflow(self, temp_dir):
    """Test complete CLI workflow."""
    # Test actual CLI execution
    result = subprocess.run([
        "counterfactual-lab", "generate",
        "--input", "tests/fixtures/sample.jpg",
        "--output", str(temp_dir)
    ])
    
    assert result.returncode == 0
    assert (temp_dir / "results.json").exists()
```

### Performance Tests

Test performance characteristics:

- **Purpose**: Verify performance requirements
- **Scope**: Performance-critical components
- **Dependencies**: Realistic or mocked for timing
- **Speed**: Variable (marked as slow)

Example:
```python
@pytest.mark.performance
def test_generation_throughput(self, performance_config):
    """Test generation throughput meets requirements."""
    start_time = time.time()
    
    for i in range(performance_config["batch_size"]):
        result = generator.generate(sample_image, sample_text)
    
    duration = time.time() - start_time
    throughput = performance_config["batch_size"] / duration
    
    assert throughput >= performance_config["min_throughput"]
```

## Test Quality

### Coverage Requirements

- **Minimum Coverage**: 80% overall
- **Critical Components**: 95% coverage required
- **New Code**: 90% coverage required for PRs

```bash
# Generate coverage reports
make test-cov

# View HTML coverage report
open htmlcov/index.html
```

### Mutation Testing

Verify test quality using mutation testing:

```bash
# Run mutation tests
mutmut run

# View mutation test results
mutmut results
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Every pull request
- Every push to main branch
- Nightly schedule for performance tests

### Test Matrix

Tests run across:
- Python versions: 3.10, 3.11, 3.12
- Operating systems: Ubuntu, macOS, Windows
- Dependencies: Latest and pinned versions

## Debugging Tests

### Running Single Tests

```bash
# Run specific test file
pytest tests/unit/test_core.py

# Run specific test function
pytest tests/unit/test_core.py::test_function_name

# Run with verbose output
pytest -v tests/unit/test_core.py

# Run with debugging
pytest --pdb tests/unit/test_core.py
```

### Test Debugging Tips

1. **Use `pytest.set_trace()`** for interactive debugging
2. **Check fixture values** with print statements
3. **Verify mock configurations** are correct
4. **Use `--capture=no`** to see print output
5. **Check environment variables** are set correctly

## Common Issues

### Mock-Related Issues

```python
# Issue: Mock not working as expected
# Solution: Verify patch target path
with patch('counterfactual_lab.module.ClassName') as mock:
    # Ensure path matches actual import

# Issue: Mock calls not recorded
# Solution: Use return_value correctly
mock.method.return_value = expected_value
```

### Fixture Issues

```python
# Issue: Fixture not available
# Solution: Check fixture scope and location
@pytest.fixture(scope="function")  # Correct scope
def my_fixture():
    return "value"
```

### Performance Test Issues

```python
# Issue: Performance tests flaky
# Solution: Use relative performance comparisons
baseline_time = measure_baseline()
test_time = measure_test_case()
assert test_time < baseline_time * 1.1  # 10% tolerance
```

## Best Practices

### General Testing

1. **Test Behavior, Not Implementation**: Focus on what the code does, not how
2. **Use Descriptive Test Names**: Make test purpose clear from name
3. **Keep Tests Independent**: Each test should be able to run in isolation
4. **Use Appropriate Assertions**: Choose specific assertions over generic ones
5. **Test Edge Cases**: Include boundary conditions and error cases

### Performance Testing

1. **Use Realistic Data**: Test with representative inputs
2. **Include Warmup**: Account for initialization overhead
3. **Measure Multiple Runs**: Average performance over multiple executions
4. **Set Reasonable Thresholds**: Balance strictness with reliability
5. **Monitor Resource Usage**: Test memory, CPU, and GPU usage

### Mock Usage

1. **Mock External Dependencies**: APIs, databases, file systems
2. **Don't Over-Mock**: Test real code paths when possible
3. **Configure Realistic Responses**: Match actual API behavior
4. **Verify Mock Usage**: Assert mocks were called correctly
5. **Clean Up Mocks**: Ensure mocks don't affect other tests

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Guide](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Mutation Testing with mutmut](https://mutmut.readthedocs.io/)

For questions about testing, please reach out to the development team or open an issue on GitHub.