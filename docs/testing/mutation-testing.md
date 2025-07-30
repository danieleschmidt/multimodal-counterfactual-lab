# Mutation Testing Guide

## Overview

Mutation testing evaluates the quality of test suites by introducing small changes (mutations) to the source code and checking if tests detect these changes. This helps identify weak spots in test coverage.

## Tools Configured

### 1. Mutmut (Primary)
- **Fast execution** with intelligent caching
- **Detailed reporting** with survival analysis
- **CI/CD integration** friendly

### 2. MutPy (Alternative)  
- **Comprehensive operators** for Python-specific mutations
- **HTML reporting** with detailed mutation analysis
- **Academic-grade** precision

## Running Mutation Tests

### Quick Start

```bash
# Install mutation testing tools
pip install -e ".[dev]"

# Run basic mutation testing
mutmut run

# View results
mutmut results
mutmut show
```

### Advanced Usage

```bash
# Test specific modules
mutmut run --paths-to-mutate src/counterfactual_lab/core.py

# Run with coverage threshold
mutmut run --runner "pytest --cov=counterfactual_lab --cov-fail-under=90"

# Parallel execution
mutmut run --processes 4

# Focus on specific test types
mutmut run --runner "pytest -m unit"
```

### MutPy Alternative

```bash
# Run MutPy mutation testing
mut.py --target src/counterfactual_lab --unit-test tests --runner pytest

# Generate HTML report
mut.py --target src/counterfactual_lab --unit-test tests --report html
```

## Configuration

### Mutmut Settings (`pyproject.toml`)

```toml
[tool.mutmut]
paths_to_mutate = "src/"
backup = false
runner = "python -m pytest"
tests_dir = "tests/"
dict_synonyms = ["Struct", "NamedStruct"] 
total = 500
suspicious_policy = "ignore"
untested_policy = "ignore"
```

### MutPy Settings

```toml
[tool.mutpy]
target = "src/counterfactual_lab"
unit-test = "tests"
runner = "pytest"
report = ["html", "json"]
timeout = 300
disable-operator = ["AOD", "COD", "COI"]
exclude = ["__init__.py", "setup.py"]
```

## Mutation Operators

### Enabled Operators

| Operator | Description | Example |
|----------|-------------|---------|
| **AOR** | Arithmetic Operator Replacement | `+` → `-`, `*` → `/` |
| **LCR** | Logical Connector Replacement | `and` → `or` |
| **ROR** | Relational Operator Replacement | `>` → `>=`, `==` → `!=` |
| **ASR** | Assignment Operator Replacement | `+=` → `-=` |
| **BCR** | Break/Continue Replacement | `break` → `continue` |
| **EXS** | Exception Swallowing | `try/except` → `try/pass` |
| **SIR** | Slice Index Remove | `list[1:]` → `list[:]` |

### Disabled Operators

- **AOD**: Arithmetic Operator Deletion (too noisy)
- **COD**: Conditional Operator Deletion (creates syntax errors)
- **COI**: Conditional Operator Insertion (low value)

## Quality Metrics

### Mutation Score Calculation

```
Mutation Score = (Killed Mutants / Total Mutants) × 100%
```

### Target Thresholds

- **Minimum Acceptable**: 70%
- **Good Quality**: 80%
- **Excellent Quality**: 90%+

### Interpreting Results

```bash
# View detailed mutation report
mutmut show

# Example output interpretation:
# ✓ Killed: Test detected the mutation (GOOD)
# ✗ Survived: Test did not detect mutation (NEEDS IMPROVEMENT)  
# ⚠ Timeout: Mutation caused infinite loop (INVESTIGATE)
# ? Suspicious: Unusual behavior (REVIEW)
```

## Integration with CI/CD

### GitHub Actions Workflow

```yaml
name: Mutation Testing
on: 
  pull_request:
    paths: ['src/**', 'tests/**']

jobs:
  mutation-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -e ".[dev]"
        
      - name: Run mutation tests
        run: |
          mutmut run
          mutmut junitxml > mutation-results.xml
          
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: mutation-test-results
          path: mutation-results.xml
```

### Quality Gates

```bash
# Fail build if mutation score below threshold
mutmut run --check-coverage --coverage-threshold 80
```

## Best Practices

### 1. Incremental Testing
```bash
# Test only changed files
mutmut run --paths-to-mutate $(git diff --name-only HEAD~1 | grep '\.py$' | tr '\n' ' ')
```

### 2. Focus on Critical Code
```bash
# Prioritize core business logic
mutmut run --paths-to-mutate src/counterfactual_lab/core.py src/counterfactual_lab/methods/
```

### 3. Exclude Low-Value Code
```python
# Mark code to exclude from mutation testing
def utility_function():
    """Utility function - not critical for mutation testing"""
    pass  # pragma: no mutate
```

### 4. Performance Optimization
```bash
# Use intelligent test selection
mutmut run --use-coverage --use-patch-file
```

## Analyzing Results

### Common Mutation Survival Patterns

1. **Boundary Conditions**: Tests miss edge cases
   ```python
   # Original: if x > 0
   # Mutant: if x >= 0  (might survive)
   # Fix: Add test for x = 0
   ```

2. **Exception Handling**: Tests don't verify exceptions
   ```python
   # Original: raise ValueError("message")
   # Mutant: raise TypeError("message")  (might survive)  
   # Fix: Assert specific exception type
   ```

3. **Boolean Logic**: Tests miss logical combinations
   ```python
   # Original: if a and b
   # Mutant: if a or b  (might survive)
   # Fix: Test all boolean combinations
   ```

### Fixing Surviving Mutants

```python
# Before: Weak test
def test_calculation():
    result = calculate(5, 10)
    assert result > 0  # Too generic

# After: Strong test  
def test_calculation():
    result = calculate(5, 10)
    assert result == 15  # Exact assertion
    
    # Test boundary conditions
    assert calculate(0, 5) == 5
    assert calculate(-1, 1) == 0
    
    # Test error conditions
    with pytest.raises(ValueError):
        calculate(None, 5)
```

## Reporting and Visualization

### Generate HTML Report
```bash
# Mutmut HTML output
mutmut html

# MutPy detailed report
mut.py --target src --unit-test tests --report html --report-dir mutation-report
```

### Integration with Coverage Tools
```bash
# Combine with coverage analysis
pytest --cov=counterfactual_lab --cov-report=html
mutmut run --use-coverage
```

## Troubleshooting

### Common Issues

1. **Slow Execution**
   ```bash
   # Solution: Use parallel processing
   mutmut run --processes $(nproc)
   ```

2. **Memory Issues**
   ```bash
   # Solution: Reduce batch size
   mutmut run --total 100  # Process in smaller batches
   ```

3. **Flaky Tests**
   ```bash
   # Solution: Run tests multiple times
   mutmut run --runner "pytest --count=3"
   ```

### Performance Tuning

```bash
# Cache mutations for faster reruns
export MUTMUT_CACHE_DIR=.mutmut-cache

# Skip equivalent mutants
mutmut run --skip-equivalent

# Use test selection heuristics
mutmut run --use-coverage --use-patch-file
```

## Advanced Features

### Custom Mutation Operators

```python
# Create custom mutations for domain-specific logic
class CustomMutationOperator:
    def mutate_fairness_threshold(self, node):
        """Custom mutations for fairness thresholds"""
        if isinstance(node, ast.Num) and 0.5 <= node.n <= 1.0:
            # Mutate fairness thresholds specifically
            return ast.Num(n=node.n + 0.1)
        return node
```

### Integration with Property-Based Testing

```python
import hypothesis
from hypothesis import strategies as st

@hypothesis.given(st.integers(min_value=0, max_value=100))
def test_with_mutation_focus(value):
    """Property-based test that helps detect mutations"""
    result = process_value(value)
    
    # Property: result should always be non-negative
    assert result >= 0
    
    # Property: boundary behavior
    if value == 0:
        assert result == 0
```

## References

- [Mutmut Documentation](https://mutmut.readthedocs.io/)
- [MutPy Project](https://github.com/mutpy/mutpy)
- [Mutation Testing Best Practices](https://pitest.org/quickstart/best_practices/)
- [Academic Research on Mutation Testing](https://doi.org/10.1109/TSE.2019.2927124)