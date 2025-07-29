---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear and concise description of the bug.

## Environment
- **OS**: [e.g., Ubuntu 22.04, macOS 13.1, Windows 11]
- **Python Version**: [e.g., 3.10.8]
- **Package Version**: [e.g., 0.1.0]
- **GPU/CUDA**: [e.g., NVIDIA RTX 4090, CUDA 12.1]

## Reproduction Steps
```python
# Minimal code example that reproduces the issue
from counterfactual_lab import CounterfactualGenerator

generator = CounterfactualGenerator(method="modicf")
# ... rest of the reproduction code
```

## Expected Behavior
A clear description of what you expected to happen.

## Actual Behavior
A clear description of what actually happened.

## Error Output
```
Paste any error messages or stack traces here
```

## Additional Context
- [ ] This is a regression (worked in a previous version)
- [ ] This affects multiple generation methods
- [ ] This affects the web interface
- [ ] This is related to model loading/inference

Add any other context about the problem here.