# Multimodal Counterfactual Lab

A data-generation studio that creates counterfactual image-text pairs for fairness and robustness research.

## Overview

With regulators now requiring bias audits for Vision-Language Models (VLMs), this lab provides automated tools for generating counterfactual multimodal data:

- **MoDiCF Pipeline**: Diffusion-based counterfactual generation
- **ICG Generator**: Interpretable Counterfactual Generation
- **Skew-Aware Sampling**: Balanced representation across attributes
- **CITS Evaluation**: Counterfactual Image-Text Score metrics
- **Bias Audit Reports**: Regulatory-compliant documentation

## Key Features

- Generate counterfactuals across protected attributes (race, gender, age)
- Control fine-grained attributes while preserving context
- Evaluate model fairness with generated counterfactuals
- Export audit-ready reports for compliance
- Integration with popular VLM frameworks

## Quick Example

```python
from counterfactual_lab import CounterfactualGenerator, BiasEvaluator

# Generate counterfactuals
generator = CounterfactualGenerator(method="modicf")
counterfactuals = generator.generate(
    image=image,
    text="A doctor examining a patient",
    attributes=["gender", "race"],
    num_samples=5
)

# Evaluate bias
evaluator = BiasEvaluator(model)
results = evaluator.evaluate(counterfactuals, metrics=["demographic_parity"])
```

## Getting Started

1. [Install the package](installation.md)
2. Follow the [Quick Start guide](quickstart.md)
3. Explore [Examples](examples.md)
4. Read the [API documentation](api/core.md)