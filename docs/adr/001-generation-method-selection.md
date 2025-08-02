# ADR-001: Generation Method Selection Strategy

Date: 2025-01-02
Status: Accepted
Deciders: Core Development Team

## Context

The Multimodal Counterfactual Lab needs to support multiple counterfactual generation methods to provide flexibility for different use cases. Users need both high-quality visual generation and interpretable text-based generation depending on their requirements for bias evaluation and regulatory compliance.

## Decision

We will implement a dual-method approach with MoDiCF (Diffusion-based) and ICG (Interpretable Counterfactual Generation) as primary methods, with a pluggable architecture for custom methods.

### Primary Methods

1. **MoDiCF (Diffusion-based)**
   - Primary method for high-quality visual counterfactuals
   - Based on Stable Diffusion with attribute control
   - Best for research and detailed bias analysis

2. **ICG (Interpretable Counterfactual Generation)**
   - Text-first approach with explicit reasoning
   - Provides human-readable explanations
   - Best for regulatory compliance and audit trails

### Extensibility

- Abstract base class `GenerationMethod` for custom implementations
- Plugin system for community-contributed methods
- Configuration-driven method selection

## Consequences

### Positive
- Flexibility to choose appropriate method for use case
- High-quality visual generation with MoDiCF
- Interpretable results with ICG for compliance
- Extensible architecture for future methods
- Comparative evaluation capabilities

### Negative
- Increased complexity in codebase
- Higher resource requirements for multiple models
- Need to maintain multiple generation pipelines
- Potential inconsistency between methods

### Risks
- Method selection confusion for users
- Maintenance overhead for multiple implementations
- Resource contention when running multiple methods

## Options Considered

### Option 1: Single Method (MoDiCF only)
- Pros: Simpler codebase, focused optimization, higher quality
- Cons: Limited flexibility, no interpretability for compliance

### Option 2: Single Method (ICG only)
- Pros: Interpretable results, regulatory compliance, faster generation
- Cons: Lower visual quality, limited research applications

### Option 3: Dual Method with Plugin Architecture (Selected)
- Pros: Maximum flexibility, extensibility, covers all use cases
- Cons: Increased complexity, resource requirements

## Implementation Notes

- Methods are selected via configuration or runtime parameters
- Default method is MoDiCF for backward compatibility
- Method capabilities are exposed through metadata
- Resource management handles GPU allocation between methods
- Evaluation framework works consistently across all methods

## Related ADRs

- [ADR-002](002-evaluation-framework-design.md) - Evaluation Framework Design
- [ADR-003](003-model-integration-approach.md) - Model Integration Approach