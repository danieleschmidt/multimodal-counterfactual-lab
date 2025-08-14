"""Multimodal Counterfactual Lab.

A data-generation studio that creates counterfactual image-text pairs
for fairness and robustness research.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

# Progressive capability detection and import
# Try scalable -> robust -> full -> lightweight in order of capability
try:
    from counterfactual_lab.scalable_core import (
        ScalableCounterfactualGenerator as CounterfactualGenerator,
        ScalableBiasEvaluator as BiasEvaluator
    )
    _implementation_level = "scalable"
    _using_lightweight = False
except ImportError:
    try:
        from counterfactual_lab.robust_core import (
            RobustCounterfactualGenerator as CounterfactualGenerator,
            RobustBiasEvaluator as BiasEvaluator
        )
        _implementation_level = "robust"
        _using_lightweight = False
    except ImportError:
        try:
            from counterfactual_lab.core import CounterfactualGenerator, BiasEvaluator
            _implementation_level = "full"
            _using_lightweight = False
        except ImportError as e:
            # Fall back to lightweight implementation
            from counterfactual_lab.lightweight_core import (
                LightweightCounterfactualGenerator as CounterfactualGenerator,
                LightweightBiasEvaluator as BiasEvaluator
            )
            _implementation_level = "lightweight"
            _using_lightweight = True
            import warnings
            warnings.warn(f"Using lightweight implementation due to missing dependencies: {e}")

__all__ = [
    "CounterfactualGenerator", 
    "BiasEvaluator",
    "_implementation_level",
    "_using_lightweight"
]