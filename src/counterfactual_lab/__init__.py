"""Multimodal Counterfactual Lab.

A data-generation studio that creates counterfactual image-text pairs
for fairness and robustness research.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

from counterfactual_lab.core import CounterfactualGenerator, BiasEvaluator

__all__ = [
    "CounterfactualGenerator",
    "BiasEvaluator",
]