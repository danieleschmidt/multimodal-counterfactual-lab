"""
multimodal-counterfactual-lab
=============================

A toolkit for generating counterfactual image-text pairs to support
fairness and robustness research in vision-language models.

Quick start
-----------
>>> from counterfactual_lab import CounterfactualGenerator, BiasEvaluator
>>> gen = CounterfactualGenerator()
>>> result = gen.generate("The doctor examined his patient carefully.")
>>> evaluator = BiasEvaluator(model=my_classifier)
>>> report = evaluator.evaluate(result)
>>> print(report.summary())
"""

__version__ = "0.2.0"
__author__ = "Daniel Schmidt"

from counterfactual_lab.perturbations import (
    LexicalSubstitution,
    SemanticParaphrase,
    Perturbation,
)
from counterfactual_lab.generator import (
    CounterfactualGenerator,
    CounterfactualPair,
    GenerationResult,
)
from counterfactual_lab.bias import (
    BiasEvaluator,
    BiasReport,
    DatasetBiasReport,
)

__all__ = [
    # perturbations
    "LexicalSubstitution",
    "SemanticParaphrase",
    "Perturbation",
    # generator
    "CounterfactualGenerator",
    "CounterfactualPair",
    "GenerationResult",
    # bias evaluation
    "BiasEvaluator",
    "BiasReport",
    "DatasetBiasReport",
]
