"""Core classes for counterfactual generation and bias evaluation."""

from typing import Dict, List, Optional, Union
from pathlib import Path


class CounterfactualGenerator:
    """Main interface for generating counterfactual image-text pairs."""
    
    def __init__(self, method: str = "modicf", device: str = "cuda"):
        """Initialize the counterfactual generator.
        
        Args:
            method: Generation method ("modicf" or "icg")
            device: Compute device ("cuda" or "cpu")
        """
        self.method = method
        self.device = device
    
    def generate(
        self,
        image,
        text: str,
        attributes: List[str],
        num_samples: int = 5
    ) -> Dict:
        """Generate counterfactual examples.
        
        Args:
            image: Input image
            text: Input text description
            attributes: Attributes to modify
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing generated counterfactuals
        """
        raise NotImplementedError("Method implementation required")
    
    def visualize_grid(self, counterfactuals: Dict, save_path: Optional[str] = None):
        """Visualize counterfactuals in a grid format."""
        raise NotImplementedError("Method implementation required")


class BiasEvaluator:
    """Evaluates bias in vision-language models using counterfactuals."""
    
    def __init__(self, model):
        """Initialize bias evaluator with a model."""
        self.model = model
    
    def evaluate(
        self,
        counterfactuals: Dict,
        metrics: List[str]
    ) -> Dict:
        """Evaluate bias using specified metrics.
        
        Args:
            counterfactuals: Generated counterfactual data
            metrics: List of bias metrics to compute
            
        Returns:
            Evaluation results
        """
        raise NotImplementedError("Method implementation required")
    
    def generate_report(
        self,
        results: Dict,
        format: str = "regulatory",
        export_path: Optional[str] = None
    ) -> Dict:
        """Generate bias audit report.
        
        Args:
            results: Evaluation results
            format: Report format ("regulatory", "academic", "technical")
            export_path: Path to save report
            
        Returns:
            Generated report
        """
        raise NotImplementedError("Method implementation required")