"""ICG: Interpretable Counterfactual Generation."""

from typing import Dict, List, Optional


class ICG:
    """Interpretable counterfactual generation method."""
    
    def __init__(
        self,
        interpreter_model: str = "bert-base",
        generator_model: str = "dalle-3",
        attribute_encoder: str = "clip"
    ):
        """Initialize ICG pipeline."""
        self.interpreter_model = interpreter_model
        self.generator_model = generator_model
        self.attribute_encoder = attribute_encoder
    
    def generate_interpretable(
        self,
        text: str,
        attribute_changes: Dict[str, str],
        explanation_level: str = "detailed"
    ):
        """Generate interpretable counterfactuals."""
        raise NotImplementedError("ICG implementation required")