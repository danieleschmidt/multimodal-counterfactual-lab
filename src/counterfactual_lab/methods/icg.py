"""ICG: Interpretable Counterfactual Generation."""

import logging
import re
from typing import Dict, List, Optional, Any
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICG:
    """Interpretable counterfactual generation method."""
    
    def __init__(
        self,
        interpreter_model: str = "bert-base",
        generator_model: str = "dalle-3",
        attribute_encoder: str = "clip",
        device: str = "cuda"
    ):
        """Initialize ICG pipeline."""
        self.interpreter_model = interpreter_model
        self.generator_model = generator_model
        self.attribute_encoder = attribute_encoder
        self.device = device
        
        logger.info(f"ICG initialized with {interpreter_model} + {generator_model}")
    
    def generate_interpretable(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        attribute_changes: Optional[Dict[str, str]] = None,
        explanation_level: str = "detailed"
    ) -> Dict[str, Any]:
        """Generate interpretable counterfactuals."""
        if attribute_changes is None:
            attribute_changes = {}
        
        logger.info(f"Generating interpretable counterfactual with changes: {attribute_changes}")
        
        # Apply changes to generate new text
        modified_text = self._apply_text_changes(text, attribute_changes)
        
        # Generate explanations
        explanation = self._generate_explanation(text, modified_text, attribute_changes, explanation_level)
        
        # Generate or modify image
        if image is not None:
            generated_image = image.copy()
        else:
            generated_image = self._create_placeholder_image(modified_text)
        
        return {
            "image": generated_image,
            "text": modified_text,
            "original_text": text,
            "attribute_changes": attribute_changes,
            "explanation": explanation,
            "reasoning": f"Applied interpretable changes to {', '.join(attribute_changes.keys())}"
        }
    
    def _apply_text_changes(self, text: str, changes: Dict[str, str]) -> str:
        """Apply attribute changes to text."""
        modified = text
        
        for attr, value in changes.items():
            if attr == "gender":
                modified = self._change_gender_in_text(modified, value)
            elif attr == "profession":
                modified = self._change_profession_in_text(modified, value)
        
        return modified
    
    def _change_gender_in_text(self, text: str, target_gender: str) -> str:
        """Change gender references in text."""
        if target_gender == "male":
            text = re.sub(r'\b(woman|female)\b', 'man', text, flags=re.IGNORECASE)
        elif target_gender == "female":
            text = re.sub(r'\b(man|male)\b', 'woman', text, flags=re.IGNORECASE)
        return text
    
    def _change_profession_in_text(self, text: str, profession: str) -> str:
        """Change profession references in text."""
        professions = ["doctor", "teacher", "engineer", "nurse"]
        for prof in professions:
            if prof in text.lower():
                text = re.sub(rf'\b{prof}\b', profession, text, flags=re.IGNORECASE)
                break
        return text
    
    def _generate_explanation(self, original: str, modified: str, changes: Dict[str, str], level: str) -> str:
        """Generate explanation for changes."""
        if level == "basic":
            return f"Applied {len(changes)} changes"
        else:
            explanations = []
            for attr, value in changes.items():
                explanations.append(f"Changed {attr} to {value}")
            return "; ".join(explanations)
    
    def _create_placeholder_image(self, text: str) -> Image.Image:
        """Create placeholder image with text."""
        image = Image.new('RGB', (400, 300), color=(240, 240, 240))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), text[:100] + "...", fill=(0, 0, 0))
        return image