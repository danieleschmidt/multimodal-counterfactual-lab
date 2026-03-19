"""
CounterfactualGenerator: takes an (image, text) pair and generates
counterfactual pairs using the registered perturbation strategies.

Vision is optional — falls back to text-only mode gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from counterfactual_lab.perturbations import (
    LexicalSubstitution,
    Perturbation,
    SemanticParaphrase,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CounterfactualPair:
    """An original input + one counterfactual perturbation."""

    original_text: str
    counterfactual_text: str
    perturbation: Perturbation
    original_image: Optional[Any] = None   # PIL.Image or path, or None
    counterfactual_image: Optional[Any] = None

    def __repr__(self) -> str:
        return (
            f"CounterfactualPair("
            f"strategy={self.perturbation.strategy!r}, "
            f"attribute={self.perturbation.attribute!r}, "
            f"direction={self.perturbation.direction!r}, "
            f"edits={self.perturbation.edit_distance_words})"
        )


@dataclass
class GenerationResult:
    """Collection of counterfactual pairs produced from a single input."""

    original_text: str
    original_image: Optional[Any]
    pairs: List[CounterfactualPair] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def by_attribute(self, attribute: str) -> List[CounterfactualPair]:
        return [p for p in self.pairs if p.perturbation.attribute == attribute]

    def by_strategy(self, strategy: str) -> List[CounterfactualPair]:
        return [p for p in self.pairs if p.perturbation.strategy == strategy]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class CounterfactualGenerator:
    """
    Generate counterfactual pairs for (image, text) inputs.

    Strategies applied:
    1. LexicalSubstitution  — direct token swaps for protected attributes
    2. SemanticParaphrase   — minimal meaning-preserving rewrites

    Vision side:
    - If an image is provided and vision_model is set, the image is passed
      through the model; future versions can do image perturbations.
    - If vision_model is None, image is carried along unchanged.

    Parameters
    ----------
    attributes : list of str
        Protected attributes to perturb (default: ["gender"]).
    lexical_directions : list of str
        Which gender-swap directions to generate.
        Options: "male→female", "female→male", "neutral".
    paraphrase_modes : list of str
        Paraphrase modes to apply. Options: "hedge", "neutral", "active-passive".
    vision_model : callable, optional
        A callable (image) → prediction dict. Used only for bias evaluation.
    """

    def __init__(
        self,
        attributes: Optional[List[str]] = None,
        lexical_directions: Optional[List[str]] = None,
        paraphrase_modes: Optional[List[str]] = None,
        vision_model: Optional[Any] = None,
    ):
        self.attributes = attributes if attributes is not None else ["gender"]
        self.lexical_directions = (
            lexical_directions
            if lexical_directions is not None
            else ["male→female", "female→male", "neutral"]
        )
        self.paraphrase_modes = (
            paraphrase_modes if paraphrase_modes is not None else ["hedge", "neutral"]
        )
        self.vision_model = vision_model

        self._lexical = LexicalSubstitution()
        self._paraphrase = SemanticParaphrase()

        logger.info(
            "CounterfactualGenerator ready | attributes=%s | lexical=%s | paraphrase=%s",
            self.attributes,
            self.lexical_directions,
            self.paraphrase_modes,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        image: Optional[Any] = None,
    ) -> GenerationResult:
        """
        Generate all counterfactual pairs for a single (text, image) input.

        Parameters
        ----------
        text : str
            Input caption or description.
        image : optional
            Input image (PIL.Image, path, or None for text-only mode).

        Returns
        -------
        GenerationResult
        """
        pairs: List[CounterfactualPair] = []

        # --- Lexical substitution ---
        for attribute in self.attributes:
            for direction in self.lexical_directions:
                perturbation = self._lexical.generate(text, attribute, direction)
                if perturbation is not None:
                    pairs.append(
                        CounterfactualPair(
                            original_text=text,
                            counterfactual_text=perturbation.perturbed_text,
                            perturbation=perturbation,
                            original_image=image,
                            counterfactual_image=image,  # image unchanged (text-side CF)
                        )
                    )

        # --- Semantic paraphrase ---
        for mode in self.paraphrase_modes:
            perturbation = self._paraphrase.generate(text, mode)
            if perturbation is not None:
                pairs.append(
                    CounterfactualPair(
                        original_text=text,
                        counterfactual_text=perturbation.perturbed_text,
                        perturbation=perturbation,
                        original_image=image,
                        counterfactual_image=image,
                    )
                )

        metadata = {
            "n_lexical": sum(1 for p in pairs if p.perturbation.strategy == "lexical_substitution"),
            "n_paraphrase": sum(1 for p in pairs if p.perturbation.strategy == "semantic_paraphrase"),
            "image_provided": image is not None,
            "vision_model": type(self.vision_model).__name__ if self.vision_model else None,
        }

        result = GenerationResult(
            original_text=text,
            original_image=image,
            pairs=pairs,
            metadata=metadata,
        )

        logger.debug("Generated %d counterfactual pairs for input: %r", len(pairs), text[:60])
        return result

    def generate_batch(
        self,
        inputs: List[Dict],
    ) -> List[GenerationResult]:
        """
        Generate counterfactuals for a list of inputs.

        Each input dict should have keys: "text" (required), "image" (optional).
        """
        results = []
        for i, inp in enumerate(inputs):
            text = inp.get("text", "")
            image = inp.get("image", None)
            logger.debug("Processing batch item %d/%d", i + 1, len(inputs))
            results.append(self.generate(text, image))
        return results
