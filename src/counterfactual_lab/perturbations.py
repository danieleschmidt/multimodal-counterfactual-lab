"""
Perturbation strategies for counterfactual text generation.

Two core strategies:
- LexicalSubstitution: swap protected-attribute terms directly
- SemanticParaphrase: rewrite sentences to be attribute-neutral or to swap attributes
                      (rule-based; no external model needed)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Protected-attribute lexicons
# ---------------------------------------------------------------------------

GENDER_MALE_TERMS: List[str] = [
    "he", "him", "his", "himself",
    "man", "men", "boy", "boys",
    "gentleman", "gentlemen", "male", "males",
    "father", "son", "brother", "uncle", "grandfather",
    "husband", "boyfriend",
    "mr", "sir",
]

GENDER_FEMALE_TERMS: List[str] = [
    "she", "her", "hers", "herself",
    "woman", "women", "girl", "girls",
    "lady", "ladies", "female", "females",
    "mother", "daughter", "sister", "aunt", "grandmother",
    "wife", "girlfriend",
    "ms", "mrs", "miss", "ma'am",
]

GENDER_NEUTRAL_MAP: Dict[str, str] = {
    # pronouns
    "he": "they",
    "she": "they",
    "him": "them",
    "her": "them",
    "his": "their",
    "hers": "theirs",
    "himself": "themselves",
    "herself": "themselves",
    # nouns
    "man": "person",
    "men": "people",
    "boy": "child",
    "boys": "children",
    "gentleman": "individual",
    "gentlemen": "individuals",
    "woman": "person",
    "women": "people",
    "girl": "child",
    "girls": "children",
    "lady": "individual",
    "ladies": "individuals",
    "male": "person",
    "males": "people",
    "female": "person",
    "females": "people",
    # family
    "father": "parent",
    "mother": "parent",
    "son": "child",
    "daughter": "child",
    "brother": "sibling",
    "sister": "sibling",
    "uncle": "relative",
    "aunt": "relative",
    "grandfather": "grandparent",
    "grandmother": "grandparent",
    "husband": "spouse",
    "wife": "spouse",
    "boyfriend": "partner",
    "girlfriend": "partner",
    # honorifics
    "mr": "mx",
    "mrs": "mx",
    "ms": "mx",
    "miss": "mx",
    "sir": "friend",
    "ma'am": "friend",
}

# Male → Female mapping (swap in both directions)
GENDER_SWAP_MAP_M2F: Dict[str, str] = {
    "he": "she",
    "him": "her",
    "his": "her",
    "himself": "herself",
    "man": "woman",
    "men": "women",
    "boy": "girl",
    "boys": "girls",
    "gentleman": "lady",
    "gentlemen": "ladies",
    "male": "female",
    "males": "females",
    "father": "mother",
    "son": "daughter",
    "brother": "sister",
    "uncle": "aunt",
    "grandfather": "grandmother",
    "husband": "wife",
    "boyfriend": "girlfriend",
    "mr": "ms",
    "sir": "ma'am",
}

GENDER_SWAP_MAP_F2M: Dict[str, str] = {v: k for k, v in GENDER_SWAP_MAP_M2F.items()}


@dataclass
class Perturbation:
    """A single counterfactual perturbation of a text."""

    strategy: str
    original_text: str
    perturbed_text: str
    attribute: str          # which protected attribute was modified
    direction: str          # e.g. "male→female", "gender-neutral", "paraphrase"
    changed_tokens: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def edit_distance_words(self) -> int:
        """Count of word-level substitutions made."""
        return len(self.changed_tokens)

    @property
    def is_minimal(self) -> bool:
        """True if only one token was changed (maximally minimal perturbation)."""
        return self.edit_distance_words == 1


# ---------------------------------------------------------------------------
# Strategy 1: Lexical Substitution
# ---------------------------------------------------------------------------

class LexicalSubstitution:
    """
    Strategy 1 — direct token-level swap of protected-attribute terms.

    Covers:
    - gender: male↔female swap, or neutralise to they/them/person
    - (extensible: race, age, religion …)
    """

    SUPPORTED_ATTRIBUTES = ("gender",)

    def generate(
        self,
        text: str,
        attribute: str = "gender",
        direction: str = "neutral",  # "male→female" | "female→male" | "neutral"
    ) -> Optional[Perturbation]:
        """
        Generate a single lexically-substituted counterfactual.

        Returns None if no substitutions were made.
        """
        if attribute == "gender":
            return self._swap_gender(text, direction)
        return None

    def generate_all_directions(self, text: str, attribute: str = "gender") -> List[Perturbation]:
        """Generate one counterfactual per supported direction."""
        results = []
        for direction in ("male→female", "female→male", "neutral"):
            p = self.generate(text, attribute, direction)
            if p is not None:
                results.append(p)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _swap_gender(self, text: str, direction: str) -> Optional[Perturbation]:
        if direction == "male→female":
            mapping = GENDER_SWAP_MAP_M2F
        elif direction == "female→male":
            mapping = GENDER_SWAP_MAP_F2M
        else:  # neutral
            mapping = GENDER_NEUTRAL_MAP

        new_text, changed = _apply_token_map(text, mapping)
        if not changed:
            return None

        return Perturbation(
            strategy="lexical_substitution",
            original_text=text,
            perturbed_text=new_text,
            attribute="gender",
            direction=direction,
            changed_tokens=changed,
            metadata={"substitution_map_size": len(mapping)},
        )


# ---------------------------------------------------------------------------
# Strategy 2: Semantic Paraphrase (rule-based, no external model)
# ---------------------------------------------------------------------------

_ACTIVE_PASSIVE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "X verb Y" → "Y was verbed by X"  (very coarse; just for demo)
    (
        re.compile(
            r"\b(he|she|they|the (?:man|woman|person))\s+(helped|assisted|saved|trained|taught|hired|fired|promoted)\s+(him|her|them|the (?:man|woman|person))\b",
            re.IGNORECASE,
        ),
        r"\3 was \2 by \1",
    ),
]

_SENTIMENT_HEDGE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(excellent|outstanding|brilliant)\b", re.IGNORECASE), "good"),
    (re.compile(r"\b(terrible|horrible|dreadful)\b", re.IGNORECASE), "bad"),
    (re.compile(r"\b(always)\b", re.IGNORECASE), "often"),
    (re.compile(r"\b(never)\b", re.IGNORECASE), "rarely"),
]


class SemanticParaphrase:
    """
    Strategy 2 — minimal meaning-preserving rewrites that change surface form
    without (ideally) changing ground-truth semantics.

    Two sub-modes:
    - "hedge": soften absolute sentiment words (tests model robustness to
               small semantic shifts)
    - "neutral": remove gendered terms via GENDER_NEUTRAL_MAP (same as
                 LexicalSubstitution neutral, but framed as paraphrase)

    No external model needed; all rule-based.
    """

    SUPPORTED_MODES = ("hedge", "neutral", "active-passive")

    def generate(
        self, text: str, mode: str = "hedge"
    ) -> Optional[Perturbation]:
        if mode == "hedge":
            return self._hedge(text)
        if mode == "neutral":
            return self._neutral(text)
        if mode == "active-passive":
            return self._active_passive(text)
        return None

    def generate_all_modes(self, text: str) -> List[Perturbation]:
        results = []
        for mode in self.SUPPORTED_MODES:
            p = self.generate(text, mode)
            if p is not None:
                results.append(p)
        return results

    # ------------------------------------------------------------------

    def _hedge(self, text: str) -> Optional[Perturbation]:
        new_text = text
        changed: List[Tuple[str, str]] = []
        for pat, replacement in _SENTIMENT_HEDGE_PATTERNS:
            match = pat.search(new_text)
            if match:
                changed.append((match.group(0), replacement))
                new_text = pat.sub(replacement, new_text)
        if not changed:
            return None
        return Perturbation(
            strategy="semantic_paraphrase",
            original_text=text,
            perturbed_text=new_text,
            attribute="sentiment",
            direction="hedge",
            changed_tokens=changed,
        )

    def _neutral(self, text: str) -> Optional[Perturbation]:
        new_text, changed = _apply_token_map(text, GENDER_NEUTRAL_MAP)
        if not changed:
            return None
        return Perturbation(
            strategy="semantic_paraphrase",
            original_text=text,
            perturbed_text=new_text,
            attribute="gender",
            direction="neutral-paraphrase",
            changed_tokens=changed,
        )

    def _active_passive(self, text: str) -> Optional[Perturbation]:
        new_text = text
        changed: List[Tuple[str, str]] = []
        for pat, template in _ACTIVE_PASSIVE_PATTERNS:
            match = pat.search(new_text)
            if match:
                replacement = pat.sub(template, new_text)
                changed.append((match.group(0), replacement))
                new_text = replacement
        if not changed:
            return None
        return Perturbation(
            strategy="semantic_paraphrase",
            original_text=text,
            perturbed_text=new_text,
            attribute="voice",
            direction="active→passive",
            changed_tokens=changed,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _apply_token_map(
    text: str, mapping: Dict[str, str]
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Replace whole-word occurrences of mapping keys with their values.
    Case-preserving: if the original token is capitalised, capitalise replacement.
    Returns (new_text, list_of_(original, replacement) pairs).
    """
    changed: List[Tuple[str, str]] = []

    # Build a single regex that matches any key as a whole word
    sorted_keys = sorted(mapping, key=len, reverse=True)  # longest first
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b",
        re.IGNORECASE,
    )

    def replace(m: re.Match) -> str:
        token = m.group(0)
        key = token.lower()
        repl = mapping.get(key, token)
        # Preserve capitalisation
        if token[0].isupper():
            repl = repl.capitalize()
        changed.append((token, repl))
        return repl

    new_text = pattern.sub(replace, text)
    return new_text, changed
