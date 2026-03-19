"""Unit tests for perturbation strategies."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from counterfactual_lab.perturbations import (
    LexicalSubstitution,
    SemanticParaphrase,
    Perturbation,
    _apply_token_map,
    GENDER_SWAP_MAP_M2F,
    GENDER_SWAP_MAP_F2M,
    GENDER_NEUTRAL_MAP,
)


# ---------------------------------------------------------------------------
# LexicalSubstitution tests
# ---------------------------------------------------------------------------

class TestLexicalSubstitution:
    def setup_method(self):
        self.lex = LexicalSubstitution()

    def test_male_to_female_pronoun(self):
        p = self.lex.generate("He went to the store.", "gender", "male→female")
        assert p is not None
        assert "She" in p.perturbed_text or "she" in p.perturbed_text
        assert p.strategy == "lexical_substitution"
        assert p.attribute == "gender"
        assert p.direction == "male→female"

    def test_female_to_male_pronoun(self):
        p = self.lex.generate("She is an excellent engineer.", "gender", "female→male")
        assert p is not None
        assert "He" in p.perturbed_text or "he" in p.perturbed_text
        assert p.direction == "female→male"

    def test_neutral_pronoun(self):
        p = self.lex.generate("He is a great doctor.", "gender", "neutral")
        assert p is not None
        assert "they" in p.perturbed_text.lower() or "them" in p.perturbed_text.lower()
        assert p.direction == "neutral"

    def test_male_noun_swap(self):
        p = self.lex.generate("The man solved the problem.", "gender", "male→female")
        assert p is not None
        assert "woman" in p.perturbed_text.lower()

    def test_no_match_returns_none(self):
        p = self.lex.generate("The sky is blue.", "gender", "male→female")
        assert p is None

    def test_generate_all_directions_returns_list(self):
        results = self.lex.generate_all_directions(
            "He is an excellent scientist.", "gender"
        )
        assert len(results) >= 1
        directions = [r.direction for r in results]
        assert "male→female" in directions
        assert "neutral" in directions

    def test_changed_tokens_populated(self):
        p = self.lex.generate("He helped her.", "gender", "male→female")
        assert p is not None
        assert len(p.changed_tokens) >= 1

    def test_edit_distance_words_property(self):
        p = self.lex.generate("He is here.", "gender", "male→female")
        assert p is not None
        assert p.edit_distance_words == len(p.changed_tokens)

    def test_is_minimal_property(self):
        p = self.lex.generate("He is here.", "gender", "male→female")
        assert p is not None
        # "he" → "she" = 1 change → minimal
        assert p.is_minimal is True

    def test_case_preservation(self):
        p = self.lex.generate("He is tall.", "gender", "male→female")
        assert p is not None
        # Capital H → capital S
        assert p.perturbed_text[0].isupper()

    def test_unsupported_attribute_returns_none(self):
        p = self.lex.generate("He is tall.", "religion", "swap")
        assert p is None

    def test_family_term_swap(self):
        p = self.lex.generate("The father went home.", "gender", "male→female")
        assert p is not None
        assert "mother" in p.perturbed_text.lower()

    def test_honorific_swap(self):
        p = self.lex.generate("Good morning, sir.", "gender", "male→female")
        assert p is not None
        assert "ma'am" in p.perturbed_text.lower() or "friend" in p.perturbed_text.lower()


# ---------------------------------------------------------------------------
# SemanticParaphrase tests
# ---------------------------------------------------------------------------

class TestSemanticParaphrase:
    def setup_method(self):
        self.para = SemanticParaphrase()

    def test_hedge_mode_replaces_extreme_word(self):
        p = self.para.generate("She is an excellent researcher.", "hedge")
        assert p is not None
        assert "excellent" not in p.perturbed_text.lower()
        assert "good" in p.perturbed_text.lower()

    def test_hedge_mode_always_never(self):
        p = self.para.generate("He always wins every time.", "hedge")
        assert p is not None
        assert "always" not in p.perturbed_text.lower()

    def test_neutral_mode_removes_pronouns(self):
        p = self.para.generate("He is a great scientist.", "neutral")
        assert p is not None
        assert "he" not in p.perturbed_text.lower().split()

    def test_neutral_mode_no_match_returns_none(self):
        p = self.para.generate("The sky is blue.", "neutral")
        assert p is None

    def test_hedge_no_match_returns_none(self):
        p = self.para.generate("The sky is blue.", "hedge")
        assert p is None

    def test_generate_all_modes_list(self):
        results = self.para.generate_all_modes("She is an outstanding engineer.")
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_strategy_label(self):
        p = self.para.generate("He is an excellent doctor.", "hedge")
        assert p is not None
        assert p.strategy == "semantic_paraphrase"

    def test_original_text_preserved(self):
        original = "She was brilliant."
        p = self.para.generate(original, "hedge")
        assert p is not None
        assert p.original_text == original

    def test_active_passive_no_crash(self):
        # Active-passive is optional — just ensure it doesn't error
        p = self.para.generate("He helped her.", "active-passive")
        # May be None or Perturbation; either is valid
        assert p is None or isinstance(p, Perturbation)


# ---------------------------------------------------------------------------
# _apply_token_map utility tests
# ---------------------------------------------------------------------------

class TestApplyTokenMap:
    def test_basic_substitution(self):
        new_text, changed = _apply_token_map("He is tall.", {"he": "she"})
        assert "she" in new_text.lower()
        assert len(changed) == 1

    def test_case_preserve_upper(self):
        new_text, changed = _apply_token_map("He is tall.", {"he": "she"})
        assert new_text.startswith("She")

    def test_no_match_unchanged(self):
        new_text, changed = _apply_token_map("The sky is blue.", {"he": "she"})
        assert new_text == "The sky is blue."
        assert changed == []

    def test_multiple_substitutions(self):
        new_text, changed = _apply_token_map(
            "He helped him.", {"he": "she", "him": "her"}
        )
        assert len(changed) == 2
        assert "she" in new_text.lower()
        assert "her" in new_text.lower()

    def test_whole_word_only(self):
        # "history" must NOT be replaced by swapping "his" → "her"
        new_text, changed = _apply_token_map("He read the history book.", {"his": "her"})
        assert "history" in new_text  # "history" intact
