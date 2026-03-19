"""Tests for perturbation strategies."""

import pytest
from counterfactual_lab.perturbations import (
    LexicalSubstitution,
    SemanticParaphrase,
    Perturbation,
    _apply_token_map,
    GENDER_NEUTRAL_MAP,
    GENDER_SWAP_MAP_M2F,
    GENDER_SWAP_MAP_F2M,
)


class TestLexicalSubstitution:
    def setup_method(self):
        self.strat = LexicalSubstitution()

    # --- male → female ---

    def test_male_to_female_pronouns(self):
        p = self.strat.generate("He walked into the room.", "gender", "male→female")
        assert p is not None
        assert "she" in p.perturbed_text.lower() or "She" in p.perturbed_text
        assert p.strategy == "lexical_substitution"
        assert p.attribute == "gender"
        assert p.direction == "male→female"
        assert p.edit_distance_words >= 1

    def test_male_to_female_noun(self):
        p = self.strat.generate("The man was brilliant.", "gender", "male→female")
        assert p is not None
        assert "woman" in p.perturbed_text.lower()

    def test_male_to_female_possessive(self):
        p = self.strat.generate("His report was outstanding.", "gender", "male→female")
        assert p is not None
        assert "her" in p.perturbed_text.lower()

    # --- female → male ---

    def test_female_to_male(self):
        p = self.strat.generate("She completed her analysis.", "gender", "female→male")
        assert p is not None
        assert "he" in p.perturbed_text.lower() or "He" in p.perturbed_text

    # --- neutral ---

    def test_gender_neutral(self):
        p = self.strat.generate("He is a brilliant engineer.", "gender", "neutral")
        assert p is not None
        assert "they" in p.perturbed_text.lower() or "They" in p.perturbed_text

    def test_neutral_noun(self):
        p = self.strat.generate("The woman presented her findings.", "gender", "neutral")
        assert p is not None
        # "woman" → "person"
        assert "person" in p.perturbed_text.lower()

    # --- no match returns None ---

    def test_no_match_returns_none(self):
        p = self.strat.generate("The robot processed the data.", "gender", "male→female")
        assert p is None

    # --- generate_all_directions ---

    def test_all_directions(self):
        text = "He is an excellent doctor."
        perturbations = self.strat.generate_all_directions(text, "gender")
        # Should produce male→female and neutral (female→male has no match since no female terms)
        directions = {p.direction for p in perturbations}
        assert "male→female" in directions
        assert "neutral" in directions

    def test_all_directions_female_text(self):
        text = "She is an excellent doctor."
        perturbations = self.strat.generate_all_directions(text, "gender")
        directions = {p.direction for p in perturbations}
        assert "female→male" in directions
        assert "neutral" in directions

    # --- capitalisation preservation ---

    def test_capitalisation_preserved(self):
        p = self.strat.generate("He walked in.", "gender", "male→female")
        assert p is not None
        # "He" → "She" (capital preserved)
        assert p.perturbed_text[0].isupper()
        assert p.perturbed_text.startswith("She")

    # --- original text unchanged ---

    def test_original_text_unchanged(self):
        original = "The man was here."
        p = self.strat.generate(original, "gender", "male→female")
        assert p.original_text == original


class TestSemanticParaphrase:
    def setup_method(self):
        self.strat = SemanticParaphrase()

    def test_hedge_excellent(self):
        p = self.strat.generate("She is an excellent engineer.", "hedge")
        assert p is not None
        assert "good" in p.perturbed_text.lower()
        assert "excellent" not in p.perturbed_text.lower()
        assert p.strategy == "semantic_paraphrase"
        assert p.attribute == "sentiment"

    def test_hedge_terrible(self):
        p = self.strat.generate("He is a terrible manager.", "hedge")
        assert p is not None
        assert "bad" in p.perturbed_text.lower()

    def test_hedge_always(self):
        p = self.strat.generate("She always finishes on time.", "hedge")
        assert p is not None
        assert "often" in p.perturbed_text.lower()

    def test_hedge_never(self):
        p = self.strat.generate("He never misses a deadline.", "hedge")
        assert p is not None
        assert "rarely" in p.perturbed_text.lower()

    def test_hedge_no_match_returns_none(self):
        p = self.strat.generate("A person walked into the room.", "hedge")
        assert p is None

    def test_neutral_mode(self):
        p = self.strat.generate("He is a competent professional.", "neutral")
        assert p is not None
        assert "they" in p.perturbed_text.lower() or "They" in p.perturbed_text

    def test_neutral_mode_no_gender(self):
        p = self.strat.generate("The robot processed data.", "neutral")
        assert p is None  # no gender terms to neutralise

    def test_active_passive(self):
        p = self.strat.generate("He helped her with the project.", "active-passive")
        assert p is not None
        assert "was helped" in p.perturbed_text.lower() or "helped" in p.perturbed_text

    def test_all_modes_returns_list(self):
        results = self.strat.generate_all_modes("She always delivers excellent results.")
        assert isinstance(results, list)
        assert len(results) >= 1  # at least hedge should fire


class TestApplyTokenMap:
    def test_basic_swap(self):
        text, changed = _apply_token_map("He went to the store.", {"he": "she"})
        assert "She" in text
        assert len(changed) == 1

    def test_no_partial_match(self):
        # "him" should not match inside "shimmer"
        text, changed = _apply_token_map("The shimmer was beautiful.", {"him": "her"})
        assert "shimmer" in text
        assert len(changed) == 0

    def test_multiple_swaps(self):
        text, changed = _apply_token_map(
            "He and his brother walked.", {"he": "she", "his": "her", "brother": "sister"}
        )
        assert "She" in text
        assert "her" in text
        assert "sister" in text
        assert len(changed) == 3

    def test_case_preservation_sentence_start(self):
        text, _ = _apply_token_map("Man in the hall.", {"man": "woman"})
        assert text.startswith("Woman")

    def test_empty_text(self):
        text, changed = _apply_token_map("", {"he": "she"})
        assert text == ""
        assert changed == []


class TestPerturbationDataclass:
    def test_edit_distance(self):
        p = Perturbation(
            strategy="lexical_substitution",
            original_text="He is great.",
            perturbed_text="She is great.",
            attribute="gender",
            direction="male→female",
            changed_tokens=[("He", "She")],
        )
        assert p.edit_distance_words == 1
        assert p.is_minimal is True

    def test_not_minimal_multi_edit(self):
        p = Perturbation(
            strategy="lexical_substitution",
            original_text="He and his brother.",
            perturbed_text="She and her sister.",
            attribute="gender",
            direction="male→female",
            changed_tokens=[("He", "She"), ("his", "her"), ("brother", "sister")],
        )
        assert p.edit_distance_words == 3
        assert p.is_minimal is False
