"""Unit tests for CounterfactualGenerator."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from counterfactual_lab import CounterfactualGenerator, CounterfactualPair, GenerationResult


class TestCounterfactualGenerator:
    def setup_method(self):
        self.gen = CounterfactualGenerator(
            attributes=["gender"],
            lexical_directions=["male→female", "female→male", "neutral"],
            paraphrase_modes=["hedge", "neutral"],
        )

    def test_generate_returns_generation_result(self):
        result = self.gen.generate("He is a doctor.")
        assert isinstance(result, GenerationResult)

    def test_generate_produces_pairs(self):
        result = self.gen.generate("He is an excellent scientist.")
        assert result.n_pairs > 0

    def test_generate_no_match_returns_empty(self):
        result = self.gen.generate("The sky is blue.")
        assert result.n_pairs == 0

    def test_original_text_preserved(self):
        text = "She is a brilliant engineer."
        result = self.gen.generate(text)
        assert result.original_text == text

    def test_pair_has_counterfactual_text(self):
        result = self.gen.generate("He is a doctor.")
        for pair in result.pairs:
            assert isinstance(pair, CounterfactualPair)
            assert pair.counterfactual_text != "" or pair.original_text == pair.counterfactual_text

    def test_by_attribute_filter(self):
        result = self.gen.generate("She is an outstanding researcher.")
        gender_pairs = result.by_attribute("gender")
        assert all(p.perturbation.attribute == "gender" for p in gender_pairs)

    def test_by_strategy_filter(self):
        result = self.gen.generate("He is excellent.")
        lex_pairs = result.by_strategy("lexical_substitution")
        para_pairs = result.by_strategy("semantic_paraphrase")
        assert len(lex_pairs) + len(para_pairs) == len(result.pairs)

    def test_image_none_by_default(self):
        result = self.gen.generate("He is a doctor.")
        assert result.original_image is None

    def test_image_passed_through(self):
        dummy_image = object()
        result = self.gen.generate("He is a doctor.", image=dummy_image)
        assert result.original_image is dummy_image

    def test_metadata_populated(self):
        result = self.gen.generate("He is excellent.")
        assert "n_lexical" in result.metadata
        assert "n_paraphrase" in result.metadata

    def test_generate_batch_processes_all(self):
        inputs = [
            {"text": "He is a great doctor."},
            {"text": "She won the award."},
            {"text": "They are talented musicians."},
        ]
        results = self.gen.generate_batch(inputs)
        assert len(results) == 3

    def test_generate_batch_empty_list(self):
        results = self.gen.generate_batch([])
        assert results == []

    def test_default_attributes(self):
        gen = CounterfactualGenerator()
        assert gen.attributes == ["gender"]
