"""Tests for CounterfactualGenerator."""

import pytest
from counterfactual_lab.generator import CounterfactualGenerator, GenerationResult, CounterfactualPair


class TestCounterfactualGenerator:
    def setup_method(self):
        self.gen = CounterfactualGenerator()

    def test_generate_returns_result(self):
        result = self.gen.generate("He is an excellent doctor.")
        assert isinstance(result, GenerationResult)

    def test_generate_produces_pairs(self):
        result = self.gen.generate("He is an excellent doctor.")
        assert result.n_pairs > 0

    def test_pairs_are_counterfactual_pair_instances(self):
        result = self.gen.generate("She is an outstanding engineer.")
        for pair in result.pairs:
            assert isinstance(pair, CounterfactualPair)

    def test_original_text_preserved(self):
        text = "He walked into the building."
        result = self.gen.generate(text)
        assert result.original_text == text

    def test_counterfactual_text_differs(self):
        result = self.gen.generate("He is a great manager.")
        for pair in result.pairs:
            assert pair.counterfactual_text != pair.original_text

    def test_lexical_pairs_present(self):
        result = self.gen.generate("He is an expert.")
        lexical = result.by_strategy("lexical_substitution")
        assert len(lexical) > 0

    def test_paraphrase_pairs_present(self):
        result = self.gen.generate("He is an excellent developer.")
        paraphrase = result.by_strategy("semantic_paraphrase")
        assert len(paraphrase) > 0

    def test_by_attribute_filter(self):
        result = self.gen.generate("He is an excellent developer.")
        gender_pairs = result.by_attribute("gender")
        assert len(gender_pairs) > 0

    def test_no_pairs_on_no_match(self):
        # Text with no gendered terms and no sentiment words
        result = self.gen.generate("The robot processed the data.")
        assert result.n_pairs == 0

    def test_image_none_is_ok(self):
        result = self.gen.generate("He is a great engineer.", image=None)
        assert result.original_image is None

    def test_image_carried_through(self):
        fake_image = object()
        result = self.gen.generate("She is talented.", image=fake_image)
        for pair in result.pairs:
            assert pair.original_image is fake_image

    def test_metadata_populated(self):
        result = self.gen.generate("He is brilliant.")
        assert "n_lexical" in result.metadata
        assert "n_paraphrase" in result.metadata
        assert "image_provided" in result.metadata
        assert result.metadata["image_provided"] is False

    def test_generate_batch(self):
        inputs = [
            {"text": "He is an excellent doctor."},
            {"text": "She is an outstanding engineer."},
            {"text": "The robot processed data."},
        ]
        results = self.gen.generate_batch(inputs)
        assert len(results) == 3
        assert results[0].n_pairs > 0
        assert results[1].n_pairs > 0
        assert results[2].n_pairs == 0  # no gendered terms or sentiment words

    def test_custom_attributes(self):
        gen = CounterfactualGenerator(attributes=["gender"])
        result = gen.generate("She is a great engineer.")
        assert all(p.perturbation.attribute in ("gender", "sentiment") for p in result.pairs)

    def test_custom_directions(self):
        gen = CounterfactualGenerator(
            attributes=["gender"],
            lexical_directions=["male→female"],
            paraphrase_modes=[],
        )
        result = gen.generate("He is brilliant.")
        assert all(
            p.perturbation.direction == "male→female"
            for p in result.by_strategy("lexical_substitution")
        )

    def test_custom_paraphrase_modes(self):
        gen = CounterfactualGenerator(
            attributes=["gender"],
            lexical_directions=[],
            paraphrase_modes=["hedge"],
        )
        result = gen.generate("He is always excellent.")
        assert result.n_pairs > 0
        assert all(p.perturbation.strategy == "semantic_paraphrase" for p in result.pairs)
