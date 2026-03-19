"""Tests for BiasEvaluator."""

import pytest
from counterfactual_lab.generator import CounterfactualGenerator
from counterfactual_lab.bias import BiasEvaluator, BiasReport, DatasetBiasReport


@pytest.fixture
def gen():
    return CounterfactualGenerator()


class TestBiasEvaluatorFairModel:
    """With a fair (consistent) model, consistency should be 1.0."""

    def test_consistent_model_rate_is_one(self, gen, simple_classifier):
        # Use text with no sentiment words (no hedge matches) so only gender
        # perturbations fire → all produce same prediction → rate == 1.0.
        result = gen.generate("He walked into the building.")
        evaluator = BiasEvaluator(simple_classifier)
        report = evaluator.evaluate(result)
        assert report.consistency_rate == 1.0
        assert not report.bias_detected

    def test_report_type(self, gen, simple_classifier):
        result = gen.generate("She is a brilliant researcher.")
        evaluator = BiasEvaluator(simple_classifier)
        report = evaluator.evaluate(result)
        assert isinstance(report, BiasReport)

    def test_n_pairs_matches(self, gen, simple_classifier):
        result = gen.generate("He is always an excellent manager.")
        evaluator = BiasEvaluator(simple_classifier)
        report = evaluator.evaluate(result)
        assert report.n_pairs == result.n_pairs

    def test_summary_string(self, gen, simple_classifier):
        result = gen.generate("She is an outstanding scientist.")
        evaluator = BiasEvaluator(simple_classifier)
        report = evaluator.evaluate(result)
        summary = report.summary()
        assert "consistency" in summary.lower() or "Consistent" in summary

    def test_no_pairs_returns_full_consistency(self, gen, simple_classifier):
        result = gen.generate("The robot processed data.")  # no gendered terms
        assert result.n_pairs == 0
        evaluator = BiasEvaluator(simple_classifier)
        report = evaluator.evaluate(result)
        assert report.consistency_rate == 1.0
        assert report.n_pairs == 0


class TestBiasEvaluatorBiasedModel:
    """With the biased model, inconsistency should be detected."""

    def test_biased_model_detects_flips(self, gen, biased_classifier):
        # biased_classifier always labels male pronouns POSITIVE
        # → swapping to female should produce NEGATIVE → inconsistency
        result = gen.generate("He is great.")
        evaluator = BiasEvaluator(biased_classifier)
        report = evaluator.evaluate(result)
        assert report.bias_detected
        assert report.consistency_rate < 1.0

    def test_attribute_rates_populated(self, gen, biased_classifier):
        result = gen.generate("He is a great leader.")
        evaluator = BiasEvaluator(biased_classifier)
        report = evaluator.evaluate(result)
        assert "gender" in report.attribute_rates

    def test_strategy_rates_populated(self, gen, biased_classifier):
        result = gen.generate("He is a great leader.")
        evaluator = BiasEvaluator(biased_classifier)
        report = evaluator.evaluate(result)
        assert len(report.strategy_rates) > 0

    def test_score_delta_computed(self, gen, biased_classifier):
        result = gen.generate("He is a great leader.")
        evaluator = BiasEvaluator(biased_classifier)
        report = evaluator.evaluate(result)
        # Biased classifier returns score in dicts, so delta should be computed
        if report.n_pairs > 0:
            assert report.mean_score_delta is not None
            assert report.mean_score_delta >= 0.0


class TestBatchEvaluation:
    def test_batch_returns_dataset_report(self, gen, simple_classifier):
        inputs = [
            {"text": "He is an excellent doctor."},
            {"text": "She is an outstanding engineer."},
        ]
        results = gen.generate_batch(inputs)
        evaluator = BiasEvaluator(simple_classifier)
        dataset_report = evaluator.evaluate_batch(results)
        assert isinstance(dataset_report, DatasetBiasReport)
        assert dataset_report.n_inputs == 2

    def test_batch_consistency_in_range(self, gen, simple_classifier):
        inputs = [{"text": "He is an excellent doctor."}]
        results = gen.generate_batch(inputs)
        evaluator = BiasEvaluator(simple_classifier)
        report = evaluator.evaluate_batch(results)
        assert 0.0 <= report.overall_consistency_rate <= 1.0

    def test_batch_summary(self, gen, simple_classifier):
        inputs = [{"text": "She is always an outstanding leader."}]
        results = gen.generate_batch(inputs)
        evaluator = BiasEvaluator(simple_classifier)
        report = evaluator.evaluate_batch(results)
        summary = report.summary()
        assert "consistency" in summary.lower() or "Consistent" in summary

    def test_batch_attribute_rates(self, gen, biased_classifier):
        inputs = [
            {"text": "He is a great manager."},
            {"text": "She is a great manager."},
        ]
        results = gen.generate_batch(inputs)
        evaluator = BiasEvaluator(biased_classifier)
        report = evaluator.evaluate_batch(results)
        assert "gender" in report.attribute_rates

    def test_batch_per_input_reports(self, gen, simple_classifier):
        inputs = [
            {"text": "He is an excellent engineer."},
            {"text": "She is outstanding."},
        ]
        results = gen.generate_batch(inputs)
        evaluator = BiasEvaluator(simple_classifier)
        report = evaluator.evaluate_batch(results)
        assert len(report.per_input_reports) == 2

    def test_batch_n_pairs_total(self, gen, simple_classifier):
        inputs = [
            {"text": "He is an excellent engineer."},
            {"text": "She is outstanding."},
        ]
        results = gen.generate_batch(inputs)
        evaluator = BiasEvaluator(simple_classifier)
        report = evaluator.evaluate_batch(results)
        expected = sum(r.n_pairs for r in results)
        assert report.n_pairs_total == expected


class TestModelOutputFormats:
    """BiasEvaluator should handle various model output formats."""

    def test_string_output(self, gen):
        # Model returns plain string label
        model = lambda text: "POSITIVE"
        gen_local = CounterfactualGenerator()
        result = gen_local.generate("He is excellent.")
        evaluator = BiasEvaluator(model)
        report = evaluator.evaluate(result)
        assert report.consistency_rate == 1.0  # always "POSITIVE"

    def test_dict_output(self, gen):
        model = lambda text: {"label": "POS", "score": 0.9}
        gen_local = CounterfactualGenerator()
        result = gen_local.generate("He is excellent.")
        evaluator = BiasEvaluator(model)
        report = evaluator.evaluate(result)
        assert report.consistency_rate == 1.0

    def test_list_output_hf_style(self, gen):
        # HuggingFace pipeline style: [{"label": ..., "score": ...}, ...]
        model = lambda text: [
            {"label": "POSITIVE", "score": 0.9},
            {"label": "NEGATIVE", "score": 0.1},
        ]
        gen_local = CounterfactualGenerator()
        result = gen_local.generate("He is excellent.")
        evaluator = BiasEvaluator(model)
        report = evaluator.evaluate(result)
        assert report.consistency_rate == 1.0  # always picks POSITIVE (highest score)
